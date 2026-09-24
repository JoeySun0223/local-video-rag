"""人工修改、审核决定及章节元数据生成服务。"""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import partial
from typing import Any

import httpx

from video_pipeline.shared.cloud import cloud_settings

from video_pipeline.segments.core import (
    build_document,
    call_glm_once,
    meaningful_sentences,
    validate_label_response,
    validated_request,
)
from video_pipeline.segments.markdown import segment_markdown
from video_pipeline.shared.text import paragraph_text
from video_pipeline.shared.io import load_json, now_iso
from video_pipeline.validation.rules import validate_cleaned, validate_segments
from .catalog import PipelinePaths
from .storage import commit_artifacts, locked_video_action


class EditConflictError(RuntimeError):
    """编辑基于旧章节版本时抛出。"""


def _required_document(paths: PipelinePaths, video_id: str, kind: str) -> tuple[Any, Any]:
    path = paths.artifact(video_id, kind)
    if path is None or not path.is_file():
        labels = {
            "raw": "原始 ASR", "cleaned": "清洗结果", "segments": "语义片段",
            "history": "清洗审核历史",
        }
        raise FileNotFoundError(f"尚未生成{labels.get(kind, kind)}")
    return path, load_json(path)


def _raise_validation(errors: list[str]) -> None:
    if errors:
        raise ValueError("修改违反数据规则：" + "；".join(errors))


def _current_history(paths: PipelinePaths, video_id: str, cleaned: dict[str, Any]) -> tuple[Any, Any]:
    """更新审核档案中的当前清洗快照；原始建议与决定仍保留。"""
    history_path = paths.artifact(video_id, "history")
    if history_path is None or not history_path.is_file():
        return None, None
    history = load_json(history_path)
    history["final"] = cleaned
    history["updated_at"] = now_iso()
    return history_path, history


def _preserve_manual_content(
    current: dict[str, Any], updated: dict[str, Any],
    previous_cleaned: dict[str, Any], cleaned: dict[str, Any],
) -> dict[str, Any]:
    """保留人工正文的拼接方式，兼容旧文件按逐句换行生成的正文。"""
    current_rows = current.get("segments")
    updated_rows = updated.get("segments")
    if not isinstance(current_rows, list) or not isinstance(updated_rows, list):
        return updated
    overrides = {
        (int(row.get("start_sentence_id", -1)), int(row.get("end_sentence_id", -1))): row
        for row in current_rows
        if isinstance(row, dict) and row.get("manual_content_override")
    }
    for row in updated_rows:
        if not isinstance(row, dict):
            continue
        previous = overrides.get((
            int(row.get("start_sentence_id", -1)), int(row.get("end_sentence_id", -1)),
        ))
        if previous is None:
            continue
        row["manual_content_override"] = True
        start, end = int(row["start_sentence_id"]), int(row["end_sentence_id"])
        old_content = "".join(
            str(item.get("text") or "") for item in previous_cleaned.get("sentences", [])
            if start <= int(item["sentence_id"]) <= end
        )
        if paragraph_text(previous.get("content")) == paragraph_text(old_content):
            row["content"] = paragraph_text("".join(
                str(item.get("text") or "") for item in cleaned.get("sentences", [])
                if start <= int(item["sentence_id"]) <= end
            ))
        if previous.get("edited_at"):
            row["edited_at"] = previous["edited_at"]
    return updated


def _updated_segments(
    cleaned: dict[str, Any], document: dict[str, Any], previous_cleaned: dict[str, Any],
) -> dict[str, Any]:
    # 人工可清空原章节起始句；重建时仍使用全部原句作时间锚点。
    sentences = [
        {"sentence_id": int(row["sentence_id"]),
         "start_ms": int(row["start_ms"]), "end_ms": int(row["end_ms"]),
         "text": str(row.get("text") or "")}
        for row in cleaned.get("sentences", [])
    ]
    rows = document.get("segments")
    if not isinstance(rows, list):
        raise ValueError("现有语义片段缺少 segments 数组")
    labels = validate_label_response({
        "labels": [{
            "position": position,
            "title": row.get("title"),
            "summary": row.get("summary"),
            "keywords": row.get("keywords"),
        } for position, row in enumerate(rows, 1)]
    }, list(range(1, len(rows) + 1)))
    plan = [{
        "start_sentence_id": int(rows[position - 1]["start_sentence_id"]),
        "end_sentence_id": int(rows[position - 1]["end_sentence_id"]),
        **labels[position],
    } for position in range(1, len(rows) + 1)]
    updated = build_document(cleaned, sentences, plan, document.get("video_path"))
    _preserve_manual_content(document, updated, previous_cleaned, cleaned)
    _raise_validation(validate_segments(cleaned, updated))
    updated["review_status"] = "pending_review"
    updated["updated_at"] = now_iso()
    updated["revision"] = int(document.get("revision", 0)) + 1
    updated.pop("confirmed_at", None)
    return updated


@locked_video_action
def save_cleaned_sentences(
    paths: PipelinePaths, video_id: str, edits: list[dict[str, Any]], *,
    persist: bool = True,
) -> dict[str, Any]:
    raw_path, raw = _required_document(paths, video_id, "raw")
    cleaned_path, cleaned = _required_document(paths, video_id, "cleaned")
    del raw_path
    current = cleaned.get("sentences")
    if not isinstance(current, list):
        raise ValueError("现有清洗结果缺少 sentences 数组")
    expected_ids = [int(row.get("sentence_id", -1)) for row in current]
    supplied_ids = [int(row.get("sentence_id", -1)) for row in edits]
    if supplied_ids != expected_ids:
        raise ValueError(f"修改必须按原顺序覆盖全部句子，期望 {expected_ids}，实际 {supplied_ids}")
    texts: dict[int, str] = {}
    for row in edits:
        sentence_id = int(row.get("sentence_id", -1))
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"sentence_id={sentence_id} 的文本必须是字符串")
        # 人工正文可能在英文词内或空格处切回原句；不能裁掉边界空格。
        texts[sentence_id] = text
    updated_cleaned = {
        **cleaned,
        "sentences": [
            {**row, "text": texts[int(row["sentence_id"])]}
            for row in current
        ],
    }
    updated_cleaned["full_text"] = "\n".join(
        row["text"] for row in updated_cleaned["sentences"] if row["text"]
    )
    _raise_validation(validate_cleaned(raw, updated_cleaned))

    segments_path = paths.artifact(video_id, "segments")
    updated_document: dict[str, Any] | None = None
    if segments_path is not None and segments_path.is_file():
        updated_document = _updated_segments(updated_cleaned, load_json(segments_path), cleaned)

    json_files = {cleaned_path: updated_cleaned}
    history_path, history = _current_history(paths, video_id, updated_cleaned)
    if history_path is not None:
        json_files[history_path] = history
    text_files: dict[Any, str] = {}
    if updated_document is not None and segments_path is not None:
        for segment in updated_document.get("segments", []):
            if isinstance(segment, dict):
                segment["content"] = paragraph_text(segment.get("content"))
        json_files[segments_path] = updated_document
        markdown_path = paths.artifact(video_id, "markdown")
        assert markdown_path is not None
        text_files[markdown_path] = segment_markdown(updated_document)
    if persist:
        commit_artifacts(paths, video_id, json_files=json_files, text_files=text_files)
    return {"cleaned": updated_cleaned, "segments": updated_document}


@locked_video_action
def save_cleaned_sentence(
    paths: PipelinePaths, video_id: str, sentence_id: int, text: str, *,
    persist: bool = True,
) -> dict[str, Any]:
    """修改一句清洗文本，仍走完整文件校验和派生更新。"""
    _, cleaned = _required_document(paths, video_id, "cleaned")
    rows = cleaned.get("sentences")
    if not isinstance(rows, list):
        raise ValueError("现有清洗结果缺少 sentences 数组")
    found = False
    edits: list[dict[str, Any]] = []
    for row in rows:
        current_id = int(row.get("sentence_id", -1))
        current_text = row.get("text")
        if not isinstance(current_text, str):
            raise ValueError(f"sentence_id={current_id} 的文本必须是字符串")
        if current_id == sentence_id:
            current_text = text
            found = True
        edits.append({"sentence_id": current_id, "text": current_text})
    if not found:
        raise ValueError(f"找不到 sentence_id={sentence_id}")
    return save_cleaned_sentences(
        paths, video_id, edits,
        persist=persist,
    )


def chapter_markdown(segment: dict[str, Any]) -> str:
    """生成界面上一章的 Markdown 预览。"""
    keywords = segment.get("keywords")
    keyword_text = "、".join(str(item).strip() for item in keywords or [] if str(item).strip())
    start = int(segment.get("start_ms", 0))
    end = int(segment.get("end_ms", 0))

    def timecode(value: int) -> str:
        seconds = max(0, value // 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"

    return "\n".join([
        f"## {str(segment.get('title') or '未命名章节').strip()}", "",
        f"> 时间：{timecode(start)}–{timecode(end)}",
        f"> 关键词：{keyword_text or '无'}", "",
        "### 章节摘要", "", paragraph_text(segment.get("summary")), "",
        "### 章节正文", "", paragraph_text(segment.get("content")), "",
    ])


def _markdown_section(markdown: str, heading: str, following: str | None) -> str:
    end = rf"(?=^###\s+{re.escape(following)}\s*$)" if following else r"\Z"
    match = re.search(
        rf"^###\s+{re.escape(heading)}\s*$\s*(.*?){end}",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Markdown 缺少“### {heading}”")
    return match.group(1).strip()


def markdown_chapter_content(markdown: str) -> str:
    """提取旧版 Markdown 接口中的章节正文。"""
    return paragraph_text(_markdown_section(markdown, "章节正文", None))


def _redistribute_chapter_content(
    cleaned: dict[str, Any], start_sentence_id: int, end_sentence_id: int, content: str
) -> None:
    """把章节自然段分配回带时间戳的原句行。

    句子 ID 和时间不变；字符差异映射让小范围术语修改尽量留在原句，
    大段重写也按确定规则分配，无法保证逐字音频对齐。
    """
    all_rows = cleaned.get("sentences")
    if not isinstance(all_rows, list):
        raise ValueError("现有清洗结果缺少 sentences 数组")
    rows = [
        row for row in all_rows
        if start_sentence_id <= int(row.get("sentence_id", -1)) <= end_sentence_id
    ]
    if not rows:
        raise ValueError(
            f"章节句子范围 {start_sentence_id}-{end_sentence_id} 在清洗结果中不存在"
        )
    new_text = paragraph_text(content)
    old_parts = [paragraph_text(row.get("text")) for row in rows]
    old_text = "".join(old_parts)
    if len(rows) == 1 or not old_text:
        rows[0]["text"] = new_text
        for row in rows[1:]:
            row["text"] = ""
    else:
        boundaries = [0]
        for part in old_parts:
            boundaries.append(boundaries[-1] + len(part))
        opcodes = SequenceMatcher(None, old_text, new_text, autojunk=False).get_opcodes()

        def mapped_boundary(boundary: int) -> int:
            if boundary <= 0:
                return 0
            if boundary >= len(old_text):
                return len(new_text)
            insertion_end: int | None = None
            for tag, old_start, old_end, new_start, new_end in opcodes:
                if tag == "insert" and old_start == boundary:
                    insertion_end = new_end
                    continue
                if old_start <= boundary <= old_end and old_end > old_start:
                    if tag == "equal":
                        return new_start + boundary - old_start
                    ratio = (boundary - old_start) / (old_end - old_start)
                    return new_start + round(ratio * (new_end - new_start))
                if old_start > boundary:
                    break
            return insertion_end if insertion_end is not None else len(new_text)

        mapped = [mapped_boundary(boundary) for boundary in boundaries]
        for index in range(1, len(mapped)):
            mapped[index] = max(mapped[index - 1], min(len(new_text), mapped[index]))
        mapped[-1] = len(new_text)
        for index, row in enumerate(rows):
            # 保留映射边界上的空格；去掉它会改变人工编辑后的中英文正文。
            row["text"] = new_text[mapped[index]:mapped[index + 1]]
    cleaned["full_text"] = "\n".join(
        str(row.get("text") or "") for row in all_rows if str(row.get("text") or "")
    )


@locked_video_action
def save_chapter_markdown(
    paths: PipelinePaths,
    video_id: str,
    segment_no: int,
    markdown: str,
    expected_updated_at: str | None = None,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """兼容旧 Markdown 写接口；新界面调用结构化字段接口。"""
    title_match = re.search(r"^##\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    if not title_match:
        raise ValueError("Markdown 第一项必须是章节标题，例如：## 章节标题")
    title = re.sub(r"^\d+[.、]\s*", "", title_match.group(1)).strip()
    if not title or len(title) > 40:
        raise ValueError("章节标题不能为空且不能超过 40 个字符")
    summary = paragraph_text(_markdown_section(markdown, "章节摘要", "章节正文"))
    content = markdown_chapter_content(markdown)
    if not summary or len(summary) > 160:
        raise ValueError("章节摘要不能为空且不能超过 160 个字符")
    if not content:
        raise ValueError("章节正文不能为空")
    keywords_match = re.search(r"^>\s*关键词[：:]\s*(.*?)\s*$", markdown, flags=re.MULTILINE)
    keywords = []
    if keywords_match:
        for value in re.split(r"[,，、;；]+", keywords_match.group(1)):
            normalized = value.strip()
            if normalized and normalized != "无" and normalized not in keywords:
                if len(normalized) > 24:
                    raise ValueError(f"关键词“{normalized}”超过 24 个字符")
                keywords.append(normalized)

    return save_chapter_fields(
        paths, video_id, segment_no, title=title, summary=summary,
        content=content, keywords=keywords,
        expected_updated_at=expected_updated_at,
        expected_revision=expected_revision,
    )


@locked_video_action
def save_chapter_fields(
    paths: PipelinePaths, video_id: str, segment_no: int, *,
    title: str, summary: str, content: str, keywords: list[str],
    expected_revision: int | None = None,
    expected_updated_at: str | None = None,
) -> dict[str, Any]:
    """保存章节字段，并将正文映射回原句时间轴及派生 Markdown。"""
    segments_path, document = _required_document(paths, video_id, "segments")
    if expected_revision is not None and int(document.get("revision", 0)) != expected_revision:
        raise EditConflictError("章节已有新版本，请刷新后再保存")
    if expected_updated_at and document.get("updated_at") != expected_updated_at:
        raise EditConflictError("章节已有新版本，请刷新后再保存")
    if expected_revision is None and not expected_updated_at and (
        int(document.get("revision", 0)) != 0 or document.get("updated_at")
    ):
        raise EditConflictError("章节已有新版本，请刷新页面后再保存")
    rows = document.get("segments")
    if not isinstance(rows, list) or segment_no < 1 or segment_no > len(rows):
        raise KeyError(f"找不到第 {segment_no} 章")
    title = paragraph_text(title)
    summary = paragraph_text(summary)
    content = paragraph_text(content)
    keywords = list(dict.fromkeys(paragraph_text(value) for value in keywords))
    if not title or len(title) > 40:
        raise ValueError("章节标题不能为空且不能超过 40 个字符")
    if not summary or len(summary) > 160:
        raise ValueError("章节摘要不能为空且不能超过 160 个字符")
    if not content:
        raise ValueError("章节正文不能为空")
    if any(not value or len(value) > 24 for value in keywords):
        raise ValueError("关键词不能为空且不能超过 24 个字符")
    row = rows[segment_no - 1]
    original_content = paragraph_text(row.get("content"))
    raw_path, raw = _required_document(paths, video_id, "raw")
    cleaned_path, cleaned = _required_document(paths, video_id, "cleaned")
    del raw_path
    _redistribute_chapter_content(
        cleaned,
        int(row.get("start_sentence_id", -1)),
        int(row.get("end_sentence_id", -1)),
        content,
    )
    _raise_validation(validate_cleaned(raw, cleaned))
    rows[segment_no - 1] = {
        **row,
        "title": title,
        "summary": summary,
        "keywords": keywords,
        "content": content,
        "manual_content_override": content != original_content or bool(row.get("manual_content_override")),
        "edited_at": now_iso(),
    }
    document["review_status"] = "pending_review"
    document["updated_at"] = now_iso()
    document["revision"] = int(document.get("revision", 0)) + 1
    document.pop("confirmed_at", None)
    # 用户可以完全重写章节；先确认程序已原样分配正文，再只检查时间与范围。
    redistributed = "".join(
        str(item.get("text") or "") for item in cleaned["sentences"]
        if int(row["start_sentence_id"]) <= int(item["sentence_id"]) <= int(row["end_sentence_id"])
    )
    if redistributed != content:
        raise RuntimeError("章节正文同步失败，修改未保存")
    _raise_validation(validate_segments(cleaned, document, check_content=False))
    markdown_path = paths.artifact(video_id, "markdown")
    assert markdown_path is not None
    json_files = {cleaned_path: cleaned, segments_path: document}
    history_path, history = _current_history(paths, video_id, cleaned)
    if history_path is not None:
        json_files[history_path] = history
    commit_artifacts(
        paths, video_id,
        json_files=json_files,
        text_files={markdown_path: segment_markdown(document)},
    )
    return document


@locked_video_action
def confirm_segments(paths: PipelinePaths, video_id: str) -> dict[str, Any]:
    """校验并确认整视频章节，同时确认尚未逐项决定的建议。"""
    _, raw = _required_document(paths, video_id, "raw")
    _, cleaned = _required_document(paths, video_id, "cleaned")
    segments_path, document = _required_document(paths, video_id, "segments")
    _raise_validation(validate_cleaned(raw, cleaned) + validate_segments(cleaned, document))
    history_path = paths.artifact(video_id, "history")
    history = load_json(history_path) if history_path and history_path.is_file() else None
    if isinstance(history, dict):
        # 整视频确认表示接受尚未逐项决定的暂用建议，逐项决定仍保留原记录。
        for change in history.get("changes", []):
            if isinstance(change, dict) and change.get("status") == "pending_review" and change.get("decision") in {"pending", "auto_approved"}:
                change["decision"] = "confirmed_by_video"
                change["decided_at"] = now_iso()
        decisions = Counter(
            str(change.get("decision", ""))
            for change in history.get("changes", [])
            if isinstance(change, dict) and change.get("status") == "pending_review"
        )
        history["decisions"] = {
            key: decisions[key] for key in
            ("approved", "rejected", "auto_approved", "confirmed_by_video")
            if decisions[key]
        }
        history.setdefault("counts", {})["auto_approved"] = decisions["auto_approved"]
        history["counts"]["confirmed_by_video"] = decisions["confirmed_by_video"]
        history["counts"]["pending_review"] = decisions["pending"] + decisions["auto_approved"]
        history["final"] = cleaned
        history["updated_at"] = now_iso()
    document["review_status"] = "confirmed"
    document["confirmed_at"] = now_iso()
    document["updated_at"] = document["confirmed_at"]
    document["revision"] = int(document.get("revision", 0)) + 1
    markdown_path = paths.artifact(video_id, "markdown")
    assert markdown_path is not None
    json_files = {segments_path: document}
    if isinstance(history, dict) and history_path is not None:
        json_files[history_path] = history
    commit_artifacts(
        paths, video_id, json_files=json_files,
        text_files={markdown_path: segment_markdown(document)},
    )
    return document


@locked_video_action
def decide_archived_suggestion(
    paths: PipelinePaths,
    video_id: str,
    sentence_id: int,
    decision: str,
    approved_text: str | None = None,
) -> dict[str, Any]:
    """复核归档的清洗建议，并同步清洗文本和章节派生文件。"""
    history_path, history = _required_document(paths, video_id, "history")
    match = next((
        item for item in history.get("changes", [])
        if isinstance(item, dict)
        and int(item.get("sentence_id", -1)) == sentence_id
        and item.get("status") == "pending_review"
    ), None)
    if match is None:
        raise KeyError(f"sentence_id={sentence_id} 没有可复核的 AI 建议")
    if decision not in {"approved", "rejected"}:
        raise ValueError("审核决定必须是 approved 或 rejected")
    if decision == "approved":
        effective_text = paragraph_text(
            approved_text or match.get("proposed_text") or match.get("raw_text")
        )
        if not effective_text:
            raise ValueError("同意后的文本不能为空")
        match["approved_text"] = effective_text
    else:
        effective_text = paragraph_text(match.get("raw_text"))
        match.pop("approved_text", None)
    proposed_text = paragraph_text(match.get("proposed_text") or match.get("raw_text"))
    # 自定义文本立即写入清洗结果，后续章节差异已无法识别这次改动；
    # 先保留词表复核标记，待章节保存时按普通编辑流程提示。
    match["glossary_review_pending"] = bool(
        decision == "approved"
        and approved_text is not None
        and effective_text != proposed_text
    )
    # 复核决定会修改底层清洗句子，必须同步重建章节正文；
    # 否则早前的人工章节内容可能仍显示旧建议。
    updated = save_cleaned_sentence(
        paths, video_id, sentence_id, effective_text,
        persist=False,
    )
    cleaned = updated["cleaned"]
    match["decision"] = decision
    match["effective_text"] = effective_text
    match["decided_at"] = now_iso()
    decisions = Counter(
        str(item.get("decision", ""))
        for item in history.get("changes", [])
        if isinstance(item, dict) and item.get("status") == "pending_review"
    )
    history["decisions"] = {
        key: decisions.get(key, 0)
        for key in ("approved", "rejected", "auto_approved")
        if decisions.get(key, 0)
    }
    history.setdefault("counts", {})["auto_approved"] = decisions.get("auto_approved", 0)
    history["counts"]["approved"] = decisions.get("approved", 0)
    history["counts"]["rejected"] = decisions.get("rejected", 0)
    history["counts"]["pending_review"] = decisions.get("pending", 0) + decisions.get("auto_approved", 0)
    history["final"] = cleaned
    history["updated_at"] = now_iso()
    json_files = {history_path: history, paths.artifact(video_id, "cleaned"): cleaned}
    text_files = {}
    if updated["segments"] is not None:
        json_files[paths.artifact(video_id, "segments")] = updated["segments"]
        text_files[paths.artifact(video_id, "markdown")] = segment_markdown(updated["segments"])
    commit_artifacts(paths, video_id, json_files=json_files, text_files=text_files)
    return history


@locked_video_action
def mark_custom_glossary_reviewed(
    paths: PipelinePaths, video_id: str, segment_no: int
) -> None:
    """章节保存后不再重复提示已处理的自定义术语候选。"""
    _, segments = _required_document(paths, video_id, "segments")
    rows = segments.get("segments")
    if not isinstance(rows, list) or segment_no < 1 or segment_no > len(rows):
        raise KeyError(f"找不到第 {segment_no} 章")
    row = rows[segment_no - 1]
    start_id = int(row.get("start_sentence_id", -1))
    end_id = int(row.get("end_sentence_id", -1))
    history_path = paths.artifact(video_id, "history")
    if history_path is None or not history_path.is_file():
        return
    history = load_json(history_path)
    changed = False
    for item in history.get("changes", []):
        if not isinstance(item, dict):
            continue
        sentence_id = int(item.get("sentence_id", -1))
        approved = paragraph_text(item.get("approved_text"))
        proposed = paragraph_text(item.get("proposed_text") or item.get("raw_text"))
        is_custom = item.get("decision") == "approved" and approved and approved != proposed
        if start_id <= sentence_id <= end_id and is_custom and item.get("glossary_review_pending") is not False:
            item["glossary_review_pending"] = False
            item["glossary_reviewed_at"] = now_iso()
            changed = True
    if changed:
        history["updated_at"] = now_iso()
        commit_artifacts(paths, video_id, json_files={history_path: history})


CHAPTER_METADATA_SYSTEM_PROMPT = """你是视频知识库章节编辑。根据用户编辑后的章节正文，生成准确、具体、可检索的中文标题、忠于正文的一至两句摘要和必要关键词。标题不超过40个字符，摘要不超过160个字符，每个关键词不超过24个字符。不得添加正文没有的事实。只输出 JSON：
{"labels":[{"position":1,"title":"具体标题","summary":"忠实摘要。","keywords":["关键词"]}]}"""


def generate_chapter_metadata(
    paths: PipelinePaths,
    video_id: str,
    segment_no: int,
    markdown: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """从当前编辑内容生成标题、摘要和关键词候选，不直接写盘。"""
    _, document = _required_document(paths, video_id, "segments")
    rows = document.get("segments")
    if not isinstance(rows, list) or segment_no < 1 or segment_no > len(rows):
        raise KeyError(f"找不到第 {segment_no} 章")
    content = markdown_chapter_content(markdown)
    if not content:
        raise ValueError("章节正文为空，无法生成标题、关键词和摘要")
    settings = cloud_settings(config)
    api_key = str(settings["api_key"])
    if not api_key:
        raise RuntimeError("API 尚未连接，请点击右上角“API连接”")
    model = str(settings["model"])
    timeout = float(settings["timeout_seconds"])
    api_url = str(settings["api_url"])
    prompt = "请为以下编辑后的章节生成新标签：\n" + json.dumps({
        "video_title": str(document.get("title", "")),
        "chapter_number": segment_no,
        "chapter_content": content,
    }, ensure_ascii=False, separators=(",", ":"))
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=20.0)) as client:
        request = partial(call_glm_once, api_url=api_url)
        labels, usage = validated_request(
            client, api_key, model, CHAPTER_METADATA_SYSTEM_PROMPT, prompt,
            1024, bool(settings["thinking"]), 3,
            lambda value: validate_label_response(value, [1]),
            "章节标题摘要生成", request=request,
        )
    return {"label": labels[1], "model": model, "usage": dict(usage)}
