"""Shared semantic-segmentation models, GLM calls, and validation helpers."""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import httpx

from ..shared.glm import DEFAULT_API_URL, request_json
from ..shared.progress import emit_progress
from ..shared.text import paragraph_text

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


SEGMENT_BOUNDARY_SYSTEM_PROMPT = """你是视频章节编辑。通读全部清洗后句子，按独立主题、完整问答或完整操作划分连续章节。语义完整优先，字数只用于防止章节过长或过碎。

- 全文不超过1300字，保持一个章节；不要因其中包含多个相关子功能而拆分。
- 1301至2200字通常划分为2章；只有确实存在3个独立内容单元且每章均不少于500字时才分3章，不得切成4章以上。
- 更长文本的单章以700至1100字为宜；超过1200字时，应优先在附近的自然语义边界拆分。
- 不要为了凑字数切断完整问题、完整回答、操作步骤、校验修改或连续业务场景。
- 原则上不得形成不足500字的章节；应移动到附近更合适的自然边界或并入相邻章节。引入、过渡语和结束语不得单独成章。
- 一个功能或操作从开始到完成应保持在同一章；只有转入另一个功能、问题或业务主题时才开始新章。

输出前根据 start_char、end_char 自检各章字数；若不符合上述区间，先移动边界或合并，再输出最终结果。

你只返回每个章节的起始句 ID。程序会自动把每段延伸到下一段起始句的前一句，最后一段自动覆盖到全文末句。第一个起始 ID 必须是输入第一条 sentence_id，后续 ID 严格递增。

输入包含 total_chars；sentences 中每项为 [sentence_id, start_char, end_char, text]。start_char 和 end_char 是去除空白和标点后的累计字符位置，不需要自行数数。
只输出 JSON，不要解释：
{"start_sentence_ids":[1,18]}"""


SEGMENT_LABEL_SYSTEM_PROMPT = """你是视频知识库的语义片段标签编辑。片段边界已经固定，不得修改、合并或拆分。

请为输入的每个片段生成：
- 具体、可检索的中文标题，准确表达该片段的问题、操作或主题，避免“功能介绍”“第一部分”等空泛表述；
- 忠于原文的一至两句摘要，不添加原文没有的事实；
- 按内容需要生成关键词，优先覆盖业务术语、模块名、操作对象、关键动作和核心概念，不为凑数添加泛词。

必须按照输入 position 逐一返回，不得遗漏、重复或增加片段。标题不超过40个字符，摘要不超过160个字符，每个关键词不超过24个字符。
只输出一个 JSON 对象，不要输出 Markdown 或解释：
{"labels":[{"position":1,"title":"具体标题","summary":"忠实摘要。","keywords":["关键词"]}]}
每个标签只能包含 position、title、summary、keywords。"""


BOUNDARY_EXPLANATION_SUFFIX = """

本次是一次性对比实验。除 start_sentence_ids 外，请额外给出简短、可核查的编辑依据；不要输出内部推理过程。说明每个章节为何从该句开始，并概括未在相邻候选位置拆分或合并的主要语义与长度依据。
只输出以下 JSON 结构：
{"start_sentence_ids":[1,18],"overall_basis":"整体划分原则的简短说明","boundary_explanations":[{"start_sentence_id":1,"reason":"该处开始独立主题"},{"start_sentence_id":18,"reason":"此处转入新的完整业务主题"}]}"""


class SegmentResponseError(ValueError):
    """GLM response does not satisfy the segment boundary or label contract."""


def effective_char_count(text: str) -> int:
    """Count content characters after excluding whitespace and punctuation."""
    return sum(
        1 for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def meaningful_sentences(source: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sentences = source.get("sentences")
    if not isinstance(raw_sentences, list):
        raise ValueError("cleaned ASR 缺少 sentences 数组")

    result: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    last_id: int | None = None
    last_start = -1
    for position, raw in enumerate(raw_sentences, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {position} 个 sentence 不是对象")
        try:
            sentence_id = int(raw["sentence_id"])
            start_ms = int(raw["start_ms"])
            end_ms = int(raw["end_ms"])
            text = str(raw["text"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"第 {position} 个 sentence 字段非法") from error
        if sentence_id in seen_ids or (last_id is not None and sentence_id <= last_id):
            raise ValueError(f"sentence_id 未严格递增：{sentence_id}")
        if start_ms < 0 or end_ms <= start_ms or start_ms < last_start:
            raise ValueError(f"句子时间戳非法：sentence_id={sentence_id}")
        seen_ids.add(sentence_id)
        last_id = sentence_id
        last_start = start_ms
        if text.strip():
            result.append({
                "sentence_id": sentence_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text.strip(),
            })
    return result


def transcript_payload(source: dict[str, Any], sentences: list[dict[str, Any]]) -> dict[str, Any]:
    compact_sentences: list[list[Any]] = []
    character_offset = 0
    for row in sentences:
        start_char = character_offset
        character_offset += effective_char_count(row["text"])
        compact_sentences.append([
            row["sentence_id"], start_char, character_offset, row["text"]
        ])
    return {
        "video_title": str(source.get("title", "")).strip(),
        "sentence_count": len(sentences),
        "total_chars": character_offset,
        "sentences": compact_sentences,
    }


def segment_prompt_for(source: dict[str, Any], sentences: list[dict[str, Any]]) -> str:
    payload = transcript_payload(source, sentences)
    return "请划分以下视频的语义片段：\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def _as_int(value: Any, field: str, item_number: int) -> int:
    if isinstance(value, bool):
        raise SegmentResponseError(f"第 {item_number} 项的 {field} 非法")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise SegmentResponseError(f"第 {item_number} 项的 {field} 非法") from error


def validate_segment_response(value: Any, sentences: list[dict[str, Any]]) -> list[int]:
    if not isinstance(value, dict) or "start_sentence_ids" not in value:
        raise SegmentResponseError("片段结果必须包含 start_sentence_ids")
    raw_starts = value["start_sentence_ids"]
    if not isinstance(raw_starts, list) or (sentences and not raw_starts):
        raise SegmentResponseError("start_sentence_ids 必须是非空数组")
    if not sentences:
        if raw_starts:
            raise SegmentResponseError("空文本不能生成片段")
        return []
    starts = [
        _as_int(raw, "start_sentence_ids", number)
        for number, raw in enumerate(raw_starts, 1)
    ]
    valid_ids = {row["sentence_id"] for row in sentences}
    if starts[0] != sentences[0]["sentence_id"]:
        raise SegmentResponseError(
            f"segment_starts 必须从 sentence_id={sentences[0]['sentence_id']} 开始"
        )
    if any(value not in valid_ids for value in starts):
        raise SegmentResponseError("segment_starts 引用了不存在或空白的句子")
    if any(right <= left for left, right in zip(starts, starts[1:])):
        raise SegmentResponseError("segment_starts 必须严格递增且不能重复")
    return starts


def validate_boundary_explanations(
    value: Any,
    starts: list[int],
) -> dict[str, Any]:
    """Validate concise editorial rationales requested for a one-time audit."""
    if not isinstance(value, dict):
        raise SegmentResponseError("分段说明必须是对象")
    overall_basis = str(value.get("overall_basis") or "").strip()
    raw_rows = value.get("boundary_explanations")
    if not overall_basis or not isinstance(raw_rows, list):
        raise SegmentResponseError("一次性分段说明缺少 overall_basis 或 boundary_explanations")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item_number, raw in enumerate(raw_rows, 1):
        if not isinstance(raw, dict):
            raise SegmentResponseError(f"第 {item_number} 条分段说明不是对象")
        start_id = _as_int(raw.get("start_sentence_id"), "start_sentence_id", item_number)
        reason = str(raw.get("reason") or "").strip()
        if start_id not in starts or start_id in seen or not reason:
            raise SegmentResponseError(f"start_sentence_id={start_id} 的分段说明缺失或重复")
        seen.add(start_id)
        rows.append({"start_sentence_id": start_id, "reason": reason[:600]})
    if seen != set(starts):
        raise SegmentResponseError("分段说明未覆盖全部章节起点")
    return {"overall_basis": overall_basis[:1200], "boundary_explanations": rows}


def ranges_from_starts(starts: list[int], sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positions = {row["sentence_id"]: index for index, row in enumerate(sentences)}
    ranges: list[dict[str, Any]] = []
    for position, start_id in enumerate(starts, 1):
        start = positions[start_id]
        end = positions[starts[position]] - 1 if position < len(starts) else len(sentences) - 1
        rows = sentences[start:end + 1]
        ranges.append({
            "position": position,
            "start_sentence_id": rows[0]["sentence_id"],
            "end_sentence_id": rows[-1]["sentence_id"],
            "text": "\n".join(row["text"] for row in rows),
        })
    return ranges


def label_prompt_for(source: dict[str, Any], ranges: list[dict[str, Any]]) -> str:
    payload = {
        "video_title": str(source.get("title", "")).strip(),
        "fixed_segments": ranges,
    }
    return "请为以下固定片段生成标签：\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def validate_label_response(value: Any, expected_positions: list[int]) -> dict[int, dict[str, Any]]:
    if not isinstance(value, dict) or "labels" not in value:
        raise SegmentResponseError("结果必须包含 labels 数组")
    raw_labels = value["labels"]
    if not isinstance(raw_labels, list):
        raise SegmentResponseError("labels 必须是数组")
    expected_fields = {"position", "title", "summary", "keywords"}
    labels: dict[int, dict[str, Any]] = {}
    for item_number, raw in enumerate(raw_labels, 1):
        if not isinstance(raw, dict) or not expected_fields.issubset(raw):
            raise SegmentResponseError(f"第 {item_number} 个标签缺少必要字段")
        position = _as_int(raw["position"], "position", item_number)
        if position not in expected_positions or position in labels:
            raise SegmentResponseError(f"标签 position={position} 不属于本批次或发生重复")
        title = raw["title"].strip() if isinstance(raw["title"], str) else ""
        summary = raw["summary"].strip() if isinstance(raw["summary"], str) else ""
        keywords = raw["keywords"]
        if not title or len(title) > 40:
            raise SegmentResponseError(f"position={position} 的标题为空或超过40字符")
        if not summary or len(summary) > 160:
            raise SegmentResponseError(f"position={position} 的摘要为空或超过160字符")
        if isinstance(keywords, str):
            keywords = re.split(r"[,，、;；\n]+", keywords)
        if not isinstance(keywords, list):
            raise SegmentResponseError(f"position={position} 的 keywords 非法")
        clean_keywords: list[str] = []
        for keyword in keywords:
            if not isinstance(keyword, str):
                continue
            normalized = keyword.strip()
            if not normalized or len(normalized) > 24:
                continue
            if normalized not in clean_keywords:
                clean_keywords.append(normalized)
        labels[position] = {"title": title, "summary": summary, "keywords": clean_keywords}
    if set(labels) != set(expected_positions):
        missing = sorted(set(expected_positions) - set(labels))
        raise SegmentResponseError(f"标签批次缺少 position={missing}")
    return labels


def make_label_batches(
    ranges: list[dict[str, Any]], max_chars: int, max_segments: int
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    characters = 0
    for segment in ranges:
        size = len(segment["text"])
        if current and (len(current) >= max_segments or characters + size > max_chars):
            batches.append(current)
            current = []
            characters = 0
        current.append(segment)
        characters += size
    if current:
        batches.append(current)
    return batches


def materialize_segments(
    sentences: list[dict[str, Any]],
    plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row["sentence_id"]: position for position, row in enumerate(sentences)}
    result: list[dict[str, Any]] = []
    for segment_no, label in enumerate(plan, 1):
        start = by_id[label["start_sentence_id"]]
        end = by_id[label["end_sentence_id"]]
        rows = sentences[start:end + 1]
        result.append({
            "segment_no": segment_no,
            "title": label["title"],
            "summary": label["summary"],
            "keywords": label["keywords"],
            "start_sentence_id": rows[0]["sentence_id"],
            "end_sentence_id": rows[-1]["sentence_id"],
            "start_ms": rows[0]["start_ms"],
            "end_ms": rows[-1]["end_ms"],
            "content": paragraph_text("\n".join(row["text"] for row in rows)),
        })
    return result


def find_video_path(video_root: Path, partition: str, video_id: str) -> str | None:
    partition_root = video_root / partition
    if not partition_root.is_dir():
        return None
    matches = sorted(path for path in partition_root.glob(f"*__{video_id[:16]}.*") if path.is_file())
    if len(matches) > 1:
        raise RuntimeError(f"video_id={video_id} 匹配到多个视频文件")
    if not matches:
        return None
    try:
        return matches[0].relative_to(video_root.parent).as_posix()
    except ValueError:
        return str(matches[0].resolve())


def build_document(
    source: dict[str, Any],
    sentences: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    video_path: str | None,
) -> dict[str, Any]:
    video_id = str(source.get("source_id", "")).strip()
    if not video_id:
        raise ValueError("cleaned ASR 缺少 source_id")
    return {
        "video_id": video_id,
        "title": str(source.get("title", "")).strip(),
        "video_path": video_path,
        "segments": materialize_segments(sentences, plan),
    }


def call_glm_once(
    client: httpx.Client,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    thinking: bool,
    api_url: str = DEFAULT_API_URL,
) -> tuple[Any, dict[str, int]]:
    return request_json(
        client,
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        thinking=thinking,
        attempts=1,
    )


def validated_request(
    client: httpx.Client,
    api_key: str,
    model: str,
    system_prompt: str,
    base_prompt: str,
    max_tokens: int,
    thinking: bool,
    attempts: int,
    validator: Callable[[Any], Any],
    progress_label: str,
    request: Callable[..., tuple[Any, dict[str, int]]] = call_glm_once,
) -> tuple[Any, Counter[str]]:
    last_error: Exception | None = None
    usage_totals: Counter[str] = Counter()
    for attempt in range(1, attempts + 1):
        correction = ""
        if last_error is not None:
            correction = (
                "\n\n上一次输出校验失败，请重新输出完整结果并修正此问题："
                + str(last_error)
            )
        try:
            value, usage = request(
                client, api_key, model, system_prompt, base_prompt + correction,
                max_tokens, thinking,
            )
            usage_totals.update(usage)
            return validator(value), usage_totals
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError,
                RuntimeError, json.JSONDecodeError, SegmentResponseError) as error:
            last_error = error
            print(
                f"  {progress_label} attempt={attempt}/{attempts} failed: {error}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(min(10, attempt * 2))
    raise RuntimeError(f"{progress_label} 在 {attempts} 次尝试后仍失败：{last_error}")


def generate_segments(
    client: httpx.Client,
    api_key: str,
    model: str,
    source: dict[str, Any],
    sentences: list[dict[str, Any]],
    max_tokens: int,
    thinking: bool,
    attempts: int,
    label_batch_max_chars: int,
    label_batch_max_segments: int,
    request: Callable[..., tuple[Any, dict[str, int]]] = call_glm_once,
    boundary_audit: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    prompt = segment_prompt_for(source, sentences)
    emit_progress("segments", 0, 100, "正在等待模型规划章节边界")

    def validate_boundaries(value: Any) -> list[int]:
        starts = validate_segment_response(value, sentences)
        if boundary_audit is not None:
            boundary_audit.update(validate_boundary_explanations(value, starts))
            boundary_audit["start_sentence_ids"] = starts
        return starts

    starts, usage_totals = validated_request(
        client, api_key, model,
        SEGMENT_BOUNDARY_SYSTEM_PROMPT + (BOUNDARY_EXPLANATION_SUFFIX if boundary_audit is not None else ""),
        prompt, max_tokens, thinking, attempts,
        validate_boundaries,
        "语义片段边界规划", request,
    )
    ranges = ranges_from_starts(starts, sentences)
    emit_progress("segments", 35, 100, f"章节边界规划完成，共 {len(ranges)} 章")
    batches = make_label_batches(
        ranges, label_batch_max_chars, label_batch_max_segments
    )
    labels: dict[int, dict[str, Any]] = {}
    for batch_number, batch in enumerate(batches, 1):
        emit_progress(
            "segments", 35 + 60 * (batch_number - 1) / max(len(batches), 1), 100,
            f"正在等待第 {batch_number}/{len(batches)} 批章节标题与摘要",
        )
        positions = [segment["position"] for segment in batch]
        batch_labels, usage = validated_request(
            client, api_key, model, SEGMENT_LABEL_SYSTEM_PROMPT,
            label_prompt_for(source, batch), max_tokens, thinking, attempts,
            lambda value, expected=positions: validate_label_response(value, expected),
            f"语义片段标签 {batch_number}/{len(batches)}", request,
        )
        labels.update(batch_labels)
        usage_totals.update(usage)
        emit_progress(
            "segments", 35 + 60 * batch_number / max(len(batches), 1), 100,
            f"已生成第 {batch_number}/{len(batches)} 批章节标签",
        )
    plan = [{
        "start_sentence_id": segment["start_sentence_id"],
        "end_sentence_id": segment["end_sentence_id"],
        **labels[segment["position"]],
    } for segment in ranges]
    return plan, usage_totals


def select_sources(input_root: Path, video_id: str | None, limit: int | None) -> list[Path]:
    # cleaned_asr/<partition>/manifest.json is aggregate metadata, not a video source.
    paths = sorted(path for path in input_root.glob("*.json") if path.name != "manifest.json")
    if video_id:
        paths = [path for path in paths if path.stem == video_id]
        if not paths:
            raise FileNotFoundError(f"找不到 cleaned ASR：{video_id}")
    if limit is not None:
        paths = paths[:limit]
    return paths
