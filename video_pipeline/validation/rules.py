"""Cross-stage validation for one file-based video record."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..shared.text import paragraph_text


def validate_raw(raw: dict[str, Any], expected_id: str | None = None) -> list[str]:
    errors: list[str] = []
    source_id = str(raw.get("id", ""))
    if expected_id and source_id != expected_id:
        errors.append(f"asr_raw.id={source_id!r} 与文件名 {expected_id!r} 不一致")
    sentences = raw.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        return [*errors, "asr_raw.sentences 为空或不是数组"]
    previous_end = -1
    for position, sentence in enumerate(sentences, 1):
        if int(sentence.get("id", -1)) != position:
            errors.append(f"asr_raw 第 {position} 句 id 不连续")
        start = int(sentence.get("start_ms", -1))
        end = int(sentence.get("end_ms", -1))
        if start < 0 or end <= start:
            errors.append(f"asr_raw sentence_id={position} 时间戳非法：{start}-{end}")
        if start < previous_end:
            errors.append(f"asr_raw sentence_id={position} 与前句时间重叠")
        if not isinstance(sentence.get("raw_text"), str):
            errors.append(f"asr_raw sentence_id={position} 缺少 raw_text")
        previous_end = end
    duration = int(raw.get("duration_ms", 0))
    if duration and previous_end > duration + 1000:
        errors.append(f"末句结束时间 {previous_end} 超过视频时长 {duration}")
    return errors


def validate_cleaned(raw: dict[str, Any], cleaned: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if str(cleaned.get("source_id", "")) != str(raw.get("id", "")):
        errors.append("cleaned_asr.source_id 与 asr_raw.id 不一致")
    raw_sentences = raw.get("sentences") or []
    clean_sentences = cleaned.get("sentences")
    if not isinstance(clean_sentences, list) or len(clean_sentences) != len(raw_sentences):
        return [*errors, "cleaned_asr 句子数量与 asr_raw 不一致"]
    for raw_sentence, clean_sentence in zip(raw_sentences, clean_sentences, strict=True):
        expected = (
            int(raw_sentence["id"]), int(raw_sentence["start_ms"]), int(raw_sentence["end_ms"])
        )
        actual = (
            int(clean_sentence.get("sentence_id", -1)),
            int(clean_sentence.get("start_ms", -1)),
            int(clean_sentence.get("end_ms", -1)),
        )
        if actual != expected:
            errors.append(f"cleaned_asr sentence_id={expected[0]} 的 ID 或时间戳发生变化")
        if not isinstance(clean_sentence.get("text"), str):
            errors.append(f"cleaned_asr sentence_id={expected[0]} 缺少 text")
    expected_full = "\n".join(str(item["text"]) for item in clean_sentences if item["text"])
    if cleaned.get("full_text") != expected_full:
        errors.append("cleaned_asr.full_text 与逐句文本不一致")
    return errors


def validate_segments(
    cleaned: dict[str, Any], document: dict[str, Any], *, check_content: bool = True
) -> list[str]:
    errors: list[str] = []
    if str(document.get("video_id", "")) != str(cleaned.get("source_id", "")):
        errors.append("semantic_segments.video_id 与 cleaned_asr.source_id 不一致")
    meaningful = [item for item in cleaned.get("sentences", []) if str(item.get("text", "")).strip()]
    segments = document.get("segments")
    if not meaningful:
        return errors if segments == [] else [*errors, "空文本视频不应生成语义片段"]
    if not isinstance(segments, list) or not segments:
        return [*errors, "semantic_segments.segments 为空或不是数组"]
    all_by_id = {int(item["sentence_id"]): item for item in cleaned.get("sentences", [])}
    previous_end: int | None = None
    for position, segment in enumerate(segments, 1):
        if int(segment.get("segment_no", -1)) != position:
            errors.append(f"第 {position} 个片段 segment_no 不连续")
        start = int(segment.get("start_sentence_id", -1))
        end = int(segment.get("end_sentence_id", -1))
        if end < start or start not in all_by_id or end not in all_by_id:
            errors.append(f"第 {position} 个片段句子范围不连续：{start}-{end}")
        if previous_end is not None and start <= previous_end:
            errors.append(f"第 {position} 个片段与上一片段重叠：{start}-{end}")
        included = [item for item in meaningful if start <= int(item["sentence_id"]) <= end]
        expected_content = "\n".join(str(item["text"]) for item in included)
        # 自动分章的逐句换行可在英文句间补空格；人工编辑后的句子可能
        # 正好切在一个英文词内部，此时连续拼接才是用户保存的原文。
        accepted_content = {paragraph_text(expected_content)}
        if segment.get("manual_content_override"):
            all_parts = [
                str(item.get("text") or "") for item in cleaned.get("sentences", [])
                if start <= int(item["sentence_id"]) <= end
            ]
            accepted_content.add(paragraph_text("".join(all_parts)))
        if check_content and paragraph_text(segment.get("content")) not in accepted_content:
            errors.append(f"第 {position} 个片段 content 与 cleaned_asr 不一致")
        if start in all_by_id and end in all_by_id:
            if int(segment.get("start_ms", -1)) != int(all_by_id[start]["start_ms"]):
                errors.append(f"第 {position} 个片段 start_ms 不一致")
            if int(segment.get("end_ms", -1)) != int(all_by_id[end]["end_ms"]):
                errors.append(f"第 {position} 个片段 end_ms 不一致")
        previous_end = end
    for item in meaningful:
        sentence_id = int(item["sentence_id"])
        covering = sum(
            int(segment.get("start_sentence_id", -1)) <= sentence_id <= int(segment.get("end_sentence_id", -1))
            for segment in segments
        )
        if covering != 1:
            errors.append(f"非空句 sentence_id={sentence_id} 未被唯一章节覆盖")
    return errors


def validate_video_path(document: dict[str, Any], output_root: Path) -> list[str]:
    value = document.get("video_path")
    if not value:
        return ["semantic_segments.video_path 为空"]
    path = Path(str(value))
    if not path.is_absolute():
        path = output_root / path
    return [] if path.is_file() else [f"视频文件不存在：{path}"]
