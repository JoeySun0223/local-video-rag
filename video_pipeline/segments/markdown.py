"""Render semantic-segment JSON as a human-readable Markdown derivative."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..shared.text import paragraph_text


def _inline(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").splitlines()).strip()


def _timecode(milliseconds: Any) -> str:
    seconds = max(0, int(milliseconds or 0) // 1000)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def segment_markdown(document: dict[str, Any]) -> str:
    """Build the complete Markdown view for one semantic-segment document."""
    title = _inline(document.get("title")) or "未命名视频"
    video_id = _inline(document.get("video_id"))
    segments = document.get("segments")
    rows = segments if isinstance(segments, list) else []
    lines = [f"# {title}", ""]
    if video_id:
        lines.extend([f"> 视频 ID：`{video_id}`  ", f"> 片段数：{len(rows)}", ""])
    for position, segment in enumerate(rows, 1):
        if not isinstance(segment, dict):
            continue
        number = int(segment.get("segment_no", position))
        segment_title = _inline(segment.get("title")) or f"片段 {number}"
        start_id = segment.get("start_sentence_id", "—")
        end_id = segment.get("end_sentence_id", "—")
        start_time = _timecode(segment.get("start_ms"))
        end_time = _timecode(segment.get("end_ms"))
        keywords = segment.get("keywords")
        keyword_text = "、".join(
            _inline(item) for item in keywords or [] if _inline(item)
        ) if isinstance(keywords, list) else _inline(keywords)
        summary = paragraph_text(segment.get("summary"))
        content = paragraph_text(segment.get("content"))
        lines.extend([
            f"## {number}. {segment_title}", "",
            f"> 时间：{start_time}–{end_time}  ",
            f"> 句子：{start_id}–{end_id}",
        ])
        if keyword_text:
            lines.append(f"> 关键词：{keyword_text}")
        lines.extend(["", "### 摘要", "", summary or "（无）", "", "### 正文", ""])
        lines.extend([content or "（无）", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_segment_markdown(path: Path, document: dict[str, Any]) -> None:
    """Atomically replace the Markdown derivative for a segment document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(segment_markdown(document), encoding="utf-8")
    os.replace(temporary, path)
