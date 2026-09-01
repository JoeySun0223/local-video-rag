from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any


SENTENCE_TERMINAL = re.compile(r"(?:[。！？!?]+|\.(?:[\"'”’）)]*)?)$")


def _candidate_id(variant: str, canonical: str) -> str:
    return hashlib.sha256(f"{variant}\x1f{canonical}".encode("utf-8")).hexdigest()[:20]


def preview_sentence_edits(sentences: list[dict[str, Any]], edits: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate edits and return an auditable diff plus optional glossary pairs."""
    current = {int(item["id"]): item for item in sentences}
    requested: dict[int, str] = {}
    for item in edits:
        sentence_id = int(item.get("sentence_id", 0))
        if sentence_id not in current:
            raise ValueError(f"句子{sentence_id}不存在于当前Build")
        if sentence_id in requested:
            raise ValueError(f"句子{sentence_id}重复提交")
        after = str(item.get("approved_text", "")).strip()
        if not after:
            raise ValueError(f"句子{sentence_id}的校订文本不能为空")
        if not SENTENCE_TERMINAL.search(after):
            raise ValueError(f"句子{sentence_id}必须保留完整句末标点")
        requested[sentence_id] = after
    if len(requested) > 500:
        raise ValueError("单次最多校订500句")

    changes: list[dict[str, Any]] = []
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for sentence_id in sorted(requested):
        row = current[sentence_id]
        before = str(row.get("approved_text") or row["raw_text"])
        after = requested[sentence_id]
        if before == after:
            continue
        parts = []
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            original, replacement = before[i1:i2], after[j1:j2]
            eligible = (
                tag == "replace" and bool(original.strip()) and bool(replacement.strip())
                and len(original) <= 80 and len(replacement) <= 80
                and any(char.isalnum() or "\u3400" <= char <= "\u9fff" for char in original + replacement)
            )
            part = {"type": tag, "original": original, "replacement": replacement, "glossary_eligible": eligible}
            parts.append(part)
            if eligible:
                key = (original, replacement)
                candidate = candidates.setdefault(key, {
                    "id": _candidate_id(original, replacement), "variant": original,
                    "canonical": replacement, "sentence_ids": [],
                })
                candidate["sentence_ids"].append(sentence_id)
        changes.append({"sentence_id": sentence_id, "before": before, "after": after, "parts": parts})
    if not changes:
        raise ValueError("没有检测到实际文字修改")
    return {"changes": changes, "glossary_candidates": list(candidates.values())}

