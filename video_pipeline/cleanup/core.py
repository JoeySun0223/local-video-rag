"""Pure cleanup validation and materialization rules."""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Any


REQUIRED_RESULT_FIELDS = {"sentence_id", "text", "needs_review", "review_reason"}


def semantic_characters(text: str) -> str:
    return "".join(
        character for character in text
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def text_similarity(left: str, right: str) -> float:
    """Compare sentence content while ignoring whitespace and punctuation."""
    normalized_left = semantic_characters(left)
    normalized_right = semantic_characters(right)
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def neighboring_source_offset(
    sentences: list[dict[str, Any]],
    sentence_index: int,
    candidate: str,
    *,
    window: int = 2,
    minimum_similarity: float = 0.45,
    minimum_margin: float = 0.20,
) -> int:
    """Return a likely neighboring source offset, or zero when alignment is plausible.

    A single large rewrite is not enough to reject a result.  The caller combines
    these signals into a consecutive run before treating a batch as shifted.
    """
    if not candidate.strip():
        return 0
    own = text_similarity(candidate, str(sentences[sentence_index]["raw_text"]))
    best_offset = 0
    best_score = own
    lower = max(0, sentence_index - window)
    upper = min(len(sentences), sentence_index + window + 1)
    for neighbor_index in range(lower, upper):
        if neighbor_index == sentence_index:
            continue
        score = text_similarity(candidate, str(sentences[neighbor_index]["raw_text"]))
        if score > best_score:
            best_score = score
            best_offset = neighbor_index - sentence_index
    if best_score >= minimum_similarity and best_score >= own + minimum_margin:
        return best_offset
    return 0


def alignment_drift_ids(
    sentences: list[dict[str, Any]],
    proposals: dict[int, dict[str, Any]],
    start: int,
    end: int,
    *,
    minimum_run: int = 3,
) -> list[int]:
    """Detect consecutive model outputs mapped to the same wrong neighbor offset.

    Requiring a run avoids penalizing legitimate aggressive rewrites.  A deletion
    represented by an empty string is also ignored, so normal filler removal stays
    available to the model.
    """
    signals: list[tuple[int, int, int]] = []
    for index in range(start, end):
        sentence_id = int(sentences[index]["id"])
        proposal = proposals.get(sentence_id)
        if proposal is None:
            continue
        offset = neighboring_source_offset(sentences, index, str(proposal.get("text", "")))
        if offset:
            signals.append((index, sentence_id, offset))

    drifted: list[int] = []
    run: list[tuple[int, int, int]] = []
    for signal in signals:
        if run and (signal[0] != run[-1][0] + 1 or signal[2] != run[-1][2]):
            if len(run) >= minimum_run:
                drifted.extend(item[1] for item in run)
            run = []
        run.append(signal)
    if len(run) >= minimum_run:
        drifted.extend(item[1] for item in run)
    return drifted


def validate_response_partial(
    value: dict[str, Any], expected_ids: list[int]
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    """Return valid rows and per-sentence errors without discarding good rows.

    Extra model fields are harmless and ignored. Missing or mistyped required
    fields remain invalid and can be retried independently by the caller.
    """
    rows = value.get("results")
    if not isinstance(rows, list):
        raise ValueError("GLM 返回对象缺少 results 数组")
    result: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}
    for position, row in enumerate(rows, 1):
        fallback_id = expected_ids[position - 1] if position <= len(expected_ids) else None
        if not isinstance(row, dict):
            if fallback_id is not None:
                errors[fallback_id] = f"第 {position} 条结果不是对象"
            continue
        try:
            sentence_id = int(row.get("sentence_id"))
        except (TypeError, ValueError):
            if fallback_id is not None:
                errors[fallback_id] = f"第 {position} 条缺少有效 sentence_id"
            continue
        if sentence_id not in expected_ids:
            continue
        missing = REQUIRED_RESULT_FIELDS.difference(row)
        if missing:
            errors[sentence_id] = f"缺少字段：{', '.join(sorted(missing))}"
            continue
        if sentence_id in result:
            result.pop(sentence_id, None)
            errors[sentence_id] = f"sentence_id={sentence_id} 重复"
            continue
        if not isinstance(row["text"], str) or not isinstance(row["review_reason"], str):
            errors[sentence_id] = "text 或 review_reason 不是字符串"
            continue
        if not isinstance(row["needs_review"], bool):
            errors[sentence_id] = "needs_review 不是布尔值"
            continue
        review_reason = row["review_reason"].strip()
        if row["needs_review"] and not review_reason:
            errors[sentence_id] = "需要审核但未提供原因"
            continue
        if not row["needs_review"] and review_reason:
            errors[sentence_id] = "无需审核时 review_reason 必须为空"
            continue
        result[sentence_id] = {
            "text": row["text"],
            "needs_review": row["needs_review"],
            "review_reason": review_reason,
        }
        errors.pop(sentence_id, None)
    for sentence_id in expected_ids:
        if sentence_id not in result:
            errors.setdefault(sentence_id, "模型未返回该句，或返回结果无法匹配")
    return result, errors


def validate_response(value: dict[str, Any], expected_ids: list[int]) -> dict[int, dict[str, Any]]:
    result, errors = validate_response_partial(value, expected_ids)
    if errors:
        sentence_id = next(item for item in expected_ids if item in errors)
        raise ValueError(f"sentence_id={sentence_id}：{errors[sentence_id]}")
    return result


def evaluate(original: str, proposal: dict[str, Any]) -> dict[str, Any]:
    candidate = proposal["text"]
    if proposal["needs_review"]:
        return {
            "text": original,
            "status": "pending_review",
            "candidate": candidate,
            "evidence": proposal["review_reason"],
        }
    if candidate == original:
        return {"text": original, "status": "unchanged", "candidate": candidate}
    return {"text": candidate, "status": "auto_accepted", "candidate": candidate}


def edits(before: str, after: str) -> list[dict[str, str]]:
    return [
        {"operation": operation, "before": before[i1:i2], "after": after[j1:j2]}
        for operation, i1, i2, j1, j2 in SequenceMatcher(None, before, after).get_opcodes()
        if operation != "equal"
    ]
