"""Per-video cloud cleanup workflow."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import httpx

from ..shared.glm import request_json
from ..shared.io import now_iso, sentence_batches
from ..shared.progress import emit_progress
from .core import (
    alignment_drift_ids,
    edits,
    evaluate,
    neighboring_source_offset,
    validate_response_partial,
)
from .prompts import SYSTEM_PROMPT


def _prompt(
    title: str,
    sentences: list[dict[str, Any]],
    start: int,
    end: int,
    confirmed_terms: list[str] | None = None,
) -> str:
    payload = {
        "video_title": title,
        "confirmed_business_terms": confirmed_terms or [],
        "context_before": [item["raw_text"] for item in sentences[max(0, start - 2):start]],
        "target_sentences": [
            {"sentence_id": int(item["id"]), "text": str(item["raw_text"])}
            for item in sentences[start:end]
        ],
        "context_after": [item["raw_text"] for item in sentences[end:end + 2]],
    }
    return json.dumps(payload, ensure_ascii=False)


def clean_document(
    client: httpx.Client,
    raw: dict[str, Any],
    *,
    api_url: str,
    api_key: str,
    model: str,
    thinking: bool,
    attempts: int,
    max_tokens: int,
    batch_max_sentences: int,
    batch_max_chars: int,
    confirmed_terms: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Counter[str]]:
    sentences = raw.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError("asr_raw 缺少 sentences 数组")
    proposals: dict[int, dict[str, Any]] = {}
    usage: Counter[str] = Counter()
    warnings: list[dict[str, Any]] = []
    sentence_indexes = {
        int(sentence["id"]): index for index, sentence in enumerate(sentences)
    }

    def short_error(value: Any) -> str:
        message = " ".join(str(value).split())
        return message[:240] or "未知错误"

    def preserve_original(sentence_index: int, reason: str) -> dict[str, Any]:
        return {
            "text": str(sentences[sentence_index]["raw_text"]),
            "needs_review": True,
            "review_reason": (
                "自动清洗未能可靠完成，系统已保留原始 ASR，请人工复核。"
                f"原因：{short_error(reason)}"
            ),
        }

    def request_span(
        start: int,
        end: int,
        request_attempts: int,
        progress_label: str,
    ) -> tuple[
        dict[str, Any] | None,
        dict[int, dict[str, Any]],
        dict[int, str],
        str | None,
    ]:
        expected = [int(item["id"]) for item in sentences[start:end]]
        base_prompt = _prompt(
            str(raw.get("title", "")), sentences, start, end, confirmed_terms,
        )
        last_error: str | None = None
        last_value: dict[str, Any] | None = None
        for request_attempt in range(1, max(1, request_attempts) + 1):
            correction = (
                f"\n\n上一次字段校验失败：{last_error}。请重新返回完整结果。"
                if last_error else ""
            )
            try:
                value, batch_usage = request_json(
                    client,
                    api_url=api_url,
                    api_key=api_key,
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=base_prompt + correction,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    attempts=1,
                    progress_label=(
                        progress_label
                        if request_attempts == 1
                        else f"{progress_label} 尝试 {request_attempt}/{request_attempts}"
                    ),
                )
                usage.update(batch_usage)
                last_value = value
                valid, row_errors = validate_response_partial(value, expected)
                return value, valid, row_errors, None
            except (RuntimeError, ValueError) as error:
                last_error = short_error(error)
        return last_value, {}, {}, last_error or "请求失败"

    def repair_sentence(
        sentence_id: int,
        reason: str,
        progress_label: str,
        *,
        verify_alignment: bool,
    ) -> None:
        sentence_index = sentence_indexes[sentence_id]
        retry_error = reason
        for retry_number in range(1, attempts + 1):
            retry_prompt = _prompt(
                str(raw.get("title", "")), sentences,
                sentence_index, sentence_index + 1, confirmed_terms,
            )
            retry_prompt += (
                f"\n\n上一轮 sentence_id={sentence_id} 结果不可用：{retry_error}。"
                "本次只处理这一句；相邻上下文只用于理解，不得把相邻句内容写入本句。"
            )
            try:
                retry_value, retry_usage = request_json(
                    client,
                    api_url=api_url,
                    api_key=api_key,
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=retry_prompt,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    attempts=1,
                    progress_label=(
                        f"{progress_label} 单句 {sentence_id} "
                        f"重试 {retry_number}/{attempts}"
                    ),
                )
                usage.update(retry_usage)
                retry_valid, retry_errors = validate_response_partial(
                    retry_value, [sentence_id]
                )
                proposal = retry_valid.get(sentence_id)
                if proposal is None:
                    retry_error = retry_errors.get(sentence_id, "返回结果仍不合法")
                    continue
                if verify_alignment:
                    offset = neighboring_source_offset(
                        sentences, sentence_index, str(proposal.get("text", ""))
                    )
                    if offset:
                        retry_error = f"内容仍疑似来自相邻第 {sentence_id + offset} 句"
                        continue
                proposals[sentence_id] = proposal
                return
            except (RuntimeError, ValueError) as error:
                retry_error = short_error(error)

        proposals[sentence_id] = preserve_original(sentence_index, retry_error)
        warnings.append({
            "kind": "sentence_fallback",
            "sentence_ids": [sentence_id],
            "reason": short_error(retry_error),
        })

    def accept_rows(
        start: int,
        end: int,
        valid: dict[int, dict[str, Any]],
        row_errors: dict[int, str],
        progress_label: str,
    ) -> None:
        proposals.update(valid)
        for sentence in sentences[start:end]:
            sentence_id = int(sentence["id"])
            if sentence_id not in row_errors:
                continue
            repair_sentence(
                sentence_id,
                row_errors[sentence_id],
                progress_label,
                verify_alignment=False,
            )

    spans = sentence_batches(sentences, batch_max_sentences, batch_max_chars)
    emit_progress("cleanup", 0, max(len(sentences), 1), f"准备清洗 {len(sentences)} 句，共 {len(spans)} 批")
    for batch_number, (start, end) in enumerate(spans, 1):
        expected = [int(item["id"]) for item in sentences[start:end]]
        label = f"清洗 {batch_number}/{len(spans)}"
        emit_progress(
            "cleanup", start, max(len(sentences), 1),
            f"正在等待模型返回第 {batch_number}/{len(spans)} 批（第 {start + 1}-{end} 句）",
        )
        _value, valid, row_errors, batch_error = request_span(
            start, end, attempts, label,
        )
        if batch_error is None:
            accept_rows(start, end, valid, row_errors, label)
        else:
            warnings.append({
                "kind": "batch_recovery",
                "sentence_ids": expected,
                "reason": short_error(batch_error),
            })
            print(
                f"[{label}] 整批重试后仍失败，改用每 5 句的小范围恢复：{batch_error}",
                flush=True,
            )
            consecutive_failures = 0
            recovery_start = start
            while recovery_start < end:
                recovery_end = min(end, recovery_start + 5)
                recovery_ids = [
                    int(item["id"]) for item in sentences[recovery_start:recovery_end]
                ]
                _recovery_value, recovery_valid, recovery_errors, recovery_error = request_span(
                    recovery_start,
                    recovery_end,
                    1,
                    f"{label} 小范围恢复 {recovery_ids[0]}-{recovery_ids[-1]}",
                )
                if recovery_error is None:
                    consecutive_failures = 0
                    accept_rows(
                        recovery_start, recovery_end,
                        recovery_valid, recovery_errors, label,
                    )
                else:
                    consecutive_failures += 1
                    for sentence_id in recovery_ids:
                        proposals[sentence_id] = preserve_original(
                            sentence_indexes[sentence_id], recovery_error,
                        )
                    warnings.append({
                        "kind": "range_fallback",
                        "sentence_ids": recovery_ids,
                        "reason": short_error(recovery_error),
                    })
                    if consecutive_failures >= 2:
                        remaining = [
                            int(item["id"])
                            for item in sentences[recovery_end:end]
                        ]
                        for sentence_id in remaining:
                            proposals[sentence_id] = preserve_original(
                                sentence_indexes[sentence_id],
                                "云端接口连续不可用，已停止本批次的额外请求",
                            )
                        if remaining:
                            warnings.append({
                                "kind": "circuit_breaker_fallback",
                                "sentence_ids": remaining,
                                "reason": "云端接口连续不可用，已停止本批次的额外请求",
                            })
                        break
                recovery_start = recovery_end

        for sentence_id in expected:
            if sentence_id not in proposals:
                proposals[sentence_id] = preserve_original(
                    sentence_indexes[sentence_id], "模型未返回可用结果",
                )
                warnings.append({
                    "kind": "sentence_fallback",
                    "sentence_ids": [sentence_id],
                    "reason": "模型未返回可用结果",
                })

        drifted = alignment_drift_ids(sentences, proposals, start, end)
        if drifted:
            warnings.append({
                "kind": "alignment_drift",
                "sentence_ids": drifted,
                "reason": "检测到连续结果更接近相邻原句，已逐句重新清洗",
            })
            print(
                f"[{label}] 检测到连续句子错位：{drifted[0]}-{drifted[-1]}，逐句恢复",
                flush=True,
            )
            for sentence_id in drifted:
                repair_sentence(
                    sentence_id,
                    "检测到内容疑似来自相邻句",
                    label,
                    verify_alignment=True,
                )
        emit_progress(
            "cleanup", end, max(len(sentences), 1),
            f"已完成第 {batch_number}/{len(spans)} 批（{end}/{len(sentences)} 句）",
        )

    cleaned_sentences: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for sentence in sentences:
        sentence_id = int(sentence["id"])
        original = str(sentence["raw_text"])
        decision = evaluate(original, proposals[sentence_id])
        counts[decision["status"]] += 1
        cleaned_sentences.append({
            "sentence_id": sentence_id,
            "start_ms": int(sentence["start_ms"]),
            "end_ms": int(sentence["end_ms"]),
            "text": str(decision["text"]),
        })
        if decision["candidate"] != original or decision["status"] == "pending_review":
            audit.append({
                "sentence_id": sentence_id,
                "raw_text": original,
                "proposed_text": decision["candidate"],
                "effective_text": decision["text"],
                "status": decision["status"],
                "review_reason": decision.get("evidence", ""),
                "edits": edits(original, decision["candidate"]),
            })
    source_id = str(raw["id"])
    cleaned = {
        "source_id": source_id,
        "title": str(raw.get("title", "")),
        "cleanup": model,
        "sentences": cleaned_sentences,
        "full_text": "\n".join(item["text"] for item in cleaned_sentences if item["text"]),
    }
    report = {
        "source_id": source_id,
        "title": str(raw.get("title", "")),
        "model": model,
        "counts": dict(counts),
        "changes": audit,
    }
    if warnings:
        report["warnings"] = warnings
    emit_progress("cleanup", 1, 1, f"文本清洗完成，共 {len(sentences)} 句")
    return cleaned, report, usage


def completed_audit(cleaned: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """生成清洗审核档案；持久化由统一事务层负责。"""
    decisions = Counter(
        str(item.get("decision", ""))
        for item in report.get("changes", [])
        if item.get("decision") in {"approved", "rejected"}
    )
    return {
        **report,
        "status": "completed",
        "completed_at": now_iso(),
        "decisions": dict(decisions),
        "final": cleaned,
    }
