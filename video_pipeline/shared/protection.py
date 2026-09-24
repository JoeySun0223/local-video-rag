"""阻止命令行重跑覆盖已存在的人工或下游成果。"""

from __future__ import annotations

from pathlib import Path

from .io import load_json


def assert_can_overwrite(
    stage: str, data_root: Path, partition: str, video_id: str,
    *, history_root: Path | None = None,
) -> None:
    """检查覆盖风险；默认的非覆盖运行不需要调用。

    上游结果已有下游文件时禁止就地覆盖，避免时间轴或审核记录失配。
    已编辑或已确认的章节也不能被模型输出直接替换。
    """
    cleaned = data_root / "cleaned_asr" / partition / f"{video_id}.json"
    segments = data_root / "semantic_segments" / partition / f"{video_id}.json"
    if stage == "asr" and cleaned.is_file():
        raise RuntimeError("已有清洗结果，禁止覆盖原始 ASR；请先另存新版本并迁移下游数据")
    if stage == "cleanup":
        history = history_root / "cleanup" / partition / f"{video_id}.json" if history_root else None
        if segments.is_file() or (history is not None and history.is_file()):
            raise RuntimeError("已有章节或清洗审核记录，禁止覆盖清洗结果；请另存新版本")
    if stage == "segments" and segments.is_file():
        document = load_json(segments)
        if (
            document.get("review_status") == "confirmed"
            or int(document.get("revision", 0)) > 0
            or any(row.get("manual_content_override") for row in document.get("segments", []))
        ):
            raise RuntimeError("章节已有人工修改或确认，禁止覆盖；请另存新版本")
