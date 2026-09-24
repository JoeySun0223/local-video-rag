"""Machine-readable progress events shared by pipeline subprocesses."""

from __future__ import annotations

import json
from typing import Any


PROGRESS_PREFIX = "PIPELINE_PROGRESS "


def emit_progress(stage: str, completed: float, total: float, detail: str) -> None:
    """Write one flush-safe event without coupling workers to the Web UI."""
    safe_total = max(float(total), 1.0)
    payload = {
        "stage": str(stage),
        "completed": max(0.0, min(float(completed), safe_total)),
        "total": safe_total,
        "detail": " ".join(str(detail).split())[:160],
    }
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=True), flush=True)


def parse_progress_event(line: str) -> dict[str, Any] | None:
    """Parse only explicitly tagged events; ordinary logs remain untouched."""
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        value = json.loads(line[len(PROGRESS_PREFIX):])
        stage = str(value["stage"])
        completed = float(value["completed"])
        total = float(value["total"])
        if total <= 0:
            return None
        return {
            "stage": stage,
            "fraction": max(0.0, min(completed / total, 1.0)),
            "detail": str(value.get("detail") or "").strip(),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
