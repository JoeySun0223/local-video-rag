"""Small shared helpers for the file-based ASR pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="microseconds")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sentence_batches(
    sentences: list[dict[str, Any]],
    max_sentences: int,
    max_chars: int,
    text_key: str = "raw_text",
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start = 0
    while start < len(sentences):
        end = start
        characters = 0
        while end < len(sentences) and end - start < max_sentences:
            size = len(str(sentences[end].get(text_key, "")))
            if end > start and characters + size > max_chars:
                break
            characters += size
            end += 1
        result.append((start, max(start + 1, end)))
        start = max(start + 1, end)
    return result
