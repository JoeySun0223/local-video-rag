"""Load verified glossary terms for ASR cleanup prompts."""

from __future__ import annotations

from typing import Any

from ..shared.config import project_path
from ..shared.io import load_json


def configured_terms(config: dict[str, Any], video_id: str) -> list[str]:
    glossary_path = project_path(config, config["glossary"])
    if not glossary_path.is_file():
        return []
    glossary = load_json(glossary_path)
    if not isinstance(glossary, dict):
        raise ValueError(f"词表顶层必须是对象：{glossary_path}")
    global_terms = glossary.get("global", {})
    source_terms = (glossary.get("sources", {}) or {}).get(video_id, {})
    result: list[str] = []
    for group in (global_terms, source_terms):
        if not isinstance(group, dict):
            raise ValueError(f"词表条目必须是对象：{glossary_path}")
        result.extend(str(term).strip() for term in group if str(term).strip())
    return list(dict.fromkeys(result))
