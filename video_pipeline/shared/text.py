"""跨处理阶段统一章节正文的展示格式。"""

from __future__ import annotations

from typing import Any


def paragraph_text(value: Any) -> str:
    """把逐句换行文本合成自然段，必要时保留英文词间空格。"""
    parts = [
        " ".join(line.split())
        for line in str(value or "").replace("\r", "").split("\n")
        if line.strip()
    ]
    result = ""
    for part in parts:
        needs_space = bool(
            result
            and result[-1] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;:!?)"
            and part[0] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789("
        )
        result += (" " if needs_space else "") + part
    return result
