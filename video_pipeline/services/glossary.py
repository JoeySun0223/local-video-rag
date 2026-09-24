"""从人工修改提取术语，并维护词表与选择性应用。"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from video_pipeline.shared.config import project_path
from video_pipeline.shared.io import load_json, write_json
from .catalog import PipelinePaths, validate_video_id
from .editing import EditConflictError, save_cleaned_sentences
from .storage import video_lock

TERM_PART = re.compile(r"[\u3400-\u9fffA-Za-z0-9·_-]+")
GLOSSARY_LOCK = threading.Lock()
CLAUSE_BOUNDARY = re.compile(r"[，,。！？!?；;：:\n\r]")


def _parts(characters: list[str]) -> list[str]:
    return [
        match.group(0)
        for match in TERM_PART.finditer("".join(characters))
        if match.group(0).strip()
    ]


def _is_han(character: str) -> bool:
    return bool(character and "\u3400" <= character <= "\u9fff")


def _is_ascii_term_character(character: str) -> bool:
    return bool(character and re.fullmatch(r"[A-Za-z0-9_-]", character))


def _ascii_term_span(
    characters: list[str], start: int, end: int
) -> tuple[int, int] | None:
    """把英文局部改动扩展到完整词项，同时不吞并相邻中文。"""
    touches_ascii = any(_is_ascii_term_character(value) for value in characters[start:end])
    touches_ascii = touches_ascii or (
        start > 0 and _is_ascii_term_character(characters[start - 1])
    ) or (end < len(characters) and _is_ascii_term_character(characters[end]))
    if not touches_ascii:
        return None
    while start > 0 and _is_ascii_term_character(characters[start - 1]):
        start -= 1
    while end < len(characters) and _is_ascii_term_character(characters[end]):
        end += 1
    return start, end


def changed_term_candidates(original: str, edited: str) -> list[dict[str, Any]]:
    """提取替换和新增的术语候选；纯删除不进入词表。"""
    before = list(original)
    after = list(edited)
    matcher = SequenceMatcher(None, before, after, autojunk=False)
    result: list[dict[str, Any]] = []
    by_term: dict[str, dict[str, Any]] = {}
    for operation, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if operation not in {"replace", "insert"}:
            continue
        old_span = _ascii_term_span(before, left_start, left_end)
        new_span = _ascii_term_span(after, right_start, right_end)
        if old_span is not None or new_span is not None:
            old_slice = before[slice(*(old_span or (left_start, left_end)))]
            new_slice = after[slice(*(new_span or (right_start, right_end)))]
            old_parts = _parts(old_slice)
            new_parts = _parts(new_slice)
        else:
            old_parts = _parts(before[left_start:left_end])
            new_parts = _parts(after[right_start:right_end])
        if not new_parts:
            continue
        old_text = "".join(old_parts)
        new_text = "".join(new_parts)
        # Character diff avoids swallowing a following person's name. For a
        # replacement ending immediately before one shared Han character, keep
        # that single character as useful term context (兔南海 → 通达海).
        if (
            old_span is None and new_span is None
            and
            operation == "replace"
            and left_end < len(before) and right_end < len(after)
            and before[left_end] == after[right_end]
            and _is_han(after[right_end])
        ):
            old_text += before[left_end]
            new_text += after[right_end]
        if not new_text or new_text == old_text or len(new_text) > 40:
            continue
        aliases = [old_text] if old_text and old_text != new_text and len(old_text) <= 40 else []
        existing = by_term.get(new_text)
        if existing is not None:
            existing["aliases"] = list(dict.fromkeys([*existing["aliases"], *aliases]))
            continue
        candidate = {
            "term": new_text,
            "aliases": aliases,
            "original": old_text,
            "replacement": new_text,
            "change_type": operation,
        }
        by_term[new_text] = candidate
        result.append(candidate)
        if len(result) >= 20:
            break
    return result


def _glossary(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = project_path(config, config["glossary"])
    value = load_json(path) if path.is_file() else {"global": {}, "sources": {}}
    if not isinstance(value, dict):
        raise ValueError(f"词表顶层必须是对象：{path}")
    if not isinstance(value.get("global", {}), dict) or not isinstance(value.get("sources", {}), dict):
        raise ValueError(f"词表 global 和 sources 必须是对象：{path}")
    value.setdefault("global", {})
    value.setdefault("sources", {})
    return path, value


def glossary_candidates(
    config: dict[str, Any], video_id: str, original: str, edited: str
) -> list[dict[str, Any]]:
    """发现候选术语，并排除词表已经完整覆盖的内容。"""
    _, glossary = _glossary(config)
    global_terms = glossary["global"]
    source_terms = glossary["sources"].get(video_id, {})
    if not isinstance(source_terms, dict):
        raise ValueError("当前视频词表必须是对象")
    result: list[dict[str, Any]] = []
    for candidate in changed_term_candidates(original, edited):
        aliases = set(candidate["aliases"])
        covered = False
        existing_scopes: list[str] = []
        for scope, group in (("global", global_terms), ("video", source_terms)):
            if candidate["term"] not in group:
                continue
            existing_scopes.append(scope)
            stored = group[candidate["term"]]
            stored_aliases = set(stored if isinstance(stored, list) else [])
            if aliases.issubset(stored_aliases):
                covered = True
        if not covered:
            result.append({**candidate, "existing_scopes": existing_scopes})
    return result


def save_glossary_terms(
    config: dict[str, Any], video_id: str, entries: list[dict[str, Any]]
) -> int:
    """把选中术语合并到全局或当前视频词表。"""
    if not entries:
        return 0
    with GLOSSARY_LOCK:
        path, glossary = _glossary(config)
        changed = 0
        for entry in entries:
            term = str(entry.get("term", "")).strip()
            scope = str(entry.get("scope", ""))
            if not term or len(term) > 40 or not TERM_PART.fullmatch(term):
                raise ValueError(f"词表术语非法：{term!r}")
            if scope == "global":
                group = glossary["global"]
            elif scope == "video":
                group = glossary["sources"].setdefault(video_id, {})
            else:
                raise ValueError(f"未知词表范围：{scope}")
            if not isinstance(group, dict):
                raise ValueError("词表分组必须是对象")
            aliases: list[str] = []
            for value in entry.get("aliases", []):
                alias = str(value).strip()
                if not alias or alias == term:
                    continue
                if len(alias) > 40 or not TERM_PART.fullmatch(alias):
                    raise ValueError(f"错误举例非法：{alias!r}")
                aliases.append(alias)
            current = group.get(term, [])
            if not isinstance(current, list):
                current = []
            merged = list(dict.fromkeys([*current, *aliases]))
            if term not in group or merged != current:
                group[term] = merged
                changed += 1
        if changed:
            write_json(path, glossary)
        return changed


def _clause_snippet(text: str, position: int) -> str:
    """只截取命中词周围未被标点切断的分句。"""
    left = 0
    right = len(text)
    for match in CLAUSE_BOUNDARY.finditer(text):
        if match.end() <= position:
            left = match.end()
            continue
        right = match.start()
        break
    return text[left:right].strip()


def _hit_id(
    partition: str, video_id: str, sentence_id: int, alias: str, term: str, text: str
) -> str:
    payload = json.dumps(
        [partition, video_id, sentence_id, alias, term, text],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def search_glossary_occurrences(
    paths: PipelinePaths,
    source_video_id: str,
    entries: list[dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    """只读搜索清洗文本中精确匹配的术语及别名。"""
    validate_video_id(source_video_id)
    if scope not in {"database", "video"}:
        raise ValueError(f"未知搜索范围：{scope}")
    replacements: list[tuple[str, str]] = []
    for entry in entries:
        term = str(entry.get("term") or "").strip()
        if not term or not TERM_PART.fullmatch(term):
            raise ValueError(f"词表术语非法：{term!r}")
        for value in entry.get("aliases", []):
            alias = str(value).strip()
            if alias and alias != term and TERM_PART.fullmatch(alias):
                replacements.append((alias, term))
    replacements = list(dict.fromkeys(sorted(replacements, key=lambda row: -len(row[0]))))
    if not replacements:
        return []

    root = paths.data_root / "cleaned_asr"
    candidates = sorted(root.glob("*/*.json")) if scope == "database" else [
        root / paths.partition / f"{source_video_id}.json"
    ]
    hits: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file() or path.stem == "manifest":
            continue
        partition = path.parent.name
        video_id = validate_video_id(path.stem)
        if scope == "video" and video_id != source_video_id:
            continue
        document = load_json(path)
        if not isinstance(document, dict) or not isinstance(document.get("sentences"), list):
            continue
        title = str(document.get("title") or video_id)
        for sentence in document["sentences"]:
            if not isinstance(sentence, dict):
                continue
            text = str(sentence.get("text") or "")
            sentence_id = int(sentence.get("sentence_id", -1))
            occupied: list[tuple[int, int]] = []
            for alias, term in replacements:
                position = -1
                for match in re.finditer(re.escape(alias), text):
                    if any(match.start() < end and match.end() > start for start, end in occupied):
                        continue
                    position = match.start()
                    occupied.append((match.start(), match.end()))
                    break
                if position < 0:
                    continue
                hits.append({
                    "hit_id": _hit_id(partition, video_id, sentence_id, alias, term, text),
                    "partition": partition,
                    "video_id": video_id,
                    "video_title": title,
                    "sentence_id": sentence_id,
                    "start_ms": int(sentence.get("start_ms", 0)),
                    "alias": alias,
                    "term": term,
                    "snippet": _clause_snippet(text, position),
                })
    return hits


def apply_glossary_occurrences(
    paths: PipelinePaths,
    source_video_id: str,
    entries: list[dict[str, Any]],
    scope: str,
    selected_hit_ids: list[str],
) -> dict[str, Any]:
    """应用选中的逐句替换，并重建对应章节派生文件。"""
    selected = set(selected_hit_ids)
    available = {
        hit["hit_id"]: hit
        for hit in search_glossary_occurrences(paths, source_video_id, entries, scope)
    }
    unknown = selected - set(available)
    if unknown:
        raise EditConflictError("搜索结果已经变化，请重新搜索后再应用")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for hit_id in selected:
        hit = available[hit_id]
        grouped.setdefault((hit["partition"], hit["video_id"]), []).append(hit)

    changed_sentences = 0
    changed_videos = 0
    for (partition, video_id), hits in grouped.items():
        local_paths = paths if partition == paths.partition else replace(paths, partition=partition)
        with video_lock(local_paths, video_id):
            # 搜索和加锁之间可能已有人工修改；锁内重新核对选中句子的内容。
            current_ids = {
                hit["hit_id"] for hit in search_glossary_occurrences(
                    local_paths, video_id, entries, "video",
                )
            }
            if any(hit["hit_id"] not in current_ids for hit in hits):
                raise EditConflictError("搜索结果已经变化，请重新搜索后再应用")
            cleaned_path = local_paths.artifact(video_id, "cleaned")
            cleaned = load_json(cleaned_path)
            rows = cleaned["sentences"]
            by_sentence: dict[int, list[dict[str, Any]]] = {}
            for hit in hits:
                by_sentence.setdefault(int(hit["sentence_id"]), []).append(hit)
            edits: list[dict[str, Any]] = []
            local_changes = 0
            for row in rows:
                sentence_id = int(row.get("sentence_id", -1))
                text = str(row.get("text") or "")
                original = text
                for hit in sorted(
                    by_sentence.get(sentence_id, []), key=lambda value: -len(value["alias"])
                ):
                    text = text.replace(str(hit["alias"]), str(hit["term"]))
                if text != original:
                    local_changes += 1
                edits.append({"sentence_id": sentence_id, "text": text})
            if local_changes:
                save_cleaned_sentences(local_paths, video_id, edits)
                changed_sentences += local_changes
                changed_videos += 1
    return {
        "selected": len(selected),
        "changed_sentences": changed_sentences,
        "changed_videos": changed_videos,
    }
