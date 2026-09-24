"""视频资料目录、路径约束、上传暂存和可读记录查询。"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from video_pipeline.shared.config import project_path
from video_pipeline.shared.io import load_json, now_iso, write_json


VIDEO_ID = re.compile(r"^[0-9a-f]{64}$")
INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class PipelinePaths:
    partition: str
    data_root: Path
    history_root: Path
    work_root: Path
    business_root: Path | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any], partition: str | None = None) -> "PipelinePaths":
        return cls(
            partition=partition or str(config["partition"]),
            data_root=project_path(config, config["data_root"]),
            history_root=project_path(config, config["history_root"]),
            work_root=project_path(config, config["work_root"]),
            business_root=project_path(
                config, config.get("business_library_root", "business_knowledge_library")
            ),
        )

    @property
    def upload_root(self) -> Path:
        return self.work_root / "uploads" / self.partition

    @property
    def markdown_root(self) -> Path:
        return self.data_root / "semantic_markdown" / self.partition

    @property
    def resolved_business_root(self) -> Path:
        return self.business_root or self.data_root.parent / "business_knowledge_library"

    def artifact(self, video_id: str, kind: str) -> Path | None:
        validate_video_id(video_id)
        roots = {
            "raw": self.data_root / "asr_raw" / self.partition,
            "cleaned": self.data_root / "cleaned_asr" / self.partition,
            "segments": self.data_root / "semantic_segments" / self.partition,
            "markdown": self.markdown_root,
            "history": self.history_root / "cleanup" / self.partition,
            "business": self.resolved_business_root / self.partition / "json",
            "business_markdown": self.resolved_business_root / self.partition / "markdown",
        }
        if kind == "video":
            raw = read_optional(roots["raw"] / f"{video_id}.json")
            value = raw.get("video") if raw else None
            if value:
                path = Path(str(value))
                resolved = (self.data_root / path).resolve() if not path.is_absolute() else path.resolve()
                if not resolved.is_relative_to((self.data_root / "videos" / self.partition).resolve()):
                    raise ValueError("视频路径不在当前分区的正式视频目录")
                return resolved
            matches = sorted((self.data_root / "videos" / self.partition).glob(f"*__{video_id[:16]}.*"))
            return matches[0].resolve() if matches else None
        if kind == "upload":
            manifest = self.upload_root / video_id / "upload.json"
            value = read_optional(manifest)
            if not value or not value.get("path"):
                return None
            resolved = Path(value["path"]).resolve()
            if not resolved.is_relative_to((self.upload_root / video_id).resolve()):
                raise ValueError("上传路径不在该视频的暂存目录")
            return resolved
        root = roots.get(kind)
        if root is None:
            raise KeyError(f"未知文件类型：{kind}")
        suffix = ".md" if kind in {"markdown", "business_markdown"} else ".json"
        return (root / f"{video_id}{suffix}").resolve()


def validate_video_id(video_id: str) -> str:
    value = video_id.lower()
    if not VIDEO_ID.fullmatch(value):
        raise ValueError("video_id 必须是64位十六进制SHA-256")
    return value


def safe_filename(name: str | None) -> str:
    value = Path(name or "video.mp4").name.strip()
    value = INVALID_FILENAME.sub("_", value).rstrip(". ")
    return value or "video.mp4"


def display_upload_title(name: str | None) -> str:
    """上传尚无正式标题时，隐藏文件名里的存储哈希和扩展名。"""
    filename = Path(str(name or "")).name.strip()
    stem = Path(filename).stem
    return re.sub(r"__[0-9a-f]{16}$", "", stem, flags=re.IGNORECASE) or stem


def read_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = load_json(path)
    return value if isinstance(value, dict) else None


async def ingest_upload(
    chunks: AsyncIterator[bytes], original_filename: str | None, paths: PipelinePaths
) -> dict[str, Any]:
    """接收异步视频字节流，按内容哈希存放；不依赖 HTTP 对象。"""
    filename = safe_filename(original_filename)
    incoming = paths.upload_root / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    temporary = incoming / f"{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as handle:
            async for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        video_id = digest.hexdigest()
        destination_root = paths.upload_root / video_id
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / filename
        if destination.exists():
            temporary.unlink()
        else:
            temporary.replace(destination)
        manifest = {
            "id": video_id,
            "filename": filename,
            "path": str(destination.resolve()),
            "size": size,
            "uploaded_at": now_iso(),
        }
        write_json(destination_root / "upload.json", manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def _ids_in(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {path.stem for path in root.glob("*.json") if VIDEO_ID.fullmatch(path.stem)}


def list_records(paths: PipelinePaths) -> list[dict[str, Any]]:
    raw_root = paths.data_root / "asr_raw" / paths.partition
    clean_root = paths.data_root / "cleaned_asr" / paths.partition
    segment_root = paths.data_root / "semantic_segments" / paths.partition
    business_root = paths.resolved_business_root / paths.partition / "json"
    history_root = paths.history_root / "cleanup" / paths.partition
    upload_ids = {
        item.parent.name for item in paths.upload_root.glob("*/upload.json")
        if VIDEO_ID.fullmatch(item.parent.name)
    } if paths.upload_root.is_dir() else set()
    all_ids = set().union(
        upload_ids, _ids_in(raw_root), _ids_in(clean_root), _ids_in(segment_root),
        _ids_in(history_root), _ids_in(business_root),
    )
    records: list[dict[str, Any]] = []
    for video_id in all_ids:
        raw = read_optional(raw_root / f"{video_id}.json")
        cleaned = read_optional(clean_root / f"{video_id}.json")
        segments = read_optional(segment_root / f"{video_id}.json")
        business = read_optional(business_root / f"{video_id}.json")
        upload = read_optional(paths.upload_root / video_id / "upload.json")
        upload_title = display_upload_title((upload or {}).get("filename"))
        title = str(
            (segments or {}).get("title") or (cleaned or {}).get("title")
            or (business or {}).get("title")
            or (raw or {}).get("title") or upload_title or video_id[:12]
        )
        review_status = (
            str((segments or {}).get("review_status") or "pending_review")
            if segments is not None else "not_ready"
        )
        records.append({
            "id": video_id,
            "title": title,
            "uploaded": bool(upload),
            "has_video": bool(paths.artifact(video_id, "video") or paths.artifact(video_id, "upload")),
            "has_raw": raw is not None,
            "has_cleaned": cleaned is not None,
            "has_segments": segments is not None,
            "has_business": business is not None,
            "has_history": (history_root / f"{video_id}.json").is_file(),
            "review_status": review_status,
            "duration_ms": int((raw or {}).get("duration_ms", 0)),
            "sentence_count": len((raw or {}).get("sentences", [])),
            "segment_count": len((segments or {}).get("segments", [])),
            "business_annotation_count": len((business or {}).get("annotations", [])),
            "processed_at": str((raw or {}).get("processed_at", "")),
            "uploaded_at": str((upload or {}).get("uploaded_at", "")),
        })

    def workflow_priority(row: dict[str, Any]) -> int:
        # Put actionable rows first instead of burying new uploads that do not yet
        # have a processed_at timestamp.
        if row["uploaded"] and not row["has_raw"]:
            return 0
        if not row["has_raw"] or not row["has_cleaned"] or not row["has_segments"]:
            return 1
        if row["review_status"] != "confirmed":
            return 2
        return 3

    records.sort(key=lambda row: row["title"])
    records.sort(
        key=lambda row: max(row["uploaded_at"], row["processed_at"]),
        reverse=True,
    )
    records.sort(key=workflow_priority)
    return records


def dashboard(paths: PipelinePaths) -> dict[str, Any]:
    rows = list_records(paths)
    return {
        "partition": paths.partition,
        "counts": {
            "videos": len(rows),
            "uploaded": sum(row["uploaded"] and not row["has_raw"] for row in rows),
            "raw": sum(row["has_raw"] for row in rows),
            "cleaned": sum(row["has_cleaned"] for row in rows),
            "segments": sum(row["has_segments"] for row in rows),
            "business": sum(row["has_business"] for row in rows),
            "pending_review": sum(
                row["has_segments"] and row["review_status"] != "confirmed" for row in rows
            ),
            "completed": sum(row["review_status"] == "confirmed" for row in rows),
        },
        "records": rows,
    }


def review_suggestions(paths: PipelinePaths, video_id: str) -> list[dict[str, Any]]:
    """从当前视频审核档案取得仍可复核的清洗建议。"""
    suggestions: list[dict[str, Any]] = []
    seen: set[int] = set()
    current = read_optional(paths.artifact(video_id, "history") or Path())
    for item in (current or {}).get("changes", []):
        if not isinstance(item, dict) or item.get("status") != "pending_review":
            continue
        sentence_id = int(item.get("sentence_id", -1))
        if sentence_id < 1:
            continue
        suggestions.append({**item, "source": "current"})
        seen.add(sentence_id)

    return sorted(suggestions, key=lambda item: int(item.get("sentence_id", 0)))


def video_detail(paths: PipelinePaths, video_id: str) -> dict[str, Any]:
    validate_video_id(video_id)
    artifacts: dict[str, Any] = {}
    artifact_paths: dict[str, str] = {}
    for kind in ("raw", "cleaned", "segments", "history", "business"):
        path = paths.artifact(video_id, kind)
        if path and path.is_file():
            artifacts[kind] = load_json(path)
            artifact_paths[kind] = str(path)
    markdown_path = paths.artifact(video_id, "markdown")
    if markdown_path and markdown_path.is_file():
        artifacts["markdown"] = markdown_path.read_text(encoding="utf-8")
        artifact_paths["markdown"] = str(markdown_path)
    business_markdown_path = paths.artifact(video_id, "business_markdown")
    if business_markdown_path and business_markdown_path.is_file():
        artifacts["business_markdown"] = business_markdown_path.read_text(encoding="utf-8")
        artifact_paths["business_markdown"] = str(business_markdown_path)
    for kind in ("video", "upload"):
        path = paths.artifact(video_id, kind)
        if path and path.is_file():
            artifact_paths[kind] = str(path)
    artifacts["suggestions"] = review_suggestions(paths, video_id)
    return {"id": video_id, "artifacts": artifacts, "paths": artifact_paths}


def delete_video(paths: PipelinePaths, video_id: str) -> int:
    # 局部导入避免路径模块与事务模块互相导入；删除也必须先恢复残留事务。
    from .storage import video_lock

    with video_lock(paths, video_id):
        return _delete_video(paths, video_id)


def _delete_video(paths: PipelinePaths, video_id: str) -> int:
    """只删除已校验视频在项目目录内的文件和上传目录。"""
    validate_video_id(video_id)
    allowed_roots = [
        paths.data_root.resolve(), paths.history_root.resolve(),
        paths.work_root.resolve(), paths.resolved_business_root.resolve(),
    ]
    candidates: set[Path] = set()
    for kind in (
        "raw", "cleaned", "segments", "markdown", "history",
        "business", "business_markdown", "video", "upload",
    ):
        candidate = paths.artifact(video_id, kind)
        if candidate:
            candidates.add(candidate.resolve())

    for candidate in candidates:
        if not any(candidate.is_relative_to(root) for root in allowed_roots):
            raise ValueError(f"拒绝删除项目目录之外的文件：{candidate}")

    owned_directories = [
        (paths.upload_root / video_id).resolve(),
        (paths.work_root / "cache" / "thumbnails" / paths.partition / video_id).resolve(),
    ]
    allowed_directories = [paths.work_root.resolve()]
    if any(not any(directory.is_relative_to(root) for root in allowed_directories) for directory in owned_directories):
        raise ValueError("待删除目录越界")

    removed = 0
    for candidate in candidates:
        if candidate.is_file():
            candidate.unlink()
            removed += 1

    for directory in owned_directories:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if candidate.is_file():
                candidate.unlink()
                removed += 1
            elif candidate.is_dir():
                candidate.rmdir()
        directory.rmdir()
    return removed
