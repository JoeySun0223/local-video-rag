"""Transcribe videos with a local ASR model and write immutable asr_raw JSON."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .engine import ASRRequest, get_asr_provider
from .media import duration_ms, extract_audio, title_from_filename, video_files
from ..shared.config import load_config, project_path
from ..shared.io import load_json, now_iso, sha256_file, write_json
from ..shared.progress import emit_progress
from ..shared.protection import assert_can_overwrite
from ..services.catalog import PipelinePaths
from ..services.storage import video_lock


def _configured_hotwords(config: dict, video_id: str, explicit: list[str]) -> list[str]:
    glossary_path = project_path(config, config["glossary"])
    configured: list[str] = []
    if glossary_path.is_file():
        glossary = load_json(glossary_path)
        if not isinstance(glossary, dict):
            raise ValueError(f"词表顶层必须是对象：{glossary_path}")
        global_terms = glossary.get("global", {})
        source_terms = (glossary.get("sources", {}) or {}).get(video_id, {})
        for group in (global_terms, source_terms):
            if not isinstance(group, dict):
                raise ValueError(f"词表条目必须是对象：{glossary_path}")
            configured.extend(str(term).strip() for term in group if str(term).strip())
    return list(dict.fromkeys([*explicit, *configured]))


def _stored_video(source: Path, data_root: Path, partition: str, video_id: str, title: str) -> Path:
    target_root = data_root / "videos" / partition
    target_root.mkdir(parents=True, exist_ok=True)
    try:
        source.resolve().relative_to(target_root.resolve())
        return source.resolve()
    except ValueError:
        target = target_root / f"{title}__{video_id[:16]}{source.suffix.lower()}"
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(target)
        elif sha256_file(target) != video_id:
            raise RuntimeError(f"目标视频同名但内容不同：{target}")
        return target


def transcribe_one(
    source: Path,
    *,
    config: dict,
    data_root: Path,
    work_root: Path,
    partition: str,
    provider_name: str | None,
    overwrite: bool,
    title: str | None = None,
    category: str = "",
    hotwords: list[str] | None = None,
) -> tuple[Path, bool]:
    source = source.resolve()
    video_id = sha256_file(source)
    hotwords = _configured_hotwords(config, video_id, hotwords or [])
    output_path = data_root / "asr_raw" / partition / f"{video_id}.json"
    if output_path.is_file() and not overwrite:
        return output_path, False
    if output_path.is_file() and overwrite:
        assert_can_overwrite("asr", data_root, partition, video_id)

    source_title = (title or title_from_filename(source)).strip() or source.stem
    stored = _stored_video(source, data_root, partition, video_id, source_title)
    ffmpeg = project_path(config, config["ffmpeg"])
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"找不到 FFmpeg：{ffmpeg}")

    provider = get_asr_provider(config, provider_name)
    emit_progress("asr", 2, 100, "正在检查本地语音模型")
    report = provider.doctor()
    if not report.get("available"):
        raise RuntimeError(f"本地 ASR 不可用：{report}")

    source_work_root = work_root / "asr" / partition / video_id
    source_work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="transcribe-", dir=source_work_root) as temporary_dir:
        temporary = Path(temporary_dir)
        audio = temporary / "audio.wav"
        emit_progress("asr", 4, 100, "正在从视频提取音频")
        extract_audio(ffmpeg, stored, audio)
        emit_progress("asr", 8, 100, "音频提取完成，正在切分")
        result = provider.transcribe(ASRRequest(
            audio=audio,
            title=source_title,
            category=category,
            hotwords=hotwords,
            work_dir=temporary,
            ffmpeg=ffmpeg,
            checkpoint_path=source_work_root / "checkpoint.json",
        ))

    try:
        relative_video = stored.relative_to(data_root).as_posix()
    except ValueError:
        relative_video = str(stored)
    languages = Counter(
        str(item.get("language", "")).strip()
        for item in result.raw.get("segments", [])
        if str(item.get("language", "")).strip()
    )
    measured_duration = duration_ms(ffmpeg, stored)
    if not measured_duration and result.sentences:
        measured_duration = int(result.sentences[-1]["end"])
    payload = {
        "id": video_id,
        "title": source_title,
        "video": relative_video,
        "duration_ms": measured_duration,
        "processed_at": now_iso(),
        "asr": {
            **result.provenance(),
            "language": languages.most_common(1)[0][0] if languages else "Chinese",
        },
        "sentences": [
            {
                "id": int(item["id"]),
                "start_ms": int(item["start"]),
                "end_ms": int(item["end"]),
                "raw_text": str(item["text"]),
            }
            for item in result.sentences
        ],
    }
    service_paths = PipelinePaths(
        partition, data_root,
        project_path(config, config["history_root"]), work_root,
    )
    with video_lock(service_paths, video_id):
        if output_path.is_file():
            if not overwrite:
                return output_path, False
            assert_can_overwrite("asr", data_root, partition, video_id)
        write_json(output_path, payload)
    emit_progress("asr", 100, 100, f"原始 ASR 已保存，共 {len(result.sentences)} 句")
    return output_path, True


def run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    partition = args.partition or str(config["partition"])
    data_root = project_path(config, args.data_root or config["data_root"])
    work_root = project_path(config, args.work_root or config["work_root"])
    paths = video_files(Path(args.input).resolve())
    if args.limit is not None:
        paths = paths[: max(0, args.limit)]
    if args.title and len(paths) != 1:
        raise ValueError("--title 只能用于单个视频")
    print(f"videos={len(paths)} provider={config['asr']['primary']} data={data_root}", flush=True)
    written = skipped = 0
    for number, path in enumerate(paths, 1):
        print(f"[{number}/{len(paths)}] {path.name}", flush=True)
        output, changed = transcribe_one(
            path,
            config=config,
            data_root=data_root,
            work_root=work_root,
            partition=partition,
            provider_name=None,
            overwrite=args.overwrite,
            title=args.title,
            category=args.category,
            hotwords=args.hotword,
        )
        written += int(changed)
        skipped += int(not changed)
        print(f"  {'wrote' if changed else 'skip'} {output}", flush=True)
    print(f"done written={written} skipped={skipped}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("input", help="单个视频或视频目录")
    result.add_argument("--config")
    result.add_argument("--data-root")
    result.add_argument("--work-root")
    result.add_argument("--partition")
    result.add_argument("--title")
    result.add_argument("--category", default="")
    result.add_argument("--hotword", action="append", default=[])
    result.add_argument("--limit", type=int)
    result.add_argument("--overwrite", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
