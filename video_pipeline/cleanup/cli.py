"""连续清洗原始 ASR，并归档供 WebUI 后续复核的 AI 建议。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import httpx

from .service import completed_audit, clean_document
from .terms import configured_terms
from ..shared.cloud import cloud_settings
from ..shared.config import load_config, project_path
from ..shared.io import load_json, now_iso, write_json
from ..shared.protection import assert_can_overwrite
from ..validation.rules import validate_cleaned, validate_raw
from ..services.catalog import PipelinePaths
from ..services.storage import commit_artifacts, video_lock


def _auto_publish(
    raw: dict, cleaned: dict, report: dict, output: Path, archive: Path,
    *, paths: PipelinePaths, overwrite: bool = False,
) -> None:
    # WebUI 不在清洗过程中暂停；不确定建议暂用，保留待复核记录。
    changes = {
        int(item["sentence_id"]): item
        for item in report.get("changes", [])
        if item.get("status") == "pending_review"
    }
    for sentence in cleaned.get("sentences", []):
        change = changes.get(int(sentence["sentence_id"]))
        if not change:
            continue
        sentence["text"] = str(change.get("proposed_text") or change.get("raw_text") or "")
        change["decision"] = "pending"
        change["effective_text"] = sentence["text"]
    cleaned["full_text"] = "\n".join(
        str(item["text"]) for item in cleaned.get("sentences", []) if item.get("text")
    )
    # 暂用建议只为后续章节生成提供文本，不等于用户同意。
    report.setdefault("counts", {})["pending_review"] = len(changes)
    errors = validate_cleaned(raw, cleaned)
    if errors:
        raise ValueError("清洗结果校验失败：" + "；".join(errors))
    audit = completed_audit(cleaned, report)
    video_id = str(raw["id"])
    with video_lock(paths, video_id):
        raw_path = paths.artifact(video_id, "raw")
        if raw_path is None or load_json(raw_path) != raw:
            raise RuntimeError("原始 ASR 在清洗期间发生变化，拒绝发布旧结果")
        if output.is_file():
            if not overwrite:
                raise RuntimeError("清洗结果在运行期间已生成，拒绝覆盖")
            assert_can_overwrite("cleanup", paths.data_root, paths.partition, video_id,
                                 history_root=paths.history_root)
        if archive.is_file():
            raise RuntimeError("已有清洗审核档案，拒绝覆盖")
        commit_artifacts(paths, video_id,
                         json_files={output: cleaned, archive: audit})


def run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    cloud = config["cloud"]
    partition = args.partition or str(config["partition"])
    data_root = project_path(config, args.data_root or config["data_root"])
    history_base = project_path(config, args.history_root or config["history_root"])
    history_root = history_base / "cleanup" / partition
    service_paths = PipelinePaths(
        partition, data_root, history_base,
        project_path(config, config["work_root"]),
    )
    failure_root = (
        project_path(config, args.history_root or config["history_root"])
        / "cleanup_failures" / partition
    )
    raw_root = data_root / "asr_raw" / partition
    cleaned_root = data_root / "cleaned_asr" / partition
    paths = sorted(raw_root.glob("*.json"))
    if args.video_id:
        paths = [path for path in paths if path.stem == args.video_id]
        if not paths:
            raise FileNotFoundError(f"找不到 asr_raw：{args.video_id}")
    settings = cloud_settings(config, model=args.model, api_key_env=args.api_key_env)
    api_key = str(settings["api_key"])
    if not api_key:
        raise RuntimeError("API 尚未连接，请在 WebUI 右上角点击“API连接”")
    model = str(settings["model"])
    totals: Counter[str] = Counter()
    with httpx.Client(timeout=httpx.Timeout(args.timeout or float(cloud["timeout_seconds"]), connect=20.0)) as client:
        for number, path in enumerate(paths, 1):
            destination = cleaned_root / path.name
            if destination.is_file() and not args.overwrite:
                print(f"[{number}/{len(paths)}] skip existing {path.name}", flush=True)
                continue
            if (history_root / path.name).is_file():
                raise RuntimeError("已有清洗审核档案，拒绝重新清洗；请使用独立版本")
            if destination.is_file() and args.overwrite:
                assert_can_overwrite(
                    "cleanup", data_root, partition, path.stem,
                    history_root=project_path(config, args.history_root or config["history_root"]),
                )
            raw = load_json(path)
            raw_errors = validate_raw(raw, path.stem)
            if raw_errors:
                raise ValueError("原始 ASR 校验失败：" + "；".join(raw_errors))
            failure_path = failure_root / path.name
            try:
                cleaned, report, usage = clean_document(
                    client, raw, api_url=str(settings["api_url"]), api_key=api_key, model=model,
                    # 清洗属于受约束的抽取和改写；高推理模式可能先耗尽输出预算，
                    # 导致尚未生成完整 JSON 就停止。
                    thinking=False, attempts=args.attempts,
                    max_tokens=args.max_tokens, batch_max_sentences=args.batch_max_sentences,
                    batch_max_chars=args.batch_max_chars,
                    confirmed_terms=configured_terms(config, str(raw["id"])),
                )
            except Exception as error:
                write_json(failure_path, {
                    "source_id": str(raw["id"]),
                    "title": str(raw.get("title", "")),
                    "model": model,
                    "failed_at": now_iso(),
                    "error": str(error),
                    "details": {},
                })
                raise
            _auto_publish(
                raw, cleaned, report, destination, history_root / path.name,
                paths=service_paths, overwrite=args.overwrite,
            )
            failure_path.unlink(missing_ok=True)
            totals.update(videos=1, **usage)
            print(f"[{number}/{len(paths)}] wrote {destination}", flush=True)
    print(dict(totals))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config")
    result.add_argument("--data-root")
    result.add_argument("--history-root")
    result.add_argument("--partition")
    result.add_argument("--video-id")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--model")
    result.add_argument("--api-key-env")
    result.add_argument("--attempts", type=int, default=3)
    result.add_argument("--max-tokens", type=int, default=4096)
    result.add_argument("--batch-max-sentences", type=int, default=40)
    result.add_argument("--batch-max-chars", type=int, default=3800)
    result.add_argument("--timeout", type=float)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
