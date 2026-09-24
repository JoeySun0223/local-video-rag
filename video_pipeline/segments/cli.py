"""Generate playable semantic segments from cleaned ASR with GLM-5.3."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import httpx

from .core import (
    build_document,
    call_glm_once,
    find_video_path,
    generate_segments,
    meaningful_sentences,
    segment_prompt_for,
    select_sources,
)
from .markdown import segment_markdown
from ..shared.cloud import cloud_settings
from ..shared.config import load_config, project_path
from ..shared.io import load_json, now_iso, write_json
from ..shared.progress import emit_progress
from ..shared.protection import assert_can_overwrite
from ..validation.rules import validate_segments
from ..services.catalog import PipelinePaths
from ..services.storage import commit_artifacts, video_lock


def run(args: argparse.Namespace) -> int:
    if args.attempts < 1:
        raise ValueError("--attempts 必须至少为1")
    config = load_config(Path(args.config) if args.config else None)
    cloud = config["cloud"]
    partition = args.partition or str(config["partition"])
    data_root = project_path(config, args.data_root or config["data_root"])
    input_root = data_root / "cleaned_asr" / partition
    result_root = data_root / "semantic_segments" / partition
    markdown_root = data_root / "semantic_markdown" / partition
    if not input_root.is_dir():
        raise FileNotFoundError(f"cleaned ASR 目录不存在：{input_root}")
    paths = select_sources(input_root, args.video_id, args.limit)
    if not paths:
        print(f"没有找到 cleaned ASR JSON：{input_root}")
        return 0
    settings = cloud_settings(config, model=args.model, api_key_env=args.api_key_env)
    api_key = str(settings["api_key"])
    if not api_key:
        raise RuntimeError("API 尚未连接，请在 WebUI 右上角点击“API连接”")
    model = str(settings["model"])
    timeout = args.timeout or float(cloud["timeout_seconds"])
    video_root = data_root / "videos"
    work_root = project_path(config, config["work_root"])
    history_root = project_path(config, config["history_root"])
    service_paths = PipelinePaths(partition, data_root, history_root, work_root)
    totals: Counter[str] = Counter()
    print(f"model={model} videos={len(paths)} segmentation=semantic input={input_root} output={result_root}", flush=True)
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=20.0)) as client:
        request = partial(call_glm_once, api_url=str(settings["api_url"]))
        for number, source_path in enumerate(paths, 1):
            output_path = result_root / source_path.name
            if output_path.is_file() and not args.overwrite:
                markdown_path = markdown_root / f"{source_path.stem}.md"
                if not markdown_path.is_file():
                    existing = load_json(output_path)
                    if isinstance(existing, dict):
                        errors = validate_segments(load_json(source_path), existing)
                        if errors:
                            raise ValueError("现有章节校验失败：" + "；".join(errors))
                        with video_lock(service_paths, source_path.stem):
                            commit_artifacts(
                                service_paths, source_path.stem,
                                text_files={markdown_path: segment_markdown(existing)},
                            )
                totals["skipped"] += 1
                print(f"[{number}/{len(paths)}] skip existing {source_path.name}", flush=True)
                continue
            if output_path.is_file() and args.overwrite:
                assert_can_overwrite("segments", data_root, partition, source_path.stem)
            try:
                source = load_json(source_path)
                if not isinstance(source, dict):
                    raise ValueError(f"cleaned ASR 顶层不是对象：{source_path}")
                video_id = str(source.get("source_id", "")).strip()
                if video_id != source_path.stem:
                    raise ValueError("cleaned_asr.source_id 与文件名不一致")
                sentences = meaningful_sentences(source)
                compact_prompt = segment_prompt_for(source, sentences)
                if len(compact_prompt) > args.max_input_chars:
                    raise RuntimeError(f"{source_path.name} 的紧凑输入为 {len(compact_prompt)} 字符，超过 --max-input-chars={args.max_input_chars}")
                title = str(source.get("title", source_path.stem))
                print(f"[{number}/{len(paths)}] {title} sentences={len(sentences)} input_chars={len(compact_prompt)}", flush=True)
                explanation_request = (
                    work_root / "segment_explanations" / partition / f"{source_path.stem}.request.json"
                )
                boundary_audit: dict | None = {} if explanation_request.is_file() else None
                if sentences:
                    plan, usage = generate_segments(
                        client, api_key, model, source, sentences,
                        args.max_tokens, args.thinking, args.attempts,
                        args.label_batch_max_chars, args.label_batch_max_segments,
                        request=request,
                        boundary_audit=boundary_audit,
                    )
                else:
                    plan, usage = [], Counter()
                document = build_document(
                    source, sentences, plan,
                    find_video_path(video_root, partition, video_id),
                )
                errors = validate_segments(source, document)
                if errors:
                    raise ValueError("章节结果校验失败：" + "；".join(errors))
                markdown_path = markdown_root / f"{source_path.stem}.md"
                with video_lock(service_paths, video_id):
                    if load_json(source_path) != source:
                        raise RuntimeError("清洗结果在分章期间发生变化，拒绝发布旧结果")
                    if output_path.is_file():
                        if not args.overwrite:
                            raise RuntimeError("章节结果在运行期间已生成，拒绝覆盖")
                        assert_can_overwrite("segments", data_root, partition, video_id)
                    commit_artifacts(
                        service_paths, video_id,
                        json_files={output_path: document},
                        text_files={markdown_path: segment_markdown(document)},
                    )
                if boundary_audit is not None:
                    explanation_output = (
                        history_root / "segment_explanations" / partition / f"{source_path.stem}.json"
                    )
                    write_json(explanation_output, {
                        "video_id": video_id,
                        "title": title,
                        "model": model,
                        "generated_at": now_iso(),
                        "one_time_request": load_json(explanation_request),
                        **boundary_audit,
                        "resulting_segments": [{
                            "segment_no": row["segment_no"],
                            "start_sentence_id": row["start_sentence_id"],
                            "end_sentence_id": row["end_sentence_id"],
                            "start_ms": row["start_ms"],
                            "end_ms": row["end_ms"],
                            "title": row["title"],
                        } for row in document["segments"]],
                    })
                    explanation_request.unlink(missing_ok=True)
                    print(f"  wrote one-time boundary explanation {explanation_output}", flush=True)
                emit_progress("segments", 100, 100, f"章节文件已保存，共 {len(plan)} 章")
                totals.update(videos=1, segments=len(plan), **usage)
                print(f"  wrote segments={len(plan)} tokens={usage.get('total_tokens', 0)} path={output_path}", flush=True)
            except Exception as error:
                totals["failed"] += 1
                print(f"  FAILED {source_path.name}: {error}", file=sys.stderr, flush=True)
                raise
    print(json.dumps({"counts": dict(totals)}, ensure_ascii=False, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config")
    result.add_argument("--data-root")
    result.add_argument("--partition")
    result.add_argument("--video-id")
    result.add_argument("--limit", type=int)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--model")
    result.add_argument("--api-key-env")
    result.add_argument("--max-input-chars", type=int, default=120_000)
    result.add_argument("--max-tokens", type=int, default=8192)
    result.add_argument("--attempts", type=int, default=3)
    result.add_argument("--label-batch-max-chars", type=int, default=12_000)
    result.add_argument("--label-batch-max-segments", type=int, default=6)
    result.add_argument("--timeout", type=float)
    result.add_argument("--thinking", action="store_true", help="默认关闭")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
