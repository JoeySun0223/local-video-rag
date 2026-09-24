"""Validate asr_raw, cleaned_asr and semantic_segments without calling a model."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .rules import validate_cleaned, validate_raw, validate_segments, validate_video_path
from ..shared.config import load_config, project_path
from ..shared.io import load_json


def run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    partition = args.partition or str(config["partition"])
    data_root = project_path(config, args.data_root or config["data_root"])
    raw_root = data_root / "asr_raw" / partition
    paths = sorted(raw_root.glob("*.json"))
    if args.video_id:
        paths = [path for path in paths if path.stem == args.video_id]
        if not paths:
            raise FileNotFoundError(f"找不到 asr_raw：{args.video_id}")
    totals: Counter[str] = Counter()
    for path in paths:
        source_id = path.stem
        errors = validate_raw(load_json(path), source_id)
        raw = load_json(path)
        clean_path = data_root / "cleaned_asr" / partition / path.name
        segment_path = data_root / "semantic_segments" / partition / path.name
        if not clean_path.is_file():
            errors.append(f"缺少 cleaned_asr：{clean_path}")
        else:
            cleaned = load_json(clean_path)
            errors.extend(validate_cleaned(raw, cleaned))
            if not segment_path.is_file():
                errors.append(f"缺少 semantic_segments：{segment_path}")
            else:
                document = load_json(segment_path)
                errors.extend(validate_segments(cleaned, document))
                errors.extend(validate_video_path(document, data_root))
        if errors:
            totals["failed"] += 1
            print(f"FAILED {source_id}")
            for error in errors:
                print(f"  - {error}")
        else:
            totals["passed"] += 1
    print(f"validation passed={totals['passed']} failed={totals['failed']} total={len(paths)}")
    return 1 if totals["failed"] else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config")
    result.add_argument("--data-root")
    result.add_argument("--partition")
    result.add_argument("--video-id")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
