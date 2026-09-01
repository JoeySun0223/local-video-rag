from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "Qwen/Qwen3-ASR-1.7B-hf": "qwen3-asr-1.7b-hf",
    "Qwen/Qwen3-ForcedAligner-0.6B-hf": "qwen3-forced-aligner-0.6b-hf",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(repo: str, target: Path, revision: str, allow_hf_fallback: bool) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    source = "modelscope-official-qwen"
    error = None
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download(model_id=repo, revision=revision, local_dir=str(target), max_workers=4)
    except Exception as exc:
        error = str(exc)
        if not allow_hf_fallback:
            raise
        from huggingface_hub import snapshot_download
        source = "huggingface-official-qwen-fallback"
        snapshot_download(repo_id=repo, revision=revision if revision != "master" else None, local_dir=str(target))
    required = [target / "config.json"]
    weights = sorted(target.glob("*.safetensors"))
    if not all(path.is_file() for path in required) or not weights:
        raise RuntimeError(f"官方模型下载不完整：{target}")
    # A deliberately interrupted single-stream attempt may leave an obsolete
    # resumable file after a later parallel download succeeds.
    for incomplete in target.glob("*.incomplete"):
        incomplete.unlink()
    critical = [*required, *weights]
    provenance = {
        "repo": repo, "source": source, "requested_revision": revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "modelscope_error_before_fallback": error,
        "total_bytes": sum(path.stat().st_size for path in target.rglob("*") if path.is_file()),
        "critical_files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in critical},
    }
    (target / "download_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="优先从ModelScope官方Qwen仓库下载Qwen3-ASR")
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--allow-hf-fallback", action="store_true")
    args = parser.parse_args()
    report = {}
    for repo, child in MODELS.items():
        report[repo] = download_model(repo, (args.models_dir / child).resolve(), args.revision, args.allow_hf_fallback)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
