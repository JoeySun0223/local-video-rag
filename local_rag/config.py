from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = (path or ROOT / "config.yaml").resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data["_config_path"] = str(config_path)
    data["_root"] = str(config_path.parent)
    return data


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_root"]) / path
    return path.resolve()


def ensure_project_dirs(config: dict[str, Any]) -> None:
    project = config["project"]
    for key in ("assets_dir", "sources_dir", "data_dir", "models_dir", "glossaries_dir"):
        if key not in project:
            continue
        project_path(config, project[key]).mkdir(parents=True, exist_ok=True)
    data = project_path(config, project["data_dir"])
    for child in ("cache", "indexes", "tasks", "builds", "clips", "runtime"):
        (data / child).mkdir(parents=True, exist_ok=True)


def database_path(config: dict[str, Any]) -> Path:
    return project_path(config, config["project"]["data_dir"]) / "metadata.db"


def index_dir(config: dict[str, Any]) -> Path:
    return project_path(config, config["project"]["data_dir"]) / "indexes"


def assets_dir(config: dict[str, Any]) -> Path:
    value = config["project"].get("assets_dir", config["project"]["data_dir"])
    return project_path(config, value)


def portable_asset_path(config: dict[str, Any], path: Path) -> str:
    resolved = path.resolve()
    root = assets_dir(config)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def resolve_asset_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (assets_dir(config) / path).resolve()
