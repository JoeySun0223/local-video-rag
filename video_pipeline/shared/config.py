"""Load active pipeline settings without depending on the retired database app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import PACKAGE_ROOT, PROJECT_ROOT, load_json


CONFIG_PATH = PACKAGE_ROOT / "config.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = (path or CONFIG_PATH).resolve()
    value = load_json(config_path)
    if not isinstance(value, dict):
        raise ValueError(f"配置顶层必须是对象：{config_path}")
    value["_config_path"] = str(config_path)
    # The bundled config always belongs to this project, even when its path is
    # supplied explicitly by the Web worker. Other configs are self-contained.
    value["_root"] = str(
        PROJECT_ROOT
        if path is None or config_path == CONFIG_PATH.resolve()
        else config_path.parent
    )
    return value


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(str(config.get("_root", PROJECT_ROOT))) / path
    return path.resolve()
