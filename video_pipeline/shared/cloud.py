"""Folder-local cloud credentials shared by both Web UIs and worker processes."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from .config import project_path
from .glm import request_json
from .io import load_json, now_iso, write_json


CREDENTIALS_FILE = "resources/cloud_credentials.json"


def credentials_path(config: dict[str, Any]) -> Path:
    return project_path(config, CREDENTIALS_FILE)


def _saved_credentials(config: dict[str, Any]) -> dict[str, Any] | None:
    path = credentials_path(config)
    if not path.is_file():
        return None
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"API 配置必须是对象：{path}")
    return value


def cloud_settings(
    config: dict[str, Any],
    *,
    model: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Resolve portable credentials first, with environment fallback before disconnect."""
    cloud = config.get("cloud")
    if not isinstance(cloud, dict):
        raise ValueError("配置缺少 cloud 设置")
    saved = _saved_credentials(config)
    disconnected = bool(saved and saved.get("disconnected"))
    saved_key = str((saved or {}).get("api_key") or "").strip()
    env_name = api_key_env or str(cloud.get("api_key_env", "ZHIPUAI_API_KEY"))
    env_key = "" if disconnected or saved_key else os.environ.get(env_name, "").strip()
    key = saved_key or env_key
    return {
        "api_key": key,
        "api_url": str((saved or {}).get("api_url") or cloud.get("api_url") or "").strip(),
        "model": str(model or (saved or {}).get("model") or cloud.get("model") or "glm-5.3").strip(),
        "thinking": bool(cloud.get("thinking", False)),
        "timeout_seconds": float(cloud.get("timeout_seconds", 300)),
        "source": "portable" if saved_key else "environment" if env_key else "none",
        "persisted": bool(saved_key),
    }


def cloud_status(config: dict[str, Any]) -> dict[str, Any]:
    settings = cloud_settings(config)
    key = str(settings["api_key"])
    return {
        "connected": bool(key),
        "persisted": bool(settings["persisted"]),
        "source": settings["source"],
        "model": settings["model"],
        "api_url": settings["api_url"],
        "key_hint": f"••••{key[-4:]}" if key else "",
    }


def test_cloud_connection(
    *, api_url: str, api_key: str, model: str, timeout_seconds: float = 30
) -> dict[str, Any]:
    if not api_url.strip():
        raise ValueError("API 地址不能为空")
    if not api_key.strip():
        raise ValueError("API Key 不能为空")
    if not model.strip():
        raise ValueError("模型名称不能为空")
    started = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=15.0)) as client:
        value, usage = request_json(
            client,
            api_url=api_url.strip(), api_key=api_key.strip(), model=model.strip(),
            system_prompt="你是 API 连通测试助手，只输出 JSON。",
            user_prompt='请只返回 {"status":"ok"}。',
            max_tokens=64, thinking=False, attempts=1, progress_label="API 连接测试",
        )
    if str(value.get("status", "")).lower() != "ok":
        raise RuntimeError("API 已响应，但没有返回预期的连通测试结果")
    return {"latency_ms": round((time.monotonic() - started) * 1000), "usage": usage}


def connect_cloud(
    config: dict[str, Any], *, api_url: str, api_key: str | None, model: str
) -> dict[str, Any]:
    current = cloud_settings(config)
    effective_key = str(api_key or "").strip() or str(current["api_key"])
    tested = test_cloud_connection(
        api_url=api_url, api_key=effective_key, model=model,
        timeout_seconds=min(60.0, float(current["timeout_seconds"])),
    )
    write_json(credentials_path(config), {
        "api_url": api_url.strip(),
        "api_key": effective_key,
        "model": model.strip(),
        "connected_at": now_iso(),
    })
    return {**cloud_status(config), **tested}


def disconnect_cloud(config: dict[str, Any]) -> dict[str, Any]:
    current = cloud_settings(config)
    write_json(credentials_path(config), {
        "api_url": current["api_url"],
        "model": current["model"],
        "disconnected": True,
        "disconnected_at": now_iso(),
    })
    return cloud_status(config)
