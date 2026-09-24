"""Shared HTTP client for GLM JSON requests."""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

import httpx


DEFAULT_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def reasoning_parameters(model: str, thinking: bool) -> dict[str, Any]:
    """Return reasoning parameters accepted by the selected GLM generation."""
    if model.strip().lower().startswith("glm-5.3"):
        return {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high" if thinking else "low",
        }
    return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型返回的 JSON 顶层不是对象")
    return value


def request_json(
    client: httpx.Client,
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    thinking: bool = False,
    attempts: int = 3,
    progress_label: str = "GLM",
) -> tuple[dict[str, Any], dict[str, int]]:
    last_error: Exception | None = None
    correction = ""
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        finished = threading.Event()

        def report_wait() -> None:
            while not finished.wait(15):
                print(
                    f"  {progress_label} attempt={attempt}/{attempts} "
                    f"waiting={int(time.monotonic() - started)}s",
                    flush=True,
                )

        reporter = threading.Thread(target=report_wait, daemon=True)
        reporter.start()
        try:
            response = client.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + correction},
                    ],
                    "stream": False,
                    "do_sample": attempt > 1,
                    "temperature": 0.1 if attempt > 1 else 0,
                    **reasoning_parameters(model, thinking),
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                    "request_id": str(uuid.uuid4()),
                },
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[-500:]}")
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            content = str(message.get("content") or "")
            reasoning_content = str(message.get("reasoning_content") or "")
            try:
                value = parse_json_object(content)
            except (ValueError, json.JSONDecodeError) as error:
                preview = content[:160].replace("\r", " ").replace("\n", " ")
                raise ValueError(
                    "模型正文不是有效 JSON："
                    f"finish_reason={choice.get('finish_reason')!r}, "
                    f"content_chars={len(content)}, "
                    f"reasoning_chars={len(reasoning_content)}, "
                    f"content_preview={preview!r}; {error}"
                ) from error
            usage = body.get("usage") or {}
            return value, {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError, IndexError, json.JSONDecodeError) as error:
            last_error = error
            print(f"  {progress_label} attempt={attempt}/{attempts} failed: {error}", flush=True)
            correction = f"\n\n上一次输出无效：{error}。请重新输出完整且严格符合格式的 JSON。"
            if attempt < attempts:
                time.sleep(min(10, attempt * 2))
        finally:
            finished.set()
            reporter.join(timeout=1)
    raise RuntimeError(f"{progress_label} 在 {attempts} 次尝试后失败：{last_error}")
