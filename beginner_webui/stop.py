"""Stop a local pipeline Web server through its loopback API."""

from __future__ import annotations

import argparse

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args()
    try:
        response = httpx.post(f"http://127.0.0.1:{args.port}/api/system/shutdown", timeout=5)
        response.raise_for_status()
        print("关闭请求已发送。")
        return 0
    except Exception as error:
        print(f"Web UI 未运行或无法关闭：{error}")
        return 1


raise SystemExit(main())
