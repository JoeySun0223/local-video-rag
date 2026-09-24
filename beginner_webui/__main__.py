"""Start the local pipeline Web server."""

from __future__ import annotations

import argparse
import os
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn

from video_pipeline.shared.config import load_config
from .app import create_app


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config")
    result.add_argument("--partition")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8876)
    result.add_argument("--no-browser", action="store_true")
    return result


def _wait_and_open(url: str) -> None:
    """Open the page only after Uvicorn is ready to accept requests."""
    for _ in range(60):
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status < 500:
                    webbrowser.open(url)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)


def _is_our_server(url: str) -> bool:
    """Return True when this UI is already running on the requested address."""
    try:
        with urllib.request.urlopen(url, timeout=0.8) as response:
            body = response.read(64_000).decode("utf-8", errors="ignore")
            return response.status < 500 and "视频数据处理" in body
    except (OSError, urllib.error.URLError):
        return False


def main() -> int:
    args = parser().parse_args()
    url = f"http://{args.host}:{args.port}"
    if _is_our_server(url):
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    config_path = Path(args.config).resolve() if args.config else None
    config = load_config(config_path)
    state_root = Path(__file__).resolve().parent / "runtime"
    state_root.mkdir(parents=True, exist_ok=True)
    pid_path = state_root / "server.pid"
    pid_path.write_text(str(os.getpid()), encoding="ascii")
    app = create_app(config_path, args.partition)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info"))
    app.state.stop_callback = lambda: setattr(server, "should_exit", True)
    if not args.no_browser:
        threading.Thread(target=_wait_and_open, args=(url,), daemon=True).start()
    try:
        server.run()
        return 0
    finally:
        if pid_path.is_file() and pid_path.read_text(encoding="ascii").strip() == str(os.getpid()):
            pid_path.unlink()


raise SystemExit(main())
