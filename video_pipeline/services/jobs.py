"""单 worker 视频任务队列、进度与取消。"""

from __future__ import annotations

import os
import locale
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from video_pipeline.shared.io import PROJECT_ROOT, load_json, now_iso, write_json
from video_pipeline.shared.progress import parse_progress_event
from .catalog import PipelinePaths, validate_video_id
from .storage import TransactionRecoveryError, recover_video_transactions


STAGES = {"pipeline", "asr", "cleanup", "segments"}
PIPELINE_STAGES = ("asr", "cleanup", "segments")
STAGE_PROGRESS = {
    "asr": (5, 70),
    "cleanup": (72, 85),
    "segments": (87, 100),
}


class JobManager:
    def __init__(self, paths: PipelinePaths, config_path: Path):
        self.paths = paths
        self.config_path = config_path.resolve()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._queue: deque[str] = deque()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._cancel_requested: set[str] = set()
        self._condition = threading.Condition()
        self._job_root = paths.work_root / "jobs" / paths.partition
        self._job_root.mkdir(parents=True, exist_ok=True)
        self._last_persist: dict[str, float] = {}
        self._load_jobs()
        self._worker = threading.Thread(target=self._work, daemon=True, name="pipeline-web-worker")
        self._worker.start()

    def _persist(self, job: dict[str, Any], *, force: bool = False) -> None:
        """写入单用户队列快照；进度事件限频，状态变化立即保存。"""
        job_id = str(job["id"])
        current = time.monotonic()
        if not force and current - self._last_persist.get(job_id, 0) < 0.5:
            return
        write_json(self._job_root / f"{job_id}.json", job)
        self._last_persist[job_id] = current

    def _load_jobs(self) -> None:
        """重启后展示历史；中断的任务明确标记，不自动重复执行。"""
        for path in sorted(self._job_root.glob("*.json")):
            try:
                job = load_json(path)
                if not isinstance(job, dict) or job.get("id") != path.stem:
                    continue
                if job.get("status") in {"queued", "running", "cancelling"}:
                    job["status"] = "interrupted"
                    job["finished_at"] = now_iso()
                    job["progress_detail"] = "服务重启，任务未自动重试"
                    for item in job.get("items", []):
                        if item.get("status") in {"queued", "running"}:
                            item["status"] = "interrupted"
                    self._persist(job, force=True)
                self._jobs[path.stem] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def create(self, stage: str, video_ids: list[str]) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"不支持的阶段：{stage}")
        ids = list(dict.fromkeys(validate_video_id(item) for item in video_ids))
        if not ids:
            raise ValueError("至少选择一个视频")
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "stage": stage,
            "status": "queued",
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "current": None,
            "current_stage": None,
            "progress_detail": "",
            "progress": 0,
            "cancellation_requested": False,
            "items": [{
                "video_id": item, "status": "queued", "log": "",
                "current_stage": None, "progress_detail": "", "progress": 0,
            } for item in ids],
        }
        with self._condition:
            self._jobs[job_id] = job
            self._queue.append(job_id)
            self._persist(job, force=True)
            self._condition.notify()
        return self.snapshot(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                **job,
                "can_cancel": job["status"] in {"queued", "running", "cancelling"},
                "items": [dict(item) for item in job["items"]],
            }

    def cancel(self, job_id: str) -> dict[str, Any]:
        """取消排队任务，或终止正在运行的子进程树。"""
        process: subprocess.Popen[str] | None = None
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] not in {"queued", "running", "cancelling"}:
                raise ValueError("任务已经结束，无法终止")
            self._cancel_requested.add(job_id)
            job["cancellation_requested"] = True
            if job["status"] == "queued":
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                for item in job["items"]:
                    item["status"] = "cancelled"
                    item["log"] = (item["log"] + "\n用户已终止任务。\n")[-30_000:]
                job["status"] = "cancelled"
                job["finished_at"] = now_iso()
                self._cancel_requested.discard(job_id)
            else:
                job["status"] = "cancelling"
                process = self._processes.get(job_id)
            self._persist(job, force=True)
        if process is not None:
            self._terminate_process_tree(process)
        return self.snapshot(job_id)

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False, capture_output=True, timeout=10,
                )
            else:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        except (OSError, subprocess.SubprocessError):
            process.kill()

    def _is_cancelled(self, job: dict[str, Any]) -> bool:
        with self._condition:
            return str(job["id"]) in self._cancel_requested

    def _set_progress(
        self, job: dict[str, Any], item: dict[str, Any], stage: str, progress: int,
        detail: str | None = None,
    ) -> None:
        with self._condition:
            item["current_stage"] = stage
            item["progress"] = max(int(item.get("progress", 0)), progress)
            if detail is not None:
                item["progress_detail"] = detail
            job["current_stage"] = stage
            job["progress_detail"] = item.get("progress_detail", "")
            item_index = job["items"].index(item)
            job["progress"] = round(
                (item_index * 100 + int(item["progress"])) / len(job["items"])
            )
            self._persist(job)

    def _accept_progress_event(
        self, job: dict[str, Any], item: dict[str, Any], expected_stage: str, line: str
    ) -> bool:
        event = parse_progress_event(line.strip())
        if event is None or event["stage"] != expected_stage:
            return False
        start, end = STAGE_PROGRESS[expected_stage]
        progress = start + round((end - start) * float(event["fraction"]))
        self._set_progress(
            job, item, expected_stage, progress, str(event.get("detail") or "")
        )
        return True

    def list(self) -> list[dict[str, Any]]:
        with self._condition:
            ids = list(self._jobs)[-20:]
        return [self.snapshot(job_id) for job_id in reversed(ids)]

    def is_busy(self) -> bool:
        with self._condition:
            return bool(self._queue) or any(
                job["status"] in {"running", "cancelling"} for job in self._jobs.values()
            )

    def is_video_busy(self, video_id: str) -> bool:
        with self._condition:
            return any(
                job["status"] in {"queued", "running", "cancelling"}
                and any(
                    item["video_id"] == video_id and item["status"] in {"queued", "running"}
                    for item in job["items"]
                )
                for job in self._jobs.values()
            )

    def _command(self, stage: str, video_id: str) -> list[str]:
        base = [sys.executable, "-m"]
        common = ["--config", str(self.config_path), "--partition", self.paths.partition]
        if stage == "asr":
            candidate = self.paths.artifact(video_id, "upload")
            if candidate is None or not candidate.is_file():
                candidate = self.paths.artifact(video_id, "video")
            if candidate is None or not candidate.is_file():
                raise FileNotFoundError(f"找不到上传或正式视频：{video_id}")
            return [*base, "video_pipeline.asr", str(candidate), *common]
        module = "video_pipeline.cleanup" if stage == "cleanup" else "video_pipeline.segments"
        return [*base, module, *common, "--video-id", video_id]

    def _already_done(self, stage: str, video_id: str) -> str | None:
        if stage == "asr" and (self.paths.artifact(video_id, "raw") or Path()).is_file():
            return "已有原始 ASR"
        if stage == "cleanup":
            if (self.paths.artifact(video_id, "cleaned") or Path()).is_file():
                return "已有清洗结果"
        if stage == "segments" and (self.paths.artifact(video_id, "segments") or Path()).is_file():
            return "已有语义片段"
        return None

    def _run_process(
        self, job: dict[str, Any], item: dict[str, Any], stage: str
    ) -> int:
        video_id = str(item["video_id"])
        command = self._command(stage, video_id)
        item["status"] = "running"
        job["current"] = video_id
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=locale.getpreferredencoding(False) if os.name == "nt" else "utf-8",
            errors="replace",
            bufsize=1,
        )
        with self._condition:
            self._processes[str(job["id"])] = process
            cancel_now = str(job["id"]) in self._cancel_requested
        if cancel_now:
            self._terminate_process_tree(process)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if self._accept_progress_event(job, item, stage, line):
                    continue
                with self._condition:
                    item["log"] = (item["log"] + line)[-30_000:]
                    self._persist(job)
            return_code = process.wait()
        finally:
            try:
                # 日志读取或进度持久化异常时，也要先确保子进程停止写入。
                if process.poll() is None:
                    self._terminate_process_tree(process)
                process.wait()
                with self._condition:
                    item["progress_detail"] = "正在检查并恢复数据"
                    job["progress_detail"] = item["progress_detail"]
                recover_video_transactions(self.paths, video_id)
            finally:
                process.stdout.close()
                with self._condition:
                    self._processes.pop(str(job["id"]), None)
        return return_code

    def _check_stage_prerequisites(self, stage: str, video_id: str) -> None:
        if stage in {"cleanup", "segments"} and not (self.paths.artifact(video_id, "raw") or Path()).is_file():
            raise FileNotFoundError("尚未生成原始 ASR")
        if stage == "segments" and not (self.paths.artifact(video_id, "cleaned") or Path()).is_file():
            raise FileNotFoundError("尚未完成清洗或人工审核")

    def _run_pipeline_item(self, job: dict[str, Any], item: dict[str, Any]) -> bool:
        video_id = str(item["video_id"])
        item["status"] = "running"
        for stage in PIPELINE_STAGES:
            if self._is_cancelled(job):
                item["status"] = "cancelled"
                return False
            stage_start, stage_end = STAGE_PROGRESS[stage]
            self._set_progress(job, item, stage, stage_start, "正在准备")
            item["log"] = (item["log"] + f"\n=== {stage} ===\n")[-30_000:]
            existing = self._already_done(stage, video_id)
            if existing:
                item["log"] = (item["log"] + existing + "\n")[-30_000:]
                self._set_progress(job, item, stage, stage_end)
                continue
            self._check_stage_prerequisites(stage, video_id)
            return_code = self._run_process(job, item, stage)
            if self._is_cancelled(job):
                item["status"] = "cancelled"
                return False
            if return_code != 0:
                item["status"] = "failed"
                return False
            self._set_progress(job, item, stage, stage_end)
        item["status"] = "completed"
        item["progress"] = 100
        return True

    def _run_item(self, job: dict[str, Any], item: dict[str, Any]) -> bool:
        stage = str(job["stage"])
        video_id = str(item["video_id"])
        if self._is_cancelled(job):
            item["status"] = "cancelled"
            return False
        recover_video_transactions(self.paths, video_id)
        if stage == "pipeline":
            return self._run_pipeline_item(job, item)
        existing = self._already_done(stage, video_id)
        if existing:
            item["status"] = "skipped"
            item["log"] = existing
            return True
        self._check_stage_prerequisites(stage, video_id)
        start, end = STAGE_PROGRESS[stage]
        self._set_progress(job, item, stage, start, "正在准备")
        return_code = self._run_process(job, item, stage)
        if self._is_cancelled(job):
            item["status"] = "cancelled"
            return False
        if return_code == 0:
            item["status"] = "completed"
            self._set_progress(job, item, stage, end)
            return True
        item["status"] = "failed"
        return False

    def _work(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                job = self._jobs[self._queue.popleft()]
                job["status"] = "running"
                job["started_at"] = now_iso()
                self._persist(job, force=True)
            failed = False
            recovery_failed = False
            for item in job["items"]:
                try:
                    if not self._run_item(job, item):
                        failed = True
                        break
                except Exception as error:
                    recovery_failed = isinstance(error, TransactionRecoveryError)
                    cancelled = self._is_cancelled(job) and not recovery_failed
                    item["status"] = "cancelled" if cancelled else "failed"
                    if not cancelled:
                        item["log"] = (item["log"] + f"\n{type(error).__name__}: {error}\n")[-30_000:]
                    if recovery_failed:
                        item["progress_detail"] = str(error)
                        job["progress_detail"] = str(error)
                    failed = True
                    break
            with self._condition:
                cancelled = str(job["id"]) in self._cancel_requested and not recovery_failed
                if failed or cancelled:
                    for item in job["items"]:
                        if item["status"] == "queued":
                            item["status"] = "cancelled"
                            item["log"] = "任务已终止" if cancelled else "前一项失败，后续处理已停止"
                job["current"] = None
                job["current_stage"] = None
                job["status"] = "cancelled" if cancelled else ("failed" if failed else "completed")
                if not failed and not cancelled:
                    job["progress"] = 100
                job["finished_at"] = now_iso()
                self._cancel_requested.discard(str(job["id"]))
                self._persist(job, force=True)
