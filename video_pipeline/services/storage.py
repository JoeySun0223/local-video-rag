"""按视频保护读改写，并让一组派生文件在失败后可恢复。"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

from filelock import FileLock

from ..shared.io import write_json
from .catalog import PipelinePaths, validate_video_id


_LOCKS: dict[Path, FileLock] = {}
_LOCKS_GUARD = threading.Lock()


class TransactionRecoveryError(RuntimeError):
    """未完成事务无法恢复时，禁止继续修改该视频。"""


def _check_targets(paths: PipelinePaths, video_id: str, targets: list[Path]) -> None:
    """事务日志只允许指向该视频的四种可变文件。"""
    allowed = {
        paths.artifact(video_id, kind)
        for kind in ("cleaned", "segments", "markdown", "history")
    }
    if any(target.resolve() not in allowed for target in targets):
        raise ValueError("事务包含非本视频文件，拒绝写入或恢复")


@contextmanager
def _file_lock(paths: PipelinePaths, video_id: str) -> Iterator[FileLock]:
    video_id = validate_video_id(video_id)
    lock_path = paths.work_root / "locks" / paths.partition / f"{video_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(lock_path.resolve(), FileLock(str(lock_path)))
    with lock:
        yield lock


@contextmanager
def video_lock(paths: PipelinePaths, video_id: str) -> Iterator[None]:
    """在读改写之前恢复残留事务；同一线程的嵌套服务调用可重入。"""
    with _file_lock(paths, video_id) as lock:
        if lock.lock_counter == 1:
            _recover_video_transactions(paths, validate_video_id(video_id))
        yield


def locked_video_action(function: Callable[..., Any]) -> Callable[..., Any]:
    """装饰以 (paths, video_id) 开头的服务写操作。"""
    @wraps(function)
    def wrapped(paths: PipelinePaths, video_id: str, *args: Any, **kwargs: Any) -> Any:
        with video_lock(paths, video_id):
            return function(paths, video_id, *args, **kwargs)
    return wrapped


def _replace_bytes(path: Path, content: bytes) -> None:
    """每次使用独立临时文件，避免同名 .tmp 在并发时互相覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def commit_artifacts(
    paths: PipelinePaths,
    video_id: str,
    *,
    json_files: dict[Path, Any] | None = None,
    text_files: dict[Path, str] | None = None,
) -> None:
    """先备份同一次修改的全部文件，再逐个替换；失败时恢复旧版本。

    进程意外退出会留下 manifest；worker 退出后或下次取得视频锁时恢复。
    调用方应持有 video_lock，且只传入服务层已解析的项目内路径。
    """
    video_id = validate_video_id(video_id)
    pending: dict[Path, bytes] = {}
    for path, value in (json_files or {}).items():
        pending[path.resolve()] = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for path, value in (text_files or {}).items():
        pending[path.resolve()] = value.encode("utf-8")
    if not pending:
        return
    _check_targets(paths, video_id, list(pending))

    transaction = paths.work_root / "transactions" / paths.partition / video_id / uuid.uuid4().hex
    transaction.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    committed = False
    restored = False
    try:
        for index, path in enumerate(pending):
            backup = f"{index}.backup"
            existed = path.is_file()
            if existed:
                shutil.copy2(path, transaction / backup)
            entries.append({"path": str(path), "backup": backup, "existed": existed})
        write_json(transaction / "manifest.json", {"entries": entries})
        for path, content in pending.items():
            _replace_bytes(path, content)
        # 完成标记必须持久化；即使随后清理目录时被终止，也不能再回滚。
        write_json(transaction / "manifest.json", {"entries": entries, "state": "committed"})
        committed = True
    except BaseException:
        if (transaction / "manifest.json").is_file():
            manifest = json.loads((transaction / "manifest.json").read_text(encoding="utf-8"))
            committed = manifest.get("state") == "committed"
            if not committed:
                _restore(transaction, entries)
                restored = True
        raise
    finally:
        if (committed or restored) and transaction.is_dir():
            try:
                _discard_transaction(transaction)
            except OSError:
                # 已提交/已恢复的数据有效；残留目录下次取得锁时继续清理。
                pass


def _discard_transaction(transaction: Path) -> None:
    # 先移除日志，再删备份。中途终止也不会留下指向已删除备份的有效日志。
    (transaction / "manifest.json").unlink(missing_ok=True)
    shutil.rmtree(transaction)


def _restore(transaction: Path, entries: list[dict[str, Any]]) -> None:
    backups = [
        (transaction / entry["backup"]).read_bytes() if entry["existed"] else None
        for entry in entries
    ]
    for entry, content in zip(entries, backups, strict=True):
        path = Path(entry["path"])
        if entry["existed"]:
            _replace_bytes(path, content)
        else:
            path.unlink(missing_ok=True)


def _recover_video_transactions(paths: PipelinePaths, video_id: str) -> int:
    """调用方持有文件锁；恢复失败时保留日志，后续写入仍会被拦截。"""
    root = paths.work_root / "transactions" / paths.partition / video_id
    recovered = 0
    try:
        if not root.is_dir():
            return 0
        for transaction in sorted(root.iterdir()):
            if not transaction.is_dir():
                continue
            manifest = transaction / "manifest.json"
            if manifest.is_file():
                value = json.loads(manifest.read_text(encoding="utf-8"))
                entries = value["entries"]
                if value.get("state") not in {None, "committed"} or not isinstance(entries, list) or any(
                    not isinstance(entry, dict)
                    or entry.get("backup") != f"{index}.backup"
                    or not isinstance(entry.get("existed"), bool)
                    or not isinstance(entry.get("path"), str)
                    for index, entry in enumerate(entries)
                ):
                    raise ValueError("事务日志格式非法，已停止自动恢复")
                _check_targets(paths, video_id, [Path(entry["path"]) for entry in entries])
                if value.get("state") != "committed":
                    _restore(transaction, entries)
                    recovered += 1
            _discard_transaction(transaction)
    except Exception as error:
        raise TransactionRecoveryError(
            f"视频 {video_id[:12]} 的数据恢复失败，已暂停该视频写入，请修复事务记录后重试：{error}"
        ) from error
    return recovered


def recover_video_transactions(paths: PipelinePaths, video_id: str) -> int:
    """子进程退出后，在任务解除占用前恢复该视频的未完成写入。"""
    video_id = validate_video_id(video_id)
    with _file_lock(paths, video_id):
        return _recover_video_transactions(paths, video_id)


def recover_transactions(paths: PipelinePaths) -> int:
    """恢复服务中断时留下的多文件修改，再开放界面读写。"""
    root = paths.work_root / "transactions" / paths.partition
    if not root.is_dir():
        return 0
    recovered = 0
    for video_dir in root.iterdir():
        if not video_dir.is_dir():
            continue
        video_id = validate_video_id(video_dir.name)
        recovered += recover_video_transactions(paths, video_id)
    return recovered
