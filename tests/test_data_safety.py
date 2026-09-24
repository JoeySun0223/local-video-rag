"""人工正文、词表并发替换及 worker 强制中断的回归测试。"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock, Timeout

from video_pipeline.services import glossary, jobs, storage
from video_pipeline.services.catalog import PipelinePaths, delete_video
from video_pipeline.services.editing import (
    EditConflictError, confirm_segments, save_chapter_fields, save_cleaned_sentence,
)
from video_pipeline.services.jobs import JobManager
from video_pipeline.shared.io import load_json, write_json
from video_pipeline.validation.rules import validate_cleaned, validate_segments


VIDEO_ID = "a" * 64
TERMS = [{"term": "新词", "aliases": ["旧词"], "scope": "global"}]


class EditingSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pipeline-safety-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = PipelinePaths("test", self.root / "data", self.root / "history", self.root / "work")
        self.seed(self.paths)

    def seed(self, paths):
        texts = ["甲乙丙丁。", "戊己庚辛。", "旧词。", "尾句。"]
        raw = {"id": VIDEO_ID, "duration_ms": 4000, "sentences": [
            {"id": i, "start_ms": (i - 1) * 1000, "end_ms": i * 1000, "raw_text": text}
            for i, text in enumerate(texts, 1)
        ]}
        cleaned = {"source_id": VIDEO_ID, "sentences": [
            {"sentence_id": row["id"], "start_ms": row["start_ms"],
             "end_ms": row["end_ms"], "text": row["raw_text"]} for row in raw["sentences"]
        ], "full_text": "\n".join(texts)}
        document = {"video_id": VIDEO_ID, "segments": [
            {"segment_no": i, "title": "章节", "summary": "摘要。", "keywords": [],
             "start_sentence_id": start, "end_sentence_id": end,
             "start_ms": (start - 1) * 1000, "end_ms": end * 1000,
             "content": "".join(texts[start - 1:end])}
            for i, (start, end) in enumerate([(1, 2), (3, 4)], 1)
        ]}
        for kind, value in [("raw", raw), ("cleaned", cleaned), ("segments", document)]:
            write_json(paths.artifact(VIDEO_ID, kind), value)

    def hits(self):
        return glossary.search_glossary_occurrences(self.paths, VIDEO_ID, TERMS, "video")

    def apply(self, hits):
        return glossary.apply_glossary_occurrences(
            self.paths, VIDEO_ID, TERMS, "video", [hit["hit_id"] for hit in hits],
        )

    def assert_valid(self):
        raw = load_json(self.paths.artifact(VIDEO_ID, "raw"))
        cleaned = load_json(self.paths.artifact(VIDEO_ID, "cleaned"))
        document = load_json(self.paths.artifact(VIDEO_ID, "segments"))
        self.assertEqual(validate_cleaned(raw, cleaned), [])
        self.assertEqual(validate_segments(cleaned, document), [])

    def test_glossary_holds_video_lock_until_save_and_preserves_other_editor(self):
        hits = self.hits()
        at_save, release = threading.Event(), threading.Event()
        original_save = glossary.save_cleaned_sentences

        def paused_save(*args, **kwargs):
            at_save.set()
            if not release.wait(5):
                raise TimeoutError("测试未释放词表保存")
            return original_save(*args, **kwargs)

        with patch.object(glossary, "save_cleaned_sentences", side_effect=paused_save):
            with ThreadPoolExecutor(max_workers=2) as executor:
                applying = executor.submit(self.apply, hits)
                try:
                    self.assertTrue(at_save.wait(5))
                    lock_path = self.paths.work_root / "locks" / "test" / f"{VIDEO_ID}.lock"
                    with self.assertRaises(Timeout):
                        with FileLock(str(lock_path), timeout=0.15):
                            pass
                    editing = executor.submit(save_cleaned_sentence, self.paths, VIDEO_ID, 1, "人工修改。")
                finally:
                    release.set()
                self.assertEqual(applying.result(timeout=5)["changed_sentences"], 1)
                editing.result(timeout=5)
        rows = load_json(self.paths.artifact(VIDEO_ID, "cleaned"))["sentences"]
        self.assertEqual(rows[0]["text"], "人工修改。")
        self.assertEqual(rows[2]["text"], "新词。")
        self.assert_valid()

    def test_edit_between_search_and_lock_is_preserved(self):
        hits = self.hits()

        @contextmanager
        def edit_then_lock(paths, video_id):
            save_cleaned_sentence(paths, video_id, 1, "已保存的人工修改。")
            with storage.video_lock(paths, video_id):
                yield

        with patch.object(glossary, "video_lock", edit_then_lock):
            self.apply(hits)
        rows = load_json(self.paths.artifact(VIDEO_ID, "cleaned"))["sentences"]
        self.assertEqual(rows[0]["text"], "已保存的人工修改。")
        self.assertEqual(rows[2]["text"], "新词。")

    def test_changed_selected_sentence_requires_new_search(self):
        hits = self.hits()
        save_cleaned_sentence(self.paths, VIDEO_ID, 3, "旧词的新解释。")
        with self.assertRaises(EditConflictError):
            self.apply(hits)
        self.assertEqual(load_json(self.paths.artifact(VIDEO_ID, "cleaned"))["sentences"][2]["text"], "旧词的新解释。")

    def test_selected_sentence_is_rechecked_after_lock_acquisition(self):
        hits = self.hits()

        @contextmanager
        def edit_then_lock(paths, video_id):
            save_cleaned_sentence(paths, video_id, 3, "旧词的新解释。")
            with storage.video_lock(paths, video_id):
                yield

        with patch.object(glossary, "video_lock", edit_then_lock):
            with self.assertRaises(EditConflictError):
                self.apply(hits)

    def test_database_scope_still_supports_multiple_partitions(self):
        other = replace(self.paths, partition="other")
        self.seed(other)
        hits = glossary.search_glossary_occurrences(self.paths, VIDEO_ID, TERMS, "database")
        result = glossary.apply_glossary_occurrences(
            self.paths, VIDEO_ID, TERMS, "database", [hit["hit_id"] for hit in hits],
        )
        self.assertEqual(result["changed_videos"], 2)
        for paths in [self.paths, other]:
            self.assertEqual(load_json(paths.artifact(VIDEO_ID, "cleaned"))["sentences"][2]["text"], "新词。")

    def test_other_chapter_updates_preserve_manual_words_and_boundary_spaces(self):
        for content in ["ABCDEFGHIJ", "ABCDE FGHIJ", "测试 ABC 中文 def。"]:
            with self.subTest(content=content):
                self.seed(self.paths)
                save_chapter_fields(
                    self.paths, VIDEO_ID, 1, title="人工章节", summary="摘要。",
                    content=content, keywords=[], expected_revision=0,
                )
                before = load_json(self.paths.artifact(VIDEO_ID, "cleaned"))["sentences"][:2]
                self.apply(self.hits())
                save_cleaned_sentence(self.paths, VIDEO_ID, 4, "另一章修改。")
                after = load_json(self.paths.artifact(VIDEO_ID, "cleaned"))["sentences"][:2]
                self.assertEqual(after, before)
                document = load_json(self.paths.artifact(VIDEO_ID, "segments"))
                self.assertEqual(document["segments"][0]["content"], content)
                self.assertIn(content, self.paths.artifact(VIDEO_ID, "markdown").read_text(encoding="utf-8"))
                confirm_segments(self.paths, VIDEO_ID)
                self.assert_valid()

    def test_manual_chapter_still_rebuilds_when_its_own_sentence_changes(self):
        save_chapter_fields(
            self.paths, VIDEO_ID, 1, title="人工章节", summary="摘要。",
            content="ABCDEFGHIJ", keywords=[], expected_revision=0,
        )
        save_cleaned_sentence(self.paths, VIDEO_ID, 2, "KLMNO")
        document = load_json(self.paths.artifact(VIDEO_ID, "segments"))
        self.assertEqual(document["segments"][0]["content"], "ABCDEKLMNO")
        self.assert_valid()

    def test_automatic_and_legacy_manual_chapters_keep_english_word_spaces(self):
        for manual in [False, True]:
            with self.subTest(manual=manual):
                self.seed(self.paths)
                cleaned = load_json(self.paths.artifact(VIDEO_ID, "cleaned"))
                cleaned["sentences"][0]["text"] = "Hello"
                cleaned["sentences"][1]["text"] = "world."
                cleaned["full_text"] = "\n".join(row["text"] for row in cleaned["sentences"])
                document = load_json(self.paths.artifact(VIDEO_ID, "segments"))
                document["segments"][0].update(content="Hello world.", manual_content_override=manual)
                write_json(self.paths.artifact(VIDEO_ID, "cleaned"), cleaned)
                write_json(self.paths.artifact(VIDEO_ID, "segments"), document)
                self.apply(self.hits())
                document = load_json(self.paths.artifact(VIDEO_ID, "segments"))
                self.assertEqual(document["segments"][0]["content"], "Hello world.")
                self.assert_valid()


WORKER_SCRIPT = r'''
import sys, threading
from pathlib import Path
from video_pipeline.services import storage
from video_pipeline.services.catalog import PipelinePaths
root, phase = Path(sys.argv[1]), sys.argv[2]
paths = PipelinePaths('test', root/'data', root/'history', root/'work')
video_id = 'a'*64
segments, cleaned = paths.artifact(video_id,'segments'), paths.artifact(video_id,'cleaned')
def pause():
    (root/'ready').write_text('ready')
    threading.Event().wait(30)
replace_bytes = storage._replace_bytes
def writing(path, content):
    replace_bytes(path, content)
    if phase in {'writing', 'broken_backup'} and path == segments:
        if phase == 'broken_backup':
            for backup in (paths.work_root/'transactions').rglob('1.backup'):
                backup.unlink()
        pause()
storage._replace_bytes = writing
write_json = storage.write_json
def writing_manifest(path, value):
    if phase == 'before_commit' and value.get('state') == 'committed':
        pause()
    write_json(path, value)
storage.write_json = writing_manifest
discard = storage._discard_transaction
def discarding(transaction):
    if phase == 'committed':
        pause()
    discard(transaction)
storage._discard_transaction = discarding
rmtree = storage.shutil.rmtree
def removing(directory):
    if phase == 'cleanup':
        # manifest 已移除，即使备份只删除一部分，也不能再执行回滚。
        for backup in Path(directory).glob('*.backup'):
            backup.unlink()
            break
        pause()
    rmtree(directory)
storage.shutil.rmtree = removing
with storage.video_lock(paths, video_id):
    storage.commit_artifacts(paths, video_id,
        json_files={segments:{'version':2}, cleaned:{'version':2}},
        text_files={paths.artifact(video_id,'markdown'):'version 2'})
'''


class WorkerRecoveryTest(unittest.TestCase):
    def wait_until(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("等待测试 worker 状态超时")

    def run_cancelled_worker(self, root, phase):
        paths = PipelinePaths("test", root / "data", root / "history", root / "work")
        write_json(paths.artifact(VIDEO_ID, "raw"), {"id": VIDEO_ID})
        write_json(paths.artifact(VIDEO_ID, "cleaned"), {"version": 1})
        manager = JobManager(paths, root / "config.json")
        recovering, release = threading.Event(), threading.Event()
        original_recovery = storage.recover_video_transactions

        def paused_recovery(*args):
            if (root / "ready").exists():
                recovering.set()
                if not release.wait(5):
                    raise TimeoutError("测试未释放恢复操作")
            return original_recovery(*args)

        command = [sys.executable, "-c", WORKER_SCRIPT, str(root), phase]
        with patch.object(manager, "_command", return_value=command), patch.object(
            jobs, "recover_video_transactions", side_effect=paused_recovery,
        ):
            job = manager.create("segments", [VIDEO_ID])
            try:
                self.wait_until(lambda: (root / "ready").exists())
                manager.cancel(job["id"])
                self.assertTrue(recovering.wait(5))
                self.assertTrue(manager.is_video_busy(VIDEO_ID))
                self.assertEqual(manager.snapshot(job["id"])["status"], "cancelling")
            finally:
                release.set()
                if manager.snapshot(job["id"])["can_cancel"]:
                    manager.cancel(job["id"])
                self.wait_until(lambda: not manager.snapshot(job["id"])["can_cancel"])
        return paths, manager.snapshot(job["id"])

    def test_forced_cancel_recovers_before_unlock_and_restart_keeps_new_save(self):
        for phase in ["writing", "before_commit", "committed", "cleanup"]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory(prefix="pipeline-worker-") as directory:
                paths, job = self.run_cancelled_worker(Path(directory), phase)
                self.assertEqual(job["status"], "cancelled", job)
                segments = paths.artifact(VIDEO_ID, "segments")
                cleaned = paths.artifact(VIDEO_ID, "cleaned")
                if phase in {"writing", "before_commit"}:
                    self.assertFalse(segments.exists())
                    self.assertFalse(paths.artifact(VIDEO_ID, "markdown").exists())
                    self.assertEqual(load_json(cleaned), {"version": 1})
                else:
                    self.assertEqual(load_json(segments), {"version": 2})
                    self.assertEqual(load_json(cleaned), {"version": 2})
                with storage.video_lock(paths, VIDEO_ID):
                    storage.commit_artifacts(paths, VIDEO_ID, json_files={segments: {"version": 3}})
                self.assertEqual(storage.recover_transactions(paths), 0)
                self.assertEqual(load_json(segments), {"version": 3})

    def test_failed_recovery_is_reported_and_blocks_further_changes(self):
        with tempfile.TemporaryDirectory(prefix="pipeline-worker-") as directory:
            paths, job = self.run_cancelled_worker(Path(directory), "broken_backup")
            self.assertEqual(job["status"], "failed", job)
            self.assertIn("数据恢复失败", job["items"][0]["log"])
            self.assertEqual(load_json(paths.artifact(VIDEO_ID, "segments")), {"version": 2})
            with self.assertRaises(storage.TransactionRecoveryError):
                with storage.video_lock(paths, VIDEO_ID):
                    self.fail("恢复失败后不应开放写入")
            with self.assertRaises(storage.TransactionRecoveryError):
                delete_video(paths, VIDEO_ID)

    def test_next_writer_recovers_legacy_journal_before_reading(self):
        with tempfile.TemporaryDirectory(prefix="pipeline-worker-") as directory:
            root = Path(directory)
            paths = PipelinePaths("test", root / "data", root / "history", root / "work")
            target = paths.artifact(VIDEO_ID, "segments")
            transaction = paths.work_root / "transactions" / "test" / VIDEO_ID / "legacy"
            write_json(transaction / "manifest.json", {"entries": [
                {"path": str(target), "backup": "0.backup", "existed": False},
            ]})
            write_json(target, {"version": "partial"})
            with storage.video_lock(paths, VIDEO_ID):
                self.assertFalse(target.exists())
                storage.commit_artifacts(paths, VIDEO_ID, json_files={target: {"version": "new"}})
            self.assertEqual(storage.recover_transactions(paths), 0)
            self.assertEqual(load_json(target), {"version": "new"})

    def test_failed_commit_marker_rolls_back_all_files(self):
        with tempfile.TemporaryDirectory(prefix="pipeline-worker-") as directory:
            root = Path(directory)
            paths = PipelinePaths("test", root / "data", root / "history", root / "work")
            target = paths.artifact(VIDEO_ID, "cleaned")
            new_file = paths.artifact(VIDEO_ID, "segments")
            write_json(target, {"version": 1})

            def fail_marker(path, value):
                if value.get("state") == "committed":
                    raise OSError("模拟完成标记写入失败")
                write_json(path, value)

            with storage.video_lock(paths, VIDEO_ID), patch.object(storage, "write_json", side_effect=fail_marker):
                with self.assertRaises(OSError):
                    storage.commit_artifacts(paths, VIDEO_ID, json_files={target: {"version": 2}, new_file: {"version": 2}})
            self.assertEqual(load_json(target), {"version": 1})
            self.assertFalse(new_file.exists())
            self.assertEqual(storage.recover_transactions(paths), 0)

    def test_interrupted_rollback_keeps_backups_and_can_be_retried(self):
        with tempfile.TemporaryDirectory(prefix="pipeline-worker-") as directory:
            root = Path(directory)
            paths = PipelinePaths("test", root / "data", root / "history", root / "work")
            targets = [paths.artifact(VIDEO_ID, kind) for kind in ["cleaned", "segments"]]
            transaction = paths.work_root / "transactions" / "test" / VIDEO_ID / "interrupted"
            entries = []
            for index, target in enumerate(targets):
                write_json(transaction / f"{index}.backup", {"version": 1})
                write_json(target, {"version": 2})
                entries.append({"path": str(target), "backup": f"{index}.backup", "existed": True})
            write_json(transaction / "manifest.json", {"entries": entries})
            original_replace = storage._replace_bytes

            def fail_second(path, content):
                if path == targets[1]:
                    raise OSError("模拟恢复中断")
                original_replace(path, content)

            with patch.object(storage, "_replace_bytes", side_effect=fail_second):
                with self.assertRaises(storage.TransactionRecoveryError):
                    storage.recover_transactions(paths)
            self.assertTrue((transaction / "manifest.json").exists())
            self.assertEqual(storage.recover_transactions(paths), 1)
            self.assertEqual([load_json(path) for path in targets], [{"version": 1}, {"version": 1}])
            self.assertFalse(transaction.exists())


if __name__ == "__main__":
    unittest.main()
