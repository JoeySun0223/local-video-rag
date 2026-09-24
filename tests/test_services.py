"""检查服务写入失败和重启后的可恢复行为。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_pipeline.services.catalog import PipelinePaths
from video_pipeline.cleanup.cli import _auto_publish
from video_pipeline.services.jobs import JobManager
from video_pipeline.services.storage import commit_artifacts, recover_transactions, video_lock
from video_pipeline.shared.io import load_json, write_json
from video_pipeline.shared.protection import assert_can_overwrite


VIDEO_ID = "a" * 64


class ServiceSafetyTest(unittest.TestCase):
    def make_paths(self, root: Path) -> PipelinePaths:
        return PipelinePaths("test", root / "data", root / "history", root / "work")

    def test_failed_multi_file_commit_restores_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(Path(directory))
            cleaned = paths.artifact(VIDEO_ID, "cleaned")
            segments = paths.artifact(VIDEO_ID, "segments")
            assert cleaned is not None and segments is not None
            write_json(cleaned, {"version": 1})
            write_json(segments, {"version": 1})
            from video_pipeline.services import storage
            original = storage._replace_bytes
            calls = 0

            def fail_second(path: Path, content: bytes) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("模拟第二个文件写入失败")
                original(path, content)

            with video_lock(paths, VIDEO_ID):
                with patch.object(storage, "_replace_bytes", side_effect=fail_second):
                    with self.assertRaises(OSError):
                        commit_artifacts(
                            paths, VIDEO_ID,
                            json_files={cleaned: {"version": 2}, segments: {"version": 2}},
                        )
            self.assertEqual(load_json(cleaned), {"version": 1})
            self.assertEqual(load_json(segments), {"version": 1})
            self.assertEqual(recover_transactions(paths), 0)

    def test_restart_marks_incomplete_job_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(Path(directory))
            job_id = "b" * 32
            job_path = paths.work_root / "jobs" / "test" / f"{job_id}.json"
            write_json(job_path, {
                "id": job_id, "status": "running", "items": [
                    {"video_id": VIDEO_ID, "status": "running"}
                ],
            })
            manager = JobManager(paths, Path(directory) / "config.yaml")
            job = manager.snapshot(job_id)
            self.assertEqual(job["status"], "interrupted")
            self.assertEqual(job["items"][0]["status"], "interrupted")
            self.assertEqual(load_json(job_path)["status"], "interrupted")

    def test_startup_recovers_interrupted_multi_file_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_paths(root)
            cleaned = paths.artifact(VIDEO_ID, "cleaned")
            segments = paths.artifact(VIDEO_ID, "segments")
            assert cleaned is not None and segments is not None
            write_json(cleaned, {"version": 1})
            write_json(segments, {"version": 1})
            transaction = paths.work_root / "transactions" / "test" / VIDEO_ID / "interrupted"
            transaction.mkdir(parents=True)
            (transaction / "0.backup").write_bytes(cleaned.read_bytes())
            (transaction / "1.backup").write_bytes(segments.read_bytes())
            write_json(transaction / "manifest.json", {"entries": [
                {"path": str(cleaned), "backup": "0.backup", "existed": True},
                {"path": str(segments), "backup": "1.backup", "existed": True},
            ]})
            write_json(cleaned, {"version": 2})
            self.assertEqual(recover_transactions(paths), 1)
            self.assertEqual(load_json(cleaned), {"version": 1})
            self.assertEqual(load_json(segments), {"version": 1})
            self.assertFalse(transaction.exists())

    def test_overwrite_protects_downstream_and_confirmed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_paths(Path(directory))
            cleaned = paths.artifact(VIDEO_ID, "cleaned")
            segments = paths.artifact(VIDEO_ID, "segments")
            assert cleaned is not None and segments is not None
            write_json(cleaned, {"source_id": VIDEO_ID})
            with self.assertRaises(RuntimeError):
                assert_can_overwrite("asr", paths.data_root, "test", VIDEO_ID)
            write_json(segments, {"review_status": "confirmed", "segments": []})
            with self.assertRaises(RuntimeError):
                assert_can_overwrite("segments", paths.data_root, "test", VIDEO_ID)
            with self.assertRaises(RuntimeError):
                assert_can_overwrite(
                    "cleanup", paths.data_root, "test", VIDEO_ID,
                    history_root=paths.history_root,
                )

    def test_media_paths_cannot_escape_project_video_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_paths(root)
            raw = paths.artifact(VIDEO_ID, "raw")
            assert raw is not None
            write_json(raw, {"video": str(root / "unrelated.mp4")})
            with self.assertRaises(ValueError):
                paths.artifact(VIDEO_ID, "video")
            write_json(paths.upload_root / VIDEO_ID / "upload.json", {
                "path": str(root / "unrelated.mp4")
            })
            with self.assertRaises(ValueError):
                paths.artifact(VIDEO_ID, "upload")

    def test_cleanup_candidate_is_provisional_until_video_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_paths(root)
            raw = {"id": VIDEO_ID, "sentences": [
                {"id": 1, "start_ms": 0, "end_ms": 1000, "raw_text": "原文。"}
            ]}
            cleaned = {"source_id": VIDEO_ID, "sentences": [
                {"sentence_id": 1, "start_ms": 0, "end_ms": 1000, "text": "原文。"}
            ], "full_text": "原文。"}
            report = {"counts": {"pending_review": 1}, "changes": [{
                "sentence_id": 1, "status": "pending_review", "raw_text": "原文。",
                "proposed_text": "建议。",
            }]}
            raw_path = paths.artifact(VIDEO_ID, "raw")
            output = paths.artifact(VIDEO_ID, "cleaned")
            archive = paths.artifact(VIDEO_ID, "history")
            assert raw_path is not None and output is not None and archive is not None
            write_json(raw_path, raw)
            _auto_publish(raw, cleaned, report, output, archive, paths=paths)
            self.assertEqual(load_json(output)["sentences"][0]["text"], "建议。")
            self.assertEqual(load_json(archive)["changes"][0]["decision"], "pending")
            self.assertEqual(load_json(archive)["counts"]["pending_review"], 1)


if __name__ == "__main__":
    unittest.main()
