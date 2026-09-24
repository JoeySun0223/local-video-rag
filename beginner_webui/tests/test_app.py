from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from beginner_webui.app import create_app
from video_pipeline.services.editing import _redistribute_chapter_content, save_cleaned_sentence
from video_pipeline.services.glossary import changed_term_candidates
from video_pipeline.services.jobs import JobManager
from video_pipeline.services.catalog import PipelinePaths
from video_pipeline.shared.io import load_json, write_json


VIDEO_ID = "d" * 64


class BeginnerWebUiTest(unittest.TestCase):
    def make_project(self, root: Path) -> tuple[Path, Path]:
        config = root / "config.json"
        write_json(config, {
            "partition": "test", "data_root": "data",
            "history_root": "history", "work_root": "work", "ffmpeg": "ffmpeg",
            "glossary": "resources/glossary.json",
        })
        write_json(root / "resources" / "glossary.json", {"global": {}, "sources": {}})
        video = root / "data" / "videos" / "test" / f"demo__{VIDEO_ID[:16]}.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        write_json(root / "data" / "asr_raw" / "test" / f"{VIDEO_ID}.json", {
            "id": VIDEO_ID, "title": "演示视频", "duration_ms": 3000,
            "video": f"videos/test/{video.name}",
            "sentences": [{"id": 1, "start_ms": 0, "end_ms": 3000, "raw_text": "原文。"}],
        })
        write_json(root / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json", {
            "source_id": VIDEO_ID, "title": "演示视频", "sentences": [
                {"sentence_id": 1, "start_ms": 0, "end_ms": 3000, "text": "原文。"},
            ], "full_text": "原文。",
        })
        segments = root / "data" / "semantic_segments" / "test" / f"{VIDEO_ID}.json"
        write_json(segments, {
            "video_id": VIDEO_ID, "title": "演示视频", "video_path": f"videos/test/{video.name}",
            "segments": [{
                "segment_no": 1, "title": "演示章节", "summary": "演示摘要。", "keywords": ["演示"],
                "start_sentence_id": 1, "end_sentence_id": 1, "start_ms": 0, "end_ms": 3000,
                "content": "第一句。\n第二句。",
            }],
        })
        write_json(root / "history" / "cleanup" / "test" / f"{VIDEO_ID}.json", {
            "source_id": VIDEO_ID, "title": "演示视频", "status": "completed",
            "counts": {"auto_approved": 1},
            "changes": [{
                "sentence_id": 1, "raw_text": "原文。", "proposed_text": "建议原文。",
                "effective_text": "原文。", "status": "pending_review",
                "decision": "auto_approved", "review_reason": "需要人工确认术语。",
            }],
        })
        return config, segments

    def test_new_ui_is_served_from_separate_package(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.make_project(Path(directory))
            with TestClient(create_app(config)) as client:
                html = client.get("/").text
                stylesheet = client.get("/static/style.css").text
                javascript = client.get("/static/app.js").text
                detail_page = client.get(f"/videos/{VIDEO_ID}")
                chapter = client.get(f"/api/videos/{VIDEO_ID}/segments/1/markdown").json()
            self.assertIn("视频数据处理", html)
            self.assertIn('id="processButton"', html)
            self.assertIn('id="chapterProgress"', html)
            self.assertIn('id="videoSeek"', html)
            self.assertIn('id="chapterTimeDisplay"', html)
            self.assertIn('id="chapterTitleInput"', html)
            self.assertIn('id="chapterKeywordsInput"', html)
            self.assertIn('id="chapterSummaryInput"', html)
            self.assertIn('id="chapterContentInput"', html)
            self.assertIn('id="playChapterButton"', html)
            self.assertIn('id="unsavedEditModal"', html)
            self.assertIn('id="saveAndReturnButton"', html)
            self.assertIn('内容已修改，是否保存？', html)
            self.assertIn('id="editorVideoPanel" class="editor-video-panel hidden"', html)
            self.assertIn('id="editorVideoCloseButton"', html)
            self.assertNotIn('id="editorChapterStartButton"', html)
            self.assertNotIn('id="editorLoopButton"', html)
            self.assertNotIn('id="markdownEditor"', html)
            self.assertIn('id="apiConnectionButton"', html)
            self.assertIn('id="apiDialog"', html)
            self.assertIn('id="taskPanel"', html)
            self.assertIn('id="cancelTaskButton"', html)
            self.assertNotIn('class="panel jobs-panel"', html)
            self.assertNotIn("已忽略纯删除内容", html)
            self.assertNotIn("一键全选", html)
            self.assertNotIn("业务深标", html)
            self.assertNotIn("格式由系统自动维护", html)
            self.assertNotIn("实时预览", html)
            self.assertIn(".editor-pane, .preview-pane { min-width: 0; min-height: 0;", stylesheet)
            self.assertIn("overflow-y: auto; overscroll-behavior: contain; scrollbar-gutter: stable;", stylesheet)
            self.assertIn("function pauseDetailVideo()", javascript)
            self.assertIn('if (id !== "videoView") pauseDetailVideo();', javascript)
            self.assertIn('text: "自定义"', javascript)
            self.assertEqual(detail_page.status_code, 200)
            self.assertIn("第一句。第二句。", chapter["markdown"])
            self.assertNotIn("第一句。\n第二句。", chapter["markdown"])

    def test_dashboard_places_unprocessed_upload_before_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.make_project(root)
            waiting_id = "e" * 64
            waiting_video = root / "data" / "videos" / "test" / f"待处理__{waiting_id[:16]}.mp4"
            waiting_video.write_bytes(b"waiting-video")
            write_json(root / "work" / "uploads" / "test" / waiting_id / "upload.json", {
                "id": waiting_id,
                "filename": "待处理视频",
                "path": str(waiting_video),
                "size": waiting_video.stat().st_size,
                "uploaded_at": "2026-09-16T12:00:00+08:00",
            })
            with TestClient(create_app(config)) as client:
                records = client.get("/api/dashboard").json()["records"]
            self.assertEqual(records[0]["id"], waiting_id)
            self.assertFalse(records[0]["has_raw"])

    def test_queued_job_can_be_cancelled_before_it_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.make_project(root)
            paths = PipelinePaths.from_config({
                **json.loads(config.read_text(encoding="utf-8")),
                "_root": str(root),
            })
            manager = JobManager(paths, config)
            with manager._condition:
                created = manager.create("pipeline", [VIDEO_ID])
                cancelled = manager.cancel(created["id"])
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(cancelled["items"][0]["status"], "cancelled")
            self.assertFalse(cancelled["can_cancel"])

    def test_cancel_job_endpoint_calls_job_manager(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.make_project(Path(directory))
            with TestClient(create_app(config)) as client:
                with patch.object(client.app.state.jobs, "cancel", return_value={
                    "id": "job-1", "status": "cancelled", "items": [],
                }) as cancel:
                    response = client.post("/api/jobs/job-1/cancel")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["status"], "cancelled")
            cancel.assert_called_once_with("job-1")

    def test_markdown_edit_and_confirmation_update_segment_json(self):
        with tempfile.TemporaryDirectory() as directory:
            config, segments = self.make_project(Path(directory))
            markdown = """## 人工章节

> 时间：0:00–0:03
> 关键词：人工、编辑

### 章节摘要

人工摘要。

### 章节正文

人工正文。
"""
            with TestClient(create_app(config)) as client:
                before = client.get("/api/dashboard").json()["counts"]
                edited = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1/markdown",
                    json={"markdown": markdown, "expected_revision": 0},
                )
                confirmed = client.post(f"/api/videos/{VIDEO_ID}/confirm")
                after = client.get("/api/dashboard").json()["counts"]
            self.assertEqual(before["pending_review"], 1)
            self.assertEqual(edited.status_code, 200, edited.text)
            self.assertEqual(confirmed.status_code, 200, confirmed.text)
            self.assertEqual(after["completed"], 1)
            document = load_json(segments)
            cleaned = load_json(
                Path(directory) / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json"
            )
            self.assertEqual(document["segments"][0]["title"], "人工章节")
            self.assertEqual(document["segments"][0]["content"], "人工正文。")
            self.assertEqual(document["review_status"], "confirmed")
            self.assertEqual(cleaned["sentences"][0]["text"], "人工正文。")
            self.assertEqual(cleaned["full_text"], "人工正文。")

    def test_chapter_content_is_redistributed_without_changing_timeline(self):
        cleaned = {
            "sentences": [
                {"sentence_id": 1, "start_ms": 0, "end_ms": 1000, "text": "下面请兔南海介绍。"},
                {"sentence_id": 2, "start_ms": 1000, "end_ms": 2000, "text": "第二句保持。"},
            ],
            "full_text": "下面请兔南海介绍。\n第二句保持。",
        }
        _redistribute_chapter_content(cleaned, 1, 2, "下面请通达海介绍。第二句保持。")
        self.assertEqual(cleaned["sentences"][0]["text"], "下面请通达海介绍。")
        self.assertEqual(cleaned["sentences"][1]["text"], "第二句保持。")
        self.assertEqual(cleaned["sentences"][1]["start_ms"], 1000)

    def test_empty_cleaned_sentence_keeps_chapter_time_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, segment_path = self.make_project(root)
            raw_path = root / "data" / "asr_raw" / "test" / f"{VIDEO_ID}.json"
            cleaned_path = root / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json"
            raw = load_json(raw_path)
            raw["sentences"] = [
                {"id": 1, "start_ms": 0, "end_ms": 1000, "raw_text": "原文。"},
                {"id": 2, "start_ms": 1000, "end_ms": 3000, "raw_text": "第二句。"},
            ]
            cleaned = load_json(cleaned_path)
            cleaned["sentences"] = [
                {"sentence_id": 1, "start_ms": 0, "end_ms": 1000, "text": "原文。"},
                {"sentence_id": 2, "start_ms": 1000, "end_ms": 3000, "text": "第二句。"},
            ]
            cleaned["full_text"] = "原文。\n第二句。"
            document = load_json(segment_path)
            document["segments"][0].update(end_sentence_id=2, content="原文。第二句。")
            write_json(raw_path, raw)
            write_json(cleaned_path, cleaned)
            write_json(segment_path, document)
            paths = PipelinePaths.from_config({
                "partition": "test", "data_root": "data", "history_root": "history",
                "work_root": "work", "_root": str(root),
            })
            save_cleaned_sentence(paths, VIDEO_ID, 1, "")
            saved = load_json(segment_path)
            self.assertEqual(saved["segments"][0]["start_sentence_id"], 1)
            self.assertEqual(saved["segments"][0]["start_ms"], 0)
            self.assertEqual(saved["segments"][0]["content"], "第二句。")

    def test_stale_markdown_save_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, segments = self.make_project(Path(directory))
            document = load_json(segments)
            document["updated_at"] = "2026-09-16T12:05:00+08:00"
            write_json(segments, document)
            markdown = """## 陈旧章节

> 时间：0:00–0:03
> 关键词：陈旧

### 章节摘要

陈旧摘要。

### 章节正文

最后保存的新正文。
"""
            with TestClient(create_app(config)) as client:
                response = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1/markdown",
                    json={
                        "markdown": markdown,
                        "expected_updated_at": "2026-09-16T12:00:00+08:00",
                        "expected_revision": 0,
                    },
                )
            self.assertEqual(response.status_code, 409, response.text)
            saved = load_json(segments)
            self.assertNotEqual(saved["segments"][0]["title"], "陈旧章节")
            self.assertNotEqual(saved["segments"][0]["content"], "最后保存的新正文。")

    def test_cached_old_page_can_save_once_without_revision_then_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            config, segments = self.make_project(Path(directory))
            markdown = """## 旧页面编辑

> 时间：0:00–0:03
> 关键词：编辑

### 章节摘要

旧页面摘要。

### 章节正文

旧页面正文。
"""
            with TestClient(create_app(config)) as client:
                first = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1/markdown",
                    json={"markdown": markdown, "expected_updated_at": ""},
                )
                second = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1/markdown",
                    json={"markdown": markdown.replace("旧页面", "陈旧页面"),
                          "expected_updated_at": ""},
                )
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(second.status_code, 409, second.text)
            self.assertEqual(load_json(segments)["segments"][0]["title"], "旧页面编辑")

    def test_cached_old_page_timestamp_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            config, segments = self.make_project(Path(directory))
            document = load_json(segments)
            document["updated_at"] = "2026-09-16T12:00:00+08:00"
            write_json(segments, document)
            markdown = """## 时间戳编辑

> 时间：0:00–0:03
> 关键词：编辑

### 章节摘要

时间戳摘要。

### 章节正文

时间戳正文。
"""
            with TestClient(create_app(config)) as client:
                first = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1/markdown",
                    json={"markdown": markdown,
                          "expected_updated_at": "2026-09-16T12:00:00+08:00"},
                )
                second = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1/markdown",
                    json={"markdown": markdown.replace("时间戳", "陈旧"),
                          "expected_updated_at": "2026-09-16T12:00:00+08:00"},
                )
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(second.status_code, 409, second.text)

    def test_structured_chapter_save_syncs_timed_sentences_and_rejects_stale_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, segments = self.make_project(root)
            cleaned_path = root / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json"
            with TestClient(create_app(config)) as client:
                first = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1",
                    json={"title": "人工章节", "summary": "人工摘要。",
                          "content": "人工修改正文。", "keywords": ["人工"],
                          "expected_revision": 0},
                )
                stale = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1",
                    json={"title": "旧编辑", "summary": "旧摘要。",
                          "content": "旧正文。", "expected_revision": 0},
                )
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(stale.status_code, 409, stale.text)
            document = load_json(segments)
            cleaned = load_json(cleaned_path)
            self.assertEqual(document["revision"], 1)
            self.assertEqual(document["segments"][0]["content"], "人工修改正文。")
            self.assertEqual(cleaned["sentences"][0]["text"], "人工修改正文。")
            self.assertEqual(cleaned["sentences"][0]["start_ms"], 0)
            markdown = (root / "data" / "semantic_markdown" / "test" / f"{VIDEO_ID}.md").read_text(encoding="utf-8")
            self.assertIn("人工修改正文。", markdown)

    def test_chapter_save_keeps_english_word_when_sentence_boundary_splits_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, segment_path = self.make_project(root)
            raw_path = root / "data" / "asr_raw" / "test" / f"{VIDEO_ID}.json"
            cleaned_path = root / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json"
            raw = load_json(raw_path)
            raw["sentences"] = [
                {"id": 1, "start_ms": 0, "end_ms": 1000, "raw_text": "测试 AB"},
                {"id": 2, "start_ms": 1000, "end_ms": 3000, "raw_text": "C 中文旧。"},
            ]
            cleaned = load_json(cleaned_path)
            cleaned["sentences"] = [
                {"sentence_id": 1, "start_ms": 0, "end_ms": 1000, "text": "测试 AB"},
                {"sentence_id": 2, "start_ms": 1000, "end_ms": 3000, "text": "C 中文旧。"},
            ]
            cleaned["full_text"] = "测试 AB\nC 中文旧。"
            document = load_json(segment_path)
            document["segments"][0].update(
                end_sentence_id=2, content="测试 AB C 中文旧。",
            )
            write_json(raw_path, raw)
            write_json(cleaned_path, cleaned)
            write_json(segment_path, document)
            with TestClient(create_app(config)) as client:
                response = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1",
                    json={"title": "测试章节", "summary": "测试摘要。",
                          "content": "测试 ABC 中文 def。", "keywords": ["测试"],
                          "expected_revision": 0},
                )
            self.assertEqual(response.status_code, 200, response.text)
            saved_cleaned = load_json(cleaned_path)
            saved_document = load_json(segment_path)
            self.assertEqual(
                "".join(item["text"] for item in saved_cleaned["sentences"]),
                "测试 ABC 中文 def。",
            )
            self.assertEqual(saved_document["segments"][0]["content"], "测试 ABC 中文 def。")
            self.assertEqual(saved_cleaned["sentences"][1]["start_ms"], 1000)

            # 正文允许与原 ASR 完全不同；保存与确认都只依赖时间锚点和同步结果。
            with TestClient(create_app(config)) as client:
                response = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1",
                    json={"title": "新标题", "summary": "新摘要。",
                          "content": "这是完全重写的新章节，与原语音内容无关。",
                          "keywords": [], "expected_revision": 1},
                )
                self.assertEqual(response.status_code, 200, response.text)
                confirmation = client.post(f"/api/videos/{VIDEO_ID}/confirm")
            self.assertEqual(confirmation.status_code, 200, confirmation.text)
            saved_cleaned = load_json(cleaned_path)
            self.assertEqual(
                "".join(item["text"] for item in saved_cleaned["sentences"]),
                "这是完全重写的新章节，与原语音内容无关。",
            )
            self.assertEqual(saved_cleaned["sentences"][1]["start_ms"], 1000)

    def test_confirm_accepts_all_remaining_provisional_suggestions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, segments = self.make_project(root)
            history_path = root / "history" / "cleanup" / "test" / f"{VIDEO_ID}.json"
            history = load_json(history_path)
            history["changes"][0]["decision"] = "pending"
            history["counts"] = {"pending_review": 1}
            write_json(history_path, history)
            document = load_json(segments)
            document["segments"][0]["content"] = "原文。"
            write_json(segments, document)
            with TestClient(create_app(config)) as client:
                response = client.post(f"/api/videos/{VIDEO_ID}/confirm")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(load_json(segments)["review_status"], "confirmed")
            self.assertEqual(load_json(history_path)["changes"][0]["decision"], "confirmed_by_video")
            self.assertEqual(load_json(history_path)["counts"]["pending_review"], 0)

    def test_stale_glossary_hits_return_conflict_without_changing_new_content(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.make_project(Path(directory))
            app = create_app(config)
            request = {"source_video_id": VIDEO_ID, "scope": "video", "terms": [
                {"term": "新词", "aliases": ["原文"], "scope": "global"},
            ]}
            with TestClient(app) as client:
                hits = client.post("/api/glossary/search", json=request).json()["hits"]
                save_cleaned_sentence(app.state.paths, VIDEO_ID, 1, "原文已经更新。")
                response = client.post("/api/glossary/apply", json={
                    **request, "selected_hit_ids": [hit["hit_id"] for hit in hits],
                })
            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("重新搜索", response.json()["detail"])
            self.assertEqual(load_json(app.state.paths.artifact(VIDEO_ID, "cleaned"))["full_text"], "原文已经更新。")

    def test_unrecoverable_transaction_blocks_edit_confirm_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, segments = self.make_project(root)
            app = create_app(config)
            original = segments.read_bytes()
            transaction = root / "work" / "transactions" / "test" / VIDEO_ID / "interrupted"
            # 模拟 worker 中断且备份不可读；服务已启动，必须在后续写入时阻止。
            write_json(transaction / "manifest.json", {"entries": [
                {"path": str(segments), "backup": "0.backup", "existed": True},
            ]})
            with TestClient(app) as client:
                responses = [
                    client.put(f"/api/videos/{VIDEO_ID}/segments/1", json={
                        "title": "修改", "summary": "摘要。", "content": "新正文。", "expected_revision": 0,
                    }),
                    client.post(f"/api/videos/{VIDEO_ID}/confirm"),
                    client.delete(f"/api/videos/{VIDEO_ID}"),
                ]
            for response in responses:
                self.assertEqual(response.status_code, 409, response.text)
                self.assertIn("数据恢复失败", response.json()["detail"])
            self.assertEqual(segments.read_bytes(), original)

    def test_delete_removes_the_selected_video_record(self):
        with tempfile.TemporaryDirectory() as directory:
            config, segments = self.make_project(Path(directory))
            with TestClient(create_app(config)) as client:
                response = client.delete(f"/api/videos/{VIDEO_ID}")
                dashboard = client.get("/api/dashboard").json()
            self.assertEqual(response.status_code, 200, response.text)
            self.assertGreater(response.json()["removed"], 0)
            self.assertFalse(segments.exists())
            self.assertEqual(dashboard["counts"]["videos"], 0)

    def test_archived_ai_suggestion_can_be_rejected_and_synchronized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, segments = self.make_project(root)
            with TestClient(create_app(config)) as client:
                response = client.post(
                    f"/api/videos/{VIDEO_ID}/suggestions/1",
                    json={"decision": "rejected", "approved_text": None},
                )
            self.assertEqual(response.status_code, 200, response.text)
            history = load_json(root / "history" / "cleanup" / "test" / f"{VIDEO_ID}.json")
            cleaned = load_json(root / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json")
            document = load_json(segments)
            self.assertEqual(history["changes"][0]["decision"], "rejected")
            self.assertEqual(cleaned["sentences"][0]["text"], "原文。")
            self.assertEqual(document["segments"][0]["content"], "原文。")

    def test_reject_rebuilds_content_even_after_manual_chapter_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, segments = self.make_project(root)
            cleaned_path = root / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json"
            history_path = root / "history" / "cleanup" / "test" / f"{VIDEO_ID}.json"
            cleaned = load_json(cleaned_path)
            cleaned["sentences"][0]["text"] = "建议原文。"
            cleaned["full_text"] = "建议原文。"
            write_json(cleaned_path, cleaned)
            document = load_json(segments)
            document["segments"][0]["content"] = "建议原文。"
            document["segments"][0]["manual_content_override"] = True
            write_json(segments, document)
            history = load_json(history_path)
            history["changes"][0]["effective_text"] = "建议原文。"
            write_json(history_path, history)
            with TestClient(create_app(config)) as client:
                response = client.post(
                    f"/api/videos/{VIDEO_ID}/suggestions/1",
                    json={"decision": "rejected", "approved_text": None},
                )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(load_json(cleaned_path)["sentences"][0]["text"], "原文。")
            self.assertEqual(load_json(segments)["segments"][0]["content"], "原文。")

    def test_suggestion_accepts_custom_combined_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, segments = self.make_project(root)
            with TestClient(create_app(config)) as client:
                response = client.post(
                    f"/api/videos/{VIDEO_ID}/suggestions/1",
                    json={"decision": "approved", "approved_text": "原文与建议结合。"},
                )
            self.assertEqual(response.status_code, 200, response.text)
            history = load_json(root / "history" / "cleanup" / "test" / f"{VIDEO_ID}.json")
            cleaned = load_json(root / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json")
            self.assertEqual(history["changes"][0]["approved_text"], "原文与建议结合。")
            self.assertTrue(history["changes"][0]["glossary_review_pending"])
            self.assertEqual(cleaned["sentences"][0]["text"], "原文与建议结合。")
            self.assertEqual(load_json(segments)["segments"][0]["content"], "原文与建议结合。")

    def test_custom_suggestion_is_offered_once_for_glossary_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.make_project(root)
            with TestClient(create_app(config)) as client:
                custom = client.post(
                    f"/api/videos/{VIDEO_ID}/suggestions/1",
                    json={"decision": "approved", "approved_text": "审管办。"},
                )
                self.assertEqual(custom.status_code, 200, custom.text)
                chapter = client.get(
                    f"/api/videos/{VIDEO_ID}/segments/1/markdown"
                ).json()["markdown"]
                detected = client.post(
                    f"/api/videos/{VIDEO_ID}/segments/1/glossary-candidates",
                    json={"markdown": chapter},
                )
                self.assertEqual(detected.status_code, 200, detected.text)
                self.assertEqual(detected.json()["candidates"][0]["term"], "审管办")
                self.assertEqual(detected.json()["candidates"][0]["aliases"], ["建议原文"])
                saved = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1/markdown",
                    json={"markdown": chapter, "glossary_terms": [],
                          "expected_revision": custom.json()["artifacts"]["segments"]["revision"]},
                )
                self.assertEqual(saved.status_code, 200, saved.text)
                after = client.post(
                    f"/api/videos/{VIDEO_ID}/segments/1/glossary-candidates",
                    json={"markdown": chapter},
                )
            self.assertEqual(after.status_code, 200, after.text)
            self.assertEqual(after.json()["candidates"], [])
            history = load_json(root / "history" / "cleanup" / "test" / f"{VIDEO_ID}.json")
            self.assertFalse(history["changes"][0]["glossary_review_pending"])

    def test_generate_metadata_uses_current_markdown_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            config, segments = self.make_project(Path(directory))
            before = segments.read_text(encoding="utf-8")
            generated = {
                "label": {"title": "新标题", "summary": "新摘要。", "keywords": ["新关键词"]},
                "model": "test-model", "usage": {"total_tokens": 10},
            }
            with patch("beginner_webui.app.generate_chapter_metadata", return_value=generated) as mocked:
                with TestClient(create_app(config)) as client:
                    response = client.post(
                        f"/api/videos/{VIDEO_ID}/segments/1/generate-metadata",
                        json={"markdown": "## 旧标题\n\n### 章节摘要\n\n旧摘要。\n\n### 章节正文\n\n用户刚修改的正文。\n"},
                    )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json(), generated)
            self.assertIn("用户刚修改的正文", mocked.call_args.args[3])
            self.assertEqual(segments.read_text(encoding="utf-8"), before)

    def test_glossary_candidates_ignore_deletion_and_save_selected_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.make_project(root)
            replacement = """## 演示章节

> 时间：0:00–0:03
> 关键词：演示

### 章节摘要

演示摘要。

### 章节正文

第一句。专业术语。
"""
            deletion = replacement.replace("第一句。专业术语。", "第一句。")
            with TestClient(create_app(config)) as client:
                candidates = client.post(
                    f"/api/videos/{VIDEO_ID}/segments/1/glossary-candidates",
                    json={"markdown": replacement},
                )
                deleted = client.post(
                    f"/api/videos/{VIDEO_ID}/segments/1/glossary-candidates",
                    json={"markdown": deletion},
                )
                saved = client.put(
                    f"/api/videos/{VIDEO_ID}/segments/1/markdown",
                    json={
                        "markdown": replacement,
                        "expected_revision": 0,
                        "glossary_terms": [{
                            "term": "专业术语", "aliases": ["第二句"], "scope": "video",
                        }],
                    },
                )
            self.assertEqual(candidates.status_code, 200, candidates.text)
            self.assertTrue(candidates.json()["candidates"])
            self.assertEqual(deleted.json()["candidates"], [])
            self.assertEqual(saved.status_code, 200, saved.text)
            glossary = load_json(root / "resources" / "glossary.json")
            self.assertEqual(glossary["sources"][VIDEO_ID]["专业术语"], ["第二句"])

    def test_glossary_candidate_does_not_swallow_following_person_name(self):
        candidates = changed_term_candidates("兔南海沈文婷", "通达海沈文婷")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["original"], "兔南海")
        self.assertEqual(candidates[0]["term"], "通达海")

    def test_glossary_candidate_keeps_changed_latin_term_whole(self):
        candidates = changed_term_candidates("专网的COCO群", "专网的CoCall群")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["original"], "COCO")
        self.assertEqual(candidates[0]["term"], "CoCall")

    def test_glossary_search_applies_only_selected_sentence_and_syncs_derivatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, segments_path = self.make_project(root)
            write_json(root / "data" / "asr_raw" / "test" / f"{VIDEO_ID}.json", {
                "id": VIDEO_ID, "title": "演示视频", "duration_ms": 3000,
                "sentences": [
                    {"id": 1, "start_ms": 0, "end_ms": 1500, "raw_text": "由省管办负责，办理事项。"},
                    {"id": 2, "start_ms": 1500, "end_ms": 3000, "raw_text": "省管办复核。"},
                ],
            })
            write_json(root / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json", {
                "source_id": VIDEO_ID, "title": "演示视频", "sentences": [
                    {"sentence_id": 1, "start_ms": 0, "end_ms": 1500, "text": "由省管办负责，办理事项。"},
                    {"sentence_id": 2, "start_ms": 1500, "end_ms": 3000, "text": "省管办复核。"},
                ], "full_text": "由省管办负责，办理事项。\n省管办复核。",
            })
            document = load_json(segments_path)
            document["segments"][0].update({
                "start_sentence_id": 1, "end_sentence_id": 2,
                "content": "由省管办负责，办理事项。\n省管办复核。",
            })
            write_json(segments_path, document)
            request = {
                "source_video_id": VIDEO_ID,
                "terms": [{"term": "审管办", "aliases": ["省管办"], "scope": "global"}],
                "scope": "video",
            }
            with TestClient(create_app(config)) as client:
                searched = client.post("/api/glossary/search", json=request)
                self.assertEqual(searched.status_code, 200, searched.text)
                hits = searched.json()["hits"]
                self.assertEqual([row["snippet"] for row in hits], ["由省管办负责", "省管办复核"])
                applied = client.post("/api/glossary/apply", json={
                    **request, "selected_hit_ids": [hits[0]["hit_id"]],
                })
            self.assertEqual(applied.status_code, 200, applied.text)
            self.assertEqual(applied.json()["changed_sentences"], 1)
            cleaned = load_json(root / "data" / "cleaned_asr" / "test" / f"{VIDEO_ID}.json")
            self.assertEqual(cleaned["sentences"][0]["text"], "由审管办负责，办理事项。")
            self.assertEqual(cleaned["sentences"][1]["text"], "省管办复核。")
            segments = load_json(segments_path)
            self.assertIn("由审管办负责", segments["segments"][0]["content"])
            self.assertIn("省管办复核", segments["segments"][0]["content"])
            markdown = root / "data" / "semantic_markdown" / "test" / f"{VIDEO_ID}.md"
            self.assertIn("由审管办负责", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
