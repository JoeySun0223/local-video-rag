from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from local_rag.config import load_config
from local_rag.webapp import create_app


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.config = copy.deepcopy(load_config())
        cls.config["project"]["data_dir"] = str(root / "data")
        cls.config["project"]["sources_dir"] = str(root / "sources")
        cls.client = TestClient(create_app(cls.config))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.temporary.cleanup()

    def test_home_and_management_apis(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/sources").json(), [])
        self.assertEqual(self.client.get("/api/corrections").json(), [])
        self.assertEqual(self.client.get("/api/query-logs").json(), [])

    def test_per_query_logging_control_and_refusal(self):
        response = self.client.post("/api/ask", json={"query": "不存在的问题", "log_query": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "当前知识库中没有足够信息。")
        self.assertEqual(self.client.get("/api/query-logs").json(), [])

        response = self.client.post("/api/ask", json={"query": "记录这个问题", "log_query": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.client.get("/api/query-logs").json()), 1)
        exported = self.client.get("/api/query-logs/export")
        self.assertIn("attachment", exported.headers["content-disposition"])
        self.assertEqual(self.client.delete("/api/query-logs").json()["deleted"], 1)

    def test_ingest_progress_is_persisted_with_storage_paths(self):
        class FakeIngestor:
            def __init__(self, config, db):
                self.config = config

            def ingest(self, path, category, title, sentences_json=None, use_llm=True, progress=None, cancelled=None, confirm_preflight=None, preflight_report=None):
                storage = [{
                    "key": "sentences", "label": "逐句ASR JSON",
                    "path": str(Path(self.config["project"]["data_dir"]) / "cache" / "sentences.json"),
                    "description": "测试路径",
                }]
                progress("asr", 25, "正在运行ASR", storage)
                progress("faiss", 90, "正在构建索引", storage)
                return {
                    "source_id": "source", "build_id": "build", "category": category, "title": title or path.stem,
                    "sentences": 3, "sections": 1, "chunks": 1, "index": {}, "storage": storage,
                }

        with patch("local_rag.webapp.Ingestor", FakeIngestor):
            response = self.client.post(
                "/api/ingest", data={"category": "测试", "title": "进度测试"},
                files={"video": ("sample.mp4", b"fake video", "video/mp4")},
            )
        self.assertEqual(response.status_code, 200)
        task_id = response.json()["task_id"]
        task = self.client.get(f"/api/tasks/{task_id}").json()
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["progress"], 100)
        self.assertTrue(any(item["key"] == "sentences" for item in task["storage"]))
        status_path = Path(self.config["project"]["data_dir"]) / "tasks" / task_id / "status.json"
        self.assertTrue(status_path.is_file())


if __name__ == "__main__":
    unittest.main()
