from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_pipeline.shared.cloud import (
    cloud_settings, cloud_status, connect_cloud, credentials_path, disconnect_cloud,
)
from video_pipeline.shared.io import load_json, write_json


class CloudCredentialsTest(unittest.TestCase):
    def make_config(self, root: Path) -> dict:
        path = root / "config.json"
        write_json(path, {
            "cloud": {
                "api_url": "https://example.test/chat/completions",
                "api_key_env": "TEST_PORTABLE_API_KEY",
                "model": "glm-5.3",
                "thinking": True,
                "timeout_seconds": 300,
            }
        })
        return {
            **load_json(path),
            "_root": str(root),
            "_config_path": str(path),
        }

    def test_successful_connection_is_saved_inside_project_and_masked(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(Path(directory))
            with patch(
                "video_pipeline.shared.cloud.test_cloud_connection",
                return_value={"latency_ms": 12, "usage": {"total_tokens": 2}},
            ):
                status = connect_cloud(
                    config,
                    api_url="https://example.test/chat/completions",
                    api_key="portable-secret-1234",
                    model="glm-5.3",
                )
            stored = load_json(credentials_path(config))
            self.assertEqual(stored["api_key"], "portable-secret-1234")
            self.assertTrue(status["persisted"])
            self.assertEqual(status["key_hint"], "••••1234")
            self.assertNotIn("api_key", status)

    def test_disconnect_blocks_environment_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(Path(directory))
            with patch.dict("os.environ", {"TEST_PORTABLE_API_KEY": "environment-secret"}):
                self.assertEqual(cloud_settings(config)["source"], "environment")
                status = disconnect_cloud(config)
                self.assertFalse(status["connected"])
                self.assertEqual(cloud_settings(config)["source"], "none")


if __name__ == "__main__":
    unittest.main()
