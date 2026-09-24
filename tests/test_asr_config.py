import unittest
import tempfile
from pathlib import Path

from video_pipeline.asr.cli import _configured_hotwords
from video_pipeline.shared.io import write_json


class AsrConfigTest(unittest.TestCase):
    def test_source_glossary_is_loaded_by_video_hash(self):
        video_id = "35f8998394899825be799893ed1435410ee6d0b6b6b59ec9ff9210b52c5b405f"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "resources" / "glossary.json", {
                "global": {"繁简识别": [], "通达海": []},
                "sources": {video_id: {"呈批": [], "管案": []}},
            })
            config = {"_root": str(root), "glossary": "resources/glossary.json"}
            self.assertEqual(
                ["临时词", "繁简识别", "通达海", "呈批", "管案"],
                _configured_hotwords(config, video_id, ["临时词"]),
            )


if __name__ == "__main__":
    unittest.main()
