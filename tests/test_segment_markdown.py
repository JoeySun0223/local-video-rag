from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from video_pipeline.segments.markdown import segment_markdown, write_segment_markdown


class SegmentMarkdownTest(unittest.TestCase):
    def test_markdown_contains_complete_segment_metadata_and_content(self):
        document = {
            "video_id": "a" * 64,
            "title": "测试视频",
            "segments": [{
                "segment_no": 1,
                "title": "查询申请",
                "summary": "介绍查询申请。",
                "keywords": ["查询", "申请"],
                "start_sentence_id": 1,
                "end_sentence_id": 2,
                "start_ms": 5_000,
                "end_ms": 125_000,
                "content": "第一句。\n第二句。",
            }],
        }
        value = segment_markdown(document)
        self.assertIn("# 测试视频", value)
        self.assertIn("## 1. 查询申请", value)
        self.assertIn("> 时间：0:05–2:05", value)
        self.assertIn("> 关键词：查询、申请", value)
        self.assertIn("第一句。第二句。", value)
        self.assertNotIn("第一句。\n\n第二句。", value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.md"
            write_segment_markdown(path, document)
            self.assertEqual(path.read_text(encoding="utf-8"), value)


if __name__ == "__main__":
    unittest.main()
