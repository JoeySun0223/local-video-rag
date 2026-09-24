import unittest

from video_pipeline.validation.rules import validate_cleaned, validate_raw, validate_segments


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "id": "abc", "duration_ms": 2000,
            "sentences": [
                {"id": 1, "start_ms": 0, "end_ms": 900, "raw_text": "第一句。"},
                {"id": 2, "start_ms": 1000, "end_ms": 1900, "raw_text": "第二句。"},
            ],
        }
        self.cleaned = {
            "source_id": "abc",
            "sentences": [
                {"sentence_id": 1, "start_ms": 0, "end_ms": 900, "text": "第一句。"},
                {"sentence_id": 2, "start_ms": 1000, "end_ms": 1900, "text": "第二句。"},
            ],
            "full_text": "第一句。\n第二句。",
        }

    def test_valid_cross_stage_record(self):
        document = {
            "video_id": "abc",
            "segments": [{
                "segment_no": 1, "start_sentence_id": 1, "end_sentence_id": 2,
                "start_ms": 0, "end_ms": 1900, "content": "第一句。\n第二句。",
            }],
        }
        self.assertEqual([], validate_raw(self.raw, "abc"))
        self.assertEqual([], validate_cleaned(self.raw, self.cleaned))
        self.assertEqual([], validate_segments(self.cleaned, document))


if __name__ == "__main__":
    unittest.main()
