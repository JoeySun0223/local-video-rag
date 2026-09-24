import contextlib
import io
import unittest

from video_pipeline.shared.progress import emit_progress, parse_progress_event


class ProgressEventTest(unittest.TestCase):
    def test_round_trip_and_clamping(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            emit_progress("asr", 7, 10, "已识别 7/10 块")
        event = parse_progress_event(output.getvalue().strip())
        self.assertEqual(event["stage"], "asr")
        self.assertAlmostEqual(event["fraction"], 0.7)
        self.assertEqual(event["detail"], "已识别 7/10 块")

    def test_ordinary_or_malformed_logs_are_ignored(self):
        self.assertIsNone(parse_progress_event("[1/3] ordinary log"))
        self.assertIsNone(parse_progress_event("PIPELINE_PROGRESS not-json"))


if __name__ == "__main__":
    unittest.main()
