from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_pipeline.segments.core import (
    SEGMENT_BOUNDARY_SYSTEM_PROMPT,
    SEGMENT_LABEL_SYSTEM_PROMPT,
    SegmentResponseError,
    build_document,
    effective_char_count,
    generate_segments,
    make_label_batches,
    meaningful_sentences,
    ranges_from_starts,
    segment_prompt_for,
    select_sources,
    transcript_payload,
    validate_segment_response,
    validate_label_response,
)


def source():
    return {
        "source_id": "abc123",
        "title": "测试视频",
        "sentences": [
            {"sentence_id": 1, "start_ms": 0, "end_ms": 30_000, "text": "介绍案件概览。"},
            {"sentence_id": 2, "start_ms": 31_000, "end_ms": 90_000, "text": "说明案件趋势。"},
            {"sentence_id": 3, "start_ms": 91_000, "end_ms": 150_000, "text": "下面演示权限设置。"},
            {"sentence_id": 4, "start_ms": 151_000, "end_ms": 190_000, "text": "点击保存权限。"},
        ],
    }


def labels_response():
    return {
        "labels": [
            {"position": 1, "title": "案件数据概览", "summary": "介绍案件概览和趋势。", "keywords": ["案件概览", "案件趋势"]},
            {"position": 2, "title": "权限设置操作", "summary": "演示权限设置及保存操作。", "keywords": ["权限设置", "保存"]},
        ]
    }


class SegmentCoreTest(unittest.TestCase):
    def test_starts_derive_complete_non_overlapping_ranges(self):
        rows = meaningful_sentences(source())
        starts = validate_segment_response({
            "start_sentence_ids": [1, 3],
        }, rows)
        ranges = ranges_from_starts(starts, rows)
        self.assertEqual(
            [(row["start_sentence_id"], row["end_sentence_id"]) for row in ranges],
            [(1, 2), (3, 4)],
        )

    def test_last_segment_automatically_reaches_final_sentence(self):
        rows = meaningful_sentences(source())
        self.assertEqual(ranges_from_starts([1], rows)[0]["end_sentence_id"], 4)

    def test_boundary_response_must_start_at_first_sentence(self):
        with self.assertRaisesRegex(SegmentResponseError, "必须从 sentence_id=1 开始"):
            validate_segment_response({
                "start_sentence_ids": [2, 3],
            }, meaningful_sentences(source()))

    def test_boundary_response_must_be_strictly_increasing(self):
        with self.assertRaisesRegex(SegmentResponseError, "严格递增"):
            validate_segment_response({
                "start_sentence_ids": [1, 3, 3],
            }, meaningful_sentences(source()))

    def test_labels_must_cover_exact_batch_positions(self):
        response = labels_response()
        response["labels"].pop()
        with self.assertRaisesRegex(SegmentResponseError, "缺少 position=\[2\]"):
            validate_label_response(response, [1, 2])

    def test_two_stage_generation_builds_plan(self):
        rows = meaningful_sentences(source())
        calls = []

        def fake_request(client, api_key, model, system_prompt, prompt, max_tokens, thinking):
            calls.append((system_prompt, prompt))
            if system_prompt == SEGMENT_BOUNDARY_SYSTEM_PROMPT:
                return {"start_sentence_ids": [1, 3]}, {"total_tokens": 10}
            self.assertEqual(system_prompt, SEGMENT_LABEL_SYSTEM_PROMPT)
            return labels_response(), {"total_tokens": 20}

        plan, usage = generate_segments(
            object(), "key", "glm-5.3", source(), rows, 8192, False, 1,
            12_000, 6, request=fake_request,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(usage["total_tokens"], 30)
        self.assertEqual(
            [(row["start_sentence_id"], row["end_sentence_id"]) for row in plan],
            [(1, 2), (3, 4)],
        )

    def test_one_time_boundary_audit_requires_and_captures_explanations(self):
        rows = meaningful_sentences(source())
        audit = {}

        def fake_request(client, api_key, model, system_prompt, prompt, max_tokens, thinking):
            if system_prompt.startswith(SEGMENT_BOUNDARY_SYSTEM_PROMPT):
                return {
                    "start_sentence_ids": [1, 3],
                    "overall_basis": "按主题变化划分。",
                    "boundary_explanations": [
                        {"start_sentence_id": 1, "reason": "案件概览开始。"},
                        {"start_sentence_id": 3, "reason": "转入权限操作。"},
                    ],
                }, {"total_tokens": 10}
            return labels_response(), {"total_tokens": 20}

        generate_segments(
            object(), "key", "glm-5.3", source(), rows, 8192, False, 1,
            12_000, 6, request=fake_request, boundary_audit=audit,
        )
        self.assertEqual(audit["start_sentence_ids"], [1, 3])
        self.assertEqual(audit["boundary_explanations"][1]["reason"], "转入权限操作。")

    def test_plan_materializes_original_text_and_timestamps(self):
        rows = meaningful_sentences(source())
        ranges = ranges_from_starts([1, 3], rows)
        labels = validate_label_response(labels_response(), [1, 2])
        plan = [{
            "start_sentence_id": chapter["start_sentence_id"],
            "end_sentence_id": chapter["end_sentence_id"],
            **labels[chapter["position"]],
        } for chapter in ranges]
        document = build_document(source(), rows, plan, "videos/test.mp4")
        self.assertEqual(document["video_id"], "abc123")
        self.assertNotIn("chapters", document)
        self.assertEqual(document["segments"][0]["segment_no"], 1)
        self.assertEqual(document["segments"][0]["content"], "介绍案件概览。说明案件趋势。")
        self.assertEqual(document["segments"][1]["end_ms"], 190_000)

    def test_label_batches_respect_count_and_character_target(self):
        ranges = [
            {"position": 1, "text": "一" * 8},
            {"position": 2, "text": "二" * 8},
            {"position": 3, "text": "三" * 8},
        ]
        batches = make_label_batches(ranges, max_chars=16, max_segments=2)
        self.assertEqual([[row["position"] for row in batch] for batch in batches], [[1, 2], [3]])

    def test_boundary_payload_uses_character_offsets_not_timestamps(self):
        rows = meaningful_sentences(source())
        payload = transcript_payload(source(), rows)
        counts = [effective_char_count(row["text"]) for row in rows]
        self.assertEqual(payload["total_chars"], sum(counts))
        self.assertEqual(payload["sentences"][0][:3], [1, 0, counts[0]])
        self.assertEqual(
            payload["sentences"][1][:3],
            [2, counts[0], counts[0] + counts[1]],
        )
        prompt = segment_prompt_for(source(), rows)
        self.assertNotIn("start_ms", prompt)
        self.assertNotIn("end_ms", prompt)

    def test_effective_character_count_excludes_whitespace_and_punctuation(self):
        self.assertEqual(effective_char_count("案件 A-1，保存。\n"), 6)

    def test_boundary_response_ignores_extra_formatting_fields(self):
        self.assertEqual(
            validate_segment_response(
                {"start_sentence_ids": [1, 3], "explanation": "多余字段"},
                meaningful_sentences(source()),
            ),
            [1, 3],
        )

    def test_label_response_normalizes_keywords_and_ignores_extra_fields(self):
        value = labels_response()
        value["explanation"] = "多余字段"
        value["labels"][0]["keywords"] = "案件概览，案件趋势"
        value["labels"][0]["extra"] = "忽略"
        self.assertEqual(
            validate_label_response(value, [1, 2])[1]["keywords"],
            ["案件概览", "案件趋势"],
        )

    def test_segment_rules_cover_qa_workflows_and_non_content(self):
        self.assertIn("完整回答", SEGMENT_BOUNDARY_SYSTEM_PROMPT)
        self.assertIn("校验修改", SEGMENT_BOUNDARY_SYSTEM_PROMPT)
        self.assertIn("结束语不得单独成章", SEGMENT_BOUNDARY_SYSTEM_PROMPT)
        self.assertIn("全文不超过1300字", SEGMENT_BOUNDARY_SYSTEM_PROMPT)
        self.assertIn("1301至2200字通常划分为2章", SEGMENT_BOUNDARY_SYSTEM_PROMPT)
        self.assertIn("每章均不少于500字", SEGMENT_BOUNDARY_SYSTEM_PROMPT)
        self.assertIn("不得形成不足500字", SEGMENT_BOUNDARY_SYSTEM_PROMPT)
        self.assertIn("输出前", SEGMENT_BOUNDARY_SYSTEM_PROMPT)

    def test_empty_sentences_are_not_sent_or_required(self):
        data = source()
        data["sentences"].insert(2, {"sentence_id": 25, "start_ms": 90_100, "end_ms": 90_500, "text": "   "})
        data["sentences"][3]["sentence_id"] = 26
        data["sentences"][4]["sentence_id"] = 27
        self.assertEqual(
            [row["sentence_id"] for row in meaningful_sentences(data)], [1, 2, 26, 27]
        )

    def test_partition_manifest_is_not_treated_as_video(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").touch()
            (root / "video123.json").touch()
            self.assertEqual([path.name for path in select_sources(root, None, None)], ["video123.json"])


if __name__ == "__main__":
    unittest.main()
