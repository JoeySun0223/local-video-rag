import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from video_pipeline.cleanup.core import alignment_drift_ids, evaluate, validate_response
from video_pipeline.cleanup.prompts import SYSTEM_PROMPT
from video_pipeline.cleanup.service import _prompt, clean_document
from video_pipeline.cleanup.terms import configured_terms
from video_pipeline.shared.io import write_json


class CleanupCoreTest(unittest.TestCase):
    def test_prompt_allows_faithful_restructuring_and_reviews_by_risk(self):
        self.assertIn("可直接进入知识库的书面语", SYSTEM_PROMPT)
        self.assertIn("无业务约束的临时选择不属于有效信息", SYSTEM_PROMPT)
        self.assertIn("没有新增操作、条件或结果的总结句", SYSTEM_PROMPT)
        self.assertIn("无明确指代的", SYSTEM_PROMPT)
        self.assertIn("同一 sentence_id 内重排语序", SYSTEM_PROMPT)
        self.assertIn("不得改变、遗漏或虚构有效信息", SYSTEM_PROMPT)
        self.assertIn("长句中的多个阶段和动作必须全部保留", SYSTEM_PROMPT)
        self.assertIn("不得跨 sentence_id 搬移、合并或去重", SYSTEM_PROMPT)
        self.assertIn("主动纠正错字、同音字、近音字及病句", SYSTEM_PROMPT)
        self.assertIn("术语证据", SYSTEM_PROMPT)
        self.assertIn("审核只取决于确定性和语义风险", SYSTEM_PROMPT)

    def test_cleanup_prompt_includes_source_confirmed_terms(self):
        video_id = "4974ac58194488992f21cb86c856e34a5eb82e1a43ee0569686d74f6fc4cccde"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "resources" / "glossary.json", {
                "global": {"繁简识别": [], "通达海": []},
                "sources": {video_id: {"呈批": [], "管案": []}},
            })
            config = {"_root": str(root), "glossary": "resources/glossary.json"}
            terms = configured_terms(config, video_id)
            self.assertEqual(["繁简识别", "通达海", "呈批", "管案"], terms)
            payload = _prompt("示例", [{"id": 1, "raw_text": "原句"}], 0, 1, terms)
            self.assertIn(
                '"confirmed_business_terms": ["繁简识别", "通达海", "呈批", "管案"]',
                payload,
            )

    def test_deletion_only_is_accepted(self):
        result = evaluate("嗯，点击保存。", {
            "text": "点击保存。", "needs_review": False, "review_reason": "",
        })
        self.assertEqual("auto_accepted", result["status"])
        self.assertEqual("点击保存。", result["text"])

    def test_confirmed_correction_is_accepted(self):
        result = evaluate("点击保成。", {
            "text": "点击保存。", "needs_review": False, "review_reason": "",
        })
        self.assertEqual("auto_accepted", result["status"])
        self.assertEqual("点击保存。", result["text"])

    def test_uncertain_unchanged_sentence_requires_review(self):
        result = evaluate("该字段为听审改革。", {
            "text": "该字段为听审改革。", "needs_review": True,
            "review_reason": "疑似界面字段识别错误，需要核对画面。",
        })
        self.assertEqual("pending_review", result["status"])
        self.assertEqual("该字段为听审改革。", result["text"])

    def test_response_is_reordered_locally_by_sentence_id(self):
        result = validate_response({"results": [
            {"sentence_id": 2, "text": "二", "needs_review": False, "review_reason": ""},
            {"sentence_id": 1, "text": "一", "needs_review": False, "review_reason": ""},
        ]}, [1, 2])
        self.assertEqual({1, 2}, set(result))

    def test_harmless_extra_model_fields_are_ignored(self):
        result = validate_response({"results": [{
            "sentence_id": 1, "text": "正文", "needs_review": False,
            "review_reason": "", "confidence": 0.99,
        }]}, [1])
        self.assertEqual("正文", result[1]["text"])
        self.assertNotIn("confidence", result[1])

    @patch("video_pipeline.cleanup.service.request_json")
    def test_only_invalid_sentence_is_retried(self, request_json):
        request_json.side_effect = [
            ({"results": [
                {"sentence_id": 1, "text": "第一句。", "needs_review": False,
                 "review_reason": ""},
                {"sentence_id": 2, "text": "第二句。", "needs_review": False},
            ]}, {"total_tokens": 10}),
            ({"results": [
                {"sentence_id": 2, "text": "第二句。", "needs_review": False,
                 "review_reason": ""},
            ]}, {"total_tokens": 3}),
        ]
        raw = {
            "id": "a" * 64, "title": "测试", "sentences": [
                {"id": 1, "start_ms": 0, "end_ms": 1000, "raw_text": "第一句。"},
                {"id": 2, "start_ms": 1000, "end_ms": 2000, "raw_text": "第二句。"},
            ],
        }
        cleaned, _report, usage = clean_document(
            object(), raw, api_url="https://example.invalid", api_key="key",
            model="glm-test", thinking=False, attempts=2, max_tokens=512,
            batch_max_sentences=40, batch_max_chars=3800,
        )
        self.assertEqual(2, request_json.call_count)
        self.assertEqual(2, len(cleaned["sentences"]))
        self.assertEqual(13, usage["total_tokens"])
        retry_prompt = request_json.call_args_list[1].kwargs["user_prompt"]
        self.assertIn('"target_sentences": [{"sentence_id": 2', retry_prompt)
        self.assertNotIn('"target_sentences": [{"sentence_id": 1', retry_prompt)

    def test_consecutive_neighbor_shift_is_detected_without_flagging_single_rewrite(self):
        sentences = [
            {"id": 1, "raw_text": "打开案件列表查看案件。"},
            {"id": 2, "raw_text": "点击新增按钮创建案件。"},
            {"id": 3, "raw_text": "填写当事人基本信息。"},
            {"id": 4, "raw_text": "上传证据材料并保存。"},
            {"id": 5, "raw_text": "提交案件进入审核流程。"},
        ]
        proposals = {
            1: {"text": sentences[0]["raw_text"]},
            2: {"text": sentences[2]["raw_text"]},
            3: {"text": sentences[3]["raw_text"]},
            4: {"text": sentences[4]["raw_text"]},
            5: {"text": sentences[4]["raw_text"]},
        }
        self.assertEqual(
            [2, 3, 4], alignment_drift_ids(sentences, proposals, 0, len(sentences))
        )

    @patch("video_pipeline.cleanup.service.request_json")
    def test_detected_alignment_drift_is_repaired_only_for_affected_rows(self, request_json):
        texts = [
            "打开案件列表查看案件。", "点击新增按钮创建案件。",
            "填写当事人基本信息。", "上传证据材料并保存。",
            "提交案件进入审核流程。",
        ]
        initial = {"results": [
            {"sentence_id": 1, "text": texts[0], "needs_review": False, "review_reason": ""},
            {"sentence_id": 2, "text": texts[2], "needs_review": False, "review_reason": ""},
            {"sentence_id": 3, "text": texts[3], "needs_review": False, "review_reason": ""},
            {"sentence_id": 4, "text": texts[4], "needs_review": False, "review_reason": ""},
            {"sentence_id": 5, "text": texts[4], "needs_review": False, "review_reason": ""},
        ]}
        repairs = [
            ({"results": [{
                "sentence_id": sentence_id, "text": texts[sentence_id - 1],
                "needs_review": False, "review_reason": "",
            }]}, {"total_tokens": 1})
            for sentence_id in (2, 3, 4)
        ]
        request_json.side_effect = [(initial, {"total_tokens": 10}), *repairs]
        raw = {
            "id": "b" * 64, "title": "测试", "sentences": [
                {"id": index, "start_ms": index * 1000, "end_ms": (index + 1) * 1000,
                 "raw_text": text}
                for index, text in enumerate(texts, 1)
            ],
        }
        cleaned, report, _usage = clean_document(
            object(), raw, api_url="https://example.invalid", api_key="key",
            model="glm-test", thinking=False, attempts=2, max_tokens=512,
            batch_max_sentences=40, batch_max_chars=3800,
        )
        self.assertEqual(4, request_json.call_count)
        self.assertEqual(texts, [item["text"] for item in cleaned["sentences"]])
        self.assertEqual("alignment_drift", report["warnings"][0]["kind"])
        self.assertEqual([2, 3, 4], report["warnings"][0]["sentence_ids"])

    @patch("video_pipeline.cleanup.service.request_json")
    def test_invalid_row_exhaustion_preserves_only_that_sentence_for_review(self, request_json):
        request_json.side_effect = [
            ({"results": [
                {"sentence_id": 1, "text": "第一句。", "needs_review": False,
                 "review_reason": ""},
                {"sentence_id": 2, "text": "第二句。", "needs_review": False},
            ]}, {"total_tokens": 10}),
            RuntimeError("单句服务失败"), RuntimeError("单句服务仍失败"),
        ]
        raw = {
            "id": "c" * 64, "title": "测试", "sentences": [
                {"id": 1, "start_ms": 0, "end_ms": 1000, "raw_text": "第一句。"},
                {"id": 2, "start_ms": 1000, "end_ms": 2000, "raw_text": "第二句原文。"},
            ],
        }
        cleaned, report, _usage = clean_document(
            object(), raw, api_url="https://example.invalid", api_key="key",
            model="glm-test", thinking=False, attempts=2, max_tokens=512,
            batch_max_sentences=40, batch_max_chars=3800,
        )
        self.assertEqual("第一句。", cleaned["sentences"][0]["text"])
        self.assertEqual("第二句原文。", cleaned["sentences"][1]["text"])
        self.assertEqual(1, report["counts"]["pending_review"])
        self.assertEqual("sentence_fallback", report["warnings"][0]["kind"])

    @patch("video_pipeline.cleanup.service.request_json")
    def test_batch_exhaustion_falls_back_and_allows_document_to_complete(self, request_json):
        request_json.side_effect = [
            RuntimeError("整批失败一"), RuntimeError("整批失败二"),
            RuntimeError("小范围恢复失败"),
        ]
        raw = {
            "id": "d" * 64, "title": "测试", "sentences": [
                {"id": 1, "start_ms": 0, "end_ms": 1000, "raw_text": "第一句原文。"},
                {"id": 2, "start_ms": 1000, "end_ms": 2000, "raw_text": "第二句原文。"},
            ],
        }
        cleaned, report, _usage = clean_document(
            object(), raw, api_url="https://example.invalid", api_key="key",
            model="glm-test", thinking=False, attempts=2, max_tokens=512,
            batch_max_sentences=40, batch_max_chars=3800,
        )
        self.assertEqual(3, request_json.call_count)
        self.assertEqual(
            ["第一句原文。", "第二句原文。"],
            [item["text"] for item in cleaned["sentences"]],
        )
        self.assertEqual(2, report["counts"]["pending_review"])
        self.assertIn("batch_recovery", {item["kind"] for item in report["warnings"]})
        self.assertIn("range_fallback", {item["kind"] for item in report["warnings"]})

    def test_beginner_cleanup_uses_low_reasoning_mode(self):
        source = (Path(__file__).parents[1] / "video_pipeline" / "cleanup" / "cli.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("thinking=False", source)

    def test_review_reason_is_required_for_review(self):
        with self.assertRaises(ValueError):
            validate_response({"results": [{
                "sentence_id": 1, "text": "原文", "needs_review": True, "review_reason": "",
            }]}, [1])


if __name__ == "__main__":
    unittest.main()
