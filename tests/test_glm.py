import unittest

from video_pipeline.shared.glm import reasoning_parameters


class GlmReasoningParametersTest(unittest.TestCase):
    def test_glm_53_uses_supported_reasoning_effort(self):
        self.assertEqual({
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }, reasoning_parameters("glm-5.3", True))
        self.assertEqual("low", reasoning_parameters("glm-5.3", False)["reasoning_effort"])

    def test_older_models_keep_boolean_thinking_switch(self):
        self.assertEqual(
            {"thinking": {"type": "disabled"}},
            reasoning_parameters("glm-4.5", False),
        )


if __name__ == "__main__":
    unittest.main()
