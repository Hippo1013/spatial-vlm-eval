import re
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_vlm_eval.models.common.runtime import GenerationResult
from spatial_vlm_eval.models.common.vision_canary import (
    BLUE_SQUARE_BOX,
    RED_CIRCLE_BOX,
    VISION_CANARY_IMAGE_SIZE,
    VISION_CANARY_QUESTION,
    make_vision_canary_image,
    validate_vision_canary_answer,
)
from spatial_vlm_eval.models.openai_compatible.client import (
    APIRequestError,
    OpenAICompatibleAdapter,
)
from spatial_vlm_eval.models.openai_compatible.vision_canary import run_vision_canary


class VisionCanaryTest(unittest.TestCase):
    def test_image_has_canonical_geometry_without_prompt_answer_leakage(self):
        image = make_vision_canary_image()
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, VISION_CANARY_IMAGE_SIZE)
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

        red_center = (
            (RED_CIRCLE_BOX[0] + RED_CIRCLE_BOX[2]) // 2,
            (RED_CIRCLE_BOX[1] + RED_CIRCLE_BOX[3]) // 2,
        )
        self.assertEqual(image.getpixel(red_center), (255, 0, 0))
        self.assertEqual(image.getpixel(RED_CIRCLE_BOX[:2]), (255, 255, 255))

        blue_center = (
            (BLUE_SQUARE_BOX[0] + BLUE_SQUARE_BOX[2]) // 2,
            (BLUE_SQUARE_BOX[1] + BLUE_SQUARE_BOX[3]) // 2,
        )
        self.assertEqual(image.getpixel(blue_center), (0, 0, 255))
        self.assertEqual(image.getpixel((BLUE_SQUARE_BOX[0] - 1, BLUE_SQUARE_BOX[1] - 1)), (254, 254, 255))
        self.assertEqual(image.getpixel((128, 48)), (255, 17, 17))

        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        self.assertGreaterEqual(len(encoded.getvalue()), 6_000)

        prompt = VISION_CANARY_QUESTION.lower()
        for leaked_answer in ("red", "blue", "circle", "square", "top", "left", "bottom", "right"):
            self.assertIsNone(re.search(rf"\b{leaked_answer}\b", prompt))

    def test_answer_validator_requires_correct_object_position_associations(self):
        accepted = (
            "A red circle is in the upper-left, and a blue square is in the lower-right.",
            "Top left: red circle; bottom right: blue square.",
            "There is a circle colored red at the top left and a square colored blue at the bottom right.",
            "The image features a blue square [0.564,0.582,0.816,0.93] and a red circle [0.176,0.086,0.426,0.43].",
        )
        for answer in accepted:
            with self.subTest(answer=answer):
                validate_vision_canary_answer(answer)

        rejected = (
            "A red circle is in the lower right and a blue square is in the upper left.",
            "A red square is top left and a blue circle is bottom right.",
            "A red circle and a blue square.",
            "A blue square [0.1,0.1,0.4,0.4] and a red circle [0.6,0.6,0.9,0.9].",
            "A red circle is top left [0.6,0.6,0.9,0.9] and a blue square is bottom right [0.1,0.1,0.4,0.4].",
            "A red circle [48,48,208,208] and a blue square [304,304,464,464].",
        )
        for answer in rejected:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                validate_vision_canary_answer(answer)

    def test_openai_compatible_canary_makes_one_call_and_writes_pass_report(self):
        adapter = OpenAICompatibleAdapter(
            profile="gpt5_openrouter_non_zdr",
            backend="openrouter",
            base_url="https://example.test/v1",
            api_key="test-key",
            metadata_retries=0,
        )
        generation = GenerationResult(
            text="A red circle is at top-left and a blue square is at bottom-right.",
            metadata={"generation_id": "gen-1", "num_media_prompt": 1},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vision_canary.json"
            with patch.object(adapter, "generate", return_value=generation) as generate:
                report = run_vision_canary(adapter, output=output)
            self.assertEqual(generate.call_count, 1)
            sent = generate.call_args.args[0]
            self.assertEqual(sent.index, -1)
            self.assertEqual(sent.image.size, VISION_CANARY_IMAGE_SIZE)
            self.assertEqual(sent.question, VISION_CANARY_QUESTION)
            self.assertTrue(report["passed"])
            self.assertEqual(report["request_count"], 1)
            self.assertEqual(report["request_image_count"], 1)
            self.assertTrue(output.is_file())

    def test_openai_compatible_canary_overwrites_stale_pass_with_failure(self):
        adapter = OpenAICompatibleAdapter(
            profile="gemini31pro_openrouter_non_zdr",
            backend="openrouter",
            base_url="https://example.test/v1",
            api_key="test-key",
            metadata_retries=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vision_canary.json"
            output.write_text('{"passed": true}\n', encoding="utf-8")
            with (
                patch.object(
                    adapter,
                    "generate",
                    return_value=GenerationResult(text="I cannot see an image."),
                ),
                self.assertRaises(ValueError),
            ):
                run_vision_canary(adapter, output=output)
            contents = output.read_text(encoding="utf-8")
            self.assertIn('"passed": false', contents)
            self.assertIn('"status": "failed"', contents)

    def test_openai_compatible_canary_persists_sanitized_api_diagnostics(self):
        adapter = OpenAICompatibleAdapter(
            profile="gemini31pro_openrouter_non_zdr",
            backend="openrouter",
            base_url="https://example.test/v1",
            api_key="test-key",
            metadata_retries=0,
        )
        failure = APIRequestError(
            "HTTP 502 from openrouter: Provider unavailable",
            status_code=502,
            error_type="provider_unavailable",
            response_headers={"x-request-id": "req-canary"},
            router_metadata={"attempt": 1, "summary": "selected=Google AI Studio"},
            elapsed_ms=123.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vision_canary.json"
            with (
                patch.object(adapter, "generate", side_effect=failure),
                self.assertRaises(APIRequestError),
            ):
                run_vision_canary(adapter, output=output)
            contents = output.read_text(encoding="utf-8")
            self.assertIn('"status_code": 502', contents)
            self.assertIn('"error_type": "provider_unavailable"', contents)
            self.assertIn('"x-request-id": "req-canary"', contents)


if __name__ == "__main__":
    unittest.main()
