import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.msmu.data import MSMUModelInput
from spatial_vlm_eval.models.openai_compatible.client import (
    APIRequestError,
    HTTPJSONResponse,
    OpenAICompatibleAdapter,
)
from spatial_vlm_eval.models.openai_compatible.vision_canary import validate_solid_color_answers


def model_input() -> MSMUModelInput:
    return MSMUModelInput(4, Image.new("RGB", (5, 3), (12, 34, 56)), "Where is the cup?")


class OpenAICompatibleContractTest(unittest.TestCase):
    def adapter(self, profile, backend, base="https://example.test/v1"):
        return OpenAICompatibleAdapter(
            profile=profile,
            backend=backend,
            base_url=base,
            api_key="test-key",
            metadata_retries=0,
        )

    def test_request_has_one_user_question_and_one_decodable_png(self):
        adapter = self.adapter("llava_next_mistral_7b", "vllm")
        payload = adapter.request_payload(model_input())
        self.assertEqual(set(payload["messages"][0]), {"role", "content"})
        self.assertEqual(payload["messages"][0]["role"], "user")
        content = payload["messages"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[1], {"type": "text", "text": "Where is the cup?"})
        uri = content[0]["image_url"]["url"]
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        decoded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        self.assertEqual(decoded.size, (5, 3))
        self.assertEqual(decoded.getpixel((0, 0)), (12, 34, 56))
        rendered = str(payload)
        self.assertNotIn("reference", rendered.lower())
        self.assertNotIn("system", rendered.lower())

    def test_gpt5_omits_temperature_and_openrouter_fails_closed(self):
        adapter = self.adapter("gpt5", "openrouter")
        payload = adapter.request_payload(model_input())
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["reasoning"], {"effort": "low", "exclude": True})
        self.assertEqual(
            payload["provider"],
            {
                "only": ["openai"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "zdr": True,
            },
        )
        self.assertEqual(payload["max_tokens"], 192)

    def test_gemini_uses_locked_temperature_and_reasoning(self):
        direct = self.adapter("gemini31pro", "google")
        payload = direct.request_payload(model_input())
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_completion_tokens"], 192)
        routed = self.adapter("gemini31pro", "openrouter")
        self.assertEqual(routed.request_payload(model_input())["provider"]["only"], ["google-ai-studio"])

    def test_openrouter_generation_metadata_is_required_and_recorded(self):
        adapter = self.adapter("gpt5", "openrouter", "https://openrouter.ai/api/v1")
        chat = HTTPJSONResponse(
            data={
                "id": "gen-123",
                "model": "openai/gpt-5",
                "choices": [{"message": {"content": "left"}, "finish_reason": "stop"}],
            },
            headers={"x-generation-id": "gen-123"},
        )
        generation = HTTPJSONResponse(
            data={
                "data": {
                    "id": "gen-123",
                    "model": "openai/gpt-5",
                    "provider_name": "OpenAI",
                    "upstream_id": "chatcmpl-1",
                    "finish_reason": "stop",
                    "tokens_prompt": 40,
                    "tokens_completion": 12,
                    "native_tokens_reasoning": 2,
                    "total_cost": 0.01,
                    "latency": 250,
                    "num_media_prompt": 1,
                }
            },
            headers={},
        )
        with patch.object(adapter, "_request_json", side_effect=[chat, generation]) as request:
            result = adapter.generate(model_input())
        self.assertEqual(result.text, "left")
        self.assertEqual(result.metadata["generation_id"], "gen-123")
        self.assertEqual(result.metadata["provider"], "OpenAI")
        self.assertEqual(result.metadata["reasoning_tokens"], 2)
        self.assertEqual(request.call_count, 2)

    def test_provider_or_image_count_mismatch_is_not_a_prediction(self):
        adapter = self.adapter("gpt5", "openrouter")
        chat = HTTPJSONResponse(
            data={"id": "gen", "choices": [{"message": {"content": "x"}}]},
            headers={},
        )
        bad_generation = HTTPJSONResponse(
            data={
                "data": {
                    "model": "openai/gpt-5",
                    "provider_name": "Azure",
                    "num_media_prompt": 0,
                }
            },
            headers={},
        )
        with (
            patch.object(adapter, "_request_json", side_effect=[chat, bad_generation]),
            self.assertRaises(APIRequestError),
        ):
            adapter.generate(model_input())

    def test_real_empty_model_content_remains_a_success(self):
        adapter = self.adapter("llava_next_mistral_7b", "vllm")
        response = HTTPJSONResponse(
            data={
                "id": "chat-1",
                "model": "llava-next-mistral-7b-msmu",
                "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                "usage": {},
            },
            headers={},
        )
        with patch.object(adapter, "_request_json", return_value=response):
            result = adapter.generate(model_input())
        self.assertEqual(result.text, "")
        self.assertIn("empty", result.warnings[0])

    def test_vllm_served_model_mismatch_is_rejected(self):
        adapter = self.adapter("llava_next_mistral_7b", "vllm")
        response = HTTPJSONResponse(
            data={
                "id": "chat-1",
                "model": "wrong-served-name",
                "choices": [{"message": {"content": "red"}, "finish_reason": "stop"}],
            },
            headers={},
        )
        with (
            patch.object(adapter, "_request_json", return_value=response),
            self.assertRaisesRegex(APIRequestError, "served-model mismatch"),
        ):
            adapter.generate(model_input())

    def test_synthetic_vision_canary_requires_both_colors(self):
        validate_solid_color_answers("red", "BLUE.")
        with self.assertRaisesRegex(ValueError, "red image"):
            validate_solid_color_answers("unknown", "blue")


if __name__ == "__main__":
    unittest.main()
