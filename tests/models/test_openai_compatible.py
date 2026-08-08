import base64
import io
import subprocess
import unittest
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.msmu.data import MSMUModelInput
from spatial_vlm_eval.benchmarks.q_spatial.data import QSpatialModelInput, STANDARD_SYSTEM_PROMPT
from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILES as QSPATIAL_PROFILES
from spatial_vlm_eval.models.openai_compatible.client import (
    APIRequestError,
    HTTPJSONResponse,
    OpenAICompatibleAdapter,
    _CURL_STATUS_MARKER,
    _curl_http_request,
    openai_compatible_model_ids,
)


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

    def test_q_spatial_system_user_and_vllm_sampling_extension_is_isolated(self):
        value = QSpatialModelInput(
            7,
            Image.new("RGB", (5, 3), (12, 34, 56)),
            STANDARD_SYSTEM_PROMPT,
            "Question: How far?",
        )
        adapter = OpenAICompatibleAdapter(
            profile=QSPATIAL_PROFILES["qwen3_vl_8b"],
            backend="vllm",
            base_url="https://example.test/v1",
            api_key="test-key",
            served_model_name="qwen3-vl-8b-qspatial",
        )
        payload = adapter.request_payload(value)
        self.assertEqual([message["role"] for message in payload["messages"]], ["system", "user"])
        self.assertEqual(payload["messages"][0]["content"], STANDARD_SYSTEM_PROMPT)
        self.assertEqual(payload["messages"][1]["content"][1]["text"], "Question: How far?")
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["presence_penalty"], 1.5)
        self.assertEqual(payload["seed"], 3407)
        self.assertEqual(payload["max_tokens"], 1024)

        legacy = self.adapter("llava_next_mistral_7b", "vllm").request_payload(model_input())
        self.assertEqual([message["role"] for message in legacy["messages"]], ["user"])
        self.assertNotIn("top_k", legacy)
        self.assertNotIn("presence_penalty", legacy)

    def test_vllm_assistant_prefill_disables_generation_prompt(self):
        adapter = self.adapter("llava_next_mistral_7b", "vllm")
        payload = adapter.request_payload(
            model_input(),
            messages=[
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "partial answer"},
            ],
            continue_final_message=True,
        )
        self.assertTrue(payload["continue_final_message"])
        self.assertFalse(payload["add_generation_prompt"])

    def test_http_headers_request_json_and_openrouter_metadata(self):
        adapter = self.adapter("gpt5_openrouter_non_zdr", "openrouter")
        headers = adapter._headers()
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["X-OpenRouter-Metadata"], "enabled")
        self.assertEqual(headers["Authorization"], "Bearer test-key")

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

    def test_openrouter_non_zdr_profiles_are_explicit_and_backend_locked(self):
        cases = [
            (
                "gpt5_openrouter_non_zdr",
                "openai",
                "msmu_gpt5_question_only_openrouter_non_zdr_v3_medium_16384",
                "medium",
                16384,
            ),
            (
                "gemini31pro_openrouter_non_zdr",
                "google-ai-studio",
                "msmu_gemini31pro_question_only_openrouter_non_zdr_v3_medium_16384",
                "medium",
                16384,
            ),
        ]
        for profile, provider, protocol, reasoning_effort, max_tokens in cases:
            with self.subTest(profile=profile):
                adapter = self.adapter(profile, "openrouter")
                payload = adapter.request_payload(model_input())
                self.assertEqual(
                    payload["provider"],
                    {
                        "only": [provider],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                        "data_collection": "deny",
                        "zdr": False,
                    },
                )
                self.assertEqual(adapter.metadata()["inference_protocol"], protocol)
                self.assertEqual(adapter.metadata()["provider_policy"]["zdr"], False)
                self.assertEqual(
                    payload["reasoning"],
                    {"effort": reasoning_effort, "exclude": True},
                )
                self.assertEqual(payload["max_tokens"], max_tokens)

        self.assertEqual(
            self.adapter("gpt5_openrouter_non_zdr", "openrouter").metadata()[
                "model_revision"
            ],
            "openrouter-canonical:openai/gpt-5-2025-08-07",
        )
        self.assertEqual(
            self.adapter("gemini31pro_openrouter_non_zdr", "openrouter").metadata()[
                "model_revision"
            ],
            "openrouter-canonical:google/gemini-3.1-pro-preview-20260219",
        )

        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.adapter("gpt5_openrouter_non_zdr", "openai")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.adapter("gemini31pro_openrouter_non_zdr", "google")

    def test_gemini_uses_locked_temperature_and_reasoning(self):
        direct = self.adapter("gemini31pro", "google")
        payload = direct.request_payload(model_input())
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_completion_tokens"], 192)
        routed = self.adapter("gemini31pro", "openrouter")
        self.assertEqual(routed.request_payload(model_input())["provider"]["only"], ["google-ai-studio"])

    def test_packyapi_gemini_uses_exact_served_model_and_openai_compatible_shape(self):
        adapter = OpenAICompatibleAdapter(
            profile="gemini31pro_openrouter_non_zdr",
            backend="packyapi",
            base_url="https://www.packyapi.com/v1",
            api_key="test-key",
            served_model_name="gemini-3.1-pro-preview",
        )
        payload = adapter.request_payload(model_input())
        self.assertEqual(payload["model"], "gemini-3.1-pro-preview")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertEqual(payload["max_tokens"], 16384)
        self.assertNotIn("provider", payload)
        self.assertNotIn("X-OpenRouter-Metadata", adapter._headers())
        self.assertEqual(adapter.metadata()["provider_policy"]["token_group"], "Gemini-slb")

        response = HTTPJSONResponse(
            data={
                "id": "packy-1",
                "model": "gemini-3.1-pro-preview",
                "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1},
            },
            headers={},
            elapsed_ms=123.0,
        )
        with patch.object(adapter, "_request_json", return_value=response):
            result = adapter.generate(model_input())
        self.assertEqual(result.text, "A")
        self.assertEqual(result.metadata["provider"], "PackyAPI Gemini-slb")
        self.assertEqual(result.metadata["api_source"], "packyapi")
        self.assertEqual(result.metadata["requested_model"], "gemini-3.1-pro-preview")

        wrong = HTTPJSONResponse(
            data={
                "id": "packy-2",
                "model": "gemini-3-pro-preview",
                "choices": [{"message": {"content": "A"}}],
            },
            headers={},
        )
        with (
            patch.object(adapter, "_request_json", return_value=wrong),
            self.assertRaisesRegex(APIRequestError, "returned-model mismatch"),
        ):
            adapter.generate(model_input())

    def test_openai_compatible_model_catalog_requires_data_array(self):
        body = b'{"data":[{"id":"gemini-3.1-pro-preview"},{"id":"other"}]}'
        with patch(
            "spatial_vlm_eval.models.openai_compatible.client._curl_http_request",
            return_value=(body, {}, 200),
        ) as request:
            identifiers = openai_compatible_model_ids(
                base_url="https://www.packyapi.com/v1/",
                api_key="test-key",
            )
        self.assertEqual(identifiers, ("gemini-3.1-pro-preview", "other"))
        self.assertEqual(request.call_args.kwargs["url"], "https://www.packyapi.com/v1/models")

    def test_openrouter_generation_metadata_is_required_and_recorded(self):
        adapter = self.adapter("gpt5", "openrouter", "https://openrouter.ai/api/v1")
        chat = HTTPJSONResponse(
            data={
                "id": "gen-123",
                "model": "openai/gpt-5-2025-08-07",
                "choices": [{"message": {"content": "left"}, "finish_reason": "stop"}],
            },
            headers={"x-generation-id": "gen-123"},
        )
        generation = HTTPJSONResponse(
            data={
                "data": {
                    "id": "gen-123",
                    "model": "openai/gpt-5-2025-08-07",
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
        self.assertEqual(result.metadata["canonical_model"], "openai/gpt-5-2025-08-07")
        self.assertEqual(result.metadata["provider"], "OpenAI")
        self.assertEqual(result.metadata["reasoning_tokens"], 2)
        self.assertEqual(request.call_count, 2)

    def test_openrouter_retries_eventually_consistent_generation_metadata(self):
        adapter = OpenAICompatibleAdapter(
            profile="gpt5_openrouter_non_zdr",
            backend="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            metadata_retries=1,
        )
        chat = HTTPJSONResponse(
            data={
                "id": "gen-delayed",
                "model": "openai/gpt-5-2025-08-07",
                "choices": [{"message": {"content": "left"}, "finish_reason": "stop"}],
            },
            headers={"x-generation-id": "gen-delayed"},
        )
        delayed = APIRequestError("Generation gen-delayed not found", status_code=404)
        generation = HTTPJSONResponse(
            data={
                "data": {
                    "id": "gen-delayed",
                    "model": "openai/gpt-5-2025-08-07",
                    "provider_name": "OpenAI",
                    "num_media_prompt": 1,
                }
            },
            headers={},
        )
        with (
            patch.object(adapter, "_request_json", side_effect=[chat, delayed, generation]) as request,
            patch("spatial_vlm_eval.models.openai_compatible.client.time.sleep") as sleep,
        ):
            result = adapter.generate(model_input())
        self.assertEqual(result.text, "left")
        self.assertEqual(result.metadata["generation_id"], "gen-delayed")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[0].kwargs["method"], "POST")
        self.assertEqual(request.call_args_list[1].kwargs["method"], "GET")
        self.assertEqual(request.call_args_list[2].kwargs["method"], "GET")
        sleep.assert_called_once_with(0.25)

    def test_http_error_preserves_typed_router_diagnostics_without_secrets(self):
        adapter = self.adapter("gpt5_openrouter_non_zdr", "openrouter")
        response_body = (
                b'{"error":{"code":502,"message":"Provider unavailable",'
                b'"metadata":{"error_type":"provider_unavailable"}},'
                b'"openrouter_metadata":{"attempt":1,"summary":"selected=OpenAI"}}'
        )
        response_headers = {
                "X-Request-Id": "req-123",
                "X-Generation-Id": "gen-123",
                "CF-Ray": "ray-123",
                "Set-Cookie": "must-not-be-recorded",
        }
        with (
            patch(
                "spatial_vlm_eval.models.openai_compatible.client._curl_http_request",
                return_value=(response_body, response_headers, 502),
            ),
            self.assertRaisesRegex(APIRequestError, "Provider unavailable") as raised,
        ):
            adapter.generate(model_input())
        diagnostics = raised.exception.diagnostics()
        self.assertEqual(diagnostics["status_code"], 502)
        self.assertEqual(diagnostics["error_type"], "provider_unavailable")
        self.assertEqual(diagnostics["response_headers"]["x-request-id"], "req-123")
        self.assertEqual(diagnostics["response_headers"]["x-generation-id"], "gen-123")
        self.assertNotIn("set-cookie", diagnostics["response_headers"])
        self.assertEqual(diagnostics["router_metadata"]["attempt"], 1)

    def test_curl_transport_keeps_authorization_out_of_process_arguments(self):
        captured: dict[str, object] = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["body_input"] = kwargs["input"]
            captured["header_input"] = kwargs["env"]["SPATIAL_VLM_CURL_HEADERS"]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b'{"ok":true}' + _CURL_STATUS_MARKER + b"200",
                stderr=b"",
            )

        with patch("subprocess.run", side_effect=fake_run):
            body, headers, status = _curl_http_request(
                method="POST",
                url="https://example.test/v1/chat/completions",
                body=b'{"hello":"world"}',
                headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
                timeout=30.0,
            )
        self.assertEqual(body, b'{"ok":true}')
        self.assertEqual(status, 200)
        self.assertEqual(headers, {})
        self.assertNotIn("test-key", " ".join(captured["command"]))
        self.assertIn("request_body=$(cat)", captured["command"][2])
        self.assertEqual(captured["body_input"], b'{"hello":"world"}')
        self.assertIn("Authorization: Bearer test-key", captured["header_input"])

    def test_provider_or_image_count_mismatch_is_not_a_prediction(self):
        adapter = self.adapter("gpt5", "openrouter")
        chat = HTTPJSONResponse(
            data={"id": "gen", "choices": [{"message": {"content": "x"}}]},
            headers={},
        )
        bad_generation = HTTPJSONResponse(
            data={
                "data": {
                    "model": "openai/gpt-5-2025-08-07",
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

    def test_openrouter_alias_or_unpinned_revision_is_not_a_prediction(self):
        adapter = self.adapter("gpt5", "openrouter")
        chat = HTTPJSONResponse(
            data={"id": "gen", "choices": [{"message": {"content": "x"}}]},
            headers={},
        )
        alias_generation = HTTPJSONResponse(
            data={
                "data": {
                    "model": "openai/gpt-5",
                    "provider_name": "OpenAI",
                    "num_media_prompt": 1,
                }
            },
            headers={},
        )
        with (
            patch.object(adapter, "_request_json", side_effect=[chat, alias_generation]),
            self.assertRaisesRegex(APIRequestError, "model mismatch"),
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

if __name__ == "__main__":
    unittest.main()
