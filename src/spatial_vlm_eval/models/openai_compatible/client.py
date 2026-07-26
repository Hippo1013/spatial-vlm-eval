"""Dependency-free OpenAI-compatible multimodal HTTP client."""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ...benchmarks.msmu.data import MSMUModelInput
from ..common.runtime import GenerationResult, InferenceAdapter
from ..profiles import InferenceProfile, get_profile

SUPPORTED_PROFILE_KEYS = {
    "gpt5",
    "gemini31pro",
    "llava_next_mistral_7b",
    "llava_next_yi_34b",
    "internvl3_8b",
    "internvl3_38b",
    "internvl3_78b",
}


class APIRequestError(RuntimeError):
    """A retryable or terminal API contract failure."""


@dataclass(frozen=True, slots=True)
class HTTPJSONResponse:
    data: dict[str, Any]
    headers: dict[str, str]
    elapsed_ms: float | None = None


def image_to_png_data_uri(image: Any) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def single_image_user_message(question: str, image_data_uri: str) -> list[dict[str, Any]]:
    """Return exactly one user message, one question, and one PNG image."""

    if not image_data_uri.startswith("data:image/png;base64,"):
        raise ValueError("MSMU OpenAI-compatible requests require a PNG data URI")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": str(question)},
            ],
        }
    ]


def _safe_api_error_body(raw: bytes) -> str:
    try:
        value = json.loads(raw.decode("utf-8", errors="replace"))
        if isinstance(value, dict):
            error = value.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error.get("code") or "API error")[:500]
            return str(value.get("message") or "API error")[:500]
    except json.JSONDecodeError:
        pass
    return raw.decode("utf-8", errors="replace")[:500]


class OpenAICompatibleAdapter(InferenceAdapter):
    """One-image chat-completions adapter with strict backend policies."""

    batch_size = 1
    supports_concurrency = True

    def __init__(
        self,
        *,
        profile: str | InferenceProfile,
        backend: str,
        base_url: str,
        api_key: str,
        served_model_name: str | None = None,
        timeout: float = 180.0,
        metadata_retries: int = 4,
    ) -> None:
        self.profile = get_profile(profile) if isinstance(profile, str) else profile
        if self.profile.key not in SUPPORTED_PROFILE_KEYS:
            raise ValueError(f"Profile {self.profile.key!r} is not OpenAI-compatible")
        self.backend = str(backend)
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.timeout = float(timeout)
        self.metadata_retries = int(metadata_retries)
        if self.timeout <= 0:
            raise ValueError("API timeout must be positive")
        if self.metadata_retries < 0:
            raise ValueError("metadata_retries must be non-negative")
        self.served_model_name = served_model_name or self.profile.served_model_name
        self._validate_backend()
        if not self.api_key:
            raise ValueError(f"No API key configured for backend {self.backend}")

    def _validate_backend(self) -> None:
        allowed = {
            "gpt5": {"openrouter", "openai"},
            "gemini31pro": {"openrouter", "google"},
        }.get(self.profile.key, {"vllm"})
        if self.backend not in allowed:
            raise ValueError(
                f"Backend {self.backend!r} is incompatible with profile {self.profile.key!r}; "
                f"allowed={sorted(allowed)}"
            )
        if self.backend == "vllm" and not self.served_model_name:
            raise ValueError("vLLM profiles require a served model name")

    @property
    def request_model(self) -> str:
        if self.backend == "vllm":
            assert self.served_model_name is not None
            return self.served_model_name
        if self.backend == "openai":
            return "gpt-5"
        if self.backend == "google":
            return "gemini-3.1-pro-preview"
        return self.profile.model

    def metadata(self) -> dict[str, Any]:
        provider_policy: dict[str, Any] | None = None
        if self.backend == "openrouter":
            provider_policy = self._openrouter_provider_policy()
        return {
            "model": self.profile.model,
            "model_revision": self.profile.revision,
            "served_model_name": self.request_model,
            "backend": self.backend,
            "api_base_url": self.base_url,
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "inference_protocol": self.profile.inference_protocol,
            "chat_template": self.profile.chat_template,
            "image_processing": {
                "source": "MSMU RGB only",
                "transport": "one canonical PNG data URI",
                "mime_type": "image/png",
                "image_count": 1,
                "content_order": ["image", "question"],
            },
            "decoding": {
                "do_sample": False if self.profile.temperature == 0 else None,
                "num_beams": 1 if self.backend == "vllm" else None,
                "temperature": self.profile.temperature,
                "reasoning_effort": self.profile.reasoning_effort,
                "max_completion_tokens": self.profile.max_new_tokens,
                "stream": False,
            },
            "provider_policy": provider_policy,
            "upstream": {
                "required_vllm_version": "0.19.0" if self.backend == "vllm" else None,
                "model_repository": self.profile.model,
            },
        }

    def _openrouter_provider_policy(self) -> dict[str, Any]:
        only = "openai" if self.profile.key == "gpt5" else "google-ai-studio"
        return {
            "only": [only],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        }

    def request_payload(self, model_input: MSMUModelInput) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.request_model,
            "messages": single_image_user_message(
                model_input.question,
                image_to_png_data_uri(model_input.image),
            ),
            "stream": False,
        }
        if self.backend in {"vllm", "openrouter"}:
            payload["max_tokens"] = self.profile.max_new_tokens
        else:
            payload["max_completion_tokens"] = self.profile.max_new_tokens
        if self.profile.temperature is not None:
            payload["temperature"] = self.profile.temperature
        if self.profile.reasoning_effort is not None:
            if self.backend == "openrouter":
                payload["reasoning"] = {
                    "effort": self.profile.reasoning_effort,
                    "exclude": True,
                }
            elif self.backend in {"openai", "google"}:
                payload["reasoning_effort"] = self.profile.reasoning_effort
        if self.backend == "openrouter":
            payload["provider"] = self._openrouter_provider_policy()
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.backend == "openrouter":
            headers["X-OpenRouter-Metadata"] = "enabled"
        return headers

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> HTTPJSONResponse:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method=method,
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
        except urllib.error.HTTPError as exc:
            detail = _safe_api_error_body(exc.read())
            raise APIRequestError(f"HTTP {exc.code} from {self.backend}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise APIRequestError(f"Network error from {self.backend}: {exc.reason}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise APIRequestError(f"Non-JSON response from {self.backend}") from exc
        if not isinstance(parsed, dict):
            raise APIRequestError(f"Unexpected non-object response from {self.backend}")
        if parsed.get("error"):
            raise APIRequestError(f"API error from {self.backend}: {parsed['error']}")
        return HTTPJSONResponse(
            data=parsed,
            headers=headers,
            elapsed_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
            ]
            return "".join(pieces)
        if content is None:
            return ""
        raise APIRequestError("Assistant content has an unsupported shape")

    def _openrouter_generation_metadata(
        self,
        response: HTTPJSONResponse,
    ) -> dict[str, Any]:
        generation_id = (
            response.headers.get("x-generation-id")
            or response.data.get("id")
            or response.data.get("generation_id")
        )
        if not generation_id:
            raise APIRequestError("OpenRouter response did not include a generation id")
        query = urllib.parse.urlencode({"id": str(generation_id)})
        last_error: Exception | None = None
        generation: dict[str, Any] | None = None
        for attempt in range(self.metadata_retries + 1):
            try:
                result = self._request_json(
                    method="GET",
                    url=f"{self.base_url}/generation?{query}",
                ).data
                value = result.get("data")
                if not isinstance(value, dict):
                    raise APIRequestError("OpenRouter generation metadata lacks data object")
                generation = value
                break
            except APIRequestError as exc:
                last_error = exc
                if attempt < self.metadata_retries:
                    time.sleep(min(0.25 * (2**attempt), 2.0))
        if generation is None:
            assert last_error is not None
            raise APIRequestError(
                "Could not retrieve OpenRouter generation metadata for "
                f"generation {generation_id}: {last_error}"
            )

        expected_provider = "OpenAI" if self.profile.key == "gpt5" else "Google AI Studio"
        provider = str(generation.get("provider_name") or response.data.get("provider") or "")
        if provider != expected_provider:
            raise APIRequestError(
                f"OpenRouter provider mismatch: got {provider!r}, expected {expected_provider!r}"
            )
        canonical_model = str(generation.get("model") or response.data.get("model") or "")
        if canonical_model != self.profile.model:
            raise APIRequestError(
                f"OpenRouter model mismatch: got {canonical_model!r}, expected {self.profile.model!r}"
            )
        media_count = generation.get("num_media_prompt")
        if media_count is None or int(media_count) != 1:
            raise APIRequestError(f"OpenRouter metadata reports num_media_prompt={media_count!r}, expected 1")
        return {
            "generation_id": str(generation_id),
            "canonical_model": canonical_model,
            "provider": provider,
            "upstream_id": generation.get("upstream_id"),
            "finish_reason": generation.get("finish_reason"),
            "native_finish_reason": generation.get("native_finish_reason"),
            "prompt_tokens": generation.get("tokens_prompt"),
            "output_tokens": generation.get("tokens_completion"),
            "native_prompt_tokens": generation.get("native_tokens_prompt"),
            "native_output_tokens": generation.get("native_tokens_completion"),
            "reasoning_tokens": generation.get("native_tokens_reasoning"),
            "cost": generation.get("total_cost"),
            "upstream_cost": generation.get("upstream_inference_cost"),
            "latency_ms": generation.get("latency"),
            "generation_time_ms": generation.get("generation_time"),
            "num_media_prompt": int(media_count),
        }

    def generate(self, model_input: MSMUModelInput) -> GenerationResult:
        response = self._request_json(
            method="POST",
            url=f"{self.base_url}/chat/completions",
            payload=self.request_payload(model_input),
        )
        choices = response.data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise APIRequestError("Chat completion response must contain exactly one choice")
        choice = choices[0]
        returned_model = response.data.get("model")
        if self.backend == "vllm" and str(returned_model or "") != self.request_model:
            raise APIRequestError(
                f"vLLM served-model mismatch: got {returned_model!r}, expected {self.request_model!r}"
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise APIRequestError("Chat completion choice lacks an assistant message")
        text = self._message_text(message)
        if self.backend == "openrouter":
            generation_metadata = self._openrouter_generation_metadata(response)
        else:
            usage = response.data.get("usage") if isinstance(response.data.get("usage"), dict) else {}
            completion_details = (
                usage.get("completion_tokens_details")
                if isinstance(usage.get("completion_tokens_details"), dict)
                else {}
            )
            generation_metadata = {
                "generation_id": response.data.get("id"),
                "provider_request_id": response.headers.get("x-request-id"),
                "canonical_model": response.data.get("model"),
                "provider": self.backend,
                "upstream_id": response.data.get("id"),
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": completion_details.get("reasoning_tokens"),
                "cost": None,
                "latency_ms": response.elapsed_ms,
                "num_media_prompt": 1,
            }
        warnings = ("model returned an empty text completion",) if not text.strip() else ()
        return GenerationResult(text=text, metadata=generation_metadata, warnings=warnings)
