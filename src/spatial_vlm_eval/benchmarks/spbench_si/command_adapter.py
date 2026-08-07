"""Fail-closed JSONL bridge to dataset-blind SPBench-SI specialized runners."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from ...models.common.runtime import GenerationResult, InferenceAdapter, pixel_sha256
from ...models.openai_compatible.client import image_to_png_data_uri
from .data import SPBenchSIModelInput
from .profiles import SPBenchSIProfile


def fold_system_user_prompt(system_prompt: str, user_prompt: str) -> str:
    if not system_prompt or not user_prompt:
        raise ValueError("SPBench-SI folded prompt requires non-empty system and user text")
    return f"{system_prompt}\n\n{user_prompt}"


def load_generation_manifest(profile: SPBenchSIProfile, path: str | Path | None) -> dict[str, Any]:
    if not profile.requires_runtime_generation_manifest:
        return dict(profile.decoding)
    if path is None:
        raise ValueError(f"Profile {profile.key} requires a locked runtime generation manifest")
    manifest = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    for key, expected in {
        "profile": profile.key,
        "model_revision": profile.revision,
        "upstream_commit": profile.upstream_commit,
    }.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Generation manifest {key} mismatch for {profile.key}: "
                f"got={manifest.get(key)!r}, expected={expected!r}"
            )
    decoding = manifest.get("decoding")
    if not isinstance(decoding, dict) or decoding != profile.decoding:
        raise ValueError(f"Generation manifest decoding mismatch for {profile.key}")
    return dict(decoding)


class UpstreamCommandAdapter(InferenceAdapter):
    """Persistent bridge whose request contains no answer, type, scene, dataset, or source row."""

    batch_size = 1
    supports_concurrency = False

    def __init__(
        self,
        *,
        profile: SPBenchSIProfile,
        command: str,
        adapter_digest: str,
        decoding: dict[str, Any],
        batch_size: int = 1,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", str(adapter_digest)):
            raise ValueError("Specialized adapter digest must be an explicit lowercase SHA-256")
        arguments = shlex.split(str(command))
        if not arguments:
            raise ValueError(f"No upstream command configured for {profile.key}")
        if not Path(arguments[0]).exists() and shutil.which(arguments[0]) is None:
            raise FileNotFoundError(f"Upstream adapter executable is unavailable: {arguments[0]}")
        self.profile = profile
        self.command = arguments
        self.adapter_digest = str(adapter_digest)
        self.decoding = dict(decoding)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Specialized adapter batch_size must be positive")
        self._lock = threading.Lock()
        self._closed = False
        self._process = subprocess.Popen(
            arguments,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Could not open JSONL pipes to the upstream adapter")

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.profile.model,
            "model_revision": self.profile.revision,
            "backend": self.profile.default_backend,
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "comparison_group": self.profile.comparison_group,
            "inference_protocol": self.profile.inference_protocol,
            "chat_template": self.profile.chat_template,
            "system_role_supported": False,
            "system_transport": self.profile.system_transport,
            "folded_prompt_separator": "\\n\\n",
            "image_processing": dict(self.profile.image_processing),
            "decoding": dict(self.decoding),
            "seed_strategy": self.profile.seed_strategy,
            "provider_nondeterministic": self.profile.provider_nondeterministic,
            "adapter_digest": self.adapter_digest,
            "known_deviation": self.profile.known_deviation,
            "upstream": {
                "repository": self.profile.upstream_url,
                "commit": self.profile.upstream_commit,
                "transport": "persistent JSON-lines subprocess",
            },
        }

    def _request(self, model_input: SPBenchSIModelInput) -> dict[str, Any]:
        source_digest = pixel_sha256(model_input.image)
        request = {
            "schema_version": 1,
            "action": "generate",
            "index": int(model_input.index),
            "system_prompt": model_input.system_prompt,
            "user_prompt": model_input.user_prompt,
            "image": {
                "count": 1,
                "mode": "RGB",
                "pixel_sha256": source_digest,
                "png_data_uri": image_to_png_data_uri(model_input.image),
            },
            "profile": self.profile.key,
            "model_revision": self.profile.revision,
            "upstream_commit": self.profile.upstream_commit,
            "inference_protocol": self.profile.inference_protocol,
            "decoding": self.decoding,
        }
        forbidden = {"ground_truth", "answer", "question_type", "scene", "scene_name", "dataset", "row"}
        if forbidden & set(request):
            raise RuntimeError("Specialized request accidentally contains SPBench-SI private fields")
        return request

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise RuntimeError(
                    f"Upstream adapter for {self.profile.key} is not running (exit={self._process.poll()})"
                )
            assert self._process.stdin is not None and self._process.stdout is not None
            self._process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"Upstream adapter for {self.profile.key} closed stdout")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise ValueError("Upstream adapter response must be a JSON object")
        return response

    def _validate_response(self, request: dict[str, Any], response: dict[str, Any]) -> GenerationResult:
        for key, expected in {
            "index": request["index"],
            "profile": self.profile.key,
            "model_revision": self.profile.revision,
            "inference_protocol": self.profile.inference_protocol,
            "decoding": self.decoding,
            "system_role_supported": False,
        }.items():
            if response.get(key) != expected:
                raise ValueError(
                    f"Upstream adapter response {key} mismatch: got={response.get(key)!r}, expected={expected!r}"
                )
        generation = response.get("generation")
        if not isinstance(generation, dict):
            raise ValueError("Upstream adapter response generation must be an object")
        if generation.get("source_rgb_count") != 1:
            raise ValueError("Upstream adapter must prove exactly one source RGB")
        source_digest = request["image"]["pixel_sha256"]
        if generation.get("source_rgb_sha256") != source_digest:
            raise ValueError("Upstream adapter source RGB digest mismatch")
        if self.profile.comparison_group != "rgb_only":
            if generation.get("derived_from_source_rgb_sha256") != source_digest:
                raise ValueError("Derived input is not proven to come from the same RGB")
        if self.profile.key == "spatialbot_zoedepth":
            if generation.get("num_model_image_tensors") != 2 or generation.get("derived_depth_count") != 1:
                raise ValueError("SpatialBot ZoeDepth must prove one RGB tensor plus one derived depth tensor")
        elif generation.get("num_model_image_tensors") != 1:
            raise ValueError("Specialized backend must prove exactly one model-bound image tensor")
        for key in ("template_sha256", "system_prompt_sha256", "user_prompt_sha256", "folded_prompt_sha256"):
            if not isinstance(generation.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", generation[key]):
                raise ValueError(f"Upstream adapter must return a valid {key}")
        if not isinstance(response.get("raw_prediction"), str):
            raise ValueError("Upstream adapter raw_prediction must be a string")
        return GenerationResult(
            response["raw_prediction"], dict(generation),
            tuple(str(value) for value in response.get("warnings") or []),
        )

    def generate(self, model_input: SPBenchSIModelInput) -> GenerationResult:
        request = self._request(model_input)
        return self._validate_response(request, self._exchange(request))

    def generate_batch(self, model_inputs: list[SPBenchSIModelInput]) -> list[GenerationResult]:
        if len(model_inputs) <= 1:
            return [self.generate(value) for value in model_inputs]
        if not self.profile.native_batch_probe:
            raise ValueError(f"{self.profile.key} does not support native batch generation")
        requests = [self._request(value) for value in model_inputs]
        response = self._exchange({"schema_version": 1, "action": "generate_batch", "requests": requests})
        responses = response.get("responses") if isinstance(response, dict) else None
        if not isinstance(responses, list) or len(responses) != len(requests):
            raise ValueError("Upstream batch response size mismatch")
        return [self._validate_response(request, value) for request, value in zip(requests, responses)]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process.poll() is None and self._process.stdin is not None:
                try:
                    self._process.stdin.write('{"schema_version":1,"action":"close"}\n')
                    self._process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=10)
            finally:
                for stream in (self._process.stdin, self._process.stdout):
                    if stream is not None and not stream.closed:
                        stream.close()
