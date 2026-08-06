"""Fail-closed JSONL bridge to dataset-blind Q-Spatial specialized runners."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from ...models.common.runtime import GenerationResult, InferenceAdapter
from ...models.openai_compatible.client import image_to_png_data_uri
from .data import QSpatialModelInput
from .profiles import QSpatialProfile


def fold_system_user_prompt(system_prompt: str, user_prompt: str) -> str:
    """Losslessly preserve both official prompt strings in one user turn."""

    if not system_prompt or not user_prompt:
        raise ValueError("Q-Spatial folded prompt requires non-empty system and user text")
    return f"{system_prompt}\n\n{user_prompt}"


def load_generation_manifest(
    profile: QSpatialProfile,
    path: str | Path | None,
) -> dict[str, Any]:
    if not profile.requires_runtime_generation_manifest:
        return dict(profile.decoding)
    if path is None:
        raise ValueError(
            f"Profile {profile.key} requires a locked runtime generation manifest"
        )
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    if not isinstance(decoding, dict) or not decoding:
        raise ValueError("Generation manifest decoding must be a non-empty object")
    if "do_sample" not in decoding or "max_new_tokens" not in decoding:
        raise ValueError("Generation manifest must resolve do_sample and max_new_tokens")
    if decoding != profile.decoding:
        raise ValueError(
            f"Generation manifest decoding mismatch for {profile.key}: "
            f"got={decoding!r}, expected={profile.decoding!r}"
        )
    return dict(decoding)


class UpstreamCommandAdapter(InferenceAdapter):
    """Persistent bridge whose request schema contains no labels or split fields."""

    batch_size = 1
    supports_concurrency = False

    def __init__(
        self,
        *,
        profile: QSpatialProfile,
        command: str,
        adapter_digest: str,
        decoding: dict[str, Any],
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", str(adapter_digest)):
            raise ValueError("Specialized adapter digest must be an explicit lowercase SHA-256")
        arguments = shlex.split(str(command))
        if not arguments:
            raise ValueError(f"No upstream command configured for {profile.key}")
        executable = arguments[0]
        if not Path(executable).exists() and shutil.which(executable) is None:
            raise FileNotFoundError(f"Upstream adapter executable is unavailable: {executable}")
        self.profile = profile
        self.command = arguments
        self.adapter_digest = str(adapter_digest)
        self.decoding = dict(decoding)
        self._lock = threading.Lock()
        self._closed = False
        self._process = subprocess.Popen(
            self.command,
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

    def generate(self, model_input: QSpatialModelInput) -> GenerationResult:
        request = {
            "schema_version": 1,
            "action": "generate",
            "index": int(model_input.index),
            "system_prompt": model_input.system_prompt,
            "user_prompt": model_input.user_prompt,
            "image": {
                "count": 1,
                "mode": "RGB",
                "png_data_uri": image_to_png_data_uri(model_input.image),
            },
            "profile": self.profile.key,
            "model_revision": self.profile.revision,
            "upstream_commit": self.profile.upstream_commit,
            "inference_protocol": self.profile.inference_protocol,
            "decoding": self.decoding,
        }
        forbidden = {"answer", "answer_value", "answer_unit", "question_type", "split"}
        if forbidden & set(request):
            raise RuntimeError("Specialized request accidentally contains Q-Spatial scoring fields")
        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise RuntimeError(
                    f"Upstream adapter for {self.profile.key} is not running "
                    f"(exit={self._process.poll()})"
                )
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(
                json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError(
                f"Upstream adapter for {self.profile.key} closed stdout "
                f"(exit={self._process.poll()})"
            )
        response = json.loads(response_line)
        if not isinstance(response, dict):
            raise ValueError("Upstream adapter response must be a JSON object")
        for key, expected in {
            "index": int(model_input.index),
            "profile": self.profile.key,
            "model_revision": self.profile.revision,
            "inference_protocol": self.profile.inference_protocol,
            "decoding": self.decoding,
            "system_role_supported": False,
        }.items():
            if response.get(key) != expected:
                raise ValueError(
                    f"Upstream adapter response {key} mismatch: "
                    f"got={response.get(key)!r}, expected={expected!r}"
                )
        generation = response.get("generation")
        if not isinstance(generation, dict):
            raise ValueError("Upstream adapter response generation must be an object")
        if generation.get("num_model_image_tensors") != 1 and generation.get("num_media_prompt") != 1:
            raise ValueError("Upstream adapter must prove exactly one model image tensor or media prompt")
        for key in ("template_sha256", "system_prompt_sha256", "user_prompt_sha256", "folded_prompt_sha256"):
            if not isinstance(generation.get(key), str) or not re.fullmatch(
                r"[0-9a-f]{64}", generation[key]
            ):
                raise ValueError(f"Upstream adapter must return a valid {key}")
        if not isinstance(response.get("raw_prediction"), str):
            raise ValueError("Upstream adapter raw_prediction must be a string")
        return GenerationResult(
            text=response["raw_prediction"],
            metadata=dict(generation),
            warnings=tuple(str(value) for value in response.get("warnings") or []),
        )

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
