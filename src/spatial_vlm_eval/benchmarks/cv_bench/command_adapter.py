"""Fail-closed JSON-lines bridge to locked upstream specialized-model runners."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from ...models.common.runtime import GenerationResult, InferenceAdapter, RestrictedVisionInput
from ...models.openai_compatible.client import image_to_png_data_uri
from .profiles import CVBenchProfile


def load_generation_manifest(profile: CVBenchProfile, path: str | Path | None) -> dict[str, Any]:
    if not profile.requires_runtime_generation_manifest:
        return dict(profile.decoding)
    if path is None:
        raise ValueError(
            f"Profile {profile.key} requires a locked runtime generation manifest; "
            "unresolved upstream/checkpoint defaults are never guessed"
        )
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "profile": profile.key,
        "model_revision": profile.revision,
        "upstream_commit": profile.upstream_commit,
    }
    for key, expected in checks.items():
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
    for key, expected in profile.decoding.items():
        if key in {"source"}:
            continue
        if decoding.get(key) != expected:
            raise ValueError(
                f"Generation manifest field {key} mismatch for {profile.key}: "
                f"got={decoding.get(key)!r}, expected={expected!r}"
            )
    return dict(decoding)


class UpstreamCommandAdapter(InferenceAdapter):
    """Persistent subprocess adapter; stdin never contains gold/task/source fields."""

    batch_size = 1
    supports_concurrency = False

    def __init__(
        self,
        *,
        profile: CVBenchProfile,
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
            raise RuntimeError("Could not open JSON-lines pipes to the upstream adapter")

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.profile.model,
            "model_revision": self.profile.revision,
            "backend": self.profile.default_backend,
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "inference_protocol": self.profile.inference_protocol,
            "chat_template": self.profile.chat_template,
            "image_processing": dict(self.profile.image_processing),
            "decoding": dict(self.decoding),
            "adapter_digest": self.adapter_digest,
            "known_deviation": self.profile.known_deviation,
            "upstream": {
                "repository": self.profile.upstream_url,
                "commit": self.profile.upstream_commit,
                "transport": "persistent JSON-lines subprocess",
            },
        }

    def generate(self, model_input: RestrictedVisionInput) -> GenerationResult:
        request = {
            "schema_version": 1,
            "action": "generate",
            "index": int(model_input.index),
            "prompt": str(model_input.question),
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
        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise RuntimeError(
                    f"Upstream adapter for {self.profile.key} is not running "
                    f"(exit={self._process.poll()})"
                )
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
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
        checks = {
            "index": int(model_input.index),
            "profile": self.profile.key,
            "model_revision": self.profile.revision,
            "inference_protocol": self.profile.inference_protocol,
            "decoding": self.decoding,
        }
        for key, expected in checks.items():
            if response.get(key) != expected:
                raise ValueError(
                    f"Upstream adapter response {key} mismatch: "
                    f"got={response.get(key)!r}, expected={expected!r}"
                )
        generation = response.get("generation")
        if not isinstance(generation, dict):
            raise ValueError("Upstream adapter response generation must be an object")
        tensor_count = generation.get("num_model_image_tensors")
        media_count = generation.get("num_media_prompt")
        if tensor_count != 1 and media_count != 1:
            raise ValueError(
                "Upstream adapter must prove exactly one model image tensor or media prompt"
            )
        template_sha256 = generation.get("template_sha256")
        if not isinstance(template_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", template_sha256
        ):
            raise ValueError("Upstream adapter must return the rendered template SHA-256")
        if not isinstance(response.get("raw_prediction"), str):
            raise ValueError("Upstream adapter raw_prediction must be a string")
        warnings = tuple(str(value) for value in response.get("warnings") or [])
        return GenerationResult(
            text=response["raw_prediction"],
            metadata=dict(generation),
            warnings=warnings,
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
                        try:
                            stream.close()
                        except (BrokenPipeError, OSError):
                            pass
