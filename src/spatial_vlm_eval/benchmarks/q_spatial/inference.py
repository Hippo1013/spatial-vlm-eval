"""Two-stage, publication-gated inference orchestration for Q-Spatial."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ...models.common.runtime import (
    GenerationResult,
    InferenceAdapter,
    atomic_write_json,
    pixel_sha256,
    run_recoverable_inference,
    utc_now,
)
from ...models.openai_compatible.client import (
    OpenAICompatibleAdapter,
    image_to_png_data_uri,
    single_image_user_message,
)
from .command_adapter import UpstreamCommandAdapter, fold_system_user_prompt, load_generation_manifest
from .data import (
    DATASET_FILES,
    DATASET_REVISION,
    LLAVA_CONTINUATION,
    LLAVA_USER_FORMAT_SUFFIX,
    OFFICIAL_TEST_SIZE,
    QSpatialModelInput,
    QSpatialTestContract,
    SMOKE8_INDICES,
    STANDARD_SYSTEM_PROMPT,
    STANDARD_SYSTEM_PROMPT_SHA256,
)
from .prediction_validation import read_jsonl, validate_prediction_rows
from .processor_audit import audit_processor
from .profiles import PROFILE_SEQUENCE, PROFILES, QSpatialProfile, ordered_profiles
from .scorer import SCORER_PROTOCOL, parse_measurement

COLOR_CANARY_PROTOCOL = "q_spatial_pure_red_blue_single_rgb_v1"


class ResourceBlockedError(RuntimeError):
    """A fail-closed resource gate suitable for explicit batch skipping."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _env_token(profile_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", profile_key).upper()


def _revision_tag(revision: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", revision).strip("-")


def track_directory(output_root: str | Path, profile: QSpatialProfile) -> Path:
    return (
        Path(output_root).resolve()
        / "runs"
        / profile.key
        / _revision_tag(profile.revision)
        / profile.inference_protocol
    )


class ProfiledContract:
    """Expose the locked prompt without revealing the private scoring row."""

    def __init__(self, contract: QSpatialTestContract) -> None:
        self.contract = contract
        self.dataset_root = contract.dataset_root
        self.dataset_fingerprint = contract.dataset_fingerprint

    def __len__(self) -> int:
        return len(self.contract)

    def model_input(self, index: int) -> QSpatialModelInput:
        return self.contract.model_input(index)

    def model_inputs(self, indices: list[int] | tuple[int, ...]) -> list[QSpatialModelInput]:
        return [self.model_input(index) for index in indices]

    def prediction_row(self, index: int, prediction: str) -> dict[str, Any]:
        return self.contract.prediction_row(index, prediction)


class BoundAdapter(InferenceAdapter):
    """Bind a delegate to the complete Q-Spatial profile provenance."""

    def __init__(
        self,
        delegate: InferenceAdapter,
        *,
        profile: QSpatialProfile,
        adapter_digest: str,
        processor_audit: dict[str, Any] | None,
    ) -> None:
        self.delegate = delegate
        self.profile = profile
        self.adapter_digest = adapter_digest
        self.processor_audit = processor_audit
        self.batch_size = int(getattr(delegate, "batch_size", 1))
        self.supports_concurrency = bool(getattr(delegate, "supports_concurrency", False))

    def metadata(self) -> dict[str, Any]:
        metadata = dict(self.delegate.metadata())
        metadata.update(
            {
                "model": self.profile.model,
                "model_revision": self.profile.revision,
                "backend": metadata.get("backend", self.profile.default_backend),
                "profile": self.profile.key,
                "input_profile": self.profile.input_profile,
                "comparison_group": self.profile.comparison_group,
                "inference_protocol": self.profile.inference_protocol,
                "chat_template": self.profile.chat_template,
                "system_role_supported": self.profile.system_role_supported,
                "image_processing": dict(self.profile.image_processing),
                "decoding": dict(self.profile.decoding),
                "seed_strategy": self.profile.seed_strategy,
                "provider_nondeterministic": self.profile.provider_nondeterministic,
                "adapter_digest": self.adapter_digest,
                "processor_audit": self.processor_audit,
                "known_deviation": self.profile.known_deviation,
            }
        )
        metadata.setdefault(
            "upstream",
            {"repository": self.profile.upstream_url or self.profile.model, "commit": self.profile.upstream_commit},
        )
        return metadata

    def generate(self, model_input: QSpatialModelInput) -> GenerationResult:
        result = self.delegate.generate(model_input)
        generation = dict(result.metadata)
        generation.setdefault("num_media_prompt", 1)
        generation.setdefault("source_rgb_count", 1)
        generation.setdefault("system_role_supported", self.profile.system_role_supported)
        generation.setdefault(
            "system_prompt_sha256",
            hashlib.sha256(model_input.system_prompt.encode("utf-8")).hexdigest(),
        )
        generation.setdefault(
            "user_prompt_sha256",
            hashlib.sha256(model_input.user_prompt.encode("utf-8")).hexdigest(),
        )
        generation.setdefault(
            "template_sha256",
            _digest(
                {
                    "chat_template": self.profile.chat_template,
                    "system_role_supported": self.profile.system_role_supported,
                    "system_prompt": model_input.system_prompt,
                    "user_prompt": model_input.user_prompt,
                    "image_pixel_sha256": pixel_sha256(model_input.image),
                    "media_count": 1,
                }
            ),
        )
        return GenerationResult(result.text, generation, tuple(result.warnings))

    def generate_batch(self, model_inputs: list[QSpatialModelInput]) -> list[GenerationResult]:
        return [self.generate(model_input) for model_input in model_inputs]

    def close(self) -> None:
        self.delegate.close()


class LlavaTwoStageAdapter(InferenceAdapter):
    """Official LLaVA formatting repair with two same-image model calls."""

    batch_size = 1
    supports_concurrency = True

    def __init__(self, base: OpenAICompatibleAdapter, profile: QSpatialProfile) -> None:
        if not profile.llava_two_stage or profile.system_role_supported:
            raise ValueError("LLaVA two-stage adapter requires the locked folded-user profile")
        self.base = base
        self.profile = profile

    def metadata(self) -> dict[str, Any]:
        metadata = dict(self.base.metadata())
        metadata["chat_template"] = self.profile.chat_template
        metadata["llava_two_stage"] = {
            "enabled": True,
            "stage_1_max_new_tokens": int(self.profile.decoding["max_new_tokens"]),
            "stage_2_max_new_tokens": int(self.profile.decoding["second_stage_max_new_tokens"]),
            "continuation_sha256": hashlib.sha256(LLAVA_CONTINUATION.encode("utf-8")).hexdigest(),
        }
        return metadata

    @staticmethod
    def _transcript_digest(
        *,
        input_value: QSpatialModelInput,
        user_text: str,
        assistant_text: str | None,
    ) -> str:
        return _digest(
            {
                "roles": ["user"] + (["assistant"] if assistant_text is not None else []),
                "user_text": user_text,
                "assistant_text": assistant_text,
                "image_pixel_sha256": pixel_sha256(input_value.image),
                "media_count": 1,
            }
        )

    def generate(self, model_input: QSpatialModelInput) -> GenerationResult:
        first_user = f"{model_input.user_prompt}\n{LLAVA_USER_FORMAT_SUFFIX}"
        folded = fold_system_user_prompt(model_input.system_prompt, first_user)
        image_uri = image_to_png_data_uri(model_input.image)
        first_messages = single_image_user_message(folded, image_uri)
        first = self.base.generate_messages(
            model_input,
            first_messages,
            max_tokens=int(self.profile.decoding["max_new_tokens"]),
        )
        partial_assistant = first.text + LLAVA_CONTINUATION
        second_messages = [*first_messages, {"role": "assistant", "content": partial_assistant}]
        second = self.base.generate_messages(
            model_input,
            second_messages,
            max_tokens=int(self.profile.decoding["second_stage_max_new_tokens"]),
            continue_final_message=True,
        )
        raw_prediction = partial_assistant + second.text
        image_hash = pixel_sha256(model_input.image)
        metadata = {
            "num_media_prompt": 1,
            "source_rgb_count": 1,
            "same_image_in_both_calls": True,
            "image_pixel_sha256": image_hash,
            "system_role_supported": False,
            "system_prompt_sha256": hashlib.sha256(model_input.system_prompt.encode("utf-8")).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(model_input.user_prompt.encode("utf-8")).hexdigest(),
            "template_sha256": self._transcript_digest(
                input_value=model_input,
                user_text=folded,
                assistant_text=partial_assistant,
            ),
            "model_calls": [
                {
                    "stage": 1,
                    "raw_output": first.text,
                    "max_new_tokens": int(self.profile.decoding["max_new_tokens"]),
                    "media_count": 1,
                    "image_pixel_sha256": image_hash,
                    "template_sha256": self._transcript_digest(
                        input_value=model_input, user_text=folded, assistant_text=None
                    ),
                    "provider": first.metadata,
                },
                {
                    "stage": 2,
                    "raw_output": second.text,
                    "max_new_tokens": int(self.profile.decoding["second_stage_max_new_tokens"]),
                    "continue_final_message": True,
                    "media_count": 1,
                    "image_pixel_sha256": image_hash,
                    "template_sha256": self._transcript_digest(
                        input_value=model_input,
                        user_text=folded,
                        assistant_text=partial_assistant,
                    ),
                    "provider": second.metadata,
                },
            ],
        }
        warnings = tuple(first.warnings) + tuple(second.warnings)
        if not raw_prediction.strip() and not warnings:
            warnings = ("model returned an empty two-stage completion",)
        return GenerationResult(raw_prediction, metadata, warnings)

    def close(self) -> None:
        self.base.close()


@dataclass(frozen=True, slots=True)
class ResolvedConfiguration:
    profile: QSpatialProfile
    backend: str
    base_urls: tuple[str, ...]
    decoding: dict[str, Any]
    adapter_digest: str
    command: str | None
    processor_audit: dict[str, Any] | None

    @property
    def sharding(self) -> dict[str, Any]:
        workers = len(self.base_urls) if self.base_urls else 1
        return {
            "strategy": "single_endpoint_concurrent_requests" if self.base_urls else "single_persistent_runner",
            "worker_count": workers,
            "tensor_parallel_size": self.profile.default_tensor_parallel_size,
        }


def _open_adapter_digest(profile: QSpatialProfile) -> str:
    from ...models.openai_compatible import client

    files = [Path(inspect.getfile(client)), Path(__file__), Path(inspect.getfile(type(profile)))]
    return _digest(
        {
            "profile_registry_digest": profile.registry_digest,
            "files": {path.name: _file_digest(path) for path in files},
        }
    )


def _profile_env(profile: QSpatialProfile, suffix: str) -> str | None:
    return os.environ.get(f"QSPATIAL_{_env_token(profile.key)}_{suffix}")


def _configured_gpu_ids(profile: QSpatialProfile) -> tuple[int, ...]:
    raw = _profile_env(profile, "GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw.strip():
        return ()
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ValueError(f"Invalid GPU selection for {profile.key}: {raw!r}")
    return values


def inspect_local_gpus(profile: QSpatialProfile, backend: str) -> dict[str, Any]:
    """Read inventory and process state without managing any existing process."""

    if backend == "openrouter":
        return {"applicable": False, "reason": "remote API backend"}
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise FileNotFoundError("Local Q-Spatial inference requires nvidia-smi for read-only preflight")
    query = subprocess.run(
        [
            executable,
            "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",", 5)]
        if len(fields) != 6:
            raise ValueError(f"Unexpected nvidia-smi inventory row: {line!r}")
        inventory.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    selected = _configured_gpu_ids(profile)
    expected_selected = profile.default_tensor_parallel_size if backend == "vllm" else 1
    if len(selected) != expected_selected:
        raise ResourceBlockedError(
            f"{profile.key} requires {expected_selected} explicit GPU id(s) for {backend}; "
            f"configure QSPATIAL_{_env_token(profile.key)}_GPU_IDS"
        )
    by_index = {item["index"]: item for item in inventory}
    missing = [index for index in selected if index not in by_index]
    if missing:
        raise ValueError(f"Configured GPUs are not present for {profile.key}: {missing}")
    if profile.key == "internvl3_78b":
        if len(selected) != 4:
            raise ResourceBlockedError(
                "InternVL3-78B requires exactly four explicit 80GB GPU ids; two-GPU execution is forbidden"
            )
        undersized = [index for index in selected if by_index[index]["memory_total_mib"] < 79_000]
        if undersized:
            raise ResourceBlockedError(
                f"InternVL3-78B requires four 80GB GPUs; undersized ids: {undersized}"
            )
    process_query = subprocess.run(
        [
            executable,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    process_lines = [line for line in process_query.stdout.splitlines() if line.strip()]
    if backend == "upstream_transformers":
        selected_uuids = {by_index[index]["uuid"] for index in selected}
        occupied = [line for line in process_lines if line.split(",", 1)[0].strip() in selected_uuids]
        if occupied:
            raise ResourceBlockedError(
                f"{profile.key} selected GPU already has compute processes; existing processes were untouched"
            )
        insufficient = [
            index for index in selected if by_index[index]["memory_free_mib"] < profile.min_free_gpu_mib
        ]
        busy = [index for index in selected if by_index[index]["utilization_percent"] > 10]
        if insufficient or busy:
            raise ResourceBlockedError(
                f"{profile.key} GPU preflight failed: insufficient_free={insufficient}, busy={busy}"
            )
    return {
        "applicable": True,
        "selected_gpu_ids": list(selected),
        "inventory": inventory,
        "compute_processes": process_lines,
        "policy": "read-only; no process is terminated or adopted",
    }


def _resolved_command_configuration(profile: QSpatialProfile) -> ResolvedConfiguration:
    command = _profile_env(profile, "COMMAND")
    adapter_digest = _profile_env(profile, "ADAPTER_DIGEST")
    if not command or not adapter_digest:
        token = _env_token(profile.key)
        raise ValueError(
            f"Set QSPATIAL_{token}_COMMAND and QSPATIAL_{token}_ADAPTER_DIGEST for {profile.key}"
        )
    decoding = load_generation_manifest(profile, _profile_env(profile, "GENERATION_MANIFEST"))
    return ResolvedConfiguration(
        profile=profile,
        backend="upstream_transformers",
        base_urls=(),
        decoding=decoding,
        adapter_digest=adapter_digest,
        command=command,
        processor_audit=None,
    )


def resolve_configuration(profile: QSpatialProfile) -> ResolvedConfiguration:
    backend = _profile_env(profile, "BACKEND") or profile.default_backend
    if profile.adapter_kind == "upstream_command":
        return _resolved_command_configuration(profile)
    if backend not in {"vllm", "openrouter"}:
        raise ValueError(f"Unsupported backend {backend!r} for {profile.key}")
    processor_report: dict[str, Any] | None = None
    if backend == "vllm":
        model_path = os.environ.get(profile.model_path_env, "")
        if not model_path:
            raise ValueError(f"Set {profile.model_path_env} for processor/template audit")
        processor_report = audit_processor(profile, model_path)
        raw_urls = _profile_env(profile, "BASE_URLS") or os.environ.get("QSPATIAL_VLLM_BASE_URLS", "")
        urls = tuple(value.strip().rstrip("/") for value in raw_urls.split(",") if value.strip())
        expected = 1
        if len(urls) != expected:
            raise ValueError(
                f"{profile.key} requires {expected} vLLM endpoint(s); set "
                f"QSPATIAL_{_env_token(profile.key)}_BASE_URLS"
            )
    else:
        urls = (os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),)
    return ResolvedConfiguration(
        profile=profile,
        backend=backend,
        base_urls=urls,
        decoding=dict(profile.decoding),
        adapter_digest=_open_adapter_digest(profile),
        command=None,
        processor_audit=processor_report,
    )


def _request_timeout_seconds(backend: str) -> float:
    variable = "QSPATIAL_VLLM_API_TIMEOUT" if backend == "vllm" else "QSPATIAL_API_TIMEOUT"
    return float(os.environ.get(variable, "600" if backend == "vllm" else "180"))


def _vllm_max_model_len(configuration: ResolvedConfiguration) -> int | None:
    if configuration.backend != "vllm":
        return None
    value = int(os.environ.get("QSPATIAL_VLLM_MAX_MODEL_LEN", "32768"))
    if value <= 0:
        raise ValueError("QSPATIAL_VLLM_MAX_MODEL_LEN must be positive")
    return value


def _runtime_retry_policy(backend: str) -> dict[str, int]:
    if backend == "vllm":
        return {
            "retries": int(os.environ.get("QSPATIAL_VLLM_INFERENCE_RETRIES", "0")),
            "retry_missing_passes": int(os.environ.get("QSPATIAL_VLLM_RETRY_MISSING_PASSES", "1")),
        }
    return {
        "retries": int(os.environ.get("QSPATIAL_INFERENCE_RETRIES", "2")),
        "retry_missing_passes": 1 if backend == "openrouter" else 0,
    }


def build_adapter(configuration: ResolvedConfiguration, endpoint_index: int = 0) -> BoundAdapter:
    profile = configuration.profile
    if configuration.command:
        delegate: InferenceAdapter = UpstreamCommandAdapter(
            profile=profile,
            command=configuration.command,
            adapter_digest=configuration.adapter_digest,
            decoding=configuration.decoding,
        )
    else:
        api_key = (
            os.environ.get("OPENROUTER_API_KEY", "")
            if configuration.backend == "openrouter"
            else os.environ.get("VLLM_API_KEY", "local") or "local"
        )
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY; API keys are never accepted on the command line")
        base = OpenAICompatibleAdapter(
            profile=profile,
            backend=configuration.backend,
            base_url=configuration.base_urls[endpoint_index],
            api_key=api_key,
            served_model_name=profile.served_model_name,
            timeout=_request_timeout_seconds(configuration.backend),
            metadata_retries=int(os.environ.get("QSPATIAL_OPENROUTER_METADATA_RETRIES", "10")),
            policy_key=profile.api_policy_key,
            image_source="Q-Spatial RGB only",
        )
        delegate = LlavaTwoStageAdapter(base, profile) if profile.llava_two_stage else base
    return BoundAdapter(
        delegate,
        profile=profile,
        adapter_digest=configuration.adapter_digest,
        processor_audit=configuration.processor_audit,
    )


def binding(configuration: ResolvedConfiguration, contract: QSpatialTestContract) -> dict[str, Any]:
    processor_identity = None
    if configuration.processor_audit:
        processor_identity = {
            key: configuration.processor_audit.get(key)
            for key in (
                "profile",
                "model_revision",
                "processor_class",
                "system_role_supported",
                "system_prompt_sha256",
                "user_prompt_sha256",
                "final_stage_1_user_prompt_sha256",
                "logical_image_placeholder",
                "logical_image_placeholder_count",
                "rendered_template_sha256",
                "pixel_values_shape",
                "model_snapshot_revision_verified",
                "llava_two_stage",
            )
        }
    return {
        "dataset": {
            "revision": DATASET_REVISION,
            "fingerprint": contract.dataset_fingerprint,
            "files": {item.name: item.sha256 for item in DATASET_FILES},
            "parquet_root": str(contract.parquet_root),
            "scannet_rgb_root": str(contract.scannet_rgb_root),
        },
        "prompt": {
            "system_prompt_sha256": STANDARD_SYSTEM_PROMPT_SHA256,
            "user_template": "Question: {question}",
            "system_role_supported": configuration.profile.system_role_supported,
            "llava_two_stage": configuration.profile.llava_two_stage,
        },
        "profile": {
            "key": configuration.profile.key,
            "model": configuration.profile.model,
            "model_revision": configuration.profile.revision,
            "input_profile": configuration.profile.input_profile,
            "comparison_group": configuration.profile.comparison_group,
            "inference_protocol": configuration.profile.inference_protocol,
            "registry_digest": configuration.profile.registry_digest,
            "upstream_commit": configuration.profile.upstream_commit,
            "seed_strategy": configuration.profile.seed_strategy,
            "provider_nondeterministic": configuration.profile.provider_nondeterministic,
        },
        "adapter": {
            "backend": configuration.backend,
            "base_urls": list(configuration.base_urls),
            "adapter_digest": configuration.adapter_digest,
            "decoding": configuration.decoding,
            "processor_audit": processor_identity,
            "vllm_max_model_len": _vllm_max_model_len(configuration),
        },
        "test_protocol": {
            "vision_canary": COLOR_CANARY_PROTOCOL,
            "smoke_indices": list(SMOKE8_INDICES),
            "capacity_candidates": list(_capacity_candidates(configuration.backend)),
        },
        "sharding": {
            **configuration.sharding,
            "configured_gpu_ids": list(_configured_gpu_ids(configuration.profile)),
        },
    }


def _color_canary_specs() -> tuple[tuple[str, QSpatialModelInput], ...]:
    prompt = "Question: What is the single solid color of this image? Answer with the color name."
    return (
        (
            "red",
            QSpatialModelInput(-1, Image.new("RGB", (512, 512), (255, 0, 0)), STANDARD_SYSTEM_PROMPT, prompt),
        ),
        (
            "blue",
            QSpatialModelInput(-2, Image.new("RGB", (512, 512), (0, 0, 255)), STANDARD_SYSTEM_PROMPT, prompt),
        ),
    )


def _canary_report(adapter: BoundAdapter) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    passed = True
    for expected, model_input in _color_canary_specs():
        result = adapter.generate(model_input)
        answer = result.text.casefold()
        color_ok = expected in answer
        media_ok = (
            result.metadata.get("num_media_prompt") == 1
            or result.metadata.get("num_model_image_tensors") == 1
        )
        template = result.metadata.get("template_sha256")
        template_ok = isinstance(template, str) and bool(re.fullmatch(r"[0-9a-f]{64}", template))
        case_passed = color_ok and media_ok and template_ok
        passed = passed and case_passed
        cases.append(
            {
                "expected": expected,
                "answer": result.text,
                "color_ok": color_ok,
                "single_image_evidence": media_ok,
                "template_sha256": template,
                "image_pixel_sha256": pixel_sha256(model_input.image),
                "passed": case_passed,
            }
        )
    return {"protocol": COLOR_CANARY_PROTOCOL, "passed": passed, "cases": cases}


def _capacity_candidates(backend: str = "vllm") -> tuple[int, ...]:
    if backend == "openrouter":
        variable = "QSPATIAL_API_CAPACITY_CANDIDATES"
        default = "8,4,2,1"
    else:
        variable = "QSPATIAL_CAPACITY_CANDIDATES"
        default = "32,16,8,4,2,1"
    raw = os.environ.get(variable, default)
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values) or tuple(sorted(set(values), reverse=True)) != values:
        raise ValueError(f"{variable} must be unique positive descending integers")
    return values


def probe_capacity(adapter: BoundAdapter, *, backend: str = "vllm") -> dict[str, Any]:
    if not adapter.supports_concurrency:
        return {"passed": True, "selected_concurrency": 1, "attempts": [{"candidate": 1, "passed": True}]}
    expected, model_input = _color_canary_specs()[0]
    attempts: list[dict[str, Any]] = []
    for candidate in _capacity_candidates(backend):
        try:
            with ThreadPoolExecutor(max_workers=candidate) as executor:
                results = list(executor.map(lambda _unused: adapter.generate(model_input), range(candidate)))
            passed = all(expected in result.text.casefold() for result in results)
            attempts.append({"candidate": candidate, "passed": passed, "num_results": len(results)})
            if passed:
                return {"passed": True, "selected_concurrency": candidate, "attempts": attempts}
        except Exception as exc:  # noqa: BLE001 - capacity probing records and lowers concurrency.
            attempts.append({"candidate": candidate, "passed": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
    raise RuntimeError(f"No stable Q-Spatial concurrency candidate: {attempts}")


def _journal_input_gate(path: Path, profile: QSpatialProfile) -> dict[str, Any]:
    successes: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("status") == "success":
                successes[int(event["index"])] = event
    errors: list[str] = []
    if set(successes) != set(SMOKE8_INDICES):
        errors.append("journal success indices differ from smoke8")
    for index, event in successes.items():
        audit = event.get("audit") if isinstance(event.get("audit"), dict) else {}
        generation = event.get("generation") if isinstance(event.get("generation"), dict) else {}
        if audit.get("image_count") != 1 or audit.get("system_prompt") != STANDARD_SYSTEM_PROMPT:
            errors.append(f"index {index} lacks exact one-image/Standard-Prompt audit")
        if audit.get("user_prompt") != audit.get("question"):
            errors.append(f"index {index} user prompt differs from runtime question alias")
        expected_roles = ["system", "user"] if profile.system_role_supported else ["user"]
        if audit.get("prompt_roles") != expected_roles:
            errors.append(f"index {index} prompt roles differ from the profile transport")
        if generation.get("num_media_prompt") != 1 and generation.get("num_model_image_tensors") != 1:
            errors.append(f"index {index} lacks one-image model-boundary evidence")
        if not re.fullmatch(r"[0-9a-f]{64}", str(generation.get("template_sha256") or "")):
            errors.append(f"index {index} lacks a template digest")
    return {"passed": not errors, "success_indices": sorted(successes), "errors": errors}


def _enrich_metadata(
    metadata_path: Path,
    *,
    contract: QSpatialTestContract,
    configuration: ResolvedConfiguration,
    binding_value: dict[str, Any],
    test_gate: Path | None,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["inference_protocol"] = configuration.profile.inference_protocol
    metadata["scorer_protocol"] = SCORER_PROTOCOL
    metadata["dataset"].update(
        {
            "benchmark": "Q-Spatial Bench",
            "parquet_root": str(contract.parquet_root),
            "scannet_rgb_root": str(contract.scannet_rgb_root),
            "revision": DATASET_REVISION,
            "fingerprint": contract.dataset_fingerprint,
            "files": {item.name: item.sha256 for item in DATASET_FILES},
        }
    )
    metadata["prompt"] = {
        "system_prompt_sha256": STANDARD_SYSTEM_PROMPT_SHA256,
        "user_template": "Question: {question}",
        "system_role_supported": configuration.profile.system_role_supported,
    }
    metadata["binding"] = binding_value
    metadata["binding_digest"] = _digest(binding_value)
    metadata["test_gate"] = str(test_gate) if test_gate else None
    atomic_write_json(metadata_path, metadata)
    return metadata


def test_gate_errors(gate: dict[str, Any], expected_binding_digest: str) -> list[str]:
    errors: list[str] = []
    if not gate.get("passed"):
        errors.append("gate did not pass")
    if gate.get("binding_digest") != expected_binding_digest:
        errors.append("binding digest differs")
    if gate.get("dataset_fingerprint") != (gate.get("binding") or {}).get("dataset", {}).get("fingerprint"):
        errors.append("dataset fingerprint is internally inconsistent")
    if gate.get("vision_canary", {}).get("passed") is not True:
        errors.append("vision canary did not pass")
    if gate.get("smoke_validation", {}).get("passed") is not True:
        errors.append("smoke subset validator did not pass")
    if gate.get("input_audit_gate", {}).get("passed") is not True:
        errors.append("smoke input audit did not pass")
    if gate.get("processor_audit", {}).get("passed") is not True:
        errors.append("processor/template audit did not pass")
    if not isinstance(gate.get("selected_concurrency"), int) or gate["selected_concurrency"] <= 0:
        errors.append("selected concurrency is missing or invalid")
    return errors


def _rotate_stale_test_artifacts(track: Path, expected_binding_digest: str) -> Path | None:
    """Preserve an invalid completed gate before starting a fresh test signature."""

    artifact_root = track / "test_artifacts"
    gate_path = track / "test_gate.json"
    if not artifact_root.exists() or not gate_path.is_file():
        return None
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        gate = {}
    if isinstance(gate, dict) and not test_gate_errors(gate, expected_binding_digest):
        return None
    old_digest = str(gate.get("binding_digest", "unknown"))[:12] if isinstance(gate, dict) else "unknown"
    suffix = f"stale-{old_digest}-{time.time_ns()}"
    archived_artifacts = track / f"test_artifacts.{suffix}"
    archived_gate = track / f"test_gate.{suffix}.json"
    artifact_root.rename(archived_artifacts)
    gate_path.rename(archived_gate)
    return archived_artifacts


def run_test_stage(
    profile: QSpatialProfile,
    contract: QSpatialTestContract,
    output_root: Path,
) -> Path:
    gpu_audit = inspect_local_gpus(profile, profile.default_backend)
    configuration = resolve_configuration(profile)
    track = track_directory(output_root, profile)
    binding_value = binding(configuration, contract)
    binding_digest = _digest(binding_value)
    _rotate_stale_test_artifacts(track, binding_digest)
    artifact_root = track / "test_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset_manifest = contract.dataset_manifest(include_images=True)
    atomic_write_json(artifact_root / "dataset_manifest.json", dataset_manifest)
    if configuration.backend != profile.default_backend:
        gpu_audit = inspect_local_gpus(profile, configuration.backend)
    atomic_write_json(artifact_root / "gpu_preflight.json", gpu_audit)
    adapters = [
        build_adapter(configuration, endpoint_index)
        for endpoint_index in range(max(1, len(configuration.base_urls)))
    ]
    adapter = adapters[0]
    try:
        canary_reports = [_canary_report(value) for value in adapters]
        canary = {
            "protocol": COLOR_CANARY_PROTOCOL,
            "passed": all(report["passed"] for report in canary_reports),
            "endpoints": [
                {"endpoint_index": index, **report}
                for index, report in enumerate(canary_reports)
            ],
        }
        atomic_write_json(artifact_root / "vision_canary.json", canary)
        if not canary["passed"]:
            raise RuntimeError(f"Q-Spatial vision canary failed for {profile.key}")
        capacity_reports = [
            probe_capacity(value, backend=configuration.backend) for value in adapters
        ]
        capacity = {
            "passed": all(report["passed"] for report in capacity_reports),
            "selected_concurrency": min(
                int(report["selected_concurrency"]) for report in capacity_reports
            ),
            "endpoints": [
                {"endpoint_index": index, **report}
                for index, report in enumerate(capacity_reports)
            ],
        }
        atomic_write_json(artifact_root / "capacity_probe.json", capacity)
        for extra in adapters[1:]:
            extra.close()
        smoke_output = artifact_root / "smoke8" / "predictions.jsonl"
        smoke_metadata = run_recoverable_inference(
            contract=ProfiledContract(contract),
            adapter=adapter,
            output=smoke_output,
            target_indices=list(SMOKE8_INDICES),
            benchmark="Q-Spatial Bench",
            split="test",
            official_size=OFFICIAL_TEST_SIZE,
            scorer_protocol=SCORER_PROTOCOL,
            workers=int(capacity["selected_concurrency"]),
            **_runtime_retry_policy(configuration.backend),
        )
    except BaseException:
        for value in adapters:
            value.close()
        raise
    smoke_metadata = _enrich_metadata(
        smoke_output.with_suffix(smoke_output.suffix + ".metadata.json"),
        contract=contract,
        configuration=configuration,
        binding_value=binding_value,
        test_gate=None,
    )
    smoke_rows = read_jsonl(smoke_output)
    smoke_validation = validate_prediction_rows(
        smoke_rows, contract, prediction_path=smoke_output, allow_subset=True
    )
    smoke_validation["expected_indices"] = list(SMOKE8_INDICES)
    smoke_validation["exact_smoke_indices"] = [row["index"] for row in smoke_rows] == sorted(SMOKE8_INDICES)
    smoke_validation["passed"] = smoke_validation["passed"] and smoke_validation["exact_smoke_indices"]
    smoke_validation["diagnostic_parse_status"] = {
        str(row["index"]): parse_measurement(row["raw_prediction"]).status for row in smoke_rows
    }
    atomic_write_json(artifact_root / "smoke8_validation.json", smoke_validation)
    input_gate = _journal_input_gate(Path(smoke_metadata["journal"]), profile)
    atomic_write_json(artifact_root / "input_audit_gate.json", input_gate)
    processor = configuration.processor_audit or {
        "passed": True,
        "profile": profile.key,
        "transport": configuration.backend,
        "system_role_supported": profile.system_role_supported,
        "system_prompt_sha256": STANDARD_SYSTEM_PROMPT_SHA256,
        "single_image_contract": True,
    }
    atomic_write_json(artifact_root / "processor_audit.json", processor)
    gate = {
        "schema_version": 1,
        "profile": profile.key,
        "passed": bool(
            canary["passed"]
            and capacity["passed"]
            and smoke_validation["passed"]
            and input_gate["passed"]
            and processor["passed"]
        ),
        "dataset_fingerprint": contract.dataset_fingerprint,
        "binding": binding_value,
        "binding_digest": binding_digest,
        "selected_concurrency": int(capacity["selected_concurrency"]),
        "vision_canary": canary,
        "smoke_validation": smoke_validation,
        "input_audit_gate": input_gate,
        "processor_audit": processor,
        "gpu_preflight": gpu_audit,
        "smoke_metadata": smoke_metadata,
        "generated_at": utc_now(),
    }
    gate_path = track / "test_gate.json"
    atomic_write_json(gate_path, gate)
    if not gate["passed"]:
        raise RuntimeError(f"Q-Spatial test gate failed for {profile.key}")
    print(f"[q-spatial] test gate passed: {gate_path}")
    return gate_path


def run_full_stage(
    profile: QSpatialProfile,
    contract: QSpatialTestContract,
    output_root: Path,
) -> Path:
    track = track_directory(output_root, profile)
    gate_path = track / "test_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"Run --stage test first; missing gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gpu_audit = inspect_local_gpus(profile, profile.default_backend)
    configuration = resolve_configuration(profile)
    binding_value = binding(configuration, contract)
    problems = test_gate_errors(gate, _digest(binding_value))
    if problems:
        raise ValueError("Q-Spatial test gate is stale or incomplete: " + "; ".join(problems))
    capacity = int(gate["selected_concurrency"])
    if configuration.backend != profile.default_backend:
        gpu_audit = inspect_local_gpus(profile, configuration.backend)
    atomic_write_json(track / "full_gpu_preflight.json", gpu_audit)
    output = track / "predictions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = run_recoverable_inference(
        contract=ProfiledContract(contract),
        adapter=build_adapter(configuration),
        output=output,
        target_indices=list(range(OFFICIAL_TEST_SIZE)),
        benchmark="Q-Spatial Bench",
        split="test",
        official_size=OFFICIAL_TEST_SIZE,
        scorer_protocol=SCORER_PROTOCOL,
        workers=capacity,
        **_runtime_retry_policy(configuration.backend),
    )
    metadata = _enrich_metadata(
        output.with_suffix(output.suffix + ".metadata.json"),
        contract=contract,
        configuration=configuration,
        binding_value=binding_value,
        test_gate=gate_path,
    )
    rows = read_jsonl(output)
    validation = validate_prediction_rows(rows, contract, prediction_path=output, allow_subset=False)
    if not validation["passed"] or not metadata.get("publishable_inference"):
        raise RuntimeError("Q-Spatial full inference did not pass its mandatory validator")
    metadata.setdefault("runtime", {})["gpu_preflight"] = gpu_audit
    atomic_write_json(output.with_suffix(output.suffix + ".metadata.json"), metadata)
    atomic_write_json(track / "prediction_validation.json", validation)
    print(f"[q-spatial] full-{OFFICIAL_TEST_SIZE} validation passed: {output}")
    return output


def _selected_profiles(args: argparse.Namespace) -> list[QSpatialProfile]:
    if args.all:
        return [PROFILES[key] for key in PROFILE_SEQUENCE]
    keys: list[str] = []
    if args.model:
        keys.append(args.model)
    if args.models:
        keys.extend(value.strip() for value in args.models.split(",") if value.strip())
    if not keys:
        raise ValueError("Select --model, --models, or --all")
    if len(keys) != len(set(keys)):
        raise ValueError("Selected Q-Spatial profiles contain duplicates")
    return ordered_profiles(keys)


def _status(profile: QSpatialProfile, output_root: Path) -> dict[str, Any]:
    track = track_directory(output_root, profile)
    gate = track / "test_gate.json"
    prediction = track / "predictions.jsonl"
    summary = track / "scores" / SCORER_PROTOCOL / "summary.json"
    return {
        "profile": profile.key,
        "test_gate": "passed" if gate.is_file() and json.loads(gate.read_text()).get("passed") else "missing",
        "full_prediction": "present" if prediction.is_file() else "missing",
        "score": "present" if summary.is_file() else "missing",
        "track": str(track),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("test", "full"))
    parser.add_argument("--model")
    parser.add_argument("--models")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--skip-resource-blocked", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parquet-root", default=os.environ.get("QSPATIAL_PARQUET_ROOT"))
    parser.add_argument("--scannet-rgb-root", default=os.environ.get("QSPATIAL_SCANNET_RGB_ROOT"))
    parser.add_argument("--output-root", default=os.environ.get("QSPATIAL_OUTPUT_ROOT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for key in PROFILE_SEQUENCE:
            profile = PROFILES[key]
            print(
                f"{profile.key}\t{profile.display_name}\t{profile.input_profile}\t"
                f"{profile.default_backend}\tTP={profile.default_tensor_parallel_size}"
            )
        return
    if args.status:
        if not args.output_root:
            raise ValueError("Set QSPATIAL_OUTPUT_ROOT or pass --output-root")
        for key in PROFILE_SEQUENCE:
            print(json.dumps(_status(PROFILES[key], Path(args.output_root)), ensure_ascii=False))
        return
    profiles = _selected_profiles(args)
    if args.dry_run:
        for profile in profiles:
            print(f"{args.stage or 'check'}\t{profile.key}\t{profile.inference_protocol}")
        return
    if not args.parquet_root or not args.scannet_rgb_root:
        raise ValueError("Set QSPATIAL_PARQUET_ROOT and QSPATIAL_SCANNET_RGB_ROOT")
    contract = QSpatialTestContract(args.parquet_root, args.scannet_rgb_root)
    contract.dataset_manifest(include_images=True)
    if args.check:
        for profile in profiles:
            preliminary_gpu = inspect_local_gpus(profile, profile.default_backend)
            configuration = resolve_configuration(profile)
            gpu = (
                preliminary_gpu
                if configuration.backend == profile.default_backend
                else inspect_local_gpus(profile, configuration.backend)
            )
            print(
                json.dumps(
                    {
                        "profile": profile.key,
                        "binding": binding(configuration, contract),
                        "gpu_preflight": gpu,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return
    if not args.stage:
        raise ValueError("Pass --stage test or --stage full")
    if not args.output_root:
        raise ValueError("Set QSPATIAL_OUTPUT_ROOT or pass --output-root")
    output_root = Path(args.output_root).resolve()
    status_path = output_root / "_batch" / "status.tsv"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        try:
            if args.stage == "test":
                run_test_stage(profile, contract, output_root)
            else:
                run_full_stage(profile, contract, output_root)
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{utc_now()}\t{args.stage.upper()}_COMPLETE\t{profile.key}\n")
        except ResourceBlockedError as exc:
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{utc_now()}\tBLOCKED_RESOURCE\t{profile.key}\t{exc}\n")
            if not args.skip_resource_blocked:
                raise
            print(f"[q-spatial] BLOCKED_RESOURCE {profile.key}: {exc}")


if __name__ == "__main__":
    main()
