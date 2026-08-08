"""Two-stage, publication-gated inference orchestration for SPBench-SI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
from typing import Any, Sequence

from PIL import Image

from ...models.common.runtime import (
    JOURNAL_SCHEMA_VERSION,
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
    openai_compatible_model_ids,
    single_image_user_message,
)
from .command_adapter import UpstreamCommandAdapter, fold_system_user_prompt, load_generation_manifest
from .data import (
    DATASET_REVISION,
    IMAGES_ARCHIVE_SHA256,
    MCQ_DIRECT_SUFFIX,
    NUMERIC_DIRECT_SUFFIX,
    OFFICIAL_TEST_SIZE,
    PARQUET_SHA256,
    SMOKE8_INDICES,
    SPBenchSIModelInput,
    SPBenchSITestContract,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
)
from .prediction_validation import read_jsonl, validate_prediction_rows
from .processor_audit import audit_processor
from .profiles import (
    PROFILE_SEQUENCE,
    PROFILES,
    SPATIALLADDER_BATCH_PADDING_SIDE,
    SPBenchSIProfile,
    ordered_profiles,
)
from .scorer import SCORER_PROTOCOL, score_main_row

COLOR_CANARY_PROTOCOL = "spbench_si_pure_red_blue_512_single_rgb_v1"
REMOTE_API_BACKENDS = frozenset({"openrouter", "packyapi"})
PACKYAPI_DEFAULT_BASE_URL = "https://www.packyapi.com/v1"
PACKYAPI_GEMINI_PROFILE = "gemini31pro_openrouter_non_zdr"


class ResourceBlockedError(RuntimeError):
    """A fail-closed resource gate; no existing process is managed or terminated."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _env_token(profile_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", profile_key).upper()


def _revision_tag(revision: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", revision).strip("-")


def track_directory(output_root: str | Path, profile: SPBenchSIProfile) -> Path:
    return Path(output_root).resolve() / "runs" / profile.key / _revision_tag(profile.revision) / profile.inference_protocol


class ProfiledContract:
    def __init__(self, contract: SPBenchSITestContract) -> None:
        self.contract = contract
        self.dataset_root = contract.dataset_root
        self.dataset_fingerprint = contract.dataset_fingerprint

    def __len__(self) -> int:
        return len(self.contract)

    def model_input(self, index: int) -> SPBenchSIModelInput:
        return self.contract.model_input(index)

    def model_inputs(self, indices: Sequence[int]) -> list[SPBenchSIModelInput]:
        return [self.model_input(index) for index in indices]

    def prediction_row(self, index: int, prediction: str) -> dict[str, Any]:
        return self.contract.prediction_row(index, prediction)


class SingleStageTransportAdapter(InferenceAdapter):
    """Use one native template call; fold system bytes only for checkpoints without a system role."""

    batch_size = 1
    supports_concurrency = True

    def __init__(self, base: OpenAICompatibleAdapter, profile: SPBenchSIProfile) -> None:
        self.base = base
        self.profile = profile

    def metadata(self) -> dict[str, Any]:
        value = dict(self.base.metadata())
        value["system_role_supported"] = self.profile.system_role_supported
        value["system_transport"] = self.profile.system_transport
        value["single_stage"] = True
        return value

    def generate(self, model_input: SPBenchSIModelInput) -> GenerationResult:
        if self.profile.system_role_supported:
            return self.base.generate(model_input)
        folded = fold_system_user_prompt(model_input.system_prompt, model_input.user_prompt)
        messages = single_image_user_message(folded, image_to_png_data_uri(model_input.image))
        return self.base.generate_messages(model_input, messages)

    def close(self) -> None:
        self.base.close()


class BoundAdapter(InferenceAdapter):
    def __init__(
        self,
        delegate: InferenceAdapter,
        *,
        profile: SPBenchSIProfile,
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
        metadata.update({
            "model": self.profile.model,
            "model_revision": self.profile.revision,
            "backend": metadata.get("backend", self.profile.default_backend),
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "comparison_group": self.profile.comparison_group,
            "inference_protocol": self.profile.inference_protocol,
            "chat_template": self.profile.chat_template,
            "system_role_supported": self.profile.system_role_supported,
            "system_transport": self.profile.system_transport,
            "image_processing": dict(self.profile.image_processing),
            "decoding": dict(self.profile.decoding),
            "seed_strategy": self.profile.seed_strategy,
            "provider_nondeterministic": self.profile.provider_nondeterministic,
            "adapter_digest": self.adapter_digest,
            "processor_audit": self.processor_audit,
            "known_deviation": self.profile.known_deviation,
        })
        metadata.setdefault("upstream", {
            "repository": self.profile.upstream_url or self.profile.model,
            "commit": self.profile.upstream_commit,
        })
        return metadata

    def _bound_result(self, model_input: SPBenchSIModelInput, result: GenerationResult) -> GenerationResult:
        generation = dict(result.metadata)
        generation.setdefault("num_media_prompt", 1)
        generation.setdefault("source_rgb_count", 1)
        generation.setdefault("source_rgb_sha256", pixel_sha256(model_input.image))
        generation.setdefault("system_role_supported", self.profile.system_role_supported)
        generation.setdefault("system_prompt_sha256", hashlib.sha256(model_input.system_prompt.encode()).hexdigest())
        generation.setdefault("user_prompt_sha256", hashlib.sha256(model_input.user_prompt.encode()).hexdigest())
        generation.setdefault("template_sha256", _digest({
            "chat_template": self.profile.chat_template,
            "system_transport": self.profile.system_transport,
            "system_prompt": model_input.system_prompt,
            "user_prompt": model_input.user_prompt,
            "image_pixel_sha256": pixel_sha256(model_input.image),
            "media_count": 1,
        }))
        if self.profile.comparison_group != "rgb_only":
            generation.setdefault("derived_from_source_rgb_sha256", pixel_sha256(model_input.image))
        return GenerationResult(result.text, generation, tuple(result.warnings))

    def generate(self, model_input: SPBenchSIModelInput) -> GenerationResult:
        return self._bound_result(model_input, self.delegate.generate(model_input))

    def generate_batch(self, model_inputs: Sequence[SPBenchSIModelInput]) -> list[GenerationResult]:
        results = self.delegate.generate_batch(list(model_inputs))
        if len(results) != len(model_inputs):
            raise ValueError("Delegate returned the wrong batch size")
        return [self._bound_result(value, result) for value, result in zip(model_inputs, results)]

    def set_batch_size(self, value: int) -> None:
        self.batch_size = int(value)
        if hasattr(self.delegate, "batch_size"):
            self.delegate.batch_size = int(value)

    def close(self) -> None:
        self.delegate.close()


@dataclass(frozen=True, slots=True)
class ResolvedConfiguration:
    profile: SPBenchSIProfile
    backend: str
    base_urls: tuple[str, ...]
    decoding: dict[str, Any]
    adapter_digest: str
    command: str | None
    processor_audit: dict[str, Any] | None
    served_model_name: str | None = None

    @property
    def endpoint_identity(self) -> list[str]:
        return [hashlib.sha256(value.encode()).hexdigest() for value in self.base_urls]


def _open_adapter_digest(profile: SPBenchSIProfile) -> str:
    from ...models.openai_compatible import client
    from . import profiles
    files = [Path(inspect.getfile(client)), Path(__file__), Path(profiles.__file__)]
    return _digest({
        "profile_registry_digest": profile.registry_digest,
        "files": {path.name: _file_digest(path) for path in files},
    })


def _profile_env(profile: SPBenchSIProfile, suffix: str) -> str | None:
    return os.environ.get(f"SPBENCH_SI_{_env_token(profile.key)}_{suffix}")


def _configured_gpu_ids(profile: SPBenchSIProfile) -> tuple[int, ...]:
    raw = _profile_env(profile, "GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw.strip():
        return ()
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ValueError(f"Invalid GPU selection for {profile.key}: {raw!r}")
    return values


def inspect_local_gpus(profile: SPBenchSIProfile, backend: str) -> dict[str, Any]:
    if backend in REMOTE_API_BACKENDS:
        return {"applicable": False, "reason": "remote API backend"}
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise FileNotFoundError("Local SPBench-SI inference requires nvidia-smi for read-only preflight")
    query = subprocess.run([
        executable, "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    inventory: list[dict[str, Any]] = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",", 5)]
        if len(fields) != 6:
            raise ValueError(f"Unexpected nvidia-smi inventory row: {line!r}")
        inventory.append({
            "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "memory_total_mib": int(fields[3]), "memory_free_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
        })
    selected = _configured_gpu_ids(profile)
    expected = profile.default_tensor_parallel_size if backend == "vllm" else 1
    if len(selected) != expected:
        raise ResourceBlockedError(
            f"{profile.key} requires {expected} explicit GPU id(s); configure "
            f"SPBENCH_SI_{_env_token(profile.key)}_GPU_IDS"
        )
    by_index = {item["index"]: item for item in inventory}
    if any(index not in by_index for index in selected):
        raise ValueError(f"Configured GPUs are absent for {profile.key}: {selected}")
    if profile.key == "internvl3_78b":
        if len(selected) != 4:
            raise ResourceBlockedError("InternVL3-78B requires exactly four GPUs; quantized two-GPU substitution is forbidden")
        undersized = [index for index in selected if by_index[index]["memory_total_mib"] < 79_000]
        if undersized:
            raise ResourceBlockedError(f"InternVL3-78B requires four 80GB GPUs; undersized={undersized}")
    process_query = subprocess.run([
        executable, "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], check=False, capture_output=True, text=True)
    process_lines = [line for line in process_query.stdout.splitlines() if line.strip()]
    selected_uuids = {by_index[index]["uuid"] for index in selected}
    occupied = [line for line in process_lines if line.split(",", 1)[0].strip() in selected_uuids]
    if backend == "upstream_transformers" and occupied:
        raise ResourceBlockedError(
            f"{profile.key} selected GPU already has compute processes; existing processes were untouched"
        )
    if backend == "upstream_transformers":
        insufficient = [index for index in selected if by_index[index]["memory_free_mib"] < profile.min_free_gpu_mib]
        busy = [index for index in selected if by_index[index]["utilization_percent"] > 10]
        if insufficient or busy:
            raise ResourceBlockedError(f"{profile.key} GPU preflight failed: insufficient={insufficient}, busy={busy}")
    return {
        "applicable": True, "selected_gpu_ids": list(selected), "inventory": inventory,
        "compute_processes": process_lines, "policy": "read-only; no process is terminated or adopted",
    }


def _resolved_command_configuration(profile: SPBenchSIProfile) -> ResolvedConfiguration:
    command = _profile_env(profile, "COMMAND")
    adapter_digest = _profile_env(profile, "ADAPTER_DIGEST")
    if not command or not adapter_digest:
        token = _env_token(profile.key)
        raise ValueError(f"Set SPBENCH_SI_{token}_COMMAND and SPBENCH_SI_{token}_ADAPTER_DIGEST")
    decoding = load_generation_manifest(profile, _profile_env(profile, "GENERATION_MANIFEST"))
    return ResolvedConfiguration(profile, "upstream_transformers", (), decoding, adapter_digest, command, None)


def resolve_configuration(profile: SPBenchSIProfile) -> ResolvedConfiguration:
    backend = _profile_env(profile, "BACKEND") or profile.default_backend
    if profile.adapter_kind == "upstream_command":
        return _resolved_command_configuration(profile)
    if backend not in {"vllm", *REMOTE_API_BACKENDS}:
        raise ValueError(f"Unsupported backend {backend!r} for {profile.key}")
    processor_report: dict[str, Any] | None = None
    if backend == "vllm":
        _vllm_runtime_version()
        model_path = os.environ.get(profile.model_path_env, "")
        if not model_path:
            raise ValueError(f"Set {profile.model_path_env} for processor/template audit")
        processor_report = audit_processor(profile, model_path)
        raw_urls = _profile_env(profile, "BASE_URLS") or os.environ.get("SPBENCH_SI_VLLM_BASE_URLS", "")
        urls = tuple(value.strip().rstrip("/") for value in raw_urls.split(",") if value.strip())
        if len(urls) != 1:
            raise ValueError(f"{profile.key} requires one vLLM endpoint")
        served_model_name = profile.served_model_name
    elif backend == "openrouter":
        urls = (os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),)
        served_model_name = None
    else:
        if profile.key != PACKYAPI_GEMINI_PROFILE:
            raise ValueError("PackyAPI is approved only as a Gemini 3.1 Pro quota source")
        api_key = os.environ.get("PACKYAPI_API_KEY", "")
        if not api_key:
            raise ValueError("Set PACKYAPI_API_KEY; keys are never accepted on the command line")
        urls = (os.environ.get("PACKYAPI_BASE_URL", PACKYAPI_DEFAULT_BASE_URL).rstrip("/"),)
        configured_model = os.environ.get("SPBENCH_SI_PACKYAPI_MODEL_ID", "").strip()
        catalog = openai_compatible_model_ids(
            base_url=urls[0],
            api_key=api_key,
            timeout=float(os.environ.get("SPBENCH_SI_PACKYAPI_CATALOG_TIMEOUT", "30")),
        )
        exact_candidates = tuple(
            model_id for model_id in catalog
            if re.search(r"(?:^|/)gemini-3\.1-pro(?:-preview)?(?:-20260219)?$", model_id, re.I)
        )
        documented_aliases = tuple(
            model_id for model_id in catalog
            if re.search(r"(?:^|/)gemini-3-pro-preview$", model_id, re.I)
        )
        candidates = exact_candidates or documented_aliases
        if configured_model:
            if configured_model not in catalog:
                raise ValueError(
                    f"Configured PackyAPI model {configured_model!r} is absent from the authenticated /models catalog"
                )
            if configured_model not in {*exact_candidates, *documented_aliases}:
                raise ValueError(
                    f"Configured PackyAPI model {configured_model!r} is not a Gemini 3.1 Pro id "
                    "or the Packy-documented gemini-3-pro-preview route alias"
                )
            served_model_name = configured_model
        elif len(candidates) == 1:
            served_model_name = candidates[0]
        else:
            raise ValueError(
                "Authenticated PackyAPI Gemini-slb catalog must expose exactly one preferred Gemini 3.1 Pro "
                "id, or one documented gemini-3-pro-preview route alias when no 3.1 id exists; "
                f"exact={list(exact_candidates)!r}, aliases={list(documented_aliases)!r}"
            )
    return ResolvedConfiguration(
        profile, backend, urls, dict(profile.decoding), _open_adapter_digest(profile), None,
        processor_report, served_model_name,
    )


def _request_timeout_seconds(backend: str) -> float:
    variable = "SPBENCH_SI_VLLM_API_TIMEOUT" if backend == "vllm" else "SPBENCH_SI_API_TIMEOUT"
    return float(os.environ.get(variable, "600" if backend == "vllm" else "180"))


def _vllm_runtime_version() -> str:
    configured = os.environ.get("SPBENCH_SI_VLLM_RUNTIME_VERSION", "").strip()
    if configured:
        version = configured
    else:
        try:
            version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(
                "Set SPBENCH_SI_VLLM_RUNTIME_VERSION to the exact audited vLLM server version"
            ) from exc
    if not re.fullmatch(r"0\.19(?:\.\d+)?(?:[+.-][A-Za-z0-9.]+)?", version):
        raise ValueError(f"SPBench-SI requires vLLM 0.19.x, got {version!r}")
    return version


def _runtime_retry_policy(backend: str) -> dict[str, int]:
    if backend == "vllm":
        return {
            "retries": int(os.environ.get("SPBENCH_SI_VLLM_INFERENCE_RETRIES", "0")),
            "retry_missing_passes": int(os.environ.get("SPBENCH_SI_VLLM_RETRY_MISSING_PASSES", "1")),
        }
    return {
        "retries": int(os.environ.get("SPBENCH_SI_INFERENCE_RETRIES", "2")),
        "retry_missing_passes": 1 if backend in REMOTE_API_BACKENDS else 0,
    }


def build_adapter(configuration: ResolvedConfiguration, *, batch_size: int = 1) -> BoundAdapter:
    profile = configuration.profile
    if configuration.command:
        delegate: InferenceAdapter = UpstreamCommandAdapter(
            profile=profile, command=configuration.command, adapter_digest=configuration.adapter_digest,
            decoding=configuration.decoding, batch_size=batch_size,
        )
    else:
        if configuration.backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
        elif configuration.backend == "packyapi":
            api_key = os.environ.get("PACKYAPI_API_KEY", "")
        else:
            api_key = os.environ.get("VLLM_API_KEY", "local") or "local"
        if not api_key:
            raise ValueError(f"Set the API key for {configuration.backend}; keys are never accepted on the command line")
        base = OpenAICompatibleAdapter(
            profile=profile, backend=configuration.backend, base_url=configuration.base_urls[0],
            api_key=api_key, served_model_name=configuration.served_model_name or profile.served_model_name,
            timeout=_request_timeout_seconds(configuration.backend),
            metadata_retries=int(os.environ.get("SPBENCH_SI_OPENROUTER_METADATA_RETRIES", "10")),
            policy_key=profile.api_policy_key, image_source="SPBench-SI locked ZIP RGB only",
        )
        delegate = SingleStageTransportAdapter(base, profile)
    return BoundAdapter(delegate, profile=profile, adapter_digest=configuration.adapter_digest, processor_audit=configuration.processor_audit)


def _capacity_candidates(backend: str = "vllm") -> tuple[int, ...]:
    variable = "SPBENCH_SI_API_CAPACITY_CANDIDATES" if backend in REMOTE_API_BACKENDS else "SPBENCH_SI_CAPACITY_CANDIDATES"
    default = "8,4,2,1" if backend in REMOTE_API_BACKENDS else "32,16,8,4,2,1"
    values = tuple(int(item.strip()) for item in os.environ.get(variable, default).split(",") if item.strip())
    if not values or any(value <= 0 for value in values) or tuple(sorted(set(values), reverse=True)) != values:
        raise ValueError(f"{variable} must be unique positive descending integers")
    return values


def _spatialladder_batch_candidates() -> tuple[int, ...]:
    raw = os.environ.get("SPBENCH_SI_SPATIALLADDER_BATCH_CANDIDATES", "16,8,4,2,1")
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or tuple(sorted(set(values), reverse=True)) != values or any(value <= 0 for value in values):
        raise ValueError("SPBENCH_SI_SPATIALLADDER_BATCH_CANDIDATES must descend uniquely")
    return values


def binding(
    configuration: ResolvedConfiguration,
    contract: SPBenchSITestContract,
    gpu_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "dataset": {
            "revision": DATASET_REVISION, "fingerprint": contract.dataset_fingerprint,
            "parquet_sha256": PARQUET_SHA256, "images_archive_sha256": IMAGES_ARCHIVE_SHA256,
            "official_test_size": OFFICIAL_TEST_SIZE,
        },
        "prompt": {
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "mcq_suffix_sha256": hashlib.sha256(MCQ_DIRECT_SUFFIX.encode()).hexdigest(),
            "numeric_suffix_sha256": hashlib.sha256(NUMERIC_DIRECT_SUFFIX.encode()).hexdigest(),
            "system_transport": configuration.profile.system_transport,
        },
        "profile": {
            "key": configuration.profile.key, "registry_digest": configuration.profile.registry_digest,
            "model": configuration.profile.model, "revision": configuration.profile.revision,
            "inference_protocol": configuration.profile.inference_protocol,
            "input_profile": configuration.profile.input_profile,
        },
        "runtime": {
            "backend": configuration.backend,
            "endpoint_sha256": configuration.endpoint_identity,
            "served_model_name": configuration.served_model_name,
            "tensor_parallel_size": configuration.profile.default_tensor_parallel_size,
            "selected_gpu_ids": (gpu_audit or {}).get("selected_gpu_ids"),
            "vllm_max_model_len": (
                int(os.environ.get("SPBENCH_SI_VLLM_MAX_MODEL_LEN", "32768"))
                if configuration.backend == "vllm" else None
            ),
            "vllm_runtime_version": (
                _vllm_runtime_version() if configuration.backend == "vllm" else None
            ),
        },
        "processor_audit": configuration.processor_audit,
        "adapter_digest": configuration.adapter_digest,
        "image_processing": configuration.profile.image_processing,
        "decoding": configuration.decoding,
        "seed_strategy": configuration.profile.seed_strategy,
        "capacity_candidates": (
            list(_spatialladder_batch_candidates()) if configuration.profile.native_batch_probe
            else list(_capacity_candidates(configuration.backend)) if configuration.backend in {"vllm", *REMOTE_API_BACKENDS}
            else [1]
        ),
        "smoke8_indices": list(SMOKE8_INDICES),
        "canary_protocol": COLOR_CANARY_PROTOCOL,
    }


def _color_inputs() -> list[tuple[str, SPBenchSIModelInput]]:
    prompt = "Question: What is the only color in this image?\n\nAnswer with one lowercase color word directly."
    return [
        (color, SPBenchSIModelInput(-offset - 1, Image.new("RGB", (512, 512), color), SYSTEM_PROMPT, prompt))
        for offset, color in enumerate(("red", "blue"))
    ]


def _native_batch_color_inputs(count: int) -> list[tuple[str, SPBenchSIModelInput]]:
    if count <= 0:
        raise ValueError("Native batch canary count must be positive")
    prompts = (
        "Question: What is the only color in this image?\n\n"
        "Answer with one lowercase color word directly.",
        "Question: Inspect the complete uniformly colored image carefully. "
        "After checking every visible pixel, what single color fills the image?\n\n"
        "Answer with one lowercase color word directly.",
    )
    combinations = (
        ("red", prompts[0]),
        ("blue", prompts[1]),
        ("blue", prompts[0]),
        ("red", prompts[1]),
    )
    cases: list[tuple[str, SPBenchSIModelInput]] = []
    for offset in range(count):
        color, prompt = combinations[offset % len(combinations)]
        cases.append((
            color,
            SPBenchSIModelInput(
                -1000 - offset,
                Image.new("RGB", (512, 512), color),
                SYSTEM_PROMPT,
                prompt,
            ),
        ))
    return cases


def _color_passed(text: str, expected: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.casefold()))
    other = "blue" if expected == "red" else "red"
    return expected in words and other not in words


def vision_canary(adapter: BoundAdapter) -> dict[str, Any]:
    cases = []
    for expected, model_input in _color_inputs():
        result = adapter.generate(model_input)
        cases.append({
            "expected": expected, "passed": _color_passed(result.text, expected),
            "raw_prediction": result.text, "source_rgb_sha256": pixel_sha256(model_input.image),
            "generation": result.metadata,
        })
    return {"protocol": COLOR_CANARY_PROTOCOL, "passed": all(case["passed"] for case in cases), "cases": cases}


def probe_capacity(adapter: BoundAdapter, *, backend: str) -> dict[str, Any]:
    expected, model_input = _color_inputs()[0]
    attempts: list[dict[str, Any]] = []
    if adapter.profile.native_batch_probe:
        candidates = _spatialladder_batch_candidates()
        for candidate in candidates:
            try:
                adapter.set_batch_size(candidate)
                cases = _native_batch_color_inputs(candidate)
                results = adapter.generate_batch([value for _color, value in cases])
                prompt_lengths = sorted({len(value.user_prompt) for _color, value in cases})
                padding_proved = len(results) == candidate and all(
                    result.metadata.get("tokenizer_padding_side")
                    == SPATIALLADDER_BATCH_PADDING_SIDE
                    for result in results
                )
                heterogeneous = candidate == 1 or len(prompt_lengths) > 1
                passed = (
                    len(results) == candidate
                    and heterogeneous
                    and padding_proved
                    and all(
                        _color_passed(result.text, color)
                        for (color, _value), result in zip(cases, results)
                    )
                )
                attempts.append({
                    "candidate": candidate,
                    "kind": "native_batch",
                    "passed": passed,
                    "prompt_character_lengths": prompt_lengths,
                    "heterogeneous_prompt_lengths": heterogeneous,
                    "tokenizer_padding_side": (
                        SPATIALLADDER_BATCH_PADDING_SIDE if padding_proved else None
                    ),
                })
                if passed:
                    return {
                        "passed": True,
                        "selected_capacity": candidate,
                        "capacity_kind": "native_batch",
                        "heterogeneous_prompt_lengths": heterogeneous,
                        "tokenizer_padding_side": SPATIALLADDER_BATCH_PADDING_SIDE,
                        "attempts": attempts,
                    }
            except Exception as exc:  # noqa: BLE001
                attempts.append({"candidate": candidate, "kind": "native_batch", "passed": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
        raise RuntimeError(f"No stable SpatialLadder native batch candidate: {attempts}")
    if not adapter.supports_concurrency:
        return {"passed": True, "selected_capacity": 1, "capacity_kind": "batch1", "attempts": [{"candidate": 1, "passed": True}]}
    for candidate in _capacity_candidates(backend):
        try:
            with ThreadPoolExecutor(max_workers=candidate) as executor:
                results = list(executor.map(lambda _unused: adapter.generate(model_input), range(candidate)))
            passed = all(_color_passed(result.text, expected) for result in results)
            attempts.append({"candidate": candidate, "kind": "request_concurrency", "passed": passed})
            if passed:
                return {"passed": True, "selected_capacity": candidate, "capacity_kind": "request_concurrency", "attempts": attempts}
        except Exception as exc:  # noqa: BLE001
            attempts.append({"candidate": candidate, "kind": "request_concurrency", "passed": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
    raise RuntimeError(f"No stable SPBench-SI capacity candidate: {attempts}")


def _journal_input_gate(path: Path, profile: SPBenchSIProfile) -> dict[str, Any]:
    successes: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                event = json.loads(line)
                if event.get("status") == "success":
                    successes[int(event["index"])] = event
    errors: list[str] = []
    if set(successes) != set(SMOKE8_INDICES):
        errors.append("journal success indices differ from smoke8")
    sizes = set()
    for index, event in successes.items():
        audit = event.get("audit") if isinstance(event.get("audit"), dict) else {}
        generation = event.get("generation") if isinstance(event.get("generation"), dict) else {}
        sizes.add(tuple(audit.get("image_size") or []))
        if audit.get("image_count") != 1 or audit.get("system_prompt") != SYSTEM_PROMPT:
            errors.append(f"index {index} lacks exact one-image/system-prompt audit")
        expected_roles = ["system", "user"] if profile.system_role_supported else ["user"]
        if audit.get("prompt_roles") != expected_roles:
            errors.append(f"index {index} prompt roles differ from locked transport")
        if generation.get("source_rgb_count") != 1 or not re.fullmatch(r"[0-9a-f]{64}", str(generation.get("template_sha256") or "")):
            errors.append(f"index {index} lacks model-boundary/template evidence")
    if len(sizes) < 2:
        errors.append("smoke8 does not prove the two locked source resolutions")
    return {"passed": not errors, "success_indices": sorted(successes), "image_sizes": [list(value) for value in sorted(sizes)], "errors": errors}


def _enrich_metadata(
    metadata_path: Path,
    *,
    contract: SPBenchSITestContract,
    configuration: ResolvedConfiguration,
    binding_value: dict[str, Any],
    test_gate: Path | None,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["inference_protocol"] = configuration.profile.inference_protocol
    metadata["scorer_protocol"] = SCORER_PROTOCOL
    metadata["dataset"].update({
        "benchmark": "SPBench-SI", "parquet": str(contract.parquet_path),
        "images_archive": str(contract.images_archive), "revision": DATASET_REVISION,
        "fingerprint": contract.dataset_fingerprint,
        "files": {"SPBench-SI.parquet": PARQUET_SHA256, "SPBench-SI-images.zip": IMAGES_ARCHIVE_SHA256},
    })
    metadata["prompt"] = {
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "user_template": "official default/direct: Question + optional Options + blank line + type suffix",
        "system_transport": configuration.profile.system_transport,
    }
    metadata["binding"] = binding_value
    metadata["binding_digest"] = _digest(binding_value)
    metadata["test_gate"] = str(test_gate) if test_gate else None
    atomic_write_json(metadata_path, metadata)
    return metadata


def test_gate_errors(gate: dict[str, Any], expected_binding_digest: str) -> list[str]:
    errors: list[str] = []
    checks = {
        "gate did not pass": gate.get("passed") is True,
        "binding digest differs": gate.get("binding_digest") == expected_binding_digest,
        "vision canary did not pass": (gate.get("vision_canary") or {}).get("passed") is True,
        "smoke subset validator did not pass": (gate.get("smoke_validation") or {}).get("passed") is True,
        "smoke input audit did not pass": (gate.get("input_audit_gate") or {}).get("passed") is True,
        "processor/template audit did not pass": (gate.get("processor_audit") or {}).get("passed") is True,
        "capacity is missing": isinstance(gate.get("selected_capacity"), int) and gate["selected_capacity"] > 0,
    }
    for message, passed in checks.items():
        if not passed:
            errors.append(message)
    if gate.get("profile") == "spatialladder3b_rgb":
        capacity = gate.get("capacity_probe") or {}
        processor = gate.get("processor_audit") or {}
        if capacity.get("passed") is not True:
            errors.append("SpatialLadder capacity probe evidence is missing")
        if capacity.get("tokenizer_padding_side") != SPATIALLADDER_BATCH_PADDING_SIDE:
            errors.append("SpatialLadder capacity probe did not prove left padding")
        if (
            gate.get("selected_capacity", 0) > 1
            and capacity.get("heterogeneous_prompt_lengths") is not True
        ):
            errors.append("SpatialLadder capacity probe did not exercise unequal prompt lengths")
        if processor.get("tokenizer_padding_side") != SPATIALLADDER_BATCH_PADDING_SIDE:
            errors.append("SpatialLadder processor audit did not prove left padding")
    return errors


def _rotate_stale_test_artifacts(track: Path, expected_binding_digest: str) -> Path | None:
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
    archived = track / f"test_artifacts.{suffix}"
    artifact_root.rename(archived)
    gate_path.rename(track / f"test_gate.{suffix}.json")
    return archived


def run_test_stage(profile: SPBenchSIProfile, contract: SPBenchSITestContract, output_root: Path) -> Path:
    gpu_audit = inspect_local_gpus(profile, profile.default_backend)
    configuration = resolve_configuration(profile)
    if configuration.backend != profile.default_backend:
        gpu_audit = inspect_local_gpus(profile, configuration.backend)
    binding_value = binding(configuration, contract, gpu_audit)
    track = track_directory(output_root, profile)
    gate_path = track / "test_gate.json"
    binding_digest = _digest(binding_value)
    if gate_path.is_file():
        existing = json.loads(gate_path.read_text(encoding="utf-8"))
        if not test_gate_errors(existing, binding_digest):
            print(f"[spbench-si] reusing current test gate: {gate_path}")
            return gate_path
    _rotate_stale_test_artifacts(track, binding_digest)
    artifact_root = track / "test_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(artifact_root / "dataset_manifest.json", contract.dataset_manifest(include_images=True))
    atomic_write_json(artifact_root / "gpu_preflight.json", gpu_audit)
    adapter = build_adapter(configuration)
    try:
        canary = vision_canary(adapter)
        atomic_write_json(artifact_root / "vision_canary.json", canary)
        if not canary["passed"]:
            raise RuntimeError(f"SPBench-SI vision canary failed for {profile.key}")
        capacity = probe_capacity(adapter, backend=configuration.backend)
        atomic_write_json(artifact_root / "capacity_probe.json", capacity)
        selected = int(capacity["selected_capacity"])
        if profile.native_batch_probe:
            adapter.set_batch_size(selected)
            workers = 1
        else:
            workers = selected
        smoke_output = artifact_root / "smoke8" / "predictions.jsonl"
        smoke_metadata = run_recoverable_inference(
            contract=ProfiledContract(contract), adapter=adapter, output=smoke_output,
            target_indices=list(SMOKE8_INDICES), benchmark="SPBench-SI", split="test",
            official_size=OFFICIAL_TEST_SIZE, scorer_protocol=SCORER_PROTOCOL, workers=workers,
            **_runtime_retry_policy(configuration.backend),
        )
    finally:
        adapter.close()
    smoke_metadata = _enrich_metadata(
        smoke_output.with_suffix(smoke_output.suffix + ".metadata.json"), contract=contract,
        configuration=configuration, binding_value=binding_value, test_gate=None,
    )
    smoke_rows = read_jsonl(smoke_output)
    smoke_validation = validate_prediction_rows(smoke_rows, contract, prediction_path=smoke_output, allow_subset=True)
    smoke_validation["expected_indices"] = list(SMOKE8_INDICES)
    smoke_validation["exact_smoke_indices"] = [row["index"] for row in smoke_rows] == sorted(SMOKE8_INDICES)
    smoke_validation["passed"] = smoke_validation["passed"] and smoke_validation["exact_smoke_indices"]
    by_index = {int(row["index"]): row for row in smoke_rows}
    smoke_validation["diagnostic_scores"] = {
        str(index): score_main_row(by_index[index], contract.scoring_row(index))["score"]
        for index in SMOKE8_INDICES
    }
    smoke_validation["diagnostic_only"] = True
    atomic_write_json(artifact_root / "smoke8_validation.json", smoke_validation)
    input_gate = _journal_input_gate(Path(smoke_metadata["journal"]), profile)
    atomic_write_json(artifact_root / "input_audit_gate.json", input_gate)
    processor = configuration.processor_audit or {
        "passed": True, "profile": profile.key, "transport": "locked upstream runner",
        "system_transport": profile.system_transport, "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "single_image_contract": True,
    }
    if profile.key == "spatialladder3b_rgb":
        processor = dict(processor)
        processor.update({
            "tokenizer_padding_side": capacity.get("tokenizer_padding_side"),
            "heterogeneous_prompt_lengths": capacity.get("heterogeneous_prompt_lengths"),
        })
        processor["passed"] = bool(
            processor.get("passed")
            and processor["tokenizer_padding_side"] == SPATIALLADDER_BATCH_PADDING_SIDE
            and (
                int(capacity["selected_capacity"]) == 1
                or processor["heterogeneous_prompt_lengths"] is True
            )
        )
    atomic_write_json(artifact_root / "processor_audit.json", processor)
    gate = {
        "schema_version": 1, "profile": profile.key,
        "passed": bool(canary["passed"] and capacity["passed"] and smoke_validation["passed"] and input_gate["passed"] and processor["passed"]),
        "dataset_fingerprint": contract.dataset_fingerprint,
        "binding": binding_value, "binding_digest": binding_digest,
        "selected_capacity": int(capacity["selected_capacity"]),
        "capacity_kind": capacity["capacity_kind"],
        "capacity_probe": capacity,
        "batch_grouping": {
            "batch_size": int(capacity["selected_capacity"]) if profile.native_batch_probe else 1,
            "seed": profile.decoding.get("seed"),
            "fixed_dataset_order": True,
        },
        "vision_canary": canary, "smoke_validation": smoke_validation,
        "input_audit_gate": input_gate, "processor_audit": processor,
        "gpu_preflight": gpu_audit, "smoke_metadata": smoke_metadata, "generated_at": utc_now(),
    }
    atomic_write_json(gate_path, gate)
    if not gate["passed"]:
        raise RuntimeError(f"SPBench-SI test gate failed for {profile.key}")
    print(f"[spbench-si] test gate passed: {gate_path}")
    return gate_path


def _api_resume_binding_errors(
    gate: dict[str, Any],
    current_binding: dict[str, Any],
    profile: SPBenchSIProfile,
) -> list[str]:
    errors: list[str] = []
    gate_digest = str(gate.get("binding_digest") or "")
    if test_gate_errors(gate, gate_digest):
        errors.append("the existing test gate does not pass its own locked binding")
    if profile.key != PACKYAPI_GEMINI_PROFILE:
        errors.append("API-source continuation is approved only for the Gemini 3.1 Pro profile")
    gate_binding = gate.get("binding") if isinstance(gate.get("binding"), dict) else {}
    gate_runtime = gate_binding.get("runtime") if isinstance(gate_binding.get("runtime"), dict) else {}
    current_runtime = (
        current_binding.get("runtime")
        if isinstance(current_binding.get("runtime"), dict)
        else {}
    )
    if gate_runtime.get("backend") != "openrouter" or current_runtime.get("backend") != "packyapi":
        errors.append("continuation must be OpenRouter -> PackyAPI")
    locked_fields = (
        "dataset",
        "prompt",
        "profile",
        "processor_audit",
        "image_processing",
        "decoding",
        "seed_strategy",
        "capacity_candidates",
        "smoke8_indices",
        "canary_protocol",
    )
    for field in locked_fields:
        if gate_binding.get(field) != current_binding.get(field):
            errors.append(f"non-source binding field changed: {field}")
    return errors


def _validated_openrouter_resume_seed(
    path: Path,
    profile: SPBenchSIProfile,
    contract: SPBenchSITestContract,
) -> tuple[dict[int, GenerationResult], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing original OpenRouter journal: {path}")
    successes: dict[int, GenerationResult] = {}
    signatures: set[str] = set()
    event_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event_count += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed OpenRouter journal JSON at line {line_number}") from exc
            if event.get("schema_version") != JOURNAL_SCHEMA_VERSION:
                raise ValueError(f"Unsupported OpenRouter journal schema at line {line_number}")
            signature = str(event.get("run_signature") or "")
            if not signature:
                raise ValueError(f"OpenRouter journal line {line_number} lacks a run signature")
            signatures.add(signature)
            if event.get("status") != "success":
                continue
            index = int(event.get("index"))
            if index in successes:
                raise ValueError(f"Duplicate OpenRouter success at index {index}")
            if not 0 <= index < OFFICIAL_TEST_SIZE:
                raise ValueError(f"OpenRouter success index outside full split: {index}")
            model_input = contract.model_input(index)
            expected_pixel_sha = pixel_sha256(model_input.image)
            audit = event.get("audit") if isinstance(event.get("audit"), dict) else {}
            generation = event.get("generation") if isinstance(event.get("generation"), dict) else {}
            expected_template_sha = _digest({
                "chat_template": profile.chat_template,
                "system_transport": profile.system_transport,
                "system_prompt": model_input.system_prompt,
                "user_prompt": model_input.user_prompt,
                "image_pixel_sha256": expected_pixel_sha,
                "media_count": 1,
            })
            exact_checks = {
                "audit profile": audit.get("profile") == profile.key,
                "audit inference protocol": audit.get("inference_protocol") == profile.inference_protocol,
                "audit chat template": audit.get("chat_template") == profile.chat_template,
                "audit system prompt": audit.get("system_prompt") == model_input.system_prompt,
                "audit user prompt": audit.get("user_prompt") == model_input.user_prompt,
                "audit image count": audit.get("image_count") == 1,
                "audit image sha": audit.get("image_pixel_sha256") == expected_pixel_sha,
                "generation model": generation.get("canonical_model") == profile.model,
                "generation provider": generation.get("provider") == "Google AI Studio",
                "generation media count": generation.get("num_media_prompt") == 1,
                "generation source rgb": generation.get("source_rgb_sha256") == expected_pixel_sha,
                "generation template": generation.get("template_sha256") == expected_template_sha,
            }
            failed = [name for name, passed in exact_checks.items() if not passed]
            if failed:
                raise ValueError(
                    f"OpenRouter journal success {index} cannot be reused: {', '.join(failed)}"
                )
            successes[index] = GenerationResult(
                text=str(event.get("prediction", "")),
                metadata=dict(generation),
                warnings=tuple(str(value) for value in event.get("warnings") or ()),
            )
    if len(signatures) != 1:
        raise ValueError(f"OpenRouter journal must have one run signature, got {len(signatures)}")
    if not successes or len(successes) >= OFFICIAL_TEST_SIZE:
        raise ValueError(
            f"OpenRouter resume seed must be non-empty and incomplete, got {len(successes)} successes"
        )
    indices = sorted(successes)
    provenance = {
        "source": "openrouter",
        "provider": "Google AI Studio",
        "canonical_model": profile.model,
        "source_journal": str(path.resolve()),
        "source_journal_sha256": _file_digest(path),
        "source_run_signature": next(iter(signatures)),
        "source_event_count": event_count,
        "success_count": len(indices),
        "success_indices_sha256": _digest(indices),
        "success_index_min": indices[0],
        "success_index_max": indices[-1],
    }
    return successes, provenance


def _completed_api_source_summary(path: Path) -> list[dict[str, Any]]:
    by_source: dict[str, list[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("status") != "success":
                continue
            generation = event.get("generation") if isinstance(event.get("generation"), dict) else {}
            source = "packyapi_gemini_slb" if generation.get("api_source") == "packyapi" else "openrouter_google_ai_studio"
            by_source.setdefault(source, []).append(int(event["index"]))
    return [
        {
            "source": source,
            "count": len(indices),
            "indices_sha256": _digest(sorted(indices)),
            "index_min": min(indices),
            "index_max": max(indices),
        }
        for source, indices in sorted(by_source.items())
    ]


def run_full_stage(
    profile: SPBenchSIProfile,
    contract: SPBenchSITestContract,
    output_root: Path,
    *,
    resume_api_source: bool = False,
) -> Path:
    track = track_directory(output_root, profile)
    gate_path = track / "test_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"Run --stage test first; missing gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gpu_audit = inspect_local_gpus(profile, profile.default_backend)
    configuration = resolve_configuration(profile)
    if configuration.backend != profile.default_backend:
        gpu_audit = inspect_local_gpus(profile, configuration.backend)
    binding_value = binding(configuration, contract, gpu_audit)
    problems = test_gate_errors(gate, _digest(binding_value))
    seed_successes: dict[int, GenerationResult] | None = None
    seed_provenance: dict[str, Any] | None = None
    continuation_journal: Path | None = None
    continuation_path = track / "api_source_continuation.json"
    if problems and resume_api_source:
        continuation_errors = _api_resume_binding_errors(gate, binding_value, profile)
        if continuation_errors:
            raise ValueError(
                "SPBench-SI API-source continuation is incompatible: "
                + "; ".join(continuation_errors)
            )
        output = track / "predictions.jsonl"
        original_journal = output.with_suffix(output.suffix + ".journal.jsonl")
        seed_successes, seed_provenance = _validated_openrouter_resume_seed(
            original_journal, profile, contract
        )
        continuation_journal = track / "predictions.jsonl.packyapi-resume.journal.jsonl"
        atomic_write_json(continuation_path, {
            "schema_version": 1,
            "status": "running",
            "profile": profile.key,
            "model_identity": profile.model,
            "model_revision": profile.revision,
            "inference_protocol": profile.inference_protocol,
            "same_model_identity": True,
            "test_stage_reused_without_retest": True,
            "capacity_reused_without_probe": True,
            "selected_capacity": int(gate["selected_capacity"]),
            "old_binding_digest": gate.get("binding_digest"),
            "new_binding_digest": _digest(binding_value),
            "new_backend": configuration.backend,
            "new_endpoint_sha256": configuration.endpoint_identity,
            "new_served_model_name": configuration.served_model_name,
            "seed": seed_provenance,
            "continuation_journal": str(continuation_journal),
            "generated_at": utc_now(),
        })
    elif problems:
        raise ValueError("SPBench-SI test gate is stale or incomplete: " + "; ".join(problems))
    elif resume_api_source:
        raise ValueError("--resume-api-source is only valid for an OpenRouter -> PackyAPI binding change")
    selected = int(gate["selected_capacity"])
    output = track / "predictions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter = build_adapter(
        configuration, batch_size=selected if profile.native_batch_probe else 1
    )
    try:
        metadata = run_recoverable_inference(
            contract=ProfiledContract(contract), adapter=adapter,
            output=output, target_indices=list(range(OFFICIAL_TEST_SIZE)), benchmark="SPBench-SI",
            split="test", official_size=OFFICIAL_TEST_SIZE, scorer_protocol=SCORER_PROTOCOL,
            workers=1 if profile.native_batch_probe else selected,
            journal_path=continuation_journal,
            seed_successes=seed_successes,
            seed_provenance=seed_provenance,
            initial_serial_requests=1 if seed_successes else 0,
            **_runtime_retry_policy(configuration.backend),
        )
    finally:
        adapter.close()
    metadata = _enrich_metadata(
        output.with_suffix(output.suffix + ".metadata.json"), contract=contract,
        configuration=configuration, binding_value=binding_value, test_gate=gate_path,
    )
    validation = validate_prediction_rows(read_jsonl(output), contract, prediction_path=output)
    if not validation["passed"] or not metadata.get("publishable_inference"):
        raise RuntimeError("SPBench-SI full inference did not pass mandatory validation")
    metadata.setdefault("runtime", {})["gpu_preflight"] = gpu_audit
    metadata["runtime"]["selected_capacity"] = selected
    metadata["runtime"]["capacity_kind"] = gate["capacity_kind"]
    if seed_successes and continuation_journal and seed_provenance:
        source_summary = _completed_api_source_summary(continuation_journal)
        metadata["api_source_continuation"] = {
            "same_model_identity": True,
            "test_stage_reused_without_retest": True,
            "old_source": seed_provenance,
            "new_source": {
                "source": "packyapi",
                "provider_pool": "Gemini-slb",
                "served_model_name": configuration.served_model_name,
                "endpoint_sha256": configuration.endpoint_identity,
            },
            "completed_source_summary": source_summary,
        }
        continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
        continuation.update({
            "status": "complete",
            "completed_source_summary": source_summary,
            "finished_at": utc_now(),
        })
        atomic_write_json(continuation_path, continuation)
    atomic_write_json(output.with_suffix(output.suffix + ".metadata.json"), metadata)
    atomic_write_json(track / "prediction_validation.json", validation)
    print(f"[spbench-si] full-{OFFICIAL_TEST_SIZE} validation passed: {output}")
    return output


def _selected_profiles(args: argparse.Namespace) -> list[SPBenchSIProfile]:
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
        raise ValueError("Selected SPBench-SI profiles contain duplicates")
    return ordered_profiles(keys)


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
    parser.add_argument(
        "--resume-api-source",
        action="store_true",
        help="resume the approved Gemini OpenRouter journal through PackyAPI without rerunning test",
    )
    parser.add_argument("--parquet", default=os.environ.get("SPBENCH_SI_PARQUET"))
    parser.add_argument("--images-archive", default=os.environ.get("SPBENCH_SI_IMAGES_ARCHIVE"))
    parser.add_argument("--output-root", default=os.environ.get("SPBENCH_SI_OUTPUT_ROOT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for key in PROFILE_SEQUENCE:
            profile = PROFILES[key]
            print(f"{profile.key}\t{profile.display_name}\t{profile.input_profile}\t{profile.default_backend}\tTP={profile.default_tensor_parallel_size}")
        return
    if args.status:
        if not args.output_root:
            raise ValueError("Set SPBENCH_SI_OUTPUT_ROOT")
        for key in PROFILE_SEQUENCE:
            profile = PROFILES[key]
            track = track_directory(args.output_root, profile)
            print(json.dumps({
                "profile": key,
                "test_gate": "present_unverified" if (track / "test_gate.json").is_file() else "missing",
                "full_prediction": "present_unverified" if (track / "predictions.jsonl").is_file() else "missing",
                "track": str(track),
            }, ensure_ascii=False))
        return
    profiles = _selected_profiles(args)
    if args.dry_run:
        for profile in profiles:
            print(f"{args.stage or 'check'}\t{profile.key}\t{profile.inference_protocol}")
        return
    if not args.parquet or not args.images_archive:
        raise ValueError("Set SPBENCH_SI_PARQUET and SPBENCH_SI_IMAGES_ARCHIVE")
    contract = SPBenchSITestContract(args.parquet, args.images_archive)
    contract.dataset_manifest(include_images=True)
    if args.check:
        for profile in profiles:
            gpu = inspect_local_gpus(profile, profile.default_backend)
            configuration = resolve_configuration(profile)
            if configuration.backend != profile.default_backend:
                gpu = inspect_local_gpus(profile, configuration.backend)
            print(json.dumps({"profile": profile.key, "binding": binding(configuration, contract, gpu), "gpu_preflight": gpu}, ensure_ascii=False, indent=2))
        return
    if not args.stage or not args.output_root:
        raise ValueError("Pass --stage test|full and set SPBENCH_SI_OUTPUT_ROOT")
    output_root = Path(args.output_root).resolve()
    status_path = output_root / "_batch" / "status.tsv"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        try:
            if args.stage == "test":
                if args.resume_api_source:
                    raise ValueError("--resume-api-source cannot be used with --stage test")
                run_test_stage(profile, contract, output_root)
            else:
                run_full_stage(
                    profile,
                    contract,
                    output_root,
                    resume_api_source=args.resume_api_source,
                )
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{utc_now()}\t{args.stage.upper()}_COMPLETE\t{profile.key}\n")
        except ResourceBlockedError as exc:
            with status_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{utc_now()}\tBLOCKED_RESOURCE\t{profile.key}\t{exc}\n")
            if not args.skip_resource_blocked:
                raise
            print(f"[spbench-si] BLOCKED_RESOURCE {profile.key}: {exc}")


if __name__ == "__main__":
    main()
