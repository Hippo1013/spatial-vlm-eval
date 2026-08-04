"""Two-stage, gate-bound CV-Bench inference orchestration."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...models.common.runtime import (
    GenerationResult,
    InferenceAdapter,
    atomic_write_json,
    atomic_write_jsonl,
    pixel_sha256,
    run_recoverable_inference,
    utc_now,
)
from ...models.common.vision_canary import (
    COLOR_CANARY_QUESTION,
    CVBENCH_COLOR_CANARY_PROTOCOL,
    make_solid_color_canary,
    validate_solid_color_canary_answer,
    validate_vision_canary_answer,
)
from ...models.openai_compatible.client import OpenAICompatibleAdapter
from .command_adapter import UpstreamCommandAdapter, load_generation_manifest
from .data import (
    CVBenchModelInput,
    CVBenchTestContract,
    DATASET_FILES,
    DATASET_REVISION,
    OFFICIAL_TEST_SIZE,
    SMOKE8_INDICES,
)
from .prediction_validation import read_jsonl, validate_prediction_rows
from .processor_audit import audit_processor
from .profiles import (
    DIRECT_ANSWER_SUFFIX,
    PROFILE_SEQUENCE,
    PROFILES,
    CVBenchProfile,
    get_profile,
    ordered_profiles,
)
from .scorer import SCORER_PROTOCOL

DEFAULT_CAPACITY_CANDIDATES = (32, 16, 8, 4, 2, 1)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _env_token(profile_key: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", profile_key.upper()).strip("_")


def _revision_tag(revision: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", revision).strip("-")


def track_directory(output_root: str | Path, profile: CVBenchProfile) -> Path:
    return (
        Path(output_root).resolve()
        / "runs"
        / profile.key
        / _revision_tag(profile.revision)
        / profile.inference_protocol
    )


def profile_prompt(prompt: str, profile: CVBenchProfile) -> str:
    dataset_prompt = str(prompt).strip()
    if not dataset_prompt:
        raise ValueError("CV-Bench dataset prompt must be non-empty")
    if not profile.prompt_prefix:
        return f"{dataset_prompt}\n{DIRECT_ANSWER_SUFFIX}"
    if "<answer>" not in profile.prompt_prefix or "</answer>" not in profile.prompt_prefix:
        raise ValueError(
            f"CV-Bench profile {profile.key} has a custom prompt without answer tags"
        )
    if "{question}" in profile.prompt_prefix:
        return profile.prompt_prefix.format(question=dataset_prompt)
    return f"{dataset_prompt}\n{profile.prompt_prefix}"


class ProfiledContract:
    """Prompt-transforming view that preserves the benchmark-owned source boundary."""

    def __init__(self, contract: CVBenchTestContract, profile: CVBenchProfile) -> None:
        self._contract = contract
        self.profile = profile
        self.dataset_root = contract.dataset_root

    def __len__(self) -> int:
        return len(self._contract)

    @property
    def dataset_fingerprint(self) -> str:
        return self._contract.dataset_fingerprint

    def model_input(self, index: int) -> CVBenchModelInput:
        value = self._contract.model_input(index)
        return CVBenchModelInput(
            index=value.index,
            image=value.image,
            question=profile_prompt(value.question, self.profile),
        )

    def model_inputs(self, indices: list[int] | tuple[int, ...]) -> list[CVBenchModelInput]:
        return [self.model_input(index) for index in indices]

    def prediction_row(self, index: int, prediction: str) -> dict[str, Any]:
        return self._contract.prediction_row(index, prediction)


class BoundAdapter(InferenceAdapter):
    """Add registry and backend-deviation identity to an existing adapter."""

    def __init__(
        self,
        delegate: InferenceAdapter,
        *,
        profile: CVBenchProfile,
        adapter_digest: str,
        backend_deviation: str | None,
        processor_audit: dict[str, Any] | None,
    ) -> None:
        self.delegate = delegate
        self.profile = profile
        self.adapter_digest = adapter_digest
        self.backend_deviation = backend_deviation
        self.processor_audit = processor_audit
        self.batch_size = int(getattr(delegate, "batch_size", 1))
        self.supports_concurrency = bool(getattr(delegate, "supports_concurrency", False))

    def metadata(self) -> dict[str, Any]:
        metadata = dict(self.delegate.metadata())
        metadata["registry_digest"] = self.profile.registry_digest
        metadata["adapter_digest"] = self.adapter_digest
        metadata["backend_deviation"] = self.backend_deviation
        metadata["processor_audit"] = self.processor_audit
        metadata["dataset_prompt_suffix"] = None
        metadata["profile_prompt_suffix"] = (
            None if self.profile.prompt_prefix else DIRECT_ANSWER_SUFFIX
        )
        metadata["profile_answer_format"] = (
            "reasoning_answer_tag" if self.profile.prompt_prefix else "direct_letter"
        )
        return metadata

    def generate(self, model_input: CVBenchModelInput) -> GenerationResult:
        return self.delegate.generate(model_input)

    def generate_batch(self, model_inputs: list[CVBenchModelInput]) -> list[GenerationResult]:
        return self.delegate.generate_batch(model_inputs)

    def close(self) -> None:
        self.delegate.close()


@dataclass(frozen=True, slots=True)
class ResolvedConfiguration:
    profile: CVBenchProfile
    backend: str
    base_urls: tuple[str, ...]
    decoding: dict[str, Any]
    adapter_digest: str
    command: str | None
    processor_audit: dict[str, Any] | None
    backend_deviation: str | None

    @property
    def sharding(self) -> dict[str, Any]:
        return {
            "strategy": "fixed_modulo" if len(self.base_urls) == 2 else "single_worker",
            "worker_count": len(self.base_urls) if self.base_urls else 1,
            "tensor_parallel_size": self.profile.default_tensor_parallel_size,
        }


def _open_adapter_digest(profile: CVBenchProfile) -> str:
    from ...models.openai_compatible import client

    files = [Path(inspect.getfile(client)), Path(__file__), Path(inspect.getfile(type(profile)))]
    return _digest(
        {
            "profile_registry_digest": profile.registry_digest,
            "files": {str(path): _file_digest(path) for path in files},
        }
    )


def _profile_env(profile: CVBenchProfile, suffix: str) -> str | None:
    return os.environ.get(f"CVBENCH_{_env_token(profile.key)}_{suffix}")


def _configured_gpu_ids(profile: CVBenchProfile) -> tuple[int, ...]:
    raw = _profile_env(profile, "GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw.strip():
        return ()
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ValueError(f"Invalid GPU selection for {profile.key}: {raw!r}")
    return values


def inspect_local_gpus(profile: CVBenchProfile, backend: str) -> dict[str, Any]:
    """Read GPU inventory/process state without mutating or terminating any process."""

    if backend == "openrouter":
        return {"applicable": False, "reason": "remote API backend"}
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise FileNotFoundError("Local CV-Bench inference requires nvidia-smi for read-only GPU preflight")
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
    by_index = {item["index"]: item for item in inventory}
    missing = [index for index in selected if index not in by_index]
    if missing:
        raise ValueError(f"Configured GPUs are not present for {profile.key}: {missing}")
    if profile.key == "internvl3_78b":
        if len(selected) != 4:
            raise ValueError(
                "InternVL3-78B requires exactly four explicit GPU ids via "
                "CVBENCH_INTERNVL3_78B_GPU_IDS or CUDA_VISIBLE_DEVICES"
            )
        undersized = [
            index for index in selected if by_index[index]["memory_total_mib"] < 79_000
        ]
        if undersized:
            raise ValueError(f"InternVL3-78B requires four 80GB GPUs; undersized ids: {undersized}")
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
    processes = [line.strip() for line in process_query.stdout.splitlines() if line.strip()]
    return {
        "applicable": True,
        "selected_gpu_ids": list(selected),
        "inventory": inventory,
        "compute_processes": processes,
        "policy": "read-only; no process is terminated",
    }


def _resolved_command_configuration(
    profile: CVBenchProfile,
    *,
    backend_deviation: str | None = None,
) -> ResolvedConfiguration:
    command = _profile_env(profile, "COMMAND")
    adapter_digest = _profile_env(profile, "ADAPTER_DIGEST")
    if not command or not adapter_digest:
        raise ValueError(
            f"Set CVBENCH_{_env_token(profile.key)}_COMMAND and "
            f"CVBENCH_{_env_token(profile.key)}_ADAPTER_DIGEST for {profile.key}"
        )
    manifest = _profile_env(profile, "GENERATION_MANIFEST")
    decoding = load_generation_manifest(profile, manifest)
    return ResolvedConfiguration(
        profile=profile,
        backend="upstream_transformers",
        base_urls=(),
        decoding=decoding,
        adapter_digest=adapter_digest,
        command=command,
        processor_audit=None,
        backend_deviation=backend_deviation,
    )


def resolve_configuration(profile: CVBenchProfile) -> ResolvedConfiguration:
    backend = _profile_env(profile, "BACKEND") or profile.default_backend
    if profile.adapter_kind == "upstream_command" or backend in {"transformers", "upstream_transformers"}:
        deviation = (
            "vLLM processor/template audit fallback to locked upstream Transformers"
            if profile.adapter_kind == "openai_compatible"
            else None
        )
        return _resolved_command_configuration(profile, backend_deviation=deviation)
    if backend not in {"vllm", "openrouter"}:
        raise ValueError(f"Unsupported backend {backend!r} for {profile.key}")
    processor_report: dict[str, Any] | None = None
    if backend == "vllm":
        model_path = os.environ.get(profile.model_path_env, "")
        if not model_path:
            raise ValueError(f"Set {profile.model_path_env} for processor/template audit")
        try:
            processor_report = audit_processor(profile, model_path)
        except Exception:
            if _profile_env(profile, "COMMAND") and _profile_env(profile, "ADAPTER_DIGEST"):
                return _resolved_command_configuration(
                    profile,
                    backend_deviation="vLLM processor/template audit failed; locked Transformers fallback",
                )
            raise
        raw_urls = _profile_env(profile, "BASE_URLS") or os.environ.get("CVBENCH_VLLM_BASE_URLS", "")
        urls = tuple(value.strip().rstrip("/") for value in raw_urls.split(",") if value.strip())
        expected = 2 if profile.default_tensor_parallel_size == 1 else 1
        if len(urls) != expected:
            raise ValueError(
                f"{profile.key} requires {expected} vLLM endpoint(s); set "
                f"CVBENCH_{_env_token(profile.key)}_BASE_URLS"
            )
    else:
        urls = (os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),)
    decoding = dict(profile.decoding)
    decoding["stream"] = False
    return ResolvedConfiguration(
        profile=profile,
        backend=backend,
        base_urls=urls,
        decoding=decoding,
        adapter_digest=_open_adapter_digest(profile),
        command=None,
        processor_audit=processor_report,
        backend_deviation=None,
    )


def _request_timeout_seconds(backend: str) -> float:
    if backend == "vllm":
        return float(os.environ.get("CVBENCH_VLLM_API_TIMEOUT", "600"))
    return float(os.environ.get("CVBENCH_API_TIMEOUT", "180"))


def _runtime_retry_policy(backend: str) -> dict[str, int]:
    if backend == "vllm":
        return {
            "retries": int(os.environ.get("CVBENCH_VLLM_INFERENCE_RETRIES", "0")),
            "retry_missing_passes": int(
                os.environ.get("CVBENCH_VLLM_RETRY_MISSING_PASSES", "1")
            ),
        }
    return {
        "retries": int(os.environ.get("CVBENCH_INFERENCE_RETRIES", "2")),
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
        delegate = OpenAICompatibleAdapter(
            profile=profile,
            backend=configuration.backend,
            base_url=configuration.base_urls[endpoint_index],
            api_key=api_key,
            served_model_name=profile.served_model_name,
            timeout=_request_timeout_seconds(configuration.backend),
            metadata_retries=int(os.environ.get("CVBENCH_OPENROUTER_METADATA_RETRIES", "10")),
            policy_key=profile.api_policy_key,
            image_source="CV-Bench RGB only",
        )
    return BoundAdapter(
        delegate,
        profile=profile,
        adapter_digest=configuration.adapter_digest,
        backend_deviation=configuration.backend_deviation,
        processor_audit=configuration.processor_audit,
    )


def binding(configuration: ResolvedConfiguration, contract: CVBenchTestContract) -> dict[str, Any]:
    processor_identity = None
    if configuration.processor_audit:
        processor_identity = {
            key: configuration.processor_audit.get(key)
            for key in (
                "profile",
                "model_revision",
                "processor_class",
                "logical_image_placeholder",
                "logical_image_placeholder_count",
                "rendered_template_sha256",
                "pixel_values_shape",
                "image_grid_rows",
                "model_snapshot_revision_verified",
            )
        }
    return {
        "dataset": {
            "revision": DATASET_REVISION,
            "fingerprint": contract.dataset_fingerprint,
            "files": {item.name: item.sha256 for item in DATASET_FILES},
        },
        "profile": {
            "key": configuration.profile.key,
            "model": configuration.profile.model,
            "model_revision": configuration.profile.revision,
            "input_profile": configuration.profile.input_profile,
            "inference_protocol": configuration.profile.inference_protocol,
            "registry_digest": configuration.profile.registry_digest,
            "upstream_commit": configuration.profile.upstream_commit,
        },
        "adapter": {
            "backend": configuration.backend,
            "adapter_digest": configuration.adapter_digest,
            "backend_deviation": configuration.backend_deviation,
            "decoding": configuration.decoding,
            "processor_audit": processor_identity,
        },
        "test_protocol": {
            "vision_canary": CVBENCH_COLOR_CANARY_PROTOCOL,
            "smoke_indices": list(SMOKE8_INDICES),
        },
        "sharding": {
            **configuration.sharding,
            "configured_gpu_ids": list(_configured_gpu_ids(configuration.profile)),
        },
    }


def _cvbench_color_canary_specs() -> tuple[tuple[str, Any], ...]:
    return tuple((color, make_solid_color_canary(color)) for color in ("red", "blue"))


def _canary_report(
    adapter: BoundAdapter,
    profile: CVBenchProfile,
    expected_color: str,
    image: Any,
) -> dict[str, Any]:
    result = adapter.generate(
        CVBenchModelInput(index=-1, image=image, question=COLOR_CANARY_QUESTION)
    )
    validate_solid_color_canary_answer(result.text, expected_color)
    generation = dict(result.metadata)
    if generation.get("num_media_prompt") != 1 and generation.get("num_model_image_tensors") != 1:
        raise ValueError("Vision canary did not prove exactly one image at the model boundary")
    return {
        "passed": True,
        "canary_protocol": CVBENCH_COLOR_CANARY_PROTOCOL,
        "expected_color": expected_color,
        "profile": profile.key,
        "question": COLOR_CANARY_QUESTION,
        "request_image_count": 1,
        "image_mode": "RGB",
        "image_size": list(image.size),
        "image_pixel_sha256": pixel_sha256(image),
        "answer": result.text,
        "generation": generation,
    }


def _capacity_candidates() -> tuple[int, ...]:
    raw = os.environ.get("CVBENCH_CAPACITY_CANDIDATES")
    if not raw:
        return DEFAULT_CAPACITY_CANDIDATES
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values) or list(values) != sorted(values, reverse=True):
        raise ValueError("CVBENCH_CAPACITY_CANDIDATES must be positive and descending")
    return values


def probe_capacity(adapter: BoundAdapter) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    image = make_solid_color_canary("red")
    model_input = CVBenchModelInput(index=-1, image=image, question=COLOR_CANARY_QUESTION)
    for candidate in _capacity_candidates():
        try:
            with ThreadPoolExecutor(max_workers=candidate) as executor:
                results = list(executor.map(lambda _: adapter.generate(model_input), range(candidate)))
            for result in results:
                generation = result.metadata
                if (
                    generation.get("num_media_prompt") != 1
                    and generation.get("num_model_image_tensors") != 1
                ):
                    raise ValueError("Capacity probe did not prove exactly one model image")
            attempts.append({"candidate": candidate, "passed": True})
            return {"passed": True, "selected_concurrency": candidate, "attempts": attempts}
        except Exception as error:  # noqa: BLE001 - capacity fallback intentionally handles OOM/API failures.
            attempts.append(
                {
                    "candidate": candidate,
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }
            )
    raise RuntimeError(f"No stable vLLM concurrency candidate: {attempts}")


def _journal_input_gate(
    journal_path: Path,
    expected_prompts: dict[int, str],
) -> dict[str, Any]:
    successful: dict[int, dict[str, Any]] = {}
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("status") == "success":
            successful[int(event["index"])] = event
    errors: list[str] = []
    if set(successful) != set(expected_prompts):
        errors.append(f"success indices mismatch: {sorted(successful)}")
    for index, event in successful.items():
        audit = event.get("audit") or {}
        generation = event.get("generation") or {}
        if audit.get("image_count") != 1 or audit.get("image_mode") != "RGB":
            errors.append(f"index={index}: input audit is not exactly one RGB image")
        if generation.get("num_media_prompt") != 1 and generation.get("num_model_image_tensors") != 1:
            errors.append(f"index={index}: model-boundary media/tensor count is not one")
        if not audit.get("question") or audit.get("question") != str(audit.get("question")).strip():
            errors.append(f"index={index}: audited prompt is empty or not outer-trimmed")
        elif audit.get("question") != expected_prompts.get(index):
            errors.append(f"index={index}: audited prompt differs from deterministic reconstruction")
    return {
        "passed": not errors,
        "num_successes": len(successful),
        "expected_indices": sorted(expected_prompts),
        "all_image_counts_one": not any("image" in error for error in errors),
        "all_model_media_counts_one": not any("media/tensor" in error for error in errors),
        "errors": errors,
    }


def _enrich_metadata(
    metadata_path: Path,
    *,
    contract: CVBenchTestContract,
    configuration: ResolvedConfiguration,
    binding_value: dict[str, Any],
    test_gate: Path | None,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset"]["revision"] = DATASET_REVISION
    metadata["dataset"]["files"] = {item.name: item.sha256 for item in DATASET_FILES}
    metadata["dataset"]["fingerprint"] = contract.dataset_fingerprint
    metadata["binding"] = binding_value
    metadata["binding_digest"] = _digest(binding_value)
    metadata["test_gate"] = str(test_gate) if test_gate else None
    metadata["scorer_protocol"] = SCORER_PROTOCOL
    atomic_write_json(metadata_path, metadata)
    return metadata


def merge_prediction_shards(
    shard_paths: list[str | Path],
    output: str | Path,
    *,
    expected_indices: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        rows.extend(read_jsonl(shard_path))
    indices = [int(row["index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("CV-Bench shard merge found duplicate indices")
    if set(indices) != set(int(index) for index in expected_indices):
        raise ValueError("CV-Bench shard merge index coverage mismatch")
    ordered = sorted(rows, key=lambda row: int(row["index"]))
    atomic_write_jsonl(Path(output).resolve(), ordered)
    return ordered


def test_gate_errors(gate: dict[str, Any], expected_binding_digest: str) -> list[str]:
    errors: list[str] = []
    if gate.get("passed") is not True:
        errors.append("gate.passed is not true")
    if gate.get("binding_digest") != expected_binding_digest:
        errors.append("binding digest mismatch")
    if not isinstance(gate.get("binding"), dict) or _digest(gate["binding"]) != expected_binding_digest:
        errors.append("embedded binding does not match its digest")
    if gate.get("smoke_indices") != list(SMOKE8_INDICES):
        errors.append("smoke8 indices mismatch")
    json_artifacts = {
        "dataset_audit": None,
        "vision_canary": "passed",
        "capacity_probe": "passed",
        "smoke_validation": "passed",
        "input_audit_gate": "passed",
    }
    loaded_artifacts: dict[str, dict[str, Any]] = {}
    for label, passed_key in json_artifacts.items():
        raw_path = gate.get(label)
        path = Path(str(raw_path)).resolve() if raw_path else None
        if path is None or not path.is_file():
            errors.append(f"missing {label} artifact")
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"malformed {label} artifact")
            continue
        if not isinstance(artifact, dict):
            errors.append(f"non-object {label} artifact")
        elif passed_key and artifact.get(passed_key) is not True:
            errors.append(f"{label} did not pass")
        elif label == "dataset_audit" and artifact.get("dataset_fingerprint") != (
            gate.get("binding", {}).get("dataset", {}).get("fingerprint")
        ):
            errors.append("dataset audit fingerprint mismatch")
        else:
            loaded_artifacts[label] = artifact
    capacity = loaded_artifacts.get("capacity_probe", {})
    if capacity and gate.get("selected_concurrency") != capacity.get("selected_concurrency"):
        errors.append("selected concurrency differs from capacity probe")
    vision_canary = loaded_artifacts.get("vision_canary", {})
    expected_canary_protocol = (
        gate.get("binding", {}).get("test_protocol", {}).get("vision_canary")
    )
    canary_endpoints = vision_canary.get("endpoints")
    if vision_canary and (
        not expected_canary_protocol
        or not isinstance(canary_endpoints, list)
        or not canary_endpoints
        or any(
            not isinstance(item, dict)
            or item.get("canary_protocol") != expected_canary_protocol
            for item in canary_endpoints
        )
    ):
        errors.append("vision canary protocol mismatch")
    prediction_raw = gate.get("smoke_predictions")
    prediction = Path(str(prediction_raw)).resolve() if prediction_raw else None
    if prediction is None or not prediction.is_file():
        errors.append("missing smoke prediction artifact")
    elif not prediction.with_suffix(prediction.suffix + ".metadata.json").is_file():
        errors.append("missing smoke inference metadata")
    processor_required = bool(
        gate.get("binding", {}).get("adapter", {}).get("processor_audit")
    )
    processor_raw = gate.get("processor_audit")
    if processor_required and (
        not processor_raw or not Path(str(processor_raw)).resolve().is_file()
    ):
        errors.append("missing required processor audit")
    return errors


def _load_gate_artifact(gate: dict[str, Any], label: str) -> dict[str, Any]:
    raw_path = gate.get(label)
    if not raw_path:
        raise ValueError(f"Legacy gate is missing {label}")
    path = Path(str(raw_path)).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Legacy {label} artifact is not an object")
    return value


def _strict_legacy_gate_migration_errors(
    gate: dict[str, Any],
    profile: CVBenchProfile,
    current_binding: dict[str, Any],
) -> list[str]:
    """Validate that an older spatial canary is stronger evidence of image receipt."""

    errors: list[str] = []
    if gate.get("passed") is not True or gate.get("profile") != profile.key:
        errors.append("legacy gate identity/pass mismatch")
    if gate.get("smoke_indices") != list(SMOKE8_INDICES):
        errors.append("legacy smoke8 indices mismatch")
    old_binding = gate.get("binding")
    if not isinstance(old_binding, dict):
        errors.append("legacy embedded binding is missing")
        return errors
    for section in ("dataset", "profile", "sharding"):
        if old_binding.get(section) != current_binding.get(section):
            errors.append(f"legacy {section} binding mismatch")
    old_adapter = dict(old_binding.get("adapter") or {})
    new_adapter = dict(current_binding.get("adapter") or {})
    old_adapter.pop("adapter_digest", None)
    new_adapter.pop("adapter_digest", None)
    if old_adapter != new_adapter:
        errors.append("legacy adapter binding mismatch beyond orchestration digest")
    try:
        dataset_audit = _load_gate_artifact(gate, "dataset_audit")
        if dataset_audit.get("dataset_fingerprint") != current_binding["dataset"]["fingerprint"]:
            errors.append("legacy dataset audit fingerprint mismatch")
        for label in ("capacity_probe", "smoke_validation", "input_audit_gate"):
            if _load_gate_artifact(gate, label).get("passed") is not True:
                errors.append(f"legacy {label} did not pass")
        prediction = Path(str(gate.get("smoke_predictions", ""))).resolve()
        if not prediction.is_file() or not prediction.with_suffix(
            prediction.suffix + ".metadata.json"
        ).is_file():
            errors.append("legacy smoke prediction/metadata is missing")
        if current_binding.get("adapter", {}).get("processor_audit"):
            processor = gate.get("processor_audit")
            if not processor or not Path(str(processor)).resolve().is_file():
                errors.append("legacy required processor audit is missing")
        canary = _load_gate_artifact(gate, "vision_canary")
        endpoints = canary.get("endpoints")
        if canary.get("passed") is not True or not isinstance(endpoints, list) or not endpoints:
            errors.append("legacy strict vision canary did not pass")
        else:
            for item in endpoints:
                protocol = str(item.get("canary_protocol", ""))
                generation = item.get("generation") or {}
                if "red_circle_top_left_blue_square_bottom_right" not in protocol:
                    errors.append("legacy canary protocol is not the strict red/blue spatial test")
                    continue
                if item.get("request_image_count") != 1 or (
                    generation.get("num_media_prompt") != 1
                    and generation.get("num_model_image_tensors") != 1
                ):
                    errors.append("legacy strict canary lacks exact-one-image evidence")
                try:
                    validate_vision_canary_answer(str(item.get("answer", "")))
                except ValueError as error:
                    errors.append(f"legacy strict canary answer is invalid: {error}")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"legacy gate artifact error: {error}")
    return errors


def _migrate_strict_legacy_gate(
    gate: dict[str, Any],
    gate_path: Path,
    run_dir: Path,
    profile: CVBenchProfile,
    current_binding: dict[str, Any],
    current_binding_digest: str,
) -> Path | None:
    errors = _strict_legacy_gate_migration_errors(gate, profile, current_binding)
    if errors:
        return None
    source_canary_path = Path(str(gate["vision_canary"])).resolve()
    source_gate_path = source_canary_path.parent / "test_gate.json"
    if not source_gate_path.is_file():
        return None
    source_canary = json.loads(source_canary_path.read_text(encoding="utf-8"))
    migrated_endpoints = []
    for item in source_canary["endpoints"]:
        migrated = dict(item)
        migrated["canary_protocol"] = CVBENCH_COLOR_CANARY_PROTOCOL
        migrated["evidence_kind"] = "stricter_legacy_evidence"
        migrated["source_canary_protocol"] = item.get("canary_protocol")
        migrated["source_image_colors_proven"] = ["red", "blue"]
        migrated_endpoints.append(migrated)
    run_dir.mkdir(parents=True, exist_ok=True)
    migrated_canary_path = run_dir / "vision_canary.json"
    atomic_write_json(
        migrated_canary_path,
        {
            "passed": True,
            "evidence_kind": "stricter_legacy_evidence",
            "source_gate": str(source_gate_path.resolve()),
            "source_vision_canary": str(source_canary_path),
            "endpoints": migrated_endpoints,
        },
    )
    migrated_gate = dict(gate)
    migrated_gate.update(
        {
            "binding": current_binding,
            "binding_digest": current_binding_digest,
            "vision_canary": str(migrated_canary_path),
            "generated_at": utc_now(),
            "evidence_migration": {
                "kind": "stricter_legacy_evidence",
                "source_gate": str(source_gate_path.resolve()),
                "source_binding_digest": gate.get("binding_digest"),
                "model_was_not_reinvoked": True,
            },
        }
    )
    atomic_write_json(run_dir / "test_gate.json", migrated_gate)
    atomic_write_json(gate_path, migrated_gate)
    return gate_path


def _adapter_digest_only_change(
    old_binding: dict[str, Any], current_binding: dict[str, Any]
) -> bool:
    old_comparable = json.loads(json.dumps(old_binding))
    new_comparable = json.loads(json.dumps(current_binding))
    old_adapter = old_comparable.get("adapter")
    new_adapter = new_comparable.get("adapter")
    if not isinstance(old_adapter, dict) or not isinstance(new_adapter, dict):
        return False
    source_adapter_digest = old_adapter.pop("adapter_digest", None)
    new_adapter_digest = new_adapter.pop("adapter_digest", None)
    return bool(
        source_adapter_digest
        and new_adapter_digest
        and source_adapter_digest != new_adapter_digest
        and old_comparable == new_comparable
    )


def _migrate_adapter_digest_only_gate(
    gate: dict[str, Any],
    gate_path: Path,
    run_dir: Path,
    current_binding: dict[str, Any],
    current_binding_digest: str,
) -> Path | None:
    """Reuse a valid current-protocol gate when only the composite source digest changed."""

    old_binding = gate.get("binding")
    old_digest = gate.get("binding_digest")
    if not isinstance(old_binding, dict) or not isinstance(old_digest, str):
        return None
    if test_gate_errors(gate, old_digest):
        return None
    if not _adapter_digest_only_change(old_binding, current_binding):
        return None
    source_adapter_digest = old_binding["adapter"]["adapter_digest"]
    source_gate_path = Path(str(gate["vision_canary"])).resolve().parent / "test_gate.json"
    if not source_gate_path.is_file():
        return None
    migrated_gate = dict(gate)
    migrated_gate.update(
        {
            "binding": current_binding,
            "binding_digest": current_binding_digest,
            "generated_at": utc_now(),
            "evidence_migration": {
                "kind": "adapter_digest_only",
                "source_gate": str(source_gate_path),
                "source_binding_digest": old_digest,
                "source_adapter_digest": source_adapter_digest,
                "model_was_not_reinvoked": True,
            },
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "test_gate.json", migrated_gate)
    atomic_write_json(gate_path, migrated_gate)
    return gate_path


def _run_dual_shard_full(
    *,
    contract: CVBenchTestContract,
    profiled: ProfiledContract,
    configuration: ResolvedConfiguration,
    output: Path,
    binding_value: dict[str, Any],
    test_gate: Path,
    capacity: int,
) -> dict[str, Any]:
    shard_root = output.parent / "shards"
    shard_outputs = [shard_root / f"worker-{worker}" / "predictions.jsonl" for worker in range(2)]

    def run_worker(worker: int) -> dict[str, Any]:
        adapter = build_adapter(configuration, worker)
        indices = [index for index in range(OFFICIAL_TEST_SIZE) if index % 2 == worker]
        return run_recoverable_inference(
            contract=profiled,
            adapter=adapter,
            output=shard_outputs[worker],
            target_indices=indices,
            benchmark="CV-Bench",
            split="test",
            official_size=OFFICIAL_TEST_SIZE,
            scorer_protocol=SCORER_PROTOCOL,
            workers=capacity,
            **_runtime_retry_policy(configuration.backend),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        shard_metadata = list(executor.map(run_worker, range(2)))
    rows = merge_prediction_shards(
        [str(path) for path in shard_outputs],
        output,
        expected_indices=list(range(OFFICIAL_TEST_SIZE)),
    )
    output_sha = _file_digest(output)
    model_metadata = dict(shard_metadata[0]["model"])
    model_metadata["backend"] = "vllm-dual-endpoint"
    model_metadata["shards"] = [
        {
            "worker": worker,
            "endpoint": shard_metadata[worker]["model"].get("api_base_url"),
            "run_signature": shard_metadata[worker]["run_signature"],
            "journal": shard_metadata[worker]["journal"],
        }
        for worker in range(2)
    ]
    final = {
        "schema_version": 1,
        "inference_protocol": configuration.profile.inference_protocol,
        "scorer_protocol": SCORER_PROTOCOL,
        "model": model_metadata,
        "dataset": {
            "benchmark": "CV-Bench",
            "root": str(contract.dataset_root),
            "split": "test",
            "revision": DATASET_REVISION,
            "fingerprint": contract.dataset_fingerprint,
            "files": {item.name: item.sha256 for item in DATASET_FILES},
            "official_test_size": OFFICIAL_TEST_SIZE,
            "loaded_size": len(contract),
            "target_indices": list(range(OFFICIAL_TEST_SIZE)),
            "num_targets": OFFICIAL_TEST_SIZE,
            "is_subset": False,
        },
        "output": str(output),
        "output_sha256": output_sha,
        "run_signature": _digest([item["run_signature"] for item in shard_metadata]),
        "num_predictions": len(rows),
        "empty_prediction_indices": [
            int(row["index"]) for row in rows if not str(row["raw_prediction"]).strip()
        ],
        "publishable_inference": True,
        "binding": binding_value,
        "binding_digest": _digest(binding_value),
        "test_gate": str(test_gate),
        "shard_metadata": [item["output"] + ".metadata.json" for item in shard_metadata],
        "started_at": min(item["started_at"] for item in shard_metadata),
        "finished_at": max(item["finished_at"] for item in shard_metadata),
        "runtime": {"sharding": configuration.sharding, "per_endpoint_concurrency": capacity},
    }
    atomic_write_json(output.with_suffix(output.suffix + ".metadata.json"), final)
    return final


def _run_test_stage_with_configuration(
    profile: CVBenchProfile,
    contract: CVBenchTestContract,
    output_root: Path,
    configuration: ResolvedConfiguration,
) -> Path:
    binding_value = binding(configuration, contract)
    binding_digest = _digest(binding_value)
    track = track_directory(output_root, profile)
    gate_path = track / "test_gate.json"
    existing: dict[str, Any] | None = None
    if gate_path.is_file():
        existing = json.loads(gate_path.read_text(encoding="utf-8"))
        if not test_gate_errors(existing, binding_digest):
            print(f"[cv-bench] current test gate already passed: {gate_path}")
            return gate_path
    run_dir = track / "test_runs" / binding_digest
    if existing is not None:
        migrated = _migrate_adapter_digest_only_gate(
            existing,
            gate_path,
            run_dir,
            binding_value,
            binding_digest,
        )
        if migrated is not None:
            migrated_gate = json.loads(migrated.read_text(encoding="utf-8"))
            errors = test_gate_errors(migrated_gate, binding_digest)
            if errors:
                raise RuntimeError(f"Adapter-digest-only migrated gate is invalid: {errors}")
            print(f"[cv-bench] test gate migrated after adapter digest-only change: {migrated}")
            return migrated
        migrated = _migrate_strict_legacy_gate(
            existing,
            gate_path,
            run_dir,
            profile,
            binding_value,
            binding_digest,
        )
        if migrated is not None:
            migrated_gate = json.loads(migrated.read_text(encoding="utf-8"))
            errors = test_gate_errors(migrated_gate, binding_digest)
            if errors:
                raise RuntimeError(f"Migrated strict legacy gate is invalid: {errors}")
            print(f"[cv-bench] test gate migrated from stricter legacy evidence: {migrated}")
            return migrated
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_audit = inspect_local_gpus(profile, configuration.backend)
    atomic_write_json(run_dir / "gpu_preflight.json", gpu_audit)
    dataset_audit = contract.dataset_manifest(include_images=True)
    atomic_write_json(run_dir / "dataset_audit.json", dataset_audit)
    if configuration.processor_audit:
        atomic_write_json(run_dir / "processor_audit.json", configuration.processor_audit)

    endpoint_count = len(configuration.base_urls) if configuration.base_urls else 1
    adapters = [build_adapter(configuration, index) for index in range(endpoint_count)]
    canaries: list[dict[str, Any]] = []
    capacities: list[dict[str, Any]] = []
    try:
        for adapter in adapters:
            canaries.extend(
                _canary_report(adapter, profile, expected_color, image)
                for expected_color, image in _cvbench_color_canary_specs()
            )
            if configuration.backend == "vllm":
                capacities.append(probe_capacity(adapter))
        atomic_write_json(run_dir / "vision_canary.json", {"passed": True, "endpoints": canaries})
        if capacities:
            selected_capacity = min(int(item["selected_concurrency"]) for item in capacities)
            capacity_report = {
                "passed": True,
                "selected_concurrency": selected_capacity,
                "endpoints": capacities,
            }
        else:
            selected_capacity = profile.default_workers
            capacity_report = {
                "passed": True,
                "selected_concurrency": selected_capacity,
                "not_applicable": configuration.backend != "vllm",
            }
        atomic_write_json(run_dir / "capacity_probe.json", capacity_report)
        for adapter in adapters[1:]:
            adapter.close()
        profiled = ProfiledContract(contract, profile)
        prediction_path = run_dir / "predictions.jsonl"
        metadata = run_recoverable_inference(
            contract=profiled,
            adapter=adapters[0],
            output=prediction_path,
            target_indices=SMOKE8_INDICES,
            benchmark="CV-Bench",
            split="test",
            official_size=OFFICIAL_TEST_SIZE,
            scorer_protocol=SCORER_PROTOCOL,
            workers=min(selected_capacity, len(SMOKE8_INDICES)),
            **_runtime_retry_policy(configuration.backend),
        )
    except BaseException:
        for adapter in adapters:
            adapter.close()
        raise

    metadata_path = prediction_path.with_suffix(prediction_path.suffix + ".metadata.json")
    _enrich_metadata(
        metadata_path,
        contract=contract,
        configuration=configuration,
        binding_value=binding_value,
        test_gate=None,
    )
    prediction_rows = read_jsonl(prediction_path)
    validation = validate_prediction_rows(
        prediction_rows,
        contract,
        prediction_path=prediction_path,
        allow_subset=True,
    )
    expected_prompts = {
        index: profiled.model_input(index).question for index in SMOKE8_INDICES
    }
    input_gate = _journal_input_gate(
        Path(metadata["journal"]),
        expected_prompts,
    )
    if not validation["passed"] or not input_gate["passed"]:
        raise RuntimeError("CV-Bench smoke8 validation/input audit failed")
    if {
        index: profiled.model_input(index).question for index in SMOKE8_INDICES
    } != expected_prompts:
        raise RuntimeError("CV-Bench smoke8 prompt reconstruction is not deterministic")
    atomic_write_json(run_dir / "prediction_validation.json", validation)
    atomic_write_json(run_dir / "input_audit_gate.json", input_gate)
    gate = {
        "schema_version": 1,
        "passed": True,
        "profile": profile.key,
        "binding": binding_value,
        "binding_digest": binding_digest,
        "dataset_audit": str(run_dir / "dataset_audit.json"),
        "processor_audit": (
            str(run_dir / "processor_audit.json") if configuration.processor_audit else None
        ),
        "vision_canary": str(run_dir / "vision_canary.json"),
        "capacity_probe": str(run_dir / "capacity_probe.json"),
        "gpu_preflight": str(run_dir / "gpu_preflight.json"),
        "selected_concurrency": selected_capacity,
        "smoke_predictions": str(prediction_path),
        "smoke_validation": str(run_dir / "prediction_validation.json"),
        "input_audit_gate": str(run_dir / "input_audit_gate.json"),
        "smoke_indices": list(SMOKE8_INDICES),
        "generated_at": utc_now(),
    }
    atomic_write_json(run_dir / "test_gate.json", gate)
    atomic_write_json(gate_path, gate)
    print(f"[cv-bench] test gate passed: {gate_path}")
    return gate_path


def run_test_stage(
    profile: CVBenchProfile,
    contract: CVBenchTestContract,
    output_root: Path,
) -> Path:
    configuration = resolve_configuration(profile)
    try:
        return _run_test_stage_with_configuration(
            profile, contract, output_root, configuration
        )
    except Exception as error:
        if (
            configuration.backend == "vllm"
            and _profile_env(profile, "COMMAND")
            and _profile_env(profile, "ADAPTER_DIGEST")
        ):
            deviation = (
                "vLLM test input/output contract failed; locked Transformers fallback "
                f"({type(error).__name__}: {str(error)[:200]})"
            )
            fallback = _resolved_command_configuration(
                profile, backend_deviation=deviation
            )
            print(f"[cv-bench] {profile.key}: {deviation}")
            return _run_test_stage_with_configuration(
                profile, contract, output_root, fallback
            )
        raise


def run_full_stage(
    profile: CVBenchProfile,
    contract: CVBenchTestContract,
    output_root: Path,
) -> Path:
    track = track_directory(output_root, profile)
    gate_path = track / "test_gate.json"
    if not gate_path.is_file():
        raise FileNotFoundError(f"Run --stage test first; missing gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    configuration = resolve_configuration(profile)
    gated_adapter = (gate.get("binding") or {}).get("adapter") or {}
    if gated_adapter.get("backend") == "upstream_transformers" and configuration.backend == "vllm":
        configuration = _resolved_command_configuration(
            profile,
            backend_deviation=gated_adapter.get("backend_deviation"),
        )
    binding_value = binding(configuration, contract)
    gate_problems = test_gate_errors(gate, _digest(binding_value))
    if gate_problems:
        raise ValueError(
            "CV-Bench test gate is stale or incomplete: " + "; ".join(gate_problems)
        )
    capacity = int(gate["selected_concurrency"])
    gpu_audit = inspect_local_gpus(profile, configuration.backend)
    atomic_write_json(track / "full_gpu_preflight.json", gpu_audit)
    profiled = ProfiledContract(contract, profile)
    output = track / "predictions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(configuration.base_urls) == 2:
        metadata = _run_dual_shard_full(
            contract=contract,
            profiled=profiled,
            configuration=configuration,
            output=output,
            binding_value=binding_value,
            test_gate=gate_path,
            capacity=capacity,
        )
    else:
        adapter = build_adapter(configuration)
        metadata = run_recoverable_inference(
            contract=profiled,
            adapter=adapter,
            output=output,
            target_indices=list(range(OFFICIAL_TEST_SIZE)),
            benchmark="CV-Bench",
            split="test",
            official_size=OFFICIAL_TEST_SIZE,
            scorer_protocol=SCORER_PROTOCOL,
            workers=capacity,
            **_runtime_retry_policy(configuration.backend),
        )
        metadata_path = output.with_suffix(output.suffix + ".metadata.json")
        metadata = _enrich_metadata(
            metadata_path,
            contract=contract,
            configuration=configuration,
            binding_value=binding_value,
            test_gate=gate_path,
        )
    rows = read_jsonl(output)
    validation = validate_prediction_rows(
        rows,
        contract,
        prediction_path=output,
        allow_subset=False,
    )
    if not validation["passed"] or not metadata.get("publishable_inference"):
        raise RuntimeError("CV-Bench full inference did not pass its mandatory validator")
    metadata.setdefault("runtime", {})["gpu_preflight"] = gpu_audit
    atomic_write_json(output.with_suffix(output.suffix + ".metadata.json"), metadata)
    atomic_write_json(track / "prediction_validation.json", validation)
    print(f"[cv-bench] full-{OFFICIAL_TEST_SIZE} validation passed: {output}")
    return output


def _selected_profiles(args: argparse.Namespace, *, default_all: bool = False) -> list[CVBenchProfile]:
    if args.model:
        keys = [args.model]
    elif args.models:
        keys = [value.strip() for value in args.models.split(",") if value.strip()]
        if not keys:
            raise ValueError("--models selected no profiles")
    elif args.all or default_all:
        keys = list(PROFILE_SEQUENCE)
    else:
        raise ValueError("Select exactly one of --model, --models, or --all")
    return ordered_profiles(keys)


def _status(profile: CVBenchProfile, output_root: Path) -> dict[str, Any]:
    track = track_directory(output_root, profile)
    gate = track / "test_gate.json"
    predictions = track / "predictions.jsonl"
    validation = track / "prediction_validation.json"
    score_gate = track / "scores" / SCORER_PROTOCOL / "publication_gates.json"
    return {
        "profile": profile.key,
        "track": str(track),
        "test_gate_passed": gate.is_file()
        and bool(json.loads(gate.read_text(encoding="utf-8")).get("passed")),
        "full_predictions": predictions.is_file(),
        "full_validation_passed": validation.is_file()
        and bool(json.loads(validation.read_text(encoding="utf-8")).get("passed")),
        "score_publication_passed": score_gate.is_file()
        and bool(json.loads(score_gate.read_text(encoding="utf-8")).get("passed")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("test", "full"), default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=PROFILE_SEQUENCE)
    selection.add_argument("--models")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-root", default=os.environ.get("CVBENCH_DATASET_ROOT"))
    parser.add_argument("--output-root", default=os.environ.get("CVBENCH_OUTPUT_ROOT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for key in PROFILE_SEQUENCE:
            profile = get_profile(key)
            print(
                f"{profile.key}\t{profile.display_name}\t{profile.default_backend}\t"
                f"tp={profile.default_tensor_parallel_size}"
            )
        return
    profiles = _selected_profiles(args, default_all=args.status)
    if not args.output_root:
        raise ValueError("Set CVBENCH_OUTPUT_ROOT or pass --output-root")
    output_root = Path(args.output_root).resolve()
    if args.status:
        for profile in profiles:
            print(json.dumps(_status(profile, output_root), ensure_ascii=False, sort_keys=True))
        return
    if not args.stage and not args.check:
        raise ValueError("--stage test or --stage full is required")
    if args.dry_run:
        for profile in profiles:
            print(
                json.dumps(
                    {
                        "stage": args.stage or "check",
                        "profile": profile.key,
                        "track": str(track_directory(output_root, profile)),
                        "required_model_path_env": profile.model_path_env or None,
                        "required_command_env": (
                            f"CVBENCH_{_env_token(profile.key)}_COMMAND"
                            if profile.adapter_kind == "upstream_command"
                            else None
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        return
    if not args.dataset_root:
        raise ValueError("Set CVBENCH_DATASET_ROOT or pass --dataset-root")
    contract = CVBenchTestContract(args.dataset_root, verify_files=True)
    if args.check:
        dataset_manifest = contract.dataset_manifest(include_images=False)
        for profile in profiles:
            configuration = resolve_configuration(profile)
            print(
                json.dumps(
                    {
                        "passed": True,
                        "profile": profile.key,
                        "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
                        "binding_digest": _digest(binding(configuration, contract)),
                        "backend": configuration.backend,
                        "sharding": configuration.sharding,
                    },
                    ensure_ascii=False,
                )
            )
        return
    for profile in profiles:
        if args.stage == "test":
            run_test_stage(profile, contract, output_root)
        else:
            run_full_stage(profile, contract, output_root)


if __name__ == "__main__":
    main()
