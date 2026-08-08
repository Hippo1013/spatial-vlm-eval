"""Benchmark-neutral recoverable vision inference with strict provenance ownership.

Adapters receive only a structural index/image/question input. This module journals
every attempt, resumes completed indices, and asks the benchmark-owned contract to
materialize prediction rows only after every selected index succeeds.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ...benchmarks.msmu.data import OFFICIAL_TEST_SIZE, MSMUModelInput, MSMUTestContract
from ...benchmarks.msmu.scorer import SCORER_PROTOCOL

JOURNAL_SCHEMA_VERSION = 1
METADATA_SCHEMA_VERSION = 1
_REQUIRED_ADAPTER_METADATA = {
    "model",
    "model_revision",
    "backend",
    "profile",
    "inference_protocol",
    "chat_template",
    "image_processing",
    "decoding",
    "upstream",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One successful model call; empty text remains a successful response."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class RestrictedVisionInput(Protocol):
    """Structural input boundary shared by image-plus-prompt benchmarks.

    ``question`` remains the compatibility surface used by MSMU and CV-Bench.
    Benchmarks with a genuine system/user protocol may additionally expose
    ``system_prompt`` and ``user_prompt``; adapters must feature-detect those
    fields instead of assuming that every benchmark owns a system message.
    """

    index: int
    image: Any
    question: str


class InferenceAdapter:
    """Small interface implemented by model-family adapters."""

    batch_size = 1
    supports_concurrency = False

    @property
    def inference_protocol(self) -> str:
        return str(self.metadata()["inference_protocol"])

    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def generate(self, model_input: RestrictedVisionInput) -> GenerationResult:
        raise NotImplementedError

    def generate_batch(self, model_inputs: Sequence[RestrictedVisionInput]) -> list[GenerationResult]:
        return [self.generate(model_input) for model_input in model_inputs]

    def close(self) -> None:
        """Release optional model resources."""


def parse_indices(specification: str | None, size: int) -> list[int]:
    """Parse comma-separated indices and inclusive ranges such as ``1,4-6``."""

    if specification is None or not specification.strip():
        return list(range(int(size)))
    selected: list[int] = []
    for piece in specification.split(","):
        token = piece.strip()
        if not token:
            raise ValueError("Empty token in --indices")
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Descending index range is not allowed: {token}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(token))
    if len(selected) != len(set(selected)):
        raise ValueError("--indices contains duplicate indices")
    outside = [index for index in selected if not 0 <= index < size]
    if outside:
        raise ValueError(f"Indices outside [0,{size}): {outside[:20]}")
    return selected


def select_target_indices(
    size: int,
    *,
    indices: str | None = None,
    limit: int | None = None,
) -> list[int]:
    selected = parse_indices(indices, size)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("No MSMU target indices were selected")
    return selected


def pixel_sha256(image: Any) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(b"RGB\0")
    digest.update(str(rgb.size[0]).encode("ascii"))
    digest.update(b"x")
    digest.update(str(rgb.size[1]).encode("ascii"))
    digest.update(b"\0")
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def input_audit(model_input: RestrictedVisionInput, adapter_metadata: dict[str, Any]) -> dict[str, Any]:
    rgb = model_input.image.convert("RGB")
    system_prompt = getattr(model_input, "system_prompt", None)
    user_prompt = getattr(model_input, "user_prompt", None)
    if system_prompt is not None or user_prompt is not None:
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise TypeError("System/user benchmark prompts must both be strings")
        question_for_audit = user_prompt
    else:
        question_for_audit = str(model_input.question)
    audit = {
        "index": int(model_input.index),
        "question": question_for_audit,
        "image_count": 1,
        "image_mode": "RGB",
        "image_size": [int(rgb.size[0]), int(rgb.size[1])],
        "image_pixel_sha256": pixel_sha256(rgb),
        "profile": adapter_metadata["profile"],
        "inference_protocol": adapter_metadata["inference_protocol"],
        "chat_template": adapter_metadata["chat_template"],
    }
    if system_prompt is not None or user_prompt is not None:
        audit["system_prompt"] = system_prompt
        audit["user_prompt"] = user_prompt
        system_role_supported = bool(adapter_metadata.get("system_role_supported", True))
        audit["system_role_supported"] = system_role_supported
        audit["prompt_roles"] = ["system", "user"] if system_role_supported else ["user"]
    return audit


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _run_signature(run_identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(run_identity).encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:  # noqa: BLE001 - clean temp files even on interruption.
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rendered = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    _atomic_write_text(path, rendered)


def default_journal_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".journal.jsonl")


def default_metadata_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".metadata.json")


def _redact_error(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    message = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", message)
    message = re.sub(r"\b(?:sk|AIza)[-_A-Za-z0-9]{12,}\b", "[REDACTED]", message)
    message = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "data:image/[REDACTED]", message)
    return message[:1000]


class PredictionJournal:
    """Append-only, fsync-backed journal scoped to a deterministic run signature."""

    def __init__(self, path: Path, run_signature: str, *, resume: bool) -> None:
        self.path = path
        self.run_signature = run_signature
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not resume and self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"Journal already exists and --no-resume was requested: {self.path}")

    def successful_results(self, target_indices: set[int]) -> dict[int, GenerationResult]:
        successes: dict[int, GenerationResult] = {}
        if not self.path.exists():
            return successes
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed journal JSON at line {line_number}") from exc
                if event.get("schema_version") != JOURNAL_SCHEMA_VERSION:
                    raise ValueError(f"Unsupported journal schema at line {line_number}")
                if event.get("run_signature") != self.run_signature:
                    raise ValueError(
                        f"Journal run signature mismatch at line {line_number}; use a separate output directory"
                    )
                try:
                    index = int(event["index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid journal index at line {line_number}") from exc
                if index not in target_indices:
                    raise ValueError(f"Journal index {index} is outside this run's targets")
                if event.get("status") != "success":
                    continue
                if index in successes:
                    raise ValueError(f"Duplicate successful journal index: {index}")
                if "prediction" not in event:
                    raise ValueError(f"Successful journal event lacks prediction at line {line_number}")
                successes[index] = GenerationResult(
                    text=str(event["prediction"]),
                    metadata=dict(event.get("generation") or {}),
                    warnings=tuple(str(item) for item in event.get("warnings") or []),
                )
        return successes

    def append_success(
        self,
        *,
        model_input: RestrictedVisionInput,
        attempt: int,
        audit: dict[str, Any],
        result: GenerationResult,
        seeded_from: dict[str, Any] | None = None,
    ) -> None:
        event = {
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "run_signature": self.run_signature,
                "timestamp": utc_now(),
                "status": "success",
                "index": int(model_input.index),
                "attempt": int(attempt),
                "audit": audit,
                "prediction": str(result.text).strip(),
                "generation": result.metadata,
                "warnings": list(result.warnings),
            }
        if seeded_from is not None:
            event["seeded_from"] = dict(seeded_from)
        self._append(event)

    def append_failure(
        self,
        *,
        model_input: RestrictedVisionInput,
        attempt: int,
        audit: dict[str, Any],
        error: BaseException,
    ) -> None:
        self._append(
            {
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "run_signature": self.run_signature,
                "timestamp": utc_now(),
                "status": "failure",
                "index": int(model_input.index),
                "attempt": int(attempt),
                "audit": audit,
                "error": _redact_error(error),
            }
        )

    def _append(self, event: dict[str, Any]) -> None:
        rendered = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())


def _package_versions() -> dict[str, str | None]:
    names = [
        "spatial-vlm-eval",
        "datasets",
        "numpy",
        "torch",
        "torchvision",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "peft",
        "accelerate",
        "vllm",
        "Pillow",
        "qwen-vl-utils",
        "timm",
        "opencv-python",
        "pycocotools",
        "flash-attn",
        "xformers",
        "s2wrapper",
        "vila",
        "bunny",
        "depth-pro",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_metadata() -> list[dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_mib": round(
                    torch.cuda.get_device_properties(index).total_memory / (1024 * 1024),
                    3,
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
    except (ImportError, OSError, RuntimeError):
        return []


def validate_adapter_metadata(metadata: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_ADAPTER_METADATA - set(metadata))
    if missing:
        raise ValueError(f"Adapter metadata is missing required fields: {missing}")
    if not isinstance(metadata.get("decoding"), dict):
        raise TypeError("Adapter decoding metadata must be an object")
    if not isinstance(metadata.get("image_processing"), dict):
        raise TypeError("Adapter image_processing metadata must be an object")
    if not isinstance(metadata.get("upstream"), dict):
        raise TypeError("Adapter upstream metadata must be an object")


def run_recoverable_inference(
    *,
    contract: Any,
    adapter: InferenceAdapter,
    output: str | Path,
    target_indices: Sequence[int],
    benchmark: str,
    split: str,
    official_size: int,
    scorer_protocol: str,
    journal_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    retries: int = 2,
    retry_missing_passes: int = 0,
    workers: int = 1,
    resume: bool = True,
    seed_successes: Mapping[int, GenerationResult] | None = None,
    seed_provenance: dict[str, Any] | None = None,
    initial_serial_requests: int = 0,
) -> dict[str, Any]:
    """Run, resume, and atomically finalize one restricted benchmark split."""

    started_wall = time.monotonic()
    started_at = utc_now()
    output_path = Path(output).resolve()
    resolved_journal = (
        Path(journal_path).resolve() if journal_path is not None else default_journal_path(output_path)
    )
    resolved_metadata = (
        Path(metadata_path).resolve() if metadata_path is not None else default_metadata_path(output_path)
    )
    resolved_paths = {output_path, resolved_journal, resolved_metadata}
    if len(resolved_paths) != 3:
        raise ValueError("output, journal, and metadata paths must be three distinct files")
    selected = [int(index) for index in target_indices]
    locked_official_size = int(official_size)
    if locked_official_size <= 0:
        raise ValueError("official_size must be positive")
    if len(selected) != len(set(selected)):
        raise ValueError("Target indices contain duplicates")
    outside = [index for index in selected if not 0 <= index < len(contract)]
    if outside:
        raise ValueError(f"Target indices outside dataset: {outside[:20]}")
    if not selected:
        raise ValueError("No target indices")
    selected = sorted(selected)
    retry_count = int(retries)
    missing_retry_pass_count = int(retry_missing_passes)
    worker_count = int(workers)
    initial_serial_request_count = int(initial_serial_requests)
    if retry_count < 0:
        raise ValueError("retries must be non-negative")
    if missing_retry_pass_count < 0:
        raise ValueError("retry_missing_passes must be non-negative")
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    if initial_serial_request_count < 0:
        raise ValueError("initial_serial_requests must be non-negative")
    if seed_successes and seed_provenance is None:
        raise ValueError("seed_provenance is required when seed_successes are supplied")

    adapter_metadata = dict(adapter.metadata())
    validate_adapter_metadata(adapter_metadata)
    batch_size = int(getattr(adapter, "batch_size", 1))
    if batch_size <= 0:
        raise ValueError("adapter.batch_size must be positive")
    if worker_count > 1 and (batch_size != 1 or not adapter.supports_concurrency):
        raise ValueError("workers > 1 requires a thread-safe single-sample adapter")

    runtime_packages = _package_versions()
    run_identity = {
        "benchmark": str(benchmark),
        "dataset_root": str(contract.dataset_root),
        "dataset_fingerprint": contract.dataset_fingerprint,
        "split": str(split),
        "target_indices": selected,
        "adapter": adapter_metadata,
        "python": sys.version.split()[0],
        "runtime_packages": runtime_packages,
    }
    signature = _run_signature(run_identity)
    journal = PredictionJournal(resolved_journal, signature, resume=resume)
    successes = journal.successful_results(set(selected))
    seeded_indices: list[int] = []
    for raw_index, raw_result in sorted((seed_successes or {}).items()):
        index = int(raw_index)
        if index not in selected:
            raise ValueError(f"Seed success index {index} is outside this run's targets")
        if not isinstance(raw_result, GenerationResult):
            raise TypeError(f"Seed success {index} is not a GenerationResult")
        normalized_seed = GenerationResult(
            text=str(raw_result.text).strip(),
            metadata=dict(raw_result.metadata),
            warnings=tuple(str(item) for item in raw_result.warnings),
        )
        existing = successes.get(index)
        if existing is not None:
            if existing != normalized_seed:
                raise ValueError(f"Existing journal success differs from supplied seed at index {index}")
            continue
        model_input = contract.model_input(index)
        journal.append_success(
            model_input=model_input,
            attempt=0,
            audit=input_audit(model_input, adapter_metadata),
            result=normalized_seed,
            seeded_from=seed_provenance,
        )
        successes[index] = normalized_seed
        seeded_indices.append(index)
    terminal_failures: set[int] = set()
    terminal_failure_lock = threading.Lock()
    if output_path.exists() and set(successes) != set(selected):
        raise FileExistsError(
            f"Refusing to overwrite an existing prediction file before the journal is complete: {output_path}"
        )
    def run_batch(
        batch: Sequence[RestrictedVisionInput],
        *,
        attempt_offset: int = 0,
    ) -> dict[int, GenerationResult]:
        audits = [input_audit(model_input, adapter_metadata) for model_input in batch]
        last_error: Exception | None = None
        for attempt in range(attempt_offset + 1, attempt_offset + retry_count + 2):
            try:
                generated = adapter.generate_batch(batch)
                if len(generated) != len(batch):
                    raise ValueError(f"Adapter returned {len(generated)} results for a batch of {len(batch)}")
                normalized_results: list[GenerationResult] = []
                for result in generated:
                    if not isinstance(result, GenerationResult):
                        raise TypeError("Adapter returned a non-GenerationResult value")
                    warnings = tuple(str(item) for item in result.warnings)
                    text = str(result.text).strip()
                    if not text and not any("empty" in warning.lower() for warning in warnings):
                        warnings += ("model returned an empty text completion",)
                    normalized = GenerationResult(
                        text=text,
                        metadata=dict(result.metadata),
                        warnings=warnings,
                    )
                    normalized_results.append(normalized)
            except Exception as exc:  # noqa: BLE001 - adapters can fail through third-party APIs.
                last_error = exc
                for model_input, audit in zip(batch, audits):
                    journal.append_failure(
                        model_input=model_input,
                        attempt=attempt,
                        audit=audit,
                        error=exc,
                    )
                status_code = getattr(exc, "status_code", None)
                explicit_retryable = getattr(exc, "retryable", None)
                retryable = (
                    bool(explicit_retryable)
                    if explicit_retryable is not None
                    else status_code is None
                    or status_code == 429
                    or (isinstance(status_code, int) and status_code >= 500)
                )
                if not retryable:
                    with terminal_failure_lock:
                        terminal_failures.update(int(model_input.index) for model_input in batch)
                    break
                if attempt < attempt_offset + retry_count + 1 and (
                    status_code == 429 or (isinstance(status_code, int) and status_code >= 500)
                ):
                    headers = getattr(exc, "response_headers", {}) or {}
                    retry_after = headers.get("retry-after")
                    try:
                        delay = float(retry_after) if retry_after is not None else 2.0 ** (
                            attempt - attempt_offset - 1
                        )
                    except (TypeError, ValueError):
                        delay = 2.0 ** (attempt - attempt_offset - 1)
                    time.sleep(min(max(delay, 0.0), 30.0))
                continue

            # Journal writes are intentionally outside the retry handler. If durable
            # persistence fails, abort instead of reissuing a paid/model request and
            # risking duplicate successful events.
            resolved: dict[int, GenerationResult] = {}
            for model_input, audit, normalized in zip(batch, audits, normalized_results):
                journal.append_success(
                    model_input=model_input,
                    attempt=attempt,
                    audit=audit,
                    result=normalized,
                )
                resolved[model_input.index] = normalized
            return resolved
        assert last_error is not None
        return {}

    def run_indices(indices: Sequence[int], *, attempt_offset: int) -> None:
        if worker_count == 1:
            for start in range(0, len(indices), batch_size):
                batch = contract.model_inputs(indices[start : start + batch_size])
                successes.update(run_batch(batch, attempt_offset=attempt_offset))
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                pending_iterator = iter(indices)
                futures: dict[Any, int] = {}

                def submit_next() -> bool:
                    try:
                        index = next(pending_iterator)
                    except StopIteration:
                        return False
                    model_input = contract.model_input(index)
                    futures[
                        executor.submit(
                            run_batch,
                            [model_input],
                            attempt_offset=attempt_offset,
                        )
                    ] = index
                    return True

                for _ in range(worker_count):
                    if not submit_next():
                        break
                while futures:
                    future = next(as_completed(tuple(futures)))
                    futures.pop(future)
                    successes.update(future.result())
                    submit_next()

    try:
        if initial_serial_request_count:
            initial_pending = [index for index in selected if index not in successes]
            for index in initial_pending[:initial_serial_request_count]:
                initial_result = run_batch([contract.model_input(index)])
                successes.update(initial_result)
                if index not in initial_result:
                    raise RuntimeError(
                        "Initial serial compatibility request failed; "
                        "no concurrent continuation requests were issued"
                    )
        for pass_index in range(missing_retry_pass_count + 1):
            pending_indices = [
                index
                for index in selected
                if index not in successes and index not in terminal_failures
            ]
            if not pending_indices:
                break
            run_indices(
                pending_indices,
                attempt_offset=pass_index * (retry_count + 1),
            )
    finally:
        adapter.close()

    missing = sorted(set(selected) - set(successes))
    if missing:
        raise RuntimeError(
            f"Inference incomplete after retries; missing {len(missing)} indices: {missing[:20]}. "
            f"No prediction JSONL was finalized; resume from {resolved_journal}."
        )

    rows = [contract.prediction_row(index, successes[index].text) for index in selected]
    atomic_write_jsonl(output_path, rows)
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    empty_indices = [index for index in selected if not successes[index].text.strip()]
    warning_count = sum(len(successes[index].warnings) for index in selected)
    is_full_split = len(contract) == locked_official_size and selected == list(
        range(locked_official_size)
    )
    finished_at = utc_now()
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "inference_protocol": adapter_metadata["inference_protocol"],
        "scorer_protocol": str(scorer_protocol),
        "model": adapter_metadata,
        "dataset": {
            "benchmark": str(benchmark),
            "root": str(contract.dataset_root),
            "split": str(split),
            "fingerprint": contract.dataset_fingerprint,
            "official_test_size": locked_official_size,
            "loaded_size": len(contract),
            "target_indices": selected,
            "num_targets": len(selected),
            "is_subset": not is_full_split,
        },
        "output": str(output_path),
        "output_sha256": output_sha256,
        "journal": str(resolved_journal),
        "run_signature": signature,
        "num_predictions": len(rows),
        "empty_prediction_indices": empty_indices,
        "adapter_warning_count": warning_count,
        "publishable_inference": is_full_split,
        "resume": {
            "seeded_success_count": len(seed_successes or {}),
            "newly_seeded_indices": seeded_indices,
            "seed_provenance": seed_provenance,
            "initial_serial_requests": initial_serial_request_count,
        },
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(time.monotonic() - started_wall, 6),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "packages": runtime_packages,
            "gpus": _gpu_metadata(),
            "retry_policy": {
                "retries_per_pass": retry_count,
                "retry_missing_passes": missing_retry_pass_count,
            },
        },
    }
    atomic_write_json(resolved_metadata, metadata)
    return metadata


def run_msmu_inference(
    *,
    contract: MSMUTestContract,
    adapter: InferenceAdapter,
    output: str | Path,
    target_indices: Sequence[int],
    journal_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    retries: int = 2,
    retry_missing_passes: int = 0,
    workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Compatibility wrapper for the canonical MSMU recoverable runner."""

    return run_recoverable_inference(
        contract=contract,
        adapter=adapter,
        output=output,
        target_indices=target_indices,
        benchmark="MSMU-Bench",
        split="test",
        official_size=OFFICIAL_TEST_SIZE,
        scorer_protocol=SCORER_PROTOCOL,
        journal_path=journal_path,
        metadata_path=metadata_path,
        retries=retries,
        retry_missing_passes=retry_missing_passes,
        workers=workers,
        resume=resume,
    )
