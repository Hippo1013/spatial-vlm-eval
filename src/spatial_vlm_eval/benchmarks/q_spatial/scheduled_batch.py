"""Two-GPU/API staged scheduler for the 20 runnable Q-Spatial tracks."""

from __future__ import annotations

import argparse
import fcntl
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from ...models.common.runtime import atomic_write_json, utc_now
from .command_adapter import load_generation_manifest
from .data import DATASET_FILES, DATASET_REVISION, OFFICIAL_TEST_SIZE, QSpatialTestContract
from .inference import (
    _digest,
    _file_digest,
    binding,
    resolve_configuration,
    test_gate_errors,
    track_directory,
)
from .prediction_validation import read_jsonl, validate_prediction_rows
from .profiles import PROFILE_SEQUENCE, PROFILES, QSpatialProfile
from .scorer import SCORER_PROTOCOL, inference_metadata_scorer_protocol_is_compatible
from .specialized_runner import adapter_digest as specialized_adapter_digest

SCHEDULE_PROTOCOL = "q_spatial_2xa800_staged_lanes_v1"
EXCLUDED_RESOURCE_PROFILE = "internvl3_78b"
LANE_PHASE_BARRIER = {"gpu0": "dual", "gpu1": "dual"}


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    phase: str
    lane: str
    order: int
    profile: str
    gpu_ids: tuple[int, ...]
    port: int | None


def _jobs(
    phase: str,
    lane: str,
    profiles: tuple[str, ...],
    gpu_ids: tuple[int, ...],
    port: int | None,
) -> tuple[ScheduledJob, ...]:
    return tuple(
        ScheduledJob(phase, lane, order, profile, gpu_ids, port)
        for order, profile in enumerate(profiles, start=1)
    )


SCHEDULE: tuple[ScheduledJob, ...] = (
    *_jobs(
        "A",
        "dual",
        ("internvl3_38b", "llava_next_yi_34b", "qwen3_vl_32b"),
        (0, 1),
        18101,
    ),
    *_jobs(
        "A",
        "api",
        ("gpt5_openrouter_non_zdr", "gemini31pro_openrouter_non_zdr"),
        (),
        None,
    ),
    *_jobs(
        "B",
        "gpu0",
        (
            "3dthinker_rgb",
            "spatialrgpt_rgb",
            "llava_next_mistral_7b",
            "internvl3_8b",
            "qwen3_vl_8b",
            "qwen3_vl_4b",
            "qwen3_vl_2b",
        ),
        (0,),
        18101,
    ),
    *_jobs(
        "B",
        "gpu1",
        (
            "ssr_native",
            "ssr_rgb",
            "hispatial3b_moge2_xyz",
            "spatialbot_zoedepth",
            "spatialbot_rgb",
            "robobrain25_8b_nv_rgb",
            "robobrain25_8b_mt_rgb",
            "spatialladder3b_rgb",
        ),
        (1,),
        18102,
    ),
)
LANE_ORDER = ("dual", "api", "gpu0", "gpu1")


class SchedulerInterrupted(RuntimeError):
    def __init__(self, signum: int) -> None:
        super().__init__(f"scheduled batch interrupted by signal {signum}")
        self.signum = signum


def jobs_for_lane(lane: str) -> tuple[ScheduledJob, ...]:
    return tuple(job for job in SCHEDULE if job.lane == lane)


def schedule_errors() -> list[str]:
    errors: list[str] = []
    scheduled = [job.profile for job in SCHEDULE]
    expected = set(PROFILE_SEQUENCE) - {EXCLUDED_RESOURCE_PROFILE}
    if len(SCHEDULE) != 20:
        errors.append(f"schedule must contain exactly 20 runnable tracks, got {len(SCHEDULE)}")
    if len(scheduled) != len(set(scheduled)):
        errors.append("schedule contains duplicate profiles")
    if set(scheduled) != expected:
        errors.append(
            f"schedule coverage differs: missing={sorted(expected - set(scheduled))}, "
            f"extra={sorted(set(scheduled) - expected)}"
        )
    if tuple(job.profile for job in jobs_for_lane("api")) != (
        "gpt5_openrouter_non_zdr",
        "gemini31pro_openrouter_non_zdr",
    ):
        errors.append("API lane must serialize GPT-5 before Gemini 3.1 Pro")
    for job in jobs_for_lane("dual"):
        if job.phase != "A" or job.gpu_ids != (0, 1):
            errors.append(f"dual lane resource conflict: {job}")
    for lane, expected_gpu in (("gpu0", (0,)), ("gpu1", (1,))):
        for job in jobs_for_lane(lane):
            if job.phase != "B" or job.gpu_ids != expected_gpu:
                errors.append(f"{lane} resource conflict: {job}")
        if LANE_PHASE_BARRIER.get(lane) != "dual":
            errors.append(f"{lane} must depend only on the successful dual lane")
    if any(job.gpu_ids for job in jobs_for_lane("api")):
        errors.append("API lane must not reserve a local GPU")
    for job in SCHEDULE:
        profile = PROFILES.get(job.profile)
        if profile is None:
            continue
        if job.lane == "dual" and profile.default_tensor_parallel_size != 2:
            errors.append(f"dual lane profile must use TP=2: {job.profile}")
        if job.lane in {"gpu0", "gpu1"} and profile.default_tensor_parallel_size != 1:
            errors.append(f"single-GPU lane profile must use TP=1: {job.profile}")
        if job.lane == "api" and profile.default_backend != "openrouter":
            errors.append(f"API lane contains a non-OpenRouter profile: {job.profile}")
    return errors


def schedule_payload(repository_root: Path) -> dict[str, Any]:
    from . import inference, profiles, scheduled_watcher

    payload = {
        "schema_version": 1,
        "schedule_protocol": SCHEDULE_PROTOCOL,
        "excluded_resource_profile": EXCLUDED_RESOURCE_PROFILE,
        "barriers": LANE_PHASE_BARRIER,
        "jobs": [
            {
                **asdict(job),
                "gpu_ids": list(job.gpu_ids),
                "profile_revision": PROFILES[job.profile].revision,
                "inference_protocol": PROFILES[job.profile].inference_protocol,
                "registry_digest": PROFILES[job.profile].registry_digest,
            }
            for job in SCHEDULE
        ],
    }
    payload["plan_digest"] = _digest(payload)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable"
    payload["repository_head"] = head
    payload["source_sha256"] = {
        path.name: _file_digest(path)
        for path in (
            Path(__file__),
            Path(inspect.getfile(inference)),
            Path(inspect.getfile(profiles)),
            Path(inspect.getfile(scheduled_watcher)),
        )
    }
    return payload


def _profile_token(profile: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", profile).upper()


PROFILE_VLLM_MAX_MODEL_LEN = {
    "llava_next_yi_34b": 4096,
    "llava_next_mistral_7b": 4096,
}


def environment_for_job(job: ScheduledJob) -> dict[str, str]:
    environment = dict(os.environ)
    profile = PROFILES[job.profile]
    token = _profile_token(job.profile)
    if job.gpu_ids:
        gpu_text = ",".join(str(value) for value in job.gpu_ids)
        environment[f"QSPATIAL_{token}_GPU_IDS"] = gpu_text
        environment["CUDA_VISIBLE_DEVICES"] = gpu_text
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    if profile.default_backend == "vllm":
        assert job.port is not None
        environment[f"QSPATIAL_{token}_BASE_URLS"] = f"http://127.0.0.1:{job.port}/v1"
        if job.profile in PROFILE_VLLM_MAX_MODEL_LEN:
            environment["QSPATIAL_VLLM_MAX_MODEL_LEN"] = str(
                PROFILE_VLLM_MAX_MODEL_LEN[job.profile]
            )
    return environment


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def reusable_gate_errors(
    profile: QSpatialProfile,
    contract: QSpatialTestContract,
    output_root: Path,
) -> list[str]:
    gate_path = track_directory(output_root, profile) / "test_gate.json"
    gate = _load_json(gate_path)
    if gate is None:
        return [f"missing or malformed test gate: {gate_path}"]
    configuration = resolve_configuration(profile)
    return test_gate_errors(gate, _digest(binding(configuration, contract)))


def complete_result_errors(
    profile: QSpatialProfile,
    contract: QSpatialTestContract,
    output_root: Path,
) -> list[str]:
    errors: list[str] = []
    track = track_directory(output_root, profile)
    prediction = track / "predictions.jsonl"
    metadata_path = prediction.with_suffix(prediction.suffix + ".metadata.json")
    validation_path = track / "prediction_validation.json"
    if not prediction.is_file():
        return [f"missing predictions: {prediction}"]
    try:
        rows = read_jsonl(prediction)
        validation = validate_prediction_rows(
            rows,
            contract,
            prediction_path=prediction,
            allow_subset=False,
        )
    except Exception as exc:  # noqa: BLE001 - completion audit is fail closed.
        return [f"prediction validation raised {type(exc).__name__}: {exc}"]
    if validation.get("passed") is not True:
        errors.append("full 271-row validator did not pass")
    stored_validation = _load_json(validation_path)
    if stored_validation is None or stored_validation.get("passed") is not True:
        errors.append("stored prediction_validation.json is missing or not passed")
    elif (
        stored_validation.get("num_prediction_rows") != OFFICIAL_TEST_SIZE
        or stored_validation.get("dataset_fingerprint") != contract.dataset_fingerprint
    ):
        errors.append("stored prediction validation does not bind the current 271-row dataset")
    metadata = _load_json(metadata_path)
    if metadata is None:
        return [*errors, "missing or malformed inference metadata"]
    configuration = resolve_configuration(profile)
    expected_binding = binding(configuration, contract)
    model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
    expected_files = {item.name: item.sha256 for item in DATASET_FILES}
    checks = {
        "publishable inference": metadata.get("publishable_inference") is True,
        "prediction count": metadata.get("num_predictions") == OFFICIAL_TEST_SIZE,
        "profile": model.get("profile") == profile.key,
        "model revision": model.get("model_revision") == profile.revision,
        "inference protocol": metadata.get("inference_protocol") == profile.inference_protocol,
        "scorer protocol": inference_metadata_scorer_protocol_is_compatible(
            metadata.get("scorer_protocol")
        ),
        "dataset revision": dataset.get("revision") == DATASET_REVISION,
        "dataset fingerprint": dataset.get("fingerprint") == contract.dataset_fingerprint,
        "dataset files": dataset.get("files") == expected_files,
        "official size": dataset.get("official_test_size") == OFFICIAL_TEST_SIZE,
        "binding": metadata.get("binding") == expected_binding,
        "binding digest": metadata.get("binding_digest") == _digest(expected_binding),
        "output hash": metadata.get("output_sha256") == _file_digest(prediction),
    }
    errors.extend(f"metadata {label} mismatch" for label, passed in checks.items() if not passed)
    errors.extend(f"test gate: {error}" for error in reusable_gate_errors(profile, contract, output_root))
    return errors


def _port_is_available(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _gpu_inventory() -> tuple[list[dict[str, Any]], list[str]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise FileNotFoundError("scheduled Q-Spatial preflight requires nvidia-smi")
    inventory_query = subprocess.run(
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
    for line in inventory_query.stdout.splitlines():
        fields = [value.strip() for value in line.split(",", 5)]
        if len(fields) != 6:
            raise ValueError(f"unexpected nvidia-smi inventory row: {line!r}")
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
    return inventory, [line for line in process_query.stdout.splitlines() if line.strip()]


def preflight(repository_root: Path, *, require_api_key: bool) -> dict[str, Any]:
    errors = schedule_errors()
    warnings: list[str] = []
    output_value = os.environ.get("QSPATIAL_OUTPUT_ROOT", "")
    parquet_value = os.environ.get("QSPATIAL_PARQUET_ROOT", "")
    rgb_value = os.environ.get("QSPATIAL_SCANNET_RGB_ROOT", "")
    if not output_value:
        errors.append("QSPATIAL_OUTPUT_ROOT is unset")
    if not parquet_value:
        errors.append("QSPATIAL_PARQUET_ROOT is unset")
    if not rgb_value:
        errors.append("QSPATIAL_SCANNET_RGB_ROOT is unset")
    output_root = Path(output_value).expanduser().resolve() if output_value else None
    if output_root and (output_root == repository_root or repository_root in output_root.parents):
        errors.append("QSPATIAL_OUTPUT_ROOT must be outside the repository")

    dataset_report: dict[str, Any] | None = None
    contract: QSpatialTestContract | None = None
    if parquet_value and rgb_value:
        try:
            contract = QSpatialTestContract(parquet_value, rgb_value)
            dataset_report = contract.dataset_manifest(include_images=True)
        except Exception as exc:  # noqa: BLE001 - aggregate read-only preflight errors.
            errors.append(f"dataset contract failed: {type(exc).__name__}: {exc}")

    gpu_report: dict[str, Any] | None = None
    try:
        inventory, processes = _gpu_inventory()
        by_index = {item["index"]: item for item in inventory}
        selected = [by_index.get(index) for index in (0, 1)]
        if any(item is None for item in selected):
            errors.append("schedule requires physical GPU ids 0 and 1")
        else:
            assert all(item is not None for item in selected)
            wrong_model = [item["index"] for item in selected if "A800" not in item["name"]]
            undersized = [item["index"] for item in selected if item["memory_total_mib"] < 79_000]
            selected_uuids = {item["uuid"] for item in selected}
            occupied = [line for line in processes if line.split(",", 1)[0].strip() in selected_uuids]
            busy = [item["index"] for item in selected if item["utilization_percent"] > 10]
            low_free = [item["index"] for item in selected if item["memory_free_mib"] < 60_000]
            if wrong_model:
                errors.append(f"scheduled GPUs are not A800: {wrong_model}")
            if undersized:
                errors.append(f"scheduled GPUs are smaller than 80GB: {undersized}")
            if occupied or busy or low_free:
                message = (
                    "scheduled GPUs are not execution-idle; existing processes were untouched: "
                    f"compute_processes={bool(occupied)}, busy={busy}, low_free={low_free}"
                )
                if require_api_key:
                    errors.append(message)
                else:
                    warnings.append(message)
        gpu_report = {"inventory": inventory, "compute_processes": processes, "selected_gpu_ids": [0, 1]}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"GPU inventory failed: {type(exc).__name__}: {exc}")

    port_report = {str(port): _port_is_available(port) for port in (18101, 18102)}
    for port, available in port_report.items():
        if not available:
            errors.append(f"port {port} is occupied; existing process was untouched")

    required_commands = ("tmux",)
    for command in required_commands:
        if shutil.which(command) is None:
            errors.append(f"required command is unavailable: {command}")
    if require_api_key and shutil.which("tmux") is not None:
        tmux_check = subprocess.run(
            ["tmux", "display-message", "-p", "#{session_id}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if tmux_check.returncode != 0 or not tmux_check.stdout.strip():
            errors.append("formal execution must run inside an active tmux session")
    if require_api_key and not os.environ.get("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY is required for authorized execution")

    profile_reports: list[dict[str, Any]] = []
    if contract is not None:
        for job in SCHEDULE:
            profile = PROFILES[job.profile]
            environment = environment_for_job(job)
            report: dict[str, Any] = {"profile": profile.key, "lane": job.lane, "passed": False}
            try:
                with _temporary_environment(environment):
                    configuration = resolve_configuration(profile)
                    if profile.default_backend == "vllm":
                        model_value = os.environ.get(profile.model_path_env, "")
                        model_path = Path(model_value).expanduser()
                        if not model_value or not model_path.exists():
                            raise FileNotFoundError(
                                f"configured model path is missing: {profile.model_path_env}={model_path}"
                            )
                    elif profile.adapter_kind == "upstream_command":
                        model_value = os.environ.get(profile.model_path_env, "")
                        model_path = Path(model_value).expanduser()
                        if not model_value or not model_path.exists():
                            raise FileNotFoundError(
                                f"configured model path is missing: {profile.model_path_env}={model_path}"
                            )
                        command = shlex.split(configuration.command or "")
                        if not command or (not Path(command[0]).exists() and shutil.which(command[0]) is None):
                            raise FileNotFoundError("configured specialized runner executable is unavailable")
                        actual_digest = specialized_adapter_digest(profile)
                        if actual_digest != configuration.adapter_digest:
                            raise ValueError(
                                f"adapter digest mismatch: configured={configuration.adapter_digest}, "
                                f"current={actual_digest}"
                            )
                        load_generation_manifest(
                            profile,
                            os.environ.get(f"QSPATIAL_{_profile_token(profile.key)}_GENERATION_MANIFEST"),
                        )
                    binding_value = binding(configuration, contract)
                report.update(
                    {
                        "passed": True,
                        "model_revision": profile.revision,
                        "adapter_digest": configuration.adapter_digest,
                        "binding_digest": _digest(binding_value),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                report["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(f"profile {profile.key} preflight failed: {report['error']}")
            profile_reports.append(report)

    return {
        "passed": not errors,
        "schedule_protocol": SCHEDULE_PROTOCOL,
        "plan_digest": schedule_payload(repository_root)["plan_digest"],
        "num_scheduled_profiles": len(SCHEDULE),
        "dataset": dataset_report,
        "gpu": gpu_report,
        "ports": port_report,
        "api_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
        "api_key_required": require_api_key,
        "profiles": profile_reports,
        "warnings": warnings,
        "errors": errors,
    }


class _temporary_environment:
    def __init__(self, replacement: dict[str, str]) -> None:
        self.replacement = replacement
        self.original: dict[str, str] | None = None

    def __enter__(self) -> None:
        self.original = dict(os.environ)
        os.environ.clear()
        os.environ.update(self.replacement)

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        assert self.original is not None
        os.environ.clear()
        os.environ.update(self.original)


def _append_status(
    status_path: Path,
    *,
    run_id: str,
    phase: str,
    lane: str,
    profile: str,
    state: str,
    detail: str,
) -> None:
    line = "\t".join((run_id, phase, lane, profile, state, utc_now(), detail.replace("\t", " ")))
    with status_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    _signal_watcher(run_id, lane)


def _signal_watcher(run_id: str, lane: str) -> None:
    tmux = shutil.which("tmux")
    if tmux:
        subprocess.run(
            [tmux, "wait-for", "-S", f"qspatial-{run_id}-{lane}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _group_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_owned_group(process: subprocess.Popen[Any] | None, timeout: int) -> None:
    if process is None:
        return
    pid = process.pid
    if _group_alive(pid):
        os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()  # Reap an exited group leader so a zombie is not treated as live.
        if not _group_alive(pid):
            break
        time.sleep(1)
    if _group_alive(pid):
        os.killpg(pid, signal.SIGKILL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _stop_recorded_group(pid: Any, timeout: int) -> None:
    if not isinstance(pid, int) or pid <= 0:
        return
    if _group_alive(pid):
        os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _group_alive(pid) and time.monotonic() < deadline:
        time.sleep(1)
    if _group_alive(pid):
        os.killpg(pid, signal.SIGKILL)


def _run_owned(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
    active_path: Path,
    active: dict[str, Any],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=environment["QSPATIAL_REPO_ROOT"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        active["worker_process_group"] = process.pid
        atomic_write_json(active_path, active)
        try:
            return process.wait()
        finally:
            active["worker_process_group"] = None
            atomic_write_json(active_path, active)


def _endpoint_ready(
    repository_root: Path,
    python: str,
    port: int,
    expected_model: str,
    service: subprocess.Popen[Any],
    timeout: int,
) -> bool:
    deadline = time.monotonic() + timeout
    probe = repository_root / "scripts" / "msmu" / "_probe_openai_models.py"
    while time.monotonic() < deadline:
        if service.poll() is not None:
            return False
        result = subprocess.run(
            [
                python,
                str(probe),
                "--base-url",
                f"http://127.0.0.1:{port}/v1",
                "--expected-model",
                expected_model,
                "--timeout",
                "5",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
        time.sleep(int(os.environ.get("QSPATIAL_SCHEDULE_POLL_SECONDS", "10")))
    return False


def _gpu_compute_pids(gpu_ids: tuple[int, ...]) -> list[str]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return ["nvidia-smi-unavailable"]
    found: list[str] = []
    for gpu in gpu_ids:
        result = subprocess.run(
            [executable, f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        found.extend(f"{gpu}:{line.strip()}" for line in result.stdout.splitlines() if line.strip().isdigit())
    return found


def _wait_for_gpu_release(gpu_ids: tuple[int, ...], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    poll = int(os.environ.get("QSPATIAL_SCHEDULE_POLL_SECONDS", "10"))
    while True:
        processes = _gpu_compute_pids(gpu_ids)
        if not processes:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU release timeout for {gpu_ids}; existing processes were untouched: {processes}"
            )
        time.sleep(poll)


def _wait_for_port_release(port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    poll = max(1, int(os.environ.get("QSPATIAL_SCHEDULE_POLL_SECONDS", "10")))
    while not _port_is_available(port):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"owned service port {port} did not release before timeout")
        time.sleep(poll)


def _start_vllm_service(
    job: ScheduledJob,
    *,
    repository_root: Path,
    environment: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[Any]:
    assert job.port is not None
    if not _port_is_available(job.port):
        raise RuntimeError(f"port {job.port} is occupied; existing process was untouched")
    gpu_text = ",".join(str(value) for value in job.gpu_ids)
    log = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                "bash",
                str(repository_root / "scripts" / "q_spatial" / "serve_vllm_profile.sh"),
                "--model",
                job.profile,
                "--gpu-ids",
                gpu_text,
                "--port",
                str(job.port),
            ],
            cwd=repository_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
    finally:
        log.close()
    profile = PROFILES[job.profile]
    if not _endpoint_ready(
        repository_root,
        environment["QSPATIAL_PYTHON"],
        job.port,
        profile.served_model_name or "",
        process,
        int(os.environ.get("QSPATIAL_SCHEDULE_SERVICE_TIMEOUT_SECONDS", "1800")),
    ):
        _stop_owned_group(process, int(os.environ.get("QSPATIAL_SCHEDULE_STOP_TIMEOUT_SECONDS", "120")))
        raise RuntimeError(f"vLLM service did not become ready for {job.profile}")
    return process


def _job_active_payload(
    job: ScheduledJob,
    *,
    run_id: str,
    lane_process_group: int,
    output_root: Path,
) -> dict[str, Any]:
    profile = PROFILES[job.profile]
    prediction = track_directory(output_root, profile) / "predictions.jsonl"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "phase": job.phase,
        "lane": job.lane,
        "profile": job.profile,
        "state": "START",
        "started_at_epoch": int(time.time()),
        "lane_process_group": lane_process_group,
        "worker_process_group": None,
        "service_process_groups": [],
        "gpu_ids": list(job.gpu_ids),
        "journal_glob": str(prediction.parent / "**" / "*.journal.jsonl"),
    }


def run_lane(
    lane: str,
    *,
    run_id: str,
    repository_root: Path,
    control_root: Path,
    skip_completed: bool,
    stage: str = "full",
) -> int:
    if stage not in {"test", "full"}:
        raise ValueError(f"unsupported scheduled stage: {stage}")
    status_path = control_root / "status.tsv"
    output_root = Path(os.environ["QSPATIAL_OUTPUT_ROOT"]).resolve()
    active_path = control_root / "active" / f"{lane}.json"
    logs_root = control_root / "logs"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    contract = QSpatialTestContract(
        os.environ["QSPATIAL_PARQUET_ROOT"],
        os.environ["QSPATIAL_SCANNET_RGB_ROOT"],
    )
    stop_timeout = int(os.environ.get("QSPATIAL_SCHEDULE_STOP_TIMEOUT_SECONDS", "120"))
    release_timeout = int(os.environ.get("QSPATIAL_SCHEDULE_GPU_RELEASE_TIMEOUT_SECONDS", "600"))
    lane_process_group = os.getpgrp()
    lane_failed = False
    for job in jobs_for_lane(lane):
        profile = PROFILES[job.profile]
        environment = environment_for_job(job)
        environment["QSPATIAL_REPO_ROOT"] = str(repository_root)
        environment["QSPATIAL_PYTHON"] = os.environ.get(
            "QSPATIAL_PYTHON", os.environ.get("PYTHON", os.environ.get("LATENT_PYTHON", sys.executable))
        )
        active = _job_active_payload(
            job,
            run_id=run_id,
            lane_process_group=lane_process_group,
            output_root=output_root,
        )
        atomic_write_json(active_path, active)
        _append_status(
            status_path,
            run_id=run_id,
            phase=job.phase,
            lane=lane,
            profile=job.profile,
            state="START",
            detail=f"gpu={','.join(map(str, job.gpu_ids)) or '-'}",
        )
        service: subprocess.Popen[Any] | None = None
        try:
            with _temporary_environment(environment):
                if skip_completed and stage == "full":
                    completion_errors = complete_result_errors(profile, contract, output_root)
                    if not completion_errors:
                        active["state"] = "PASS"
                        atomic_write_json(active_path, active)
                        _append_status(
                            status_path,
                            run_id=run_id,
                            phase=job.phase,
                            lane=lane,
                            profile=job.profile,
                            state="PASS",
                            detail="skip-completed; validator+metadata+revision+protocol+dataset+binding passed",
                        )
                        continue
                gate_errors = reusable_gate_errors(profile, contract, output_root)
                if stage == "test" and not gate_errors:
                    active["state"] = "PASS"
                    atomic_write_json(active_path, active)
                    _append_status(
                        status_path,
                        run_id=run_id,
                        phase=job.phase,
                        lane=lane,
                        profile=job.profile,
                        state="PASS",
                        detail="reused current test gate",
                    )
                    continue
                if job.gpu_ids:
                    _wait_for_gpu_release(job.gpu_ids, release_timeout)
                if profile.default_backend == "vllm":
                    service = _start_vllm_service(
                        job,
                        repository_root=repository_root,
                        environment=environment,
                        log_path=logs_root / f"{run_id}.{lane}.{job.profile}.vllm.log",
                    )
                    active["service_process_groups"] = [service.pid]
                    atomic_write_json(active_path, active)
                if gate_errors:
                    test_status = _run_owned(
                        [
                            environment["QSPATIAL_PYTHON"],
                            "-m",
                            "spatial_vlm_eval.benchmarks.q_spatial.inference",
                            "--stage",
                            "test",
                            "--model",
                            job.profile,
                        ],
                        environment=environment,
                        log_path=logs_root / f"{run_id}.{lane}.{job.profile}.test.log",
                        active_path=active_path,
                        active=active,
                    )
                    if test_status != 0:
                        raise RuntimeError(f"test stage exited {test_status}")
                    gate_errors = reusable_gate_errors(profile, contract, output_root)
                    if gate_errors:
                        raise RuntimeError("test gate still invalid: " + "; ".join(gate_errors))
                if stage == "test":
                    active["state"] = "PASS"
                    atomic_write_json(active_path, active)
                    _append_status(
                        status_path,
                        run_id=run_id,
                        phase=job.phase,
                        lane=lane,
                        profile=job.profile,
                        state="PASS",
                        detail="test gate passed",
                    )
                    continue
                full_status = _run_owned(
                    [
                        environment["QSPATIAL_PYTHON"],
                        "-m",
                        "spatial_vlm_eval.benchmarks.q_spatial.inference",
                        "--stage",
                        "full",
                        "--model",
                        job.profile,
                    ],
                    environment=environment,
                    log_path=logs_root / f"{run_id}.{lane}.{job.profile}.full.log",
                    active_path=active_path,
                    active=active,
                )
                if full_status != 0:
                    raise RuntimeError(f"full stage exited {full_status}")
                completion_errors = complete_result_errors(profile, contract, output_root)
                if completion_errors:
                    raise RuntimeError("formal completion audit failed: " + "; ".join(completion_errors))
            active["state"] = "PASS"
            atomic_write_json(active_path, active)
            _append_status(
                status_path,
                run_id=run_id,
                phase=job.phase,
                lane=lane,
                profile=job.profile,
                state="PASS",
                detail="full-271 validator and provenance passed",
            )
        except BaseException as exc:  # noqa: BLE001 - isolate failure to this lane.
            lane_failed = True
            active["state"] = "FAIL"
            active["error"] = f"{type(exc).__name__}: {exc}"[:1000]
            atomic_write_json(active_path, active)
            _append_status(
                status_path,
                run_id=run_id,
                phase=job.phase,
                lane=lane,
                profile=job.profile,
                state="FAIL",
                detail=active["error"],
            )
        finally:
            _stop_owned_group(service, stop_timeout)
            if service is not None and job.port is not None:
                try:
                    _wait_for_port_release(job.port, stop_timeout)
                except RuntimeError as exc:
                    lane_failed = True
                    if active.get("state") != "FAIL":
                        active["state"] = "FAIL"
                        active["error"] = str(exc)
                        atomic_write_json(active_path, active)
                        _append_status(
                            status_path,
                            run_id=run_id,
                            phase=job.phase,
                            lane=lane,
                            profile=job.profile,
                            state="FAIL",
                            detail=str(exc),
                        )
            active["service_process_groups"] = []
            atomic_write_json(active_path, active)
            if job.gpu_ids:
                try:
                    _wait_for_gpu_release(job.gpu_ids, release_timeout)
                except RuntimeError as exc:
                    lane_failed = True
                    if active.get("state") != "FAIL":
                        active["state"] = "FAIL"
                        active["error"] = str(exc)
                        atomic_write_json(active_path, active)
                        _append_status(
                            status_path,
                            run_id=run_id,
                            phase=job.phase,
                            lane=lane,
                            profile=job.profile,
                            state="FAIL",
                            detail=str(exc),
                        )
        if lane_failed:
            break
    _append_status(
        status_path,
        run_id=run_id,
        phase=jobs_for_lane(lane)[0].phase,
        lane=lane,
        profile="-",
        state="COMPLETE",
        detail="failed" if lane_failed else "passed",
    )
    active = _load_json(active_path) or {}
    active.update({"state": "COMPLETE", "lane_failed": lane_failed})
    atomic_write_json(active_path, active)
    _signal_watcher(run_id, lane)
    return 1 if lane_failed else 0


def _start_watcher(
    lane: str,
    *,
    run_id: str,
    repository_root: Path,
    control_root: Path,
) -> tuple[subprocess.Popen[Any], TextIO]:
    log_path = control_root / "logs" / f"{run_id}.{lane}.health.log"
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            os.environ.get("QSPATIAL_PYTHON", os.environ.get("PYTHON", os.environ.get("LATENT_PYTHON", sys.executable))),
            "-m",
            "spatial_vlm_eval.benchmarks.q_spatial.scheduled_watcher",
            "--lane",
            lane,
            "--run-id",
            run_id,
            "--control-root",
            str(control_root),
        ],
        cwd=repository_root,
        env=os.environ,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    return process, log_handle


def _start_lane(
    lane: str,
    *,
    run_id: str,
    repository_root: Path,
    control_root: Path,
    skip_completed: bool,
    stage: str,
) -> tuple[subprocess.Popen[Any], TextIO]:
    log_path = control_root / "logs" / f"{run_id}.{lane}.lane.log"
    log_handle = log_path.open("a", encoding="utf-8")
    command = [
        os.environ.get("QSPATIAL_PYTHON", os.environ.get("PYTHON", os.environ.get("LATENT_PYTHON", sys.executable))),
        "-m",
        "spatial_vlm_eval.benchmarks.q_spatial.scheduled_batch",
        "--internal-lane",
        lane,
        "--run-id",
        run_id,
        "--control-root",
        str(control_root),
        "--stage",
        stage,
    ]
    if skip_completed:
        command.append("--skip-completed")
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        env=os.environ,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    return process, log_handle


def _freeze_plan(control_root: Path, payload: dict[str, Any]) -> None:
    path = control_root / "plan.json"
    existing = _load_json(path)
    if path.exists() and existing is None:
        raise RuntimeError(f"frozen schedule is malformed and was not overwritten: {path}")
    if existing is not None and existing != payload:
        raise RuntimeError(
            f"frozen schedule differs from current code: {path}; use a new QSPATIAL_OUTPUT_ROOT"
        )
    if existing is None:
        atomic_write_json(path, payload)


def run_controller(repository_root: Path, *, skip_completed: bool, stage: str = "full") -> int:
    if stage not in {"test", "full"}:
        raise ValueError(f"unsupported scheduled stage: {stage}")
    if stage == "test" and skip_completed:
        raise ValueError("--skip-completed applies only to --stage full; test gates are reused automatically")
    report = preflight(repository_root, require_api_key=True)
    if not report["passed"]:
        raise RuntimeError("scheduled preflight failed: " + "; ".join(report["errors"]))
    output_root = Path(os.environ["QSPATIAL_OUTPUT_ROOT"]).resolve()
    control_root = output_root / "_scheduled_batch"
    (control_root / "logs").mkdir(parents=True, exist_ok=True)
    lock_path = control_root / "lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _freeze_plan(control_root, schedule_payload(repository_root))
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_id = f"{run_id}.{time.time_ns() % 1_000_000_000:09d}"
        status_path = control_root / "status.tsv"
        if not status_path.exists():
            status_path.write_text(
                "run_id\tphase\tlane\tprofile\tstate\ttimestamp\tdetail\n",
                encoding="utf-8",
            )
        elif not status_path.read_text(encoding="utf-8", errors="replace").startswith(
            "run_id\tphase\tlane\tprofile\tstate\ttimestamp\tdetail\n"
        ):
            raise RuntimeError(f"scheduled status header is malformed: {status_path}")
        lanes: dict[str, tuple[subprocess.Popen[Any], TextIO]] = {}
        watchers: dict[str, tuple[subprocess.Popen[Any], TextIO]] = {}

        def start(lane: str) -> None:
            watchers[lane] = _start_watcher(
                lane,
                run_id=run_id,
                repository_root=repository_root,
                control_root=control_root,
            )
            lanes[lane] = _start_lane(
                lane,
                run_id=run_id,
                repository_root=repository_root,
                control_root=control_root,
                skip_completed=skip_completed,
                stage=stage,
            )

        try:
            start("dual")
            start("api")
            dual_status = lanes["dual"][0].wait()
            if dual_status == 0:
                start("gpu0")
                start("gpu1")
            else:
                for lane in ("gpu0", "gpu1"):
                    _append_status(
                        status_path,
                        run_id=run_id,
                        phase="B",
                        lane=lane,
                        profile="-",
                        state="FAIL",
                        detail="phase B not started because dual lane failed",
                    )
            statuses = {"dual": dual_status}
            for lane, (process, _handle) in lanes.items():
                if lane == "dual":
                    continue
                statuses[lane] = process.wait()
            batch_passed = all(status == 0 for status in statuses.values()) and dual_status == 0
            _append_status(
                status_path,
                run_id=run_id,
                phase="-",
                lane="batch",
                profile="-",
                state="COMPLETE",
                detail="passed" if batch_passed else "failed",
            )
            return 0 if batch_passed else 1
        finally:
            stop_timeout = int(os.environ.get("QSPATIAL_SCHEDULE_STOP_TIMEOUT_SECONDS", "120"))
            for lane, (process, handle) in lanes.items():
                if process.poll() is None:
                    _stop_owned_group(process, stop_timeout)
                active = _load_json(control_root / "active" / f"{lane}.json") or {}
                if active.get("run_id") == run_id:
                    _stop_recorded_group(active.get("worker_process_group"), stop_timeout)
                    for pid in active.get("service_process_groups") or []:
                        _stop_recorded_group(pid, stop_timeout)
                handle.close()
            for lane, (process, handle) in watchers.items():
                if process.poll() is None:
                    _signal_watcher(run_id, lane)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        _stop_owned_group(process, 5)
                handle.close()


def _print_schedule() -> None:
    print("phase\tlane\torder\tprofile\tbackend\ttp\tgpus\tport\tbarrier")
    for job in SCHEDULE:
        profile = PROFILES[job.profile]
        print(
            "\t".join(
                (
                    job.phase,
                    job.lane,
                    str(job.order),
                    job.profile,
                    profile.default_backend,
                    str(profile.default_tensor_parallel_size),
                    ",".join(map(str, job.gpu_ids)) or "-",
                    str(job.port) if job.port is not None else "-",
                    LANE_PHASE_BARRIER.get(job.lane, "-")
                )
            )
        )


def _print_dry_run(stage: str) -> None:
    errors = schedule_errors()
    if errors:
        raise RuntimeError("invalid schedule: " + "; ".join(errors))
    for job in SCHEDULE:
        actions = "gate-or-test" if stage == "test" else "gate-or-test,full,validator"
        if PROFILES[job.profile].default_backend == "vllm":
            actions = "owned-vllm," + actions
        print(
            f"[q-spatial-schedule] dry-run stage={stage} phase={job.phase} lane={job.lane} "
            f"order={job.order} profile={job.profile} gpu={','.join(map(str, job.gpu_ids)) or '-'} "
            f"actions={actions}"
        )
    print(
        "[q-spatial-schedule] dry-run complete; no dataset/model/runner/GPU/API/scorer "
        "call or output write was made"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    parser.add_argument("--without-internvl78", action="store_true")
    parser.add_argument("--with-paid-api", action="store_true")
    parser.add_argument("--stage", choices=("test", "full"), default="full")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--internal-lane", choices=LANE_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--control-root", help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_signal_safe(operation: Any) -> int:
    previous: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise SchedulerInterrupted(signum)

    for signum in (signal.SIGHUP, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    try:
        return int(operation())
    except SchedulerInterrupted as exc:
        return 128 + exc.signum
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main() -> int:
    args = parse_args()
    repository_root = Path(os.environ.get("QSPATIAL_REPO_ROOT", Path.cwd())).resolve()
    if args.internal_lane:
        if not args.run_id or not args.control_root:
            raise ValueError("internal lane requires --run-id and --control-root")
        return _run_signal_safe(
            lambda: run_lane(
                args.internal_lane,
                run_id=args.run_id,
                repository_root=repository_root,
                control_root=Path(args.control_root).resolve(),
                skip_completed=args.skip_completed,
                stage=args.stage,
            )
        )
    if args.list:
        _print_schedule()
        return 0
    if args.dry_run:
        _print_dry_run(args.stage)
        return 0
    if args.check:
        report = preflight(repository_root, require_api_key=False)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    if not args.without_internvl78 or not args.with_paid_api:
        raise ValueError(
            "formal execution requires both --without-internvl78 and --with-paid-api"
        )
    if args.stage == "test" and args.skip_completed:
        raise ValueError(
            "--skip-completed applies only to --stage full; test gates are reused automatically"
        )
    return _run_signal_safe(
        lambda: run_controller(
            repository_root,
            skip_completed=args.skip_completed,
            stage=args.stage,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
