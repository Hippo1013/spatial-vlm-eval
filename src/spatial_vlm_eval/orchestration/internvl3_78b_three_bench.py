"""Run Q-Spatial, SPBench-SI, and CV-Bench against one owned vLLM service."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_KEY = "internvl3_78b"
SERVED_MODEL_NAME = "internvl3-78b-three-bench"
MODEL_REVISION = "3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356"
MODEL_ID = "OpenGVLab/InternVL3-78B-hf"
BENCHMARK_ORDER = ("q_spatial", "spbench_si", "cv_bench")
EXPECTED_BASELINE = {"q_spatial": 20, "spbench_si": 20, "cv_bench": 22}
EXIT_CONFIGURATION = 2
EXIT_RESOURCE_BLOCKED = 4


class ConfigurationError(RuntimeError):
    """The requested workflow does not match the locked configuration."""


class ResourceBlocked(RuntimeError):
    """A port, GPU, or controller lock is already owned elsewhere."""


class StepFailed(RuntimeError):
    """One owned workflow step exited unsuccessfully."""


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    key: str
    label: str
    scripts_dir: Path
    output_root: Path
    profile: Any
    track: Path
    predictions: Path
    validation_report: Path
    report: Path
    total_profiles: int
    official_size: int
    scorer_protocol: str

    @property
    def inference_script(self) -> Path:
        return self.scripts_dir / "run_inference.sh"

    @property
    def validation_script(self) -> Path:
        return self.scripts_dir / "validate_predictions.sh"

    @property
    def scoring_script(self) -> Path:
        return self.scripts_dir / "score_results.sh"

    @property
    def report_script(self) -> Path:
        return self.scripts_dir / "build_results_report.sh"


@dataclass(frozen=True, slots=True)
class WorkflowOutcome:
    inference_error: str | None
    publication_status: Mapping[str, int]
    publication_start_order: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.inference_error is None and all(
            status == 0 for status in self.publication_status.values()
        )


@dataclass(frozen=True, slots=True)
class ReportPlan:
    rebuild: bool
    detail: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def assert_shared_profile_identity() -> dict[str, Any]:
    from spatial_vlm_eval.benchmarks.cv_bench.profiles import get_profile as cv_profile
    from spatial_vlm_eval.benchmarks.q_spatial.profiles import get_profile as q_profile
    from spatial_vlm_eval.benchmarks.spbench_si.profiles import get_profile as sp_profile

    profiles = {
        "q_spatial": q_profile(PROFILE_KEY),
        "spbench_si": sp_profile(PROFILE_KEY),
        "cv_bench": cv_profile(PROFILE_KEY),
    }
    errors: list[str] = []
    for key, profile in profiles.items():
        checks = {
            "model": getattr(profile, "model", None) == MODEL_ID,
            "revision": getattr(profile, "revision", None) == MODEL_REVISION,
            "served_model_name": getattr(profile, "served_model_name", None)
            == SERVED_MODEL_NAME,
            "tensor_parallel_size": getattr(profile, "default_tensor_parallel_size", None)
            == 4,
            "family": getattr(profile, "family", None) == "internvl3",
            "processor_family": getattr(profile, "processor_family", None) == "internvl3",
            "model_path_env": getattr(profile, "model_path_env", None) == "INTERNVL3_78B_MODEL",
            "seed": (getattr(profile, "decoding", {}) or {}).get("seed") == 42,
        }
        errors.extend(f"{key} {name} mismatch" for name, passed in checks.items() if not passed)
    if errors:
        raise ConfigurationError("shared profile identity failed: " + "; ".join(errors))
    return profiles


def resolve_benchmarks(
    root: Path, environment: Mapping[str, str], *, allow_placeholders: bool = False
) -> tuple[BenchmarkSpec, ...]:
    profiles = assert_shared_profile_identity()

    from spatial_vlm_eval.benchmarks.cv_bench.data import OFFICIAL_TEST_SIZE as cv_size
    from spatial_vlm_eval.benchmarks.cv_bench.inference import track_directory as cv_track
    from spatial_vlm_eval.benchmarks.cv_bench.profiles import PROFILE_SEQUENCE as cv_sequence
    from spatial_vlm_eval.benchmarks.cv_bench.scorer import SCORER_PROTOCOL as cv_scorer
    from spatial_vlm_eval.benchmarks.q_spatial.data import OFFICIAL_TEST_SIZE as q_size
    from spatial_vlm_eval.benchmarks.q_spatial.inference import track_directory as q_track
    from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILE_SEQUENCE as q_sequence
    from spatial_vlm_eval.benchmarks.q_spatial.scorer import SCORER_PROTOCOL as q_scorer
    from spatial_vlm_eval.benchmarks.spbench_si.data import OFFICIAL_TEST_SIZE as sp_size
    from spatial_vlm_eval.benchmarks.spbench_si.inference import track_directory as sp_track
    from spatial_vlm_eval.benchmarks.spbench_si.profiles import PROFILE_SEQUENCE as sp_sequence
    from spatial_vlm_eval.benchmarks.spbench_si.scorer import SCORER_PROTOCOL as sp_scorer

    def output(variable: str) -> Path:
        value = environment.get(variable)
        if not value:
            if not allow_placeholders:
                raise ConfigurationError(f"set {variable}")
            value = f"/required/{variable.lower()}"
        return Path(value).expanduser().resolve()

    q_root = output("QSPATIAL_OUTPUT_ROOT")
    sp_root = output("SPBENCH_SI_OUTPUT_ROOT")
    cv_root = output("CVBENCH_OUTPUT_ROOT")
    definitions = {
        "q_spatial": (
            "Q-Spatial",
            root / "scripts" / "q_spatial",
            q_root,
            q_track(q_root, profiles["q_spatial"]),
            len(q_sequence),
            q_size,
            q_scorer,
            "q-spatial-result.md",
        ),
        "spbench_si": (
            "SPBench-SI",
            root / "scripts" / "spbench_si",
            sp_root,
            sp_track(sp_root, profiles["spbench_si"]),
            len(sp_sequence),
            sp_size,
            sp_scorer,
            "spbench-si-result.md",
        ),
        "cv_bench": (
            "CV-Bench",
            root / "scripts" / "cv_bench",
            cv_root,
            cv_track(cv_root, profiles["cv_bench"]),
            len(cv_sequence),
            cv_size,
            cv_scorer,
            "cv-bench-result.md",
        ),
    }
    resolved: list[BenchmarkSpec] = []
    for key in BENCHMARK_ORDER:
        label, scripts, output_root, track, total, size, scorer, report_name = definitions[key]
        resolved.append(
            BenchmarkSpec(
                key=key,
                label=label,
                scripts_dir=scripts,
                output_root=output_root,
                profile=profiles[key],
                track=track,
                predictions=track / "predictions.jsonl",
                validation_report=track / "prediction_validation.json",
                report=output_root / report_name,
                total_profiles=total,
                official_size=size,
                scorer_protocol=scorer,
            )
        )
    return tuple(resolved)


def prediction_reuse_errors(spec: BenchmarkSpec) -> list[str]:
    errors: list[str] = []
    if not spec.predictions.is_file():
        return ["prediction is missing"]
    metadata_path = spec.predictions.with_suffix(spec.predictions.suffix + ".metadata.json")
    metadata = _load_json(metadata_path)
    validation = _load_json(spec.validation_report)
    gate = _load_json(spec.track / "test_gate.json")
    if metadata is None:
        return ["metadata is missing or malformed"]
    if validation is None:
        errors.append("full validator is missing or malformed")
        validation = {}
    if gate is None:
        errors.append("test gate is missing or malformed")
        gate = {}
    model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
    binding = metadata.get("binding") if isinstance(metadata.get("binding"), dict) else {}
    binding_profile = binding.get("profile") if isinstance(binding.get("profile"), dict) else {}
    expected_gate = (spec.track / "test_gate.json").resolve()
    declared_gate = Path(str(metadata.get("test_gate") or "")).resolve()
    checks = {
        "publishable inference": metadata.get("publishable_inference") is True,
        "prediction count": metadata.get("num_predictions") == spec.official_size,
        "profile": model.get("profile") == PROFILE_KEY,
        "model revision": model.get("model_revision") == MODEL_REVISION,
        "inference protocol": metadata.get("inference_protocol") == spec.profile.inference_protocol,
        "scorer protocol": metadata.get("scorer_protocol") == spec.scorer_protocol,
        "dataset size": dataset.get("official_test_size") == spec.official_size,
        "binding profile": binding_profile.get("key") == PROFILE_KEY,
        "binding registry digest": binding_profile.get("registry_digest")
        == spec.profile.registry_digest,
        "binding digest": metadata.get("binding_digest") == _digest(binding),
        "output hash": metadata.get("output_sha256") == _file_digest(spec.predictions),
        "test gate path": declared_gate == expected_gate,
        "test gate passed": gate.get("passed") is True,
        "test gate binding": gate.get("binding_digest") == metadata.get("binding_digest"),
        "stored validator passed": validation.get("passed") is True,
        "stored validator size": validation.get("official_test_size") == spec.official_size,
        "stored validator rows": validation.get("num_prediction_rows") == spec.official_size,
        "stored validator indices": validation.get("num_unique_indices") == spec.official_size,
        "stored validator dataset": validation.get("dataset_fingerprint")
        == dataset.get("fingerprint"),
    }
    errors.extend(name + " mismatch" for name, passed in checks.items() if not passed)
    return errors


def _report_profiles(spec: BenchmarkSpec) -> set[str]:
    if spec.key == "q_spatial":
        from spatial_vlm_eval.benchmarks.q_spatial.report import discover_results
    elif spec.key == "spbench_si":
        from spatial_vlm_eval.benchmarks.spbench_si.report import discover_results
    else:
        from spatial_vlm_eval.benchmarks.cv_bench.report import discover_results
    return {item.profile for item in discover_results(spec.output_root)}


def _expected_profiles(spec: BenchmarkSpec) -> set[str]:
    if spec.key == "q_spatial":
        from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILE_SEQUENCE
    elif spec.key == "spbench_si":
        from spatial_vlm_eval.benchmarks.spbench_si.profiles import PROFILE_SEQUENCE
    else:
        from spatial_vlm_eval.benchmarks.cv_bench.profiles import PROFILE_SEQUENCE
    return set(PROFILE_SEQUENCE)


def report_rebuild_plan(spec: BenchmarkSpec) -> ReportPlan:
    try:
        present = _report_profiles(spec)
    except Exception as exc:  # noqa: BLE001 - unrelated report sources must not block scoring.
        return ReportPlan(
            rebuild=False,
            detail=f"report discovery unavailable: {type(exc).__name__}: {exc}",
        )
    expected = _expected_profiles(spec)
    missing = expected - present
    unexpected = present - expected
    if present == expected:
        return ReportPlan(
            rebuild=True,
            detail=f"report sources already complete: {len(present)}/{spec.total_profiles}",
        )
    if (
        len(present) == EXPECTED_BASELINE[spec.key]
        and missing == {PROFILE_KEY}
        and not unexpected
    ):
        return ReportPlan(
            rebuild=True,
            detail=(
                f"report baseline ready: {len(present)}/{spec.total_profiles}; "
                f"only {PROFILE_KEY} missing"
            ),
        )
    return ReportPlan(
        rebuild=False,
        detail=(
            f"report baseline incomplete: {len(present)}/{spec.total_profiles}; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        ),
    )


def _score_state(spec: BenchmarkSpec) -> str:
    if spec.key == "spbench_si":
        from spatial_vlm_eval.benchmarks.spbench_si.score_results import score_state
    elif spec.key == "q_spatial":
        from spatial_vlm_eval.benchmarks.q_spatial.score_results import score_state
    else:
        from spatial_vlm_eval.benchmarks.cv_bench.score_results import score_state
    return score_state(spec.predictions).state


def target_result_errors(spec: BenchmarkSpec) -> list[str]:
    errors = prediction_reuse_errors(spec)
    if _score_state(spec) != "complete":
        errors.append("target score did not reach complete publication gates")
    return errors


def report_result_errors(spec: BenchmarkSpec) -> list[str]:
    errors: list[str] = []
    try:
        present = _report_profiles(spec)
    except Exception as exc:  # noqa: BLE001
        return [f"report discovery failed: {type(exc).__name__}: {exc}"]
    if len(present) != spec.total_profiles or PROFILE_KEY not in present:
        errors.append(
            f"report sources are incomplete: {len(present)}/{spec.total_profiles}, "
            f"internvl3_78b_present={PROFILE_KEY in present}"
        )
    if not spec.report.is_file() or spec.report.stat().st_size == 0:
        errors.append(f"global report is missing: {spec.report}")
    return errors


def lock_paths(control_root: Path, specs: Sequence[BenchmarkSpec]) -> tuple[Path, ...]:
    by_key = {spec.key: spec for spec in specs}
    return (
        control_root / "lock",
        by_key["q_spatial"].output_root / "_scheduled_batch" / "lock",
        by_key["q_spatial"].output_root / "_single_model_evaluation" / "lock",
        by_key["spbench_si"].output_root / "_scheduled_batch" / "lock",
        by_key["spbench_si"].output_root / "_single_model_evaluation" / "lock",
        by_key["cv_bench"].output_root / "_serial_full" / "lock",
        by_key["cv_bench"].output_root / "_single_model_evaluation" / "lock",
    )


class MultiLock:
    def __init__(self, paths: Iterable[Path]) -> None:
        self.paths = tuple(paths)
        self.handles: list[Any] = []

    def acquire(self) -> None:
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    handle.close()
                    raise ResourceBlocked(f"lock is already held: {path}") from exc
                self.handles.append(handle)
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        for handle in reversed(self.handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        self.handles.clear()

    def __enter__(self) -> MultiLock:  # noqa: PYI034 - Python 3.10 has no typing.Self.
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()

    @staticmethod
    def unavailable(paths: Iterable[Path]) -> list[Path]:
        unavailable: list[Path] = []
        handles: list[Any] = []
        try:
            for path in paths:
                if not path.is_file():
                    continue
                handle = path.open("r+", encoding="utf-8")
                handles.append(handle)
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    unavailable.append(path)
        finally:
            for handle in handles:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
        return unavailable


def execute_workflow(
    specs: Sequence[BenchmarkSpec],
    *,
    is_reusable: Callable[[BenchmarkSpec], bool],
    ensure_service: Callable[[], None],
    run_inference: Callable[[BenchmarkSpec], None],
    start_publication: Callable[[BenchmarkSpec], Any],
    stop_service: Callable[[], None],
    wait_publications: Callable[[Mapping[str, Any]], Mapping[str, int]],
) -> WorkflowOutcome:
    publications: dict[str, Any] = {}
    start_order: list[str] = []
    inference_error: str | None = None
    try:
        for spec in specs:
            if not is_reusable(spec):
                ensure_service()
                run_inference(spec)
            try:
                publications[spec.key] = start_publication(spec)
                start_order.append(spec.key)
            except Exception as exc:  # noqa: BLE001 - publication failure is lane-isolated.
                publications[spec.key] = exc
                start_order.append(spec.key)
    except Exception as exc:  # noqa: BLE001 - fail-fast only the inference lane.
        inference_error = f"{type(exc).__name__}: {exc}"
    finally:
        stop_service()
    statuses = dict(wait_publications(publications))
    return WorkflowOutcome(inference_error, statuses, tuple(start_order))


class Controller:
    def __init__(
        self,
        root: Path,
        specs: Sequence[BenchmarkSpec],
        control_root: Path,
        environment: Mapping[str, str],
    ) -> None:
        self.root = root
        self.specs = tuple(specs)
        self.control_root = control_root
        self.environment = dict(environment)
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f".{os.getpid()}"
        self.logs_root = control_root / "logs"
        self.status_path = control_root / "status.tsv"
        self.service: subprocess.Popen[Any] | None = None
        self.active_step: subprocess.Popen[Any] | None = None
        self.publication_workers: dict[str, subprocess.Popen[Any]] = {}

    def prepare(self) -> None:
        self.logs_root.mkdir(parents=True, exist_ok=True)
        if not self.status_path.exists():
            self.status_path.write_text(
                "run_id\tbenchmark\tstage\tstate\ttimestamp\tdetail\n", encoding="utf-8"
            )

    def log(self, message: str) -> None:
        line = f"[internvl3-78b-three-bench] {message}"
        print(line, flush=True)
        with (self.logs_root / f"{self.run_id}.controller.log").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(line + "\n")

    def status(self, benchmark: str, stage: str, state: str, detail: str) -> None:
        line = (
            f"{self.run_id}\t{benchmark}\t{stage}\t{state}\t"
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{detail[:1000]}\n"
        )
        with self.status_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _run_step(self, spec: BenchmarkSpec, stage: str, command: Sequence[str]) -> None:
        log = self.logs_root / f"{self.run_id}.{spec.key}.{stage}.log"
        self.log(f"START benchmark={spec.label} stage={stage} log={log}")
        self.status(spec.key, stage, "START", str(log))
        with log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                list(command),
                cwd=self.root,
                env=self.environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            self.active_step = process
            try:
                status = process.wait()
            except BaseException:
                self._stop_process_group(process, f"{spec.key} {stage}")
                self.active_step = None
                raise
            else:
                self.active_step = None
        if status != 0:
            self.status(spec.key, stage, "FAIL", f"exit={status}; log={log}")
            raise StepFailed(f"{spec.label} {stage} exited {status}; log={log}")
        self.status(spec.key, stage, "PASS", str(log))
        self.log(f"PASS benchmark={spec.label} stage={stage}")

    def _service_command(self) -> list[str]:
        return ["bash", str(self.root / "scripts" / "internvl3_78b" / "serve_shared_vllm.sh")]

    def ensure_service(self) -> None:
        if self.service is not None:
            return
        log = self.logs_root / f"{self.run_id}.vllm.log"
        self.log(f"START shared vLLM log={log}")
        handle = log.open("a", encoding="utf-8")
        try:
            self.service = subprocess.Popen(
                self._service_command(),
                cwd=self.root,
                env=self.environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
        finally:
            handle.close()
        timeout = int(
            self.environment.get("INTERNVL3_78B_THREE_BENCH_SERVICE_TIMEOUT_SECONDS", "1800")
        )
        poll = int(self.environment.get("INTERNVL3_78B_THREE_BENCH_POLL_SECONDS", "10"))
        deadline = time.monotonic() + timeout
        probe = self.root / "scripts" / "msmu" / "_probe_openai_models.py"
        base_url = self.environment["INTERNVL3_78B_THREE_BENCH_BASE_URL"]
        while time.monotonic() < deadline:
            assert self.service is not None
            if self.service.poll() is not None:
                raise StepFailed(f"shared vLLM exited before readiness; log={log}")
            ready = subprocess.run(
                [
                    sys.executable,
                    str(probe),
                    "--base-url",
                    base_url,
                    "--expected-model",
                    SERVED_MODEL_NAME,
                    "--timeout",
                    "5",
                ],
                cwd=self.root,
                env=self.environment,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ready.returncode == 0:
                self.status("service", "vllm", "PASS", f"pid={self.service.pid}")
                self.log(f"PASS shared vLLM pid={self.service.pid}")
                return
            time.sleep(poll)
        raise StepFailed(f"shared vLLM readiness timed out; log={log}")

    def _stop_process_group(self, process: subprocess.Popen[Any] | None, label: str) -> None:
        if process is None or process.poll() is not None:
            if process is not None:
                process.wait()
            return
        self.log(f"stopping owned {label} process_group={process.pid}")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        timeout = int(
            self.environment.get("INTERNVL3_78B_THREE_BENCH_STOP_TIMEOUT_SECONDS", "120")
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)

    def stop_service(self) -> None:
        self._stop_process_group(self.service, "vLLM")
        self.service = None

    def cleanup(self) -> None:
        self._stop_process_group(self.active_step, "active step")
        self.active_step = None
        self.stop_service()
        for key, process in list(self.publication_workers.items()):
            self._stop_process_group(process, f"{key} publication")
            self.publication_workers.pop(key, None)

    def is_reusable(self, spec: BenchmarkSpec) -> bool:
        errors = prediction_reuse_errors(spec)
        if errors:
            self.log(f"RUN inference benchmark={spec.label} reason={'; '.join(errors)}")
            return False
        try:
            self._run_step(
                spec,
                "reuse-validator",
                [
                    "bash",
                    str(spec.validation_script),
                    "--predictions",
                    str(spec.predictions),
                    "--report",
                    str(spec.validation_report),
                ],
            )
        except StepFailed as exc:
            self.log(f"RUN inference benchmark={spec.label} reason=revalidation failed: {exc}")
            return False
        self.status(spec.key, "inference", "SKIP", "current full prediction revalidated")
        self.log(f"SKIP inference benchmark={spec.label}; current full result revalidated")
        return True

    def run_inference(self, spec: BenchmarkSpec) -> None:
        self._run_step(
            spec,
            "test",
            ["bash", str(spec.inference_script), "--stage", "test", "--model", PROFILE_KEY],
        )
        self._run_step(
            spec,
            "full",
            ["bash", str(spec.inference_script), "--stage", "full", "--model", PROFILE_KEY],
        )
        self._run_step(
            spec,
            "validator",
            [
                "bash",
                str(spec.validation_script),
                "--predictions",
                str(spec.predictions),
                "--report",
                str(spec.validation_report),
            ],
        )
        errors = prediction_reuse_errors(spec)
        if errors:
            raise StepFailed(f"{spec.label} full provenance audit failed: {'; '.join(errors)}")

    def start_publication(
        self, spec: BenchmarkSpec, report_plan: ReportPlan
    ) -> subprocess.Popen[Any]:
        log = self.logs_root / f"{self.run_id}.{spec.key}.publication.log"
        report_mode = "build" if report_plan.rebuild else "skip"
        self.log(
            f"START background scoring benchmark={spec.label} "
            f"report={report_mode} log={log}"
        )
        self.status(
            spec.key,
            "publication",
            "START",
            f"report={report_mode}; {report_plan.detail}; log={log}",
        )
        handle = log.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "spatial_vlm_eval.orchestration.internvl3_78b_three_bench",
                    "--publication-worker",
                    spec.key,
                    "--predictions",
                    str(spec.predictions),
                    "--report-mode",
                    report_mode,
                ],
                cwd=self.root,
                env=self.environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            self.publication_workers[spec.key] = process
            return process
        finally:
            handle.close()

    def wait_publications(self, workers: Mapping[str, Any]) -> Mapping[str, int]:
        statuses: dict[str, int] = {}
        for key, worker in workers.items():
            spec = next(item for item in self.specs if item.key == key)
            if isinstance(worker, BaseException):
                statuses[key] = 1
                self.status(key, "publication", "FAIL", f"start failed: {worker}")
                continue
            status = int(worker.wait())
            self.publication_workers.pop(key, None)
            statuses[key] = status
            state = "PASS" if status == 0 else "FAIL"
            self.status(key, "publication", state, f"exit={status}")
            self.log(f"{state} background scoring/report benchmark={spec.label} exit={status}")
        return statuses


def _environment_errors(root: Path, specs: Sequence[BenchmarkSpec], environment: Mapping[str, str]) -> list[str]:
    errors: list[str] = []
    required_paths = {
        "CVBENCH_DATASET_ROOT": "dir",
        "QSPATIAL_PARQUET_ROOT": "dir",
        "QSPATIAL_SCANNET_RGB_ROOT": "dir",
        "SPBENCH_SI_PARQUET": "file",
        "SPBENCH_SI_IMAGES_ARCHIVE": "file",
        "INTERNVL3_78B_MODEL": "exists",
    }
    for variable, kind in required_paths.items():
        value = environment.get(variable)
        if not value:
            errors.append(f"set {variable}")
            continue
        path = Path(value).expanduser()
        passed = path.is_dir() if kind == "dir" else path.is_file() if kind == "file" else path.exists()
        if not passed:
            errors.append(f"{variable} is unavailable: {path}")
    for spec in specs:
        if not spec.output_root.is_dir():
            errors.append(f"{spec.label} existing output root is unavailable: {spec.output_root}")
    control_value = environment.get("INTERNVL3_78B_THREE_BENCH_CONTROL_ROOT")
    if not control_value:
        errors.append("set INTERNVL3_78B_THREE_BENCH_CONTROL_ROOT")
    else:
        control = Path(control_value).expanduser().resolve()
        try:
            control.relative_to(root)
        except ValueError:
            pass
        else:
            errors.append("INTERNVL3_78B_THREE_BENCH_CONTROL_ROOT must stay outside the repository")
        for spec in specs:
            try:
                control.relative_to(spec.output_root)
            except ValueError:
                continue
            errors.append(
                "INTERNVL3_78B_THREE_BENCH_CONTROL_ROOT must be independent of "
                f"{spec.label} output root"
            )
    return errors


def _run_checked(command: Sequence[str], *, root: Path, environment: Mapping[str, str]) -> int:
    return subprocess.run(list(command), cwd=root, env=dict(environment), check=False).returncode


def run_preflight(
    root: Path,
    specs: Sequence[BenchmarkSpec],
    control_root: Path,
    environment: Mapping[str, str],
) -> None:
    errors = _environment_errors(root, specs, environment)
    if errors:
        raise ConfigurationError("; ".join(errors))
    service_status = _run_checked(
        ["bash", str(root / "scripts" / "internvl3_78b" / "serve_shared_vllm.sh"), "--check"],
        root=root,
        environment=environment,
    )
    if service_status == EXIT_RESOURCE_BLOCKED:
        raise ResourceBlocked("four-GPU service preflight is blocked")
    if service_status != 0:
        raise ConfigurationError(f"shared service preflight exited {service_status}")
    unavailable = MultiLock.unavailable(lock_paths(control_root, specs))
    if unavailable:
        raise ResourceBlocked("workflow lock is already held: " + ", ".join(map(str, unavailable)))
    for spec in specs:
        status = _run_checked(
            ["bash", str(spec.inference_script), "--check", "--model", PROFILE_KEY],
            root=root,
            environment=environment,
        )
        if status == EXIT_RESOURCE_BLOCKED:
            raise ResourceBlocked(f"{spec.label} preflight is resource-blocked")
        if status != 0:
            raise ConfigurationError(f"{spec.label} preflight exited {status}")


def run_publication_worker(spec: BenchmarkSpec, report_plan: ReportPlan) -> int:
    score_command = [
        "bash",
        str(spec.scoring_script),
        "--predictions",
        str(spec.predictions),
    ]
    status = subprocess.run(score_command, check=False).returncode
    if status != 0:
        return status
    errors = target_result_errors(spec)
    if errors:
        print(
            f"[{spec.key}-publication] target score verification failed: {'; '.join(errors)}",
            file=sys.stderr,
        )
        return 1
    if not report_plan.rebuild:
        print(
            f"[{spec.key}-publication] SKIP report; target score complete; "
            f"{report_plan.detail}"
        )
        return 0
    for command in (
        ["bash", str(spec.report_script), "--check"],
        ["bash", str(spec.report_script)],
    ):
        status = subprocess.run(command, check=False).returncode
        if status != 0:
            return status
    errors = report_result_errors(spec)
    if errors:
        print(
            f"[{spec.key}-publication] report verification failed: {'; '.join(errors)}",
            file=sys.stderr,
        )
        return 1
    return 0


def print_dry_run(root: Path, specs: Sequence[BenchmarkSpec], environment: Mapping[str, str]) -> int:
    service = root / "scripts" / "internvl3_78b" / "serve_shared_vllm.sh"
    status = subprocess.run(["bash", str(service), "--dry-run"], env=dict(environment), check=False)
    if status.returncode != 0:
        return status.returncode
    for spec in specs:
        print(f"[three-bench-dry-run] {spec.label}: bash {spec.inference_script} --stage test --model {PROFILE_KEY}")
        print(f"[three-bench-dry-run] {spec.label}: bash {spec.inference_script} --stage full --model {PROFILE_KEY}")
        print(
            f"[three-bench-dry-run] {spec.label}: bash {spec.validation_script} "
            f"--predictions {spec.predictions} --report {spec.validation_report}"
        )
        print(
            f"[three-bench-dry-run] {spec.label}: background bash {spec.scoring_script} "
            f"--predictions {spec.predictions}; rebuild report only when the existing "
            f"baseline is {EXPECTED_BASELINE[spec.key]}/{spec.total_profiles} with only "
            f"{PROFILE_KEY} missing"
        )
    print("[three-bench-dry-run] stop owned shared vLLM; wait for all background publication workers")
    print("[three-bench-dry-run] verify all three target scores; reports may be complete or skipped")
    print("[three-bench-dry-run] no files, GPU processes, inference, scoring, or reports were changed")
    return 0


def print_status(specs: Sequence[BenchmarkSpec], control_root: Path) -> int:
    print(f"control_root\t{control_root}")
    status = control_root / "status.tsv"
    if status.is_file():
        lines = status.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-40:]:
            print(f"control\t{line}")
    else:
        print(f"control\tmissing\t{status}")
    for spec in specs:
        reuse = prediction_reuse_errors(spec)
        try:
            present = len(_report_profiles(spec))
            report_state = f"{present}/{spec.total_profiles}"
        except Exception as exc:  # noqa: BLE001 - status is informative.
            report_state = f"error:{type(exc).__name__}:{exc}"
        print(
            f"benchmark\t{spec.key}\tfull={'reusable' if not reuse else 'missing_or_stale'}\t"
            f"report={report_state}\tpath={spec.predictions}"
        )
    return 0


def _positive_integer(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def configured_environment(root: Path, source: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(source or os.environ)
    python = environment.get("LATENT_PYTHON") or environment.get("PYTHON") or sys.executable
    environment["LATENT_PYTHON"] = python
    environment["CVBENCH_PYTHON"] = python
    environment["QSPATIAL_PYTHON"] = python
    environment["SPBENCH_SI_PYTHON"] = python
    environment["CVBENCH_ENV_FILE"] = "/dev/null"
    environment["QSPATIAL_ENV_FILE"] = "/dev/null"
    environment["SPBENCH_SI_ENV_FILE"] = "/dev/null"
    gpu_ids = environment.get("INTERNVL3_78B_THREE_BENCH_GPU_IDS", "0,1,2,3")
    if len(gpu_ids.split(",")) != 4 or len(set(gpu_ids.split(","))) != 4:
        raise ConfigurationError(
            "INTERNVL3_78B_THREE_BENCH_GPU_IDS must contain four distinct GPU ids"
        )
    port = _positive_integer(environment, "INTERNVL3_78B_THREE_BENCH_PORT", 18103)
    if port > 65535:
        raise ConfigurationError("INTERNVL3_78B_THREE_BENCH_PORT must be at most 65535")
    base_url = f"http://127.0.0.1:{port}/v1"
    environment["INTERNVL3_78B_THREE_BENCH_GPU_IDS"] = gpu_ids
    environment["INTERNVL3_78B_THREE_BENCH_PORT"] = str(port)
    environment["INTERNVL3_78B_THREE_BENCH_BASE_URL"] = base_url
    environment["CUDA_VISIBLE_DEVICES"] = gpu_ids
    environment["CVBENCH_INTERNVL3_78B_GPU_IDS"] = gpu_ids
    environment["QSPATIAL_INTERNVL3_78B_GPU_IDS"] = gpu_ids
    environment["SPBENCH_SI_INTERNVL3_78B_GPU_IDS"] = gpu_ids
    environment["CVBENCH_INTERNVL3_78B_BASE_URLS"] = base_url
    environment["QSPATIAL_INTERNVL3_78B_BASE_URLS"] = base_url
    environment["SPBENCH_SI_INTERNVL3_78B_BASE_URLS"] = base_url
    environment["CVBENCH_VLLM_MAX_MODEL_LEN"] = "32768"
    environment["QSPATIAL_VLLM_MAX_MODEL_LEN"] = "32768"
    environment["SPBENCH_SI_VLLM_MAX_MODEL_LEN"] = "32768"
    environment["SPBENCH_SI_VLLM_RUNTIME_VERSION"] = "0.19.0"
    pythonpath = str(root / "src")
    if environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath
    return environment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--check", action="store_true")
    operation.add_argument("--status", action="store_true")
    operation.add_argument("--dry-run", action="store_true")
    operation.add_argument("--publication-worker", choices=BENCHMARK_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--predictions", help=argparse.SUPPRESS)
    parser.add_argument(
        "--report-mode",
        choices=("auto", "build", "skip"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repository_root()
    controller: Controller | None = None
    try:
        environment = configured_environment(root)
        specs = resolve_benchmarks(root, environment, allow_placeholders=args.dry_run)
        if args.publication_worker:
            spec = next(item for item in specs if item.key == args.publication_worker)
            if not args.predictions or Path(args.predictions).resolve() != spec.predictions:
                raise ConfigurationError("publication worker prediction path differs from canonical track")
            if args.report_mode == "auto":
                report_plan = report_rebuild_plan(spec)
            else:
                report_plan = ReportPlan(
                    rebuild=args.report_mode == "build",
                    detail=f"controller selected report mode: {args.report_mode}",
                )
            return run_publication_worker(spec, report_plan)
        if args.dry_run:
            return print_dry_run(root, specs, environment)
        control_value = environment.get("INTERNVL3_78B_THREE_BENCH_CONTROL_ROOT")
        if not control_value:
            raise ConfigurationError("set INTERNVL3_78B_THREE_BENCH_CONTROL_ROOT")
        control_root = Path(control_value).expanduser().resolve()
        if args.status:
            return print_status(specs, control_root)
        run_preflight(root, specs, control_root, environment)
        if args.check:
            print("[internvl3-78b-three-bench] CHECK shared_profile_identity=passed")
            for spec in specs:
                plan = report_rebuild_plan(spec)
                mode = "build" if plan.rebuild else "skip"
                print(
                    f"[internvl3-78b-three-bench] CHECK report benchmark={spec.label} "
                    f"mode={mode} detail={plan.detail}"
                )
            print("[internvl3-78b-three-bench] CHECK report_gaps_do_not_block_target_scoring")
            print("[internvl3-78b-three-bench] CHECK service=BF16,TP4,vLLM0.19.0,max_model_len32768")
            print("[internvl3-78b-three-bench] CHECK locks=available")
            return 0
        controller = Controller(root, specs, control_root, environment)
        controller.prepare()
        previous_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGHUP, signal.SIGTERM)
        }

        def interrupt_controller(_signum: int, _frame: object) -> None:
            raise KeyboardInterrupt

        for signum in previous_handlers:
            signal.signal(signum, interrupt_controller)
        with MultiLock(lock_paths(control_root, specs)):
            try:
                report_plans = {spec.key: report_rebuild_plan(spec) for spec in specs}
                for spec in specs:
                    plan = report_plans[spec.key]
                    mode = "READY" if plan.rebuild else "SKIP"
                    controller.status(spec.key, "report", mode, plan.detail)
                    controller.log(
                        f"REPORT {mode} benchmark={spec.label} detail={plan.detail}"
                    )
                outcome = execute_workflow(
                    specs,
                    is_reusable=controller.is_reusable,
                    ensure_service=controller.ensure_service,
                    run_inference=controller.run_inference,
                    start_publication=lambda spec: controller.start_publication(
                        spec, report_plans[spec.key]
                    ),
                    stop_service=controller.stop_service,
                    wait_publications=controller.wait_publications,
                )
                if outcome.inference_error:
                    controller.log(f"FAIL inference_lane={outcome.inference_error}")
                final_errors: list[str] = []
                if outcome.inference_error is None and all(
                    status == 0 for status in outcome.publication_status.values()
                ):
                    for spec in specs:
                        final_errors.extend(
                            f"{spec.label}: {error}" for error in target_result_errors(spec)
                        )
                        if report_plans[spec.key].rebuild:
                            final_errors.extend(
                                f"{spec.label}: {error}"
                                for error in report_result_errors(spec)
                            )
                if not outcome.passed or final_errors:
                    detail = outcome.inference_error or "; ".join(final_errors) or str(
                        dict(outcome.publication_status)
                    )
                    controller.status("workflow", "final", "FAIL", detail)
                    return 1
                completion = "; ".join(
                    f"{spec.label}:score=complete,report="
                    f"{'complete' if report_plans[spec.key].rebuild else 'skipped'}"
                    for spec in specs
                )
                controller.status("workflow", "final", "COMPLETE", completion)
                controller.log(f"COMPLETE {completion}")
                return 0
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
    except ResourceBlocked as exc:
        print(f"[internvl3-78b-three-bench] BLOCKED {exc}", file=sys.stderr)
        return EXIT_RESOURCE_BLOCKED
    except ConfigurationError as exc:
        print(f"[internvl3-78b-three-bench] ERROR {exc}", file=sys.stderr)
        return EXIT_CONFIGURATION
    except KeyboardInterrupt:
        if controller is not None:
            controller.cleanup()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
