#!/usr/bin/env python3
"""Two-lane MSMU SOTA supplement inference, scoring, and report controller."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

from spatial_vlm_eval.benchmarks.msmu.data import (
    OFFICIAL_TEST_SIZE,
    MSMUTestContract,
    load_arrow_split,
)
from spatial_vlm_eval.benchmarks.msmu.scorer import SCORER_PROTOCOL
from spatial_vlm_eval.benchmarks.msmu.smoke_indices import select_type_covering_indices
from spatial_vlm_eval.models.common.provenance import (
    verify_git_checkout,
    verify_hf_snapshot_revision,
)
from spatial_vlm_eval.models.sota_spatial.common import adapter_source_digest
from spatial_vlm_eval.models.sota_spatial.hispatial import (
    MOGE2_MODEL_ID,
    MOGE2_REVISION,
    MOGE2_UPSTREAM_COMMIT,
    MOGE2_UTILS3D_COMMIT,
)
from spatial_vlm_eval.models.profiles import (
    PROFILES,
    SOTA_SUPPLEMENT_PROFILE_KEYS,
    SOTA_SUPPLEMENT_REPORT_PROFILE_KEYS,
)


LANES: dict[str, tuple[str, ...]] = {
    "gpu0": (
        "robobrain25_8b_nv_rgb",
        "hispatial3b_moge2_xyz",
        "spatialladder3b_rgb",
    ),
    "gpu1": (
        "robobrain25_8b_mt_rgb",
        "spatialladder3b_thinking",
    ),
}
SCORE_ORDER = SOTA_SUPPLEMENT_PROFILE_KEYS
RUN_SLUGS = {
    "robobrain25_8b_nv_rgb": "robobrain25-8b-nv-rgb",
    "robobrain25_8b_mt_rgb": "robobrain25-8b-mt-rgb",
    "hispatial3b_moge2_xyz": "hispatial3b-moge2-xyz",
    "spatialladder3b_rgb": "spatialladder3b-rgb-direct",
    "spatialladder3b_thinking": "spatialladder3b-thinking",
}
PROFILE_MODEL_ENVS = {
    "robobrain25_8b_nv_rgb": "ROBOBRAIN25_8B_NV_MODEL",
    "robobrain25_8b_mt_rgb": "ROBOBRAIN25_8B_MT_MODEL",
    "hispatial3b_moge2_xyz": "HISPATIAL_3B_MODEL",
    "spatialladder3b_rgb": "SPATIALLADDER_3B_MODEL",
    "spatialladder3b_thinking": "SPATIALLADDER_3B_MODEL",
}
FAMILY_UPSTREAM_ENVS = {
    "robobrain25": "ROBOBRAIN25_UPSTREAM_ROOT",
    "hispatial": "HISPATIAL_UPSTREAM_ROOT",
    "spatialladder": "SPATIALLADDER_UPSTREAM_ROOT",
}
FAMILY_PYTHON_ENVS = {
    "robobrain25": "ROBOBRAIN25_PYTHON",
    "hispatial": "HISPATIAL_PYTHON",
    "spatialladder": "SPATIALLADDER_PYTHON",
}
STATUS_FIELDS = ("timestamp", "lane", "profile", "phase", "state", "detail")


class ConfigurationError(RuntimeError):
    pass


class UnsafeArtifactError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UnsafeArtifactError(f"invalid JSON artifact {path}: {exc}") from exc


class FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: TextIO | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise ConfigurationError(f"lock is already held: {self.path}") from exc
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


class StatusTable:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.events: list[dict[str, str]] = []
        self.lock = threading.Lock()
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as handle:
                self.events.extend(dict(row) for row in csv.DictReader(handle, delimiter="\t"))
        self.notifiers: dict[str, int] = {}

    def attach_notifier(self, lane: str, descriptor: int) -> None:
        self.notifiers[lane] = descriptor

    def add(self, lane: str, profile: str, phase: str, state: str, detail: str) -> None:
        event = {
            "timestamp": utc_now(),
            "lane": lane,
            "profile": profile,
            "phase": phase,
            "state": state,
            "detail": str(detail).replace("\t", " ").replace("\n", " ")[:1000],
        }
        with self.lock:
            self.events.append(event)
            lines = ["\t".join(STATUS_FIELDS)]
            lines.extend("\t".join(row.get(field, "") for field in STATUS_FIELDS) for row in self.events)
            atomic_write_text(self.path, "\n".join(lines) + "\n")
            descriptor = self.notifiers.get(lane)
            if descriptor is not None:
                try:
                    os.write(descriptor, b"1")
                except OSError:
                    pass

    def close_notifiers(self) -> None:
        for descriptor in self.notifiers.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.notifiers.clear()


class Controller:
    def __init__(self, manual_root: Path, repository: Path) -> None:
        self.manual_root = manual_root.resolve()
        self.stage3_root = self.manual_root / "03_full987"
        self.control_root = self.stage3_root / "_sota_supplement"
        self.repository = repository.resolve()
        self.status = StatusTable(self.control_root / "status.tsv")
        self.active: dict[int, subprocess.Popen[Any]] = {}
        self.active_lock = threading.Lock()
        self.failure = threading.Event()
        self.watchers: list[subprocess.Popen[Any]] = []
        self.smoke_indices: list[int] = []
        self.dataset_fingerprint = ""
        self.selected_capacities: dict[str, int] = {}

    def stage_run_dir(self, stage: str, profile_key: str) -> Path:
        profile = PROFILES[profile_key]
        return (
            self.manual_root
            / stage
            / RUN_SLUGS[profile_key]
            / profile.revision
            / profile.inference_protocol
            / SCORER_PROTOCOL
        )

    def predictions_path(self, stage: str, profile_key: str) -> Path:
        return self.stage_run_dir(stage, profile_key) / "predictions.jsonl"

    def start_watchers(self) -> None:
        watcher = self.repository / "scripts" / "msmu" / "_sota_event_watcher.py"
        for lane in LANES:
            read_fd, write_fd = os.pipe()
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(watcher),
                    "--fd",
                    str(read_fd),
                    "--lane",
                    lane,
                    "--status",
                    str(self.status.path),
                ],
                pass_fds=(read_fd,),
            )
            os.close(read_fd)
            self.status.attach_notifier(lane, write_fd)
            self.watchers.append(process)

    def stop_watchers(self) -> None:
        self.status.close_notifiers()
        for watcher in self.watchers:
            try:
                watcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                watcher.terminate()
                watcher.wait(timeout=10)
        self.watchers.clear()

    def run_owned(
        self,
        command: list[str],
        *,
        environment: dict[str, str],
        log_path: Path,
    ) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{utc_now()}] command={subprocess.list2cmdline(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=self.repository,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            with self.active_lock:
                self.active[process.pid] = process
            try:
                return process.wait()
            finally:
                with self.active_lock:
                    self.active.pop(process.pid, None)

    def stop_owned_processes(self) -> None:
        with self.active_lock:
            processes = list(self.active.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and any(process.poll() is None for process in processes):
            time.sleep(0.25)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def select_smoke_indices(self, *, write_report: bool) -> None:
        test = load_arrow_split(os.environ["DATASET_ROOT"], "test")
        selections = select_type_covering_indices(test)
        self.smoke_indices = [int(item["index"]) for item in selections]
        report = {
            "dataset_root": str(Path(os.environ["DATASET_ROOT"]).resolve()),
            "split": "test",
            "official_test_size": OFFICIAL_TEST_SIZE,
            "selections": selections,
            "indices_csv": ",".join(str(index) for index in self.smoke_indices),
            "debug_subset_only": True,
        }
        if write_report:
            atomic_write_json(self.manual_root / "02_smoke8" / "selected_indices.json", report)

    def inspect_inference_artifacts(
        self,
        *,
        stage: str,
        profile_key: str,
        expected_indices: list[int],
        require_canary: bool,
    ) -> bool:
        run_dir = self.stage_run_dir(stage, profile_key)
        predictions = run_dir / "predictions.jsonl"
        if not predictions.exists():
            return False
        metadata_path = run_dir / "predictions.jsonl.metadata.json"
        validation_path = run_dir / "prediction_validation.json"
        required = [metadata_path, validation_path]
        if require_canary:
            required.append(run_dir / "vision_canary.json")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise UnsafeArtifactError(
                f"finalized predictions have missing required artifacts: {missing}; path={predictions}"
            )

        profile = PROFILES[profile_key]
        metadata = read_json(metadata_path)
        model = metadata.get("model") if isinstance(metadata, dict) else None
        dataset = metadata.get("dataset") if isinstance(metadata, dict) else None
        checks = {
            "top-level inference protocol": metadata.get("inference_protocol") == profile.inference_protocol,
            "scorer protocol": metadata.get("scorer_protocol") == SCORER_PROTOCOL,
            "model profile": isinstance(model, dict) and model.get("profile") == profile_key,
            "model revision": isinstance(model, dict) and model.get("model_revision") == profile.revision,
            "nested inference protocol": isinstance(model, dict)
            and model.get("inference_protocol") == profile.inference_protocol,
            "adapter digest": isinstance(model, dict)
            and model.get("adapter_source_sha256") == adapter_source_digest(profile_key),
            "dataset fingerprint": isinstance(dataset, dict)
            and dataset.get("fingerprint") == self.dataset_fingerprint,
            "target indices": isinstance(dataset, dict)
            and dataset.get("target_indices") == expected_indices,
            "prediction count": metadata.get("num_predictions") == len(expected_indices),
            "publishable flag": metadata.get("publishable_inference")
            is (len(expected_indices) == OFFICIAL_TEST_SIZE),
        }
        failures = [name for name, passed in checks.items() if not passed]

        rows: list[Any] = []
        try:
            with predictions.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(f"predictions JSONL invalid: {exc}")
        if len(rows) != len(expected_indices) or [row.get("index") for row in rows] != expected_indices:
            failures.append("prediction indices do not match the frozen target")

        validation = read_json(validation_path)
        if validation.get("passed") is not True or validation.get("errors") != []:
            failures.append("prediction validator did not pass")
        if validation.get("num_prediction_rows") != len(expected_indices):
            failures.append("prediction validator row count mismatch")
        if validation.get("allow_subset") is not (len(expected_indices) != OFFICIAL_TEST_SIZE):
            failures.append("prediction validator subset identity mismatch")

        if require_canary:
            canary = read_json(run_dir / "vision_canary.json")
            canary_checks = {
                "passed": canary.get("passed") is True,
                "profile": canary.get("profile") == profile_key,
                "model_revision": canary.get("model_revision") == profile.revision,
                "inference_protocol": canary.get("inference_protocol") == profile.inference_protocol,
                "adapter_digest": canary.get("adapter_source_sha256")
                == adapter_source_digest(profile_key),
                "image_count": canary.get("request_image_count") == 1,
            }
            failures.extend(
                f"vision canary {name}" for name, passed in canary_checks.items() if not passed
            )
            if profile.family == "spatialladder":
                capacity = canary.get("capacity_probe")
                if not isinstance(capacity, dict) or capacity.get("passed") is not True:
                    failures.append("SpatialLadder native batch capacity probe did not pass")
                elif capacity.get("tokenizer_padding_side") != "left":
                    failures.append("SpatialLadder capacity probe did not prove left padding")
                else:
                    self.selected_capacities[profile_key] = int(
                        capacity["selected_capacity"]
                    )

        if profile.family == "spatialladder":
            selected_capacity = self.selected_capacities.get(profile_key)
            decoding = model.get("decoding") if isinstance(model, dict) else None
            if selected_capacity is None:
                failures.append("SpatialLadder selected native capacity is unavailable")
            elif not isinstance(decoding, dict) or decoding.get(
                "native_batch_size"
            ) != selected_capacity:
                failures.append("SpatialLadder metadata native batch size mismatch")

        if failures:
            raise UnsafeArtifactError(
                f"refusing to overwrite invalid finalized artifact {predictions}: "
                + "; ".join(failures)
            )
        return True

    def run_stage(self, lane: str, gpu: str, profile_key: str, stage: str) -> None:
        stage_config = {
            "01_canary": ([0], True, "1"),
            "02_smoke8": (self.smoke_indices, False, "2"),
            "03_full987": (list(range(OFFICIAL_TEST_SIZE)), False, "3"),
        }
        expected_indices, require_canary, stage_number = stage_config[stage]
        if self.inspect_inference_artifacts(
            stage=stage,
            profile_key=profile_key,
            expected_indices=expected_indices,
            require_canary=require_canary,
        ):
            self.status.add(lane, profile_key, stage, "PASS", "reused complete canonical artifacts")
            return
        environment = dict(os.environ)
        environment.update(
            {
                "MANUAL_DRY_RUN": "0",
                "MANUAL_CUDA_VISIBLE_DEVICES": gpu,
                "CUDA_VISIBLE_DEVICES": gpu,
                "MSMU_SMOKE_INDICES": ",".join(str(index) for index in self.smoke_indices),
                "SPATIALLADDER_BATCH_SIZE": str(
                    self.selected_capacities.get(profile_key, 1)
                ),
            }
        )
        command = [
            "bash",
            str(self.repository / "scripts" / "msmu" / f"run_manual_stage{stage_number}.sh"),
            profile_key,
        ]
        self.status.add(lane, profile_key, stage, "RUNNING", "owned process group started")
        return_code = self.run_owned(
            command,
            environment=environment,
            log_path=self.control_root / "logs" / lane / f"{profile_key}.{stage}.log",
        )
        if return_code != 0:
            self.status.add(lane, profile_key, stage, "FAIL", f"exit={return_code}")
            raise RuntimeError(f"{profile_key} {stage} failed with exit {return_code}")
        if not self.inspect_inference_artifacts(
            stage=stage,
            profile_key=profile_key,
            expected_indices=expected_indices,
            require_canary=require_canary,
        ):
            self.status.add(lane, profile_key, stage, "FAIL", "artifacts absent after exit 0")
            raise RuntimeError(f"{profile_key} {stage} produced no finalized artifacts")
        self.status.add(lane, profile_key, stage, "PASS", "validator and provenance passed")

    def run_lane(self, lane: str, gpu: str) -> None:
        try:
            for profile_key in LANES[lane]:
                if self.failure.is_set():
                    return
                for stage in ("01_canary", "02_smoke8", "03_full987"):
                    self.run_stage(lane, gpu, profile_key, stage)
            self.status.add(lane, "-", "lane", "COMPLETE", "all frozen profiles complete")
        except BaseException as exc:
            self.failure.set()
            self.status.add(
                lane,
                profile_key if "profile_key" in locals() else "-",
                "lane",
                "FAULT",
                f"{type(exc).__name__}: {exc}",
            )
            raise

    def run_lanes(self, gpu0: str, gpu1: str) -> None:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="msmu-sota-lane") as executor:
            futures = [
                executor.submit(self.run_lane, "gpu0", gpu0),
                executor.submit(self.run_lane, "gpu1", gpu1),
            ]
            completed, _pending = wait(futures, return_when=FIRST_EXCEPTION)
            error = next((future.exception() for future in completed if future.exception()), None)
            if error is not None:
                self.failure.set()
                self.stop_owned_processes()
                wait(futures)
                raise error
            wait(futures)
            for future in futures:
                future.result()

    def score_and_report(self, gpu0: str) -> None:
        from _score_pending_results import complete_state_errors

        for profile_key in SCORE_ORDER:
            if not self.inspect_inference_artifacts(
                stage="03_full987",
                profile_key=profile_key,
                expected_indices=list(range(OFFICIAL_TEST_SIZE)),
                require_canary=False,
            ):
                raise RuntimeError(f"full inference is incomplete: {profile_key}")

        pending_profiles = []
        for profile_key in SCORE_ORDER:
            predictions = self.predictions_path("03_full987", profile_key)
            score_dir = predictions.parent / "scores" / SCORER_PROTOCOL
            if complete_state_errors(predictions, score_dir):
                pending_profiles.append(profile_key)
            else:
                self.status.add(
                    "control", profile_key, "score", "PASS", "reused publication-gated summary"
                )

        environment = dict(os.environ)
        environment.update(
            {
                "MANUAL_DRY_RUN": "0",
                "MANUAL_JUDGE_CUDA_VISIBLE_DEVICES": gpu0,
                "CUDA_VISIBLE_DEVICES": gpu0,
            }
        )
        if pending_profiles:
            self._score_pending_profiles(pending_profiles, environment)

        check_command = [
            "bash",
            str(self.repository / "scripts" / "msmu" / "build_results_report.sh"),
            "--check",
            "--results-root",
            str(self.stage3_root),
        ]
        if self.run_owned(
            check_command,
            environment=environment,
            log_path=self.control_root / "logs" / "report.check.log",
        ) != 0:
            raise RuntimeError("frozen 23-profile report check failed")

        report = self.stage3_root / "msmu-result.md"
        build_command = [
            "bash",
            str(self.repository / "scripts" / "msmu" / "build_results_report.sh"),
            "--results-root",
            str(self.stage3_root),
            "--output",
            str(report),
        ]
        for profile_key in SOTA_SUPPLEMENT_REPORT_PROFILE_KEYS:
            build_command.extend(("--profile", profile_key))
        if self.run_owned(
            build_command,
            environment=environment,
            log_path=self.control_root / "logs" / "report.build.log",
        ) != 0:
            raise RuntimeError("atomic MSMU report rebuild failed")
        row_count = sum(
            1
            for line in report.read_text(encoding="utf-8").splitlines()
            if line.startswith("| ")
        ) - 2
        if row_count != 23:
            raise RuntimeError(f"rebuilt MSMU report has {row_count} rows instead of 23")
        self.status.add("control", "report", "report", "COMPLETE", "23 rows atomically rebuilt")

    def _score_pending_profiles(
        self,
        pending_profiles: list[str],
        environment: dict[str, str],
    ) -> None:
        from _score_pending_results import complete_state_errors

        judge_command = [
            "bash",
            str(self.repository / "scripts" / "msmu" / "run_manual_stage3.sh"),
            "judge",
            "serve",
        ]
        judge_log = self.control_root / "logs" / "judge.log"
        judge_log.parent.mkdir(parents=True, exist_ok=True)
        handle = judge_log.open("a", encoding="utf-8")
        judge = subprocess.Popen(
            judge_command,
            cwd=self.repository,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        with self.active_lock:
            self.active[judge.pid] = judge
        self.status.add("control", "judge", "scoring", "RUNNING", "single judge started")
        try:
            wait_for_model(
                repository=self.repository,
                python=os.environ.get("LATENT_PYTHON", sys.executable),
                base_url=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:18080/v1"),
                expected_model=os.environ.get("JUDGE_MODEL_NAME", "msmu-judge"),
                process=judge,
            )
            for profile_key in pending_profiles:
                predictions = self.predictions_path("03_full987", profile_key)
                score_dir = predictions.parent / "scores" / SCORER_PROTOCOL
                command = [
                    "bash",
                    str(self.repository / "scripts" / "msmu" / "score_pending_results.sh"),
                    "--results-root",
                    str(self.stage3_root),
                    "--predictions",
                    str(predictions),
                ]
                return_code = self.run_owned(
                    command,
                    environment=environment,
                    log_path=self.control_root / "logs" / f"score.{profile_key}.log",
                )
                if return_code != 0:
                    self.status.add(
                        "control", profile_key, "score", "FAIL", f"exit={return_code}"
                    )
                    raise RuntimeError(f"scoring failed for {profile_key}: exit {return_code}")
                errors = complete_state_errors(predictions, score_dir)
                if errors:
                    raise RuntimeError(
                        f"publication gates incomplete for {profile_key}: {'; '.join(errors)}"
                    )
                self.status.add(
                    "control", profile_key, "score", "PASS", "publication gates complete"
                )
        finally:
            if judge.poll() is None:
                try:
                    os.killpg(judge.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    judge.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(judge.pid, signal.SIGKILL)
                    judge.wait(timeout=30)
            with self.active_lock:
                self.active.pop(judge.pid, None)
            handle.close()



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list", action="store_true", dest="list_plan")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--status", action="store_true")
    parser.add_argument("--manual-output-root")
    return parser.parse_args(argv)


def print_plan() -> None:
    for lane, profiles in LANES.items():
        for order, profile_key in enumerate(profiles, start=1):
            print(f"{lane}\t{order}\t{profile_key}")
    print("score\t" + "\t".join(SCORE_ORDER))
    print("report\tbaseline18+main4+thinking1\t23")


def required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"missing required environment value: {name}")
    return value


def check_endpoint_free(host: str, port: int) -> None:
    with socket.socket() as connection:
        connection.settimeout(1)
        if connection.connect_ex((host, port)) == 0:
            raise ConfigurationError(f"owned judge endpoint is already occupied: {host}:{port}")


def gpu_inventory(gpu_ids: tuple[str, str]) -> None:
    if len(set(gpu_ids)) != 2:
        raise ConfigurationError("SOTA supplement lanes require two distinct GPU IDs")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory: dict[str, tuple[int, int, int]] = {}
    for line in completed.stdout.splitlines():
        index, total, free, utilization = (item.strip() for item in line.split(","))
        inventory[index] = (int(total), int(free), int(utilization))
    for gpu in gpu_ids:
        if gpu not in inventory:
            raise ConfigurationError(f"configured GPU does not exist: {gpu}")
        total, free, utilization = inventory[gpu]
        if total < 79_000 or free < 70_000 or utilization > 10:
            raise ConfigurationError(
                f"GPU {gpu} is not an idle 80GB device: total={total} free={free} util={utilization}"
            )
        processes = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if processes.returncode != 0:
            raise ConfigurationError(
                f"could not inspect compute processes on GPU {gpu}: {processes.stderr.strip()}"
            )
        if processes.stdout.strip() and "No running processes found" not in processes.stdout:
            raise ConfigurationError(
                f"GPU {gpu} has existing compute processes; no process was stopped"
            )


def check_assets_and_imports(repository: Path) -> None:
    for profile_key in SOTA_SUPPLEMENT_PROFILE_KEYS:
        profile = PROFILES[profile_key]
        model_path = required_environment(PROFILE_MODEL_ENVS[profile_key])
        if not verify_hf_snapshot_revision(model_path, profile.revision, profile.model):
            raise ConfigurationError(f"model revision is not verifiable: {profile_key} -> {model_path}")
        upstream = required_environment(FAMILY_UPSTREAM_ENVS[profile.family])
        if not verify_git_checkout(upstream, profile.upstream_commit or "", profile.family):
            raise ConfigurationError(f"upstream revision is not verifiable: {profile.family} -> {upstream}")

    moge_model = required_environment("MOGE2_MODEL")
    if not verify_hf_snapshot_revision(moge_model, MOGE2_REVISION, MOGE2_MODEL_ID):
        raise ConfigurationError(f"MoGe-2 revision is not verifiable: {moge_model}")
    for variable, commit, label in (
        ("MOGE2_UPSTREAM_ROOT", MOGE2_UPSTREAM_COMMIT, "MoGe-2"),
        ("MOGE2_UTILS3D_ROOT", MOGE2_UTILS3D_COMMIT, "MoGe-2 utils3d"),
    ):
        path = required_environment(variable)
        if not verify_git_checkout(path, commit, label):
            raise ConfigurationError(f"{label} revision is not verifiable: {path}")

    probes = {
        "robobrain25": (
            "import torch, transformers, qwen_vl_utils; "
            "from transformers import AutoModelForImageTextToText, AutoProcessor"
        ),
        "hispatial": (
            "import torch, transformers, utils3d; "
            "from hispatial.inference import HiSpatialPredictor, MoGeProcessor; "
            "from moge.model.v2 import MoGeModel"
        ),
        "spatialladder": (
            "import torch, flash_attn, qwen_vl_utils; "
            "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration"
        ),
    }
    for family, code in probes.items():
        python = required_environment(FAMILY_PYTHON_ENVS[family])
        if not Path(python).is_file() or not os.access(python, os.X_OK):
            raise ConfigurationError(f"configured {family} interpreter is unavailable: {python}")
        environment = dict(os.environ)
        python_paths = [str(repository / "src")]
        if family == "hispatial":
            python_paths.extend(
                [
                    required_environment("HISPATIAL_UPSTREAM_ROOT"),
                    required_environment("MOGE2_UPSTREAM_ROOT"),
                ]
            )
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        completed = subprocess.run(
            [python, "-c", code],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()[-1:]
            raise ConfigurationError(
                f"{family} import probe failed in {python}: {' '.join(detail)}"
            )


def build_plan(controller: Controller, gpu0: str, gpu1: str) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(controller.repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    profiles = {
        key: {
            "model": PROFILES[key].model,
            "revision": PROFILES[key].revision,
            "inference_protocol": PROFILES[key].inference_protocol,
            "adapter_source_sha256": adapter_source_digest(key),
        }
        for key in SOTA_SUPPLEMENT_PROFILE_KEYS
    }
    payload = {
        "schema_version": 1,
        "repository_head": head,
        "dataset_fingerprint": controller.dataset_fingerprint,
        "scorer_protocol": SCORER_PROTOCOL,
        "lanes": {"gpu0": list(LANES["gpu0"]), "gpu1": list(LANES["gpu1"])},
        "gpu_ids": {"gpu0": gpu0, "gpu1": gpu1},
        "smoke8_indices": controller.smoke_indices,
        "score_order": list(SCORE_ORDER),
        "report_profiles": list(SOTA_SUPPLEMENT_REPORT_PROFILE_KEYS),
        "profiles": profiles,
    }
    payload["plan_digest"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def validate_or_write_plan(controller: Controller, plan: dict[str, Any], *, write: bool) -> None:
    path = controller.control_root / "frozen-plan.json"
    if path.exists():
        existing = read_json(path)
        if existing != plan:
            raise ConfigurationError(
                f"existing frozen plan differs from current code/config; use a new output root: {path}"
            )
    elif write:
        atomic_write_json(path, plan)


def wait_for_model(
    *,
    repository: Path,
    python: str,
    base_url: str,
    expected_model: str,
    process: subprocess.Popen[Any],
    timeout: int = 1800,
) -> None:
    probe = repository / "scripts" / "msmu" / "_probe_openai_models.py"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("judge exited before readiness")
        completed = subprocess.run(
            [
                python,
                str(probe),
                "--base-url",
                base_url,
                "--expected-model",
                expected_model,
                "--timeout",
                "5",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError(f"timed out waiting for judge model {expected_model}")


def dry_run(controller: Controller, gpu0: str, gpu1: str) -> int:
    for lane, gpu in (("gpu0", gpu0), ("gpu1", gpu1)):
        for profile_key in LANES[lane]:
            for stage in (1, 2, 3):
                print(
                    f"[msmu-sota] dry-run lane={lane} gpu={gpu}: "
                    f"bash scripts/msmu/run_manual_stage{stage}.sh {profile_key}"
                )
    print("[msmu-sota] dry-run: start one judge on released gpu0")
    for profile_key in SCORE_ORDER:
        print(f"[msmu-sota] dry-run: score exact predictions for {profile_key}")
    print("[msmu-sota] dry-run: report --check, then atomically rebuild 23 rows")
    print("[msmu-sota] dry-run complete; no GPU/model/judge/scorer/report action was taken")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.list_plan:
            print_plan()
            return 0
        raw_root = args.manual_output_root or os.environ.get("MSMU_SOTA_MANUAL_OUTPUT_ROOT", "")
        if not raw_root:
            raise ConfigurationError("manual output root is not configured")
        manual_root = Path(raw_root).expanduser()
        if not manual_root.is_absolute():
            raise ConfigurationError(f"manual output root must be absolute: {manual_root}")
        repository = Path(__file__).resolve().parents[2]
        controller = Controller(manual_root, repository)
        if args.status:
            if controller.status.path.is_file():
                print(controller.status.path.read_text(encoding="utf-8"), end="")
            else:
                print(f"missing\t{controller.status.path}")
            return 0

        gpu0 = os.environ.get("SOTA_SUPPLEMENT_GPU0", "0")
        gpu1 = os.environ.get("SOTA_SUPPLEMENT_GPU1", "1")
        if os.environ.get("MANUAL_DRY_RUN", "0") == "1":
            return dry_run(controller, gpu0, gpu1)

        storage_root_raw = os.environ.get("SERVER_STORAGE_ROOT", "").strip()
        if storage_root_raw:
            storage_root = Path(storage_root_raw).expanduser().resolve()
            try:
                manual_root.resolve().relative_to(storage_root)
            except ValueError as exc:
                raise ConfigurationError(
                    f"formal outputs must stay under SERVER_STORAGE_ROOT={storage_root}: {manual_root}"
                ) from exc
        required_environment("DATASET_ROOT")
        contract = MSMUTestContract(os.environ["DATASET_ROOT"], require_official_size=True)
        controller.dataset_fingerprint = contract.dataset_fingerprint
        controller.select_smoke_indices(write_report=False)
        check_assets_and_imports(repository)
        gpu_inventory((gpu0, gpu1))
        check_endpoint_free("127.0.0.1", 18080)
        plan = build_plan(controller, gpu0, gpu1)
        validate_or_write_plan(controller, plan, write=False)

        supplement_lock = controller.control_root / "lock"
        inference_lock = controller.stage3_root / "_serial_inference" / "batch.lock"
        with ExitStack() as stack:
            stack.enter_context(FileLock(supplement_lock))
            stack.enter_context(FileLock(inference_lock))
        if args.check:
            print(f"[msmu-sota] CHECK dataset_fingerprint={controller.dataset_fingerprint}")
            print(f"[msmu-sota] CHECK gpu0={gpu0} gpu1={gpu1}")
            print("[msmu-sota] CHECK models=5 upstreams=verified imports=verified")
            print("[msmu-sota] CHECK locks=available judge_port=available")
            print(f"[msmu-sota] CHECK control_root={controller.control_root}")
            return 0

        controller.control_root.mkdir(parents=True, exist_ok=True)
        controller.select_smoke_indices(write_report=True)
        validate_or_write_plan(controller, plan, write=True)
        controller.start_watchers()
        try:
            with FileLock(supplement_lock):
                with FileLock(inference_lock):
                    controller.run_lanes(gpu0, gpu1)
                controller.score_and_report(gpu0)
            return 0
        except BaseException:
            controller.failure.set()
            controller.stop_owned_processes()
            raise
        finally:
            controller.stop_watchers()
    except (ConfigurationError, UnsafeArtifactError) as exc:
        print(f"[msmu-sota] configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[msmu-sota] interrupted; rerun the same command to resume", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - preserve a resumable controller failure.
        print(f"[msmu-sota] FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
