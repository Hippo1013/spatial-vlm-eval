"""Fail-isolated two-GPU/API schedule for the 20 runnable SPBench-SI tracks."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...models.common.runtime import atomic_write_json, utc_now
from .profiles import PROFILE_SEQUENCE, PROFILES


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    phase: str
    lane: str
    profile: str
    gpu_ids: tuple[int, ...]
    port: int | None = None


SCHEDULE = (
    ScheduledJob("A", "dual", "internvl3_38b", (0, 1), 18100),
    ScheduledJob("A", "dual", "llava_next_yi_34b", (0, 1), 18100),
    ScheduledJob("A", "dual", "qwen3_vl_32b", (0, 1), 18100),
    ScheduledJob("A", "api", "gpt5_openrouter_non_zdr", ()),
    ScheduledJob("A", "api", "gemini31pro_openrouter_non_zdr", ()),
    ScheduledJob("B", "gpu0", "3dthinker_rgb", (0,)),
    ScheduledJob("B", "gpu0", "spatialrgpt_rgb", (0,)),
    ScheduledJob("B", "gpu0", "llava_next_mistral_7b", (0,), 18101),
    ScheduledJob("B", "gpu0", "internvl3_8b", (0,), 18101),
    ScheduledJob("B", "gpu0", "qwen3_vl_8b", (0,), 18101),
    ScheduledJob("B", "gpu0", "qwen3_vl_4b", (0,), 18101),
    ScheduledJob("B", "gpu0", "qwen3_vl_2b", (0,), 18101),
    ScheduledJob("B", "gpu1", "ssr_native", (1,)),
    ScheduledJob("B", "gpu1", "ssr_rgb", (1,)),
    ScheduledJob("B", "gpu1", "hispatial3b_moge2_xyz", (1,)),
    ScheduledJob("B", "gpu1", "spatialbot_zoedepth", (1,)),
    ScheduledJob("B", "gpu1", "spatialbot_rgb", (1,)),
    ScheduledJob("B", "gpu1", "robobrain25_8b_nv_rgb", (1,)),
    ScheduledJob("B", "gpu1", "robobrain25_8b_mt_rgb", (1,)),
    ScheduledJob("B", "gpu1", "spatialladder3b_rgb", (1,)),
)
LANE_PHASE_BARRIER = {"gpu0": "dual", "gpu1": "dual"}


def jobs_for_lane(lane: str) -> tuple[ScheduledJob, ...]:
    return tuple(job for job in SCHEDULE if job.lane == lane)


def schedule_errors() -> list[str]:
    errors: list[str] = []
    keys = [job.profile for job in SCHEDULE]
    expected = set(PROFILE_SEQUENCE) - {"internvl3_78b"}
    if len(keys) != 20 or len(keys) != len(set(keys)) or set(keys) != expected:
        errors.append("schedule must cover exactly the 20 non-InternVL3-78B profiles once")
    if [job.profile for job in jobs_for_lane("api")] != ["gpt5_openrouter_non_zdr", "gemini31pro_openrouter_non_zdr"]:
        errors.append("API lane must be strictly serial GPT-5 then Gemini 3.1 Pro")
    for job in SCHEDULE:
        profile = PROFILES[job.profile]
        if job.lane == "dual" and (job.gpu_ids != (0, 1) or profile.default_tensor_parallel_size != 2):
            errors.append(f"dual lane resource mismatch: {job.profile}")
        if job.lane == "gpu0" and job.gpu_ids != (0,):
            errors.append(f"gpu0 lane resource mismatch: {job.profile}")
        if job.lane == "gpu1" and job.gpu_ids != (1,):
            errors.append(f"gpu1 lane resource mismatch: {job.profile}")
        if job.lane == "api" and job.gpu_ids:
            errors.append(f"API job unexpectedly owns GPUs: {job.profile}")
    if LANE_PHASE_BARRIER != {"gpu0": "dual", "gpu1": "dual"}:
        errors.append("phase-B lanes must wait only for the dual lane")
    return errors


def schedule_payload(stage: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "excluded": {"profile": "internvl3_78b", "reason": "fixed TP=4 four-80GB-GPU route"},
        "jobs": [{**asdict(job), "gpu_ids": list(job.gpu_ids)} for job in SCHEDULE],
        "api_overlap": False,
        "controller_scores": False,
    }


def environment_for_job(job: ScheduledJob) -> dict[str, str]:
    profile = PROFILES[job.profile]
    token = "".join(character if character.isalnum() else "_" for character in job.profile).upper()
    environment = os.environ.copy()
    if job.gpu_ids:
        value = ",".join(str(index) for index in job.gpu_ids)
        environment["CUDA_VISIBLE_DEVICES"] = value
        environment[f"SPBENCH_SI_{token}_GPU_IDS"] = value
    if job.port is not None:
        environment[f"SPBENCH_SI_{token}_BASE_URLS"] = f"http://127.0.0.1:{job.port}/v1"
        environment["SPBENCH_SI_VLLM_PORT"] = str(job.port)
        environment["SPBENCH_SI_VLLM_MAX_MODEL_LEN"] = "4096" if profile.family == "llava_next" else environment.get("SPBENCH_SI_VLLM_MAX_MODEL_LEN", "32768")
    return environment


def _append_status(
    control_root: Path, *, run_id: str, phase: str, lane: str, profile: str, state: str, detail: str
) -> None:
    path = control_root / "status.tsv"
    line = f"{run_id}\t{phase}\t{lane}\t{profile}\t{state}\t{utc_now()}\t{detail}\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _signal_watcher(run_id: str, lane: str) -> None:
    if shutil.which("tmux"):
        subprocess.run(["tmux", "wait-for", "-S", f"spbench-si-{run_id}-{lane}"], check=False, capture_output=True)


def _port_is_available(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def _port_is_listening(port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def _wait_for_port_release(port: int, timeout: float | None = None) -> None:
    if timeout is None:
        timeout = float(os.environ.get("SPBENCH_SI_SCHEDULE_PORT_RELEASE_TIMEOUT_SECONDS", "180"))
    poll = max(0.1, float(os.environ.get("SPBENCH_SI_SCHEDULE_POLL_SECONDS", "1")))
    deadline = time.monotonic() + timeout
    while not _port_is_available(port):
        if time.monotonic() >= deadline:
            raise ResourceWarning(f"owned vLLM stopped but port {port} did not release")
        time.sleep(poll)


def _stop_owned_group(process: subprocess.Popen[Any], timeout: float = 20.0) -> None:
    if process.poll() is not None:
        process.wait()
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _wait_for_service(port: int, process: subprocess.Popen[Any], timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"owned vLLM exited early with status {process.returncode}")
        if _port_is_listening(port):
            return
        time.sleep(2)
    raise TimeoutError(f"owned vLLM did not bind port {port} within {timeout}s")


def _run_owned(command: list[str], *, environment: dict[str, str], log: Any) -> int:
    return subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def _run_job(
    job: ScheduledJob,
    *,
    stage: str,
    repository_root: Path,
    control_root: Path,
    run_id: str,
) -> int:
    environment = environment_for_job(job)
    log_path = control_root / "logs" / f"{job.phase}-{job.lane}-{job.profile}.log"
    server: subprocess.Popen[Any] | None = None
    _append_status(control_root, run_id=run_id, phase=job.phase, lane=job.lane, profile=job.profile, state="START", detail=stage)
    _signal_watcher(run_id, job.lane)
    error: Exception | None = None
    try:
        with log_path.open("a", encoding="utf-8") as log:
            if job.port is not None:
                if not _port_is_available(job.port):
                    raise RuntimeError(f"port {job.port} is occupied; controller will not adopt or terminate it")
                server = subprocess.Popen(
                    [
                        "bash", "scripts/spbench_si/serve_vllm_profile.sh",
                        "--model", job.profile,
                        "--gpu-ids", ",".join(str(index) for index in job.gpu_ids),
                        "--port", str(job.port),
                    ],
                    cwd=repository_root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _wait_for_service(job.port, server)
            stages = ("test", "full") if stage == "full" else ("test",)
            status = 0
            for requested_stage in stages:
                command = [
                    "bash", "scripts/spbench_si/run_inference.sh",
                    "--stage", requested_stage, "--model", job.profile,
                ]
                status = _run_owned(command, environment=environment, log=log)
                if status != 0:
                    break
        if status != 0:
            raise RuntimeError(f"run_inference exited {status}")
    except Exception as exc:  # noqa: BLE001
        error = exc
    finally:
        if server is not None:
            try:
                _stop_owned_group(server)
                assert job.port is not None
                _wait_for_port_release(job.port)
            except Exception as exc:  # noqa: BLE001
                if error is None:
                    error = exc
    if error is None:
        _append_status(control_root, run_id=run_id, phase=job.phase, lane=job.lane, profile=job.profile, state="PASS", detail=f"{stage} complete")
        _signal_watcher(run_id, job.lane)
        return 0
    _append_status(control_root, run_id=run_id, phase=job.phase, lane=job.lane, profile=job.profile, state="FAIL", detail=f"{type(error).__name__}: {error}"[:500])
    _signal_watcher(run_id, job.lane)
    return 1


def run_lane(
    lane: str,
    *,
    stage: str,
    run_id: str,
    repository_root: Path,
    control_root: Path,
) -> int:
    for job in jobs_for_lane(lane):
        if _run_job(job, stage=stage, repository_root=repository_root, control_root=control_root, run_id=run_id):
            _append_status(control_root, run_id=run_id, phase=job.phase, lane=lane, profile="-", state="FAIL", detail="lane stopped after job failure")
            _signal_watcher(run_id, lane)
            return 1
    phase = jobs_for_lane(lane)[0].phase
    _append_status(control_root, run_id=run_id, phase=phase, lane=lane, profile="-", state="COMPLETE", detail="lane passed")
    _signal_watcher(run_id, lane)
    return 0


def _start_lane(lane: str, *, stage: str, run_id: str, repository_root: Path, control_root: Path) -> subprocess.Popen[Any]:
    return subprocess.Popen([
        sys.executable, "-m", "spatial_vlm_eval.benchmarks.spbench_si.scheduled_batch",
        "--internal-lane", lane, "--stage", stage, "--run-id", run_id,
        "--control-root", str(control_root), "--repository-root", str(repository_root),
    ])


def _start_watcher(lane: str, *, run_id: str, control_root: Path) -> subprocess.Popen[Any]:
    return subprocess.Popen([
        sys.executable, "-m", "spatial_vlm_eval.benchmarks.spbench_si.scheduled_watcher",
        "--lane", lane, "--run-id", run_id, "--control-root", str(control_root),
    ])


def _run_controller_locked(repository_root: Path, *, stage: str, control_root: Path) -> int:
    """Run one schedule after the caller has acquired the output-root lock."""

    run_id = utc_now().replace(":", "-")
    status = control_root / "status.tsv"
    if not status.exists():
        status.write_text("run_id\tphase\tlane\tprofile\tstate\ttimestamp\tdetail\n", encoding="utf-8")
    atomic_write_json(control_root / "frozen-plan.json", schedule_payload(stage))
    watchers = {
        lane: _start_watcher(lane, run_id=run_id, control_root=control_root)
        for lane in ("dual", "api", "gpu0", "gpu1")
    }
    dual = _start_lane("dual", stage=stage, run_id=run_id, repository_root=repository_root, control_root=control_root)
    api = _start_lane("api", stage=stage, run_id=run_id, repository_root=repository_root, control_root=control_root)
    dual_status = dual.wait()
    phase_b: list[subprocess.Popen[Any]] = []
    if dual_status == 0:
        phase_b = [
            _start_lane(lane, stage=stage, run_id=run_id, repository_root=repository_root, control_root=control_root)
            for lane in ("gpu0", "gpu1")
        ]
    statuses = [dual_status, api.wait(), *(process.wait() for process in phase_b)]
    if dual_status != 0:
        for lane in ("gpu0", "gpu1"):
            _append_status(
                control_root, run_id=run_id, phase="B", lane=lane, profile="-",
                state="FAIL", detail="not started because dual lane failed",
            )
            _signal_watcher(run_id, lane)
    for watcher in watchers.values():
        watcher.wait(timeout=30)
    return 0 if all(status == 0 for status in statuses) else 1


def run_controller(repository_root: Path, *, stage: str) -> int:
    output_root = Path(os.environ["SPBENCH_SI_OUTPUT_ROOT"]).resolve()
    control_root = output_root / "_scheduled_batch"
    control_root.mkdir(parents=True, exist_ok=True)
    (control_root / "logs").mkdir(exist_ok=True)
    lock_path = control_root / "lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[spbench-si-schedule] another run holds {lock_path}", file=sys.stderr)
            return 4
        return _run_controller_locked(repository_root, stage=stage, control_root=control_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("test", "full"), default="full")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--without-internvl78", action="store_true")
    parser.add_argument("--with-paid-api", action="store_true")
    parser.add_argument("--internal-lane", choices=("dual", "api", "gpu0", "gpu1"))
    parser.add_argument("--run-id")
    parser.add_argument("--control-root")
    parser.add_argument("--repository-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = schedule_errors()
    if errors:
        raise RuntimeError("; ".join(errors))
    repository_root = Path(args.repository_root).resolve() if args.repository_root else Path(__file__).resolve().parents[4]
    if args.internal_lane:
        if not args.run_id or not args.control_root:
            raise ValueError("internal lane requires run id and control root")
        raise SystemExit(run_lane(
            args.internal_lane, stage=args.stage, run_id=args.run_id, repository_root=repository_root,
            control_root=Path(args.control_root).resolve(),
        ))
    if args.list:
        for job in SCHEDULE:
            print(f"{job.phase}\t{job.lane}\t{job.profile}\t{','.join(map(str, job.gpu_ids)) or '-'}")
        print("excluded\tinternvl3_78b\tTP=4")
        return
    if args.check:
        print(json.dumps({"passed": True, "schedule": schedule_payload(args.stage)}, ensure_ascii=False))
        return
    if args.dry_run:
        for job in SCHEDULE:
            print(f"dry-run stage={args.stage} phase={job.phase} lane={job.lane} profile={job.profile}; no dataset/model/runner/GPU/API/scorer call")
        return
    if not (args.without_internvl78 and args.with_paid_api):
        raise ValueError("formal double-A800 schedule requires both --without-internvl78 and --with-paid-api")
    if "SPBENCH_SI_OUTPUT_ROOT" not in os.environ:
        raise ValueError("Set SPBENCH_SI_OUTPUT_ROOT")
    raise SystemExit(run_controller(repository_root, stage=args.stage))


if __name__ == "__main__":
    main()
