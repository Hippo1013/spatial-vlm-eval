"""Read-only, event-driven health watcher for one scheduled Q-Spatial lane."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

LANES = ("dual", "api", "gpu0", "gpu1")
VISIBLE_STATES = {"PASS", "FAIL", "COMPLETE"}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _group_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _status_rows(path: Path, run_id: str, lane: str) -> list[tuple[str, ...]]:
    if not path.is_file():
        return []
    rows: list[tuple[str, ...]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        fields = tuple(line.split("\t", 6))
        if len(fields) == 7 and fields[0] == run_id and fields[2] == lane:
            rows.append(fields)
    return rows


def _latest_run_id(status_path: Path) -> str:
    if not status_path.is_file():
        raise FileNotFoundError(f"scheduled status is missing: {status_path}")
    rows = [line.split("\t", 1)[0] for line in status_path.read_text(encoding="utf-8").splitlines()[1:] if line]
    if not rows:
        raise ValueError(f"scheduled status has no run rows: {status_path}")
    return rows[-1]


def _gpu_is_unexpectedly_idle(gpu_ids: list[int]) -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable or not gpu_ids:
        return False
    for gpu in gpu_ids:
        result = subprocess.run(
            [
                executable,
                f"--id={gpu}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if any(line.strip().isdigit() for line in result.stdout.splitlines()):
            return False
    return True


def _health_anomaly(active: dict[str, Any] | None, now: float) -> str | None:
    if not active or active.get("state") != "START":
        return None
    profile = str(active.get("profile") or "-")
    lane_pid = active.get("lane_process_group")
    worker_pid = active.get("worker_process_group")
    if not _group_alive(lane_pid):
        return f"FAIL profile={profile} abnormal lane exit"
    if isinstance(worker_pid, int) and worker_pid > 0 and not _group_alive(worker_pid):
        return f"FAIL profile={profile} abnormal worker exit"
    started = float(active.get("started_at_epoch") or now)
    pattern = str(active.get("journal_glob") or "")
    journal_paths = [Path(value) for value in glob.glob(pattern, recursive=True)] if pattern else []
    last_progress = max(
        [started, *(path.stat().st_mtime for path in journal_paths if path.is_file())]
    )
    stall_seconds = int(os.environ.get("QSPATIAL_SCHEDULE_STALL_SECONDS", "1800"))
    if now - last_progress >= stall_seconds:
        return f"STALL profile={profile} seconds_without_durable_progress={int(now - last_progress)}"
    idle_grace = int(os.environ.get("QSPATIAL_SCHEDULE_GPU_IDLE_GRACE_SECONDS", "600"))
    gpu_ids = [int(value) for value in active.get("gpu_ids") or []]
    owned_local_process = bool(worker_pid or active.get("service_process_groups"))
    if owned_local_process and now - started >= idle_grace and _gpu_is_unexpectedly_idle(gpu_ids):
        return f"GPU_IDLE profile={profile} gpu={','.join(map(str, gpu_ids))}"
    return None


def _wait_for_event(channel: str, poll_seconds: int) -> bool:
    tmux = shutil.which("tmux")
    if not tmux:
        return False
    cancelled = threading.Event()

    def wake_after_timeout() -> None:
        if not cancelled.wait(poll_seconds):
            subprocess.run(
                [tmux, "wait-for", "-S", channel],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    timer = threading.Thread(target=wake_after_timeout, daemon=True)
    timer.start()
    result = subprocess.run(
        [tmux, "wait-for", channel],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cancelled.set()
    timer.join(timeout=1)
    return result.returncode == 0


def watch(control_root: Path, *, lane: str, run_id: str) -> int:
    status_path = control_root / "status.tsv"
    active_path = control_root / "active" / f"{lane}.json"
    channel = f"qspatial-{run_id}-{lane}"
    poll_seconds = int(os.environ.get("QSPATIAL_SCHEDULE_HEALTH_POLL_SECONDS", "60"))
    if poll_seconds <= 0:
        raise ValueError("QSPATIAL_SCHEDULE_HEALTH_POLL_SECONDS must be positive")
    seen_rows = 0
    last_anomaly: str | None = None
    while True:
        rows = _status_rows(status_path, run_id, lane)
        for row in rows[seen_rows:]:
            _run, _phase, _lane, profile, state, _timestamp, detail = row
            if state in VISIBLE_STATES:
                print(
                    f"[q-spatial-health] {state} lane={lane} profile={profile} detail={detail}",
                    flush=True,
                )
            if state == "COMPLETE":
                return 0
        seen_rows = len(rows)
        anomaly = _health_anomaly(_load_json(active_path), time.time())
        if anomaly and anomaly != last_anomaly:
            print(f"[q-spatial-health] {anomaly} lane={lane}", flush=True)
        last_anomaly = anomaly
        if anomaly and anomaly.startswith("FAIL "):
            return 1
        if not _wait_for_event(channel, poll_seconds):
            print(
                f"[q-spatial-health] FAIL lane={lane} profile=- detail=tmux wait-for unavailable",
                flush=True,
            )
            return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True, choices=LANES)
    parser.add_argument("--run-id")
    parser.add_argument("--control-root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.control_root:
        control_root = Path(args.control_root).resolve()
    else:
        output_root = os.environ.get("QSPATIAL_OUTPUT_ROOT")
        if not output_root:
            raise ValueError("Set QSPATIAL_OUTPUT_ROOT or pass --control-root")
        control_root = Path(output_root).resolve() / "_scheduled_batch"
    run_id = args.run_id or _latest_run_id(control_root / "status.tsv")
    return watch(control_root, lane=args.lane, run_id=run_id)


if __name__ == "__main__":
    raise SystemExit(main())
