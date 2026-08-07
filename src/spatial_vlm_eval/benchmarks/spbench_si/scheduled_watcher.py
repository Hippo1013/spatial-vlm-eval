"""Read-only, event-driven health watcher for SPBench-SI scheduled lanes."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def _events(path: Path, *, lane: str, run_id: str, start_line: int) -> tuple[list[str], int, bool]:
    if not path.is_file():
        return [], start_line, False
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    complete = False
    for line in lines[start_line:]:
        fields = line.split("\t")
        if len(fields) < 7 or fields[0] == "run_id" or fields[0] != run_id or fields[2] != lane:
            continue
        state = fields[4]
        if state in {"PASS", "FAIL", "COMPLETE"}:
            output.append(f"{state} lane={lane} profile={fields[3]} detail={fields[6]}")
        if state in {"FAIL", "COMPLETE"} and fields[3] == "-":
            complete = True
    return output, len(lines), complete


def watch(control_root: str | Path, *, lane: str, run_id: str) -> int:
    root = Path(control_root).resolve()
    status = root / "status.tsv"
    cursor = 0
    while True:
        messages, cursor, complete = _events(status, lane=lane, run_id=run_id, start_line=cursor)
        for message in messages:
            print(message, flush=True)
        if complete:
            return 0
        channel = f"spbench-si-{run_id}-{lane}"
        if shutil.which("tmux") and subprocess.run(["tmux", "has-session"], check=False, capture_output=True).returncode == 0:
            subprocess.run(["tmux", "wait-for", channel], check=False)
        else:
            time.sleep(float(os.environ.get("SPBENCH_SI_WATCHER_POLL_SECONDS", "5")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    raise SystemExit(watch(args.control_root, lane=args.lane, run_id=args.run_id))


if __name__ == "__main__":
    main()
