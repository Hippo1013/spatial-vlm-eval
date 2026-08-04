#!/usr/bin/env bash
# Read-only live display of CV-Bench journal predictions across serial profiles.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cv_bench/watch_live_predictions.sh
  bash scripts/cv_bench/watch_live_predictions.sh --from-start

Follow the active CV-Bench serial profile and print each newly appended journal
success or failure. The watcher is read-only; Ctrl-C stops only the watcher.

Options:
  --from-start   Replay existing journal rows for the currently active profile.
  --help, -h     Show this help.
EOF
}

from_start=0
while (( $# > 0 )); do
  case "$1" in
    --from-start) from_start=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[cv-bench-watch] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

poll_seconds="${CVBENCH_WATCH_POLL_SECONDS:-1}"
if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[cv-bench-watch] CVBENCH_WATCH_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ -z "${CVBENCH_OUTPUT_ROOT:-}" ]]; then
  echo "[cv-bench-watch] set CVBENCH_OUTPUT_ROOT" >&2
  exit 2
fi

exec "${CVBENCH_PYTHON}" -u - \
  "${CVBENCH_OUTPUT_ROOT}" "${poll_seconds}" "${from_start}" <<'PY'
import json
import sys
import time
from pathlib import Path


output_root = Path(sys.argv[1]).resolve()
poll_seconds = int(sys.argv[2])
from_start = sys.argv[3] == "1"
control_root = output_root / "_serial_full"
status_path = control_root / "status.tsv"
runs_root = output_root / "runs"


def status_rows():
    if not status_path.is_file():
        return []
    rows = []
    for line in status_path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) >= 4:
            rows.append(tuple(fields[:4]))
    return rows


def active_profile(rows):
    active = None
    for _, profile, state, _ in rows:
        if state == "START":
            active = profile
        elif state in {"PASS", "FAIL"} and profile == active:
            active = None
    return active


def journal_paths(profile):
    profile_root = runs_root / profile
    if not profile_root.is_dir():
        return []
    return sorted(
        path
        for path in profile_root.rglob("*.journal.jsonl")
        if "test_runs" not in path.parts and "failed_attempts" not in path.parts
    )


def batch_complete(rows):
    if not rows:
        return False
    run_id = rows[-1][0]
    log_path = control_root / "logs" / f"{run_id}.controller.log"
    return log_path.is_file() and "[cv-bench-full-serial] COMPLETE" in log_path.read_text(
        encoding="utf-8", errors="replace"
    )


def print_event(profile, event):
    timestamp = event.get("timestamp", "")
    index = event.get("index", "?")
    status = event.get("status")
    if status == "success":
        print(f"\n[{timestamp}] {profile} index={index}")
        print(str(event.get("prediction", "")))
    elif status == "failure":
        error = json.dumps(event.get("error", {}), ensure_ascii=False, sort_keys=True)
        print(f"\n[{timestamp}] {profile} index={index} FAILED")
        print(error)
    else:
        return
    print("-" * 72, flush=True)


positions = {}
rows = status_rows()
current = active_profile(rows)
if current and not from_start:
    for path in journal_paths(current):
        try:
            positions[path] = path.stat().st_size
        except OSError:
            pass

if current:
    print(f"===== active model: {current} =====", flush=True)
else:
    print("===== waiting for an active CV-Bench serial profile =====", flush=True)

try:
    while True:
        rows = status_rows()
        next_profile = active_profile(rows)
        if next_profile != current:
            current = next_profile
            if current:
                print(f"\n===== active model: {current} =====", flush=True)

        if current:
            for path in journal_paths(current):
                try:
                    size = path.stat().st_size
                    offset = positions.get(path, 0)
                    if size < offset:
                        offset = 0
                    with path.open("rb") as handle:
                        handle.seek(offset)
                        chunk = handle.read()
                    newline = chunk.rfind(b"\n")
                    if newline < 0:
                        positions[path] = offset
                        continue
                    complete = chunk[: newline + 1]
                    positions[path] = offset + newline + 1
                    for raw_line in complete.splitlines():
                        if raw_line.strip():
                            print_event(current, json.loads(raw_line.decode("utf-8")))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue

        if batch_complete(rows):
            print("\n===== CV-Bench serial inference COMPLETE =====", flush=True)
            raise SystemExit(0)
        time.sleep(poll_seconds)
except KeyboardInterrupt:
    print("\n[cv-bench-watch] watcher stopped; inference was not interrupted.", flush=True)
PY
