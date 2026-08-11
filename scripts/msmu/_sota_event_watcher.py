#!/usr/bin/env python3
"""Pipe-driven MSMU SOTA lane watcher; it never polls or invokes a model."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


VISIBLE_STATES = {"PASS", "FAIL", "COMPLETE", "FAULT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--status", type=Path, required=True)
    return parser.parse_args()


def latest_visible_event(path: Path, lane: str) -> tuple[str, ...] | None:
    if not path.is_file():
        return None
    latest: tuple[str, ...] | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("lane") != lane or row.get("state") not in VISIBLE_STATES:
                continue
            latest = (
                row.get("timestamp", ""),
                row.get("state", ""),
                row.get("profile", ""),
                row.get("phase", ""),
                row.get("detail", ""),
            )
    return latest


def main() -> int:
    args = parse_args()
    previous: tuple[str, ...] | None = None
    with os.fdopen(args.fd, "rb", buffering=0) as pipe:
        while pipe.read(1):
            event = latest_visible_event(args.status, args.lane)
            if event is None or event == previous:
                continue
            _timestamp, state, profile, phase, detail = event
            print(
                f"[msmu-sota-watcher {args.lane}] {state} "
                f"profile={profile} phase={phase} detail={detail}",
                flush=True,
            )
            previous = event
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
