"""Directory-driven serial scoring for complete CV-Bench prediction tracks."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import CVBenchTestContract, OFFICIAL_TEST_SIZE
from .scorer import SCORER_PROTOCOL, score_predictions


@dataclass(frozen=True, slots=True)
class Candidate:
    predictions: Path
    state: str
    reason: str


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def score_state(predictions: str | Path) -> Candidate:
    path = Path(predictions).resolve()
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata = _load_json(metadata_path)
    if metadata is None:
        return Candidate(path, "invalid", "missing or malformed inference metadata")
    dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
    if not metadata.get("publishable_inference") or dataset.get("official_test_size") != OFFICIAL_TEST_SIZE:
        return Candidate(path, "subset", "inference metadata is not a full publishable split")
    score_dir = path.parent / "scores" / SCORER_PROTOCOL
    gates = _load_json(score_dir / "publication_gates.json")
    summary = _load_json(score_dir / "summary.json")
    if (
        gates
        and gates.get("passed")
        and all((gates.get("gates") or {}).values())
        and summary
        and summary.get("scorer_protocol") == SCORER_PROTOCOL
        and summary.get("num_scored_rows") == OFFICIAL_TEST_SIZE
    ):
        return Candidate(path, "complete", "publication gates passed")
    if score_dir.exists():
        return Candidate(path, "retry", "canonical score artifacts are incomplete or invalid")
    return Candidate(path, "new", "unscored full prediction track")


def discover_candidates(output_root: str | Path) -> list[Candidate]:
    root = Path(output_root).resolve()
    candidates: list[Candidate] = []
    for path in sorted(root.rglob("predictions.jsonl")):
        relative_parts = path.relative_to(root).parts
        if "test_runs" in relative_parts or "shards" in relative_parts or "scores" in relative_parts:
            continue
        candidates.append(score_state(path))
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--dataset-root", default=os.environ.get("CVBENCH_DATASET_ROOT"))
    parser.add_argument("--output-root", default=os.environ.get("CVBENCH_OUTPUT_ROOT"))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predictions:
        candidates = [score_state(args.predictions)]
        output_root = Path(args.output_root).resolve() if args.output_root else candidates[0].predictions.parent
    else:
        if not args.output_root:
            raise ValueError("Set CVBENCH_OUTPUT_ROOT or pass --output-root")
        output_root = Path(args.output_root).resolve()
        candidates = discover_candidates(output_root)
    if args.list or args.status:
        for candidate in candidates:
            print(f"{candidate.state}\t{candidate.predictions}\t{candidate.reason}")
        return
    if args.predictions and candidates[0].state in {"invalid", "subset"}:
        raise ValueError(
            f"Selected prediction is not scoreable: {candidates[0].state}: {candidates[0].reason}"
        )
    pending = [candidate for candidate in candidates if candidate.state in {"new", "retry"}]
    if args.check:
        invalid = [candidate for candidate in candidates if candidate.state in {"invalid", "subset"}]
        print(
            json.dumps(
                {
                    "passed": not invalid,
                    "num_candidates": len(candidates),
                    "num_pending": len(pending),
                    "num_complete": sum(candidate.state == "complete" for candidate in candidates),
                    "invalid": [str(candidate.predictions) for candidate in invalid],
                },
                ensure_ascii=False,
            )
        )
        if invalid:
            raise SystemExit(1)
        return
    if args.dry_run:
        for candidate in pending:
            print(f"score\t{candidate.predictions}")
        return
    if not args.dataset_root:
        raise ValueError("Set CVBENCH_DATASET_ROOT or pass --dataset-root")
    contract = CVBenchTestContract(args.dataset_root, verify_files=True)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".cvbench-scoring.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = [score_state(candidate.predictions) for candidate in pending]
        for candidate in frozen:
            if candidate.state == "complete":
                continue
            if candidate.state not in {"new", "retry"}:
                raise RuntimeError(
                    f"Candidate changed state before scoring: {candidate.predictions} -> {candidate.state}"
                )
            print(f"[cv-bench-score] scoring {candidate.predictions}")
            score_predictions(candidate.predictions, contract)
            completed = score_state(candidate.predictions)
            if completed.state != "complete":
                raise RuntimeError(
                    f"Scoring did not produce complete publication gates: {candidate.predictions}"
                )


if __name__ == "__main__":
    main()
