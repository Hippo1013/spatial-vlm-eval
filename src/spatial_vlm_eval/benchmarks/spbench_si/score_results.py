"""Directory-driven, locked SPBench-SI scoring orchestration."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .data import SPBenchSITestContract
from .scorer import AUDIT_SCORER_PROTOCOL, SCORER_PROTOCOL, score_predictions


@dataclass(frozen=True, slots=True)
class ScoreCandidate:
    predictions: Path
    state: str
    reason: str


def discover_predictions(output_root: str | Path) -> list[Path]:
    root = Path(output_root).resolve()
    return sorted(
        path for path in root.rglob("predictions.jsonl")
        if "test_artifacts" not in path.parts and "scores" not in path.parts
    )


def score_state(predictions: str | Path) -> ScoreCandidate:
    path = Path(predictions).resolve()
    main = path.parent / "scores" / SCORER_PROTOCOL
    audit = path.parent / "scores" / AUDIT_SCORER_PROTOCOL
    summary = main / "summary.json"
    gates = main / "publication_gates.json"
    audit_summary = audit / "summary.json"
    if not path.is_file():
        return ScoreCandidate(path, "invalid", "prediction file is missing")
    if summary.is_file() and gates.is_file() and audit_summary.is_file():
        try:
            gate_value = json.loads(gates.read_text(encoding="utf-8"))
            summary_value = json.loads(summary.read_text(encoding="utf-8"))
            audit_value = json.loads(audit_summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ScoreCandidate(path, "retry", f"malformed score artifact: {exc}")
        if (
            gate_value.get("passed") is True
            and summary_value.get("scorer_protocol") == SCORER_PROTOCOL
            and audit_value.get("scorer_protocol") == AUDIT_SCORER_PROTOCOL
        ):
            return ScoreCandidate(path, "complete", "both protocols and publication gates passed")
        return ScoreCandidate(path, "retry", "canonical score artifacts are incomplete")
    if summary.exists() or gates.exists() or audit_summary.exists():
        return ScoreCandidate(path, "retry", "partial score artifacts exist")
    metadata = path.with_suffix(path.suffix + ".metadata.json")
    validation = path.parent / "prediction_validation.json"
    if not metadata.is_file() or not validation.is_file():
        return ScoreCandidate(path, "invalid", "full inference metadata/validator is missing")
    try:
        validation_value = json.loads(validation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ScoreCandidate(path, "invalid", f"invalid full validator: {exc}")
    if validation_value.get("passed") is not True:
        return ScoreCandidate(path, "invalid", "full validator did not pass")
    return ScoreCandidate(path, "new", "ready")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=os.environ.get("SPBENCH_SI_OUTPUT_ROOT"))
    parser.add_argument("--predictions")
    parser.add_argument("--parquet", default=os.environ.get("SPBENCH_SI_PARQUET"))
    parser.add_argument("--images-archive", default=os.environ.get("SPBENCH_SI_IMAGES_ARCHIVE"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_root and not args.predictions:
        raise ValueError("Set SPBENCH_SI_OUTPUT_ROOT or pass --predictions")
    output_root = Path(args.output_root).resolve() if args.output_root else Path(args.predictions).resolve().parent
    paths = [Path(args.predictions).resolve()] if args.predictions else discover_predictions(output_root)
    candidates = [score_state(path) for path in paths]
    invalid = [candidate for candidate in candidates if candidate.state == "invalid"]
    pending = [candidate for candidate in candidates if candidate.state in {"new", "retry"}]
    if args.check:
        print(json.dumps({
            "passed": not invalid,
            "num_candidates": len(candidates),
            "num_pending": len(pending),
            "num_complete": sum(item.state == "complete" for item in candidates),
            "candidates": [
                {"predictions": str(item.predictions), "state": item.state, "reason": item.reason}
                for item in candidates
            ],
        }, ensure_ascii=False))
        if invalid:
            raise SystemExit(1)
        return
    if invalid:
        raise ValueError("Invalid SPBench-SI prediction candidates: " + "; ".join(
            f"{item.predictions}: {item.reason}" for item in invalid
        ))
    if args.dry_run:
        for candidate in pending:
            print(f"score\t{candidate.predictions}")
        return
    if not args.parquet or not args.images_archive:
        raise ValueError("Set SPBENCH_SI_PARQUET and SPBENCH_SI_IMAGES_ARCHIVE")
    contract = SPBenchSITestContract(args.parquet, args.images_archive)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".spbench-si-scoring.lock"
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
            print(f"[spbench-si-score] scoring {candidate.predictions}")
            score_predictions(candidate.predictions, contract)
            if score_state(candidate.predictions).state != "complete":
                raise RuntimeError(f"Scoring did not complete both protocols: {candidate.predictions}")


if __name__ == "__main__":
    main()
