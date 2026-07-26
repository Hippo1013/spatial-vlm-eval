"""Validate prediction JSONL files against the exact MSMU-Bench test split.

An empty prediction is intentionally a warning rather than a hard error: the
row is structurally valid and the scorer can assign it zero. All provenance
fields and indices remain hard validation requirements.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data import clean_question, load_arrow_split, task_family


REQUIRED_FIELDS = {
    "index",
    "raw_type",
    "task_family",
    "question",
    "reference",
    "prediction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def validate_prediction_rows(
    rows: list[dict[str, Any]],
    test: Any,
    *,
    prediction_path: Path,
    dataset_root: str | Path,
    allow_subset: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable validation report for already loaded rows."""

    expected_size = len(test)
    errors: list[str] = []
    warnings: list[str] = []

    for position, row in enumerate(rows):
        row_fields = set(row)
        missing = sorted(REQUIRED_FIELDS - row_fields)
        extra = sorted(row_fields - REQUIRED_FIELDS)
        if missing:
            errors.append(f"row_position={position}: missing keys {missing}")
        if extra:
            errors.append(f"row_position={position}: unexpected keys {extra}")
        if missing:
            continue
        try:
            index = int(row["index"])
        except (TypeError, ValueError):
            errors.append(f"row_position={position}: invalid index={row.get('index')!r}")
            continue
        if not 0 <= index < expected_size:
            errors.append(f"row_position={position}: index {index} outside [0,{expected_size})")
            continue
        source = test[index]
        expected_raw_type = str(source.get("type") or "")
        expected_question = clean_question(source["conversations"]["value"][0])
        expected_reference = str(source["conversations"]["value"][1]).strip()
        expected_family = task_family(expected_raw_type)
        checks = {
            "raw_type": expected_raw_type,
            "task_family": expected_family,
            "question": expected_question,
            "reference": expected_reference,
        }
        for key, expected in checks.items():
            if str(row[key]) != expected:
                errors.append(
                    f"index={index}: {key} mismatch; got={row[key]!r}, expected={expected!r}"
                )
        if not str(row["prediction"]).strip():
            warnings.append(f"index={index}: empty prediction (will score zero or parse failure)")

    indices = [int(row["index"]) for row in rows if "index" in row and str(row["index"]).lstrip("-").isdigit()]
    duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate indices: {duplicates[:20]}")
    if not allow_subset:
        expected_indices = set(range(expected_size))
        actual_indices = set(indices)
        missing_indices = sorted(expected_indices - actual_indices)
        extra_indices = sorted(actual_indices - expected_indices)
        if len(rows) != expected_size:
            errors.append(f"expected {expected_size} rows, got {len(rows)}")
        if missing_indices:
            errors.append(f"missing indices: {missing_indices[:20]} (total={len(missing_indices)})")
        if extra_indices:
            errors.append(f"extra indices: {extra_indices[:20]} (total={len(extra_indices)})")

    type_counts = Counter(str(row.get("raw_type", "")) for row in rows)
    return {
        "passed": not errors,
        "predictions": str(prediction_path),
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": "test",
        "official_test_size": expected_size,
        "num_prediction_rows": len(rows),
        "allow_subset": bool(allow_subset),
        "num_unique_indices": len(set(indices)),
        "raw_type_counts": dict(sorted(type_counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def validate_predictions(
    prediction_path: str | Path,
    dataset_root: str | Path,
    *,
    allow_subset: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate predictions, returning both rows and the report."""

    resolved_predictions = Path(prediction_path).resolve()
    rows = read_jsonl(resolved_predictions)
    test = load_arrow_split(dataset_root, "test")
    report = validate_prediction_rows(
        rows,
        test,
        prediction_path=resolved_predictions,
        dataset_root=dataset_root,
        allow_subset=allow_subset,
    )
    return rows, report


def write_validation_report(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _, report = validate_predictions(
        args.predictions,
        args.dataset_root,
        allow_subset=args.allow_subset,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        write_validation_report(args.report, report)
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
