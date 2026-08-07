"""Validate the strict two-field SPBench-SI prediction schema."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from ...models.common.runtime import atomic_write_json
from .data import DATASET_REVISION, OFFICIAL_TEST_SIZE, SPBenchSITestContract

REQUIRED_FIELDS = frozenset({"index", "raw_prediction"})


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
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
    contract: SPBenchSITestContract,
    *,
    prediction_path: str | Path,
    allow_subset: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    indices: list[int] = []
    for position, row in enumerate(rows):
        missing = sorted(REQUIRED_FIELDS - set(row))
        extra = sorted(set(row) - REQUIRED_FIELDS)
        if missing:
            errors.append(f"row_position={position}: missing keys {missing}")
        if extra:
            errors.append(f"row_position={position}: unexpected keys {extra}")
        if missing:
            continue
        value = row["index"]
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"row_position={position}: index must be an integer, got {value!r}")
            continue
        if not 0 <= value < len(contract):
            errors.append(f"row_position={position}: index {value} outside [0,{len(contract)})")
            continue
        indices.append(value)
        prediction = row["raw_prediction"]
        if not isinstance(prediction, str):
            errors.append(f"index={value}: raw_prediction must be a string")
        elif not prediction.strip():
            warnings.append(f"index={value}: empty raw_prediction will score zero")
    duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate indices: {duplicates[:20]}")
    actual = set(indices)
    expected = set(range(len(contract)))
    if not allow_subset:
        if len(rows) != len(contract):
            errors.append(f"expected {len(contract)} rows, got {len(rows)}")
        missing_indices = sorted(expected - actual)
        if missing_indices:
            errors.append(f"missing indices: {missing_indices[:20]} (total={len(missing_indices)})")
    return {
        "passed": not errors,
        "predictions": str(Path(prediction_path).resolve()),
        "dataset_revision": DATASET_REVISION,
        "dataset_fingerprint": contract.dataset_fingerprint,
        "official_test_size": OFFICIAL_TEST_SIZE,
        "loaded_size": len(contract),
        "num_prediction_rows": len(rows),
        "num_unique_indices": len(actual),
        "allow_subset": bool(allow_subset),
        "errors": errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--parquet", default=os.environ.get("SPBENCH_SI_PARQUET"))
    parser.add_argument("--images-archive", default=os.environ.get("SPBENCH_SI_IMAGES_ARCHIVE"))
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.parquet or not args.images_archive:
        raise ValueError("Set SPBENCH_SI_PARQUET and SPBENCH_SI_IMAGES_ARCHIVE")
    contract = SPBenchSITestContract(args.parquet, args.images_archive)
    rows = read_jsonl(args.predictions)
    report = validate_prediction_rows(
        rows, contract, prediction_path=args.predictions, allow_subset=args.allow_subset
    )
    report_path = Path(args.report).resolve() if args.report else Path(args.predictions).resolve().parent / "prediction_validation.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
