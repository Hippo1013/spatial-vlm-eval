"""Validate the minimal two-field CV-Bench prediction schema."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .data import CVBenchTestContract, DATASET_REVISION, OFFICIAL_TEST_SIZE

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
    contract: CVBenchTestContract,
    *,
    prediction_path: str | Path,
    allow_subset: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    indices: list[int] = []
    for position, row in enumerate(rows):
        fields = set(row)
        missing = sorted(REQUIRED_FIELDS - fields)
        extra = sorted(fields - REQUIRED_FIELDS)
        if missing:
            errors.append(f"row_position={position}: missing keys {missing}")
        if extra:
            errors.append(f"row_position={position}: unexpected keys {extra}")
        if missing:
            continue
        value = row["index"]
        if isinstance(value, bool):
            errors.append(f"row_position={position}: invalid boolean index={value!r}")
            continue
        try:
            index = int(value)
        except (TypeError, ValueError):
            errors.append(f"row_position={position}: invalid index={value!r}")
            continue
        if str(value).strip() != str(index):
            errors.append(f"row_position={position}: index is not a canonical integer: {value!r}")
            continue
        if not 0 <= index < len(contract):
            errors.append(f"row_position={position}: index {index} outside [0,{len(contract)})")
            continue
        indices.append(index)
        if not isinstance(row["raw_prediction"], str):
            errors.append(f"index={index}: raw_prediction must be a string")
        elif not row["raw_prediction"].strip():
            warnings.append(f"index={index}: empty raw_prediction will score zero")

    duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate indices: {duplicates[:20]}")
    expected = set(range(len(contract)))
    actual = set(indices)
    if not allow_subset:
        missing_indices = sorted(expected - actual)
        extra_indices = sorted(actual - expected)
        if len(rows) != len(contract):
            errors.append(f"expected {len(contract)} rows, got {len(rows)}")
        if missing_indices:
            errors.append(f"missing indices: {missing_indices[:20]} (total={len(missing_indices)})")
        if extra_indices:
            errors.append(f"extra indices: {extra_indices[:20]} (total={len(extra_indices)})")
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


def validate_predictions(
    prediction_path: str | Path,
    dataset_root: str | Path,
    *,
    allow_subset: bool = False,
    verify_files: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = CVBenchTestContract(dataset_root, verify_files=verify_files)
    rows = read_jsonl(prediction_path)
    return rows, validate_prediction_rows(
        rows,
        contract,
        prediction_path=prediction_path,
        allow_subset=allow_subset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset-root", default=os.environ.get("CVBENCH_DATASET_ROOT"))
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> None:
    from ...models.common.runtime import atomic_write_json

    args = parse_args()
    if not args.dataset_root:
        raise ValueError("Set CVBENCH_DATASET_ROOT or pass --dataset-root")
    _, report = validate_predictions(
        args.predictions,
        args.dataset_root,
        allow_subset=args.allow_subset,
    )
    if args.report:
        atomic_write_json(Path(args.report).resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
