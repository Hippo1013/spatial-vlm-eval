"""Robust deterministic numeric scorer for the locked Q-Spatial benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

from ...models.common.runtime import atomic_write_json, atomic_write_jsonl, utc_now
from .data import (
    DATASET_REVISION,
    EXPECTED_SPLIT_TYPE_COUNTS,
    OFFICIAL_TEST_SIZE,
    QSpatialTestContract,
    SCANNET_CANONICAL_TYPES,
)
from .prediction_validation import read_jsonl, validate_prediction_rows

SCORER_PROTOCOL = (
    "q_spatial_robust_numeric_v1_standard_prompt_tag_first_unique_fallback_"
    "paper_inclusive_ratio"
)
RESULT_KIND = "q_spatial_official_formula_robust_numeric_parser_internal_score"

_TAG_TRACE_RE = re.compile(r"(?i)\b(?:scalar|distance_unit)\b")
_SCALAR_TAG_RE = re.compile(r"(?i)\\?scalar\s*\{([^{}]*)\}")
_UNIT_TAG_RE = re.compile(r"(?i)\\?distance_unit\s*\{([^{}]*)\}")
_DECIMAL_TEXT_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")
_NUMBER_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_FALLBACK_PAIR_RE = re.compile(
    r"(?i)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-z]+)"
)

_UNIT_MULTIPLIERS = {
    "m": Decimal("100"),
    "meter": Decimal("100"),
    "meters": Decimal("100"),
    "metre": Decimal("100"),
    "metres": Decimal("100"),
    "cm": Decimal("1"),
    "centimeter": Decimal("1"),
    "centimeters": Decimal("1"),
    "centimetre": Decimal("1"),
    "centimetres": Decimal("1"),
    "mm": Decimal("0.1"),
    "millimeter": Decimal("0.1"),
    "millimeters": Decimal("0.1"),
    "millimetre": Decimal("0.1"),
    "millimetres": Decimal("0.1"),
    "ft": Decimal("30.48"),
    "foot": Decimal("30.48"),
    "feet": Decimal("30.48"),
    "in": Decimal("2.54"),
    "inch": Decimal("2.54"),
    "inches": Decimal("2.54"),
}


@dataclass(frozen=True, slots=True)
class ParsedMeasurement:
    value: Decimal | None
    unit: str | None
    centimeters: Decimal | None
    status: str
    mode: str


def _measurement(value_text: str, unit_text: str, *, mode: str) -> ParsedMeasurement:
    stripped_value = value_text.strip()
    if not _DECIMAL_TEXT_RE.fullmatch(stripped_value):
        return ParsedMeasurement(None, None, None, f"{mode}_invalid_scalar", mode)
    try:
        value = Decimal(stripped_value)
    except InvalidOperation:
        return ParsedMeasurement(None, None, None, f"{mode}_invalid_scalar", mode)
    if not value.is_finite() or value <= 0:
        return ParsedMeasurement(None, None, None, f"{mode}_non_positive_scalar", mode)
    unit = re.sub(r"\s+", " ", unit_text.strip().casefold())
    multiplier = _UNIT_MULTIPLIERS.get(unit)
    if multiplier is None:
        return ParsedMeasurement(value, unit or None, None, f"{mode}_unknown_unit", mode)
    return ParsedMeasurement(value, unit, value * multiplier, f"{mode}_valid", mode)


def _fallback_region(text: str) -> tuple[str, str]:
    markers = (
        ("final_answer", re.compile(r"(?i)\bfinal\s+answer\b\s*(?:is\s*)?(?::|=|-)?")),
        ("answer", re.compile(r"(?i)\banswer\s*:")),
        ("in_conclusion", re.compile(r"(?i)\bin\s+conclusion\b\s*[,;:]?")),
    )
    for name, pattern in markers:
        matches = list(pattern.finditer(text))
        if matches:
            return text[matches[-1].end() :].strip(), name
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1] if lines else ""), "last_nonempty_line"


def parse_measurement(raw_prediction: Any) -> ParsedMeasurement:
    text = str(raw_prediction)
    if not text.strip():
        return ParsedMeasurement(None, None, None, "empty", "none")
    if _TAG_TRACE_RE.search(text):
        scalars = _SCALAR_TAG_RE.findall(text)
        units = _UNIT_TAG_RE.findall(text)
        if len(scalars) != 1 or len(units) != 1:
            return ParsedMeasurement(None, None, None, "tag_malformed_or_non_unique", "tag")
        return _measurement(scalars[0], units[0], mode="tag")

    region, region_name = _fallback_region(text)
    if not region:
        return ParsedMeasurement(None, None, None, f"fallback_{region_name}_empty", "fallback")
    numbers = _NUMBER_TOKEN_RE.findall(region)
    if len(numbers) != 1:
        return ParsedMeasurement(
            None,
            None,
            None,
            f"fallback_{region_name}_{'missing' if not numbers else 'multiple'}_numbers",
            "fallback",
        )
    pairs = _FALLBACK_PAIR_RE.findall(region)
    if len(pairs) != 1:
        return ParsedMeasurement(
            None,
            None,
            None,
            f"fallback_{region_name}_{'missing' if not pairs else 'multiple'}_pairs",
            "fallback",
        )
    number_text, unit_text = pairs[0]
    if number_text != numbers[0]:
        return ParsedMeasurement(None, None, None, f"fallback_{region_name}_conflict", "fallback")
    if any(character in number_text for character in "+-eE"):
        return ParsedMeasurement(None, None, None, f"fallback_{region_name}_invalid_scalar", "fallback")
    parsed = _measurement(number_text, unit_text, mode="fallback")
    return ParsedMeasurement(
        parsed.value,
        parsed.unit,
        parsed.centimeters,
        f"fallback_{region_name}_{parsed.status.removeprefix('fallback_')}",
        "fallback",
    )


def parse_legacy_notebook(raw_prediction: Any) -> ParsedMeasurement:
    """Conservatively emulate the official notebook without aborting a batch."""

    text = str(raw_prediction)
    scalar_tags = _SCALAR_TAG_RE.findall(text)
    unit_tags = _UNIT_TAG_RE.findall(text)
    if not scalar_tags or not unit_tags:
        return ParsedMeasurement(None, None, None, "audit_missing_tag", "audit")
    scalar_values = re.findall(r"\d+\.?\d*", scalar_tags[-1])
    if not scalar_values:
        return ParsedMeasurement(None, None, None, "audit_invalid_scalar", "audit")
    try:
        values = [Decimal(value) for value in scalar_values]
        with localcontext() as context:
            context.prec = 40
            mean = sum(values, Decimal("0")) / Decimal(len(values))
    except (InvalidOperation, ZeroDivisionError):
        return ParsedMeasurement(None, None, None, "audit_invalid_scalar", "audit")
    if not mean.is_finite() or mean <= 0:
        return ParsedMeasurement(None, None, None, "audit_non_positive_scalar", "audit")
    unit = unit_tags[-1].strip().casefold()
    multiplier = _UNIT_MULTIPLIERS.get(unit, Decimal("1"))
    status = "audit_valid" if unit in _UNIT_MULTIPLIERS else "audit_unknown_unit_as_centimeter"
    return ParsedMeasurement(mean, unit or None, mean * multiplier, status, "audit")


def _ground_truth_centimeters(value: str, unit: str) -> Decimal:
    try:
        numeric = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid Q-Spatial ground truth {value!r}") from exc
    multiplier = _UNIT_MULTIPLIERS.get(str(unit).casefold())
    if multiplier is None or numeric <= 0:
        raise ValueError(f"Invalid Q-Spatial ground truth unit/value: {value!r} {unit!r}")
    return numeric * multiplier


def _ratio(predicted: Decimal | None, truth: Decimal) -> Decimal | None:
    if predicted is None or predicted <= 0 or truth <= 0:
        return None
    with localcontext() as context:
        context.prec = 40
        return max(predicted / truth, truth / predicted)


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot compute Q-Spatial accuracy for an empty group")
    correct = sum(bool(row[key]) for row in rows)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}


def score_rows(
    prediction_rows: list[dict[str, Any]],
    contract: QSpatialTestContract,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for prediction in sorted(prediction_rows, key=lambda row: int(row["index"])):
        index = int(prediction["index"])
        source = contract.scoring_row(index)
        truth_cm = _ground_truth_centimeters(source["answer_value"], source["answer_unit"])
        parsed = parse_measurement(prediction["raw_prediction"])
        ratio = _ratio(parsed.centimeters, truth_cm)
        audit = parse_legacy_notebook(prediction["raw_prediction"])
        audit_ratio = _ratio(audit.centimeters, truth_cm)
        main_125 = ratio is not None and ratio <= Decimal("1.25")
        main_2 = ratio is not None and ratio <= Decimal("2")
        audit_125 = audit_ratio is not None and audit_ratio < Decimal("1.25")
        audit_2 = audit_ratio is not None and audit_ratio < Decimal("2")
        scored.append(
            {
                "index": index,
                "raw_prediction": str(prediction["raw_prediction"]),
                "parse_mode": parsed.mode,
                "parse_status": parsed.status,
                "parsed_value": _decimal_text(parsed.value),
                "parsed_unit": parsed.unit,
                "predicted_centimeters": _decimal_text(parsed.centimeters),
                "ground_truth_value": source["answer_value"],
                "ground_truth_unit": source["answer_unit"],
                "ground_truth_centimeters": _decimal_text(truth_cm),
                "ratio": _decimal_text(ratio),
                "success_delta_le_1_25": main_125,
                "success_delta_le_2": main_2,
                "legacy_parse_status": audit.status,
                "legacy_parsed_value": _decimal_text(audit.value),
                "legacy_parsed_unit": audit.unit,
                "legacy_predicted_centimeters": _decimal_text(audit.centimeters),
                "legacy_ratio": _decimal_text(audit_ratio),
                "legacy_success_delta_lt_1_25": audit_125,
                "legacy_success_delta_lt_2": audit_2,
                "main_vs_legacy_difference": main_125 != audit_125 or main_2 != audit_2,
                "split": source["split"],
                "raw_type": source["raw_type"],
                "canonical_type": source["canonical_type"],
            }
        )

    split_rows = {
        split: [row for row in scored if row["split"] == split]
        for split in ("QSpatial_scannet", "QSpatial_plus")
    }
    expected_sizes = {"QSpatial_scannet": 170, "QSpatial_plus": 101}
    if {key: len(value) for key, value in split_rows.items()} != expected_sizes:
        raise ValueError("Q-Spatial scored split counts differ from the locked 170/101 contract")
    split_metrics = {
        split: {
            "delta_le_1_25": _metric(rows, "success_delta_le_1_25"),
            "delta_le_2": _metric(rows, "success_delta_le_2"),
            "legacy_delta_lt_1_25": _metric(rows, "legacy_success_delta_lt_1_25"),
            "legacy_delta_lt_2": _metric(rows, "legacy_success_delta_lt_2"),
        }
        for split, rows in split_rows.items()
    }
    scannet_type_metrics: dict[str, dict[str, Any]] = {}
    for canonical_type in SCANNET_CANONICAL_TYPES:
        rows = [
            row
            for row in split_rows["QSpatial_scannet"]
            if row["canonical_type"] == canonical_type
        ]
        expected = EXPECTED_SPLIT_TYPE_COUNTS["QSpatial_scannet"][canonical_type]
        if len(rows) != expected:
            raise ValueError(f"Q-Spatial ScanNet type count mismatch for {canonical_type}")
        scannet_type_metrics[canonical_type] = {
            "delta_le_1_25": _metric(rows, "success_delta_le_1_25"),
            "delta_le_2": _metric(rows, "success_delta_le_2"),
        }
    scan_125 = split_metrics["QSpatial_scannet"]["delta_le_1_25"]["accuracy"]
    plus_125 = split_metrics["QSpatial_plus"]["delta_le_1_25"]["accuracy"]
    scan_2 = split_metrics["QSpatial_scannet"]["delta_le_2"]["accuracy"]
    plus_2 = split_metrics["QSpatial_plus"]["delta_le_2"]["accuracy"]
    audit_scan_125 = split_metrics["QSpatial_scannet"]["legacy_delta_lt_1_25"]["accuracy"]
    audit_plus_125 = split_metrics["QSpatial_plus"]["legacy_delta_lt_1_25"]["accuracy"]
    audit_scan_2 = split_metrics["QSpatial_scannet"]["legacy_delta_lt_2"]["accuracy"]
    audit_plus_2 = split_metrics["QSpatial_plus"]["legacy_delta_lt_2"]["accuracy"]
    aggregate = {
        "split_metrics": split_metrics,
        "scannet_type_metrics": scannet_type_metrics,
        "metrics": {
            "overall_delta_le_1_25": (scan_125 + plus_125) / 2,
            "overall_delta_le_2": (scan_2 + plus_2) / 2,
            "micro_delta_le_1_25_audit_only": _metric(scored, "success_delta_le_1_25")["accuracy"],
            "micro_delta_le_2_audit_only": _metric(scored, "success_delta_le_2")["accuracy"],
            "legacy_notebook_overall_delta_lt_1_25": (audit_scan_125 + audit_plus_125) / 2,
            "legacy_notebook_overall_delta_lt_2": (audit_scan_2 + audit_plus_2) / 2,
        },
        "parse_status_counts": dict(sorted(Counter(row["parse_status"] for row in scored).items())),
        "legacy_parse_status_counts": dict(
            sorted(Counter(row["legacy_parse_status"] for row in scored).items())
        ),
        "num_main_vs_legacy_differences": sum(
            bool(row["main_vs_legacy_difference"]) for row in scored
        ),
    }
    return scored, aggregate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_and_validate_metadata(
    prediction_path: Path,
    contract: QSpatialTestContract,
    *,
    required: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    metadata_path = prediction_path.with_suffix(prediction_path.suffix + ".metadata.json")
    errors: list[str] = []
    if not metadata_path.is_file():
        if required:
            errors.append(f"missing inference metadata: {metadata_path}")
        return None, errors
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
    model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    checks = {
        "metadata.output": (str(Path(metadata.get("output", "")).resolve()), str(prediction_path)),
        "metadata.output_sha256": (metadata.get("output_sha256"), _sha256(prediction_path)),
        "metadata.scorer_protocol": (metadata.get("scorer_protocol"), SCORER_PROTOCOL),
        "metadata.dataset.fingerprint": (dataset.get("fingerprint"), contract.dataset_fingerprint),
        "metadata.dataset.official_test_size": (dataset.get("official_test_size"), OFFICIAL_TEST_SIZE),
        "metadata.dataset.revision": (dataset.get("revision"), DATASET_REVISION),
        "metadata.publishable_inference": (metadata.get("publishable_inference"), True),
        "metadata.model.inference_protocol": (
            model.get("inference_protocol"),
            metadata.get("inference_protocol"),
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{label} mismatch: got={actual!r}, expected={expected!r}")
    return metadata, errors


def score_predictions(
    prediction_path: str | Path,
    contract: QSpatialTestContract,
    *,
    output_dir: str | Path | None = None,
    require_metadata: bool = True,
) -> dict[str, Any]:
    predictions = Path(prediction_path).resolve()
    rows = read_jsonl(predictions)
    validation = validate_prediction_rows(
        rows, contract, prediction_path=predictions, allow_subset=False
    )
    metadata, metadata_errors = _load_and_validate_metadata(
        predictions, contract, required=require_metadata
    )
    if metadata_errors:
        validation["errors"].extend(metadata_errors)
        validation["passed"] = False
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else predictions.parent / "scores" / SCORER_PROTOCOL
    )
    destination.mkdir(parents=True, exist_ok=True)
    validation_path = destination / "prediction_validation.json"
    atomic_write_json(validation_path, validation)
    if not validation["passed"]:
        raise ValueError("Q-Spatial full prediction validation failed; see prediction_validation.json")
    scored, aggregate = score_rows(rows, contract)
    scored_path = destination / "scored_rows.jsonl"
    atomic_write_jsonl(scored_path, scored)
    inference_model = metadata.get("model", {}) if metadata else {}
    summary = {
        "schema_version": 1,
        "result_kind": RESULT_KIND,
        "scorer_protocol": SCORER_PROTOCOL,
        "dataset": {
            "repository": "andrewliao11/Q-Spatial-Bench",
            "revision": DATASET_REVISION,
            "fingerprint": contract.dataset_fingerprint,
            "official_test_size": OFFICIAL_TEST_SIZE,
        },
        "inference": {
            "profile": inference_model.get("profile"),
            "model": inference_model.get("model"),
            "model_revision": inference_model.get("model_revision"),
            "input_profile": inference_model.get("input_profile"),
            "comparison_group": inference_model.get("comparison_group"),
            "inference_protocol": inference_model.get("inference_protocol"),
            "backend": inference_model.get("backend"),
            "decoding": inference_model.get("decoding"),
            "seed_strategy": inference_model.get("seed_strategy"),
        },
        "num_scored_rows": len(scored),
        **aggregate,
        "artifacts": {
            "predictions": str(predictions),
            "predictions_sha256": _sha256(predictions),
            "validation": str(validation_path),
            "scored_rows": str(scored_path),
            "scored_rows_sha256": _sha256(scored_path),
        },
        "generated_at": utc_now(),
    }
    summary_path = destination / "summary.json"
    atomic_write_json(summary_path, summary)
    gates = {
        "schema_version": 1,
        "scorer_protocol": SCORER_PROTOCOL,
        "passed": True,
        "gates": {
            "full_prediction_validation": True,
            "locked_dataset_identity": True,
            "full_scored_rows": len(scored) == OFFICIAL_TEST_SIZE,
            "complete_split_groups": set(aggregate["split_metrics"])
            == {"QSpatial_scannet", "QSpatial_plus"},
            "complete_scannet_types": set(aggregate["scannet_type_metrics"])
            == set(SCANNET_CANONICAL_TYPES),
            "protocol_consistency": summary["scorer_protocol"] == SCORER_PROTOCOL,
            "inference_metadata_consistency": metadata is not None or not require_metadata,
        },
        "summary": str(summary_path),
        "generated_at": utc_now(),
    }
    gates["passed"] = all(gates["gates"].values())
    atomic_write_json(destination / "publication_gates.json", gates)
    if not gates["passed"]:
        raise RuntimeError("Q-Spatial publication gates failed")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--parquet-root", required=True)
    parser.add_argument("--scannet-rgb-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = QSpatialTestContract(args.parquet_root, args.scannet_rgb_root)
    summary = score_predictions(args.predictions, contract, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
