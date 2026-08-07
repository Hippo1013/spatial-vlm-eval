"""Original-MRA robust scorer plus a byte-semantic upstream compatibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from ...models.common.runtime import atomic_write_json, atomic_write_jsonl, utc_now
from .data import (
    DATASET_REVISION,
    EXPECTED_TASK_COUNTS,
    MULTIPLE_CHOICE_TASKS,
    NUMERIC_TASKS,
    OFFICIAL_TEST_SIZE,
    TASK_SEQUENCE,
    SPBenchSITestContract,
)
from .prediction_validation import read_jsonl, validate_prediction_rows

SCORER_PROTOCOL = "spbench_si_original_mra10_strict_robust_direct_four_task_macro_v1"
AUDIT_SCORER_PROTOCOL = "spbench_si_upstream_7a0d2ee_default_direct_compat_v1"
RESULT_KIND = "spbench_si_official_definition_robust_internal_score"
AUDIT_RESULT_KIND = "spbench_si_upstream_code_compatibility_audit"
MRA_THRESHOLDS = tuple(Decimal(value) / Decimal(100) for value in range(50, 100, 5))
MAIN_FORMULA = {
    "metric": "mean_relative_accuracy",
    "thresholds": [str(value) for value in MRA_THRESHOLDS],
    "relative_error_comparison": "strict_less_than_1_minus_theta",
    "task_aggregation": "four_question_types_equal_weight_macro",
    "nq_mcq_aggregation": "two_constituent_types_equal_weight",
}
AUDIT_FORMULA = {
    "source_commit": "7a0d2ee85c28728835300310a349a53a15967f2e",
    "mode": "default_direct",
    "relative_error_comparison": "inclusive_less_than_or_equal_1_minus_theta",
    "task_aggregation": "upstream_four_question_types_equal_weight_macro",
}

_ANSWER_TAG = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)
_FINAL_REGION = re.compile(
    r"(?im)^\s*(?:the\s+)?(?:final\s+answer|answer)\s*(?:is|:)\s*(.+?)\s*$"
)
_CHOICE = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])(?:\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?![A-Za-z0-9_.])"
)
_NONFINITE = re.compile(r"(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])", re.IGNORECASE)
_RANGE = re.compile(
    r"(?:\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b)\s*"
    r"(?:-|–|—|to|through)\s*"
    r"(?:\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b)",
    re.IGNORECASE,
)

_WORD_VALUES = {
    "zero": Decimal(0), "one": Decimal(1), "two": Decimal(2), "three": Decimal(3),
    "four": Decimal(4), "five": Decimal(5), "six": Decimal(6), "seven": Decimal(7),
    "eight": Decimal(8), "nine": Decimal(9), "ten": Decimal(10), "eleven": Decimal(11),
    "twelve": Decimal(12), "thirteen": Decimal(13), "fourteen": Decimal(14),
    "fifteen": Decimal(15), "sixteen": Decimal(16), "seventeen": Decimal(17),
    "eighteen": Decimal(18), "nineteen": Decimal(19), "twenty": Decimal(20),
    "thirty": Decimal(30), "forty": Decimal(40), "fifty": Decimal(50),
    "sixty": Decimal(60), "seventy": Decimal(70), "eighty": Decimal(80),
    "ninety": Decimal(90), "a": Decimal(1), "an": Decimal(1),
}
_WORD_PATTERN = re.compile(
    r"(?<![A-Za-z])(" + "|".join(sorted(_WORD_VALUES, key=len, reverse=True)) + r")(?![A-Za-z])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParseResult:
    status: str
    value: str | None
    evidence: dict[str, Any]

    @property
    def succeeded(self) -> bool:
        return self.status == "ok" and self.value is not None


def _focus_region(text: str) -> tuple[str | None, dict[str, Any]]:
    tags = _ANSWER_TAG.findall(text)
    if len(tags) > 1:
        return None, {"source": "answer_tag", "reason": "multiple_complete_answer_tags", "count": len(tags)}
    if len(tags) == 1:
        return tags[0].strip(), {"source": "answer_tag", "count": 1}
    finals = _FINAL_REGION.findall(text)
    if len(finals) > 1:
        return None, {"source": "explicit_final", "reason": "multiple_final_regions", "count": len(finals)}
    if len(finals) == 1:
        return finals[0].strip(), {"source": "explicit_final", "count": 1}
    return text.strip(), {"source": "full_response"}


def parse_choice(raw_prediction: Any, allowed_letters: Iterable[str] = ("A", "B", "C", "D")) -> ParseResult:
    if not isinstance(raw_prediction, str) or not raw_prediction.strip():
        return ParseResult("empty", None, {"reason": "empty_or_non_string"})
    region, evidence = _focus_region(raw_prediction)
    if region is None:
        return ParseResult("conflict", None, evidence)
    allowed = {str(letter).upper() for letter in allowed_letters}
    matches = [match for match in _CHOICE.findall(region) if match in allowed]
    unique = sorted(set(matches))
    details = {**evidence, "region": region, "matches": matches, "unique": unique, "allowed": sorted(allowed)}
    if len(unique) == 1:
        return ParseResult("ok", unique[0], details)
    if not unique:
        return ParseResult("no_choice", None, details)
    return ParseResult("conflict", None, details)


def _decimal_string(value: Decimal) -> str:
    if value == value.to_integral():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def parse_numeric(raw_prediction: Any) -> ParseResult:
    if not isinstance(raw_prediction, str) or not raw_prediction.strip():
        return ParseResult("empty", None, {"reason": "empty_or_non_string"})
    region, evidence = _focus_region(raw_prediction)
    if region is None:
        return ParseResult("conflict", None, evidence)
    if _NONFINITE.search(region):
        return ParseResult("nonfinite", None, {**evidence, "region": region})
    if _RANGE.search(region):
        return ParseResult("range", None, {**evidence, "region": region})
    candidates: list[tuple[str, Decimal]] = []
    occupied: list[tuple[int, int]] = []
    for match in _NUMBER.finditer(region):
        try:
            value = Decimal(match.group(0).lstrip("+"))
        except InvalidOperation:
            continue
        candidates.append((match.group(0), value))
        occupied.append(match.span())
    for match in _WORD_PATTERN.finditer(region):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        candidates.append((match.group(0), _WORD_VALUES[match.group(1).casefold()]))
    negatives = re.search(r"(?<![A-Za-z0-9_.])-\s*(?:\d|\.\d)", region)
    if negatives:
        return ParseResult("negative", None, {**evidence, "region": region})
    unique = sorted({value for _raw, value in candidates})
    details = {
        **evidence,
        "region": region,
        "candidates": [{"raw": raw, "value": _decimal_string(value)} for raw, value in candidates],
        "unique": [_decimal_string(value) for value in unique],
    }
    if len(unique) == 1 and unique[0].is_finite() and unique[0] >= 0:
        return ParseResult("ok", _decimal_string(unique[0]), details)
    if not unique:
        return ParseResult("no_number", None, details)
    return ParseResult("conflict", None, details)


def relative_error(prediction: Decimal, target: Decimal) -> Decimal:
    if target == 0:
        return abs(prediction - target)
    return abs(prediction - target) / abs(target)


def mean_relative_accuracy_strict(prediction: Decimal | None, target: Decimal) -> Decimal:
    if prediction is None or not prediction.is_finite() or prediction < 0:
        return Decimal(0)
    error = relative_error(prediction, target)
    passed = sum(error < (Decimal(1) - threshold) for threshold in MRA_THRESHOLDS)
    return Decimal(passed) / Decimal(len(MRA_THRESHOLDS))


def mean_relative_accuracy_upstream(prediction: float | None, target: float) -> float:
    if prediction is None or not math.isfinite(prediction):
        return 0.0
    error = abs(prediction - target) if target == 0 else abs((prediction - target) / target)
    return sum(error <= 1.0 - float(threshold) for threshold in MRA_THRESHOLDS) / len(MRA_THRESHOLDS)


def _upstream_fuzzy_numeric(raw_prediction: str) -> str:
    text = raw_prediction.strip().lower()
    # Preserve the exact insertion order of SpatialLadder's current
    # ``fuzzy_match_number`` audit implementation: one..ninety, then
    # zero/a/an.  This is intentionally separate from the robust parser.
    upstream_words = tuple(
        (word, value) for word, value in _WORD_VALUES.items()
        if word not in {"zero", "a", "an"}
    ) + (
        ("zero", Decimal(0)), ("a", Decimal(1)), ("an", Decimal(1)),
    )
    for word, value in upstream_words:
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            return _decimal_string(value)
    match = re.search(r"(\d+(\.\d+)?)", text)
    return match.group(1) if match else "None"


def _upstream_choice(raw_prediction: str, target: str) -> tuple[str, float]:
    prediction = raw_prediction.replace("Answer:", "")
    first = prediction.split(" ")[0].strip()
    parsed = first.rstrip(".").upper() if re.fullmatch(r"[A-D]\.?", first) else prediction.strip()
    correct = parsed.lower() == target.lower()
    return parsed, 1.0 if correct else 0.0


def score_main_row(prediction_row: dict[str, Any], scoring_row: dict[str, Any]) -> dict[str, Any]:
    index = int(prediction_row["index"])
    raw = prediction_row["raw_prediction"]
    question_type = scoring_row["question_type"]
    target = scoring_row["ground_truth"]
    if question_type in MULTIPLE_CHOICE_TASKS:
        allowed = [option.split(".", 1)[0].strip() for option in scoring_row["options"]]
        parsed = parse_choice(raw, allowed)
        score = Decimal(1) if parsed.value == target else Decimal(0)
    else:
        parsed = parse_numeric(raw)
        try:
            prediction = Decimal(parsed.value) if parsed.succeeded else None
            target_value = Decimal(target)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid locked numeric target at index {index}") from exc
        score = mean_relative_accuracy_strict(prediction, target_value)
    return {
        "index": index,
        "official_id": scoring_row["official_id"],
        "question_type": question_type,
        "raw_prediction": raw,
        "ground_truth": target,
        "parse_status": parsed.status,
        "parsed_answer": parsed.value,
        "parse_evidence": parsed.evidence,
        "score": float(score),
    }


def score_upstream_row(prediction_row: dict[str, Any], scoring_row: dict[str, Any]) -> dict[str, Any]:
    raw = str(prediction_row["raw_prediction"])
    target = str(scoring_row["ground_truth"])
    question_type = str(scoring_row["question_type"])
    if question_type in MULTIPLE_CHOICE_TASKS:
        parsed, score = _upstream_choice(raw, target)
    else:
        parsed = _upstream_fuzzy_numeric(raw)
        try:
            prediction = float(parsed)
        except (TypeError, ValueError):
            prediction = None
        score = mean_relative_accuracy_upstream(prediction, float(target))
    return {
        "index": int(prediction_row["index"]),
        "official_id": scoring_row["official_id"],
        "question_type": question_type,
        "raw_prediction": raw,
        "ground_truth": target,
        "upstream_extracted_answer": parsed,
        "score": float(score),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    type_metrics: dict[str, dict[str, Any]] = {}
    for question_type in TASK_SEQUENCE:
        selected = [row for row in rows if row["question_type"] == question_type]
        score_sum = sum(Decimal(str(row["score"])) for row in selected)
        total = len(selected)
        type_metrics[question_type] = {
            "total": total,
            "score_sum": float(score_sum),
            "mean": float(score_sum / Decimal(total)) if total else 0.0,
        }
    nq = sum(Decimal(str(type_metrics[key]["mean"])) for key in NUMERIC_TASKS) / Decimal(2)
    mcq = sum(Decimal(str(type_metrics[key]["mean"])) for key in MULTIPLE_CHOICE_TASKS) / Decimal(2)
    overall = sum(Decimal(str(type_metrics[key]["mean"])) for key in TASK_SEQUENCE) / Decimal(4)
    micro = sum(Decimal(str(row["score"])) for row in rows) / Decimal(len(rows)) if rows else Decimal(0)
    return {
        "task_metrics": type_metrics,
        "metrics": {
            "nq_macro": float(nq),
            "mcq_macro": float(mcq),
            "overall_four_task_macro": float(overall),
            "micro_audit": float(micro),
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata_for_predictions(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing inference metadata: {metadata_path}")
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Inference metadata is not an object: {metadata_path}")
    return value


def _validate_inference_metadata(
    metadata: dict[str, Any], prediction_path: Path, contract: SPBenchSITestContract
) -> None:
    dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
    if metadata.get("publishable_inference") is not True:
        raise ValueError("Subset/non-publishable SPBench-SI predictions cannot be scored")
    if metadata.get("num_predictions") != OFFICIAL_TEST_SIZE:
        raise ValueError("Inference metadata does not cover full-1009")
    if metadata.get("output_sha256") != _sha256(prediction_path):
        raise ValueError("Inference metadata prediction digest mismatch")
    if dataset.get("fingerprint") != contract.dataset_fingerprint:
        raise ValueError("Inference metadata dataset fingerprint mismatch")
    if dataset.get("official_test_size") != OFFICIAL_TEST_SIZE:
        raise ValueError("Inference metadata official size mismatch")


def score_predictions(
    prediction_path: str | Path,
    contract: SPBenchSITestContract,
) -> tuple[Path, Path]:
    predictions = Path(prediction_path).resolve()
    rows = read_jsonl(predictions)
    validation = validate_prediction_rows(rows, contract, prediction_path=predictions)
    if not validation["passed"]:
        raise ValueError("SPBench-SI predictions failed mandatory full validation")
    metadata = _metadata_for_predictions(predictions)
    _validate_inference_metadata(metadata, predictions, contract)
    rows_by_index = {int(row["index"]): row for row in rows}
    main_rows = [score_main_row(rows_by_index[index], contract.scoring_row(index)) for index in range(len(contract))]
    audit_rows = [score_upstream_row(rows_by_index[index], contract.scoring_row(index)) for index in range(len(contract))]
    differences = [
        index for index, (main, audit) in enumerate(zip(main_rows, audit_rows))
        if abs(float(main["score"]) - float(audit["score"])) > 1e-12
    ]
    main_dir = predictions.parent / "scores" / SCORER_PROTOCOL
    audit_dir = predictions.parent / "scores" / AUDIT_SCORER_PROTOCOL
    main_scored = main_dir / "scored_rows.jsonl"
    audit_scored = audit_dir / "scored_rows.jsonl"
    atomic_write_jsonl(main_scored, main_rows)
    atomic_write_jsonl(audit_scored, audit_rows)
    main_aggregate = aggregate_rows(main_rows)
    audit_aggregate = aggregate_rows(audit_rows)
    common = {
        "dataset": {
            "repository": "hongxingli/SPBench",
            "revision": DATASET_REVISION,
            "fingerprint": contract.dataset_fingerprint,
            "official_test_size": OFFICIAL_TEST_SIZE,
            "task_counts": EXPECTED_TASK_COUNTS,
        },
        "num_scored_rows": len(main_rows),
        "inference": metadata.get("model"),
        "generated_at": utc_now(),
    }
    metadata_path = predictions.with_suffix(predictions.suffix + ".metadata.json")
    audit_summary = {
        **common,
        "result_kind": AUDIT_RESULT_KIND,
        "scorer_protocol": AUDIT_SCORER_PROTOCOL,
        "formula": AUDIT_FORMULA,
        **audit_aggregate,
        "num_main_vs_audit_differences": len(differences),
        "main_vs_audit_difference_indices": differences,
        "artifacts": {
            "predictions": str(predictions),
            "predictions_sha256": _sha256(predictions),
            "scored_rows": str(audit_scored),
            "scored_rows_sha256": _sha256(audit_scored),
        },
    }
    audit_summary_path = audit_dir / "summary.json"
    atomic_write_json(audit_summary_path, audit_summary)
    validation_path = main_dir / "prediction_validation.json"
    atomic_write_json(validation_path, validation)
    main_summary = {
        **common,
        "result_kind": RESULT_KIND,
        "scorer_protocol": SCORER_PROTOCOL,
        "formula": MAIN_FORMULA,
        **main_aggregate,
        "parse_status_counts": dict(Counter(row["parse_status"] for row in main_rows)),
        "compatibility_audit": {
            "protocol": AUDIT_SCORER_PROTOCOL,
            "summary": str(audit_summary_path),
            "summary_sha256": _sha256(audit_summary_path),
            "num_row_differences": len(differences),
        },
        "artifacts": {
            "predictions": str(predictions),
            "predictions_sha256": _sha256(predictions),
            "inference_metadata": str(metadata_path),
            "inference_metadata_sha256": _sha256(metadata_path),
            "prediction_validation": str(validation_path),
            "prediction_validation_sha256": _sha256(validation_path),
            "scored_rows": str(main_scored),
            "scored_rows_sha256": _sha256(main_scored),
        },
    }
    main_summary_path = main_dir / "summary.json"
    atomic_write_json(main_summary_path, main_summary)
    gates = {
        "passed": True,
        "scorer_protocol": SCORER_PROTOCOL,
        "summary": str(main_summary_path),
        "gates": {
            "full_prediction_validation": validation["passed"] and validation["num_unique_indices"] == OFFICIAL_TEST_SIZE,
            "locked_dataset_identity": validation["dataset_fingerprint"] == contract.dataset_fingerprint,
            "complete_four_task_distribution": dict(Counter(row["question_type"] for row in main_rows)) == EXPECTED_TASK_COUNTS,
            "complete_main_scored_rows": len(main_rows) == OFFICIAL_TEST_SIZE,
            "separate_upstream_compatibility_audit": audit_summary_path.is_file(),
            "publishable_inference_metadata": metadata.get("publishable_inference") is True,
            "locked_main_formula": main_summary["formula"] == MAIN_FORMULA,
            "locked_upstream_audit_formula": audit_summary["formula"] == AUDIT_FORMULA,
        },
    }
    gates["passed"] = all(gates["gates"].values())
    atomic_write_json(main_dir / "publication_gates.json", gates)
    if not gates["passed"]:
        raise RuntimeError("SPBench-SI publication gates failed")
    return main_summary_path, audit_summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--parquet", default=os.environ.get("SPBENCH_SI_PARQUET"))
    parser.add_argument("--images-archive", default=os.environ.get("SPBENCH_SI_IMAGES_ARCHIVE"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.parquet or not args.images_archive:
        raise ValueError("Set SPBENCH_SI_PARQUET and SPBENCH_SI_IMAGES_ARCHIVE")
    contract = SPBenchSITestContract(args.parquet, args.images_archive)
    main_summary, audit_summary = score_predictions(args.predictions, contract)
    print(json.dumps({"main": str(main_summary), "audit": str(audit_summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
