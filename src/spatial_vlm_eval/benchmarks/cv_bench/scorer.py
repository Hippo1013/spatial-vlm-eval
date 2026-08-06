"""Robust deterministic multiple-choice scorer for the locked CV-Bench split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...models.common.runtime import atomic_write_json, atomic_write_jsonl, utc_now
from .data import (
    CVBenchTestContract,
    DATASET_REVISION,
    EXPECTED_SOURCE_COUNTS,
    EXPECTED_TASK_COUNTS,
    OFFICIAL_TEST_SIZE,
)
from .prediction_validation import read_jsonl, validate_prediction_rows

LEGACY_SCORER_PROTOCOL_V2 = (
    "cv_bench_robust_mcq_v2_answer_tag_unique_letter_or_exact_option_text"
)
SCORER_PROTOCOL = (
    "cv_bench_robust_mcq_v3_answer_tag_unique_declared_letter_or_exact_option_text"
)
COMPATIBLE_INFERENCE_SCORER_PROTOCOLS = frozenset(
    {LEGACY_SCORER_PROTOCOL_V2, SCORER_PROTOCOL}
)
RESULT_KIND = "cv_bench_official_formula_robust_parser_internal_score"

TERMINAL_TOKENS = ("<eos>", "<|im_end|>", "<|endoftext|>")

TASK_KEYS = {
    "Relation": "spatial_relationship",
    "Count": "object_count",
    "Depth": "depth_order",
    "Distance": "relative_distance",
}


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    answer: str | None
    status: str
    evidence: tuple[str, ...] = ()


def _normalized_text(value: Any) -> str:
    text = str(value).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.!,;:'\"`“”‘’")


def inference_metadata_scorer_protocol_is_compatible(value: Any) -> bool:
    return str(value) in COMPATIBLE_INFERENCE_SCORER_PROTOCOLS


def _strip_terminal_tokens(text: str) -> tuple[str, tuple[str, ...]]:
    """Remove only known generation terminators from the parser view.

    ``raw_prediction`` remains unchanged in scored artifacts. Returning the exact
    stripped tokens makes the normalization auditable rather than silently
    rewriting a model response.
    """

    cleaned = str(text).strip()
    stripped: list[str] = []
    changed = True
    while changed:
        changed = False
        folded = cleaned.casefold()
        for token in TERMINAL_TOKENS:
            if folded.endswith(token.casefold()):
                cleaned = cleaned[: -len(token)].strip()
                stripped.append(token)
                changed = True
                break
    return cleaned, tuple(stripped)


def _explicit_letter_candidates(text: str) -> dict[str, set[str]]:
    candidates: dict[str, set[str]] = {}

    def add(letter: str, source: str) -> None:
        candidates.setdefault(str(letter).upper(), set()).add(source)

    upper = text.upper()
    for letter in re.findall(r"[\(\[]\s*([A-Z])\s*[\)\]]", upper):
        add(letter, "parenthesized_letter")
    for letter in re.findall(
        r"\b(?:FINAL\s+)?(?:ANSWER|OPTION|CHOICE)\b"
        r"(?:\s+IS\b)?\s*(?::|=|-)?\s*(?:\*\*)?"
        r"[\(\[]?([A-Z])(?:[\)\]]|\b)(?:\*\*)?",
        upper,
    ):
        add(letter, "labelled_answer_letter")
    whole = re.fullmatch(r"\s*[\(\[]?([A-Z])[\)\].,:;]?\s*", upper)
    if whole:
        add(whole.group(1), "whole_letter")
    leading = re.match(r"\s*[\(\[]?([A-Z])[\)\].:]\s*", upper)
    if leading:
        add(leading.group(1), "leading_label_letter")
    leading_parenthetical_text = re.match(r"\s*([A-Z])\s+\(", upper)
    if leading_parenthetical_text:
        add(leading_parenthetical_text.group(1), "leading_parenthetical_text_letter")

    nonempty_lines = [line.strip() for line in upper.splitlines() if line.strip()]
    if nonempty_lines:
        first = re.fullmatch(r"[\(\[]?([A-Z])[\)\].,:;]?", nonempty_lines[0])
        if first:
            add(first.group(1), "first_line_letter")
        last = re.fullmatch(r"[\(\[]?([A-Z])[\)\].,:;]?", nonempty_lines[-1])
        if last:
            add(last.group(1), "last_line_letter")

    two_letter_response = re.fullmatch(
        r"\s*[\(\[]?([A-Z])[\)\]]?\s+[\(\[]?([A-Z])[\)\]]?\s*",
        upper,
    )
    if two_letter_response:
        for letter in two_letter_response.groups():
            add(letter, "two_letter_response")

    for match in re.finditer(
        r"\b(?:OPTION\b\s+)?([A-Z])\b\s+(?:OR|AND)\s+"
        r"(?:OPTION\b\s+)?([A-Z])\b",
        upper,
    ):
        for letter in match.groups():
            add(letter, "paired_letter_expression")
    return candidates


def _text_candidates(text: str) -> list[str]:
    values = [_normalized_text(text)]
    stripped_prefix = re.sub(
        r"^\s*(?:the\s+)?(?:(?:final|correct)\s+)*(?:answer|option|choice)\b"
        r"(?:\s+is\b)?\s*(?::|=|-)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    values.append(_normalized_text(stripped_prefix))
    labelled = re.match(
        r"^\s*[\(\[]?[A-Z][\)\].:]\s*(.+?)\s*$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if labelled:
        values.append(_normalized_text(labelled.group(1)))
    parenthetical_text = re.match(
        r"^\s*[A-Z]\s+\((.+?)\)\s*$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if parenthetical_text:
        values.append(_normalized_text(parenthetical_text.group(1)))
    return list(dict.fromkeys(value for value in values if value))


def _letter_status(sources: set[str]) -> str:
    if "whole_letter" in sources:
        return "explicit_letter"
    if "first_line_letter" in sources and "last_line_letter" not in sources:
        return "first_line_explicit_letter"
    if "last_line_letter" in sources and "first_line_letter" not in sources:
        return "last_line_explicit_letter"
    if {"leading_label_letter", "leading_parenthetical_text_letter"} & sources:
        return "leading_explicit_letter"
    return "explicit_letter"


def parse_answer(raw_prediction: Any, choices: list[str]) -> ParsedAnswer:
    text = str(raw_prediction)
    if not text.strip():
        return ParsedAnswer(None, "empty")
    tagged_answers = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL
    )
    if len(tagged_answers) > 1:
        return ParsedAnswer(
            None,
            "multiple_answer_tags",
            (f"answer_tag_count:{len(tagged_answers)}",),
        )
    parse_text, terminal_tokens = _strip_terminal_tokens(
        tagged_answers[0] if tagged_answers else text
    )
    evidence_prefix = (("answer_tag",) if tagged_answers else ()) + tuple(
        f"stripped_terminal_token:{token}" for token in terminal_tokens
    )
    status_prefix = "answer_tag_" if tagged_answers else ""
    if terminal_tokens:
        status_prefix += "terminal_token_stripped_"
    if tagged_answers and not parse_text:
        return ParsedAnswer(None, "empty_answer_tag", evidence_prefix)
    if not parse_text:
        return ParsedAnswer(None, "empty_after_terminal_tokens", evidence_prefix)
    letters = _explicit_letter_candidates(parse_text)
    letter_evidence = tuple(
        f"letter:{letter}:{source}"
        for letter in sorted(letters)
        for source in sorted(letters[letter])
    )
    if len(letters) > 1:
        return ParsedAnswer(
            None,
            status_prefix + "multiple_answers",
            evidence_prefix + letter_evidence,
        )
    if letters:
        letter_answer = next(iter(letters))
        position = string.ascii_uppercase.index(letter_answer)
        if position >= len(choices):
            return ParsedAnswer(
                None,
                status_prefix + "out_of_range",
                evidence_prefix + letter_evidence,
            )
    else:
        letter_answer = None

    normalized_choices = [_normalized_text(choice) for choice in choices]
    matched_positions = {
        position
        for candidate in _text_candidates(parse_text)
        for position, choice in enumerate(normalized_choices)
        if candidate == choice
    }
    if len(matched_positions) > 1:
        return ParsedAnswer(
            None,
            status_prefix + "multiple_answers",
            evidence_prefix + letter_evidence + ("multiple_exact_option_texts",),
        )
    text_answer = (
        string.ascii_uppercase[next(iter(matched_positions))] if matched_positions else None
    )
    option_text_evidence = (
        (f"exact_option_text:{text_answer}",) if text_answer is not None else ()
    )
    if letter_answer and text_answer:
        if letter_answer != text_answer:
            return ParsedAnswer(
                None,
                status_prefix + "conflict",
                evidence_prefix + letter_evidence + option_text_evidence,
            )
        return ParsedAnswer(
            letter_answer,
            status_prefix + "letter_and_option_text",
            evidence_prefix + letter_evidence + option_text_evidence,
        )
    if letter_answer:
        return ParsedAnswer(
            letter_answer,
            status_prefix + _letter_status(letters[letter_answer]),
            evidence_prefix + letter_evidence,
        )
    if text_answer:
        return ParsedAnswer(
            text_answer,
            status_prefix + "option_text",
            evidence_prefix + option_text_evidence,
        )
    return ParsedAnswer(None, status_prefix + "unparsed", evidence_prefix)


def _accuracy(correct: int, total: int) -> float:
    if total <= 0:
        raise ValueError("Cannot compute an accuracy for an empty CV-Bench group")
    return correct / total


def _group_metrics(
    rows: list[dict[str, Any]], key: str, expected_counts: dict[str, int]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    if {name: len(values) for name, values in grouped.items()} != expected_counts:
        actual_counts = {name: len(values) for name, values in grouped.items()}
        raise ValueError(
            f"CV-Bench {key} counts do not match locked counts: {actual_counts}"
        )
    return {
        name: {
            "correct": sum(bool(row["correct"]) for row in values),
            "total": len(values),
            "accuracy": _accuracy(sum(bool(row["correct"]) for row in values), len(values)),
        }
        for name, values in sorted(grouped.items())
    }


def score_rows(
    prediction_rows: list[dict[str, Any]], contract: CVBenchTestContract
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for prediction in sorted(prediction_rows, key=lambda row: int(row["index"])):
        index = int(prediction["index"])
        source = contract.scoring_row(index)
        parsed = parse_answer(prediction["raw_prediction"], source["choices"])
        correct = parsed.answer == source["gold"]
        scored.append(
            {
                "index": index,
                "raw_prediction": str(prediction["raw_prediction"]),
                "parsed_answer": parsed.answer,
                "parse_status": parsed.status,
                "parse_evidence": list(parsed.evidence),
                "gold": source["gold"],
                "correct": bool(correct),
                "type": source["type"],
                "task": source["task"],
                "source": source["source"],
            }
        )
    task_raw = _group_metrics(scored, "task", EXPECTED_TASK_COUNTS)
    source_metrics = _group_metrics(scored, "source", EXPECTED_SOURCE_COUNTS)
    task_metrics = {TASK_KEYS[name]: metrics for name, metrics in task_raw.items()}
    accuracy_2d = (source_metrics["ADE20K"]["accuracy"] + source_metrics["COCO"]["accuracy"]) / 2
    accuracy_3d = source_metrics["Omni3D"]["accuracy"]
    overall = (accuracy_2d + accuracy_3d) / 2
    aggregate = {
        "task_metrics": task_metrics,
        "source_metrics": source_metrics,
        "metrics": {
            "accuracy_2d": accuracy_2d,
            "accuracy_3d": accuracy_3d,
            "overall_accuracy": overall,
            "micro_accuracy_audit_only": _accuracy(
                sum(bool(row["correct"]) for row in scored), len(scored)
            ),
        },
        "parse_status_counts": dict(sorted(Counter(row["parse_status"] for row in scored).items())),
    }
    return scored, aggregate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_and_validate_metadata(
    prediction_path: Path,
    contract: CVBenchTestContract,
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
    declared_scorer_protocol = metadata.get("scorer_protocol")
    if not inference_metadata_scorer_protocol_is_compatible(declared_scorer_protocol):
        errors.append(
            "metadata.scorer_protocol is not compatible with the current CV-Bench scorer: "
            f"got={declared_scorer_protocol!r}, "
            f"allowed={sorted(COMPATIBLE_INFERENCE_SCORER_PROTOCOLS)!r}"
        )
    return metadata, errors


def score_predictions(
    prediction_path: str | Path,
    contract: CVBenchTestContract,
    *,
    output_dir: str | Path | None = None,
    require_metadata: bool = True,
) -> dict[str, Any]:
    predictions = Path(prediction_path).resolve()
    rows = read_jsonl(predictions)
    validation = validate_prediction_rows(
        rows,
        contract,
        prediction_path=predictions,
        allow_subset=False,
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
        raise ValueError("CV-Bench full prediction validation failed; see prediction_validation.json")

    scored, aggregate = score_rows(rows, contract)
    scored_path = destination / "scored_rows.jsonl"
    atomic_write_jsonl(scored_path, scored)
    inference_model = metadata.get("model", {}) if metadata else {}
    summary = {
        "schema_version": 1,
        "result_kind": RESULT_KIND,
        "scorer_protocol": SCORER_PROTOCOL,
        "dataset": {
            "repository": "nyu-visionx/CV-Bench",
            "revision": DATASET_REVISION,
            "fingerprint": contract.dataset_fingerprint,
            "official_test_size": OFFICIAL_TEST_SIZE,
        },
        "inference": {
            "profile": inference_model.get("profile"),
            "model": inference_model.get("model"),
            "model_revision": inference_model.get("model_revision"),
            "input_profile": inference_model.get("input_profile"),
            "inference_protocol": inference_model.get("inference_protocol"),
            "declared_scorer_protocol": metadata.get("scorer_protocol") if metadata else None,
            "backend": inference_model.get("backend"),
            "decoding": inference_model.get("decoding"),
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
            "complete_task_groups": set(aggregate["task_metrics"])
            == set(TASK_KEYS.values()),
            "complete_source_groups": set(aggregate["source_metrics"])
            == set(EXPECTED_SOURCE_COUNTS),
            "protocol_consistency": summary["scorer_protocol"] == SCORER_PROTOCOL,
            "inference_metadata_consistency": metadata is not None or not require_metadata,
            "inference_scorer_protocol_compatible": (
                metadata is not None
                and inference_metadata_scorer_protocol_is_compatible(
                    metadata.get("scorer_protocol")
                )
            )
            or not require_metadata,
        },
        "summary": str(summary_path),
        "generated_at": utc_now(),
    }
    gates["passed"] = all(gates["gates"].values())
    atomic_write_json(destination / "publication_gates.json", gates)
    if not gates["passed"]:
        raise RuntimeError("CV-Bench publication gates failed")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = CVBenchTestContract(args.dataset_root)
    summary = score_predictions(
        args.predictions,
        contract,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
