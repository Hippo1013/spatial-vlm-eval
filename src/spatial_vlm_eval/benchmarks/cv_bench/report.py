"""Publication-gated CV-Bench result discovery and Markdown rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...models.common.runtime import _atomic_write_text, utc_now
from .data import (
    DATASET_REVISION,
    EXPECTED_SOURCE_COUNTS,
    EXPECTED_TASK_COUNTS,
    OFFICIAL_TEST_SIZE,
)
from .profiles import PROFILE_SEQUENCE, PROFILES
from .scorer import RESULT_KIND, SCORER_PROTOCOL

DEFAULT_OUTPUT_NAME = "cv-bench-result.md"
METRIC_COLUMNS = (
    ("Spatial Relationship", "task_metrics", "spatial_relationship"),
    ("Object Count", "task_metrics", "object_count"),
    ("Depth Order", "task_metrics", "depth_order"),
    ("Relative Distance", "task_metrics", "relative_distance"),
    ("2D", "metrics", "accuracy_2d"),
    ("3D", "metrics", "accuracy_3d"),
    ("Overall", "metrics", "overall_accuracy"),
)


@dataclass(frozen=True, slots=True)
class ReportResult:
    profile: str
    summary_path: Path
    summary: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_artifact(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing artifact path {label}")
    return Path(value).resolve()


def _same_number(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and abs(float(actual) - float(expected)) <= 1e-12


def _validate_result(summary_path: Path) -> ReportResult:
    summary = _load_json(summary_path)
    score_dir = summary_path.parent
    gates = _load_json(score_dir / "publication_gates.json")
    gate_values = gates.get("gates") if isinstance(gates.get("gates"), dict) else {}
    if not gates.get("passed") or not gate_values or not all(gate_values.values()):
        raise ValueError(f"Publication gates did not pass: {summary_path}")
    if gates.get("scorer_protocol") != SCORER_PROTOCOL:
        raise ValueError(f"Publication gate protocol mismatch: {summary_path}")
    if Path(str(gates.get("summary", ""))).resolve() != summary_path.resolve():
        raise ValueError(f"Publication gate summary path mismatch: {summary_path}")
    if summary.get("scorer_protocol") != SCORER_PROTOCOL:
        raise ValueError(f"Unexpected scorer protocol: {summary_path}")
    if summary.get("result_kind") != RESULT_KIND:
        raise ValueError(f"Unexpected result kind: {summary_path}")
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), dict) else {}
    if dataset.get("revision") != DATASET_REVISION or dataset.get("official_test_size") != OFFICIAL_TEST_SIZE:
        raise ValueError(f"Locked dataset identity mismatch: {summary_path}")
    if summary.get("num_scored_rows") != OFFICIAL_TEST_SIZE:
        raise ValueError(f"Incomplete scored rows: {summary_path}")
    if not isinstance(dataset.get("fingerprint"), str) or not dataset["fingerprint"]:
        raise ValueError(f"Missing locked dataset fingerprint: {summary_path}")
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    expected_predictions = score_dir.parent.parent / "predictions.jsonl"
    predictions = _resolved_artifact(artifacts.get("predictions"), label="predictions")
    if predictions != expected_predictions.resolve() or not predictions.is_file():
        raise ValueError(f"Prediction artifact path mismatch: {summary_path}")
    if artifacts.get("predictions_sha256") != _sha256(predictions):
        raise ValueError(f"Prediction artifact hash mismatch: {summary_path}")
    scored_rows = _resolved_artifact(artifacts.get("scored_rows"), label="scored_rows")
    if scored_rows != (score_dir / "scored_rows.jsonl").resolve() or not scored_rows.is_file():
        raise ValueError(f"Scored-row artifact path mismatch: {summary_path}")
    if artifacts.get("scored_rows_sha256") != _sha256(scored_rows):
        raise ValueError(f"Scored-row artifact hash mismatch: {summary_path}")
    scored: list[dict[str, Any]] = []
    with scored_rows.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Scored row {line_number} is not an object: {summary_path}")
            if not isinstance(value.get("correct"), bool):
                raise ValueError(f"Scored row {line_number} has invalid correctness: {summary_path}")
            scored.append(value)
    if len(scored) != OFFICIAL_TEST_SIZE:
        raise ValueError(f"Scored-row artifact is not full-{OFFICIAL_TEST_SIZE}: {summary_path}")
    scored_indices = [row.get("index") for row in scored]
    if scored_indices != list(range(OFFICIAL_TEST_SIZE)):
        raise ValueError(f"Scored-row index coverage/order mismatch: {summary_path}")
    scored_task_counts = Counter(str(row.get("task")) for row in scored)
    scored_source_counts = Counter(str(row.get("source")) for row in scored)
    if dict(scored_task_counts) != EXPECTED_TASK_COUNTS:
        raise ValueError(f"Scored-row task distribution mismatch: {summary_path}")
    if dict(scored_source_counts) != EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"Scored-row source distribution mismatch: {summary_path}")

    metadata_path = predictions.with_suffix(predictions.suffix + ".metadata.json")
    metadata = _load_json(metadata_path)
    metadata_dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
    metadata_model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    metadata_checks = {
        "output": (str(Path(str(metadata.get("output", ""))).resolve()), str(predictions)),
        "output_sha256": (metadata.get("output_sha256"), _sha256(predictions)),
        "scorer_protocol": (metadata.get("scorer_protocol"), SCORER_PROTOCOL),
        "dataset.revision": (metadata_dataset.get("revision"), DATASET_REVISION),
        "dataset.fingerprint": (metadata_dataset.get("fingerprint"), dataset.get("fingerprint")),
        "dataset.official_test_size": (
            metadata_dataset.get("official_test_size"),
            OFFICIAL_TEST_SIZE,
        ),
        "publishable_inference": (metadata.get("publishable_inference"), True),
    }
    for label, (actual, expected) in metadata_checks.items():
        if actual != expected:
            raise ValueError(f"Inference metadata {label} mismatch: {summary_path}")
    inference = summary.get("inference") if isinstance(summary.get("inference"), dict) else {}
    profile_key = str(inference.get("profile") or "")
    if profile_key not in PROFILES:
        raise ValueError(f"Summary has an unregistered profile: {summary_path}")
    profile = PROFILES[profile_key]
    checks = {
        "profile": profile.key,
        "model_revision": profile.revision,
        "input_profile": profile.input_profile,
        "inference_protocol": profile.inference_protocol,
    }
    for key, expected in checks.items():
        if inference.get(key) != expected:
            raise ValueError(
                f"{profile_key} {key} mismatch: got={inference.get(key)!r}, expected={expected!r}"
            )
        if metadata_model.get(key) != expected:
            raise ValueError(f"{profile_key} metadata {key} mismatch: {summary_path}")
    decoding = inference.get("decoding") if isinstance(inference.get("decoding"), dict) else {}
    metadata_decoding = (
        metadata_model.get("decoding")
        if isinstance(metadata_model.get("decoding"), dict)
        else {}
    )
    for key, expected in profile.decoding.items():
        if key == "source":
            continue
        if decoding.get(key) != expected:
            raise ValueError(
                f"{profile_key} decoding {key} mismatch: got={decoding.get(key)!r}, expected={expected!r}"
            )
        if metadata_decoding.get(key) != expected:
            raise ValueError(f"{profile_key} metadata decoding {key} mismatch: {summary_path}")
    task_metrics = summary.get("task_metrics") if isinstance(summary.get("task_metrics"), dict) else {}
    source_metrics = (
        summary.get("source_metrics") if isinstance(summary.get("source_metrics"), dict) else {}
    )
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    if set(task_metrics) != {
        "spatial_relationship",
        "object_count",
        "depth_order",
        "relative_distance",
    }:
        raise ValueError(f"Incomplete task metrics: {summary_path}")
    task_names = {
        "spatial_relationship": "Relation",
        "object_count": "Count",
        "depth_order": "Depth",
        "relative_distance": "Distance",
    }
    for key, source_name in task_names.items():
        metric = task_metrics.get(key) or {}
        if metric.get("total") != EXPECTED_TASK_COUNTS[source_name]:
            raise ValueError(f"Task metric count mismatch for {key}: {summary_path}")
        correct = sum(bool(row["correct"]) for row in scored if row["task"] == source_name)
        if metric.get("correct") != correct or not _same_number(
            metric.get("accuracy"), correct / EXPECTED_TASK_COUNTS[source_name]
        ):
            raise ValueError(f"Task metric value mismatch for {key}: {summary_path}")
    if set(source_metrics) != set(EXPECTED_SOURCE_COUNTS):
        raise ValueError(f"Incomplete source metrics: {summary_path}")
    for source_name, expected_count in EXPECTED_SOURCE_COUNTS.items():
        metric = source_metrics.get(source_name) or {}
        if metric.get("total") != expected_count:
            raise ValueError(f"Source metric count mismatch for {source_name}: {summary_path}")
        correct = sum(bool(row["correct"]) for row in scored if row["source"] == source_name)
        if metric.get("correct") != correct or not _same_number(
            metric.get("accuracy"), correct / expected_count
        ):
            raise ValueError(f"Source metric value mismatch for {source_name}: {summary_path}")
    for key in ("accuracy_2d", "accuracy_3d", "overall_accuracy"):
        if not isinstance(metrics.get(key), (int, float)):
            raise ValueError(f"Missing numeric metric {key}: {summary_path}")
    expected_2d = (
        float(source_metrics["ADE20K"]["accuracy"])
        + float(source_metrics["COCO"]["accuracy"])
    ) / 2
    expected_3d = float(source_metrics["Omni3D"]["accuracy"])
    expected_overall = (expected_2d + expected_3d) / 2
    if not _same_number(metrics.get("accuracy_2d"), expected_2d):
        raise ValueError(f"2D aggregation mismatch: {summary_path}")
    if not _same_number(metrics.get("accuracy_3d"), expected_3d):
        raise ValueError(f"3D aggregation mismatch: {summary_path}")
    if not _same_number(metrics.get("overall_accuracy"), expected_overall):
        raise ValueError(f"Overall aggregation mismatch: {summary_path}")
    return ReportResult(profile_key, summary_path, summary)


def discover_results(output_root: str | Path) -> list[ReportResult]:
    root = Path(output_root).resolve()
    suffix = Path("scores") / SCORER_PROTOCOL / "summary.json"
    found: list[ReportResult] = []
    errors: list[str] = []
    for path in sorted(root.rglob("summary.json")):
        if not str(path).endswith(str(suffix)):
            continue
        try:
            found.append(_validate_result(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
    if errors:
        raise ValueError("Invalid canonical CV-Bench summaries:\n" + "\n".join(errors))
    duplicates: dict[str, list[Path]] = {}
    for result in found:
        duplicates.setdefault(result.profile, []).append(result.summary_path)
    ambiguous = {key: paths for key, paths in duplicates.items() if len(paths) > 1}
    if ambiguous:
        rendered = "; ".join(f"{key}={paths}" for key, paths in ambiguous.items())
        raise ValueError(f"Multiple publishable results for one CV-Bench profile: {rendered}")
    by_profile = {result.profile: result for result in found}
    return [by_profile[key] for key in PROFILE_SEQUENCE if key in by_profile]


def _metric(result: ReportResult, group: str, key: str) -> float:
    value = result.summary[group][key]
    if group == "task_metrics":
        value = value["accuracy"]
    return float(value)


def render_markdown(results: list[ReportResult], *, generated_at: str | None = None) -> str:
    present = {result.profile for result in results}
    missing = [key for key in PROFILE_SEQUENCE if key not in present]
    complete = not missing
    maxima = {
        (group, key): max(_metric(result, group, key) for result in results)
        for _, group, key in METRIC_COLUMNS
    } if results else {}
    lines = [
        "# CV-Bench 评测结果",
        "",
        f"- 数据 revision：`{DATASET_REVISION}`（完整 test：{OFFICIAL_TEST_SIZE} 条）",
        f"- Scorer protocol：`{SCORER_PROTOCOL}`",
        f"- 生成时间：{generated_at or utc_now()}",
        f"- 目标矩阵状态：{'完整（23/23）' if complete else f'未完整（{len(results)}/23）'}",
        f"- 缺失 profile：{', '.join(f'`{key}`' for key in missing) if missing else '无'}",
        "",
        "| 模型 | Spatial Relationship | Object Count | Depth Order | Relative Distance | 2D | 3D | Overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        values: list[str] = []
        for _, group, key in METRIC_COLUMNS:
            value = _metric(result, group, key)
            rendered = f"{value * 100:.2f}"
            if value == maxima[(group, key)]:
                rendered = f"**{rendered}**"
            values.append(rendered)
        lines.append(f"| {PROFILES[result.profile].display_name} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_report(output_root: str | Path, output: str | Path | None = None) -> Path:
    root = Path(output_root).resolve()
    results = discover_results(root)
    destination = Path(output).resolve() if output else root / DEFAULT_OUTPUT_NAME
    _atomic_write_text(destination, render_markdown(results))
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=os.environ.get("CVBENCH_OUTPUT_ROOT"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_root:
        raise ValueError("Set CVBENCH_OUTPUT_ROOT or pass --output-root")
    results = discover_results(args.output_root)
    if args.list or args.check:
        for result in results:
            print(f"{result.profile}\t{result.summary_path}")
        missing = [key for key in PROFILE_SEQUENCE if key not in {item.profile for item in results}]
        print(f"missing\t{','.join(missing)}")
        return
    destination = build_report(args.output_root, args.output)
    print(f"[cv-bench-report] wrote {destination}")


if __name__ == "__main__":
    main()
