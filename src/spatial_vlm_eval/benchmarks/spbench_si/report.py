"""Publication-gated SPBench-SI result discovery and two-protocol Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...models.common.runtime import _atomic_write_text, utc_now
from .data import DATASET_REVISION, EXPECTED_TASK_COUNTS, OFFICIAL_TEST_SIZE, TASK_SEQUENCE
from .profiles import PROFILE_SEQUENCE, PROFILES
from .scorer import (
    AUDIT_FORMULA,
    AUDIT_SCORER_PROTOCOL,
    AUDIT_RESULT_KIND,
    COMPATIBLE_INFERENCE_SCORER_PROTOCOLS,
    MAIN_FORMULA,
    RESULT_KIND,
    SCORER_PROTOCOL,
    inference_metadata_scorer_protocol_is_compatible,
)

DEFAULT_OUTPUT_NAME = "spbench-si-result.md"
TYPE_DISPLAY = {
    "object_abs_distance": "Absolute distance",
    "object_size_estimation": "Object size",
    "object_rel_distance": "Relative distance",
    "object_rel_direction": "Relative direction",
}
INPUT_DISPLAY = {
    "rgb": "RGB",
    "rgb_depthpro_midi_tor10": "RGB + DepthPro/MIDI/TOR10",
    "rgb_zoedepth": "RGB + ZoeDepth",
    "rgb_moge2_xyz": "RGB + MoGe-2 XYZ",
}


@dataclass(frozen=True, slots=True)
class ReportResult:
    profile: str
    summary_path: Path
    summary: dict[str, Any]
    audit: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and abs(float(actual) - expected) <= 1e-12


def _validate_result(summary_path: Path) -> ReportResult:
    summary = _load_json(summary_path)
    score_dir = summary_path.parent
    gates = _load_json(score_dir / "publication_gates.json")
    gate_values = gates.get("gates") if isinstance(gates.get("gates"), dict) else {}
    if gates.get("passed") is not True or not gate_values or not all(gate_values.values()):
        raise ValueError(f"Publication gates did not pass: {summary_path}")
    if summary.get("scorer_protocol") != SCORER_PROTOCOL or summary.get("result_kind") != RESULT_KIND:
        raise ValueError(f"Unexpected main scorer identity: {summary_path}")
    if summary.get("formula") != MAIN_FORMULA:
        raise ValueError(f"Main scorer formula mismatch: {summary_path}")
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), dict) else {}
    if dataset.get("revision") != DATASET_REVISION or dataset.get("official_test_size") != OFFICIAL_TEST_SIZE:
        raise ValueError(f"Locked dataset identity mismatch: {summary_path}")
    if dataset.get("task_counts") != EXPECTED_TASK_COUNTS or summary.get("num_scored_rows") != OFFICIAL_TEST_SIZE:
        raise ValueError(f"Incomplete SPBench-SI score: {summary_path}")
    inference = summary.get("inference") if isinstance(summary.get("inference"), dict) else {}
    profile_key = str(inference.get("profile") or "")
    if profile_key not in PROFILES:
        raise ValueError(f"Unregistered SPBench-SI profile: {summary_path}")
    profile = PROFILES[profile_key]
    for key, expected in {
        "profile": profile.key,
        "model_revision": profile.revision,
        "input_profile": profile.input_profile,
        "comparison_group": profile.comparison_group,
        "inference_protocol": profile.inference_protocol,
        "decoding": profile.decoding,
        "system_transport": profile.system_transport,
    }.items():
        if inference.get(key) != expected:
            raise ValueError(f"{profile_key} provenance mismatch for {key}: {summary_path}")
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    predictions = Path(str(artifacts.get("predictions", ""))).resolve()
    expected_predictions = score_dir.parent.parent / "predictions.jsonl"
    if predictions != expected_predictions.resolve() or not predictions.is_file():
        raise ValueError(f"Prediction artifact mismatch: {summary_path}")
    if artifacts.get("predictions_sha256") != _sha256(predictions):
        raise ValueError(f"Prediction digest mismatch: {summary_path}")
    inference_metadata = Path(str(artifacts.get("inference_metadata", ""))).resolve()
    if inference_metadata != predictions.with_suffix(predictions.suffix + ".metadata.json") or not inference_metadata.is_file():
        raise ValueError(f"Inference metadata artifact mismatch: {summary_path}")
    if artifacts.get("inference_metadata_sha256") != _sha256(inference_metadata):
        raise ValueError(f"Inference metadata digest mismatch: {summary_path}")
    metadata = _load_json(inference_metadata)
    declared_scorer_protocol = metadata.get("scorer_protocol")
    if not inference_metadata_scorer_protocol_is_compatible(declared_scorer_protocol):
        raise ValueError(
            "Inference metadata scorer_protocol is not compatible: "
            f"got={declared_scorer_protocol!r}, "
            f"allowed={sorted(COMPATIBLE_INFERENCE_SCORER_PROTOCOLS)!r}: {summary_path}"
        )
    if inference.get("declared_scorer_protocol") != declared_scorer_protocol:
        raise ValueError(f"Summary/inference scorer declaration mismatch: {summary_path}")
    prediction_validation = Path(str(artifacts.get("prediction_validation", ""))).resolve()
    if prediction_validation != (score_dir / "prediction_validation.json").resolve() or not prediction_validation.is_file():
        raise ValueError(f"Prediction validation artifact mismatch: {summary_path}")
    if artifacts.get("prediction_validation_sha256") != _sha256(prediction_validation):
        raise ValueError(f"Prediction validation digest mismatch: {summary_path}")
    scored = Path(str(artifacts.get("scored_rows", ""))).resolve()
    if scored != (score_dir / "scored_rows.jsonl").resolve() or not scored.is_file():
        raise ValueError(f"Main scored rows mismatch: {summary_path}")
    if artifacts.get("scored_rows_sha256") != _sha256(scored):
        raise ValueError(f"Main scored digest mismatch: {summary_path}")
    task_metrics = summary.get("task_metrics") if isinstance(summary.get("task_metrics"), dict) else {}
    if set(task_metrics) != set(TASK_SEQUENCE):
        raise ValueError(f"Incomplete four-task metrics: {summary_path}")
    for task, expected_count in EXPECTED_TASK_COUNTS.items():
        if task_metrics[task].get("total") != expected_count:
            raise ValueError(f"Task count mismatch for {task}: {summary_path}")
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    expected_overall = sum(float(task_metrics[key]["mean"]) for key in TASK_SEQUENCE) / 4
    if not _close(metrics.get("overall_four_task_macro"), expected_overall):
        raise ValueError(f"Overall four-task macro mismatch: {summary_path}")
    audit_info = summary.get("compatibility_audit") if isinstance(summary.get("compatibility_audit"), dict) else {}
    audit_path = Path(str(audit_info.get("summary", ""))).resolve()
    expected_audit = score_dir.parent / AUDIT_SCORER_PROTOCOL / "summary.json"
    if audit_path != expected_audit.resolve() or not audit_path.is_file():
        raise ValueError(f"Compatibility audit artifact mismatch: {summary_path}")
    if audit_info.get("summary_sha256") != _sha256(audit_path):
        raise ValueError(f"Compatibility audit digest mismatch: {summary_path}")
    audit = _load_json(audit_path)
    if audit.get("scorer_protocol") != AUDIT_SCORER_PROTOCOL or audit.get("result_kind") != AUDIT_RESULT_KIND:
        raise ValueError(f"Unexpected compatibility audit identity: {audit_path}")
    if audit.get("formula") != AUDIT_FORMULA:
        raise ValueError(f"Compatibility audit formula mismatch: {audit_path}")
    audit_dataset = audit.get("dataset") if isinstance(audit.get("dataset"), dict) else {}
    if audit_dataset != dataset or audit.get("num_scored_rows") != OFFICIAL_TEST_SIZE:
        raise ValueError(f"Compatibility audit dataset/count mismatch: {audit_path}")
    audit_tasks = audit.get("task_metrics") if isinstance(audit.get("task_metrics"), dict) else {}
    if set(audit_tasks) != set(TASK_SEQUENCE) or any(
        audit_tasks[task].get("total") != count for task, count in EXPECTED_TASK_COUNTS.items()
    ):
        raise ValueError(f"Compatibility audit task distribution mismatch: {audit_path}")
    audit_artifacts = audit.get("artifacts") if isinstance(audit.get("artifacts"), dict) else {}
    audit_scored = Path(str(audit_artifacts.get("scored_rows", ""))).resolve()
    if audit_scored != (audit_path.parent / "scored_rows.jsonl").resolve() or not audit_scored.is_file():
        raise ValueError(f"Compatibility audit scored rows mismatch: {audit_path}")
    if audit_artifacts.get("scored_rows_sha256") != _sha256(audit_scored):
        raise ValueError(f"Compatibility audit scored digest mismatch: {audit_path}")
    if audit_artifacts.get("predictions_sha256") != _sha256(predictions):
        raise ValueError(f"Compatibility audit prediction digest mismatch: {audit_path}")
    if audit_info.get("num_row_differences") != audit.get("num_main_vs_audit_differences"):
        raise ValueError(f"Compatibility audit difference count mismatch: {audit_path}")
    return ReportResult(profile_key, summary_path, summary, audit)


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
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise ValueError("Invalid canonical SPBench-SI summaries:\n" + "\n".join(errors))
    by_profile: dict[str, list[ReportResult]] = {}
    for result in found:
        by_profile.setdefault(result.profile, []).append(result)
    duplicates = {key: values for key, values in by_profile.items() if len(values) > 1}
    if duplicates:
        raise ValueError("Multiple publishable results for one SPBench-SI profile: " + "; ".join(
            f"{key}={[value.summary_path for value in values]}" for key, values in duplicates.items()
        ))
    unique = {key: values[0] for key, values in by_profile.items()}
    return [unique[key] for key in PROFILE_SEQUENCE if key in unique]


def _percent(value: float) -> str:
    return f"{value * 100:.2f}"


def render_markdown(results: list[ReportResult], *, generated_at: str | None = None) -> str:
    present = {result.profile for result in results}
    missing = [key for key in PROFILE_SEQUENCE if key not in present]
    provisional = missing == ["internvl3_78b"] and len(results) == 20
    complete = not missing and len(results) == 21
    if not (complete or provisional):
        raise ValueError(
            "SPBench-SI report requires 21/21, or exactly the approved 20/21 state missing internvl3_78b"
        )
    title = "# SPBench-SI 评测结果" if complete else "# SPBench-SI 评测结果（暂行 20/21）"
    status = "完整 21/21" if complete else "暂行 20/21；仅缺四卡 InternVL3-78B"
    lines = [
        title,
        "",
        f"- 状态：{status}",
        f"- 缺失 profile：{', '.join(f'`{key}`' for key in missing) if missing else '无'}",
        f"- 数据 revision：`{DATASET_REVISION}`（test {OFFICIAL_TEST_SIZE} 题）",
        f"- 主 scorer：`{SCORER_PROTOCOL}`",
        f"- 生成时间：{generated_at or utc_now()}",
        "- 主表按四题型等权宏平均；NQ/MCQ 各自为两题型等权。输入配置不同的轨不混同。",
        "",
        "## 主协议结果",
        "",
        "| 模型 | 实际输入配置 | Absolute distance | Object size | Relative distance | Relative direction | NQ | MCQ | Overall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        profile = PROFILES[result.profile]
        task = result.summary["task_metrics"]
        metrics = result.summary["metrics"]
        values = [_percent(float(task[key]["mean"])) for key in TASK_SEQUENCE]
        values.extend([
            _percent(float(metrics["nq_macro"])),
            _percent(float(metrics["mcq_macro"])),
            _percent(float(metrics["overall_four_task_macro"])),
        ])
        lines.append(
            f"| {profile.display_name} | {INPUT_DISPLAY[profile.input_profile]} | "
            + " | ".join(values) + " |"
        )
    lines.extend([
        "",
        "## Upstream compatibility audit（非主分）",
        "",
        "下表逐行复刻锁定 SpatialLadder 代码的 direct-mode 提取、`<=` MRA 边界与四题型聚合；"
        "它使用独立 score 目录，不与主协议逐行混表。",
        "",
        "| 模型 | Input | 主协议 Overall | Upstream audit Overall | 差值（主-audit） | 逐行差异数 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for result in results:
        profile = PROFILES[result.profile]
        main = float(result.summary["metrics"]["overall_four_task_macro"])
        audit = float(result.audit["metrics"]["overall_four_task_macro"])
        differences = int(result.audit["num_main_vs_audit_differences"])
        lines.append(
            f"| {profile.display_name} | {INPUT_DISPLAY[profile.input_profile]} | {_percent(main)} | "
            f"{_percent(audit)} | {(main - audit) * 100:+.2f} | {differences} |"
        )
    return "\n".join(lines) + "\n"


def build_report(output_root: str | Path, output: str | Path | None = None) -> Path:
    root = Path(output_root).resolve()
    destination = Path(output).resolve() if output else root / DEFAULT_OUTPUT_NAME
    _atomic_write_text(destination, render_markdown(discover_results(root)))
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=os.environ.get("SPBENCH_SI_OUTPUT_ROOT"))
    parser.add_argument("--output")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_root:
        raise ValueError("Set SPBENCH_SI_OUTPUT_ROOT or pass --output-root")
    results = discover_results(args.output_root)
    if args.list or args.check:
        for result in results:
            print(f"{result.profile}\t{result.summary_path}")
        missing = [key for key in PROFILE_SEQUENCE if key not in {result.profile for result in results}]
        print(f"completeness\t{len(results)}/21")
        print(f"missing\t{','.join(missing)}")
        if args.check and not (not missing or missing == ["internvl3_78b"]):
            raise SystemExit(1)
        return
    destination = build_report(args.output_root, args.output)
    print(f"[spbench-si-report] wrote {destination}")


if __name__ == "__main__":
    main()
