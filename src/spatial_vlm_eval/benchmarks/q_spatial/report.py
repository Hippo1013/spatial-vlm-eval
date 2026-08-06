"""Publication-gated Q-Spatial result discovery and Markdown rendering."""

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
    EXPECTED_SPLIT_TYPE_COUNTS,
    OFFICIAL_TEST_SIZE,
    SCANNET_CANONICAL_TYPES,
    STANDARD_SYSTEM_PROMPT_SHA256,
)
from .profiles import PROFILE_SEQUENCE, PROFILES, RGB_PROFILE_KEYS
from .scorer import RESULT_KIND, SCORER_PROTOCOL

DEFAULT_OUTPUT_NAME = "q-spatial-result.md"
COMPARISON_DISPLAY = {
    "rgb_only": "RGB",
    "rgb_derived_depth": "RGB + 派生深度",
    "rgb_derived_xyz": "RGB + MoGe-2 XYZ",
}
TYPE_DISPLAY = {
    "object_width": "Object width",
    "object_height": "Object height",
    "horizontal_distance": "Horizontal distance",
    "vertical_distance": "Vertical distance",
    "direct_distance": "Direct distance",
}


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


def _same_number(actual: Any, expected: float) -> bool:
    return isinstance(actual, (int, float)) and abs(float(actual) - float(expected)) <= 1e-12


def _read_scored(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Scored row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


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
    if summary.get("scorer_protocol") != SCORER_PROTOCOL or summary.get("result_kind") != RESULT_KIND:
        raise ValueError(f"Unexpected scorer identity: {summary_path}")
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), dict) else {}
    if dataset.get("revision") != DATASET_REVISION or dataset.get("official_test_size") != OFFICIAL_TEST_SIZE:
        raise ValueError(f"Locked dataset identity mismatch: {summary_path}")
    if summary.get("num_scored_rows") != OFFICIAL_TEST_SIZE:
        raise ValueError(f"Incomplete scored rows: {summary_path}")
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    expected_predictions = score_dir.parent.parent / "predictions.jsonl"
    predictions = Path(str(artifacts.get("predictions", ""))).resolve()
    if predictions != expected_predictions.resolve() or not predictions.is_file():
        raise ValueError(f"Prediction artifact path mismatch: {summary_path}")
    if artifacts.get("predictions_sha256") != _sha256(predictions):
        raise ValueError(f"Prediction artifact hash mismatch: {summary_path}")
    scored_path = Path(str(artifacts.get("scored_rows", ""))).resolve()
    if scored_path != (score_dir / "scored_rows.jsonl").resolve() or not scored_path.is_file():
        raise ValueError(f"Scored-row artifact path mismatch: {summary_path}")
    if artifacts.get("scored_rows_sha256") != _sha256(scored_path):
        raise ValueError(f"Scored-row artifact hash mismatch: {summary_path}")
    scored = _read_scored(scored_path)
    if len(scored) != OFFICIAL_TEST_SIZE or [row.get("index") for row in scored] != list(
        range(OFFICIAL_TEST_SIZE)
    ):
        raise ValueError(f"Scored-row coverage/order mismatch: {summary_path}")
    split_counts = Counter(str(row.get("split")) for row in scored)
    if dict(split_counts) != {"QSpatial_scannet": 170, "QSpatial_plus": 101}:
        raise ValueError(f"Scored-row split distribution mismatch: {summary_path}")
    scannet_type_counts = Counter(
        str(row.get("canonical_type"))
        for row in scored
        if row.get("split") == "QSpatial_scannet"
    )
    if dict(scannet_type_counts) != EXPECTED_SPLIT_TYPE_COUNTS["QSpatial_scannet"]:
        raise ValueError(f"Scored-row ScanNet type distribution mismatch: {summary_path}")

    metadata = _load_json(predictions.with_suffix(predictions.suffix + ".metadata.json"))
    metadata_dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
    metadata_model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    for label, actual, expected in (
        ("output", str(Path(str(metadata.get("output", ""))).resolve()), str(predictions)),
        ("output_sha256", metadata.get("output_sha256"), _sha256(predictions)),
        ("scorer_protocol", metadata.get("scorer_protocol"), SCORER_PROTOCOL),
        ("dataset.revision", metadata_dataset.get("revision"), DATASET_REVISION),
        ("dataset.fingerprint", metadata_dataset.get("fingerprint"), dataset.get("fingerprint")),
        ("dataset.official_test_size", metadata_dataset.get("official_test_size"), OFFICIAL_TEST_SIZE),
        ("publishable_inference", metadata.get("publishable_inference"), True),
    ):
        if actual != expected:
            raise ValueError(f"Inference metadata {label} mismatch: {summary_path}")

    inference = summary.get("inference") if isinstance(summary.get("inference"), dict) else {}
    profile_key = str(inference.get("profile") or "")
    if profile_key not in PROFILES:
        raise ValueError(f"Summary has an unregistered profile: {summary_path}")
    profile = PROFILES[profile_key]
    profile_checks = {
        "profile": profile.key,
        "model_revision": profile.revision,
        "input_profile": profile.input_profile,
        "comparison_group": profile.comparison_group,
        "inference_protocol": profile.inference_protocol,
        "seed_strategy": profile.seed_strategy,
    }
    for key, expected in profile_checks.items():
        if inference.get(key) != expected or metadata_model.get(key) != expected:
            raise ValueError(f"{profile_key} profile provenance mismatch for {key}: {summary_path}")
    decoding = inference.get("decoding") if isinstance(inference.get("decoding"), dict) else {}
    metadata_decoding = (
        metadata_model.get("decoding") if isinstance(metadata_model.get("decoding"), dict) else {}
    )
    if decoding != profile.decoding or metadata_decoding != profile.decoding:
        raise ValueError(f"{profile_key} decoding provenance mismatch: {summary_path}")
    if metadata_model.get("system_role_supported") != profile.system_role_supported:
        raise ValueError(f"{profile_key} system-role provenance mismatch: {summary_path}")
    prompt = metadata.get("prompt") if isinstance(metadata.get("prompt"), dict) else {}
    if prompt.get("system_prompt_sha256") != STANDARD_SYSTEM_PROMPT_SHA256:
        raise ValueError(f"{profile_key} Standard Prompt digest mismatch: {summary_path}")

    split_metrics = summary.get("split_metrics") if isinstance(summary.get("split_metrics"), dict) else {}
    type_metrics = (
        summary.get("scannet_type_metrics")
        if isinstance(summary.get("scannet_type_metrics"), dict)
        else {}
    )
    if set(split_metrics) != {"QSpatial_scannet", "QSpatial_plus"}:
        raise ValueError(f"Incomplete split metrics: {summary_path}")
    if set(type_metrics) != set(SCANNET_CANONICAL_TYPES):
        raise ValueError(f"Incomplete ScanNet type metrics: {summary_path}")
    for split, count in (("QSpatial_scannet", 170), ("QSpatial_plus", 101)):
        split_rows = [row for row in scored if row["split"] == split]
        for metric_name, row_key in (
            ("delta_le_1_25", "success_delta_le_1_25"),
            ("delta_le_2", "success_delta_le_2"),
            ("legacy_delta_lt_1_25", "legacy_success_delta_lt_1_25"),
            ("legacy_delta_lt_2", "legacy_success_delta_lt_2"),
        ):
            correct = sum(bool(row[row_key]) for row in split_rows)
            metric = split_metrics[split][metric_name]
            if metric.get("total") != count or metric.get("correct") != correct or not _same_number(
                metric.get("accuracy"), correct / count
            ):
                raise ValueError(f"Split metric mismatch for {split}/{metric_name}: {summary_path}")
    for canonical_type, expected_count in EXPECTED_SPLIT_TYPE_COUNTS["QSpatial_scannet"].items():
        rows = [
            row
            for row in scored
            if row["split"] == "QSpatial_scannet" and row["canonical_type"] == canonical_type
        ]
        metric = type_metrics[canonical_type]["delta_le_2"]
        correct = sum(bool(row["success_delta_le_2"]) for row in rows)
        if metric.get("total") != expected_count or metric.get("correct") != correct or not _same_number(
            metric.get("accuracy"), correct / expected_count
        ):
            raise ValueError(f"ScanNet type metric mismatch for {canonical_type}: {summary_path}")
    expected_overall = (
        split_metrics["QSpatial_scannet"]["delta_le_2"]["accuracy"]
        + split_metrics["QSpatial_plus"]["delta_le_2"]["accuracy"]
    ) / 2
    if not _same_number((summary.get("metrics") or {}).get("overall_delta_le_2"), expected_overall):
        raise ValueError(f"Overall split-macro mismatch: {summary_path}")
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
        raise ValueError("Invalid canonical Q-Spatial summaries:\n" + "\n".join(errors))
    by_key: dict[str, list[ReportResult]] = {}
    for result in found:
        by_key.setdefault(result.profile, []).append(result)
    ambiguous = {key: values for key, values in by_key.items() if len(values) > 1}
    if ambiguous:
        rendered = "; ".join(
            f"{key}={[value.summary_path for value in values]}" for key, values in ambiguous.items()
        )
        raise ValueError(f"Multiple publishable results for one Q-Spatial profile: {rendered}")
    unique = {key: values[0] for key, values in by_key.items()}
    return [unique[key] for key in PROFILE_SEQUENCE if key in unique]


def _split_accuracy(result: ReportResult, split: str, metric: str) -> float:
    return float(result.summary["split_metrics"][split][metric]["accuracy"])


def _format_percent(value: float, *, bold: bool = False) -> str:
    rendered = f"{value * 100:.2f}"
    return f"**{rendered}**" if bold else rendered


def render_markdown(results: list[ReportResult], *, generated_at: str | None = None) -> str:
    present = {result.profile for result in results}
    missing = [key for key in PROFILE_SEQUENCE if key not in present]
    rgb_present = [key for key in RGB_PROFILE_KEYS if key in present]
    maxima: dict[tuple[str, str], float] = {}
    for comparison_group in COMPARISON_DISPLAY:
        group_results = [
            result
            for result in results
            if PROFILES[result.profile].comparison_group == comparison_group
        ]
        for metric in ("scannet", "plus", "overall"):
            values: list[float] = []
            for result in group_results:
                scan = _split_accuracy(result, "QSpatial_scannet", "delta_le_2")
                plus = _split_accuracy(result, "QSpatial_plus", "delta_le_2")
                values.append({"scannet": scan, "plus": plus, "overall": (scan + plus) / 2}[metric])
            if values:
                maxima[(comparison_group, metric)] = max(values)
    lines = [
        "# Q-Spatial Bench 评测结果",
        "",
        f"- 数据 revision：`{DATASET_REVISION}`（完整评测：{OFFICIAL_TEST_SIZE} 条）",
        f"- Scorer protocol：`{SCORER_PROTOCOL}`",
        f"- 生成时间：{generated_at or utc_now()}",
        f"- RGB 轨完整度：{len(rgb_present)}/18",
        f"- 全轨完整度：{len(results)}/21",
        f"- 缺失 profile：{', '.join(f'`{key}`' for key in missing) if missing else '无'}",
        "- 比较规则：加粗只在相同 comparison group 内计算，不跨 RGB、派生深度与派生 XYZ 比较。",
        "",
        "## 主结果（δ≤2）",
        "",
        "| 模型 | Input track | Comparison group | ScanNet | Q-Spatial++ | Overall |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        profile = PROFILES[result.profile]
        scan = _split_accuracy(result, "QSpatial_scannet", "delta_le_2")
        plus = _split_accuracy(result, "QSpatial_plus", "delta_le_2")
        overall = (scan + plus) / 2
        group = profile.comparison_group
        values = [
            _format_percent(scan, bold=scan == maxima.get((group, "scannet"))),
            _format_percent(plus, bold=plus == maxima.get((group, "plus"))),
            _format_percent(overall, bold=overall == maxima.get((group, "overall"))),
        ]
        lines.append(
            f"| {profile.display_name} | `{profile.input_profile}` | "
            f"{COMPARISON_DISPLAY[group]} | " + " | ".join(values) + " |"
        )
    lines.extend(
        [
            "",
            "## ScanNet 五类明细（δ≤2）",
            "",
            "| 模型 | " + " | ".join(TYPE_DISPLAY[key] for key in SCANNET_CANONICAL_TYPES) + " |",
            "| --- | " + " | ".join("---:" for _ in SCANNET_CANONICAL_TYPES) + " |",
        ]
    )
    for result in results:
        values = [
            _format_percent(
                float(result.summary["scannet_type_metrics"][key]["delta_le_2"]["accuracy"])
            )
            for key in SCANNET_CANONICAL_TYPES
        ]
        lines.append(f"| {PROFILES[result.profile].display_name} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 解析与严格阈值审计",
            "",
            "| 模型 | Overall δ≤1.25 | 旧 notebook Overall δ<2 | 主/旧审计差异条数 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        metrics = result.summary["metrics"]
        lines.append(
            f"| {PROFILES[result.profile].display_name} | "
            f"{_format_percent(float(metrics['overall_delta_le_1_25']))} | "
            f"{_format_percent(float(metrics['legacy_notebook_overall_delta_lt_2']))} | "
            f"{int(result.summary['num_main_vs_legacy_differences'])} |"
        )
    differing = [
        result.profile
        for result in results
        if int(result.summary.get("num_main_vs_legacy_differences", 0)) > 0
    ]
    lines.extend(
        [
            "",
            "- 存在主 scorer / 旧 notebook 差异的 profile："
            + (", ".join(f"`{key}`" for key in differing) if differing else "无"),
            "- Overall 为 ScanNet 与 Q-Spatial++ 成功率等权平均；271 条 micro accuracy 仅保存在 summary 审计字段。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(output_root: str | Path, output: str | Path | None = None) -> Path:
    root = Path(output_root).resolve()
    results = discover_results(root)
    destination = Path(output).resolve() if output else root / DEFAULT_OUTPUT_NAME
    _atomic_write_text(destination, render_markdown(results))
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=os.environ.get("QSPATIAL_OUTPUT_ROOT"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output_root:
        raise ValueError("Set QSPATIAL_OUTPUT_ROOT or pass --output-root")
    results = discover_results(args.output_root)
    if args.list or args.check:
        for result in results:
            print(f"{result.profile}\t{result.summary_path}")
        missing = [key for key in PROFILE_SEQUENCE if key not in {item.profile for item in results}]
        print(f"rgb-completeness\t{sum(key in {item.profile for item in results} for key in RGB_PROFILE_KEYS)}/18")
        print(f"all-completeness\t{len(results)}/21")
        print(f"missing\t{','.join(missing)}")
        return
    destination = build_report(args.output_root, args.output)
    print(f"[q-spatial-report] wrote {destination}")


if __name__ == "__main__":
    main()
