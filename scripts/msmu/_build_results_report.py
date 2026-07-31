#!/usr/bin/env python3
"""Build a provenance-preserving MSMU-Bench Markdown results table."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from _score_pending_results import (  # noqa: E402
    MSMU_OFFICIAL_TEST_SIZE,
    OFFICIAL_QUAL_TYPES,
    OFFICIAL_QUANT_TYPES,
    SCORER_PROTOCOL,
    complete_state_errors,
    read_json,
)


REPORT_TITLE = "# MSMU-Bench评测结果"
REPORT_NOTE = (
    "注：公平版仅使用 MSMU 提供的 RGB 图像与原始问题；原生版保留模型官方设计中由同一 RGB "
    "派生的额外输入、提示或组件，且不使用 GT 深度、reference 或额外标注。"
)
DEFAULT_OUTPUT_NAME = "msmu-result.md"
OFFICIAL_METRICS = (
    ("existence", "存在性"),
    ("count", "物体计数"),
    ("scale_estimation", "尺度估计"),
    ("grounding", "空间定位"),
    ("relative_position", "相对位置"),
    ("absolute_distance", "绝对距离"),
    ("scale_compare", "尺度比较"),
    ("refer_obj_estimation", "参照物估计"),
)
EXPECTED_RESULT_KIND = "official-compatible internal score"


class ConfigurationError(RuntimeError):
    """The requested report cannot be generated safely."""


@dataclass(frozen=True)
class ScoreResult:
    result_id: str
    summary_path: Path
    predictions_path: Path
    metadata_path: Path
    score_dir: Path
    eligible: bool
    reason: str
    model: str = ""
    profile: str = ""
    model_revision: str = ""
    inference_protocol: str = ""
    input_profile: str = ""
    scorer_protocol: str = ""
    result_kind: str = ""
    accuracies: tuple[float, ...] = ()
    average: float | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover MSMU score summaries across protocols and build one "
            "publication-gated Markdown table."
        )
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_results",
        help="list every discovered score summary without writing a report",
    )
    parser.add_argument(
        "--results-root",
        help="absolute stage-three results root; defaults to .env.server configuration",
    )
    parser.add_argument(
        "--output",
        help=(
            "absolute Markdown output path; defaults to "
            "<results-root>/msmu-result.md"
        ),
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help=(
            "include this exact metadata profile; repeat or pass comma-separated "
            "values; omitted means every profile"
        ),
    )
    parser.add_argument(
        "--scorer-protocol",
        action="append",
        default=[],
        help=(
            "include exactly one scorer protocol; defaults to the current "
            "canonical protocol"
        ),
    )
    return parser.parse_args(argv)


def split_filters(values: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for item in raw.split(","):
            value = item.strip()
            if not value or value in seen:
                continue
            resolved.append(value)
            seen.add(value)
    return resolved


def resolve_results_root(argument: str | None) -> Path:
    raw = (
        argument
        or os.environ.get("MSMU_REPORT_RESULTS_ROOT", "")
        or os.environ.get("MSMU_SCORE_RESULTS_ROOT", "")
    )
    if not raw:
        raise ConfigurationError(
            "results root is not configured; set MANUAL_TEST_OUTPUT_ROOT in .env.server"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ConfigurationError(f"results root must be absolute: {raw}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ConfigurationError(f"results root is not a directory: {resolved}")
    return resolved


def resolve_output_path(argument: str | None, results_root: Path) -> Path:
    path = Path(argument).expanduser() if argument else results_root / DEFAULT_OUTPUT_NAME
    if not path.is_absolute():
        raise ConfigurationError(f"output path must be absolute: {path}")
    resolved = path.resolve()
    if resolved.exists() and not resolved.is_file():
        raise ConfigurationError(f"output path is not a regular file: {resolved}")
    return resolved


def score_summary_paths(results_root: Path) -> list[Path]:
    return sorted(
        (
            path.resolve()
            for path in results_root.rglob("summary.json")
            if path.is_file()
            and path.parent.parent.name == "scores"
            and path.parent.name
        ),
        key=str,
    )


def compact_model_name(value: str) -> str:
    parts = []
    for raw_part in value.split(" + "):
        part = raw_part.strip()
        if not part:
            continue
        if part.startswith("/"):
            part = Path(part).name
        elif "/" in part:
            part = part.rsplit("/", 1)[1]
        parts.append(part)
    return " + ".join(parts) or value


def finite_accuracy(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        return None
    return resolved


def metadata_values(
    metadata: Any,
    *,
    prediction_dir: Path,
    scorer_protocol: str,
) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    if not isinstance(metadata, dict):
        return {}, ["prediction metadata is not a JSON object"]

    model_metadata = metadata.get("model")
    if not isinstance(model_metadata, dict):
        return {}, ["prediction metadata model is not a JSON object"]

    raw_model = model_metadata.get("model")
    profile = model_metadata.get("profile")
    model_revision = model_metadata.get("model_revision")
    inference_protocol = metadata.get("inference_protocol")
    nested_inference_protocol = model_metadata.get("inference_protocol")
    metadata_scorer_protocol = metadata.get("scorer_protocol")
    input_profile = model_metadata.get("input_profile")

    required_strings = {
        "model": raw_model,
        "profile": profile,
        "model_revision": model_revision,
        "inference_protocol": inference_protocol,
        "model.inference_protocol": nested_inference_protocol,
        "scorer_protocol": metadata_scorer_protocol,
    }
    for name, value in required_strings.items():
        if not isinstance(value, str) or not value.strip():
            errors.append(f"prediction metadata has invalid {name}")

    if metadata.get("publishable_inference") is not True:
        errors.append("prediction metadata publishable_inference is not true")
    if metadata.get("num_predictions") != MSMU_OFFICIAL_TEST_SIZE:
        errors.append(
            "prediction metadata num_predictions is not "
            f"{MSMU_OFFICIAL_TEST_SIZE}"
        )
    if (
        isinstance(inference_protocol, str)
        and prediction_dir.parent.name != inference_protocol
    ):
        errors.append("prediction directory does not match inference_protocol")
    if (
        isinstance(nested_inference_protocol, str)
        and nested_inference_protocol != inference_protocol
    ):
        errors.append(
            "top-level and model inference_protocol metadata do not match"
        )
    if (
        isinstance(metadata_scorer_protocol, str)
        and metadata_scorer_protocol != scorer_protocol
    ):
        errors.append("prediction metadata scorer_protocol does not match score")

    if errors:
        return {}, errors
    assert isinstance(raw_model, str)
    assert isinstance(profile, str)
    assert isinstance(model_revision, str)
    assert isinstance(inference_protocol, str)
    return {
        "model": compact_model_name(raw_model.strip()),
        "profile": profile.strip(),
        "model_revision": model_revision.strip(),
        "inference_protocol": inference_protocol.strip(),
        "input_profile": (
            input_profile.strip() if isinstance(input_profile, str) else ""
        ),
    }, []


def summary_values(
    summary: Any,
    *,
    scorer_protocol: str,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(summary, dict):
        return {}, ["summary is not a JSON object"]

    if summary.get("protocol") != scorer_protocol:
        errors.append("summary protocol does not match score directory")
    if summary.get("result_kind") != EXPECTED_RESULT_KIND:
        errors.append(
            "summary result_kind is not official-compatible internal score"
        )

    official_types = summary.get("official_types")
    expected_types = OFFICIAL_QUANT_TYPES | OFFICIAL_QUAL_TYPES
    if not isinstance(official_types, dict) or set(official_types) != expected_types:
        errors.append("summary official_types are incomplete")
        return {}, errors

    accuracies: list[float] = []
    for key, _header in OFFICIAL_METRICS:
        item = official_types.get(key)
        accuracy = (
            finite_accuracy(item.get("accuracy"))
            if isinstance(item, dict)
            else None
        )
        if accuracy is None:
            errors.append(f"summary has invalid accuracy for {key}")
        else:
            accuracies.append(accuracy)

    average = finite_accuracy(summary.get("official_macro8_accuracy"))
    if average is None:
        errors.append("summary has invalid official_macro8_accuracy")
    elif len(accuracies) == len(OFFICIAL_METRICS):
        recomputed = sum(accuracies) / len(accuracies)
        if not math.isclose(average, recomputed, rel_tol=0.0, abs_tol=1e-12):
            errors.append(
                "official_macro8_accuracy does not equal the eight-type mean"
            )

    if errors:
        return {}, errors
    assert average is not None
    return {
        "result_kind": str(summary["result_kind"]),
        "accuracies": tuple(accuracies),
        "average": average,
    }, []


def inspect_summary(summary_path: Path, results_root: Path) -> ScoreResult:
    score_dir = summary_path.parent
    scorer_protocol = score_dir.name
    prediction_dir = score_dir.parent.parent
    predictions_path = prediction_dir / "predictions.jsonl"
    metadata_path = prediction_dir / "predictions.jsonl.metadata.json"
    result_id = score_dir.relative_to(results_root).as_posix()

    errors = complete_state_errors(
        predictions_path,
        score_dir,
        expected_protocol=scorer_protocol,
    )
    summary, summary_error = read_json(summary_path)
    if summary_error is not None:
        errors.append(f"summary.json is invalid: {summary_error}")
    metadata, metadata_error = read_json(metadata_path)
    if metadata_error is not None:
        errors.append(
            f"predictions.jsonl.metadata.json is invalid: {metadata_error}"
        )

    model_values, metadata_errors = metadata_values(
        metadata,
        prediction_dir=prediction_dir,
        scorer_protocol=scorer_protocol,
    )
    result_values, summary_errors = summary_values(
        summary,
        scorer_protocol=scorer_protocol,
    )
    errors.extend(metadata_errors)
    errors.extend(summary_errors)
    unique_errors = list(dict.fromkeys(errors))
    if unique_errors:
        return ScoreResult(
            result_id=result_id,
            summary_path=summary_path,
            predictions_path=predictions_path,
            metadata_path=metadata_path,
            score_dir=score_dir,
            eligible=False,
            reason="; ".join(unique_errors),
            model=model_values.get("model", ""),
            profile=model_values.get("profile", ""),
            model_revision=model_values.get("model_revision", ""),
            inference_protocol=model_values.get("inference_protocol", ""),
            input_profile=model_values.get("input_profile", ""),
            scorer_protocol=scorer_protocol,
            result_kind=(
                str(summary.get("result_kind", ""))
                if isinstance(summary, dict)
                else ""
            ),
        )

    return ScoreResult(
        result_id=result_id,
        summary_path=summary_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        score_dir=score_dir,
        eligible=True,
        reason="complete canonical artifacts and publication gates",
        model=model_values["model"],
        profile=model_values["profile"],
        model_revision=model_values["model_revision"],
        inference_protocol=model_values["inference_protocol"],
        input_profile=model_values["input_profile"],
        scorer_protocol=scorer_protocol,
        result_kind=result_values["result_kind"],
        accuracies=result_values["accuracies"],
        average=result_values["average"],
    )


def discover_results(results_root: Path) -> list[ScoreResult]:
    resolved_root = results_root.resolve()
    return [
        inspect_summary(path, resolved_root)
        for path in score_summary_paths(resolved_root)
    ]


def tsv_value(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def print_listing(results: Iterable[ScoreResult]) -> None:
    print(
        "eligible\tprofile\tmodel\tinference_protocol\t"
        "scorer_protocol\treason\tresult_id"
    )
    for result in results:
        print(
            "\t".join(
                (
                    "yes" if result.eligible else "no",
                    tsv_value(result.profile),
                    tsv_value(result.model),
                    tsv_value(result.inference_protocol),
                    tsv_value(result.scorer_protocol),
                    tsv_value(result.reason),
                    tsv_value(result.result_id),
                )
            )
        )


def selected_results(
    results: list[ScoreResult],
    *,
    profiles: list[str],
    scorer_protocols: list[str],
) -> list[ScoreResult]:
    profile_set = set(profiles)
    protocol_set = set(scorer_protocols)
    matched = [
        result
        for result in results
        if (not profile_set or result.profile in profile_set)
        and (
            not protocol_set
            or result.scorer_protocol in protocol_set
        )
    ]

    discovered_profiles = {result.profile for result in results if result.profile}
    missing_profiles = [
        profile for profile in profiles if profile not in discovered_profiles
    ]
    if missing_profiles:
        raise ConfigurationError(
            "requested profile(s) have no discovered score summary: "
            + ", ".join(missing_profiles)
        )

    discovered_protocols = {
        result.scorer_protocol for result in results if result.scorer_protocol
    }
    missing_protocols = [
        protocol
        for protocol in scorer_protocols
        if protocol not in discovered_protocols
    ]
    if missing_protocols:
        raise ConfigurationError(
            "requested scorer protocol(s) have no discovered score summary: "
            + ", ".join(missing_protocols)
        )

    if not matched:
        raise ConfigurationError("no score result matches the requested filters")

    reportable = [result for result in matched if result.eligible]
    if not reportable:
        ineligible = [result for result in matched if not result.eligible]
        rendered = "; ".join(
            f"{result.result_id}: {result.reason}" for result in ineligible
        )
        raise ConfigurationError(
            "no matching score result is reportable; run --list for details"
            + (f": {rendered}" if rendered else "")
        )

    profile_order = {profile: index for index, profile in enumerate(profiles)}
    protocol_order = {
        protocol: index for index, protocol in enumerate(scorer_protocols)
    }

    def key(result: ScoreResult) -> tuple[Any, ...]:
        if profiles:
            primary: tuple[Any, ...] = (
                profile_order[result.profile],
                result.model.casefold(),
            )
        else:
            primary = (result.model.casefold(), result.profile)
        return (
            *primary,
            protocol_order.get(result.scorer_protocol, len(protocol_order)),
            result.model_revision,
            result.inference_protocol,
            result.scorer_protocol,
            result.result_id,
        )

    return sorted(reportable, key=key)


def markdown_text(value: str, *, code: bool = False) -> str:
    escaped = html.escape(value, quote=False).replace("|", "&#124;")
    escaped = escaped.replace("\r\n", "<br>").replace("\n", "<br>")
    return f"<code>{escaped}</code>" if code else escaped


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}"


def presentation_model_names(results: list[ScoreResult]) -> list[str]:
    selected_profiles = {result.profile for result in results}
    names: list[str] = []
    for result in results:
        is_native = result.profile.endswith("_native") or (
            result.input_profile.endswith("_native")
        )
        has_native_sibling = f"{result.profile}_native" in selected_profiles
        is_fair_specialized = (
            not is_native
            and (
                has_native_sibling
                or result.input_profile == "rgb_only"
            )
        )
        if is_native:
            names.append(f"{result.model}（原生版）")
        elif is_fair_specialized:
            names.append(f"{result.model}（公平版）")
        else:
            names.append(result.model)
    return names


def render_markdown(results: list[ScoreResult]) -> str:
    scorer_protocols = {result.scorer_protocol for result in results}
    if len(scorer_protocols) != 1:
        raise ConfigurationError(
            "the concise report omits protocol columns and requires exactly "
            "one scorer protocol"
        )
    headers = [
        "模型名称",
        *(header for _key, header in OFFICIAL_METRICS),
        "平均",
    ]
    lines = [
        REPORT_TITLE,
        "",
        REPORT_NOTE,
        "",
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            ["---"] + ["---:"] * (len(OFFICIAL_METRICS) + 1)
        )
        + " |",
    ]
    for result, model_name in zip(
        results,
        presentation_model_names(results),
        strict=True,
    ):
        assert result.average is not None
        cells = [
            markdown_text(model_name),
            *(format_percentage(value) for value in result.accuracies),
            format_percentage(result.average),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        results_root = resolve_results_root(args.results_root)
        results = discover_results(results_root)
        if args.list_results:
            print_listing(results)
            return 0
        if not results:
            raise ConfigurationError(
                f"no score summary was discovered under {results_root}"
            )

        profiles = split_filters(args.profile)
        scorer_protocols = split_filters(args.scorer_protocol)
        if not scorer_protocols:
            scorer_protocols = [SCORER_PROTOCOL]
        if len(scorer_protocols) != 1:
            raise ConfigurationError(
                "the concise report requires exactly one scorer protocol"
            )
        selected = selected_results(
            results,
            profiles=profiles,
            scorer_protocols=scorer_protocols,
        )
        matched_ineligible = [
            result
            for result in results
            if not result.eligible
            and (not profiles or result.profile in set(profiles))
            and (
                not scorer_protocols
                or result.scorer_protocol in set(scorer_protocols)
            )
        ]
        if matched_ineligible:
            print(
                "[msmu-results-report] warning: skipped "
                f"{len(matched_ineligible)} non-reportable score summary/summaries; "
                "run --list for details",
                file=sys.stderr,
            )
        output = resolve_output_path(args.output, results_root)
        write_atomic(output, render_markdown(selected))
        print(
            f"[msmu-results-report] wrote {output} "
            f"with {len(selected)} result row(s)"
        )
        return 0
    except ConfigurationError as exc:
        print(
            f"[msmu-results-report] configuration error: {exc}",
            file=sys.stderr,
        )
        return 2
    except OSError as exc:
        print(f"[msmu-results-report] filesystem error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
