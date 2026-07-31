#!/usr/bin/env python3
"""Build a local-viewable Markdown audit of sampled MSMU stage-3 answers."""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

OFFICIAL_TEST_SIZE = 987
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_SEED = 20260730
DEFAULT_IMAGE_WIDTH = 240


@dataclass(frozen=True, slots=True)
class ModelSpec:
    display_name: str
    run_slug: str


@dataclass(frozen=True, slots=True)
class ModelResult:
    spec: ModelSpec
    predictions_path: Path
    rows_by_index: dict[int, dict[str, object]]


MODEL_SPECS = (
    ModelSpec("LLaVA-NeXT Mistral-7B", "llava-next-mistral-7b-vllm"),
    ModelSpec("LLaVA-NeXT Yi-34B", "llava-next-yi-34b-vllm"),
    ModelSpec("InternVL3-8B", "internvl3-8b-vllm"),
    ModelSpec("InternVL3-38B", "internvl3-38b-vllm"),
    ModelSpec("Qwen2.5-VL-7B", "qwen25-vl-base"),
    ModelSpec("Qwen2.5-VL-32B", "qwen25-vl-32b"),
    ModelSpec("SSR（公平轨 / RGB-only）", "ssr-rgb-only"),
    ModelSpec("SSR（原生轨 / DepthPro + MIDI + TOR）", "ssr-native"),
    ModelSpec("SpatialRGPT（RGB-only）", "spatialrgpt-rgb-only"),
    ModelSpec("3DThinker（公平轨）", "3dthinker-fair"),
    ModelSpec("3DThinker（原生 mental-3D）", "3dthinker-native"),
    ModelSpec("SpatialBot（公平轨 / RGB-only）", "spatialbot-rgb-only"),
    ModelSpec("SpatialBot（原生 ZoeDepth RGB-D）", "spatialbot-native"),
)


def select_sample_indices(
    *,
    population_size: int = OFFICIAL_TEST_SIZE,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if not 1 <= sample_size <= population_size:
        raise ValueError(
            f"sample_size must be within [1,{population_size}], got {sample_size}"
        )
    return sorted(random.Random(seed).sample(range(population_size), sample_size))


def load_prediction_rows(
    predictions_path: Path,
    *,
    expected_size: int = OFFICIAL_TEST_SIZE,
) -> dict[int, dict[str, object]]:
    rows_by_index: dict[int, dict[str, object]] = {}
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{predictions_path}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{predictions_path}:{line_number} is not a JSON object")
            index = row.get("index")
            if not isinstance(index, int):
                raise ValueError(
                    f"{predictions_path}:{line_number} has a non-integer index"
                )
            if index in rows_by_index:
                raise ValueError(f"{predictions_path} contains duplicate index {index}")
            if not isinstance(row.get("question"), str):
                raise ValueError(f"{predictions_path}:{line_number} has no string question")
            if not isinstance(row.get("prediction"), str):
                raise ValueError(f"{predictions_path}:{line_number} has no string prediction")
            rows_by_index[index] = row
    expected_indices = set(range(expected_size))
    actual_indices = set(rows_by_index)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise ValueError(
            f"{predictions_path} is not a complete {expected_size}-row result: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    return rows_by_index


def load_model_result(
    stage3_root: Path,
    spec: ModelSpec,
    *,
    expected_size: int = OFFICIAL_TEST_SIZE,
) -> ModelResult:
    run_root = stage3_root / spec.run_slug
    validations = sorted(run_root.rglob("prediction_validation.json"))
    if len(validations) != 1:
        raise ValueError(
            f"expected exactly one validator for {spec.display_name} under {run_root}, "
            f"found {len(validations)}"
        )
    validation_path = validations[0]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    required_values = {
        "passed": True,
        "allow_subset": False,
        "official_test_size": expected_size,
        "num_prediction_rows": expected_size,
        "num_unique_indices": expected_size,
    }
    mismatches = {
        key: (validation.get(key), expected)
        for key, expected in required_values.items()
        if validation.get(key) != expected
    }
    if validation.get("errors"):
        mismatches["errors"] = (validation.get("errors"), [])
    if mismatches:
        raise ValueError(
            f"{spec.display_name} has no accepted full validator: "
            f"{validation_path} mismatches={mismatches}"
        )
    predictions_value = validation.get("predictions")
    if not isinstance(predictions_value, str) or not predictions_value:
        raise ValueError(f"{validation_path} has no predictions path")
    predictions_path = Path(predictions_value).expanduser().resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"{spec.display_name} predictions are missing: {predictions_path}"
        )
    rows = load_prediction_rows(predictions_path, expected_size=expected_size)
    return ModelResult(spec=spec, predictions_path=predictions_path, rows_by_index=rows)


def fenced_text(text: str) -> str:
    longest_run = 0
    current_run = 0
    for character in text:
        if character == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}text\n{text}\n{fence}"


def render_markdown(
    model_results: list[ModelResult],
    sampled_indices: list[int],
    image_links: dict[int, str],
    *,
    seed: int,
    image_width: int = DEFAULT_IMAGE_WIDTH,
) -> str:
    lines = [
        "# 答案抽查",
        "",
        (
            f"> 从 MSMU official test 987 条中以固定随机种子 `{seed}` "
            f"抽取同一组 {len(sampled_indices)} 条，供 {len(model_results)} 个模型/轨道横向抽查。"
        ),
        "> 图片为随文档复制的本地相对链接；点击小图可以打开完整尺寸图片。",
        "",
        f"> 抽中 MSMU index：`{','.join(str(index) for index in sampled_indices)}`",
        "",
    ]
    for result in model_results:
        lines.extend([f"## {result.spec.display_name}", ""])
        for ordinal, index in enumerate(sampled_indices, start=1):
            row = result.rows_by_index[index]
            question = str(row["question"])
            prediction = str(row["prediction"])
            image_link = image_links[index]
            escaped_link = html.escape(quote(image_link, safe="/._-"), quote=True)
            alt = html.escape(f"MSMU index {index}", quote=True)
            lines.extend(
                [
                    f"### {ordinal}（MSMU index {index}）",
                    "",
                    "**完整题干**",
                    "",
                    fenced_text(question),
                    "",
                    "**题目图片**",
                    "",
                    (
                        f'<a href="{escaped_link}">'
                        f'<img src="{escaped_link}" alt="{alt}" width="{image_width}">'
                        "</a>"
                    ),
                    "",
                    "**完整回答**",
                    "",
                    fenced_text(prediction),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the same MSMU indices across every accepted stage-3 model, "
            "export local image assets, and build one Markdown answer-audit document."
        )
    )
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage3_root = args.stage3_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    assets_path = (
        args.assets_dir.expanduser().resolve()
        if args.assets_dir
        else output_path.with_name(f"{output_path.stem}-assets")
    )
    if not stage3_root.is_dir():
        raise FileNotFoundError(f"stage-3 root does not exist: {stage3_root}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")
    if assets_path.exists():
        raise FileExistsError(f"refusing to overwrite existing assets: {assets_path}")
    if args.image_width <= 0:
        raise ValueError("image-width must be positive")

    sampled_indices = select_sample_indices(
        sample_size=args.sample_size,
        seed=args.seed,
    )
    model_results = [
        load_model_result(stage3_root, spec)
        for spec in MODEL_SPECS
    ]

    # Import the benchmark dataset stack only for an actual server-side run so
    # --help and unit tests do not require the heavy inference environment.
    from spatial_vlm_eval.benchmarks.msmu.data import MSMUTestContract

    contract = MSMUTestContract(dataset_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".msmu-answer-audit-", dir=output_path.parent)
    )
    temporary_assets = temporary_root / assets_path.name
    temporary_assets.mkdir()
    try:
        image_links: dict[int, str] = {}
        for index in sampled_indices:
            model_input = contract.model_input(index)
            expected_question = model_input.question
            for result in model_results:
                actual_question = str(result.rows_by_index[index]["question"])
                if actual_question != expected_question:
                    raise ValueError(
                        f"{result.spec.display_name} question mismatch at index {index}"
                    )
            image_name = f"msmu-{index:04d}.jpg"
            image_path = temporary_assets / image_name
            model_input.image.convert("RGB").save(
                image_path,
                format="JPEG",
                quality=92,
                optimize=True,
            )
            final_image_path = assets_path / image_name
            relative_link = os.path.relpath(final_image_path, output_path.parent)
            image_links[index] = Path(relative_link).as_posix()

        markdown = render_markdown(
            model_results,
            sampled_indices,
            image_links,
            seed=args.seed,
            image_width=args.image_width,
        )
        temporary_report = temporary_root / output_path.name
        temporary_report.write_text(markdown, encoding="utf-8")
        temporary_assets.replace(assets_path)
        temporary_report.replace(output_path)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    print(f"[msmu-audit] models={len(model_results)}")
    print(f"[msmu-audit] samples_per_model={len(sampled_indices)}")
    print(f"[msmu-audit] unique_images={len(sampled_indices)}")
    print(f"[msmu-audit] indices={','.join(str(index) for index in sampled_indices)}")
    print(f"[msmu-audit] report={output_path}")
    print(f"[msmu-audit] assets={assets_path}")


if __name__ == "__main__":
    main()
