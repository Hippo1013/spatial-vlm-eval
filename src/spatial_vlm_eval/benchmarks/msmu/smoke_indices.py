"""Select a deterministic eight-type MSMU debug subset without exposing labels to models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import OFFICIAL_TEST_SIZE, load_arrow_split, official_type_for_raw_type

OFFICIAL_TYPE_ORDER = (
    "scale_estimation",
    "absolute_distance",
    "count",
    "grounding",
    "refer_obj_estimation",
    "relative_position",
    "scale_compare",
    "existence",
)


def select_type_covering_indices(test: Any) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for index in range(len(test)):
        raw_type = str(test[index].get("type") or "")
        official_type = official_type_for_raw_type(raw_type)
        if official_type not in selected:
            selected[official_type] = {
                "index": index,
                "official_type": official_type,
                "raw_type": raw_type,
            }
        if len(selected) == len(OFFICIAL_TYPE_ORDER):
            break
    missing = [official_type for official_type in OFFICIAL_TYPE_ORDER if official_type not in selected]
    if missing:
        raise ValueError(f"MSMU split cannot cover all official types; missing={missing}")
    return [selected[official_type] for official_type in OFFICIAL_TYPE_ORDER]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--format", choices=["csv", "json"], default="json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test = load_arrow_split(args.dataset_root, "test")
    if len(test) != OFFICIAL_TEST_SIZE:
        raise ValueError(
            f"MSMU smoke selection requires the official {OFFICIAL_TEST_SIZE}-row test split; got {len(test)}"
        )
    selections = select_type_covering_indices(test)
    report = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": "test",
        "official_test_size": OFFICIAL_TEST_SIZE,
        "selections": selections,
        "indices_csv": ",".join(str(item["index"]) for item in selections),
        "debug_subset_only": True,
    }
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report["indices_csv"] if args.format == "csv" else json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
