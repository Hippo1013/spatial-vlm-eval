"""Locked CV-Bench data ownership and restricted one-image model inputs."""

from __future__ import annotations

import hashlib
import json
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...models.common.runtime import pixel_sha256

DATASET_REPOSITORY = "nyu-visionx/CV-Bench"
DATASET_REVISION = "bc284db50d036958861cb60cdd7b77612052ce0d"
OFFICIAL_2D_SIZE = 1438
OFFICIAL_3D_SIZE = 1200
OFFICIAL_TEST_SIZE = OFFICIAL_2D_SIZE + OFFICIAL_3D_SIZE
SMOKE8_INDICES = (0, 633, 342, 1080, 1438, 1442, 2038, 2042)

REQUIRED_SOURCE_FIELDS = frozenset(
    {
        "type",
        "task",
        "image",
        "question",
        "choices",
        "answer",
        "prompt",
        "filename",
        "source",
        "source_dataset",
        "source_filename",
        "target_class",
        "target_size",
        "bbox",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetFile:
    name: str
    split_type: str
    rows: int
    sha256: str
    size_bytes: int


DATASET_FILES = (
    DatasetFile(
        name="test_2d.parquet",
        split_type="2D",
        rows=OFFICIAL_2D_SIZE,
        sha256="33196034ef4bf3265cae4a7ff5c4071b2ff1cc21123e8e285c6a91393897ecbc",
        size_bytes=184_906_137,
    ),
    DatasetFile(
        name="test_3d.parquet",
        split_type="3D",
        rows=OFFICIAL_3D_SIZE,
        sha256="ef91fe8b5392eb2a16e318ca68fa02449d45ba1e152afece12a0a526e9fbbc25",
        size_bytes=219_902_227,
    ),
)

TASK_DISPLAY_NAMES = {
    "Relation": "Spatial Relationship",
    "Count": "Object Count",
    "Depth": "Depth Order",
    "Distance": "Relative Distance",
}
EXPECTED_TASK_COUNTS = {
    "Relation": 650,
    "Count": 788,
    "Depth": 600,
    "Distance": 600,
}
EXPECTED_SOURCE_COUNTS = {"ADE20K": 633, "COCO": 805, "Omni3D": 1200}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def answer_letter(answer: Any) -> str:
    text = str(answer).strip().upper()
    if len(text) == 3 and text[0] == "(" and text[2] == ")":
        text = text[1]
    if len(text) != 1 or text not in string.ascii_uppercase:
        raise ValueError(f"CV-Bench answer is not one parenthesized option letter: {answer!r}")
    return text


def dataset_prompt(row: Mapping[str, Any]) -> str:
    """Validate and return the dataset-owned question and ordered choices."""

    question = str(row.get("question") or "").strip()
    prompt = str(row.get("prompt") or "").strip()
    choices = row.get("choices")
    if not question or not prompt:
        raise ValueError("CV-Bench question and prompt must be non-empty")
    if not prompt.startswith(question):
        raise ValueError("CV-Bench prompt does not begin with its question")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ValueError("CV-Bench choices must be a non-empty sequence")
    if len(choices) > 26:
        raise ValueError("CV-Bench supports at most 26 choices")
    cursor = len(question)
    for position, choice in enumerate(choices):
        rendered = f"({string.ascii_uppercase[position]}) {str(choice).strip()}"
        located = prompt.find(rendered, cursor)
        if located < 0:
            raise ValueError(f"CV-Bench prompt is missing ordered choice {rendered!r}")
        cursor = located + len(rendered)
    gold = answer_letter(row.get("answer"))
    if string.ascii_uppercase.index(gold) >= len(choices):
        raise ValueError(f"CV-Bench gold option {gold} is outside {len(choices)} choices")
    return prompt


def _as_row(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else dict(value)


def _validate_source_row(row: Mapping[str, Any], *, index: int) -> None:
    missing = sorted(REQUIRED_SOURCE_FIELDS - set(row))
    if missing:
        raise ValueError(f"CV-Bench index {index} is missing source fields: {missing}")
    split_type = str(row["type"])
    task = str(row["task"])
    source = str(row["source"])
    allowed = {
        ("2D", "Relation", "ADE20K"),
        ("2D", "Relation", "COCO"),
        ("2D", "Count", "ADE20K"),
        ("2D", "Count", "COCO"),
        ("3D", "Depth", "Omni3D"),
        ("3D", "Distance", "Omni3D"),
    }
    if (split_type, task, source) not in allowed:
        raise ValueError(
            f"CV-Bench index {index} has unsupported type/task/source "
            f"{(split_type, task, source)!r}"
        )
    dataset_prompt(row)
    image = row.get("image")
    if image is None or not hasattr(image, "convert"):
        raise ValueError(f"CV-Bench index {index} has no PIL-compatible image")


def _load_parquet(path: Path) -> Any:
    from datasets import Dataset

    return Dataset.from_parquet(str(path))


@dataclass(frozen=True, slots=True)
class CVBenchModelInput:
    """The complete object visible to adapters: one RGB image and one safe prompt."""

    index: int
    image: Any
    question: str

    @property
    def prompt(self) -> str:
        return self.question


class CVBenchTestContract:
    """Own all source rows while exposing only prompt-and-image model inputs."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split_datasets: Mapping[str, Any] | None = None,
        require_official_size: bool = True,
        verify_files: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        if split_datasets is None:
            loaded: dict[str, Any] = {}
            for specification in DATASET_FILES:
                path = self.dataset_root / specification.name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing locked CV-Bench file: {path}")
                if verify_files:
                    if path.stat().st_size != specification.size_bytes:
                        raise ValueError(
                            f"{path.name} size is {path.stat().st_size}, expected {specification.size_bytes}"
                        )
                    actual = sha256_file(path)
                    if actual != specification.sha256:
                        raise ValueError(
                            f"{path.name} SHA-256 is {actual}, expected {specification.sha256}"
                        )
                loaded[specification.split_type] = _load_parquet(path)
            split_datasets = loaded
        if set(split_datasets) != {"2D", "3D"}:
            raise ValueError("CV-Bench contract requires exactly the 2D and 3D datasets")
        self.__splits = {name: split_datasets[name] for name in ("2D", "3D")}
        sizes = {name: len(dataset) for name, dataset in self.__splits.items()}
        if require_official_size and sizes != {"2D": OFFICIAL_2D_SIZE, "3D": OFFICIAL_3D_SIZE}:
            raise ValueError(
                "CV-Bench locked sizes are 2D=1438 and 3D=1200; "
                f"found 2D={sizes['2D']} and 3D={sizes['3D']}"
            )
        self.__offsets = {"2D": 0, "3D": sizes["2D"]}
        self.__size = sizes["2D"] + sizes["3D"]
        self.__official_files_verified = split_datasets is not None and verify_files
        file_identity = {
            specification.name: specification.sha256 for specification in DATASET_FILES
        }
        self.__dataset_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "repository": DATASET_REPOSITORY,
                    "revision": DATASET_REVISION,
                    "files": file_identity,
                    "sizes": sizes,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return self.__size

    @property
    def dataset_fingerprint(self) -> str:
        return self.__dataset_fingerprint

    def _source_row(self, index: int) -> dict[str, Any]:
        resolved = int(index)
        if not 0 <= resolved < len(self):
            raise IndexError(f"CV-Bench index {resolved} outside [0,{len(self)})")
        if resolved < self.__offsets["3D"]:
            split_type, local_index = "2D", resolved
        else:
            split_type, local_index = "3D", resolved - self.__offsets["3D"]
        row = _as_row(self.__splits[split_type][local_index])
        _validate_source_row(row, index=resolved)
        if str(row["type"]) != split_type:
            raise ValueError(
                f"CV-Bench index {resolved} is stored in {split_type} but declares {row['type']!r}"
            )
        if "idx" in row and row["idx"] is not None and int(row["idx"]) != resolved:
            raise ValueError(
                f"CV-Bench index field mismatch at global index {resolved}: {row['idx']!r}"
            )
        return row

    def model_input(self, index: int) -> CVBenchModelInput:
        resolved = int(index)
        row = self._source_row(resolved)
        return CVBenchModelInput(
            index=resolved,
            image=row["image"].convert("RGB"),
            question=dataset_prompt(row),
        )

    def model_inputs(self, indices: Iterable[int]) -> list[CVBenchModelInput]:
        return [self.model_input(index) for index in indices]

    def prediction_row(self, index: int, prediction: str) -> dict[str, Any]:
        self._source_row(int(index))
        return {"index": int(index), "raw_prediction": str(prediction)}

    def scoring_row(self, index: int) -> dict[str, Any]:
        row = self._source_row(index)
        return {
            "index": int(index),
            "type": str(row["type"]),
            "task": str(row["task"]),
            "source": str(row["source"]),
            "choices": [str(choice) for choice in row["choices"]],
            "gold": answer_letter(row["answer"]),
        }

    def dataset_manifest(self, *, include_images: bool = True) -> dict[str, Any]:
        task_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        schema_by_split: dict[str, list[str]] = {}
        prompt_digest = hashlib.sha256()
        image_digest = hashlib.sha256()
        for split_type, dataset in self.__splits.items():
            column_names = getattr(dataset, "column_names", None)
            if column_names is None and len(dataset):
                column_names = sorted(_as_row(dataset[0]))
            schema_by_split[split_type] = sorted(str(name) for name in (column_names or []))
        for index in range(len(self)):
            row = self._source_row(index)
            task_counts[str(row["task"])] += 1
            source_counts[str(row["source"])] += 1
            prompt_digest.update(f"{index}\0{dataset_prompt(row)}\0".encode("utf-8"))
            if include_images:
                image = row["image"].convert("RGB")
                image_digest.update(
                    f"{index}\0{image.size[0]}x{image.size[1]}\0{pixel_sha256(image)}\0".encode(
                        "ascii"
                    )
                )
        manifest = {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "files": [
                {
                    "name": item.name,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                    "rows": item.rows,
                }
                for item in DATASET_FILES
            ],
            "schema_by_split": schema_by_split,
            "split_counts": {"2D": len(self.__splits["2D"]), "3D": len(self.__splits["3D"])},
            "task_counts": dict(sorted(task_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "prompt_sha256": prompt_digest.hexdigest(),
            "image_pixel_manifest_sha256": image_digest.hexdigest() if include_images else None,
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        if len(self) == OFFICIAL_TEST_SIZE:
            if dict(task_counts) != EXPECTED_TASK_COUNTS:
                raise ValueError(
                    f"CV-Bench task counts differ from the locked dataset: {dict(task_counts)}"
                )
            if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
                raise ValueError(
                    f"CV-Bench source counts differ from the locked dataset: {dict(source_counts)}"
                )
        return manifest
