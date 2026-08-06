"""Locked Q-Spatial data ownership and restricted system/user one-image inputs."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...models.common.runtime import pixel_sha256

DATASET_REPOSITORY = "andrewliao11/Q-Spatial-Bench"
DATASET_REVISION = "17b92e470d58fa46859ebd48ff35a1669828c9be"
UPSTREAM_REPOSITORY = "andrewliao11/Q-Spatial-Bench-code"
UPSTREAM_COMMIT = "ebe8137eae9781aaf7e29691ce8bc68b2a498a83"

OFFICIAL_SCANNET_SIZE = 170
OFFICIAL_PLUS_SIZE = 101
OFFICIAL_TEST_SIZE = OFFICIAL_SCANNET_SIZE + OFFICIAL_PLUS_SIZE
SMOKE8_INDICES = (0, 1, 3, 9, 14, 205, 247, 250)

STANDARD_SYSTEM_PROMPT = (
    "You will be provided with a question and a 2D image. The question involves measuring "
    "the precise distance in 3D space through a 2D image. You will answer the question by "
    "providing a numeric answer consisting of a scalar and a distance unit in the format of "
    '"""\\scalar{scalar} \\distance_unit{distance unit}""" at the end of your response.'
)
STANDARD_SYSTEM_PROMPT_SHA256 = "b3da32feb428a7840ecaf1d08ef095b9cd72ff6ef34d5b2b05ec1c1599bb613c"
if hashlib.sha256(STANDARD_SYSTEM_PROMPT.encode("utf-8")).hexdigest() != STANDARD_SYSTEM_PROMPT_SHA256:
    raise RuntimeError("Q-Spatial Standard system prompt bytes changed without a protocol update")
LLAVA_USER_FORMAT_SUFFIX = (
    "Answer by providing a numeric answer consisting of a scalar and a distance unit in the "
    'format of """\\scalar{scalar} \\distance_unit{distance unit}""" at the end of your response.'
)
LLAVA_CONTINUATION = ' In conclusion, the final answer in the specified format is: """\\scalar{'
REQUIRED_SOURCE_FIELDS = frozenset(
    {"question", "answer_value", "answer_unit", "question_type", "image_path", "image"}
)


@dataclass(frozen=True, slots=True)
class DatasetFile:
    name: str
    split: str
    rows: int
    sha256: str
    size_bytes: int


DATASET_FILES = (
    DatasetFile(
        "data/QSpatial_scannet-00000-of-00001.parquet",
        "QSpatial_scannet",
        OFFICIAL_SCANNET_SIZE,
        "a5b0a37443b4ae18c837e4df7fe60411f869f282aa5803b8a7d509ba381286ba",
        12_022,
    ),
    DatasetFile(
        "data/QSpatial_plus-00000-of-00001.parquet",
        "QSpatial_plus",
        OFFICIAL_PLUS_SIZE,
        "30ff075480f7fe0497122c8251f5d529f2241dda1387038e2a0ed802ae8615e2",
        129_408_418,
    ),
)

EXPECTED_SCHEMA = tuple(sorted(REQUIRED_SOURCE_FIELDS))
EXPECTED_PARQUET_SCHEMA = (
    ("question", "string"),
    ("answer_value", "float"),
    ("answer_unit", "string"),
    ("question_type", "string"),
    ("image_path", "string"),
    ("image", "struct<bytes: binary, path: string>"),
)
EXPECTED_SPLIT_TYPE_COUNTS = {
    "QSpatial_scannet": {
        "direct_distance": 36,
        "horizontal_distance": 60,
        "object_height": 22,
        "object_width": 23,
        "vertical_distance": 29,
    },
    "QSpatial_plus": {
        "1d_horizontal": 1,
        "horizontal_distance": 98,
        "vertical_distance": 2,
    },
}
EXPECTED_SPLIT_UNIT_COUNTS = {
    "QSpatial_scannet": {"centimeter": 129, "meter": 41},
    "QSpatial_plus": {"centimeter": 94, "meter": 7},
}
EXPECTED_DISTINCT_IMAGE_COUNTS = {"QSpatial_scannet": 99, "QSpatial_plus": 87}
SCANNET_SCENE_COUNT = 66
SCANNET_FRAME_COUNT = 99
SCANNET_FILE_MANIFEST_SHA256 = (
    "4485132ff448f43bdfb1283743825995823487a37f74ae4ab5a8e9d4b653751b"
)

_SCANNET_ID_AND_FRAMES = {
    "scene0015_00": (0,),
    "scene0019_00": (400,),
    "scene0025_00": (500,),
    "scene0025_02": (400,),
    "scene0030_00": (2300,),
    "scene0030_02": (900,),
    "scene0050_01": (0,),
    "scene0084_00": (600,),
    "scene0144_00": (0, 700),
    "scene0164_00": (1600, 1700, 800),
    "scene0169_00": (0,),
    "scene0169_01": (1000,),
    "scene0193_01": (200, 400),
    "scene0217_00": (0, 1100, 400),
    "scene0222_00": (4800,),
    "scene0257_00": (1200, 300),
    "scene0278_00": (0, 300),
    "scene0278_01": (0,),
    "scene0304_00": (1500,),
    "scene0329_00": (0,),
    "scene0329_02": (1000,),
    "scene0351_00": (0,),
    "scene0353_00": (2000,),
    "scene0353_01": (100, 2100),
    "scene0378_00": (0, 1800),
    "scene0378_01": (0, 1500, 900),
    "scene0406_02": (800,),
    "scene0423_01": (0,),
    "scene0427_00": (900,),
    "scene0430_00": (2300,),
    "scene0458_01": (0,),
    "scene0462_00": (300,),
    "scene0488_00": (100,),
    "scene0535_00": (400,),
    "scene0553_02": (700, 900),
    "scene0580_00": (0,),
    "scene0591_01": (1000, 1300, 1700, 300, 1400),
    "scene0593_01": (400,),
    "scene0595_00": (0,),
    "scene0598_01": (600,),
    "scene0599_00": (300,),
    "scene0599_02": (2100,),
    "scene0606_00": (0,),
    "scene0608_01": (1900,),
    "scene0608_02": (0, 2300),
    "scene0616_00": (0,),
    "scene0616_01": (1400, 2200),
    "scene0621_00": (2300,),
    "scene0629_00": (1400,),
    "scene0633_00": (300,),
    "scene0643_00": (0, 100, 1500, 300, 500),
    "scene0644_00": (0, 900),
    "scene0645_01": (1900, 2900),
    "scene0647_00": (600, 500),
    "scene0647_01": (800, 200),
    "scene0651_00": (400,),
    "scene0653_01": (0, 3900, 4700),
    "scene0678_00": (2400, 700),
    "scene0678_01": (0, 2100, 800),
    "scene0678_02": (1300, 700),
    "scene0696_01": (900,),
    "scene0701_01": (800,),
    "scene0025_01": (1500,),
    "scene0300_00": (0,),
    "scene0406_00": (700,),
    "scene0583_00": (100,),
}
EXPECTED_SCANNET_FRAMES = tuple(
    sorted(
        f"{scene}/color/{frame}.jpg"
        for scene, frames in _SCANNET_ID_AND_FRAMES.items()
        for frame in frames
    )
)
if len(_SCANNET_ID_AND_FRAMES) != SCANNET_SCENE_COUNT or len(EXPECTED_SCANNET_FRAMES) != SCANNET_FRAME_COUNT:
    raise RuntimeError("Locked ScanNet scene/frame registry is incomplete")

CANONICAL_TYPES = {
    "object_width": "object_width",
    "object_height": "object_height",
    "horizontal_distance": "horizontal_distance",
    "vertical_distance": "vertical_distance",
    "direct_distance": "direct_distance",
    "1d_horizontal": "object_width",
}
SCANNET_CANONICAL_TYPES = (
    "object_width",
    "object_height",
    "horizontal_distance",
    "vertical_distance",
    "direct_distance",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def user_prompt(question: Any) -> str:
    text = str(question).strip()
    if not text:
        raise ValueError("Q-Spatial question must be non-empty")
    return f"Question: {text}"


def _as_row(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else dict(value)


def _load_parquet(path: Path) -> Any:
    from datasets import Dataset

    return Dataset.from_parquet(str(path))


def _parquet_schema(path: Path) -> tuple[tuple[str, str], ...]:
    import pyarrow.parquet as parquet

    return tuple((field.name, str(field.type)) for field in parquet.ParquetFile(path).schema_arrow)


def _embedded_image(value: Any) -> Any:
    if value is not None and hasattr(value, "convert"):
        return value.convert("RGB")
    if isinstance(value, Mapping) and isinstance(value.get("bytes"), (bytes, bytearray)):
        from PIL import Image

        with Image.open(io.BytesIO(bytes(value["bytes"]))) as loaded:
            image = loaded.convert("RGB")
            image.load()
        return image
    raise ValueError("Q-Spatial++ row has no decodable embedded image")


def _validate_source_row(row: Mapping[str, Any], *, split: str, index: int) -> None:
    missing = sorted(REQUIRED_SOURCE_FIELDS - set(row))
    if missing:
        raise ValueError(f"Q-Spatial index {index} is missing source fields: {missing}")
    if not str(row["question"]).strip():
        raise ValueError(f"Q-Spatial index {index} has an empty question")
    raw_type = str(row["question_type"])
    if raw_type not in EXPECTED_SPLIT_TYPE_COUNTS[split]:
        raise ValueError(f"Q-Spatial index {index} has unsupported raw type {raw_type!r}")
    unit = str(row["answer_unit"])
    if unit not in {"centimeter", "meter"}:
        raise ValueError(f"Q-Spatial index {index} has unsupported ground-truth unit {unit!r}")
    try:
        value = float(row["answer_value"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Q-Spatial index {index} has invalid answer value") from exc
    if not value > 0:
        raise ValueError(f"Q-Spatial index {index} has non-positive answer value")
    image_path = str(row["image_path"])
    prefix = f"{split}/images/"
    if not image_path.startswith(prefix):
        raise ValueError(
            f"Q-Spatial index {index} image path {image_path!r} does not begin with {prefix!r}"
        )


@dataclass(frozen=True, slots=True)
class QSpatialModelInput:
    """The complete safe object visible to an adapter."""

    index: int
    image: Any
    system_prompt: str
    user_prompt: str

    @property
    def question(self) -> str:
        """Compatibility alias for model-neutral runtime auditing only."""

        return self.user_prompt


class QSpatialTestContract:
    """Own private labels while exposing one RGB and the official Standard Prompt."""

    def __init__(
        self,
        parquet_root: str | Path,
        scannet_rgb_root: str | Path,
        *,
        split_datasets: Mapping[str, Any] | None = None,
        require_official_size: bool = True,
        verify_files: bool = True,
    ) -> None:
        self.parquet_root = Path(parquet_root).expanduser().absolute()
        self.scannet_rgb_root = Path(scannet_rgb_root).expanduser().absolute()
        self.__scannet_rgb_real_root = self.scannet_rgb_root.resolve()
        self.dataset_root = self.parquet_root
        parquet_schemas: dict[str, tuple[tuple[str, str], ...]] = {}
        if split_datasets is None:
            loaded: dict[str, Any] = {}
            for specification in DATASET_FILES:
                path = self.parquet_root / specification.name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing locked Q-Spatial file: {path}")
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
                    actual_schema = _parquet_schema(path)
                    if actual_schema != EXPECTED_PARQUET_SCHEMA:
                        raise ValueError(
                            f"{path.name} schema is {actual_schema}, expected {EXPECTED_PARQUET_SCHEMA}"
                        )
                    parquet_schemas[specification.split] = actual_schema
                loaded[specification.split] = _load_parquet(path)
            split_datasets = loaded
        expected_splits = {"QSpatial_scannet", "QSpatial_plus"}
        if set(split_datasets) != expected_splits:
            raise ValueError("Q-Spatial contract requires exactly QSpatial_scannet and QSpatial_plus")
        self.__splits = {
            name: split_datasets[name] for name in ("QSpatial_scannet", "QSpatial_plus")
        }
        sizes = {name: len(dataset) for name, dataset in self.__splits.items()}
        expected_sizes = {
            "QSpatial_scannet": OFFICIAL_SCANNET_SIZE,
            "QSpatial_plus": OFFICIAL_PLUS_SIZE,
        }
        if require_official_size and sizes != expected_sizes:
            raise ValueError(f"Q-Spatial locked sizes differ: got={sizes}, expected={expected_sizes}")
        self.__sizes = sizes
        self.__parquet_schemas = parquet_schemas
        self.__size = sum(sizes.values())
        file_identity = {item.name: item.sha256 for item in DATASET_FILES}
        self.__dataset_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "repository": DATASET_REPOSITORY,
                    "revision": DATASET_REVISION,
                    "files": file_identity,
                    "sizes": sizes,
                    "scannet_file_manifest": SCANNET_FILE_MANIFEST_SHA256,
                    "system_prompt_sha256": STANDARD_SYSTEM_PROMPT_SHA256,
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

    def _resolve(self, index: int) -> tuple[str, int]:
        resolved = int(index)
        if not 0 <= resolved < len(self):
            raise IndexError(f"Q-Spatial index {resolved} outside [0,{len(self)})")
        scan_size = self.__sizes["QSpatial_scannet"]
        return (
            ("QSpatial_scannet", resolved)
            if resolved < scan_size
            else ("QSpatial_plus", resolved - scan_size)
        )

    def _source_row(self, index: int) -> tuple[str, dict[str, Any]]:
        split, local_index = self._resolve(index)
        row = _as_row(self.__splits[split][local_index])
        _validate_source_row(row, split=split, index=int(index))
        return split, row

    def _scannet_image(self, image_path: str) -> Any:
        from PIL import Image

        prefix = "QSpatial_scannet/images/"
        relative = Path(image_path[len(prefix) :])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe Q-Spatial ScanNet image path: {image_path!r}")
        path = (self.scannet_rgb_root / relative).resolve()
        try:
            path.relative_to(self.__scannet_rgb_real_root)
        except ValueError as exc:
            raise ValueError(f"ScanNet image escaped configured root: {image_path!r}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing Q-Spatial ScanNet RGB frame: {path}")
        with Image.open(path) as loaded:
            image = loaded.convert("RGB")
            image.load()
        return image

    def _row_image(self, split: str, row: Mapping[str, Any]) -> Any:
        if split == "QSpatial_scannet":
            return self._scannet_image(str(row["image_path"]))
        return _embedded_image(row["image"])

    def model_input(self, index: int) -> QSpatialModelInput:
        resolved = int(index)
        split, row = self._source_row(resolved)
        return QSpatialModelInput(
            index=resolved,
            image=self._row_image(split, row),
            system_prompt=STANDARD_SYSTEM_PROMPT,
            user_prompt=user_prompt(row["question"]),
        )

    def model_inputs(self, indices: Iterable[int]) -> list[QSpatialModelInput]:
        return [self.model_input(index) for index in indices]

    def prediction_row(self, index: int, prediction: str) -> dict[str, Any]:
        self._source_row(int(index))
        return {"index": int(index), "raw_prediction": str(prediction)}

    def scoring_row(self, index: int) -> dict[str, Any]:
        split, row = self._source_row(index)
        raw_type = str(row["question_type"])
        return {
            "index": int(index),
            "split": split,
            "raw_type": raw_type,
            "canonical_type": CANONICAL_TYPES[raw_type],
            "answer_value": str(row["answer_value"]),
            "answer_unit": str(row["answer_unit"]),
        }

    def _scannet_file_manifest(self) -> tuple[str, list[str]]:
        files = sorted(self.scannet_rgb_root.rglob("*.jpg"))
        relative = [str(path.relative_to(self.scannet_rgb_root)) for path in files]
        digest = hashlib.sha256()
        for path, name in zip(files, relative):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest(), relative

    def dataset_manifest(self, *, include_images: bool = True) -> dict[str, Any]:
        schema_by_split: dict[str, list[str]] = {}
        type_counts: dict[str, Counter[str]] = {}
        unit_counts: dict[str, Counter[str]] = {}
        image_paths: dict[str, set[str]] = {}
        prompt_digest = hashlib.sha256()
        record_digest = hashlib.sha256()
        pixel_digest = hashlib.sha256()
        for split, dataset in self.__splits.items():
            columns = getattr(dataset, "column_names", None)
            if columns is None and len(dataset):
                columns = sorted(_as_row(dataset[0]))
            schema_by_split[split] = sorted(str(value) for value in (columns or []))
            type_counts[split] = Counter()
            unit_counts[split] = Counter()
            image_paths[split] = set()
        for index in range(len(self)):
            split, row = self._source_row(index)
            raw_type = str(row["question_type"])
            unit = str(row["answer_unit"])
            path = str(row["image_path"])
            type_counts[split][raw_type] += 1
            unit_counts[split][unit] += 1
            image_paths[split].add(path)
            prompt = user_prompt(row["question"])
            prompt_digest.update(f"{index}\0{prompt}\0".encode("utf-8"))
            record_digest.update(
                f"{index}\0{split}\0{row['question']}\0{row['answer_value']}\0{unit}\0{raw_type}\0{path}\0".encode(
                    "utf-8"
                )
            )
            if include_images:
                image = self._row_image(split, row)
                pixel_digest.update(
                    f"{index}\0{image.size[0]}x{image.size[1]}\0{pixel_sha256(image)}\0".encode(
                        "ascii"
                    )
                )
        manifest = {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "files": [
                {
                    "name": item.name,
                    "split": item.split,
                    "rows": item.rows,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in DATASET_FILES
            ],
            "roots": {
                "parquet": str(self.parquet_root),
                "scannet_rgb": str(self.scannet_rgb_root),
            },
            "schema_by_split": schema_by_split,
            "parquet_schema_by_split": {
                split: [list(field) for field in schema]
                for split, schema in self.__parquet_schemas.items()
            },
            "split_counts": dict(self.__sizes),
            "type_counts": {key: dict(sorted(value.items())) for key, value in type_counts.items()},
            "unit_counts": {key: dict(sorted(value.items())) for key, value in unit_counts.items()},
            "distinct_image_counts": {key: len(value) for key, value in image_paths.items()},
            "system_prompt_sha256": STANDARD_SYSTEM_PROMPT_SHA256,
            "user_prompt_manifest_sha256": prompt_digest.hexdigest(),
            "record_manifest_sha256": record_digest.hexdigest(),
            "image_pixel_manifest_sha256": pixel_digest.hexdigest() if include_images else None,
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        if len(self) == OFFICIAL_TEST_SIZE:
            if schema_by_split != {
                "QSpatial_scannet": list(EXPECTED_SCHEMA),
                "QSpatial_plus": list(EXPECTED_SCHEMA),
            }:
                raise ValueError(f"Q-Spatial schema differs from locked schema: {schema_by_split}")
            if {key: dict(value) for key, value in type_counts.items()} != EXPECTED_SPLIT_TYPE_COUNTS:
                raise ValueError("Q-Spatial question-type distribution differs from locked data")
            if {key: dict(value) for key, value in unit_counts.items()} != EXPECTED_SPLIT_UNIT_COUNTS:
                raise ValueError("Q-Spatial unit distribution differs from locked data")
            if manifest["distinct_image_counts"] != EXPECTED_DISTINCT_IMAGE_COUNTS:
                raise ValueError("Q-Spatial distinct-image distribution differs from locked data")
            scannet_paths = {
                path.removeprefix("QSpatial_scannet/images/")
                for path in image_paths["QSpatial_scannet"]
            }
            if scannet_paths != set(EXPECTED_SCANNET_FRAMES):
                raise ValueError("Q-Spatial ScanNet dataset rows do not reference the locked 99 frames")
            if include_images:
                digest, external_paths = self._scannet_file_manifest()
                if external_paths != list(EXPECTED_SCANNET_FRAMES):
                    raise ValueError("Configured ScanNet RGB root is not the locked 99-frame directory")
                if digest != SCANNET_FILE_MANIFEST_SHA256:
                    raise ValueError(
                        f"ScanNet RGB manifest digest is {digest}, expected {SCANNET_FILE_MANIFEST_SHA256}"
                    )
                manifest["scannet_file_manifest_sha256"] = digest
                manifest["scannet_scene_count"] = SCANNET_SCENE_COUNT
                manifest["scannet_frame_count"] = SCANNET_FRAME_COUNT
        return manifest
