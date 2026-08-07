"""Locked SPBench-SI data ownership and restricted one-image prompt inputs."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ...models.common.runtime import pixel_sha256

DATASET_REPOSITORY = "hongxingli/SPBench"
DATASET_REVISION = "03611025a4e6032c558117c0e86b76c8b084c305"
UPSTREAM_REPOSITORY = "ZJU-REAL/SpatialLadder"
UPSTREAM_COMMIT = "7a0d2ee85c28728835300310a349a53a15967f2e"

PARQUET_SHA256 = "72aa46f998212a0d0a9c93ea24107eea086425ccc610083ede35c6218050c9a4"
PARQUET_SIZE_BYTES = 24_423
IMAGES_ARCHIVE_SHA256 = "bb53190a1eacf4268fb109b0d8e353c750908bdf33cad8a9221b187d81439461"
IMAGES_ARCHIVE_SIZE_BYTES = 49_171_512

OFFICIAL_TEST_SIZE = 1_009
OFFICIAL_IMAGE_COUNT = 524
OFFICIAL_SCENE_COUNT = 170
SMOKE8_INDICES = (4, 297, 306, 410, 460, 518, 918, 1008)

SYSTEM_PROMPT = "You are a helpful assistant."
MCQ_DIRECT_SUFFIX = (
    "Please answer with the option's letter from the given choices "
    "(e.g., A, B, etc.) directly."
)
NUMERIC_DIRECT_SUFFIX = (
    "Please answer the question using a numerical value (e.g., 42 or 3.1) directly."
)
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

NUMERIC_TASKS = ("object_abs_distance", "object_size_estimation")
MULTIPLE_CHOICE_TASKS = ("object_rel_distance", "object_rel_direction")
TASK_SEQUENCE = (*NUMERIC_TASKS, *MULTIPLE_CHOICE_TASKS)
EXPECTED_TASK_COUNTS = {
    "object_abs_distance": 149,
    "object_size_estimation": 463,
    "object_rel_distance": 91,
    "object_rel_direction": 306,
}
EXPECTED_PARQUET_SCHEMA = (
    ("id", "int64"),
    ("dataset", "string"),
    ("scene_name", "string"),
    ("question_type", "string"),
    ("question", "string"),
    ("ground_truth", "string"),
    ("options", "list<element: string>"),
    ("images", "list<element: string>"),
)
REQUIRED_SOURCE_FIELDS = frozenset(name for name, _kind in EXPECTED_PARQUET_SCHEMA)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parquet_schema(path: Path) -> tuple[tuple[str, str], ...]:
    import pyarrow.parquet as parquet

    return tuple((field.name, str(field.type)) for field in parquet.ParquetFile(path).schema_arrow)


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return [dict(row) for row in parquet.read_table(path).to_pylist()]


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def build_user_prompt(question: Any, options: Any, question_type: str) -> str:
    text = str(question).strip()
    if not text:
        raise ValueError("SPBench-SI question must be non-empty")
    if question_type in MULTIPLE_CHOICE_TASKS:
        rendered_options = [str(option) for option in _as_sequence(options)]
        if not rendered_options:
            raise ValueError(f"SPBench-SI {question_type} requires choices")
        body = f"{text}\nOptions:\n" + "\n".join(rendered_options)
        suffix = MCQ_DIRECT_SUFFIX
    elif question_type in NUMERIC_TASKS:
        if _as_sequence(options):
            raise ValueError(f"SPBench-SI {question_type} must not contain choices")
        body = text
        suffix = NUMERIC_DIRECT_SUFFIX
    else:
        raise ValueError(f"Unsupported SPBench-SI question type: {question_type!r}")
    return f"Question: {body}\n\n{suffix}"


def _validate_source_row(row: Mapping[str, Any], *, index: int) -> None:
    missing = sorted(REQUIRED_SOURCE_FIELDS - set(row))
    if missing:
        raise ValueError(f"SPBench-SI index {index} is missing source fields: {missing}")
    try:
        official_id = int(row["id"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SPBench-SI index {index} has invalid id") from exc
    if official_id != index + 1:
        raise ValueError(f"SPBench-SI index {index} has id={official_id}, expected {index + 1}")
    if str(row["dataset"]) != "scannet":
        raise ValueError(f"SPBench-SI index {index} is not from scannet")
    scene = str(row["scene_name"])
    if not scene or "/" in scene or ".." in scene:
        raise ValueError(f"SPBench-SI index {index} has unsafe scene_name={scene!r}")
    question_type = str(row["question_type"])
    if question_type not in EXPECTED_TASK_COUNTS:
        raise ValueError(f"SPBench-SI index {index} has unsupported type={question_type!r}")
    images = [str(value) for value in _as_sequence(row["images"])]
    if len(images) != 1:
        raise ValueError(f"SPBench-SI index {index} must reference exactly one image")
    image_name = images[0]
    if PurePosixPath(image_name).name != image_name or not image_name.lower().endswith((".jpg", ".jpeg")):
        raise ValueError(f"SPBench-SI index {index} has unsafe image name={image_name!r}")
    ground_truth = str(row["ground_truth"]).strip()
    if not ground_truth:
        raise ValueError(f"SPBench-SI index {index} has an empty ground truth")
    options = [str(value) for value in _as_sequence(row["options"])]
    if question_type in MULTIPLE_CHOICE_TASKS:
        if len(options) not in {2, 4}:
            raise ValueError(f"SPBench-SI index {index} has invalid option count={len(options)}")
        expected_letters = [chr(ord("A") + offset) for offset in range(len(options))]
        actual_letters = [option.split(".", 1)[0].strip() for option in options]
        if actual_letters != expected_letters or ground_truth not in expected_letters:
            raise ValueError(f"SPBench-SI index {index} has invalid choice labels")
    elif options:
        raise ValueError(f"SPBench-SI numeric index {index} unexpectedly has options")
    build_user_prompt(row["question"], row["options"], question_type)


@dataclass(frozen=True, slots=True)
class SPBenchSIModelInput:
    """The complete safe object visible to every SPBench-SI adapter."""

    index: int
    image: Any
    system_prompt: str
    user_prompt: str


class SPBenchSITestContract:
    """Keep labels private while reading each referenced JPEG directly from the locked ZIP."""

    def __init__(
        self,
        parquet_path: str | Path,
        images_archive: str | Path,
        *,
        rows: Sequence[Mapping[str, Any]] | None = None,
        require_official_size: bool = True,
        verify_files: bool = True,
        verify_images: bool = True,
    ) -> None:
        self.parquet_path = Path(parquet_path).expanduser().absolute()
        self.images_archive = Path(images_archive).expanduser().absolute()
        self.dataset_root = self.parquet_path.parent
        parquet_schema: tuple[tuple[str, str], ...] | None = None
        if rows is None:
            if not self.parquet_path.is_file():
                raise FileNotFoundError(f"Missing locked SPBench-SI Parquet: {self.parquet_path}")
            if verify_files:
                self._verify_file(
                    self.parquet_path, PARQUET_SIZE_BYTES, PARQUET_SHA256, "SPBench-SI Parquet"
                )
                parquet_schema = _parquet_schema(self.parquet_path)
                if parquet_schema != EXPECTED_PARQUET_SCHEMA:
                    raise ValueError(
                        f"SPBench-SI Parquet schema is {parquet_schema}, expected {EXPECTED_PARQUET_SCHEMA}"
                    )
            rows = _load_parquet(self.parquet_path)
        self.__rows = [dict(row) for row in rows]
        if require_official_size and len(self.__rows) != OFFICIAL_TEST_SIZE:
            raise ValueError(
                f"SPBench-SI requires {OFFICIAL_TEST_SIZE} rows, got {len(self.__rows)}"
            )
        for index, row in enumerate(self.__rows):
            _validate_source_row(row, index=index)
        ids = [int(row["id"]) for row in self.__rows]
        if len(ids) != len(set(ids)):
            raise ValueError("SPBench-SI official ids are not unique")
        self.__parquet_schema = parquet_schema
        self.__task_counts = Counter(str(row["question_type"]) for row in self.__rows)
        self.__scene_count = len({str(row["scene_name"]) for row in self.__rows})
        if require_official_size:
            if dict(self.__task_counts) != EXPECTED_TASK_COUNTS:
                raise ValueError(
                    f"SPBench-SI task counts differ: got={dict(self.__task_counts)}, expected={EXPECTED_TASK_COUNTS}"
                )
            if self.__scene_count != OFFICIAL_SCENE_COUNT:
                raise ValueError(
                    f"SPBench-SI scene count is {self.__scene_count}, expected {OFFICIAL_SCENE_COUNT}"
                )
            smoke_rows = [self.__rows[index] for index in SMOKE8_INDICES]
            smoke_counts = Counter(str(row["question_type"]) for row in smoke_rows)
            if smoke_counts != Counter({task: 2 for task in TASK_SEQUENCE}):
                raise ValueError(f"SPBench-SI smoke8 task coverage differs: {dict(smoke_counts)}")
            if len({str(row["scene_name"]) for row in smoke_rows}) != len(SMOKE8_INDICES):
                raise ValueError("SPBench-SI smoke8 must cover eight distinct scenes")
        if not self.images_archive.is_file():
            raise FileNotFoundError(f"Missing locked SPBench-SI image archive: {self.images_archive}")
        if verify_files:
            self._verify_file(
                self.images_archive,
                IMAGES_ARCHIVE_SIZE_BYTES,
                IMAGES_ARCHIVE_SHA256,
                "SPBench-SI image archive",
            )
        self.__image_members = self._index_archive()
        referenced = {
            f"{row['scene_name']}/{_as_sequence(row['images'])[0]}" for row in self.__rows
        }
        actual = set(self.__image_members)
        if referenced != actual:
            missing = sorted(referenced - actual)
            extra = sorted(actual - referenced)
            raise ValueError(
                f"SPBench-SI ZIP/reference mismatch: missing={missing[:10]}, extra={extra[:10]}"
            )
        if require_official_size and len(actual) != OFFICIAL_IMAGE_COUNT:
            raise ValueError(
                f"SPBench-SI ZIP contains {len(actual)} referenced JPEGs, expected {OFFICIAL_IMAGE_COUNT}"
            )
        image_manifest: list[dict[str, Any]] = []
        if verify_images:
            for relative in sorted(actual):
                image = self._read_image(relative)
                image_manifest.append(
                    {
                        "path": relative,
                        "size": list(image.size),
                        "pixel_sha256": pixel_sha256(image),
                    }
                )
        self.__image_manifest = tuple(image_manifest)
        self.__image_manifest_sha256 = hashlib.sha256(
            json.dumps(image_manifest or sorted(actual), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.__dataset_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "repository": DATASET_REPOSITORY,
                    "revision": DATASET_REVISION,
                    "parquet_sha256": PARQUET_SHA256,
                    "images_archive_sha256": IMAGES_ARCHIVE_SHA256,
                    "rows": len(self.__rows),
                    "task_counts": dict(self.__task_counts),
                    "image_manifest_sha256": self.__image_manifest_sha256,
                    "prompt": {
                        "system": SYSTEM_PROMPT_SHA256,
                        "mcq": hashlib.sha256(MCQ_DIRECT_SUFFIX.encode()).hexdigest(),
                        "numeric": hashlib.sha256(NUMERIC_DIRECT_SUFFIX.encode()).hexdigest(),
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    @staticmethod
    def _verify_file(path: Path, size: int, digest: str, label: str) -> None:
        if path.stat().st_size != size:
            raise ValueError(f"{label} size is {path.stat().st_size}, expected {size}")
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"{label} SHA-256 is {actual}, expected {digest}")

    def _index_archive(self) -> dict[str, str]:
        suffix_to_member: dict[str, str] = {}
        with zipfile.ZipFile(self.images_archive) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"SPBench-SI ZIP has a corrupt member: {bad_member}")
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith((".jpg", ".jpeg")):
                    continue
                parts = PurePosixPath(info.filename).parts
                if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
                    raise ValueError(f"Unsafe SPBench-SI ZIP member: {info.filename!r}")
                suffix = "/".join(parts[-2:])
                if suffix in suffix_to_member:
                    raise ValueError(f"Duplicate SPBench-SI ZIP image suffix: {suffix}")
                suffix_to_member[suffix] = info.filename
        if not suffix_to_member:
            raise ValueError("SPBench-SI ZIP contains no JPEG members")
        return suffix_to_member

    def _read_image(self, relative: str) -> Any:
        from PIL import Image

        try:
            member = self.__image_members[relative]
        except KeyError as exc:
            raise FileNotFoundError(f"Missing SPBench-SI image in ZIP: {relative}") from exc
        with zipfile.ZipFile(self.images_archive) as archive:
            raw = archive.read(member)
        with Image.open(io.BytesIO(raw)) as loaded:
            image = loaded.convert("RGB")
            image.load()
        return image

    def __len__(self) -> int:
        return len(self.__rows)

    @property
    def dataset_fingerprint(self) -> str:
        return self.__dataset_fingerprint

    def _source_row(self, index: int) -> dict[str, Any]:
        resolved = int(index)
        if not 0 <= resolved < len(self):
            raise IndexError(f"SPBench-SI index {resolved} outside [0,{len(self)})")
        row = dict(self.__rows[resolved])
        _validate_source_row(row, index=resolved)
        return row

    def model_input(self, index: int) -> SPBenchSIModelInput:
        resolved = int(index)
        row = self._source_row(resolved)
        relative = f"{row['scene_name']}/{_as_sequence(row['images'])[0]}"
        return SPBenchSIModelInput(
            index=resolved,
            image=self._read_image(relative),
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(row["question"], row["options"], str(row["question_type"])),
        )

    def model_inputs(self, indices: Iterable[int]) -> list[SPBenchSIModelInput]:
        return [self.model_input(index) for index in indices]

    def prediction_row(self, index: int, prediction: str) -> dict[str, Any]:
        self._source_row(int(index))
        return {"index": int(index), "raw_prediction": str(prediction)}

    def scoring_row(self, index: int) -> dict[str, Any]:
        row = self._source_row(int(index))
        return {
            "index": int(index),
            "official_id": int(row["id"]),
            "question_type": str(row["question_type"]),
            "ground_truth": str(row["ground_truth"]),
            "options": [str(value) for value in _as_sequence(row["options"])],
        }

    def dataset_manifest(self, *, include_images: bool = False) -> dict[str, Any]:
        manifest = {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "parquet": {
                "path": str(self.parquet_path),
                "size_bytes": PARQUET_SIZE_BYTES,
                "sha256": PARQUET_SHA256,
                "schema": self.__parquet_schema or EXPECTED_PARQUET_SCHEMA,
            },
            "images_archive": {
                "path": str(self.images_archive),
                "size_bytes": IMAGES_ARCHIVE_SIZE_BYTES,
                "sha256": IMAGES_ARCHIVE_SHA256,
                "jpeg_count": len(self.__image_members),
                "extraction": "none; decoded directly from ZIP",
            },
            "rows": len(self),
            "official_ids": [1, len(self)],
            "scene_count": self.__scene_count,
            "task_counts": dict(self.__task_counts),
            "smoke8": {
                "indices": list(SMOKE8_INDICES),
                "task_counts": dict(Counter(
                    str(self.__rows[index]["question_type"]) for index in SMOKE8_INDICES
                    if index < len(self.__rows)
                )),
                "scene_count": len({
                    str(self.__rows[index]["scene_name"]) for index in SMOKE8_INDICES
                    if index < len(self.__rows)
                }),
            },
            "image_manifest_sha256": self.__image_manifest_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        if include_images:
            manifest["images"] = list(self.__image_manifest)
        return manifest
