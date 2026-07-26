"""MSMU test-set ownership boundary and model-safe input objects.

The benchmark layer owns provenance fields.  Model adapters receive only
``MSMUModelInput`` instances, which deliberately contain no reference answer,
raw type, task family, or conversation history.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OFFICIAL_TEST_SIZE = 987


def load_arrow_split(dataset_root: str | Path, split: str) -> Any:
    """Load all Arrow shards for one MSMU split in filename order."""

    from datasets import Dataset, concatenate_datasets

    split_dir = Path(dataset_root) / split
    paths = sorted(split_dir.glob("*.arrow"))
    if not paths:
        raise FileNotFoundError(f"No Arrow shards found in {split_dir}")
    shards = [Dataset.from_file(str(path)) for path in paths]
    return concatenate_datasets(shards) if len(shards) > 1 else shards[0]


def clean_question(text: str) -> str:
    """Remove the dataset's textual image placeholder and outer whitespace."""

    return str(text).replace("<image>", "").strip()


def _conversation_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    conversations = row.get("conversations")
    if isinstance(conversations, dict):
        roles = list(conversations.get("from", []))
        values = list(conversations.get("value", []))
        if len(roles) != len(values):
            raise ValueError("MSMU conversation role/value lengths do not match")
        return [(str(role), str(value)) for role, value in zip(roles, values)]
    if isinstance(conversations, list):
        return [
            (str(turn.get("from", turn.get("role", ""))), str(turn.get("value", turn.get("content", ""))))
            for turn in conversations
        ]
    raise ValueError("MSMU row has no supported conversations structure")


def _first_question_and_reference(row: dict[str, Any]) -> tuple[str, str]:
    pairs = _conversation_pairs(row)
    if len(pairs) < 2:
        raise ValueError("MSMU row must contain a first user question and reference answer")
    first_role, first_value = pairs[0]
    second_role, second_value = pairs[1]
    if first_role.lower() not in {"human", "user"}:
        raise ValueError(f"MSMU first turn is not a user turn: {first_role!r}")
    if second_role.lower() not in {"gpt", "assistant"}:
        raise ValueError(f"MSMU second turn is not an assistant reference: {second_role!r}")
    if first_value.count("<image>") != 1:
        raise ValueError("MSMU first user question must contain exactly one literal <image>")
    return clean_question(first_value), second_value.strip()


@dataclass(frozen=True, slots=True)
class MSMUModelInput:
    """The complete and intentionally narrow object visible to a model adapter."""

    index: int
    image: Any
    question: str


class MSMUTestContract:
    """Own source rows while exposing only restricted per-model inputs."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        dataset: Any | None = None,
        require_official_size: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.__dataset = dataset if dataset is not None else load_arrow_split(self.dataset_root, "test")
        if require_official_size and len(self.__dataset) != OFFICIAL_TEST_SIZE:
            raise ValueError(
                f"MSMU official test split must contain {OFFICIAL_TEST_SIZE} rows; "
                f"found {len(self.__dataset)}"
            )

    def __len__(self) -> int:
        return len(self.__dataset)

    @property
    def dataset_fingerprint(self) -> str | None:
        value = getattr(self.__dataset, "_fingerprint", None)
        return str(value) if value is not None else None

    def _source_row(self, index: int) -> dict[str, Any]:
        resolved = int(index)
        if not 0 <= resolved < len(self):
            raise IndexError(f"MSMU index {resolved} outside [0,{len(self)})")
        row = self.__dataset[resolved]
        if not isinstance(row, dict):
            row = dict(row)
        return row

    def model_input(self, index: int) -> MSMUModelInput:
        """Return a fresh RGB image plus the clean first-user question only."""

        resolved = int(index)
        row = self._source_row(resolved)
        question, _ = _first_question_and_reference(row)
        image = row.get("image")
        if image is None or not hasattr(image, "convert"):
            raise ValueError(f"MSMU index {resolved} has no PIL-compatible image")
        return MSMUModelInput(index=resolved, image=image.convert("RGB"), question=question)

    def model_inputs(self, indices: Iterable[int]) -> list[MSMUModelInput]:
        return [self.model_input(index) for index in indices]

    def prediction_row(self, index: int, prediction: str) -> dict[str, Any]:
        """Reattach dataset-owned provenance to one model-generated string."""

        resolved = int(index)
        row = self._source_row(resolved)
        question, reference = _first_question_and_reference(row)
        raw_type = str(row.get("type") or "")
        return {
            "index": resolved,
            "raw_type": raw_type,
            "task_family": task_family(raw_type),
            "question": question,
            "reference": reference,
            "prediction": str(prediction),
        }


def qwen_user_messages(model_input: MSMUModelInput) -> list[dict[str, Any]]:
    """Build the one-turn native structured message used by Qwen-family processors."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": model_input.question},
            ],
        }
    ]


def conversation_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Compatibility helper that now returns only the first user turn."""

    question, _ = _first_question_and_reference(row)
    return qwen_user_messages(MSMUModelInput(index=-1, image=None, question=question))


class MSMUArrowDataset:
    """A restricted model-input view over selected original test indices."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "test",
        limit: int | None = None,
        *,
        indices: Sequence[int] | None = None,
        require_official_size: bool = True,
    ) -> None:
        if split != "test":
            raise ValueError("MSMU inference is locked to the official test split")
        self.__contract = MSMUTestContract(
            dataset_root,
            require_official_size=require_official_size,
        )
        selected = list(range(len(self.__contract))) if indices is None else [int(i) for i in indices]
        if len(selected) != len(set(selected)):
            raise ValueError("MSMU selected indices contain duplicates")
        if limit is not None:
            selected = selected[: max(0, int(limit))]
        for index in selected:
            if not 0 <= index < len(self.__contract):
                raise IndexError(f"MSMU index {index} outside [0,{len(self.__contract)})")
        self.__indices = tuple(selected)

    def __len__(self) -> int:
        return len(self.__indices)

    def __getitem__(self, position: int) -> MSMUModelInput:
        return self.__contract.model_input(self.__indices[int(position)])


class QwenGenerationCollator:
    """Build native Qwen generation tensors from restricted model inputs only."""

    def __init__(
        self,
        processor: Any,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = 112896,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if image_min_pixels is not None:
            self.processor.image_processor.min_pixels = int(image_min_pixels)
        if image_max_pixels is not None:
            self.processor.image_processor.max_pixels = int(image_max_pixels)

    def __call__(self, features: Sequence[MSMUModelInput]) -> dict[str, Any]:
        texts = [
            self.processor.apply_chat_template(
                qwen_user_messages(feature),
                tokenize=False,
                add_generation_prompt=True,
            )
            for feature in features
        ]
        encoded = self.processor(
            text=texts,
            images=[feature.image for feature in features],
            padding=True,
            return_tensors="pt",
        )
        return {
            "model_inputs": dict(encoded),
            "indices": [int(feature.index) for feature in features],
        }


RAW_TYPE_TO_FAMILY = {
    "width": "scale_estimation",
    "height": "scale_estimation",
    "size": "scale_estimation",
    "distance": "absolute_distance",
    "count": "object_counting",
    "position": "grounding",
    "refer_two_objects": "reference_object_estimation",
    "refer_three_objects": "reference_object_estimation",
    "left/right": "relative_position",
    "taller_two_object": "scale_comparison",
    "tall_three_objects": "scale_comparison",
    "zero": "existence",
}

FAMILY_TO_OFFICIAL_TYPE = {
    "scale_estimation": "scale_estimation",
    "absolute_distance": "absolute_distance",
    "object_counting": "count",
    "grounding": "grounding",
    "reference_object_estimation": "refer_obj_estimation",
    "relative_position": "relative_position",
    "scale_comparison": "scale_compare",
    "existence": "existence",
}


def task_family(raw_type: str) -> str:
    try:
        return RAW_TYPE_TO_FAMILY[str(raw_type)]
    except KeyError as exc:
        raise ValueError(f"Unknown MSMU-Bench raw type: {raw_type}") from exc


def official_type_for_raw_type(raw_type: str) -> str:
    """Map one dataset-owned raw type to the locked scorer category."""

    family = task_family(raw_type)
    try:
        return FAMILY_TO_OFFICIAL_TYPE[family]
    except KeyError as exc:
        raise ValueError(f"Unknown MSMU-Bench task family: {family}") from exc
