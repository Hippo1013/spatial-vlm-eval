"""MSMU Arrow test-set adapter and native Qwen generation collator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


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


def conversation_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an MSMU conversation to native Qwen multimodal messages."""

    conversations = row["conversations"]
    roles = list(conversations["from"])
    values = list(conversations["value"])
    if len(roles) != len(values):
        raise ValueError("MSMU conversation role/value lengths do not match")
    messages: list[dict[str, Any]] = []
    image_inserted = False
    for role, value in zip(roles, values):
        role_name = "assistant" if str(role).lower() in {"gpt", "assistant"} else "user"
        text = clean_question(value) if role_name == "user" else str(value).strip()
        if role_name == "user" and not image_inserted:
            content: Any = [
                {"type": "image"},
                {"type": "text", "text": text},
            ]
            image_inserted = True
        else:
            content = text
        messages.append({"role": role_name, "content": content})
    if not image_inserted:
        raise ValueError("MSMU conversation contains no user turn")
    return messages


class MSMUArrowDataset:
    """Expose MSMU Arrow rows without changing their original indices."""

    def __init__(self, dataset_root: str | Path, split: str, limit: int | None = None) -> None:
        self.dataset = load_arrow_split(dataset_root, split)
        self.limit = min(int(limit), len(self.dataset)) if limit is not None else len(self.dataset)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[int(index)]
        return {
            "index": int(index),
            "image": row["image"].convert("RGB"),
            "messages": conversation_messages(row),
            "raw_type": str(row.get("type") or ""),
            "conversations": row["conversations"],
        }


class QwenGenerationCollator:
    """Build generation inputs containing only the first user turn and image."""

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

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        prompt_messages = []
        for feature in features:
            first_user = next(
                message for message in feature["messages"] if message["role"] == "user"
            )
            prompt_messages.append([first_user])
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in prompt_messages
        ]
        encoded = self.processor(
            text=texts,
            images=[feature["image"] for feature in features],
            padding=True,
            return_tensors="pt",
        )
        return {
            "model_inputs": dict(encoded),
            "indices": [int(feature["index"]) for feature in features],
            "raw_types": [str(feature["raw_type"]) for feature in features],
            "questions": [
                clean_question(feature["conversations"]["value"][0]) for feature in features
            ],
            "references": [
                str(feature["conversations"]["value"][1]).strip() for feature in features
            ],
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
