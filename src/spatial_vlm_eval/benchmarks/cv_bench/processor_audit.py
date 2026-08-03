"""Official Transformers processor/template audit before CV-Bench vLLM use."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

from ...models.common.provenance import verify_hf_snapshot_revision
from ...models.common.runtime import pixel_sha256
from .profiles import CVBenchProfile

PLACEHOLDERS = {
    "llava_next": "<image>",
    "internvl3": "<IMG_CONTEXT>",
    "qwen3_vl": "<|image_pad|>",
}


def processor_messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": str(prompt)},
            ],
        }
    ]


def validate_processor_audit(
    *,
    profile: CVBenchProfile,
    rendered_prompt: str,
    encoded: Any,
    image: Image.Image,
) -> dict[str, Any]:
    try:
        placeholder = PLACEHOLDERS[str(profile.processor_family)]
    except KeyError as exc:
        raise ValueError(f"No processor placeholder contract for {profile.processor_family!r}") from exc
    placeholder_count = rendered_prompt.count(placeholder)
    if placeholder_count != 1:
        raise ValueError(
            f"{profile.key} rendered {placeholder_count} {placeholder} placeholders; expected one"
        )
    pixel_values = encoded.get("pixel_values") if hasattr(encoded, "get") else None
    if pixel_values is None:
        raise ValueError(f"{profile.key} processor produced no pixel_values")
    numel = int(pixel_values.numel()) if hasattr(pixel_values, "numel") else 0
    if numel <= 0:
        raise ValueError(f"{profile.key} processor produced empty pixel_values")
    image_grid = encoded.get("image_grid_thw") if hasattr(encoded, "get") else None
    if image_grid is not None and len(image_grid) != 1:
        raise ValueError(f"{profile.key} processor did not produce exactly one image-grid row")
    return {
        "passed": True,
        "profile": profile.key,
        "model": profile.model,
        "model_revision": profile.revision,
        "processor_family": profile.processor_family,
        "logical_image_placeholder": placeholder,
        "logical_image_placeholder_count": placeholder_count,
        "rendered_template_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
        "rendered_template_characters": len(rendered_prompt),
        "pixel_values_shape": list(pixel_values.shape) if hasattr(pixel_values, "shape") else None,
        "pixel_values_numel": numel,
        "image_grid_rows": len(image_grid) if image_grid is not None else None,
        "input_image_count": 1,
        "input_image_mode": "RGB",
        "input_image_size": list(image.size),
        "input_image_pixel_sha256": pixel_sha256(image),
    }


def audit_processor(profile: CVBenchProfile, model_path: str | Path) -> dict[str, Any]:
    from transformers import AutoProcessor

    path_or_id = str(model_path)
    snapshot_verified = verify_hf_snapshot_revision(path_or_id, profile.revision, profile.model)
    kwargs: dict[str, Any] = {
        "revision": profile.revision,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if Path(path_or_id).exists():
        kwargs.pop("revision")
    processor = AutoProcessor.from_pretrained(path_or_id, **kwargs)
    image = Image.new("RGB", (64, 48), (220, 30, 30))
    prompt = "What object is shown?\nAnswer with one short phrase."
    messages = processor_messages(prompt)
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = processor(
        text=[rendered],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    report = validate_processor_audit(
        profile=profile,
        rendered_prompt=rendered,
        encoded=encoded,
        image=image,
    )
    report["processor_class"] = type(processor).__name__
    report["model_snapshot_revision_verified"] = snapshot_verified
    return report
