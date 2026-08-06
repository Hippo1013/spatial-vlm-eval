"""Official processor/template single-image audit before Q-Spatial vLLM use."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

from ...models.common.provenance import verify_hf_snapshot_revision
from ...models.common.runtime import pixel_sha256
from .command_adapter import fold_system_user_prompt
from .data import LLAVA_CONTINUATION, LLAVA_USER_FORMAT_SUFFIX, STANDARD_SYSTEM_PROMPT
from .profiles import QSpatialProfile

PLACEHOLDERS = {
    "llava_next": "<image>",
    "internvl3": "<IMG_CONTEXT>",
    "qwen3_vl": "<|image_pad|>",
}


def processor_messages(
    profile: QSpatialProfile,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, Any]]:
    final_user_prompt = (
        f"{user_prompt}\n{LLAVA_USER_FORMAT_SUFFIX}" if profile.llava_two_stage else user_prompt
    )
    user = {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    final_user_prompt
                    if profile.system_role_supported
                    else fold_system_user_prompt(system_prompt, final_user_prompt)
                ),
            },
        ],
    }
    if profile.system_role_supported:
        return [{"role": "system", "content": system_prompt}, user]
    return [user]


def validate_processor_audit(
    *,
    profile: QSpatialProfile,
    rendered_prompt: str,
    encoded: Any,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
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
    if pixel_values is None or int(pixel_values.numel()) <= 0:
        raise ValueError(f"{profile.key} processor produced no non-empty pixel_values")
    image_grid = encoded.get("image_grid_thw") if hasattr(encoded, "get") else None
    if image_grid is not None and len(image_grid) != 1:
        raise ValueError(f"{profile.key} processor did not produce exactly one image-grid row")
    final_user_prompt = (
        f"{user_prompt}\n{LLAVA_USER_FORMAT_SUFFIX}" if profile.llava_two_stage else user_prompt
    )
    for label, required_text in (("system prompt", system_prompt), ("final user prompt", final_user_prompt)):
        if required_text not in rendered_prompt:
            raise ValueError(f"{profile.key} rendered template lost the exact {label} bytes")
    folded = fold_system_user_prompt(system_prompt, final_user_prompt)
    return {
        "passed": True,
        "profile": profile.key,
        "model": profile.model,
        "model_revision": profile.revision,
        "processor_family": profile.processor_family,
        "system_role_supported": profile.system_role_supported,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
        "final_stage_1_user_prompt_sha256": hashlib.sha256(
            final_user_prompt.encode("utf-8")
        ).hexdigest(),
        "folded_prompt_sha256": (
            hashlib.sha256(folded.encode("utf-8")).hexdigest()
            if not profile.system_role_supported
            else None
        ),
        "logical_image_placeholder": placeholder,
        "logical_image_placeholder_count": placeholder_count,
        "rendered_template_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
        "rendered_template_characters": len(rendered_prompt),
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_numel": int(pixel_values.numel()),
        "image_grid_rows": len(image_grid) if image_grid is not None else None,
        "input_image_count": 1,
        "input_image_mode": "RGB",
        "input_image_size": list(image.size),
        "input_image_pixel_sha256": pixel_sha256(image),
        "llava_two_stage": (
            {
                "enabled": True,
                "stage_1_user_suffix_sha256": hashlib.sha256(
                    LLAVA_USER_FORMAT_SUFFIX.encode("utf-8")
                ).hexdigest(),
                "stage_2_continuation_sha256": hashlib.sha256(
                    LLAVA_CONTINUATION.encode("utf-8")
                ).hexdigest(),
            }
            if profile.llava_two_stage
            else {"enabled": False}
        ),
    }


def audit_processor(profile: QSpatialProfile, model_path: str | Path) -> dict[str, Any]:
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
    user_prompt = "Question: What color is the image?"
    messages = processor_messages(profile, STANDARD_SYSTEM_PROMPT, user_prompt)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        system_prompt=STANDARD_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    report["processor_class"] = type(processor).__name__
    report["model_snapshot_revision_verified"] = snapshot_verified
    return report
