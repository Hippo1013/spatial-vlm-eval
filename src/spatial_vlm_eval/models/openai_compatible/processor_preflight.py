"""Static image-path preflight for vLLM LLaVA-NeXT and InternVL profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..common.provenance import verify_hf_snapshot_revision
from ..common.runtime import atomic_write_json
from ..profiles import get_profile, profile_keys

PROFILE_KEYS = profile_keys("llava_next", "internvl3")


def logical_image_placeholder(profile_key: str) -> str:
    profile = get_profile(profile_key)
    if profile.family == "llava_next":
        return "<image>"
    if profile.family == "internvl3":
        return "<IMG_CONTEXT>"
    raise ValueError(f"Unsupported processor preflight family: {profile.family}")


def processor_messages(
    profile_key: str,
    question: str = "What color is the square?",
) -> list[dict[str, Any]]:
    """Mimic vLLM's one-image structured chat message before HF processing.

    The checkpoint's native chat template, rather than this helper, must expand
    the structured image item to the family-specific logical placeholder.
    LLaVA's template silently drops a plain-string ``<image>`` message, so
    inserting the placeholder ourselves would make this preflight validate a
    path different from the one used by an OpenAI-compatible multimodal request.
    """

    logical_image_placeholder(profile_key)  # fail early for unsupported families
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]


def validate_processor_contract(
    *,
    profile_key: str,
    rendered_prompt: str,
    encoded: Any,
) -> dict[str, Any]:
    profile = get_profile(profile_key)
    placeholder = logical_image_placeholder(profile_key)
    logical_count = rendered_prompt.count(placeholder)
    if logical_count != 1:
        raise ValueError(
            f"{profile.key} rendered {logical_count} logical {placeholder} placeholders; expected one"
        )
    other_placeholder = "<IMG_CONTEXT>" if placeholder == "<image>" else "<image>"
    if rendered_prompt.count(other_placeholder):
        raise ValueError(f"{profile.key} rendered unexpected cross-family placeholder {other_placeholder}")
    pixel_values = encoded.get("pixel_values") if hasattr(encoded, "get") else None
    if pixel_values is None:
        raise ValueError(f"{profile.key} processor output has no pixel_values")
    numel = int(pixel_values.numel()) if hasattr(pixel_values, "numel") else 0
    if numel <= 0:
        raise ValueError(f"{profile.key} processor returned empty pixel_values")
    shape = list(pixel_values.shape) if hasattr(pixel_values, "shape") else None
    return {
        "profile": profile.key,
        "model": profile.model,
        "revision": profile.revision,
        "logical_model_placeholder": placeholder,
        "logical_model_placeholder_count": logical_count,
        "pixel_values_shape": shape,
        "pixel_values_numel": numel,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=PROFILE_KEYS)
    parser.add_argument("--model", default=None, help="Local model/cache path; defaults to profile model id.")
    parser.add_argument(
        "--image", default=None, help="Optional local canary image; otherwise a synthetic PNG is used."
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from PIL import Image
    from transformers import AutoProcessor

    profile = get_profile(args.profile)
    model = args.model or profile.model
    model_snapshot_revision_verified = verify_hf_snapshot_revision(
        model,
        profile.revision,
        profile.model,
    )
    image = Image.open(args.image).convert("RGB") if args.image else Image.new("RGB", (32, 24), (220, 30, 30))
    processor = AutoProcessor.from_pretrained(
        model,
        revision=profile.revision,
        trust_remote_code=True,
        local_files_only=True,
    )
    messages = processor_messages(profile.key)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=rendered, images=image, return_tensors="pt")
    report = validate_processor_contract(
        profile_key=profile.key,
        rendered_prompt=rendered,
        encoded=encoded,
    )
    report["processor_class"] = type(processor).__name__
    report["model_snapshot_revision_verified"] = model_snapshot_revision_verified
    if args.output:
        atomic_write_json(Path(args.output).resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
