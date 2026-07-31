"""Run native Qwen2.5-VL/PEFT inference under the common MSMU contract."""

from __future__ import annotations

import argparse

from ..common.cli import add_msmu_run_arguments, execute_msmu_cli
from ..profiles import profile_keys
from ..qwen_vl.base import QwenVLAdapterBase

QWEN_MAX_NEW_TOKENS = 192
QWEN_IMAGE_MIN_PIXELS = 12544
QWEN_IMAGE_MAX_PIXELS = 112896
QWEN_PROFILE_KEYS = profile_keys("qwen25_vl")


class QwenPeftAdapter(QwenVLAdapterBase):
    """Retained Qwen2.5-VL adapter, including its 7B PEFT path."""

    PROFILE_FAMILY = "qwen25_vl"
    MODEL_CLASS_NAME = "Qwen2_5_VLForConditionalGeneration"
    MODEL_LABEL = "Qwen2.5-VL"
    BACKEND_NAME = "transformers-qwen25-vl-peft"
    CHAT_TEMPLATE_DESCRIPTION = "Qwen2.5-VL native structured image chat template"
    TRUST_REMOTE_CODE = True
    SUPPORTS_PEFT = True
    REQUIRED_TRANSFORMERS = "with Qwen2.5-VL support"

    def __init__(
        self,
        *,
        profile_key: str,
        base_model: str,
        base_model_revision: str,
        checkpoint: str | None,
        checkpoint_revision: str | None,
        batch_size: int,
        max_new_tokens: int,
        image_min_pixels: int,
        image_max_pixels: int,
        device_map: str,
    ) -> None:
        super().__init__(
            profile_key=profile_key,
            base_model=base_model,
            base_model_revision=base_model_revision,
            checkpoint=checkpoint,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            device_map=device_map,
            expected_max_new_tokens=QWEN_MAX_NEW_TOKENS,
            expected_image_min_pixels=QWEN_IMAGE_MIN_PIXELS,
            expected_image_max_pixels=QWEN_IMAGE_MAX_PIXELS,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=QWEN_PROFILE_KEYS, default="qwen25_vl_7b")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-revision", default="local-unspecified")
    parser.add_argument("--checkpoint", default=None, help="PEFT adapter checkpoint; omit for base.")
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=QWEN_MAX_NEW_TOKENS)
    parser.add_argument("--image-min-pixels", type=int, default=QWEN_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=QWEN_IMAGE_MAX_PIXELS)
    parser.add_argument("--device-map", choices=("single", "balanced"), default="single")
    parser.add_argument("--vision-canary-report", default=None)
    add_msmu_run_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = QwenPeftAdapter(
        profile_key=args.profile,
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        device_map=args.device_map,
    )
    if args.vision_canary_report:
        try:
            adapter.run_vision_canary(args.vision_canary_report)
        except Exception:
            adapter.close()
            raise
    execute_msmu_cli(args, adapter)


if __name__ == "__main__":
    main()
