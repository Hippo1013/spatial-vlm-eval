"""Run native Qwen3-VL-Instruct inference under the common MSMU contract."""

from __future__ import annotations

import argparse

from ..common.cli import add_msmu_run_arguments, execute_msmu_cli
from ..profiles import profile_keys
from ..qwen_vl.base import QwenVLAdapterBase

QWEN3_MAX_NEW_TOKENS = 192
QWEN3_IMAGE_MIN_PIXELS = 16 * 32 * 32
QWEN3_IMAGE_MAX_PIXELS = 144 * 32 * 32
QWEN3_PROFILE_KEYS = profile_keys("qwen3_vl")


class Qwen3VLAdapter(QwenVLAdapterBase):
    PROFILE_FAMILY = "qwen3_vl"
    MODEL_CLASS_NAME = "Qwen3VLForConditionalGeneration"
    MODEL_LABEL = "Qwen3-VL"
    BACKEND_NAME = "transformers-qwen3-vl"
    CHAT_TEMPLATE_DESCRIPTION = "Qwen3-VL native structured image chat template"
    REQUIRED_TRANSFORMERS = ">=4.57.0"
    PIXEL_CONFIG_STYLE = "size_edges"

    def __init__(
        self,
        *,
        profile_key: str,
        base_model: str,
        base_model_revision: str,
        batch_size: int,
        max_new_tokens: int,
        image_min_pixels: int,
        image_max_pixels: int,
        device_map: str,
    ) -> None:
        if device_map != "single":
            raise ValueError("Canonical Qwen3-VL supplement profiles require single-GPU loading")
        if profile_key == "qwen3_vl_32b" and int(batch_size) != 1:
            raise ValueError("Canonical qwen3_vl_32b inference requires batch_size=1")
        super().__init__(
            profile_key=profile_key,
            base_model=base_model,
            base_model_revision=base_model_revision,
            checkpoint=None,
            checkpoint_revision=None,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            device_map=device_map,
            expected_max_new_tokens=QWEN3_MAX_NEW_TOKENS,
            expected_image_min_pixels=QWEN3_IMAGE_MIN_PIXELS,
            expected_image_max_pixels=QWEN3_IMAGE_MAX_PIXELS,
        )

    def metadata(self):
        metadata = super().metadata()
        metadata["model_loading"] = {
            "model_class": self.MODEL_CLASS_NAME,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "trust_remote_code": False,
        }
        metadata["image_processing"]["processor_size_fields"] = {
            "shortest_edge": self.image_min_pixels,
            "longest_edge": self.image_max_pixels,
        }
        return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=QWEN3_PROFILE_KEYS, default="qwen3_vl_2b")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-revision", default="local-unspecified")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=QWEN3_MAX_NEW_TOKENS)
    parser.add_argument("--image-min-pixels", type=int, default=QWEN3_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=QWEN3_IMAGE_MAX_PIXELS)
    parser.add_argument("--device-map", choices=("single", "balanced"), default="single")
    parser.add_argument("--vision-canary-report", default=None)
    add_msmu_run_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = Qwen3VLAdapter(
        profile_key=args.profile,
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
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
