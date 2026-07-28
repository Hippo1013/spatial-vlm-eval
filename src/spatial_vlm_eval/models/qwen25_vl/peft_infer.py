"""Run native Qwen2.5-VL/PEFT inference under the common MSMU contract."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...benchmarks.msmu.data import MSMUModelInput, QwenGenerationCollator
from ..common.cli import add_msmu_run_arguments, execute_msmu_cli
from ..common.provenance import verify_hf_snapshot_revision
from ..common.runtime import GenerationResult, InferenceAdapter
from ..profiles import get_profile, profile_keys

QWEN_MAX_NEW_TOKENS = 192
QWEN_IMAGE_MIN_PIXELS = 12544
QWEN_IMAGE_MAX_PIXELS = 112896
QWEN_PROFILE_KEYS = profile_keys("qwen25_vl")


class QwenPeftAdapter(InferenceAdapter):
    supports_concurrency = False

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
        self.profile = get_profile(profile_key)
        if self.profile.family != "qwen25_vl":
            raise ValueError(f"Profile {profile_key!r} is not a Qwen2.5-VL profile")
        self.base_model = str(base_model)
        self.base_model_revision = str(base_model_revision)
        if self.base_model_revision != self.profile.revision:
            raise ValueError(
                f"Profile {self.profile.key} is locked to revision {self.profile.revision}; "
                f"got {self.base_model_revision}"
            )
        self.base_model_snapshot_revision_verified = verify_hf_snapshot_revision(
            self.base_model,
            self.base_model_revision,
            "Qwen2.5-VL base model",
        )
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.checkpoint_revision = str(checkpoint_revision) if checkpoint_revision else None
        self.batch_size = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.image_min_pixels = int(image_min_pixels)
        self.image_max_pixels = int(image_max_pixels)
        self.device_map = str(device_map)
        if self.device_map not in {"single", "balanced"}:
            raise ValueError("device_map must be 'single' or 'balanced'")
        if self.profile.default_tensor_parallel_size == 2 and self.device_map != "balanced":
            raise ValueError(f"Profile {self.profile.key} requires balanced two-GPU loading")
        locked = (
            self.max_new_tokens,
            self.image_min_pixels,
            self.image_max_pixels,
        )
        expected = (
            QWEN_MAX_NEW_TOKENS,
            QWEN_IMAGE_MIN_PIXELS,
            QWEN_IMAGE_MAX_PIXELS,
        )
        if locked != expected:
            raise ValueError(
                "The canonical Qwen inference protocol is locked to "
                f"max_new_tokens/min_pixels/max_pixels={expected}; got {locked}. "
                "A different decoding/image profile requires a new inference protocol."
            )
        self.processor: Any | None = None
        self.model: Any | None = None
        self.collator: QwenGenerationCollator | None = None
        self._runtime_versions: dict[str, str] = {}

    def _ensure_processor(self) -> None:
        if self.processor is not None:
            return
        import transformers
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.base_model,
            revision=None if self.base_model_revision == "local-unspecified" else self.base_model_revision,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.collator = QwenGenerationCollator(
            self.processor,
            image_min_pixels=self.image_min_pixels,
            image_max_pixels=self.image_max_pixels,
        )
        self._runtime_versions["transformers"] = transformers.__version__

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        self._ensure_processor()
        import peft
        import torch
        from peft import PeftModel
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model,
            revision=None if self.base_model_revision == "local-unspecified" else self.base_model_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            device_map={"": 0} if self.device_map == "single" else "balanced",
        )
        if self.device_map == "balanced":
            placements = set(getattr(self.model, "hf_device_map", {}).values())
            if placements & {"cpu", "disk"}:
                raise RuntimeError(
                    f"Balanced Qwen loading used forbidden CPU/disk offload: {sorted(map(str, placements))}"
                )
            cuda_devices = {
                int(value)
                for value in placements
                if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
            }
            if self.profile.default_tensor_parallel_size == 2 and len(cuda_devices) != 2:
                raise RuntimeError(
                    f"Profile {self.profile.key} must be distributed over two visible GPUs; "
                    f"got device map {getattr(self.model, 'hf_device_map', {})}"
                )
        if self.checkpoint:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.checkpoint,
                revision=self.checkpoint_revision,
                is_trainable=False,
                local_files_only=True,
            )
        self.model.eval()
        self._runtime_versions.update({"torch": torch.__version__, "peft": peft.__version__})

    def metadata(self) -> dict[str, Any]:
        self._ensure_processor()
        assert self.processor is not None
        checkpoint = str(Path(self.checkpoint).resolve()) if self.checkpoint else None
        return {
            "model": self.base_model,
            "model_revision": self.base_model_revision,
            "checkpoint": checkpoint,
            "checkpoint_revision": self.checkpoint_revision,
            "backend": "transformers-qwen25-vl-peft",
            "profile": self.profile.key,
            "input_profile": "question_only",
            "inference_protocol": self.profile.inference_protocol,
            "chat_template": self.processor.chat_template,
            "system_prompt": (
                "You are a helpful assistant."
                if "You are a helpful assistant." in str(self.processor.chat_template)
                else None
            ),
            "image_processing": {
                "source": "MSMU RGB only",
                "image_count": 1,
                "min_pixels": self.image_min_pixels,
                "max_pixels": self.image_max_pixels,
            },
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.max_new_tokens,
                "use_cache": True,
            },
            "batch_size": self.batch_size,
            "device_map": self.device_map,
            "model_snapshot_revision_verified": self.base_model_snapshot_revision_verified,
            "processor_class": type(self.processor).__name__,
            "runtime": self._runtime_versions,
            "upstream": {
                "model_repository": self.base_model,
                "adapter_repository": self.checkpoint,
            },
        }

    def generate_batch(self, model_inputs: Sequence[MSMUModelInput]) -> list[GenerationResult]:
        self._ensure_model()
        assert self.model is not None and self.processor is not None and self.collator is not None
        import torch

        batch = self.collator(model_inputs)
        inputs = {
            key: value.to(self.model.device, non_blocking=True)
            for key, value in batch["model_inputs"].items()
        }
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        predictions = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [
            GenerationResult(
                text=prediction,
                metadata={"finish_reason": None, "num_model_image_tensors": 1},
                warnings=("model returned an empty text completion",) if not prediction.strip() else (),
            )
            for prediction in predictions
        ]


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
    execute_msmu_cli(args, adapter)


if __name__ == "__main__":
    main()
