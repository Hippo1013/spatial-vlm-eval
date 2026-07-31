"""Shared Qwen-VL adapter mechanics with family-specific protocol locks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from ...benchmarks.msmu.data import MSMUModelInput, QwenGenerationCollator
from ..common.provenance import verify_hf_snapshot_revision
from ..common.runtime import GenerationResult, InferenceAdapter, atomic_write_json
from ..common.vision_canary import SOLID_COLOR_QUESTION, validate_solid_color_answers
from ..profiles import get_profile


class QwenVLAdapterBase(InferenceAdapter):
    """Common processor, model loading, metadata, and generation path."""

    supports_concurrency = False

    PROFILE_FAMILY: ClassVar[str]
    MODEL_CLASS_NAME: ClassVar[str]
    MODEL_LABEL: ClassVar[str]
    BACKEND_NAME: ClassVar[str]
    CHAT_TEMPLATE_DESCRIPTION: ClassVar[str]
    TRUST_REMOTE_CODE: ClassVar[bool] = False
    SUPPORTS_PEFT: ClassVar[bool] = False
    REQUIRED_TRANSFORMERS: ClassVar[str]
    PIXEL_CONFIG_STYLE: ClassVar[str] = "legacy_attrs"

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
        expected_max_new_tokens: int,
        expected_image_min_pixels: int,
        expected_image_max_pixels: int,
    ) -> None:
        self.profile = get_profile(profile_key)
        if self.profile.family != self.PROFILE_FAMILY:
            raise ValueError(
                f"Profile {profile_key!r} is not a {self.MODEL_LABEL} profile"
            )
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
            f"{self.MODEL_LABEL} base model",
        )
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.checkpoint_revision = str(checkpoint_revision) if checkpoint_revision else None
        if self.checkpoint and not self.SUPPORTS_PEFT:
            raise ValueError(f"{self.MODEL_LABEL} adapter does not accept a PEFT checkpoint")
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
            int(expected_max_new_tokens),
            int(expected_image_min_pixels),
            int(expected_image_max_pixels),
        )
        if locked != expected:
            raise ValueError(
                f"The canonical {self.MODEL_LABEL} inference protocol is locked to "
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
            trust_remote_code=self.TRUST_REMOTE_CODE,
            local_files_only=True,
        )
        self.collator = QwenGenerationCollator(
            self.processor,
            image_min_pixels=self.image_min_pixels,
            image_max_pixels=self.image_max_pixels,
            pixel_config_style=self.PIXEL_CONFIG_STYLE,
        )
        self._runtime_versions["transformers"] = transformers.__version__

    def _model_class(self, transformers_module: Any) -> Any:
        try:
            return getattr(transformers_module, self.MODEL_CLASS_NAME)
        except AttributeError as exc:
            raise RuntimeError(
                f"{self.MODEL_LABEL} requires transformers {self.REQUIRED_TRANSFORMERS}; "
                f"{self.MODEL_CLASS_NAME} is unavailable"
            ) from exc

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        self._ensure_processor()
        import torch
        import transformers

        model_class = self._model_class(transformers)
        self.model = model_class.from_pretrained(
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
                    f"Balanced Qwen loading used forbidden CPU/disk offload: "
                    f"{sorted(map(str, placements))}"
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
        if self.SUPPORTS_PEFT:
            import peft

            self._runtime_versions["peft"] = peft.__version__
            if self.checkpoint:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.checkpoint,
                    revision=self.checkpoint_revision,
                    is_trainable=False,
                    local_files_only=True,
                )
        self.model.eval()
        self._runtime_versions["torch"] = torch.__version__

    def metadata(self) -> dict[str, Any]:
        self._ensure_processor()
        assert self.processor is not None
        checkpoint = str(Path(self.checkpoint).resolve()) if self.checkpoint else None
        return {
            "model": self.base_model,
            "model_revision": self.base_model_revision,
            "checkpoint": checkpoint,
            "checkpoint_revision": self.checkpoint_revision,
            "backend": self.BACKEND_NAME,
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

    def run_vision_canary(self, output: str | Path) -> dict[str, Any]:
        """Run two semantic image probes through the same loaded model and processor."""

        from PIL import Image

        # Establish the same processor/runtime metadata as an ordinary MSMU run
        # before model loading adds transient runtime fields such as torch.
        self._ensure_processor()
        runtime_versions_before_canary = dict(self._runtime_versions)
        answers: dict[str, GenerationResult] = {}
        try:
            for color in ("red", "blue"):
                result = self.generate_batch(
                    [
                        MSMUModelInput(
                            index=-1,
                            image=Image.new("RGB", (64, 64), color),
                            question=SOLID_COLOR_QUESTION,
                        )
                    ]
                )
                if len(result) != 1:
                    raise ValueError(
                        f"Qwen vision canary expected one {color} result; got {len(result)}"
                    )
                if result[0].metadata.get("num_model_image_tensors") != 1:
                    raise ValueError(
                        f"Qwen vision canary {color} probe did not report exactly one image tensor"
                    )
                answers[color] = result[0]
            validate_solid_color_answers(answers["red"].text, answers["blue"].text)
        finally:
            # Loading the canary model must not change the MSMU journal identity.
            self._runtime_versions = runtime_versions_before_canary
        report = {
            "passed": True,
            "profile": self.profile.key,
            "model": self.base_model,
            "model_revision": self.base_model_revision,
            "inference_protocol": self.profile.inference_protocol,
            "question": SOLID_COLOR_QUESTION,
            "request_image_count": 1,
            "image_mode": "RGB",
            "image_size": [64, 64],
            "red_answer": answers["red"].text,
            "blue_answer": answers["blue"].text,
            "generation": {
                color: dict(answers[color].metadata) for color in ("red", "blue")
            },
        }
        atomic_write_json(Path(output).resolve(), report)
        return report
