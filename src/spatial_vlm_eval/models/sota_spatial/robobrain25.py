"""RoboBrain2.5 official general-VQA inference under the restricted MSMU input."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from ..common.provenance import verify_git_checkout, verify_hf_snapshot_revision
from ..common.runtime import GenerationResult, InferenceAdapter, RestrictedVisionInput, pixel_sha256
from ..profiles import PROFILES
from .common import (
    adapter_source_digest,
    close_torch_model,
    generation_kwargs,
    run_msmu_vision_canary,
    seed_everything,
)


ROBOBRAIN_PROFILE_KEYS = (
    "robobrain25_8b_nv_rgb",
    "robobrain25_8b_mt_rgb",
)


class RoboBrain25Adapter(InferenceAdapter):
    batch_size = 1
    supports_concurrency = False

    def __init__(self, *, profile_key: str, model_path: str, upstream_root: str) -> None:
        if profile_key not in ROBOBRAIN_PROFILE_KEYS:
            raise ValueError(f"Unsupported RoboBrain2.5 MSMU profile: {profile_key}")
        self.profile = PROFILES[profile_key]
        self.model_path = str(model_path)
        self.upstream_root = Path(upstream_root).resolve()
        if not verify_hf_snapshot_revision(
            self.model_path, self.profile.revision, self.profile.model
        ):
            raise ValueError("RoboBrain2.5 local checkpoint revision is not verifiable")
        if not verify_git_checkout(
            self.upstream_root,
            self.profile.upstream_commit or "",
            "RoboBrain2.5",
        ):
            raise ValueError("RoboBrain2.5 upstream checkout is not verifiable")
        self._loaded = False

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.profile.model,
            "model_revision": self.profile.revision,
            "model_path": self.model_path,
            "backend": "official-transformers-auto-image-text-to-text",
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "inference_protocol": self.profile.inference_protocol,
            "adapter_source_sha256": adapter_source_digest(self.profile.key),
            "chat_template": self.profile.chat_template,
            "image_processing": {
                "source": "current MSMU RGB only",
                "source_image_count": 1,
                "model_image_tensor_count": 1,
                "task": "general",
                "external_depth_or_xyz": None,
            },
            "decoding": {**generation_kwargs(self.profile.key), "seed": self.profile.seed},
            "upstream": {
                "repository": self.profile.upstream_url,
                "commit": self.profile.upstream_commit,
                "checkout": str(self.upstream_root),
                "commit_verified": True,
                "model_snapshot_revision_verified": True,
                "entrypoint_equivalent": "inference.py UnifiedInference.inference(task='general')",
            },
        }

    def _load(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model.eval()
        self._loaded = True

    def generate_batch(
        self, model_inputs: Sequence[RestrictedVisionInput]
    ) -> list[GenerationResult]:
        if len(model_inputs) != 1:
            raise ValueError("RoboBrain2.5 MSMU inference is locked to batch size 1")
        return [self.generate(model_inputs[0])]

    def generate(self, model_input: RestrictedVisionInput) -> GenerationResult:
        self._load()
        import torch
        from qwen_vl_utils import process_vision_info

        seed_everything(self.profile.seed)
        rgb = model_input.image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb},
                    {"type": "text", "text": str(model_input.question)},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages)
        if len(images or []) != 1 or videos:
            raise ValueError("RoboBrain2.5 processor did not resolve exactly one image")
        inputs = self.processor(
            text=[rendered],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **generation_kwargs(self.profile.key),
            )
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated, strict=True)
        ]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = decoded[0] if decoded else ""
        return GenerationResult(
            text=text,
            metadata={
                "num_model_image_tensors": 1,
                "source_rgb_sha256": pixel_sha256(rgb),
                "template_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "seed": self.profile.seed,
            },
            warnings=("model returned an empty text completion",) if not text.strip() else (),
        )

    def run_vision_canary(self, output: str | Path) -> dict[str, Any]:
        return run_msmu_vision_canary(self, output, native_batch_probe=False)

    def close(self) -> None:
        close_torch_model(self, ("model", "processor"))
        self._loaded = False
