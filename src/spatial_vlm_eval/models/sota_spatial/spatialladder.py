"""SpatialLadder official Qwen2.5-VL direct and generic-thinking MSMU tracks."""

from __future__ import annotations

import hashlib
import re
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


SPATIALLADDER_PROFILE_KEYS = (
    "spatialladder3b_rgb",
    "spatialladder3b_thinking",
)
SPATIALLADDER_MIN_PIXELS = 16 * 28 * 28
SPATIALLADDER_MAX_PIXELS = 512 * 28 * 28
SPATIALLADDER_PADDING_SIDE = "left"
SPATIALLADDER_THINKING_TEMPLATE = (
    "Question: {question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', "
    "'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. \n"
)
SPATIALLADDER_GENERIC_SPECIAL_POST_PROMPT = (
    "Please provide your detailed reasoning between the <think> </think> tags, "
    "and then answer the question simply within the <answer> </answer> tags."
)
_ANSWER_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)


def prepare_spatialladder_config(config: Any) -> Any:
    text_config = getattr(config, "text_config", None)
    tied = (
        text_config.get("tie_word_embeddings", False)
        if isinstance(text_config, dict)
        else getattr(text_config, "tie_word_embeddings", False)
    )
    if not bool(tied):
        raise ValueError("SpatialLadder checkpoint requires tied text output embeddings")
    config.tie_word_embeddings = True
    if isinstance(text_config, dict) and "text_config" not in (
        getattr(type(config), "sub_configs", {}) or {}
    ):
        delattr(config, "text_config")
    return config


def prepare_spatialladder_processor(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("SpatialLadder processor must expose its tokenizer")
    tokenizer.padding_side = SPATIALLADDER_PADDING_SIDE
    if tokenizer.padding_side != SPATIALLADDER_PADDING_SIDE:
        raise ValueError("SpatialLadder tokenizer must use official left padding")
    return processor


def spatialladder_prompt(profile_key: str, question: str) -> str:
    if profile_key == "spatialladder3b_rgb":
        return str(question)
    if profile_key == "spatialladder3b_thinking":
        return (
            SPATIALLADDER_THINKING_TEMPLATE.format(question=str(question))
            + "\n"
            + SPATIALLADDER_GENERIC_SPECIAL_POST_PROMPT
        )
    raise ValueError(f"Unsupported SpatialLadder MSMU profile: {profile_key}")


def extract_last_complete_answer(response: str) -> tuple[str, bool]:
    matches = list(_ANSWER_PATTERN.finditer(str(response)))
    if not matches:
        return str(response).strip(), False
    return matches[-1].group(1).strip(), True


def select_spatialladder_prediction(
    profile_key: str,
    raw_response: str,
    *,
    index: int,
) -> tuple[str, bool, tuple[str, ...]]:
    if profile_key != "spatialladder3b_thinking":
        return str(raw_response).strip(), False, ()
    prediction, extracted = extract_last_complete_answer(raw_response)
    warnings = (
        ()
        if extracted
        else ("no complete <answer>...</answer> tag; raw response preserved as prediction",)
    )
    return prediction, extracted, warnings


class SpatialLadderAdapter(InferenceAdapter):
    supports_concurrency = False

    def __init__(
        self,
        *,
        profile_key: str,
        model_path: str,
        upstream_root: str,
        batch_size: int = 1,
    ) -> None:
        if profile_key not in SPATIALLADDER_PROFILE_KEYS:
            raise ValueError(f"Unsupported SpatialLadder MSMU profile: {profile_key}")
        if int(batch_size) <= 0:
            raise ValueError("SpatialLadder batch_size must be positive")
        self.profile = PROFILES[profile_key]
        self.model_path = str(model_path)
        self.upstream_root = Path(upstream_root).resolve()
        self.batch_size = int(batch_size)
        if not verify_hf_snapshot_revision(
            self.model_path, self.profile.revision, self.profile.model
        ):
            raise ValueError("SpatialLadder local checkpoint revision is not verifiable")
        if not verify_git_checkout(
            self.upstream_root,
            self.profile.upstream_commit or "",
            "SpatialLadder",
        ):
            raise ValueError("SpatialLadder upstream checkout is not verifiable")
        self._loaded = False

    def metadata(self) -> dict[str, Any]:
        thinking = self.profile.key == "spatialladder3b_thinking"
        return {
            "model": self.profile.model,
            "model_revision": self.profile.revision,
            "model_path": self.model_path,
            "backend": "official-transformers-qwen25-vl-native-batch",
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "inference_protocol": self.profile.inference_protocol,
            "adapter_source_sha256": adapter_source_digest(self.profile.key),
            "chat_template": self.profile.chat_template,
            "image_processing": {
                "source": "current MSMU RGB only",
                "source_image_count": 1,
                "model_image_tensor_count": 1,
                "min_pixels": SPATIALLADDER_MIN_PIXELS,
                "max_pixels": SPATIALLADDER_MAX_PIXELS,
                "processor_use_fast": True,
                "attention_implementation": "flash_attention_2",
                "dtype": "bfloat16",
                "tokenizer_padding_side": SPATIALLADDER_PADDING_SIDE,
                "external_depth_or_xyz": None,
            },
            "decoding": {
                **generation_kwargs(self.profile.key),
                "seed": self.profile.seed,
                "native_batch": True,
                "native_batch_size": self.batch_size,
                "fixed_dataset_order": True,
                "answer_extraction": "last complete <answer> tag" if thinking else None,
                "missing_answer_tag": "preserve raw response with warning" if thinking else None,
            },
            "prompt": {
                "variant": "official SPAR-Bench generic special thinking" if thinking else "MSMU original first question direct",
                "choice_letter_instruction": False,
            },
            "upstream": {
                "repository": self.profile.upstream_url,
                "commit": self.profile.upstream_commit,
                "checkout": str(self.upstream_root),
                "commit_verified": True,
                "model_snapshot_revision_verified": True,
                "entrypoint_equivalent": "eval_spld/data_utils/sparbench.py native left-padded batch",
            },
        }

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        config = prepare_spatialladder_config(config)
        self.processor = prepare_spatialladder_processor(
            AutoProcessor.from_pretrained(
                self.model_path,
                use_fast=True,
                min_pixels=SPATIALLADDER_MIN_PIXELS,
                max_pixels=SPATIALLADDER_MAX_PIXELS,
                local_files_only=True,
            )
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            local_files_only=True,
        ).eval()
        self._loaded = True

    def generate_batch(
        self, model_inputs: Sequence[RestrictedVisionInput]
    ) -> list[GenerationResult]:
        self._load()
        import torch
        from qwen_vl_utils import process_vision_info

        if self.processor.tokenizer.padding_side != SPATIALLADDER_PADDING_SIDE:
            raise RuntimeError("SpatialLadder tokenizer padding side changed after model load")
        seed_everything(self.profile.seed)
        messages_list = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": value.image.convert("RGB")},
                        {
                            "type": "text",
                            "text": spatialladder_prompt(self.profile.key, value.question),
                        },
                    ],
                }
            ]
            for value in model_inputs
        ]
        rendered = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        if len(image_inputs) != len(model_inputs):
            raise ValueError("SpatialLadder processor did not resolve one image per MSMU input")
        inputs = self.processor(
            text=rendered,
            images=image_inputs,
            videos=video_inputs or None,
            padding=True,
            return_tensors="pt",
            min_pixels=SPATIALLADDER_MIN_PIXELS,
            max_pixels=SPATIALLADDER_MAX_PIXELS,
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **generation_kwargs(self.profile.key),
            )
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated, strict=True)
        ]
        raw_responses = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results: list[GenerationResult] = []
        for model_input, prompt, raw_response in zip(
            model_inputs, rendered, raw_responses, strict=True
        ):
            prediction, extracted, warnings = select_spatialladder_prediction(
                self.profile.key,
                raw_response,
                index=int(model_input.index),
            )
            if not prediction.strip():
                warnings += ("model returned an empty text completion",)
            results.append(
                GenerationResult(
                    text=prediction,
                    metadata={
                        "num_model_image_tensors": 1,
                        "native_batch_size": len(model_inputs),
                        "attention_implementation": "flash_attention_2",
                        "tokenizer_padding_side": self.processor.tokenizer.padding_side,
                        "source_rgb_sha256": pixel_sha256(model_input.image),
                        "template_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "raw_response_sha256": hashlib.sha256(
                            str(raw_response).encode("utf-8")
                        ).hexdigest(),
                        "raw_response_characters": len(str(raw_response)),
                        "answer_tag_extracted": extracted,
                    },
                    warnings=warnings,
                )
            )
        return results

    def generate(self, model_input: RestrictedVisionInput) -> GenerationResult:
        return self.generate_batch([model_input])[0]

    def run_vision_canary(self, output: str | Path) -> dict[str, Any]:
        return run_msmu_vision_canary(self, output, native_batch_probe=True)

    def close(self) -> None:
        close_torch_model(self, ("model", "processor"))
        self._loaded = False
