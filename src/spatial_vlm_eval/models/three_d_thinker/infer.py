"""3DThinker MindCube-trained stage-1 checkpoint inference for MSMU."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

from ...benchmarks.msmu.data import MSMUModelInput
from ..common.cli import add_msmu_run_arguments, execute_msmu_cli
from ..common.provenance import verify_git_checkout, verify_hf_snapshot_revision
from ..common.runtime import GenerationResult, InferenceAdapter
from ..profiles import get_profile

THREEDTHINKER_REVISION = "69a70411605f86ec69bada0a625bb96ddee995d9"
MENTAL_3D_CONTROL_PROMPT = (
    "\nFirst imagine the mental 3D scene, think about the reasoning process in the mind and then "
    "provide the user with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively. Special tokens should be used "
    "to represent mental 3D scene at the beginning of your response, i.e., mental 3D scene here "
    "<think> reasoning process here </think><answer> answer here </answer>."
)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE)


def three_d_thinker_prompt(profile_key: str, question: str) -> str:
    if profile_key == "3dthinker":
        return str(question)
    if profile_key == "3dthinker_native":
        return str(question) + MENTAL_3D_CONTROL_PROMPT
    raise ValueError(f"Unknown 3DThinker profile: {profile_key}")


def extract_last_complete_answer(response: str) -> tuple[str, bool]:
    """Extract the last complete answer tag, falling back to the full raw response."""

    matches = list(_ANSWER_PATTERN.finditer(str(response)))
    if not matches:
        return str(response).strip(), False
    return matches[-1].group(1).strip(), True


def place_input_image(
    text: str,
    *,
    image_pad: str = "<|vision_start|><|image_pad|><|vision_end|>",
    image_placeholder: str = "<image>",
) -> str:
    """Equivalent to the upstream stage-1 helper with ``sep_token=None``."""

    return str(text).replace(image_pad, "").replace(image_placeholder, image_pad)


def _pretrained_kwargs(path_or_id: str, revision: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": True}
    if not Path(path_or_id).exists():
        kwargs["revision"] = revision
    return kwargs


def ensure_processor_chat_template(processor: Any) -> str:
    """Expose the checkpoint tokenizer template through newer processor APIs.

    The public MindCube snapshot stores its native Qwen template in
    ``tokenizer_config.json`` but has no processor-level ``chat_template.json``.
    Recent Transformers releases require the latter attribute when
    ``processor.apply_chat_template`` is called, even though the tokenizer loaded
    the exact checkpoint template successfully.
    """

    processor_template = getattr(processor, "chat_template", None)
    if processor_template:
        return str(processor_template)
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_template = getattr(tokenizer, "chat_template", None)
    if not tokenizer_template:
        raise ValueError("3DThinker checkpoint exposes no native chat template")
    processor.chat_template = tokenizer_template
    return str(tokenizer_template)


class ThreeDThinkerAdapter(InferenceAdapter):
    batch_size = 1
    supports_concurrency = False

    def __init__(
        self,
        *,
        profile_key: str,
        upstream_root: str,
        model_path: str,
        model_revision: str = THREEDTHINKER_REVISION,
        device_map: str = "auto",
    ) -> None:
        if profile_key not in {"3dthinker", "3dthinker_native"}:
            raise ValueError("3DThinker profile must be 3dthinker or 3dthinker_native")
        if model_revision != THREEDTHINKER_REVISION:
            raise ValueError(f"3DThinker-Mindcube is locked to {THREEDTHINKER_REVISION}")
        self.profile = get_profile(profile_key)
        self.upstream_root = Path(upstream_root).resolve()
        if not (self.upstream_root / "tests" / "infer.py").exists():
            raise FileNotFoundError(f"Not a 3DThinker upstream checkout: {self.upstream_root}")
        self.model_path = str(model_path)
        self.model_revision = str(model_revision)
        self.model_snapshot_revision_verified = verify_hf_snapshot_revision(
            self.model_path,
            self.model_revision,
            "3DThinker-Mindcube",
        )
        self.upstream_commit_verified = verify_git_checkout(
            self.upstream_root,
            self.profile.upstream_commit or "",
            "3DThinker",
        )
        self.device_map = str(device_map)
        self._loaded = False

    def metadata(self) -> dict[str, Any]:
        native = self.profile.key == "3dthinker_native"
        return {
            "model": "jankin123/3DThinker-Mindcube (MindCube-trained stage-1 checkpoint)",
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "backend": "official-modified-transformers-qwen25-vl",
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "inference_protocol": self.profile.inference_protocol,
            "chat_template": self.profile.chat_template,
            "chat_template_resolution": (
                "checkpoint processor template, or the identical tokenizer_config.json "
                "template when recent Transformers does not surface it on the processor"
            ),
            "image_processing": {
                "source": "MSMU RGB only",
                "image_count": 1,
                "processor_use_fast": False,
                "external_3d_input": None,
                "mental_3d_is_model_generated": native,
            },
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.profile.max_new_tokens,
                "use_cache": True,
                "answer_extraction": "last complete <answer> tag" if native else None,
                "missing_answer_tag": "preserve raw response with warning" if native else None,
            },
            "control_prompt": MENTAL_3D_CONTROL_PROMPT if native else None,
            "checkpoint_scope_warning": (
                "Public checkpoint is stage-1 and MindCube-trained; it is not the paper's complete final model."
            ),
            "upstream": {
                "repository": self.profile.upstream_url,
                "commit": self.profile.upstream_commit,
                "commit_verified": self.upstream_commit_verified,
                "checkout": str(self.upstream_root),
                "model_snapshot_revision_verified": self.model_snapshot_revision_verified,
                "entrypoint_equivalent": "tests/infer.py begin-position mental-3D profile",
            },
        }

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.transformers_version = transformers.__version__
        kwargs = _pretrained_kwargs(self.model_path, self.model_revision)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
            **kwargs,
        )
        ensure_processor_chat_template(self.processor)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **kwargs,
        )
        for token in ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]:
            self.processor.tokenizer.add_tokens(token, special_tokens=True)
        self.model.eval()
        self._loaded = True

    @staticmethod
    def _strip_terminal_tokens(text: str) -> str:
        cleaned = str(text).strip()
        changed = True
        while changed:
            changed = False
            for token in ["<|im_end|>", "<|endoftext|>"]:
                if cleaned.endswith(token):
                    cleaned = cleaned[: -len(token)].strip()
                    changed = True
        return cleaned

    def generate(self, model_input: MSMUModelInput) -> GenerationResult:
        self._load()
        from qwen_vl_utils import process_vision_info

        prompt = three_d_thinker_prompt(self.profile.key, model_input.question)
        content = [
            {"type": "image", "image": model_input.image.convert("RGB")},
            {"type": "text", "text": "<image>" + prompt},
        ]
        conversations = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(conversations, tokenize=False)
        rendered = place_input_image(rendered)
        image_inputs, _ = process_vision_info(conversations)
        inputs = self.processor(
            text=[rendered + "<|im_start|>assistant"],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.profile.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                tokenizer=self.processor.tokenizer,
            )
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        raw_response = self.processor.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=False,
        )
        raw_response = self._strip_terminal_tokens(raw_response)
        warnings: tuple[str, ...] = ()
        extracted = False
        prediction = raw_response
        if self.profile.key == "3dthinker_native":
            prediction, extracted = extract_last_complete_answer(raw_response)
            if not extracted:
                warnings = ("no complete <answer>...</answer> tag; raw response preserved as prediction",)
        if not prediction.strip():
            warnings += ("model returned an empty text completion",)
        return GenerationResult(
            text=prediction,
            metadata={
                "num_model_image_tensors": 1,
                "mental_3d_control_prompt": self.profile.key == "3dthinker_native",
                "answer_tag_extracted": extracted,
                "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
                "raw_response_characters": len(raw_response),
                "transformers_version": self.transformers_version,
            },
            warnings=warnings,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=["3dthinker", "3dthinker_native"])
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=THREEDTHINKER_REVISION)
    parser.add_argument("--device-map", default="auto")
    add_msmu_run_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = ThreeDThinkerAdapter(
        profile_key=args.profile,
        upstream_root=args.upstream_root,
        model_path=args.model,
        model_revision=args.model_revision,
        device_map=args.device_map,
    )
    execute_msmu_cli(args, adapter)


if __name__ == "__main__":
    main()
