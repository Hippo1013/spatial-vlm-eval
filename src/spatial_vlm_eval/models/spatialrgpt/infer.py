"""Official VILA/SpatialRGPT RGB-only inference path for MSMU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ...benchmarks.msmu.data import MSMUModelInput
from ..common.cli import add_msmu_run_arguments, execute_msmu_cli
from ..common.provenance import verify_git_checkout, verify_hf_snapshot_revision
from ..common.runtime import GenerationResult, InferenceAdapter
from ..profiles import get_profile

SPATIALRGPT_REVISION = "64df7902f82b5053f5a53455095805e6de3a1f87"
CONVERSATION_MODE = "llama_3"


def spatialrgpt_question(question: str, *, use_image_start_end: bool) -> str:
    if use_image_start_end:
        return f"<im_start><image><im_end>\n{question}"
    return f"<image>\n{question}"


class SpatialRGPTAdapter(InferenceAdapter):
    batch_size = 1
    supports_concurrency = False

    def __init__(
        self,
        *,
        upstream_root: str,
        model_path: str,
        model_revision: str = SPATIALRGPT_REVISION,
        model_base: str | None = None,
        conversation_mode: str = CONVERSATION_MODE,
        device: str = "cuda",
    ) -> None:
        if model_revision != SPATIALRGPT_REVISION:
            raise ValueError(f"SpatialRGPT checkpoint is locked to {SPATIALRGPT_REVISION}")
        if conversation_mode != CONVERSATION_MODE:
            raise ValueError(f"SpatialRGPT-VILA1.5-8B requires conversation mode {CONVERSATION_MODE!r}")
        self.profile = get_profile("spatialrgpt")
        self.upstream_root = Path(upstream_root).resolve()
        if not (self.upstream_root / "llava" / "eval" / "model_vqa.py").exists():
            raise FileNotFoundError(f"Not a SpatialRGPT upstream checkout: {self.upstream_root}")
        model_location = Path(model_path).expanduser()
        if not model_location.exists():
            raise FileNotFoundError(
                "SpatialRGPT's upstream loader does not accept an HF revision argument; "
                f"provide the locked local snapshot, got {model_path!r}"
            )
        self.model_path = str(model_location.resolve())
        self.model_revision = str(model_revision)
        self.model_snapshot_revision_verified = verify_hf_snapshot_revision(
            self.model_path,
            self.model_revision,
            "SpatialRGPT-VILA1.5-8B",
        )
        self.upstream_commit_verified = verify_git_checkout(
            self.upstream_root,
            self.profile.upstream_commit or "",
            "SpatialRGPT",
        )
        self.model_base = str(model_base) if model_base else None
        self.conversation_mode = str(conversation_mode)
        self.device_name = str(device)
        self._loaded = False

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.profile.model,
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "model_base": self.model_base,
            "backend": "official-spatialrgpt-vila",
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "inference_protocol": self.profile.inference_protocol,
            "chat_template": f"SpatialRGPT conv_templates[{self.conversation_mode!r}]",
            "image_processing": {
                "source": "MSMU RGB only",
                "image_count": 1,
                "region_masks": None,
                "depth": None,
                "processor": "checkpoint vision tower image_processor.preprocess",
            },
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.profile.max_new_tokens,
                "use_cache": True,
            },
            "upstream": {
                "repository": self.profile.upstream_url,
                "commit": self.profile.upstream_commit,
                "commit_verified": self.upstream_commit_verified,
                "checkout": str(self.upstream_root),
                "model_snapshot_revision_verified": self.model_snapshot_revision_verified,
                "entrypoint_equivalent": "llava/eval/model_vqa.py",
            },
        }

    def _load(self) -> None:
        if self._loaded:
            return
        if str(self.upstream_root) not in sys.path:
            sys.path.insert(0, str(self.upstream_root))
        import torch
        from llava.constants import (
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path,
            model_name,
            self.model_base,
            device_map="auto",
            device=self.device_name,
        )
        self.torch = torch
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token
        self._loaded = True

    def generate(self, model_input: MSMUModelInput) -> GenerationResult:
        self._load()
        use_start_end = bool(getattr(self.model.config, "mm_use_im_start_end", False))
        if use_start_end:
            question = (
                self.DEFAULT_IM_START_TOKEN
                + self.DEFAULT_IMAGE_TOKEN
                + self.DEFAULT_IM_END_TOKEN
                + "\n"
                + model_input.question
            )
        else:
            question = self.DEFAULT_IMAGE_TOKEN + "\n" + model_input.question
        if "<mask>" in question or "<depth>" in question:
            raise ValueError("MSMU SpatialRGPT prompt unexpectedly contains a region/depth token")
        conv = self.conv_templates[self.conversation_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            self.tokenizer_image_token(
                prompt,
                self.tokenizer,
                self.IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        image = model_input.image.convert("RGB")
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(self.model.device, dtype=self.torch.float16),
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.profile.max_new_tokens,
                use_cache=True,
            )
        if output_ids.shape[1] >= input_ids.shape[1] and self.torch.equal(
            output_ids[:, : input_ids.shape[1]], input_ids
        ):
            output_ids = output_ids[:, input_ids.shape[1] :]
        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        stop = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        if stop and prediction.endswith(stop):
            prediction = prediction[: -len(stop)].strip()
        warnings = ("model returned an empty text completion",) if not prediction else ()
        return GenerationResult(
            text=prediction,
            metadata={
                "num_model_image_tensors": 1,
                "region_masks": 0,
                "depth_tensors": 0,
                "conversation_mode": self.conversation_mode,
            },
            warnings=warnings,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=SPATIALRGPT_REVISION)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conversation-mode", default=CONVERSATION_MODE)
    parser.add_argument("--device", default="cuda")
    add_msmu_run_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = SpatialRGPTAdapter(
        upstream_root=args.upstream_root,
        model_path=args.model,
        model_revision=args.model_revision,
        model_base=args.model_base,
        conversation_mode=args.conversation_mode,
        device=args.device,
    )
    execute_msmu_cli(args, adapter)


if __name__ == "__main__":
    main()
