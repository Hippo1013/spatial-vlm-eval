"""Persistent, dataset-blind runners for the twelve CV-Bench specialized tracks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ...models.common.provenance import verify_git_checkout, verify_hf_snapshot_revision
from ...models.common.runtime import GenerationResult, InferenceAdapter
from .command_adapter import load_generation_manifest
from .profiles import PROFILE_SEQUENCE, PROFILES, CVBenchProfile

SPECIALIZED_PROFILE_KEYS = tuple(
    key for key in PROFILE_SEQUENCE if PROFILES[key].adapter_kind == "upstream_command"
)
MOGE2_MODEL_ID = "Ruicheng/moge-2-vitl-normal"
MOGE2_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MOGE2_UPSTREAM_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
_REQUEST_KEYS = {
    "schema_version",
    "action",
    "index",
    "prompt",
    "image",
    "profile",
    "model_revision",
    "upstream_commit",
    "inference_protocol",
    "decoding",
}
_MODEL_GENERATION_KEYS = {
    "do_sample",
    "num_beams",
    "max_new_tokens",
    "use_cache",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
}


@dataclass(frozen=True, slots=True)
class RunnerModelInput:
    index: int
    image: Image.Image
    question: str


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} for the CV-Bench specialized runner")
    return value


def _model_generation(decoding: dict[str, Any]) -> dict[str, Any]:
    values = {key: value for key, value in decoding.items() if key in _MODEL_GENERATION_KEYS}
    if int(values.get("max_new_tokens", 0)) <= 0:
        raise ValueError("Resolved decoding must contain a positive max_new_tokens")
    if "do_sample" not in values:
        raise ValueError("Resolved decoding must contain do_sample")
    return values


def _seed(decoding: dict[str, Any]) -> None:
    seed = int(decoding.get("seed", 42))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_image(value: Any) -> Image.Image:
    if not isinstance(value, dict) or set(value) != {"count", "mode", "png_data_uri"}:
        raise ValueError("Runner image input must contain only count, mode, and png_data_uri")
    if value["count"] != 1 or value["mode"] != "RGB":
        raise ValueError("Runner accepts exactly one RGB source image")
    prefix = "data:image/png;base64,"
    uri = value["png_data_uri"]
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise ValueError("Runner image must be a PNG data URI")
    raw = base64.b64decode(uri[len(prefix) :], validate=True)
    with Image.open(io.BytesIO(raw)) as loaded:
        image = loaded.convert("RGB")
        image.load()
    return image


def _verify_request(
    request: Any,
    profile: CVBenchProfile,
    decoding: dict[str, Any],
) -> RunnerModelInput:
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS:
        extra = sorted(set(request) - _REQUEST_KEYS) if isinstance(request, dict) else []
        missing = sorted(_REQUEST_KEYS - set(request)) if isinstance(request, dict) else []
        raise ValueError(f"Runner request schema mismatch: extra={extra}, missing={missing}")
    checks = {
        "schema_version": 1,
        "action": "generate",
        "profile": profile.key,
        "model_revision": profile.revision,
        "upstream_commit": profile.upstream_commit,
        "inference_protocol": profile.inference_protocol,
        "decoding": decoding,
    }
    for key, expected in checks.items():
        if request.get(key) != expected:
            raise ValueError(
                f"Runner request {key} mismatch: got={request.get(key)!r}, expected={expected!r}"
            )
    if not isinstance(request.get("index"), int) or not isinstance(request.get("prompt"), str):
        raise ValueError("Runner index must be int and prompt must be string")
    return RunnerModelInput(
        index=request["index"],
        image=_decode_image(request["image"]),
        question=request["prompt"],
    )


def _source_digest_files(profile: CVBenchProfile) -> list[Path]:
    from . import command_adapter, profiles

    files = [Path(__file__), Path(command_adapter.__file__), Path(profiles.__file__)]
    if profile.family == "ssr":
        from ...models.ssr import infer

        files.append(Path(infer.__file__))
    elif profile.family == "spatialrgpt":
        from ...models.spatialrgpt import infer

        files.append(Path(infer.__file__))
    elif profile.family == "3dthinker":
        from ...models.three_d_thinker import infer

        files.append(Path(infer.__file__))
    elif profile.family == "spatialbot":
        from ...models.spatialbot import infer

        files.append(Path(infer.__file__))
    return sorted(set(path.resolve() for path in files), key=str)


def adapter_digest(profile: CVBenchProfile) -> str:
    digest = hashlib.sha256()
    for path in _source_digest_files(profile):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class RoboBrainAdapter(InferenceAdapter):
    def __init__(self, profile: CVBenchProfile, decoding: dict[str, Any]) -> None:
        self.profile = profile
        self.decoding = dict(decoding)
        self.model_path = _required_env(profile.model_path_env)
        self.upstream_root = _required_env("ROBOBRAIN25_UPSTREAM_ROOT")
        if not verify_hf_snapshot_revision(self.model_path, profile.revision, profile.model):
            raise ValueError("RoboBrain local checkpoint revision is not verifiable")
        if not verify_git_checkout(self.upstream_root, profile.upstream_commit or "", "RoboBrain2.5"):
            raise ValueError("RoboBrain upstream checkout is not a verifiable Git checkout")
        self._loaded = False

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
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self._loaded = True

    def generate(self, model_input: RunnerModelInput) -> GenerationResult:
        self._load()
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": model_input.image.convert("RGB")},
                    {"type": "text", "text": model_input.question},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        generated = self.model.generate(**inputs, **_model_generation(self.decoding))
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return GenerationResult(
            text=text,
            metadata={
                "num_model_image_tensors": 1,
                "template_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            },
            warnings=("model returned an empty text completion",) if not text.strip() else (),
        )


class HiSpatialAdapter(InferenceAdapter):
    def __init__(self, profile: CVBenchProfile, decoding: dict[str, Any]) -> None:
        self.profile = profile
        self.decoding = dict(decoding)
        self.model_path = _required_env(profile.model_path_env)
        self.moge_path = _required_env("MOGE2_MODEL")
        self.moge_upstream_root = _required_env("MOGE2_UPSTREAM_ROOT")
        self.upstream_root = _required_env("HISPATIAL_UPSTREAM_ROOT")
        if not verify_hf_snapshot_revision(self.model_path, profile.revision, profile.model):
            raise ValueError("HiSpatial local checkpoint revision is not verifiable")
        if not verify_hf_snapshot_revision(self.moge_path, MOGE2_REVISION, MOGE2_MODEL_ID):
            raise ValueError("MoGe-2 local checkpoint revision is not verifiable")
        if not verify_git_checkout(
            self.moge_upstream_root, MOGE2_UPSTREAM_COMMIT, "MoGe-2"
        ):
            raise ValueError("MoGe-2 upstream checkout is not a verifiable Git checkout")
        if not verify_git_checkout(self.upstream_root, profile.upstream_commit or "", "HiSpatial"):
            raise ValueError("HiSpatial upstream checkout is not a verifiable Git checkout")
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self.upstream_root not in sys.path:
            sys.path.insert(0, self.upstream_root)
        import torch
        from hispatial.inference import HiSpatialPredictor, MoGeProcessor
        from moge.model.v2 import MoGeModel

        self.moge = MoGeProcessor.__new__(MoGeProcessor)
        self.moge.device = torch.device("cuda")
        self.moge.model = MoGeModel.from_pretrained(self.moge_path).to(self.moge.device).eval()
        self.moge.img_size = 448
        self.predictor = HiSpatialPredictor(model_load_path=self.model_path, gpu_rank=0)
        self._loaded = True

    def generate(self, model_input: RunnerModelInput) -> GenerationResult:
        self._load()
        import numpy as np

        image = model_input.image.convert("RGB")
        xyz = self.moge.apply_transform(image)
        text = self.predictor.query(
            image=np.asarray(image), prompt=model_input.question, xyz_values=xyz
        )
        rendered = (
            model_input.question
            if "<image>" in model_input.question
            else "<image>" + model_input.question
        )
        return GenerationResult(
            text=text,
            metadata={
                "num_model_image_tensors": 1,
                "derived_xyz_tensors": 1,
                "derived_xyz_model": MOGE2_MODEL_ID,
                "derived_xyz_revision": MOGE2_REVISION,
                "derived_xyz_upstream_commit": MOGE2_UPSTREAM_COMMIT,
                "template_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            },
            warnings=("model returned an empty text completion",) if not text.strip() else (),
        )


class SpatialLadderAdapter(InferenceAdapter):
    def __init__(self, profile: CVBenchProfile, decoding: dict[str, Any]) -> None:
        self.profile = profile
        self.decoding = dict(decoding)
        self.model_path = _required_env(profile.model_path_env)
        self.upstream_root = _required_env("SPATIALLADDER_UPSTREAM_ROOT")
        if not verify_hf_snapshot_revision(self.model_path, profile.revision, profile.model):
            raise ValueError("SpatialLadder local checkpoint revision is not verifiable")
        if not verify_git_checkout(
            self.upstream_root, profile.upstream_commit or "", "SpatialLadder"
        ):
            raise ValueError("SpatialLadder upstream checkout is not a verifiable Git checkout")
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            use_fast=True,
            min_pixels=16 * 28 * 28,
            max_pixels=512 * 28 * 28,
            local_files_only=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            local_files_only=True,
        ).eval()
        self._loaded = True

    def generate(self, model_input: RunnerModelInput) -> GenerationResult:
        self._load()
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": model_input.image.convert("RGB")},
                    {"type": "text", "text": model_input.question},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            min_pixels=16 * 28 * 28,
            max_pixels=512 * 28 * 28,
        ).to(self.model.device)
        generated = self.model.generate(**inputs, **_model_generation(self.decoding))
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return GenerationResult(
            text=text,
            metadata={
                "num_model_image_tensors": 1,
                "template_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            },
            warnings=("model returned an empty text completion",) if not text.strip() else (),
        )


def _build_backend(profile: CVBenchProfile, decoding: dict[str, Any]) -> InferenceAdapter:
    if profile.family == "ssr":
        from ...models.ssr.infer import SSRAdapter

        native = profile.key == "ssr_native"
        return SSRAdapter(
            profile_key="ssr_native" if native else "ssr",
            upstream_root=_required_env("SSR_UPSTREAM_ROOT"),
            base_model=_required_env("BASE_MODEL"),
            ssr_vlm=_required_env("SSR_VLM_MODEL"),
            ssr_midi=_required_env("SSR_MIDI_MODEL") if native else None,
            clip_model=_required_env("CLIP_MODEL") if native else None,
            siglip_model=_required_env("SIGLIP_MODEL") if native else None,
            mamba_model=_required_env("MAMBA_MODEL") if native else None,
            midi_llm_model=_required_env("MIDI_LLM_MODEL") if native else None,
            depthpro_root=_required_env("SSR_DEPTHPRO_ROOT") if native else None,
            depthpro_checkpoint=_required_env("DEPTHPRO_CHECKPOINT") if native else None,
            generation_kwargs=_model_generation(decoding),
        )
    if profile.family == "spatialrgpt":
        from ...models.spatialrgpt.infer import SpatialRGPTAdapter

        return SpatialRGPTAdapter(
            upstream_root=_required_env("SPATIALRGPT_UPSTREAM_ROOT"),
            model_path=_required_env(profile.model_path_env),
            max_new_tokens=int(decoding["max_new_tokens"]),
        )
    if profile.family == "3dthinker":
        from ...models.three_d_thinker.infer import ThreeDThinkerAdapter

        mental = profile.key == "3dthinker_mental3d"
        return ThreeDThinkerAdapter(
            profile_key="3dthinker_native" if mental else "3dthinker",
            upstream_root=_required_env("THREEDTHINKER_UPSTREAM_ROOT"),
            model_path=_required_env(profile.model_path_env),
            generation_kwargs=_model_generation(decoding),
            control_prompt_already_present=mental,
        )
    if profile.family == "spatialbot":
        from ...models.spatialbot.infer import SpatialBotAdapter

        native = profile.key == "spatialbot_zoedepth"
        return SpatialBotAdapter(
            profile_key="spatialbot_native" if native else "spatialbot",
            upstream_root=_required_env("SPATIALBOT_UPSTREAM_ROOT"),
            model_path=_required_env(profile.model_path_env),
            zoedepth_root=_required_env("ZOEDEPTH_ROOT") if native else None,
            zoedepth_checkpoint=_required_env("ZOEDEPTH_CHECKPOINT") if native else None,
            max_new_tokens=int(decoding["max_new_tokens"]),
        )
    if profile.family == "robobrain":
        return RoboBrainAdapter(profile, decoding)
    if profile.family == "hispatial":
        return HiSpatialAdapter(profile, decoding)
    if profile.family == "spatialladder":
        return SpatialLadderAdapter(profile, decoding)
    raise ValueError(f"No specialized runner backend for {profile.key}")


def _response(
    request: dict[str, Any],
    profile: CVBenchProfile,
    decoding: dict[str, Any],
    result: GenerationResult,
) -> dict[str, Any]:
    generation = dict(result.metadata)
    generation["num_media_prompt"] = 1
    generation["source_rgb_count"] = 1
    template_sha = generation.get("template_sha256")
    if not isinstance(template_sha, str) or len(template_sha) != 64:
        raise ValueError("Specialized backend did not return a rendered template SHA-256")
    return {
        "index": request["index"],
        "profile": profile.key,
        "model_revision": profile.revision,
        "inference_protocol": profile.inference_protocol,
        "decoding": decoding,
        "raw_prediction": result.text,
        "generation": generation,
        "warnings": list(result.warnings),
    }


def serve(profile: CVBenchProfile, decoding: dict[str, Any]) -> None:
    backend: InferenceAdapter | None = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request == {"schema_version": 1, "action": "close"}:
                return
            model_input = _verify_request(request, profile, decoding)
            if backend is None:
                with redirect_stdout(sys.stderr):
                    backend = _build_backend(profile, decoding)
            _seed(decoding)
            with redirect_stdout(sys.stderr):
                result = backend.generate(model_input)
            print(
                json.dumps(
                    _response(request, profile, decoding, result),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )
    finally:
        if backend is not None:
            backend.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=SPECIALIZED_PROFILE_KEYS)
    parser.add_argument("--generation-manifest")
    parser.add_argument("--print-adapter-digest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    if args.print_adapter_digest:
        print(adapter_digest(profile))
        return
    decoding = load_generation_manifest(profile, args.generation_manifest)
    serve(profile, decoding)


if __name__ == "__main__":
    main()
