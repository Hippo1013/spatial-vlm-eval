"""Persistent dataset-blind runners for the SPBench-SI specialized profiles."""

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
from ...models.common.runtime import GenerationResult, InferenceAdapter, pixel_sha256
from .command_adapter import fold_system_user_prompt, load_generation_manifest
from .profiles import PROFILE_SEQUENCE, PROFILES, SPBenchSIProfile

SPECIALIZED_PROFILE_KEYS = tuple(
    key for key in PROFILE_SEQUENCE if PROFILES[key].adapter_kind == "upstream_command"
)
MOGE2_MODEL_ID = "Ruicheng/moge-2-vitl-normal"
MOGE2_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MOGE2_UPSTREAM_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE2_CHECKPOINT_FILENAME = "model.pt"
MOGE2_UTILS3D_COMMIT = "3fab839f0be9931dac7c8488eb0e1600c236e183"
_REQUEST_KEYS = {
    "schema_version", "action", "index", "system_prompt", "user_prompt", "image",
    "profile", "model_revision", "upstream_commit", "inference_protocol", "decoding",
}
_MODEL_GENERATION_KEYS = {
    "do_sample", "num_beams", "max_new_tokens", "use_cache", "temperature", "top_p",
    "top_k", "repetition_penalty",
}


@dataclass(frozen=True, slots=True)
class RunnerModelInput:
    index: int
    image: Image.Image
    question: str
    system_prompt: str
    user_prompt: str
    source_rgb_sha256: str


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} for the SPBench-SI specialized runner")
    return value


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_generation(decoding: dict[str, Any]) -> dict[str, Any]:
    values = {key: value for key, value in decoding.items() if key in _MODEL_GENERATION_KEYS}
    if int(values.get("max_new_tokens", 0)) <= 0 or "do_sample" not in values:
        raise ValueError("Resolved decoding must contain do_sample and positive max_new_tokens")
    return values


def _prepare_spatialladder_config(config: Any) -> Any:
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


def _decode_image(value: Any) -> tuple[Image.Image, str]:
    expected = {"count", "mode", "pixel_sha256", "png_data_uri"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Runner image input schema mismatch")
    if value["count"] != 1 or value["mode"] != "RGB":
        raise ValueError("Runner accepts exactly one RGB source image")
    prefix = "data:image/png;base64,"
    uri = value["png_data_uri"]
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise ValueError("Runner image must be a PNG data URI")
    raw = base64.b64decode(uri[len(prefix):], validate=True)
    with Image.open(io.BytesIO(raw)) as loaded:
        image = loaded.convert("RGB")
        image.load()
    digest = pixel_sha256(image)
    if value["pixel_sha256"] != digest:
        raise ValueError("Runner decoded RGB digest mismatch")
    return image, digest


def _verify_request(
    request: Any, profile: SPBenchSIProfile, decoding: dict[str, Any]
) -> RunnerModelInput:
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS:
        extra = sorted(set(request) - _REQUEST_KEYS) if isinstance(request, dict) else []
        missing = sorted(_REQUEST_KEYS - set(request)) if isinstance(request, dict) else []
        raise ValueError(f"Runner request schema mismatch: extra={extra}, missing={missing}")
    for key, expected in {
        "schema_version": 1, "action": "generate", "profile": profile.key,
        "model_revision": profile.revision, "upstream_commit": profile.upstream_commit,
        "inference_protocol": profile.inference_protocol, "decoding": decoding,
    }.items():
        if request.get(key) != expected:
            raise ValueError(f"Runner request {key} mismatch")
    if not isinstance(request.get("index"), int):
        raise ValueError("Runner index must be an integer")
    system_prompt = request.get("system_prompt")
    user_prompt = request.get("user_prompt")
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        raise ValueError("Runner requires separate system_prompt and user_prompt")
    image, digest = _decode_image(request["image"])
    return RunnerModelInput(
        request["index"], image, fold_system_user_prompt(system_prompt, user_prompt),
        system_prompt, user_prompt, digest,
    )


def _source_digest_files(profile: SPBenchSIProfile) -> list[Path]:
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


def adapter_digest(profile: SPBenchSIProfile) -> str:
    digest = hashlib.sha256()
    for path in _source_digest_files(profile):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class RoboBrainAdapter(InferenceAdapter):
    def __init__(self, profile: SPBenchSIProfile, decoding: dict[str, Any]) -> None:
        self.profile = profile
        self.decoding = dict(decoding)
        self.model_path = _required_env(profile.model_path_env)
        self.upstream_root = _required_env("ROBOBRAIN25_UPSTREAM_ROOT")
        if not verify_hf_snapshot_revision(self.model_path, profile.revision, profile.model):
            raise ValueError("RoboBrain checkpoint revision is not verifiable")
        if not verify_git_checkout(self.upstream_root, profile.upstream_commit or "", "RoboBrain2.5"):
            raise ValueError("RoboBrain upstream checkout is not verifiable")
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, dtype="auto", device_map="auto", local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self._loaded = True

    def generate(self, model_input: RunnerModelInput) -> GenerationResult:
        self._load()
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": model_input.image},
            {"type": "text", "text": model_input.question},
        ]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[rendered], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda")
        generated = self.model.generate(**inputs, **_model_generation(self.decoding))
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return GenerationResult(text, {"num_model_image_tensors": 1, "template_sha256": hashlib.sha256(rendered.encode()).hexdigest()})


class HiSpatialAdapter(InferenceAdapter):
    def __init__(self, profile: SPBenchSIProfile, decoding: dict[str, Any]) -> None:
        self.profile = profile
        self.decoding = dict(decoding)
        self.model_path = _required_env(profile.model_path_env)
        self.moge_path = _required_env("MOGE2_MODEL")
        self.moge_upstream_root = _required_env("MOGE2_UPSTREAM_ROOT")
        self.moge_utils3d_root = _required_env("MOGE2_UTILS3D_ROOT")
        self.upstream_root = _required_env("HISPATIAL_UPSTREAM_ROOT")
        if not verify_hf_snapshot_revision(self.model_path, profile.revision, profile.model):
            raise ValueError("HiSpatial checkpoint revision is not verifiable")
        if not verify_hf_snapshot_revision(self.moge_path, MOGE2_REVISION, MOGE2_MODEL_ID):
            raise ValueError("MoGe-2 checkpoint revision is not verifiable")
        self.moge_checkpoint = str(Path(self.moge_path) / MOGE2_CHECKPOINT_FILENAME)
        if not Path(self.moge_checkpoint).is_file():
            raise FileNotFoundError(f"Locked MoGe-2 checkpoint is missing: {self.moge_checkpoint}")
        for root, commit, label in (
            (self.moge_upstream_root, MOGE2_UPSTREAM_COMMIT, "MoGe-2"),
            (self.moge_utils3d_root, MOGE2_UTILS3D_COMMIT, "MoGe-2 utils3d"),
            (self.upstream_root, profile.upstream_commit or "", "HiSpatial"),
        ):
            if not verify_git_checkout(root, commit, label):
                raise ValueError(f"{label} checkout is not verifiable")
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self.upstream_root not in sys.path:
            sys.path.insert(0, self.upstream_root)
        import torch
        import utils3d
        from hispatial.inference import HiSpatialPredictor, MoGeProcessor
        from moge.model.v2 import MoGeModel
        installed_root = Path(utils3d.__file__).resolve().parent
        source_root = Path(self.moge_utils3d_root).resolve() / "utils3d"
        for relative in (Path("__init__.py"), Path("torch/__init__.py")):
            installed = installed_root / relative
            source = source_root / relative
            if not installed.is_file() or not source.is_file():
                raise FileNotFoundError(f"Locked MoGe-2 utils3d file is missing: {relative}")
            if _sha256_file(installed) != _sha256_file(source):
                raise ValueError(f"Installed utils3d differs from locked checkout: {relative}")
        if not hasattr(utils3d, "pt"):
            raise ValueError("Locked MoGe-2 utils3d must expose the utils3d.pt compatibility alias")
        self.moge = MoGeProcessor.__new__(MoGeProcessor)
        self.moge.device = torch.device("cuda")
        self.moge.model = MoGeModel.from_pretrained(self.moge_checkpoint).to(self.moge.device).eval()
        self.moge.img_size = 448
        self.predictor = HiSpatialPredictor(model_load_path=self.model_path, gpu_rank=0)
        self._loaded = True

    def generate(self, model_input: RunnerModelInput) -> GenerationResult:
        self._load()
        import numpy as np
        xyz = self.moge.apply_transform(model_input.image)
        text = self.predictor.query(
            image=np.asarray(model_input.image), prompt=model_input.question, xyz_values=xyz
        )
        rendered = model_input.question if "<image>" in model_input.question else "<image>" + model_input.question
        return GenerationResult(text, {
            "num_model_image_tensors": 1, "derived_xyz_tensors": 1,
            "derived_xyz_model": MOGE2_MODEL_ID, "derived_xyz_revision": MOGE2_REVISION,
            "derived_xyz_upstream_commit": MOGE2_UPSTREAM_COMMIT,
            "derived_xyz_checkpoint_filename": MOGE2_CHECKPOINT_FILENAME,
            "derived_xyz_utils3d_commit": MOGE2_UTILS3D_COMMIT,
            "template_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        })


class SpatialLadderAdapter(InferenceAdapter):
    def __init__(self, profile: SPBenchSIProfile, decoding: dict[str, Any]) -> None:
        self.profile = profile
        self.decoding = dict(decoding)
        self.model_path = _required_env(profile.model_path_env)
        self.upstream_root = _required_env("SPATIALLADDER_UPSTREAM_ROOT")
        if not verify_hf_snapshot_revision(self.model_path, profile.revision, profile.model):
            raise ValueError("SpatialLadder checkpoint revision is not verifiable")
        if not verify_git_checkout(self.upstream_root, profile.upstream_commit or "", "SpatialLadder"):
            raise ValueError("SpatialLadder upstream checkout is not verifiable")
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        config = _prepare_spatialladder_config(config)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, use_fast=True, min_pixels=16 * 28 * 28,
            max_pixels=512 * 28 * 28, local_files_only=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, config=config, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="auto", local_files_only=True,
        ).eval()
        self._loaded = True

    def generate_batch(self, model_inputs: list[RunnerModelInput]) -> list[GenerationResult]:
        self._load()
        from qwen_vl_utils import process_vision_info
        messages_list = [[{"role": "user", "content": [
            {"type": "image", "image": value.image},
            {"type": "text", "text": value.question},
        ]}] for value in model_inputs]
        rendered = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]
        image_inputs: list[Any] = []
        video_inputs: list[Any] = []
        for messages in messages_list:
            images, videos = process_vision_info(messages)
            image_inputs.extend(images or [])
            video_inputs.extend(videos or [])
        inputs = self.processor(
            text=rendered, images=image_inputs, videos=video_inputs or None, padding=True,
            return_tensors="pt", min_pixels=16 * 28 * 28, max_pixels=512 * 28 * 28,
        ).to(self.model.device)
        generated = self.model.generate(**inputs, **_model_generation(self.decoding))
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [GenerationResult(text, {
            "num_model_image_tensors": 1, "native_batch_size": len(model_inputs),
            "attention_implementation": "flash_attention_2",
            "template_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }) for text, prompt in zip(texts, rendered)]

    def generate(self, model_input: RunnerModelInput) -> GenerationResult:
        return self.generate_batch([model_input])[0]


def _build_backend(profile: SPBenchSIProfile, decoding: dict[str, Any]) -> InferenceAdapter:
    if profile.family == "ssr":
        from ...models.ssr.infer import SSRAdapter
        native = profile.key == "ssr_native"
        return SSRAdapter(
            profile_key="ssr_native" if native else "ssr", upstream_root=_required_env("SSR_UPSTREAM_ROOT"),
            base_model=_required_env("BASE_MODEL"), ssr_vlm=_required_env("SSR_VLM_MODEL"),
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
            model_path=_required_env(profile.model_path_env), max_new_tokens=int(decoding["max_new_tokens"]),
        )
    if profile.family == "3dthinker":
        from ...models.three_d_thinker.infer import ThreeDThinkerAdapter
        return ThreeDThinkerAdapter(
            profile_key="3dthinker", upstream_root=_required_env("THREEDTHINKER_UPSTREAM_ROOT"),
            model_path=_required_env(profile.model_path_env), generation_kwargs=_model_generation(decoding),
            control_prompt_already_present=False,
        )
    if profile.family == "spatialbot":
        from ...models.spatialbot.infer import (
            SPATIALBOT_MIDAS_COMMIT, SPATIALBOT_SIGLIP_REVISION, SpatialBotAdapter,
            ZOEDEPTH_REVISION,
        )
        native = profile.key == "spatialbot_zoedepth"
        return SpatialBotAdapter(
            profile_key="spatialbot_native" if native else "spatialbot",
            upstream_root=_required_env("SPATIALBOT_UPSTREAM_ROOT"),
            model_path=_required_env(profile.model_path_env),
            siglip_model=_required_env("SPATIALBOT_SIGLIP_MODEL"),
            siglip_revision=SPATIALBOT_SIGLIP_REVISION,
            midas_root=_required_env("SPATIALBOT_MIDAS_ROOT") if native else None,
            midas_commit=SPATIALBOT_MIDAS_COMMIT,
            zoedepth_root=_required_env("ZOEDEPTH_ROOT") if native else None,
            zoedepth_revision=ZOEDEPTH_REVISION if native else None,
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
    request: dict[str, Any], profile: SPBenchSIProfile, decoding: dict[str, Any],
    result: GenerationResult, model_input: RunnerModelInput,
) -> dict[str, Any]:
    generation = dict(result.metadata)
    generation.update({
        "num_media_prompt": 1,
        "source_rgb_count": 1,
        "source_rgb_sha256": model_input.source_rgb_sha256,
        "system_role_supported": False,
        "system_prompt_sha256": hashlib.sha256(model_input.system_prompt.encode()).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(model_input.user_prompt.encode()).hexdigest(),
        "folded_prompt_sha256": hashlib.sha256(model_input.question.encode()).hexdigest(),
    })
    if profile.comparison_group != "rgb_only":
        generation["derived_from_source_rgb_sha256"] = model_input.source_rgb_sha256
    if profile.key == "ssr_native":
        generation["derived_depth_count"] = 1
    if profile.key == "spatialbot_zoedepth":
        for key, expected in {"num_model_image_tensors": 2, "derived_depth_count": 1}.items():
            if generation.get(key) != expected:
                raise ValueError(f"SpatialBot ZoeDepth evidence mismatch for {key}")
    elif generation.get("num_model_image_tensors") != 1:
        raise ValueError("Specialized backend must prove exactly one model-bound image tensor")
    if not isinstance(generation.get("template_sha256"), str) or len(generation["template_sha256"]) != 64:
        raise ValueError("Specialized backend did not return a template SHA-256")
    return {
        "index": request["index"], "profile": profile.key, "model_revision": profile.revision,
        "inference_protocol": profile.inference_protocol, "decoding": decoding,
        "system_role_supported": False, "raw_prediction": result.text,
        "generation": generation, "warnings": list(result.warnings),
    }


def serve(profile: SPBenchSIProfile, decoding: dict[str, Any]) -> None:
    backend: InferenceAdapter | None = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request == {"schema_version": 1, "action": "close"}:
                return
            batch_wrapper = isinstance(request, dict) and request.get("action") == "generate_batch"
            requests = request.get("requests") if batch_wrapper else [request]
            if not isinstance(requests, list) or not requests:
                raise ValueError("Runner batch request must contain a non-empty requests list")
            model_inputs = [_verify_request(value, profile, decoding) for value in requests]
            if backend is None:
                with redirect_stdout(sys.stderr):
                    backend = _build_backend(profile, decoding)
            _seed(decoding)
            with redirect_stdout(sys.stderr):
                results = backend.generate_batch(model_inputs)
            if len(results) != len(requests):
                raise ValueError("Specialized backend returned the wrong batch size")
            responses = [
                _response(value, profile, decoding, result, model_input)
                for value, result, model_input in zip(requests, results, model_inputs)
            ]
            payload: Any = {"responses": responses} if batch_wrapper else responses[0]
            print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
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
    serve(profile, load_generation_manifest(profile, args.generation_manifest))


if __name__ == "__main__":
    main()
