"""Official Bunny/SpatialBot inference with optional derived ZoeDepth input."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

from ...benchmarks.msmu.data import MSMUModelInput
from ..common.cli import add_msmu_run_arguments, execute_msmu_cli
from ..common.provenance import verify_git_checkout, verify_hf_snapshot_revision
from ..common.runtime import GenerationResult, InferenceAdapter
from ..profiles import get_profile

SPATIALBOT_REVISION = "41d3b52c642058dfb087885bec0b8e37e0e67f8d"
SPATIALBOT_SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
SPATIALBOT_SIGLIP_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
ZOEDEPTH_REVISION = "d87f17b2f5fdcb174cf4fb115491f4a6c60de152"
ZOEDEPTH_DERIVED_BUFFER_COUNT = 24
MIDAS_RELATIVE_POSITION_MODULE_COUNT = 24


def meters_to_uint16_millimeters(depth_meters: Any) -> Any:
    import numpy as np

    depth = np.asarray(depth_meters, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"ZoeDepth output must be 2-D, got shape={depth.shape}")
    if not np.isfinite(depth).all():
        raise ValueError("ZoeDepth output contains non-finite values")
    return np.clip(np.rint(depth * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def encode_spatialbot_depth(depth_millimeters: Any) -> Any:
    """Encode uint16 millimetres into the official three-channel SpatialBot layout."""

    import numpy as np

    depth = np.asarray(depth_millimeters)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("SpatialBot depth encoder requires a 2-D uint16 millimetre map")
    encoded = np.zeros((*depth.shape, 3), dtype=np.uint8)
    encoded[:, :, 0] = (depth // 1024) * 4
    encoded[:, :, 1] = (depth // 32) * 8
    encoded[:, :, 2] = (depth % 32) * 8
    return encoded


def load_zoedepth_checkpoint_compat(model: Any, checkpoint: Path, torch_module: Any) -> list[str]:
    """Strictly load ZoeDepth while ignoring TIMM-derived position-index buffers."""

    payload = torch_module.load(checkpoint, map_location="cpu")
    state_dict = payload.get("model", payload)
    normalized = {
        (key[7:] if key.startswith("module.") else key): value for key, value in state_dict.items()
    }
    derived = sorted(key for key in normalized if key.endswith(".attn.relative_position_index"))
    if len(derived) != ZOEDEPTH_DERIVED_BUFFER_COUNT:
        raise RuntimeError(
            "Unexpected ZoeDepth derived-buffer count: "
            f"expected {ZOEDEPTH_DERIVED_BUFFER_COUNT}, got {len(derived)}"
        )
    expected_keys = set(model.state_dict())
    ignored = [key for key in derived if key not in expected_keys]
    if len(ignored) not in {0, ZOEDEPTH_DERIVED_BUFFER_COUNT}:
        raise RuntimeError(
            "ZoeDepth target model only partially matches derived buffers: "
            f"{ZOEDEPTH_DERIVED_BUFFER_COUNT - len(ignored)} present, {len(ignored)} absent"
        )
    for key in ignored:
        del normalized[key]
    model.load_state_dict(normalized, strict=True)
    return ignored


def patch_zoedepth_resize_python_int(resize_class: type[Any]) -> None:
    """Make ZoeDepth/TIMMs NumPy scalar sizes acceptable to Torch 2.1."""

    if getattr(resize_class, "_msmu_python_int_compat", False):
        return
    original = resize_class.constrain_to_multiple_of

    def constrain_to_multiple_of(instance: Any, *args: Any, **kwargs: Any) -> int:
        return int(original(instance, *args, **kwargs))

    resize_class.constrain_to_multiple_of = constrain_to_multiple_of
    resize_class._msmu_python_int_compat = True


def install_legacy_timm_layers_alias(
    module_registry: dict[str, Any],
    legacy_layers: Any,
    legacy_norm_act: Any,
) -> None:
    """Expose TIMM 0.6 layer modules under the paths used by newer Bunny code."""

    module_registry.setdefault("timm.layers", legacy_layers)
    module_registry.setdefault("timm.layers.norm_act", legacy_norm_act)


def patch_midas_relative_position_sizes(model: Any) -> int:
    """Cast MiDaS BEiT window sizes to Python ints for Torch 2.1 interpolation."""

    patched = 0
    for module in model.modules():
        original = getattr(module, "_get_rel_pos_bias", None)
        if original is None or getattr(module, "_msmu_python_int_compat", False):
            continue

        def get_rel_pos_bias(window_size: Any, _original: Any = original) -> Any:
            return _original(tuple(int(value) for value in window_size))

        module._get_rel_pos_bias = get_rel_pos_bias
        module._msmu_python_int_compat = True
        patched += 1
    if patched != MIDAS_RELATIVE_POSITION_MODULE_COUNT:
        raise RuntimeError(
            "Unexpected MiDaS relative-position module count: "
            f"expected {MIDAS_RELATIVE_POSITION_MODULE_COUNT}, got {patched}"
        )
    return patched


def spatialbot_prompt(profile_key: str, question: str) -> str:
    if profile_key == "spatialbot":
        return f"<image>\n{question}"
    if profile_key == "spatialbot_native":
        return f"<image 1>\n<image 2>\n{question}"
    raise ValueError(f"Unknown SpatialBot profile: {profile_key}")


def bind_spatialbot_vision_tower(config: Any, local_model_path: str) -> Any:
    """Bind the checkpoint's locked SigLIP tower to a verified local snapshot."""

    configured = getattr(config, "mm_vision_tower", None)
    if configured != SPATIALBOT_SIGLIP_MODEL_ID:
        raise RuntimeError(
            "Unexpected SpatialBot vision tower: "
            f"expected {SPATIALBOT_SIGLIP_MODEL_ID!r}, got {configured!r}"
        )
    config.mm_vision_tower = str(Path(local_model_path).resolve())
    return config


class SpatialBotAdapter(InferenceAdapter):
    batch_size = 1
    supports_concurrency = False

    def __init__(
        self,
        *,
        profile_key: str,
        upstream_root: str,
        model_path: str,
        model_revision: str = SPATIALBOT_REVISION,
        model_base: str | None = None,
        model_type: str = "phi-2",
        conversation_mode: str = "bunny",
        siglip_model: str | None = None,
        siglip_revision: str = SPATIALBOT_SIGLIP_REVISION,
        zoedepth_root: str | None = None,
        zoedepth_revision: str | None = None,
        zoedepth_checkpoint: str | None = None,
        device: str = "cuda",
        max_new_tokens: int | None = None,
    ) -> None:
        if profile_key not in {"spatialbot", "spatialbot_native"}:
            raise ValueError("SpatialBot profile must be spatialbot or spatialbot_native")
        if model_revision != SPATIALBOT_REVISION:
            raise ValueError(f"SpatialBot-3B is locked to {SPATIALBOT_REVISION}")
        if model_type != "phi-2" or conversation_mode != "bunny":
            raise ValueError("Merged SpatialBot-3B is locked to model_type=phi-2 and conv_mode=bunny")
        self.profile = get_profile(profile_key)
        self.upstream_root = Path(upstream_root).resolve()
        if not (self.upstream_root / "bunny" / "serve" / "cli.py").exists():
            raise FileNotFoundError(f"Not a SpatialBot upstream checkout: {self.upstream_root}")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                "SpatialBot-3B is gated. Accept its Hugging Face license and download the locked "
                f"revision locally before inference; expected local path, got {model_path!r}."
            )
        self.model_revision = str(model_revision)
        self.model_snapshot_revision_verified = verify_hf_snapshot_revision(
            self.model_path,
            self.model_revision,
            "SpatialBot-3B",
        )
        self.upstream_commit_verified = verify_git_checkout(
            self.upstream_root,
            self.profile.upstream_commit or "",
            "SpatialBot",
        )
        self.model_base = str(Path(model_base).resolve()) if model_base else None
        self.model_type = model_type
        self.conversation_mode = conversation_mode
        self.siglip_model = Path(siglip_model).resolve() if siglip_model else None
        self.siglip_revision = str(siglip_revision) if siglip_model else None
        if self.siglip_model is not None:
            if self.siglip_revision != SPATIALBOT_SIGLIP_REVISION:
                raise ValueError(f"SpatialBot SigLIP is locked to {SPATIALBOT_SIGLIP_REVISION}")
            self.siglip_snapshot_revision_verified = verify_hf_snapshot_revision(
                self.siglip_model,
                SPATIALBOT_SIGLIP_REVISION,
                "SpatialBot SigLIP vision tower",
            )
        else:
            self.siglip_snapshot_revision_verified = False
        self.zoedepth_root = Path(zoedepth_root).resolve() if zoedepth_root else None
        self.zoedepth_revision = str(zoedepth_revision) if zoedepth_revision else None
        self.zoedepth_checkpoint = str(Path(zoedepth_checkpoint).resolve()) if zoedepth_checkpoint else None
        self.device_name = str(device)
        self.max_new_tokens = int(max_new_tokens or self.profile.max_new_tokens)
        if self.max_new_tokens <= 0:
            raise ValueError("SpatialBot max_new_tokens must be positive")
        if self.profile.key == "spatialbot_native":
            missing = [
                name
                for name, value in {
                    "zoedepth_root": self.zoedepth_root,
                    "zoedepth_revision": self.zoedepth_revision,
                    "zoedepth_checkpoint": self.zoedepth_checkpoint,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(f"SpatialBot native profile is missing ZoeDepth provenance: {missing}")
            if self.zoedepth_revision != ZOEDEPTH_REVISION:
                raise ValueError(f"ZoeDepth is locked to {ZOEDEPTH_REVISION}")
            assert self.zoedepth_root is not None
            self.zoedepth_commit_verified = verify_git_checkout(
                self.zoedepth_root,
                ZOEDEPTH_REVISION,
                "ZoeDepth",
            )
        else:
            self.zoedepth_commit_verified = False
        self._loaded = False

    def metadata(self) -> dict[str, Any]:
        native = self.profile.key == "spatialbot_native"
        return {
            "model": self.profile.model,
            "model_revision": self.model_revision,
            "model_path": str(self.model_path),
            "model_base": self.model_base,
            "backend": "official-spatialbot-bunny",
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "inference_protocol": self.profile.inference_protocol,
            "chat_template": self.profile.chat_template,
            "image_processing": {
                "source": "MSMU RGB only",
                "input_rgb_count": 1,
                "model_image_tensor_count": 2 if native else 1,
                "depth_source": "ZoeDepth estimate from the same MSMU RGB" if native else None,
                "depth_units": "uint16 millimetres" if native else None,
                "depth_quantization": "clip(round(metres * 1000), 0, 65535)" if native else None,
                "depth_encoding": "official SpatialBot 3-channel uint16 packing" if native else None,
                "zoedepth_revision": self.zoedepth_revision if native else None,
                "zoedepth_checkpoint": self.zoedepth_checkpoint if native else None,
                "zoedepth_derived_buffer_compat_counts": (
                    [0, ZOEDEPTH_DERIVED_BUFFER_COUNT] if native else None
                ),
                "zoedepth_ignored_derived_buffer_suffix": (
                    ".attn.relative_position_index" if native else None
                ),
                "zoedepth_resize_size_cast": "numpy scalar to Python int" if native else None,
                "midas_relative_position_size_cast_modules": (
                    MIDAS_RELATIVE_POSITION_MODULE_COUNT if native else None
                ),
                "timm_layers_compat": "alias timm.models.layers when timm.layers is absent",
                "vision_tower_model": SPATIALBOT_SIGLIP_MODEL_ID,
                "vision_tower_revision": self.siglip_revision,
                "vision_tower_path": str(self.siglip_model) if self.siglip_model else None,
            },
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.max_new_tokens,
                "use_cache": True,
            },
            "upstream": {
                "repository": self.profile.upstream_url,
                "commit": self.profile.upstream_commit,
                "commit_verified": self.upstream_commit_verified,
                "checkout": str(self.upstream_root),
                "model_snapshot_revision_verified": self.model_snapshot_revision_verified,
                "vision_tower_snapshot_revision_verified": (
                    self.siglip_snapshot_revision_verified
                ),
                "entrypoint_equivalent": ("bunny/serve/cli_depth.py" if native else "bunny/serve/cli.py"),
                "zoedepth_checkout": str(self.zoedepth_root) if self.zoedepth_root else None,
                "zoedepth_commit_verified": self.zoedepth_commit_verified,
                "known_depth_export_deviation": (
                    "Project protocol quantizes metres to millimetres; pinned ZoeDepth's "
                    "save_raw_16bit helper instead multiplies by 256."
                    if native
                    else None
                ),
            },
        }

    def _load(self) -> None:
        if self._loaded:
            return
        if str(self.upstream_root) not in sys.path:
            sys.path.insert(0, str(self.upstream_root))
        import torch
        try:
            import timm.layers  # noqa: F401
        except ModuleNotFoundError:
            from timm.models import layers as legacy_layers
            from timm.models.layers import norm_act as legacy_norm_act

            install_legacy_timm_layers_alias(sys.modules, legacy_layers, legacy_norm_act)
        from bunny.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from bunny.conversation import SeparatorStyle, conv_templates
        from bunny.model.builder import load_pretrained_model
        from bunny.util.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
            tokenizer_multi_image_token,
        )
        from bunny.util.utils import disable_torch_init
        from transformers import AutoConfig

        disable_torch_init()
        model_name = get_model_name_from_path(str(self.model_path))
        load_kwargs: dict[str, Any] = {}
        if self.siglip_model is not None:
            config = AutoConfig.from_pretrained(str(self.model_path), local_files_only=True)
            load_kwargs["config"] = bind_spatialbot_vision_tower(config, str(self.siglip_model))
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            str(self.model_path),
            self.model_base,
            model_name,
            self.model_type,
            device=self.device_name,
            **load_kwargs,
        )
        self.torch = torch
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.tokenizer_multi_image_token = tokenizer_multi_image_token
        if self.profile.key == "spatialbot_native":
            self._load_zoedepth()
        self._loaded = True

    def _load_zoedepth(self) -> None:
        assert self.zoedepth_root and self.zoedepth_checkpoint
        if str(self.zoedepth_root) not in sys.path:
            sys.path.insert(0, str(self.zoedepth_root))
        from zoedepth.models.builder import build_model
        from zoedepth.models.base_models.midas import Resize
        from zoedepth.utils.config import get_config

        patch_zoedepth_resize_python_int(Resize)
        config = get_config("zoedepth_nk", "infer")
        if hasattr(config, "pretrained_resource"):
            config.pretrained_resource = None
        self.zoedepth = build_model(config).to(self.device_name).eval()
        load_zoedepth_checkpoint_compat(
            self.zoedepth,
            Path(self.zoedepth_checkpoint),
            self.torch,
        )
        patch_midas_relative_position_sizes(self.zoedepth)
        for parameter in self.zoedepth.parameters():
            parameter.requires_grad_(False)

    def _model_images(self, image: Any) -> tuple[Any, int]:
        from PIL import Image

        rgb = image.convert("RGB")
        if self.profile.key == "spatialbot":
            tensor = self.process_images([rgb], self.image_processor, self.model.config)
            if isinstance(tensor, list):
                tensor = [value.to(self.model.device, dtype=self.model.dtype) for value in tensor]
            else:
                tensor = tensor.to(self.model.device, dtype=self.model.dtype)
            return tensor, 1
        with self.torch.inference_mode():
            depth_meters = self.zoedepth.infer_pil(rgb)
        depth_mm = meters_to_uint16_millimeters(depth_meters)
        depth_rgb = Image.fromarray(encode_spatialbot_depth(depth_mm), mode="RGB")
        values = [rgb, depth_rgb]
        tensors = [
            self.image_processor.preprocess(value, return_tensors="pt")["pixel_values"][0].to(
                self.model.device,
                dtype=self.model.dtype,
            )
            for value in values
        ]
        return tensors, 2

    def generate(self, model_input: MSMUModelInput) -> GenerationResult:
        self._load()
        images, tensor_count = self._model_images(model_input.image)
        conv = self.conv_templates[self.conversation_mode].copy()
        prompt_question = spatialbot_prompt(self.profile.key, model_input.question)
        conv.append_message(conv.roles[0], prompt_question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if self.profile.key == "spatialbot_native":
            input_ids = (
                self.tokenizer_multi_image_token(
                    prompt,
                    self.tokenizer,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(self.model.device)
            )
        else:
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
        stop = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping = self.KeywordsStoppingCriteria([stop], self.tokenizer, input_ids)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping],
            )
        prediction = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        if stop and prediction.endswith(stop):
            prediction = prediction[: -len(stop)].strip()
        warnings = ("model returned an empty text completion",) if not prediction else ()
        return GenerationResult(
            text=prediction,
            metadata={
                "num_model_image_tensors": tensor_count,
                "input_rgb_count": 1,
                "derived_depth_count": 1 if self.profile.key == "spatialbot_native" else 0,
                "depth_derived_from_same_rgb": self.profile.key == "spatialbot_native",
                "template_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            },
            warnings=warnings,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=["spatialbot", "spatialbot_native"])
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=SPATIALBOT_REVISION)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--model-type", default="phi-2")
    parser.add_argument("--conversation-mode", default="bunny")
    parser.add_argument("--siglip-model", default=None)
    parser.add_argument("--siglip-revision", default=SPATIALBOT_SIGLIP_REVISION)
    parser.add_argument("--zoedepth-root", default=None)
    parser.add_argument("--zoedepth-revision", default=ZOEDEPTH_REVISION)
    parser.add_argument("--zoedepth-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    add_msmu_run_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = SpatialBotAdapter(
        profile_key=args.profile,
        upstream_root=args.upstream_root,
        model_path=args.model,
        model_revision=args.model_revision,
        model_base=args.model_base,
        model_type=args.model_type,
        conversation_mode=args.conversation_mode,
        siglip_model=args.siglip_model,
        siglip_revision=args.siglip_revision,
        zoedepth_root=args.zoedepth_root,
        zoedepth_revision=args.zoedepth_revision,
        zoedepth_checkpoint=args.zoedepth_checkpoint,
        device=args.device,
    )
    execute_msmu_cli(args, adapter)


if __name__ == "__main__":
    main()
