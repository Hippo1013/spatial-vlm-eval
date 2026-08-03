"""Official-component SSR inference tracks for MSMU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ...benchmarks.msmu.data import MSMUModelInput
from ..common.cli import add_msmu_run_arguments, execute_msmu_cli
from ..common.provenance import verify_git_checkout, verify_hf_snapshot_revision
from ..common.runtime import GenerationResult, InferenceAdapter
from ..profiles import get_profile

SSR_VLM_REVISION = "7bcb4636f1396325f27f7fbb2f2df121128931bf"
SSR_MIDI_REVISION = "8ed878fa16e3e440741ed8c1fedfcfe40710258d"
SSR_DEPTHPRO_REVISION = "edb23bbab37cfc4d3fe1048a2f126ca7c590ab64"
SSR_DEPTHPRO_CHECKPOINT_SHA256 = "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"
SSR_BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SSR_BASE_MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SSR_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
SSR_CLIP_MODEL_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
SSR_SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
SSR_SIGLIP_MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
SSR_MAMBA_MODEL_ID = "state-spaces/mamba-130m-hf"
SSR_MAMBA_MODEL_REVISION = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
SSR_MIDI_LLM_MODEL_ID = "Qwen/Qwen2.5-7B"
SSR_MIDI_LLM_MODEL_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
TOR_COUNT = 10


@contextmanager
def ssr_autoroot_entrypoint(upstream_root: Path) -> Iterator[None]:
    """Anchor SSR's ``autoroot`` import to the locked upstream checkout.

    Upstream ``ssr.models.vlm`` imports ``autoroot==1.0.1``, which searches for
    ``.project-root`` starting at ``sys.argv[0]``.  Our module entrypoint lives in
    this repository, so expose SSR's own ``infer.py`` only for that import and
    restore the real CLI entrypoint immediately afterwards.
    """

    entrypoint = Path(upstream_root).resolve() / "infer.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"SSR upstream entrypoint is missing: {entrypoint}")
    original = sys.argv[0]
    sys.argv[0] = str(entrypoint)
    try:
        yield
    finally:
        sys.argv[0] = original


def tor_prefix(count: int = TOR_COUNT) -> str:
    return "<tor>" * int(count)


def ssr_question(profile_key: str, question: str) -> str:
    if profile_key == "ssr":
        return str(question)
    if profile_key == "ssr_native":
        return f"{tor_prefix()}\n{question}"
    raise ValueError(f"Unknown SSR profile: {profile_key}")


def ssr_component_switches(profile_key: str) -> dict[str, Any]:
    if profile_key == "ssr":
        return {"depthpro": False, "midi": False, "tor_count": 0, "model_image_tensor_count": 1}
    if profile_key == "ssr_native":
        return {"depthpro": True, "midi": True, "tor_count": TOR_COUNT, "model_image_tensor_count": 1}
    raise ValueError(f"Unknown SSR profile: {profile_key}")


def ssr_image_views(image: Any) -> tuple[Any, Any]:
    """Keep upstream's original-resolution auxiliary view and 256px VLM view distinct."""

    original_rgb = image.convert("RGB")
    return original_rgb, original_rgb.resize((256, 256))


def _local_pretrained_kwargs(path_or_id: str, revision: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": True}
    if not Path(path_or_id).exists() and revision and revision != "local-unspecified":
        kwargs["revision"] = revision
    return kwargs


def _local_adapter_kwargs(path_or_id: str, revision: str | None) -> dict[str, Any]:
    """Map offline-loading arguments onto Transformers 4.49's adapter API."""

    pretrained_kwargs = _local_pretrained_kwargs(path_or_id, revision)
    adapter_kwargs = {"local_files_only": pretrained_kwargs.pop("local_files_only")}
    return {**pretrained_kwargs, "adapter_kwargs": adapter_kwargs}


def _verify_local_hidden_size(
    path_or_id: str,
    *,
    expected: int,
    label: str,
    nested_key: str | None = None,
) -> bool:
    """Fail early when a local auxiliary checkpoint has incompatible dimensions."""

    root = Path(path_or_id).expanduser()
    if not root.exists():
        return False
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{label} local checkpoint has no config.json: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if nested_key:
        nested = config.get(nested_key)
        if nested is not None:
            if not isinstance(nested, dict):
                raise ValueError(f"{label} config field {nested_key!r} is not an object")
            config = nested
    hidden_size = config.get("hidden_size")
    if int(hidden_size or -1) != int(expected):
        raise ValueError(f"{label} hidden_size is {hidden_size}, expected {expected}")
    return True


def _verify_file_sha256(path: str, *, expected: str, label: str) -> str:
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint is not a file: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"{label} checkpoint SHA-256 is {actual}, expected {expected}")
    return actual


class SSRAdapter(InferenceAdapter):
    batch_size = 1
    supports_concurrency = False

    def __init__(
        self,
        *,
        profile_key: str,
        upstream_root: str,
        base_model: str,
        base_model_revision: str = SSR_BASE_MODEL_REVISION,
        ssr_vlm: str,
        ssr_vlm_revision: str = SSR_VLM_REVISION,
        ssr_midi: str | None = None,
        ssr_midi_revision: str = SSR_MIDI_REVISION,
        clip_model: str | None = None,
        clip_model_revision: str = SSR_CLIP_MODEL_REVISION,
        siglip_model: str | None = None,
        siglip_model_revision: str = SSR_SIGLIP_MODEL_REVISION,
        mamba_model: str | None = None,
        mamba_model_revision: str = SSR_MAMBA_MODEL_REVISION,
        midi_llm_model: str | None = None,
        midi_llm_model_revision: str = SSR_MIDI_LLM_MODEL_REVISION,
        depthpro_root: str | None = None,
        depthpro_checkpoint: str | None = None,
        depthpro_checkpoint_sha256: str = SSR_DEPTHPRO_CHECKPOINT_SHA256,
        depthpro_revision: str = SSR_DEPTHPRO_REVISION,
        device: str = "cuda",
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if profile_key not in {"ssr", "ssr_native"}:
            raise ValueError("SSR profile must be ssr or ssr_native")
        if ssr_vlm_revision != SSR_VLM_REVISION:
            raise ValueError(f"SSR-VLM is locked to {SSR_VLM_REVISION}")
        if ssr_midi_revision != SSR_MIDI_REVISION:
            raise ValueError(f"SSR-MIDI is locked to {SSR_MIDI_REVISION}")
        if base_model_revision != SSR_BASE_MODEL_REVISION:
            raise ValueError(f"SSR Qwen2.5-VL base is locked to {SSR_BASE_MODEL_REVISION}")
        locked_auxiliary_revisions = {
            "CLIP": (clip_model_revision, SSR_CLIP_MODEL_REVISION),
            "SigLIP": (siglip_model_revision, SSR_SIGLIP_MODEL_REVISION),
            "Mamba": (mamba_model_revision, SSR_MAMBA_MODEL_REVISION),
            "MIDI internal Qwen2.5-7B": (
                midi_llm_model_revision,
                SSR_MIDI_LLM_MODEL_REVISION,
            ),
        }
        for label, (actual, expected) in locked_auxiliary_revisions.items():
            if actual != expected:
                raise ValueError(f"SSR {label} is locked to {expected}")
        self.profile = get_profile(profile_key)
        self.upstream_root = Path(upstream_root).resolve()
        if not (self.upstream_root / "ssr" / "models" / "vlm.py").exists():
            raise FileNotFoundError(f"Not an SSR upstream checkout: {self.upstream_root}")
        self.base_model = str(base_model)
        self.base_model_revision = str(base_model_revision)
        self.ssr_vlm = str(ssr_vlm)
        self.ssr_vlm_revision = str(ssr_vlm_revision)
        self.ssr_midi = str(ssr_midi) if ssr_midi else None
        self.ssr_midi_revision = str(ssr_midi_revision)
        self.clip_model = str(clip_model) if clip_model else None
        self.clip_model_revision = str(clip_model_revision)
        self.siglip_model = str(siglip_model) if siglip_model else None
        self.siglip_model_revision = str(siglip_model_revision)
        self.mamba_model = str(mamba_model) if mamba_model else None
        self.mamba_model_revision = str(mamba_model_revision)
        self.midi_llm_model = str(midi_llm_model) if midi_llm_model else None
        self.midi_llm_model_revision = str(midi_llm_model_revision)
        self.depthpro_root = Path(depthpro_root).resolve() if depthpro_root else None
        self.depthpro_checkpoint = str(depthpro_checkpoint) if depthpro_checkpoint else None
        self.depthpro_checkpoint_sha256 = str(depthpro_checkpoint_sha256)
        self.depthpro_revision = str(depthpro_revision)
        self.device_name = str(device)
        self.generation_kwargs = dict(
            generation_kwargs
            or {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.profile.max_new_tokens,
                "use_cache": True,
            }
        )
        if int(self.generation_kwargs.get("max_new_tokens", 0)) <= 0:
            raise ValueError("SSR generation kwargs require a positive max_new_tokens")
        self.ssr_vlm_snapshot_revision_verified = verify_hf_snapshot_revision(
            self.ssr_vlm,
            self.ssr_vlm_revision,
            "SSR-VLM-7B",
        )
        self.base_model_snapshot_revision_verified = verify_hf_snapshot_revision(
            self.base_model,
            self.base_model_revision,
            "Qwen2.5-VL-7B-Instruct",
        )
        _verify_local_hidden_size(
            self.base_model,
            expected=3584,
            label=SSR_BASE_MODEL_ID,
            nested_key="text_config",
        )
        self.ssr_midi_snapshot_revision_verified = (
            verify_hf_snapshot_revision(
                self.ssr_midi,
                self.ssr_midi_revision,
                "SSR-MIDI-7B",
            )
            if self.ssr_midi
            else False
        )
        self.upstream_commit_verified = verify_git_checkout(
            self.upstream_root,
            self.profile.upstream_commit or "",
            "SSR",
        )
        if self.profile.key == "ssr_native":
            missing = [
                name
                for name, value in {
                    "ssr_midi": self.ssr_midi,
                    "clip_model": self.clip_model,
                    "siglip_model": self.siglip_model,
                    "mamba_model": self.mamba_model,
                    "midi_llm_model": self.midi_llm_model,
                    "depthpro_root": self.depthpro_root,
                    "depthpro_checkpoint": self.depthpro_checkpoint,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(f"SSR native profile is missing component paths: {missing}")
            if self.depthpro_revision != SSR_DEPTHPRO_REVISION:
                raise ValueError(f"SSR DepthPro fork is locked to {SSR_DEPTHPRO_REVISION}")
            assert self.depthpro_root
            if not (self.depthpro_root / "src" / "depth_pro" / "depth_pro.py").is_file():
                raise FileNotFoundError(
                    f"Not the locked SSR DepthPro checkout: {self.depthpro_root}"
                )
            self.depthpro_commit_verified = verify_git_checkout(
                self.depthpro_root,
                self.depthpro_revision,
                "SSR DepthPro",
            )
            if self.depthpro_checkpoint_sha256 != SSR_DEPTHPRO_CHECKPOINT_SHA256:
                raise ValueError(
                    f"SSR DepthPro checkpoint is locked to {SSR_DEPTHPRO_CHECKPOINT_SHA256}"
                )
            assert self.clip_model and self.siglip_model and self.mamba_model and self.midi_llm_model
            assert self.depthpro_checkpoint
            _verify_file_sha256(
                self.depthpro_checkpoint,
                expected=self.depthpro_checkpoint_sha256,
                label="DepthPro",
            )
            self.clip_model_snapshot_revision_verified = verify_hf_snapshot_revision(
                self.clip_model,
                self.clip_model_revision,
                SSR_CLIP_MODEL_ID,
            )
            self.siglip_model_snapshot_revision_verified = verify_hf_snapshot_revision(
                self.siglip_model,
                self.siglip_model_revision,
                SSR_SIGLIP_MODEL_ID,
            )
            self.mamba_model_snapshot_revision_verified = verify_hf_snapshot_revision(
                self.mamba_model,
                self.mamba_model_revision,
                SSR_MAMBA_MODEL_ID,
            )
            self.midi_llm_model_snapshot_revision_verified = verify_hf_snapshot_revision(
                self.midi_llm_model,
                self.midi_llm_model_revision,
                SSR_MIDI_LLM_MODEL_ID,
            )
            _verify_local_hidden_size(
                self.clip_model,
                expected=1024,
                label=SSR_CLIP_MODEL_ID,
                nested_key="vision_config",
            )
            _verify_local_hidden_size(
                self.siglip_model,
                expected=1152,
                label=SSR_SIGLIP_MODEL_ID,
                nested_key="vision_config",
            )
            _verify_local_hidden_size(
                self.mamba_model,
                expected=768,
                label=SSR_MAMBA_MODEL_ID,
            )
            _verify_local_hidden_size(
                self.midi_llm_model,
                expected=3584,
                label=SSR_MIDI_LLM_MODEL_ID,
            )
        else:
            self.depthpro_commit_verified = False
            self.clip_model_snapshot_revision_verified = False
            self.siglip_model_snapshot_revision_verified = False
            self.mamba_model_snapshot_revision_verified = False
            self.midi_llm_model_snapshot_revision_verified = False
        self._loaded = False

    def metadata(self) -> dict[str, Any]:
        switches = ssr_component_switches(self.profile.key)
        return {
            "model": self.profile.model,
            "model_revision": self.profile.revision,
            "base_model": self.base_model,
            "base_model_revision": self.base_model_revision,
            "ssr_vlm": self.ssr_vlm,
            "ssr_vlm_revision": self.ssr_vlm_revision,
            "ssr_midi": self.ssr_midi,
            "ssr_midi_revision": self.ssr_midi_revision if self.ssr_midi else None,
            "backend": "official-ssr-transformers",
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "inference_protocol": self.profile.inference_protocol,
            "chat_template": self.profile.chat_template,
            "image_processing": {
                "source": "MSMU RGB only",
                "model_rgb_resize": [256, 256],
                "native_auxiliary_rgb_resolution": "original" if switches["midi"] else None,
                "depth_estimator": "DepthPro from the same RGB" if switches["depthpro"] else None,
                "clip_model": self.clip_model if switches["midi"] else None,
                "siglip_model": self.siglip_model if switches["midi"] else None,
                "model_image_tensor_count": switches["model_image_tensor_count"],
            },
            "decoding": {**self.generation_kwargs, "tor_count": switches["tor_count"]},
            "components": switches,
            "upstream": {
                "repository": self.profile.upstream_url,
                "commit": self.profile.upstream_commit,
                "commit_verified": self.upstream_commit_verified,
                "checkout": str(self.upstream_root),
                "base_model_id": SSR_BASE_MODEL_ID,
                "base_model_snapshot_revision_verified": self.base_model_snapshot_revision_verified,
                "depthpro_checkpoint": self.depthpro_checkpoint,
                "depthpro_checkout": str(self.depthpro_root) if switches["depthpro"] else None,
                "depthpro_checkpoint_sha256": (
                    self.depthpro_checkpoint_sha256 if switches["depthpro"] else None
                ),
                "depthpro_revision": self.depthpro_revision if switches["depthpro"] else None,
                "depthpro_commit_verified": (
                    self.depthpro_commit_verified if switches["depthpro"] else False
                ),
                "clip_model_id": SSR_CLIP_MODEL_ID if switches["midi"] else None,
                "clip_model_revision": self.clip_model_revision if switches["midi"] else None,
                "clip_model_snapshot_revision_verified": (
                    self.clip_model_snapshot_revision_verified if switches["midi"] else False
                ),
                "siglip_model_id": SSR_SIGLIP_MODEL_ID if switches["midi"] else None,
                "siglip_model_revision": self.siglip_model_revision if switches["midi"] else None,
                "siglip_model_snapshot_revision_verified": (
                    self.siglip_model_snapshot_revision_verified if switches["midi"] else False
                ),
                "mamba_model": self.mamba_model,
                "mamba_model_id": SSR_MAMBA_MODEL_ID if switches["midi"] else None,
                "mamba_model_revision": self.mamba_model_revision if switches["midi"] else None,
                "mamba_model_snapshot_revision_verified": (
                    self.mamba_model_snapshot_revision_verified if switches["midi"] else False
                ),
                "midi_llm_model": self.midi_llm_model,
                "midi_llm_model_id": SSR_MIDI_LLM_MODEL_ID if switches["midi"] else None,
                "midi_llm_model_revision": self.midi_llm_model_revision if switches["midi"] else None,
                "midi_llm_model_snapshot_revision_verified": (
                    self.midi_llm_model_snapshot_revision_verified if switches["midi"] else False
                ),
                "ssr_vlm_snapshot_revision_verified": self.ssr_vlm_snapshot_revision_verified,
                "ssr_midi_snapshot_revision_verified": self.ssr_midi_snapshot_revision_verified,
            },
        }

    @staticmethod
    def _freeze(module: Any) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _load(self) -> None:
        if self._loaded:
            return
        if str(self.upstream_root) not in sys.path:
            sys.path.insert(0, str(self.upstream_root))
        import torch
        from transformers import Qwen2_5_VLProcessor

        with ssr_autoroot_entrypoint(self.upstream_root):
            from ssr.models.vlm import SSRVLM

        self.torch = torch
        self.device = torch.device(self.device_name)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            self.base_model,
            **_local_pretrained_kwargs(self.base_model, self.base_model_revision),
        )
        self.vlm = SSRVLM.from_pretrained(
            self.base_model,
            device_map=self.device,
            **_local_pretrained_kwargs(self.base_model, self.base_model_revision),
        )
        self.vlm.load_adapter(
            self.ssr_vlm,
            **_local_adapter_kwargs(self.ssr_vlm, self.ssr_vlm_revision),
        )
        self._freeze(self.vlm)
        if self.profile.key == "ssr_native":
            self._load_native_components()
        self._loaded = True

    def _load_native_components(self) -> None:
        import depth_pro
        from ssr.models.midi import MIDI, MIDIConfig
        from transformers import (
            AutoTokenizer,
            CLIPProcessor,
            CLIPVisionModel,
            SiglipProcessor,
            SiglipVisionModel,
        )

        assert self.clip_model and self.siglip_model and self.mamba_model and self.ssr_midi
        assert self.midi_llm_model
        assert self.depthpro_root
        assert self.depthpro_checkpoint
        imported_depthpro = Path(depth_pro.__file__).resolve()
        if not imported_depthpro.is_relative_to(self.depthpro_root):
            raise RuntimeError(
                f"Imported depth_pro from {imported_depthpro}, expected checkout {self.depthpro_root}"
            )
        self.clip_processor = CLIPProcessor.from_pretrained(
            self.clip_model,
            **_local_pretrained_kwargs(self.clip_model, self.clip_model_revision),
        )
        self.clip = CLIPVisionModel.from_pretrained(
            self.clip_model,
            **_local_pretrained_kwargs(self.clip_model, self.clip_model_revision),
        ).to(self.device)
        self.siglip_processor = SiglipProcessor.from_pretrained(
            self.siglip_model,
            **_local_pretrained_kwargs(self.siglip_model, self.siglip_model_revision),
        )
        self.siglip = SiglipVisionModel.from_pretrained(
            self.siglip_model,
            **_local_pretrained_kwargs(self.siglip_model, self.siglip_model_revision),
        ).to(self.device)
        self.depthpro, self.depth_transform = depth_pro.create_model_and_transforms(
            checkpoint_uri=self.depthpro_checkpoint
        )
        self.depthpro = self.depthpro.to(self.device)
        self.mamba_tokenizer = AutoTokenizer.from_pretrained(
            self.mamba_model,
            **_local_pretrained_kwargs(self.mamba_model, self.mamba_model_revision),
        )
        self.mamba_tokenizer.add_tokens("<tor>", special_tokens=True)
        self.processor.tokenizer.add_tokens("<tor>", special_tokens=True)
        self.tor_token_ids = (
            self.mamba_tokenizer.convert_tokens_to_ids("<tor>"),
            self.processor.tokenizer.convert_tokens_to_ids("<tor>"),
        )
        midi_config = MIDIConfig.from_pretrained(
            self.ssr_midi,
            **_local_pretrained_kwargs(self.ssr_midi, self.ssr_midi_revision),
        )
        midi_config.mamba_path_or_name = self.mamba_model
        midi_config.llm_path_or_name = self.midi_llm_model
        self.midi = MIDI.from_pretrained(
            self.ssr_midi,
            config=midi_config,
            device_map=self.device,
            **_local_pretrained_kwargs(self.ssr_midi, self.ssr_midi_revision),
        )
        for module in [self.clip, self.siglip, self.depthpro, self.midi]:
            self._freeze(module)

    def _depth_rgb(self, image: Any) -> Any:
        import depth_pro
        import numpy as np

        loaded, _, f_px = depth_pro.load_pil(image.convert("RGB"))
        transformed = self.depth_transform(loaded).to(self.device)
        prediction = self.depthpro.infer(transformed, f_px=f_px)["depth"]
        raw = prediction.detach().cpu().numpy()
        span = float(raw.max() - raw.min())
        if span <= 0:
            raise ValueError("DepthPro returned a constant depth map")
        normalized = ((raw - raw.min()) / span * 255.0).astype(np.uint8)
        return np.stack([normalized, normalized, normalized], axis=-1)

    def generate(self, model_input: MSMUModelInput) -> GenerationResult:
        self._load()
        from qwen_vl_utils import process_vision_info

        original_image, vlm_image = ssr_image_views(model_input.image)
        question = ssr_question(self.profile.key, model_input.question)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": vlm_image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        generation_kwargs: dict[str, Any] = {
            "input_ids": inputs.input_ids.to(self.device),
            "attention_mask": inputs.attention_mask.to(self.device),
            "pixel_values": inputs.pixel_values.to(self.device),
            "image_grid_thw": inputs.image_grid_thw.to(self.device),
            **self.generation_kwargs,
        }
        generation_metadata: dict[str, Any] = {
            "num_model_image_tensors": 1,
            "tor_count": 0,
            "template_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        }
        if self.profile.key == "ssr_native":
            import numpy as np

            depth = self._depth_rgb(original_image)
            with self.torch.no_grad():
                image_embeds = self.clip(
                    **self.clip_processor(
                        images=np.array(original_image),
                        return_tensors="pt",
                    ).to(self.device)
                ).last_hidden_state.detach()
                depth_embeds = self.siglip(
                    **self.siglip_processor(images=depth, return_tensors="pt").to(self.device)
                ).last_hidden_state.detach()
            mamba_inputs = self.mamba_tokenizer(
                model_input.question + tor_prefix(),
                add_special_tokens=False,
                return_tensors="pt",
            )
            prefix_mask = self.torch.ones(
                (1, image_embeds.size(1) + depth_embeds.size(1)),
                dtype=mamba_inputs.attention_mask.dtype,
            )
            mamba_attention = self.torch.cat((prefix_mask, mamba_inputs.attention_mask), dim=1)
            with self.torch.no_grad():
                tor_embeds = self.midi(
                    mamba_input_ids=mamba_inputs.input_ids.to(self.device),
                    mamba_attention_mask=mamba_attention.to(self.device),
                    image_embeds=image_embeds.to(self.device),
                    depth_embeds=depth_embeds.to(self.device),
                    tor_token_id=self.tor_token_ids,
                    alignment=False,
                ).tor_embeds
            generation_kwargs.update({"tor_embeds": tor_embeds, "tor_token_id": self.tor_token_ids[1]})
            generation_metadata.update(
                {
                    "tor_count": TOR_COUNT,
                    "depth_derived_from_input_pixel_sha256": True,
                    "midi_enabled": True,
                }
            )
        with self.torch.inference_mode():
            output_ids = self.vlm.generate(**generation_kwargs)
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        prediction = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        warnings = ("model returned an empty text completion",) if not prediction.strip() else ()
        return GenerationResult(text=prediction, metadata=generation_metadata, warnings=warnings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=["ssr", "ssr_native"])
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-revision", default=SSR_BASE_MODEL_REVISION)
    parser.add_argument("--ssr-vlm", required=True)
    parser.add_argument("--ssr-vlm-revision", default=SSR_VLM_REVISION)
    parser.add_argument("--ssr-midi", default=None)
    parser.add_argument("--ssr-midi-revision", default=SSR_MIDI_REVISION)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--clip-model-revision", default=SSR_CLIP_MODEL_REVISION)
    parser.add_argument("--siglip-model", default=None)
    parser.add_argument("--siglip-model-revision", default=SSR_SIGLIP_MODEL_REVISION)
    parser.add_argument("--mamba-model", default=None)
    parser.add_argument("--mamba-model-revision", default=SSR_MAMBA_MODEL_REVISION)
    parser.add_argument("--midi-llm-model", default=None)
    parser.add_argument("--midi-llm-model-revision", default=SSR_MIDI_LLM_MODEL_REVISION)
    parser.add_argument("--depthpro-checkpoint", default=None)
    parser.add_argument("--depthpro-root", default=None)
    parser.add_argument(
        "--depthpro-checkpoint-sha256",
        default=SSR_DEPTHPRO_CHECKPOINT_SHA256,
    )
    parser.add_argument("--depthpro-revision", default=SSR_DEPTHPRO_REVISION)
    parser.add_argument("--device", default="cuda")
    add_msmu_run_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = SSRAdapter(
        profile_key=args.profile,
        upstream_root=args.upstream_root,
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        ssr_vlm=args.ssr_vlm,
        ssr_vlm_revision=args.ssr_vlm_revision,
        ssr_midi=args.ssr_midi,
        ssr_midi_revision=args.ssr_midi_revision,
        clip_model=args.clip_model,
        clip_model_revision=args.clip_model_revision,
        siglip_model=args.siglip_model,
        siglip_model_revision=args.siglip_model_revision,
        mamba_model=args.mamba_model,
        mamba_model_revision=args.mamba_model_revision,
        midi_llm_model=args.midi_llm_model,
        midi_llm_model_revision=args.midi_llm_model_revision,
        depthpro_root=args.depthpro_root,
        depthpro_checkpoint=args.depthpro_checkpoint,
        depthpro_checkpoint_sha256=args.depthpro_checkpoint_sha256,
        depthpro_revision=args.depthpro_revision,
        device=args.device,
    )
    execute_msmu_cli(args, adapter)


if __name__ == "__main__":
    main()
