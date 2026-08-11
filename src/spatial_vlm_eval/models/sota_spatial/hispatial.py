"""HiSpatial official RGB-to-MoGe-2-XYZ inference under the MSMU contract."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from ..common.provenance import verify_git_checkout, verify_hf_snapshot_revision
from ..common.runtime import GenerationResult, InferenceAdapter, RestrictedVisionInput, pixel_sha256
from ..profiles import PROFILES
from .common import (
    adapter_source_digest,
    close_torch_model,
    run_msmu_vision_canary,
    seed_everything,
    sha256_file,
    tensor_sha256,
)


MOGE2_MODEL_ID = "Ruicheng/moge-2-vitl-normal"
MOGE2_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MOGE2_UPSTREAM_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE2_UTILS3D_COMMIT = "3fab839f0be9931dac7c8488eb0e1600c236e183"
MOGE2_CHECKPOINT_FILENAME = "model.pt"


class HiSpatialAdapter(InferenceAdapter):
    batch_size = 1
    supports_concurrency = False

    def __init__(
        self,
        *,
        model_path: str,
        upstream_root: str,
        moge_model_path: str,
        moge_upstream_root: str,
        moge_utils3d_root: str,
    ) -> None:
        self.profile = PROFILES["hispatial3b_moge2_xyz"]
        self.model_path = str(model_path)
        self.upstream_root = Path(upstream_root).resolve()
        self.moge_model_path = str(moge_model_path)
        self.moge_upstream_root = Path(moge_upstream_root).resolve()
        self.moge_utils3d_root = Path(moge_utils3d_root).resolve()
        if not verify_hf_snapshot_revision(
            self.model_path, self.profile.revision, self.profile.model
        ):
            raise ValueError("HiSpatial local checkpoint revision is not verifiable")
        if not verify_hf_snapshot_revision(
            self.moge_model_path, MOGE2_REVISION, MOGE2_MODEL_ID
        ):
            raise ValueError("MoGe-2 local checkpoint revision is not verifiable")
        self.moge_checkpoint = Path(self.moge_model_path) / MOGE2_CHECKPOINT_FILENAME
        if not self.moge_checkpoint.is_file():
            raise FileNotFoundError(f"Locked MoGe-2 checkpoint is missing: {self.moge_checkpoint}")
        for root, commit, label in (
            (self.upstream_root, self.profile.upstream_commit or "", "HiSpatial"),
            (self.moge_upstream_root, MOGE2_UPSTREAM_COMMIT, "MoGe-2"),
            (self.moge_utils3d_root, MOGE2_UTILS3D_COMMIT, "MoGe-2 utils3d"),
        ):
            if not verify_git_checkout(root, commit, label):
                raise ValueError(f"{label} checkout is not verifiable")
        self._loaded = False

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.profile.model,
            "model_revision": self.profile.revision,
            "model_path": self.model_path,
            "backend": "official-hispatial-predictor-with-locked-moge2",
            "profile": self.profile.key,
            "input_profile": self.profile.input_profile,
            "inference_protocol": self.profile.inference_protocol,
            "adapter_source_sha256": adapter_source_digest(self.profile.key),
            "chat_template": self.profile.chat_template,
            "image_processing": {
                "source": "current MSMU RGB only",
                "source_image_count": 1,
                "model_image_tensor_count": 1,
                "derived_xyz_tensor_count": 1,
                "derived_xyz_source": "same current MSMU RGB",
                "derived_xyz_model": MOGE2_MODEL_ID,
                "derived_xyz_revision": MOGE2_REVISION,
                "derived_xyz_upstream_commit": MOGE2_UPSTREAM_COMMIT,
                "derived_xyz_utils3d_commit": MOGE2_UTILS3D_COMMIT,
                "derived_xyz_shape": [4, 448, 448],
                "ground_truth_depth_or_xyz": None,
            },
            "decoding": {
                "do_sample": False,
                "max_new_tokens": 100,
                "seed": self.profile.seed,
                "implemented_by": "HiSpatialPredictor.query",
            },
            "upstream": {
                "repository": self.profile.upstream_url,
                "commit": self.profile.upstream_commit,
                "checkout": str(self.upstream_root),
                "commit_verified": True,
                "model_snapshot_revision_verified": True,
                "moge_repository": "https://github.com/microsoft/MoGe",
                "moge_checkout": str(self.moge_upstream_root),
                "moge_checkpoint": str(self.moge_checkpoint),
                "moge_utils3d_checkout": str(self.moge_utils3d_root),
                "entrypoint_equivalent": "hispatial.inference.HiSpatialPredictor + MoGeProcessor",
            },
        }

    def _load(self) -> None:
        if self._loaded:
            return
        if str(self.upstream_root) not in sys.path:
            sys.path.insert(0, str(self.upstream_root))
        if str(self.moge_upstream_root) not in sys.path:
            sys.path.insert(0, str(self.moge_upstream_root))
        import torch
        import utils3d
        from hispatial.inference import HiSpatialPredictor, MoGeProcessor
        from moge.model.v2 import MoGeModel

        installed_root = Path(utils3d.__file__).resolve().parent
        source_root = self.moge_utils3d_root / "utils3d"
        for relative in (Path("__init__.py"), Path("torch/__init__.py")):
            installed = installed_root / relative
            source = source_root / relative
            if not installed.is_file() or not source.is_file():
                raise FileNotFoundError(f"Locked MoGe-2 utils3d file is missing: {relative}")
            if sha256_file(installed) != sha256_file(source):
                raise ValueError(f"Installed utils3d differs from locked checkout: {relative}")
        if not hasattr(utils3d, "pt"):
            raise ValueError("Locked MoGe-2 utils3d must expose the utils3d.pt alias")

        self.moge = MoGeProcessor.__new__(MoGeProcessor)
        self.moge.device = torch.device("cuda")
        self.moge.model = MoGeModel.from_pretrained(str(self.moge_checkpoint)).to(
            self.moge.device
        ).eval()
        self.moge.img_size = 448
        self.predictor = HiSpatialPredictor(model_load_path=self.model_path, gpu_rank=0)
        self._loaded = True

    def generate(self, model_input: RestrictedVisionInput) -> GenerationResult:
        self._load()
        import numpy as np

        seed_everything(self.profile.seed)
        rgb = model_input.image.convert("RGB")
        source_digest = pixel_sha256(rgb)
        xyz = self.moge.apply_transform(rgb)
        if tuple(int(value) for value in xyz.shape) != (4, 448, 448):
            raise ValueError(f"HiSpatial derived XYZ shape mismatch: {tuple(xyz.shape)}")
        text = self.predictor.query(
            image=np.asarray(rgb),
            prompt=str(model_input.question),
            xyz_values=xyz,
        )
        rendered = (
            str(model_input.question)
            if "<image>" in str(model_input.question)
            else "<image>" + str(model_input.question)
        )
        return GenerationResult(
            text=str(text),
            metadata={
                "num_model_image_tensors": 1,
                "derived_xyz_tensors": 1,
                "source_rgb_sha256": source_digest,
                "derived_from_source_rgb_sha256": source_digest,
                "derived_xyz_sha256": tensor_sha256(xyz),
                "derived_xyz_model": MOGE2_MODEL_ID,
                "derived_xyz_revision": MOGE2_REVISION,
                "derived_xyz_upstream_commit": MOGE2_UPSTREAM_COMMIT,
                "derived_xyz_utils3d_commit": MOGE2_UTILS3D_COMMIT,
                "template_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            },
            warnings=("model returned an empty text completion",) if not str(text).strip() else (),
        )

    def run_vision_canary(self, output: str | Path) -> dict[str, Any]:
        return run_msmu_vision_canary(self, output, native_batch_probe=False)

    def close(self) -> None:
        if hasattr(self, "moge") and hasattr(self.moge, "model"):
            del self.moge.model
        close_torch_model(self, ("moge", "predictor"))
        self._loaded = False
