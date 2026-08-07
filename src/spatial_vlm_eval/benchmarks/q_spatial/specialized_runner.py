"""Persistent dataset-blind runners for Q-Spatial specialized tracks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from PIL import Image

from ..cv_bench import specialized_runner as shared_specialized
from ...models.common.runtime import GenerationResult, InferenceAdapter
from .command_adapter import fold_system_user_prompt, load_generation_manifest
from .profiles import PROFILE_SEQUENCE, PROFILES, QSpatialProfile

SPECIALIZED_PROFILE_KEYS = tuple(
    key for key in PROFILE_SEQUENCE if PROFILES[key].adapter_kind == "upstream_command"
)
_REQUEST_KEYS = {
    "schema_version",
    "action",
    "index",
    "system_prompt",
    "user_prompt",
    "image",
    "profile",
    "model_revision",
    "upstream_commit",
    "inference_protocol",
    "decoding",
}


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
    profile: QSpatialProfile,
    decoding: dict[str, Any],
) -> tuple[shared_specialized.RunnerModelInput, str, str, str]:
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS:
        extra = sorted(set(request) - _REQUEST_KEYS) if isinstance(request, dict) else []
        missing = sorted(_REQUEST_KEYS - set(request)) if isinstance(request, dict) else []
        raise ValueError(f"Runner request schema mismatch: extra={extra}, missing={missing}")
    for key, expected in {
        "schema_version": 1,
        "action": "generate",
        "profile": profile.key,
        "model_revision": profile.revision,
        "upstream_commit": profile.upstream_commit,
        "inference_protocol": profile.inference_protocol,
        "decoding": decoding,
    }.items():
        if request.get(key) != expected:
            raise ValueError(
                f"Runner request {key} mismatch: got={request.get(key)!r}, expected={expected!r}"
            )
    if not isinstance(request.get("index"), int):
        raise ValueError("Runner index must be an integer")
    system_prompt = request.get("system_prompt")
    user_prompt = request.get("user_prompt")
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        raise ValueError("Runner requires separate string system_prompt and user_prompt fields")
    folded = fold_system_user_prompt(system_prompt, user_prompt)
    return (
        shared_specialized.RunnerModelInput(
            index=request["index"],
            image=_decode_image(request["image"]),
            question=folded,
        ),
        system_prompt,
        user_prompt,
        folded,
    )


def _source_digest_files(profile: QSpatialProfile) -> list[Path]:
    from . import command_adapter, profiles

    files = [
        Path(__file__),
        Path(command_adapter.__file__),
        Path(profiles.__file__),
        Path(shared_specialized.__file__),
    ]
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


def adapter_digest(profile: QSpatialProfile) -> str:
    digest = hashlib.sha256()
    for path in _source_digest_files(profile):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _response(
    request: dict[str, Any],
    profile: QSpatialProfile,
    decoding: dict[str, Any],
    result: GenerationResult,
    system_prompt: str,
    user_prompt: str,
    folded_prompt: str,
) -> dict[str, Any]:
    generation = dict(result.metadata)
    if profile.key == "spatialbot_zoedepth":
        expected_depth_evidence = {
            "num_model_image_tensors": 2,
            "input_rgb_count": 1,
            "derived_depth_count": 1,
            "depth_derived_from_same_rgb": True,
        }
        mismatches = {
            key: {"expected": expected, "actual": generation.get(key)}
            for key, expected in expected_depth_evidence.items()
            if generation.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "SpatialBot ZoeDepth backend must prove one input RGB plus one "
                f"same-RGB-derived depth tensor: {mismatches}"
            )
    elif generation.get("num_model_image_tensors") != 1:
        raise ValueError("Specialized backend must prove exactly one model-bound image tensor")
    generation.update(
        {
            "num_media_prompt": 1,
            "source_rgb_count": 1,
            "system_role_supported": False,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
            "folded_prompt_sha256": hashlib.sha256(folded_prompt.encode("utf-8")).hexdigest(),
        }
    )
    template_sha = generation.get("template_sha256")
    if not isinstance(template_sha, str) or len(template_sha) != 64:
        raise ValueError("Specialized backend did not return a rendered template SHA-256")
    return {
        "index": request["index"],
        "profile": profile.key,
        "model_revision": profile.revision,
        "inference_protocol": profile.inference_protocol,
        "decoding": decoding,
        "system_role_supported": False,
        "raw_prediction": result.text,
        "generation": generation,
        "warnings": list(result.warnings),
    }


def serve(profile: QSpatialProfile, decoding: dict[str, Any]) -> None:
    backend: InferenceAdapter | None = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request == {"schema_version": 1, "action": "close"}:
                return
            model_input, system_prompt, user_prompt, folded_prompt = _verify_request(
                request, profile, decoding
            )
            if backend is None:
                with redirect_stdout(sys.stderr):
                    backend = shared_specialized._build_backend(profile, decoding)
            shared_specialized._seed(decoding)
            with redirect_stdout(sys.stderr):
                result = backend.generate(model_input)
            print(
                json.dumps(
                    _response(
                        request,
                        profile,
                        decoding,
                        result,
                        system_prompt,
                        user_prompt,
                        folded_prompt,
                    ),
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
