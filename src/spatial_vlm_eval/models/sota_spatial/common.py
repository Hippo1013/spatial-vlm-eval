"""Shared provenance and MSMU vision-canary helpers for SOTA model families."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ...benchmarks.msmu.data import MSMUModelInput
from ..common.runtime import (
    GenerationResult,
    InferenceAdapter,
    atomic_write_json,
    pixel_sha256,
)
from ..common.vision_canary import (
    COLOR_CANARY_QUESTION,
    VISION_CANARY_PROTOCOL,
    VISION_CANARY_QUESTION,
    make_solid_color_canary,
    make_vision_canary_image,
    validate_solid_color_canary_answer,
    validate_vision_canary_answer,
)
from ..profiles import PROFILES


SPATIALLADDER_BATCH_CANDIDATES = (16, 8, 4, 2, 1)


def seed_everything(seed: int) -> None:
    """Reset every available local RNG before one locked model generation."""

    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except ImportError:
        pass
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(value: Any) -> str:
    """Hash tensor shape, dtype, and contiguous CPU bytes without serializing it."""

    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def adapter_source_digest(profile_key: str) -> str:
    """Bind a supplement journal to the selected family implementation."""

    profile = PROFILES[str(profile_key)]
    directory = Path(__file__).resolve().parent
    family_file = {
        "robobrain25": directory / "robobrain25.py",
        "hispatial": directory / "hispatial.py",
        "spatialladder": directory / "spatialladder.py",
    }.get(profile.family)
    if family_file is None:
        raise ValueError(f"Profile is not an MSMU SOTA supplement family: {profile_key}")
    files = [
        Path(__file__).resolve(),
        family_file.resolve(),
        (directory / "cli.py").resolve(),
    ]
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            asdict(profile),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    for path in sorted(files, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CanaryInput:
    index: int
    image: Image.Image
    question: str


def _capacity_candidates() -> tuple[int, ...]:
    raw = os.environ.get("SPATIALLADDER_BATCH_CANDIDATES", "").strip()
    if not raw:
        return SPATIALLADDER_BATCH_CANDIDATES
    candidates = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("SPATIALLADDER_BATCH_CANDIDATES must contain positive integers")
    if len(candidates) != len(set(candidates)):
        raise ValueError("SPATIALLADDER_BATCH_CANDIDATES contains duplicates")
    return candidates


def _color_cases(count: int) -> list[tuple[str, CanaryInput]]:
    short = COLOR_CANARY_QUESTION
    long = (
        "Inspect the complete uniformly colored image carefully. After checking every visible "
        "pixel, state the single color that fills the image. Answer concisely in English."
    )
    combinations = (("red", short), ("blue", long), ("blue", short), ("red", long))
    return [
        (
            color,
            CanaryInput(
                index=-1000 - offset,
                image=make_solid_color_canary(color),
                question=question,
            ),
        )
        for offset in range(count)
        for color, question in (combinations[offset % len(combinations)],)
    ]


def _solid_color_passed(text: str, expected: str) -> bool:
    try:
        validate_solid_color_canary_answer(text, expected)
    except ValueError:
        return False
    other = "blue" if expected == "red" else "red"
    words = set(re.findall(r"[a-z]+", str(text).casefold()))
    return other not in words


def probe_spatialladder_native_batch(adapter: InferenceAdapter) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for candidate in _capacity_candidates():
        try:
            adapter.batch_size = candidate
            cases = _color_cases(candidate)
            results = adapter.generate_batch([value for _expected, value in cases])
            prompt_lengths = sorted({len(value.question) for _expected, value in cases})
            heterogeneous = candidate == 1 or len(prompt_lengths) > 1
            padding_proved = len(results) == candidate and all(
                result.metadata.get("tokenizer_padding_side") == "left" for result in results
            )
            passed = (
                len(results) == candidate
                and heterogeneous
                and padding_proved
                and all(
                    _solid_color_passed(result.text, expected)
                    for (expected, _value), result in zip(cases, results, strict=True)
                )
            )
            attempts.append(
                {
                    "candidate": candidate,
                    "passed": passed,
                    "prompt_character_lengths": prompt_lengths,
                    "heterogeneous_prompt_lengths": heterogeneous,
                    "tokenizer_padding_side": "left" if padding_proved else None,
                }
            )
            if passed:
                adapter.batch_size = candidate
                return {
                    "passed": True,
                    "selected_capacity": candidate,
                    "capacity_kind": "native_batch",
                    "heterogeneous_prompt_lengths": heterogeneous,
                    "tokenizer_padding_side": "left",
                    "attempts": attempts,
                }
        except Exception as exc:  # noqa: BLE001 - a larger candidate may legitimately OOM.
            attempts.append(
                {
                    "candidate": candidate,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )
    raise RuntimeError(f"No stable SpatialLadder native batch candidate: {attempts}")


def run_msmu_vision_canary(
    adapter: InferenceAdapter,
    output: str | Path,
    *,
    native_batch_probe: bool,
) -> dict[str, Any]:
    """Prove MSMU visual binding, then optionally prove native left-padded batching."""

    metadata = adapter.metadata()
    image = make_vision_canary_image()
    result = adapter.generate(
        CanaryInput(index=-1, image=image, question=VISION_CANARY_QUESTION)
    )
    if result.metadata.get("num_model_image_tensors") != 1:
        raise ValueError("MSMU vision canary did not prove exactly one model image tensor")
    validate_vision_canary_answer(result.text)
    capacity = (
        probe_spatialladder_native_batch(adapter)
        if native_batch_probe
        else {
            "passed": True,
            "selected_capacity": 1,
            "capacity_kind": "batch1",
            "attempts": [{"candidate": 1, "passed": True}],
        }
    )
    report = {
        "passed": True,
        "profile": metadata["profile"],
        "model": metadata["model"],
        "model_revision": metadata["model_revision"],
        "inference_protocol": metadata["inference_protocol"],
        "adapter_source_sha256": metadata["adapter_source_sha256"],
        "canary_protocol": VISION_CANARY_PROTOCOL,
        "question": VISION_CANARY_QUESTION,
        "request_count": 1,
        "request_image_count": 1,
        "image_mode": "RGB",
        "image_size": list(image.size),
        "image_pixel_sha256": pixel_sha256(image),
        "answer": result.text,
        "generation": dict(result.metadata),
        "capacity_probe": capacity,
    }
    atomic_write_json(Path(output).resolve(), report)
    return report


def generation_kwargs(profile_key: str) -> dict[str, Any]:
    profile = PROFILES[profile_key]
    values: dict[str, Any] = {
        "do_sample": profile.do_sample,
        "num_beams": profile.num_beams,
        "max_new_tokens": profile.max_new_tokens,
        "use_cache": profile.use_cache,
    }
    if profile.temperature is not None:
        values["temperature"] = profile.temperature
    if profile.top_p is not None:
        values["top_p"] = profile.top_p
    if profile.repetition_penalty is not None:
        values["repetition_penalty"] = profile.repetition_penalty
    return values


def close_torch_model(adapter: Any, names: Iterable[str]) -> None:
    for name in names:
        if hasattr(adapter, name):
            delattr(adapter, name)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
