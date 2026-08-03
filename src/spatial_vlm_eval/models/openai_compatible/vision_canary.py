"""Synthetic spatial canary proving that an OpenAI-compatible endpoint reads images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ...benchmarks.msmu.data import MSMUModelInput
from ..common.runtime import atomic_write_json, pixel_sha256
from ..common.vision_canary import (
    VISION_CANARY_PROTOCOL,
    VISION_CANARY_QUESTION,
    make_vision_canary_image,
    validate_vision_canary_answer,
)
from ..profiles import get_profile, profile_keys
from .client import APIRequestError, OpenAICompatibleAdapter
from .infer import DEFAULT_BASE_URLS, DEFAULT_KEY_ENVS

PROFILE_KEYS = profile_keys("closed", "llava_next", "internvl3")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=PROFILE_KEYS)
    parser.add_argument("--backend", required=True, choices=sorted(DEFAULT_BASE_URLS))
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--metadata-retries", type=int, default=10)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def run_vision_canary(
    adapter: OpenAICompatibleAdapter,
    *,
    output: str | Path | None = None,
) -> dict[str, object]:
    """Make exactly one generation call and persist a fail-closed report."""

    image = make_vision_canary_image()
    adapter_metadata = adapter.metadata()
    report: dict[str, object] = {
        "passed": False,
        "status": "running",
        "canary_protocol": VISION_CANARY_PROTOCOL,
        "profile": adapter.profile.key,
        "model": adapter.profile.model,
        "model_revision": adapter.profile.revision,
        "inference_protocol": adapter.profile.inference_protocol,
        "backend": adapter.backend,
        "served_model_name": adapter.request_model,
        "base_url": adapter.base_url,
        "question": VISION_CANARY_QUESTION,
        "request_count": 1,
        "request_image_count": 1,
        "image_mode": image.mode,
        "image_size": list(image.size),
        "image_pixel_sha256": pixel_sha256(image),
        "provider_policy": adapter_metadata.get("provider_policy"),
    }
    report_path = Path(output).resolve() if output is not None else None
    if report_path is not None:
        atomic_write_json(report_path, report)
    try:
        result = adapter.generate(
            MSMUModelInput(index=-1, image=image, question=VISION_CANARY_QUESTION)
        )
        report["answer"] = result.text
        report["generation"] = dict(result.metadata)
        validate_vision_canary_answer(result.text)
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        if isinstance(error, APIRequestError):
            report["error_details"] = error.diagnostics()
        if report_path is not None:
            atomic_write_json(report_path, report)
        raise
    report["passed"] = True
    report["status"] = "passed"
    if report_path is not None:
        atomic_write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    key_env = args.api_key_env or DEFAULT_KEY_ENVS[args.backend]
    api_key = os.environ.get(key_env, "")
    if args.backend == "vllm" and not api_key:
        api_key = "local"
    if not api_key:
        raise RuntimeError(
            f"Set {key_env} in the environment; API keys are never accepted as CLI arguments"
        )
    adapter = OpenAICompatibleAdapter(
        profile=profile,
        backend=args.backend,
        base_url=args.base_url or DEFAULT_BASE_URLS[args.backend],
        api_key=api_key,
        served_model_name=args.served_model_name,
        timeout=args.timeout,
        metadata_retries=args.metadata_retries,
    )
    report = run_vision_canary(adapter, output=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
