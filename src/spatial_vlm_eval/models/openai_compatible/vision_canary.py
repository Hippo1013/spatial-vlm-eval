"""Synthetic red/blue canary proving that a vLLM chat endpoint reads images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from ...benchmarks.msmu.data import MSMUModelInput
from ..common.runtime import atomic_write_json
from ..common.vision_canary import SOLID_COLOR_QUESTION, validate_solid_color_answers
from ..profiles import get_profile, profile_keys
from .client import OpenAICompatibleAdapter

PROFILE_KEYS = profile_keys("llava_next", "internvl3")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=PROFILE_KEYS)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    adapter = OpenAICompatibleAdapter(
        profile=profile,
        backend="vllm",
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env, "local"),
        served_model_name=args.served_model_name,
        timeout=args.timeout,
    )
    red = adapter.generate(
        MSMUModelInput(0, Image.new("RGB", (64, 64), "red"), SOLID_COLOR_QUESTION)
    )
    blue = adapter.generate(
        MSMUModelInput(1, Image.new("RGB", (64, 64), "blue"), SOLID_COLOR_QUESTION)
    )
    validate_solid_color_answers(red.text, blue.text)
    report = {
        "passed": True,
        "profile": profile.key,
        "model": profile.model,
        "model_revision": profile.revision,
        "served_model_name": args.served_model_name,
        "base_url": args.base_url,
        "request_image_count": 1,
        "red_answer": red.text,
        "blue_answer": blue.text,
        "generation_ids": [red.metadata.get("generation_id"), blue.metadata.get("generation_id")],
    }
    if args.output:
        atomic_write_json(Path(args.output).resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
