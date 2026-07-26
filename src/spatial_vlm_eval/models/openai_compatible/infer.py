"""Run closed APIs or vLLM-served general VLMs on MSMU."""

from __future__ import annotations

import argparse
import os

from ..common.cli import add_msmu_run_arguments, execute_msmu_cli
from ..profiles import get_profile, profile_keys
from .client import OpenAICompatibleAdapter

PROFILE_KEYS = profile_keys("closed", "llava_next", "internvl3")
DEFAULT_BASE_URLS = {
    "vllm": "http://127.0.0.1:18081/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
}
DEFAULT_KEY_ENVS = {
    "vllm": "VLLM_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=PROFILE_KEYS)
    parser.add_argument("--backend", required=True, choices=sorted(DEFAULT_BASE_URLS))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--metadata-retries", type=int, default=4)
    add_msmu_run_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_profile(args.profile)
    if args.model_revision is not None and args.model_revision != profile.revision:
        raise ValueError(
            f"Profile {profile.key} is locked to revision {profile.revision}; got {args.model_revision}"
        )
    key_env = args.api_key_env or DEFAULT_KEY_ENVS[args.backend]
    api_key = os.environ.get(key_env, "")
    if args.backend == "vllm" and not api_key:
        api_key = "local"
    if not api_key:
        raise RuntimeError(f"Set {key_env} in the environment; API keys are never accepted as CLI arguments")
    adapter = OpenAICompatibleAdapter(
        profile=profile,
        backend=args.backend,
        base_url=args.base_url or DEFAULT_BASE_URLS[args.backend],
        api_key=api_key,
        served_model_name=args.served_model_name,
        timeout=args.timeout,
        metadata_retries=args.metadata_retries,
    )
    execute_msmu_cli(args, adapter)


if __name__ == "__main__":
    main()
