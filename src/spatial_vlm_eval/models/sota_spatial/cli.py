"""MSMU-only CLI binding the SOTA family adapters to the benchmark contract."""

from __future__ import annotations

import argparse

from ..common.cli import add_msmu_run_arguments, execute_msmu_cli
from ..profiles import SOTA_SUPPLEMENT_PROFILE_KEYS
from .common import adapter_source_digest
from .hispatial import HiSpatialAdapter
from .robobrain25 import ROBOBRAIN_PROFILE_KEYS, RoboBrain25Adapter
from .spatialladder import SPATIALLADDER_PROFILE_KEYS, SpatialLadderAdapter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=SOTA_SUPPLEMENT_PROFILE_KEYS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--moge-model")
    parser.add_argument("--moge-upstream-root")
    parser.add_argument("--moge-utils3d-root")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vision-canary-report")
    parser.add_argument("--print-adapter-digest", action="store_true")
    add_msmu_run_arguments(parser)
    return parser.parse_args(argv)


def build_adapter(args: argparse.Namespace):
    if args.profile in ROBOBRAIN_PROFILE_KEYS:
        return RoboBrain25Adapter(
            profile_key=args.profile,
            model_path=args.model,
            upstream_root=args.upstream_root,
        )
    if args.profile == "hispatial3b_moge2_xyz":
        missing = [
            name
            for name, value in (
                ("--moge-model", args.moge_model),
                ("--moge-upstream-root", args.moge_upstream_root),
                ("--moge-utils3d-root", args.moge_utils3d_root),
            )
            if not value
        ]
        if missing:
            raise ValueError("HiSpatial requires " + ", ".join(missing))
        return HiSpatialAdapter(
            model_path=args.model,
            upstream_root=args.upstream_root,
            moge_model_path=args.moge_model,
            moge_upstream_root=args.moge_upstream_root,
            moge_utils3d_root=args.moge_utils3d_root,
        )
    if args.profile in SPATIALLADDER_PROFILE_KEYS:
        return SpatialLadderAdapter(
            profile_key=args.profile,
            model_path=args.model,
            upstream_root=args.upstream_root,
            batch_size=args.batch_size,
        )
    raise ValueError(f"Unsupported MSMU SOTA supplement profile: {args.profile}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.print_adapter_digest:
        print(adapter_source_digest(args.profile))
        return
    adapter = build_adapter(args)
    if args.vision_canary_report:
        try:
            adapter.run_vision_canary(args.vision_canary_report)
        except Exception:
            adapter.close()
            raise
    execute_msmu_cli(args, adapter)


if __name__ == "__main__":
    main()
