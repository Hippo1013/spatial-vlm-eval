"""Shared command-line arguments for MSMU inference adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ...benchmarks.msmu.data import MSMUTestContract
from .runtime import InferenceAdapter, run_msmu_inference, select_target_indices


def add_msmu_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", "--run-metadata", dest="metadata", default=None)
    parser.add_argument("--journal", default=None)
    parser.add_argument(
        "--indices",
        default=None,
        help="Debug-only comma/range selection, for example 0,7,10-14.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug-only prefix of selected indices.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")


def execute_msmu_cli(args: Any, adapter: InferenceAdapter) -> dict[str, Any]:
    contract = MSMUTestContract(args.dataset_root, require_official_size=True)
    indices = select_target_indices(len(contract), indices=args.indices, limit=args.limit)
    metadata = run_msmu_inference(
        contract=contract,
        adapter=adapter,
        output=Path(args.output),
        target_indices=indices,
        journal_path=args.journal,
        metadata_path=args.metadata,
        retries=args.retries,
        workers=args.workers,
        resume=not args.no_resume,
    )
    print(f"Wrote {metadata['num_predictions']} predictions to {metadata['output']}")
    print(
        f"Run metadata: {Path(args.metadata).resolve() if args.metadata else metadata['output'] + '.metadata.json'}"
    )
    if metadata["dataset"]["is_subset"]:
        print("WARNING: subset inference is debug-only and cannot pass the scorer preflight")
    return metadata
