"""Shared, benchmark-safe inference runtime primitives."""

from .runtime import GenerationResult, InferenceAdapter, run_msmu_inference

__all__ = ["GenerationResult", "InferenceAdapter", "run_msmu_inference"]
