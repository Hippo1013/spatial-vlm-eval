"""CV-Bench benchmark-owned contracts, validation, scoring, and reporting."""

from .data import DATASET_REVISION, OFFICIAL_TEST_SIZE
from .scorer import SCORER_PROTOCOL

__all__ = [
    "DATASET_REVISION",
    "OFFICIAL_TEST_SIZE",
    "SCORER_PROTOCOL",
]
