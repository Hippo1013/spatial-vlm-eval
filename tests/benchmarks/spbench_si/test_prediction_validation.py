from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.spbench_si.prediction_validation import validate_prediction_rows

from tests.benchmarks.spbench_si.helpers import small_contract


class SPBenchSIPredictionValidationTest(unittest.TestCase):
    def test_full_and_subset_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = small_contract(Path(directory))
            rows = [{"index": index, "raw_prediction": "A"} for index in range(4)]
            full = validate_prediction_rows(rows, contract, prediction_path="predictions.jsonl")
            subset = validate_prediction_rows(rows[:2], contract, prediction_path="subset.jsonl", allow_subset=True)
            rejected = validate_prediction_rows(rows[:2], contract, prediction_path="subset.jsonl")
        self.assertTrue(full["passed"])
        self.assertTrue(subset["passed"])
        self.assertFalse(rejected["passed"])

    def test_extra_fields_duplicates_and_non_integer_indices_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = small_contract(Path(directory))
            rows = [
                {"index": 0, "raw_prediction": "1", "ground_truth": "leak"},
                {"index": 0, "raw_prediction": "1"},
                {"index": "2", "raw_prediction": "1"},
            ]
            report = validate_prediction_rows(rows, contract, prediction_path="bad.jsonl", allow_subset=True)
        self.assertFalse(report["passed"])
        self.assertTrue(any("unexpected keys" in value for value in report["errors"]))
        self.assertTrue(any("duplicate" in value for value in report["errors"]))


if __name__ == "__main__":
    unittest.main()
