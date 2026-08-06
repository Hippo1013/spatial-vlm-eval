from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.q_spatial.prediction_validation import validate_prediction_rows

from tests.benchmarks.q_spatial.helpers import small_contract


class QSpatialPredictionValidationTest(unittest.TestCase):
    def test_schema_is_exact_and_empty_prediction_is_only_a_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = small_contract(Path(directory))
            report = validate_prediction_rows(
                [{"index": 0, "raw_prediction": ""}, {"index": 1, "raw_prediction": "2 cm"}],
                contract,
                prediction_path=Path(directory) / "predictions.jsonl",
            )
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["warnings"]), 1)
            report = validate_prediction_rows(
                [{"index": 0, "raw_prediction": "1 m", "answer": 1}],
                contract,
                prediction_path=Path(directory) / "predictions.jsonl",
                allow_subset=True,
            )
            self.assertFalse(report["passed"])
            self.assertIn("unexpected keys", report["errors"][0])

    def test_subset_never_passes_as_full(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = small_contract(Path(directory))
            rows = [{"index": 0, "raw_prediction": "1 meter"}]
            debug = validate_prediction_rows(rows, contract, prediction_path="debug.jsonl", allow_subset=True)
            full = validate_prediction_rows(rows, contract, prediction_path="full.jsonl")
            self.assertTrue(debug["passed"])
            self.assertFalse(full["passed"])


if __name__ == "__main__":
    unittest.main()
