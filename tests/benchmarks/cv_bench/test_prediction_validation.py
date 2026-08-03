from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.cv_bench.data import OFFICIAL_TEST_SIZE
from spatial_vlm_eval.benchmarks.cv_bench.prediction_validation import validate_prediction_rows

from tests.benchmarks.cv_bench.helpers import official_fake_contract


class CVBenchPredictionValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.contract = official_fake_contract(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_schema_and_index_coverage_pass(self):
        rows = [{"index": index, "raw_prediction": "A"} for index in range(OFFICIAL_TEST_SIZE)]
        report = validate_prediction_rows(
            rows,
            self.contract,
            prediction_path=self.root / "predictions.jsonl",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["num_unique_indices"], OFFICIAL_TEST_SIZE)

    def test_duplicate_missing_extra_fields_and_non_string_predictions_fail(self):
        rows = [
            {"index": 0, "raw_prediction": "A", "answer": "(A)"},
            {"index": 0, "raw_prediction": 1},
        ]
        report = validate_prediction_rows(
            rows,
            self.contract,
            prediction_path=self.root / "bad.jsonl",
        )
        self.assertFalse(report["passed"])
        rendered = "\n".join(report["errors"])
        self.assertIn("unexpected keys", rendered)
        self.assertIn("duplicate indices", rendered)
        self.assertIn("missing indices", rendered)
        self.assertIn("must be a string", rendered)

    def test_subset_only_passes_with_explicit_debug_flag(self):
        rows = [{"index": 0, "raw_prediction": ""}]
        failed = validate_prediction_rows(
            rows, self.contract, prediction_path=self.root / "subset.jsonl"
        )
        passed = validate_prediction_rows(
            rows,
            self.contract,
            prediction_path=self.root / "subset.jsonl",
            allow_subset=True,
        )
        self.assertFalse(failed["passed"])
        self.assertTrue(passed["passed"])
        self.assertTrue(passed["warnings"])


if __name__ == "__main__":
    unittest.main()
