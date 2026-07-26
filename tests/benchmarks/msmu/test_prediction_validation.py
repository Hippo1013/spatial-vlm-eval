import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_vlm_eval.benchmarks.msmu.prediction_validation import (
    validate_prediction_rows,
)
from spatial_vlm_eval.benchmarks.msmu.scorer import (
    validated_predictions_for_scoring,
)


def source_row() -> dict:
    return {
        "type": "width",
        "conversations": {
            "from": ["human", "gpt"],
            "value": ["<image>\nHow wide is the table?", "The table is 1 meter wide."],
        },
    }


def prediction_row(prediction: str = "1 meter") -> dict:
    return {
        "index": 0,
        "raw_type": "width",
        "task_family": "scale_estimation",
        "question": "How wide is the table?",
        "reference": "The table is 1 meter wide.",
        "prediction": prediction,
    }


class PredictionValidationTest(unittest.TestCase):
    def validate(self, row: dict) -> dict:
        return validate_prediction_rows(
            [row],
            [source_row()],
            prediction_path=Path("predictions.jsonl"),
            dataset_root=Path("MSMU"),
        )

    def test_empty_prediction_is_a_non_blocking_warning(self):
        report = self.validate(prediction_row("  "))
        self.assertTrue(report["passed"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("empty prediction", report["warnings"][0])

    def test_provenance_mismatch_is_a_hard_error(self):
        row = prediction_row()
        row["reference"] = "rewritten reference"
        report = self.validate(row)
        self.assertFalse(report["passed"])
        self.assertIn("reference mismatch", report["errors"][0])


class ScorerPreflightTest(unittest.TestCase):
    def test_invalid_report_aborts_before_scoring(self):
        report = {
            "passed": False,
            "num_prediction_rows": 1,
            "warnings": [],
            "errors": ["index=0: reference mismatch"],
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "prediction_validation.json"
            with patch(
                "spatial_vlm_eval.benchmarks.msmu.scorer.validate_predictions",
                return_value=([prediction_row()], report),
            ):
                with self.assertRaisesRegex(ValueError, "no judge requests were made"):
                    validated_predictions_for_scoring("predictions.jsonl", "MSMU", report_path)
            self.assertTrue(report_path.exists())

    def test_warning_only_report_is_allowed(self):
        rows = [prediction_row("")]
        report = {
            "passed": True,
            "num_prediction_rows": 1,
            "warnings": ["index=0: empty prediction (will score zero or parse failure)"],
            "errors": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "prediction_validation.json"
            with patch(
                "spatial_vlm_eval.benchmarks.msmu.scorer.validate_predictions",
                return_value=(rows, report),
            ):
                validated = validated_predictions_for_scoring(
                    "predictions.jsonl", "MSMU", report_path
                )
            self.assertEqual(validated, rows)
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
