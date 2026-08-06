from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.q_spatial.scorer import (
    SCORER_PROTOCOL,
    parse_legacy_notebook,
    parse_measurement,
    score_predictions,
    score_rows,
)
from spatial_vlm_eval.benchmarks.q_spatial.data import DATASET_REVISION
from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILES

from tests.benchmarks.q_spatial.helpers import OfficialScoringContract


class QSpatialScorerTest(unittest.TestCase):
    def test_protocol_is_locked(self):
        self.assertEqual(
            SCORER_PROTOCOL,
            "q_spatial_robust_numeric_v1_standard_prompt_tag_first_unique_fallback_paper_inclusive_ratio",
        )

    def test_tag_mode_is_unique_strict_and_never_falls_back(self):
        valid = parse_measurement(r"reasoning \scalar{2.5} \distance_unit{meters}")
        self.assertEqual(valid.status, "tag_valid")
        self.assertEqual(str(valid.centimeters), "250.0")
        cases = {
            r"\scalar{1} final answer: 1 meter": "tag_malformed_or_non_unique",
            "scalar 1 cm": "tag_malformed_or_non_unique",
            r"\scalar{1 2} \distance_unit{cm}": "tag_invalid_scalar",
            r"\scalar{-1} \distance_unit{cm}": "tag_invalid_scalar",
            r"\scalar{1e2} \distance_unit{cm}": "tag_invalid_scalar",
            r"\scalar{1} \distance_unit{yard}": "tag_unknown_unit",
            r"\scalar{1} \distance_unit{cm} \scalar{2} \distance_unit{cm}": "tag_malformed_or_non_unique",
        }
        for text, status in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_measurement(text).status, status)

    def test_fallback_uses_final_region_and_rejects_ranges_or_conflicts(self):
        self.assertEqual(parse_measurement("work 8 cm\nFinal answer: 2 feet").status, "fallback_final_answer_valid")
        self.assertEqual(str(parse_measurement("Answer: .5 m").centimeters), "50.0")
        self.assertIn("multiple_numbers", parse_measurement("Answer: 2 to 3 meters").status)
        self.assertIn("multiple_numbers", parse_measurement("Answer: 2 m or 3 ft").status)
        self.assertIn("invalid_scalar", parse_measurement("Answer: 1e2 cm").status)
        self.assertIn("unknown_unit", parse_measurement("Answer: 2 yards").status)
        self.assertEqual(parse_measurement("thinking 2 cm\n3 mm").status, "fallback_last_nonempty_line_valid")

    def test_units_and_legacy_notebook_behavior(self):
        expected = {"1 m": "100", "1 mm": "0.1", "1 ft": "30.48", "1 inch": "2.54"}
        for text, centimeters in expected.items():
            self.assertEqual(str(parse_measurement(text).centimeters), centimeters)
        legacy = parse_legacy_notebook(r"\scalar{2 and 4} \distance_unit{unknown}")
        self.assertEqual(str(legacy.centimeters), "3")
        self.assertEqual(legacy.status, "audit_unknown_unit_as_centimeter")

    def test_inclusive_main_boundaries_and_split_macro(self):
        contract = OfficialScoringContract()
        rows = [
            {"index": index, "raw_prediction": r"\scalar{1} \distance_unit{cm}"}
            for index in range(len(contract))
        ]
        rows[0]["raw_prediction"] = r"\scalar{2} \distance_unit{cm}"
        rows[1]["raw_prediction"] = r"\scalar{1.25} \distance_unit{cm}"
        scored, summary = score_rows(rows, contract)
        self.assertTrue(scored[0]["success_delta_le_2"])
        self.assertFalse(scored[0]["legacy_success_delta_lt_2"])
        self.assertTrue(scored[1]["success_delta_le_1_25"])
        self.assertFalse(scored[1]["legacy_success_delta_lt_1_25"])
        scan = summary["split_metrics"]["QSpatial_scannet"]["delta_le_1_25"]["accuracy"]
        plus = summary["split_metrics"]["QSpatial_plus"]["delta_le_1_25"]["accuracy"]
        self.assertEqual(summary["metrics"]["overall_delta_le_1_25"], (scan + plus) / 2)
        self.assertEqual(summary["metrics"]["micro_delta_le_1_25_audit_only"], 270 / 271)
        self.assertEqual(set(summary["scannet_type_metrics"]), {
            "object_width", "object_height", "horizontal_distance", "vertical_distance", "direct_distance"
        })

    def test_full_scorer_writes_validation_rows_summary_and_gates(self):
        contract = OfficialScoringContract()
        profile = PROFILES["qwen3_vl_2b"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "".join(
                    json.dumps({"index": index, "raw_prediction": r"\scalar{1} \distance_unit{cm}"}) + "\n"
                    for index in range(271)
                ),
                encoding="utf-8",
            )
            metadata = {
                "output": str(predictions.resolve()),
                "output_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "scorer_protocol": SCORER_PROTOCOL,
                "inference_protocol": profile.inference_protocol,
                "publishable_inference": True,
                "dataset": {
                    "revision": DATASET_REVISION,
                    "fingerprint": contract.dataset_fingerprint,
                    "official_test_size": 271,
                },
                "model": {
                    "profile": profile.key,
                    "model": profile.model,
                    "model_revision": profile.revision,
                    "input_profile": profile.input_profile,
                    "comparison_group": profile.comparison_group,
                    "inference_protocol": profile.inference_protocol,
                    "backend": "vllm",
                    "decoding": profile.decoding,
                    "seed_strategy": profile.seed_strategy,
                },
            }
            predictions.with_suffix(".jsonl.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            summary = score_predictions(predictions, contract)
            score_dir = root / "scores" / SCORER_PROTOCOL
            self.assertEqual(summary["num_scored_rows"], 271)
            self.assertTrue((score_dir / "prediction_validation.json").is_file())
            self.assertTrue((score_dir / "scored_rows.jsonl").is_file())
            gates = json.loads((score_dir / "publication_gates.json").read_text())
            self.assertTrue(gates["passed"])


if __name__ == "__main__":
    unittest.main()
