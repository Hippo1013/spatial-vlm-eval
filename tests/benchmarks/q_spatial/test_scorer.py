from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.q_spatial.scorer import (
    LEGACY_SCORER_PROTOCOL_V1,
    SCORER_PROTOCOL,
    inference_metadata_scorer_protocol_is_compatible,
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
            "q_spatial_robust_numeric_v2_standard_prompt_declared_final_equivalent_tags_"
            "controlled_wrappers_paper_inclusive_ratio",
        )
        self.assertTrue(inference_metadata_scorer_protocol_is_compatible(SCORER_PROTOCOL))
        self.assertTrue(
            inference_metadata_scorer_protocol_is_compatible(LEGACY_SCORER_PROTOCOL_V1)
        )
        self.assertFalse(inference_metadata_scorer_protocol_is_compatible("unknown"))

    def test_tag_mode_accepts_equivalent_repetition_and_declared_final(self):
        valid = parse_measurement(r"reasoning \scalar{2.5} \distance_unit{meters}")
        self.assertEqual(valid.status, "tag_valid")
        self.assertEqual(str(valid.centimeters), "250.0")
        repeated = parse_measurement(
            r"\scalar{1.5} \distance_unit{feet} In conclusion, the final answer in the "
            r'specified format is: """\scalar{1.5} \distance_unit{feet}"""'
        )
        self.assertEqual(repeated.status, "tag_equivalent_repeated_valid")
        self.assertEqual(str(repeated.centimeters), "45.720")
        repeated_unit = parse_measurement(
            r"1.5 \distance_unit{feet}. In conclusion, the final answer in the specified "
            r'format is: """\scalar{1.5} \distance_unit{feet}"""'
        )
        self.assertEqual(repeated_unit.status, "tag_equivalent_repeated_valid")
        unit_only = parse_measurement(r'"""85 \distance_unit{cm}"""')
        self.assertEqual(unit_only.status, "tag_unit_only_valid")
        self.assertEqual(str(unit_only.centimeters), "85")
        declared_final = parse_measurement(
            r"\scalar{10} \distance_unit{cm} - \scalar{1.5} \distance_unit{cm} = "
            r"\scalar{8.5} \distance_unit{cm}. In conclusion, the final answer in the "
            r'specified format is: """\scalar{8.5} \distance_unit{cm}"""'
        )
        self.assertEqual(declared_final.status, "tag_final_declared_valid")
        self.assertEqual(str(declared_final.centimeters), "8.5")

    def test_conflicting_or_malformed_tags_remain_invalid(self):
        cases = (
            r"\scalar{1} final answer: 1 meter",
            r"\scalar{1 2} \distance_unit{cm}",
            r"\scalar{1e2} \distance_unit{cm}",
            r"\scalar{1} \distance_unit{yard}",
            r"\scalar{1} \distance_unit{cm} \scalar{2} \distance_unit{cm}",
            r"\scalar{1} \distance_unit{inch} to \scalar{2} \distance_unit{inch}",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(parse_measurement(text).centimeters)

    def test_controlled_real_output_wrappers_are_parsed(self):
        cases = {
            r"\(\boxed{57 \text{ inches}}\)": "144.78",
            r"```distance{15 \text{ cm}}```": "15",
            r"The height is \(\boxed{20}\) \(\text{cm}\).": "20",
            r'"""0.46{m}"""': "46.00",
            r'"""0.26, meter"""': "26.00",
            r'"""0.465\meter"""': "46.500",
            r'"""0.68m"""': "68.00",
            r'"""1/8 inch"""': "0.31750",
            r"\diameter{19} \units{cm}": "19",
        }
        for text, centimeters in cases.items():
            with self.subTest(text=text):
                self.assertEqual(str(parse_measurement(text).centimeters), centimeters)

    def test_prompt_placeholder_does_not_poison_later_explicit_answer(self):
        parsed = parse_measurement(
            r'The requested format is "\scalar{scalar} \distance_unit{distance unit}". '
            r'Final answer: """10 cm"""'
        )
        self.assertEqual(str(parsed.centimeters), "10")

    def test_fallback_uses_final_region_and_rejects_ranges_or_conflicts(self):
        self.assertEqual(str(parse_measurement("work 8 cm\nFinal answer: 2 feet").centimeters), "60.96")
        self.assertEqual(str(parse_measurement("Answer: .5 m").centimeters), "50.0")
        self.assertIn("multiple_numbers", parse_measurement("Answer: 2 to 3 meters").status)
        self.assertIn("conflicting_pairs", parse_measurement("Answer: 2 m or 3 ft").status)
        self.assertIn("invalid_scalar", parse_measurement("Answer: 1e2 cm").status)
        self.assertIn("unknown_unit", parse_measurement("Answer: 2 yards").status)
        self.assertEqual(parse_measurement("thinking 2 cm\n3 mm").status, "fallback_last_nonempty_line_valid")
        self.assertEqual(
            str(
                parse_measurement(
                    "The minimum distance between the plant spray and the PS4 is 10 inches."
                ).centimeters
            ),
            "25.40",
        )
        self.assertEqual(
            str(
                parse_measurement(
                    "The distance of Region [0] from Region [1] is 4.15 inches."
                ).centimeters
            ),
            "10.5410",
        )

    def test_units_and_legacy_notebook_behavior(self):
        expected = {"1 m": "100", "1 mm": "0.1", "1 ft": "30.48", "1 inch": "2.54"}
        for text, centimeters in expected.items():
            self.assertEqual(str(parse_measurement(text).centimeters), centimeters)
        legacy = parse_legacy_notebook(r"\scalar{2 and 4} \distance_unit{unknown}")
        self.assertEqual(str(legacy.centimeters), "3")
        self.assertEqual(legacy.status, "audit_unknown_unit_as_centimeter")

    def test_zero_is_preserved_but_remains_invalid_for_scoring(self):
        parsed = parse_measurement(r"\scalar{0} \distance_unit{cm}")
        self.assertEqual(parsed.status, "tag_non_positive_scalar")
        self.assertEqual(str(parsed.value), "0")
        self.assertEqual(parsed.unit, "cm")
        self.assertEqual(str(parsed.centimeters), "0")
        repeated = parse_measurement(
            r"\scalar{0} \distance_unit{cm} In conclusion, the final answer in the specified "
            r'format is: """\scalar{0} \distance_unit{cm}"""'
        )
        self.assertEqual(repeated.status, "tag_non_positive_scalar")
        self.assertEqual(str(repeated.centimeters), "0")
        negative = parse_measurement(r"\scalar{-1} \distance_unit{cm}")
        self.assertEqual(negative.status, "tag_non_positive_scalar")
        self.assertEqual(str(negative.value), "-1")
        self.assertEqual(str(negative.centimeters), "-1")

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
                "scorer_protocol": LEGACY_SCORER_PROTOCOL_V1,
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
            self.assertEqual(
                summary["inference"]["declared_scorer_protocol"],
                LEGACY_SCORER_PROTOCOL_V1,
            )
            self.assertTrue((score_dir / "prediction_validation.json").is_file())
            self.assertTrue((score_dir / "scored_rows.jsonl").is_file())
            gates = json.loads((score_dir / "publication_gates.json").read_text())
            self.assertTrue(gates["passed"])


if __name__ == "__main__":
    unittest.main()
