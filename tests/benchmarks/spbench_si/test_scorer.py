from __future__ import annotations

import unittest
from decimal import Decimal

from spatial_vlm_eval.benchmarks.spbench_si.scorer import (
    LEGACY_SCORER_PROTOCOL_V1,
    MRA_THRESHOLDS,
    SCORER_PROTOCOL,
    aggregate_rows,
    inference_metadata_scorer_protocol_is_compatible,
    mean_relative_accuracy_strict,
    mean_relative_accuracy_upstream,
    parse_choice,
    parse_numeric,
    score_main_row,
    score_upstream_row,
)


class SPBenchSIScorerTest(unittest.TestCase):
    def test_choice_parser_prefers_unique_answer_region_and_fails_conflicts(self):
        self.assertEqual(parse_choice("analysis A maybe\n<answer>B</answer>", ("A", "B")).value, "B")
        self.assertEqual(parse_choice("Final answer: C", ("A", "B", "C", "D")).value, "C")
        self.assertEqual(parse_choice("A or B", ("A", "B")).status, "conflict")
        self.assertEqual(parse_choice("<answer>A</answer><answer>A</answer>").status, "conflict")

    def test_numeric_parser_words_repeats_conflicts_ranges_and_nonfinite(self):
        self.assertEqual(parse_numeric("Final answer: forty").value, "40")
        self.assertEqual(parse_numeric("<answer>2 meters; 2 m</answer>").value, "2")
        self.assertEqual(parse_numeric("1 or 2").status, "conflict")
        self.assertEqual(parse_numeric("1 and 2 meters").status, "conflict")
        self.assertEqual(parse_numeric("1-2 meters").status, "range")
        self.assertEqual(parse_numeric("NaN").status, "nonfinite")
        self.assertEqual(parse_numeric("-1").status, "negative")

    def test_numeric_parser_does_not_treat_articles_or_choice_a_as_one(self):
        refusal = "I can’t determine that from the image alone without a known scale."
        self.assertEqual(parse_numeric(refusal).status, "no_number")
        self.assertEqual(parse_numeric("The object is a meter away.").status, "no_number")
        self.assertEqual(parse_numeric("<answer>A. No distance</answer>").status, "no_number")

    def test_numeric_parser_strips_standard_choice_labels_only_in_strong_regions(self):
        self.assertEqual(parse_numeric("<answer>A. 20</answer>").value, "20")
        self.assertEqual(parse_numeric("Final answer: C.1").value, "1")
        self.assertEqual(parse_numeric("<answer>M.1</answer>").status, "no_number")
        self.assertEqual(parse_numeric("A. 20").status, "conflict")
        self.assertEqual(parse_numeric("<answer>A. More than 90 meters</answer>").status, "bound")

    def test_numeric_parser_uses_controlled_declared_final_regions(self):
        spatialbot = (
            "The monitor is 40 inches and the door is 36 inches wide, so 40 + 36 = 76 inches. "
            "Converting inches to meters, the distance is approximately 1.9 meters."
        )
        equation = (
            "The table is 1 meter long and the mirror is 0.5 meters wide, so the distance "
            "between them is 1 + 0.5 = 1.5 meters."
        )
        longest = (
            "Width: 0.95 m; height: 0.75 m; depth: 0.46 m.\n"
            "The longest dimension of the table is 0.46 meters.<eos>"
        )
        provide = "Width: 3.60 m; height: 0.12 m. Provide 3.60 m as the longest dimension."
        self.assertEqual(parse_numeric(spatialbot, expected_unit="meter").value, "1.9")
        self.assertEqual(parse_numeric(equation, expected_unit="meter").value, "1.5")
        self.assertEqual(parse_numeric(longest, expected_unit="centimeter").value, "0.46")
        self.assertEqual(parse_numeric(provide, expected_unit="centimeter").value, "3.6")

    def test_numeric_parser_prefers_explicit_expected_unit_without_converting(self):
        paired = parse_numeric("Width: 0.42 m (42 cm)<eos>", expected_unit="centimeter")
        self.assertEqual(paired.value, "42")
        self.assertTrue(paired.evidence["unit_filter_applied"])
        self.assertEqual(
            parse_numeric("Answer: 1.44 meters; 42 centimeters.<eos>", expected_unit="centimeter").value,
            "42",
        )
        self.assertEqual(parse_numeric("0.42 meters", expected_unit="centimeter").value, "0.42")
        self.assertEqual(parse_numeric("0.42 meters; 42 centimeters").status, "conflict")

    def test_score_row_supplies_the_question_type_expected_unit(self):
        scoring = {
            "index": 0,
            "official_id": 1,
            "question_type": "object_size_estimation",
            "ground_truth": "42",
            "options": [],
        }
        scored = score_main_row(
            {"index": 0, "raw_prediction": "Width: 0.42 m (42 cm)<eos>"}, scoring
        )
        self.assertEqual(scored["parsed_answer"], "42")
        self.assertEqual(scored["score"], 1.0)

    def test_v1_inference_metadata_remains_compatible_with_parser_only_v2(self):
        self.assertTrue(inference_metadata_scorer_protocol_is_compatible(LEGACY_SCORER_PROTOCOL_V1))
        self.assertTrue(inference_metadata_scorer_protocol_is_compatible(SCORER_PROTOCOL))
        self.assertFalse(inference_metadata_scorer_protocol_is_compatible("unknown"))

    def test_original_mra_has_ten_strict_thresholds_and_audit_is_inclusive(self):
        self.assertEqual(MRA_THRESHOLDS, tuple(Decimal(value) / 100 for value in range(50, 100, 5)))
        self.assertEqual(mean_relative_accuracy_strict(Decimal("150"), Decimal("100")), Decimal("0"))
        self.assertEqual(mean_relative_accuracy_upstream(150.0, 100.0), 0.1)
        self.assertEqual(mean_relative_accuracy_strict(Decimal("105"), Decimal("100")), Decimal("0.9"))
        self.assertEqual(mean_relative_accuracy_upstream(105.0, 100.0), 1.0)

    def test_main_and_upstream_extraction_differ_without_mixing_rows(self):
        prediction = {"index": 0, "raw_prediction": "Reasoning 999.\nFinal answer: 2"}
        scoring = {"index": 0, "official_id": 1, "question_type": "object_abs_distance", "ground_truth": "2", "options": []}
        main = score_main_row(prediction, scoring)
        audit = score_upstream_row(prediction, scoring)
        self.assertEqual(main["score"], 1.0)
        self.assertEqual(audit["score"], 0.0)
        self.assertIn("parse_evidence", main)
        self.assertNotIn("parse_evidence", audit)

    def test_upstream_choice_audit_keeps_exact_current_code_semantics(self):
        scoring = {
            "index": 0, "official_id": 1, "question_type": "object_rel_distance",
            "ground_truth": "A", "options": ["A. chair", "B. table"],
        }
        self.assertEqual(
            score_upstream_row({"index": 0, "raw_prediction": "A."}, scoring)["score"], 1.0
        )
        self.assertEqual(
            score_upstream_row({"index": 0, "raw_prediction": "The answer is A"}, scoring)["score"],
            0.0,
        )

    def test_upstream_numeric_audit_keeps_a_as_one_while_main_rejects_it(self):
        scoring = {
            "index": 0,
            "official_id": 1,
            "question_type": "object_abs_distance",
            "ground_truth": "1",
            "options": [],
        }
        prediction = {
            "index": 0,
            "raw_prediction": "I cannot determine that without a known scale.",
        }
        self.assertEqual(score_main_row(prediction, scoring)["parse_status"], "no_number")
        audit = score_upstream_row(prediction, scoring)
        self.assertEqual(audit["upstream_extracted_answer"], "1")
        self.assertEqual(audit["score"], 1.0)

    def test_four_task_macro_is_not_micro(self):
        rows = []
        for task, scores in {
            "object_abs_distance": [1.0, 0.0],
            "object_size_estimation": [1.0],
            "object_rel_distance": [0.0],
            "object_rel_direction": [1.0, 1.0, 1.0, 1.0],
        }.items():
            rows.extend({"question_type": task, "score": value} for value in scores)
        aggregate = aggregate_rows(rows)
        self.assertEqual(aggregate["metrics"]["nq_macro"], 0.75)
        self.assertEqual(aggregate["metrics"]["mcq_macro"], 0.5)
        self.assertEqual(aggregate["metrics"]["overall_four_task_macro"], 0.625)
        self.assertNotEqual(aggregate["metrics"]["micro_audit"], 0.625)


if __name__ == "__main__":
    unittest.main()
