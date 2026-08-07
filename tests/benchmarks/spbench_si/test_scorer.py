from __future__ import annotations

import unittest
from decimal import Decimal

from spatial_vlm_eval.benchmarks.spbench_si.scorer import (
    MRA_THRESHOLDS,
    aggregate_rows,
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
        self.assertEqual(parse_numeric("1-2 meters").status, "range")
        self.assertEqual(parse_numeric("NaN").status, "nonfinite")
        self.assertEqual(parse_numeric("-1").status, "negative")

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
