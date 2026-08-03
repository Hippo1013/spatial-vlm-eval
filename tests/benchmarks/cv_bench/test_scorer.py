from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.cv_bench.data import OFFICIAL_TEST_SIZE
from spatial_vlm_eval.benchmarks.cv_bench.scorer import (
    SCORER_PROTOCOL,
    parse_answer,
    score_predictions,
    score_rows,
)

from tests.benchmarks.cv_bench.helpers import official_fake_contract


class CVBenchScorerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.contract = official_fake_contract(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_robust_parser_letters_text_conflicts_and_invalid_outputs(self):
        choices = ["left", "right", "above", "below", "near", "far"]
        accepted = {
            "A": ("A", "explicit_letter"),
            "(F)": ("F", "explicit_letter"),
            "Answer is: C": ("C", "explicit_letter"),
            "right": ("B", "option_text"),
            "(A) left": ("A", "letter_and_option_text"),
            "I considered A and C. <answer>B</answer>": (
                "B",
                "answer_tag_explicit_letter",
            ),
        }
        for text, expected in accepted.items():
            with self.subTest(text=text):
                parsed = parse_answer(text, choices)
                self.assertEqual((parsed.answer, parsed.status), expected)
        rejected = {
            "(A) right": "conflict",
            "(A) or (B)": "multiple_answers",
            "(G)": "out_of_range",
            "": "empty",
            "I am unsure": "unparsed",
            "<answer>A</answer><answer>B</answer>": "multiple_answer_tags",
            "<answer></answer>": "empty_answer_tag",
        }
        for text, status in rejected.items():
            with self.subTest(text=text):
                self.assertEqual(parse_answer(text, choices).status, status)

    def test_official_equal_weight_formula_is_not_micro_accuracy(self):
        predictions = []
        for index in range(OFFICIAL_TEST_SIZE):
            row = self.contract.scoring_row(index)
            if row["source"] == "ADE20K":
                answer = "A"
            elif row["source"] == "COCO":
                answer = "B"
            else:
                answer = "A" if index % 2 == 0 else "B"
            predictions.append({"index": index, "raw_prediction": answer})
        _, aggregate = score_rows(predictions, self.contract)
        self.assertEqual(aggregate["source_metrics"]["ADE20K"]["accuracy"], 1.0)
        self.assertEqual(aggregate["source_metrics"]["COCO"]["accuracy"], 0.0)
        self.assertEqual(aggregate["source_metrics"]["Omni3D"]["accuracy"], 0.5)
        self.assertEqual(aggregate["metrics"]["accuracy_2d"], 0.5)
        self.assertEqual(aggregate["metrics"]["accuracy_3d"], 0.5)
        self.assertEqual(aggregate["metrics"]["overall_accuracy"], 0.5)
        self.assertNotEqual(aggregate["metrics"]["micro_accuracy_audit_only"], 0.5)

    def test_scorer_writes_atomic_canonical_artifacts_and_gates(self):
        predictions = self.root / "predictions.jsonl"
        with predictions.open("w", encoding="utf-8") as handle:
            for index in range(OFFICIAL_TEST_SIZE):
                handle.write(json.dumps({"index": index, "raw_prediction": "A"}) + "\n")
        output = self.root / "scores" / SCORER_PROTOCOL
        summary = score_predictions(
            predictions,
            self.contract,
            output_dir=output,
            require_metadata=False,
        )
        self.assertEqual(summary["num_scored_rows"], OFFICIAL_TEST_SIZE)
        self.assertTrue((output / "scored_rows.jsonl").is_file())
        gates = json.loads((output / "publication_gates.json").read_text(encoding="utf-8"))
        self.assertTrue(gates["passed"])
        self.assertTrue(all(gates["gates"].values()))


if __name__ == "__main__":
    unittest.main()
