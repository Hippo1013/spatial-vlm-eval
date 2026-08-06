from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.cv_bench.data import OFFICIAL_TEST_SIZE
from spatial_vlm_eval.benchmarks.cv_bench.scorer import (
    LEGACY_SCORER_PROTOCOL_V2,
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
            "B<eos>": ("B", "terminal_token_stripped_explicit_letter"),
            "A\n\nThe object is on the left.": ("A", "first_line_explicit_letter"),
            "The object is on the right.\n\nB": ("B", "last_line_explicit_letter"),
            "B.right": ("B", "letter_and_option_text"),
            "A (left)": ("A", "letter_and_option_text"),
            "right<|im_end|>": ("B", "terminal_token_stripped_option_text"),
            (
                "The platforms exceed all options and the answer is zero.\n\n(C)"
            ): ("C", "last_line_explicit_letter"),
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
            "(A) B<eos>": "terminal_token_stripped_multiple_answers",
            "(A) right<eos>": "terminal_token_stripped_conflict",
            "M. 0": "out_of_range",
        }
        for text, status in rejected.items():
            with self.subTest(text=text):
                self.assertEqual(parse_answer(text, choices).status, status)
        compact = parse_answer("B.1", ["0", "1"])
        self.assertEqual((compact.answer, compact.status), ("B", "letter_and_option_text"))
        self.assertEqual(parse_answer("B.0", ["0", "1"]).status, "conflict")

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
                raw_prediction = "A<eos>" if index == 0 else "A"
                handle.write(
                    json.dumps({"index": index, "raw_prediction": raw_prediction}) + "\n"
                )
        output = self.root / "scores" / SCORER_PROTOCOL
        summary = score_predictions(
            predictions,
            self.contract,
            output_dir=output,
            require_metadata=False,
        )
        self.assertEqual(summary["num_scored_rows"], OFFICIAL_TEST_SIZE)
        self.assertTrue((output / "scored_rows.jsonl").is_file())
        first_scored = json.loads(
            (output / "scored_rows.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertEqual(first_scored["raw_prediction"], "A<eos>")
        self.assertEqual(first_scored["parsed_answer"], "A")
        self.assertEqual(
            first_scored["parse_status"], "terminal_token_stripped_explicit_letter"
        )
        self.assertEqual(
            first_scored["parse_evidence"][0], "stripped_terminal_token:<eos>"
        )
        gates = json.loads((output / "publication_gates.json").read_text(encoding="utf-8"))
        self.assertTrue(gates["passed"])
        self.assertTrue(all(gates["gates"].values()))

    def test_v3_scorer_accepts_locked_v2_inference_metadata_without_rewriting_it(self):
        predictions = self.root / "predictions.jsonl"
        with predictions.open("w", encoding="utf-8") as handle:
            for index in range(OFFICIAL_TEST_SIZE):
                handle.write(json.dumps({"index": index, "raw_prediction": "A"}) + "\n")
        metadata = {
            "output": str(predictions.resolve()),
            "output_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "scorer_protocol": LEGACY_SCORER_PROTOCOL_V2,
            "inference_protocol": "test-inference-v1",
            "publishable_inference": True,
            "dataset": {
                "revision": "bc284db50d036958861cb60cdd7b77612052ce0d",
                "fingerprint": self.contract.dataset_fingerprint,
                "official_test_size": OFFICIAL_TEST_SIZE,
            },
            "model": {
                "profile": "test-profile",
                "model": "test-model",
                "model_revision": "test-revision",
                "input_profile": "rgb",
                "inference_protocol": "test-inference-v1",
                "backend": "test",
                "decoding": {},
            },
        }
        predictions.with_suffix(".jsonl.metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        summary = score_predictions(predictions, self.contract)
        self.assertEqual(summary["scorer_protocol"], SCORER_PROTOCOL)
        self.assertEqual(
            summary["inference"]["declared_scorer_protocol"],
            LEGACY_SCORER_PROTOCOL_V2,
        )
        metadata["scorer_protocol"] = "unknown-cv-bench-scorer"
        predictions.with_suffix(".jsonl.metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "full prediction validation failed"):
            score_predictions(
                predictions,
                self.contract,
                output_dir=self.root / "unknown-scorer-output",
            )


if __name__ == "__main__":
    unittest.main()
