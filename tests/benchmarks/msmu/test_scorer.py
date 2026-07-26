import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from spatial_vlm_eval.benchmarks.msmu import scorer
from spatial_vlm_eval.benchmarks.msmu.data import official_type_for_raw_type
from spatial_vlm_eval.benchmarks.msmu.scorer import (
    cache_entry_is_usable,
    judge_cache_candidate,
    official_quant_score,
    official_type,
    publication_gate_status,
)


def judge_row(raw_type: str = "zero") -> dict:
    return {
        "index": 0,
        "raw_type": raw_type,
        "task_family": "existence" if raw_type == "zero" else "scale_estimation",
        "question": "Is there a chair?" if raw_type == "zero" else "How wide is the table?",
        "reference": "Yes." if raw_type == "zero" else "1 meter.",
        "prediction": "Yes." if raw_type == "zero" else "1 meter.",
    }


class OfficialTypeRoutingTest(unittest.TestCase):
    def test_all_raw_types_use_the_canonical_mapping_chain(self):
        expected = {
            "width": "scale_estimation",
            "height": "scale_estimation",
            "size": "scale_estimation",
            "distance": "absolute_distance",
            "count": "count",
            "position": "grounding",
            "refer_two_objects": "refer_obj_estimation",
            "refer_three_objects": "refer_obj_estimation",
            "left/right": "relative_position",
            "taller_two_object": "scale_compare",
            "tall_three_objects": "scale_compare",
            "zero": "existence",
        }
        for raw_type, scorer_type in expected.items():
            with self.subTest(raw_type=raw_type):
                self.assertEqual(official_type_for_raw_type(raw_type), scorer_type)

    def test_untrusted_official_type_cannot_override_raw_type(self):
        row = {
            "raw_type": "width",
            "task_family": "scale_estimation",
            "official_type": "existence",
        }
        self.assertEqual(official_type(row), "scale_estimation")


class JudgeCacheReliabilityTest(unittest.TestCase):
    def test_request_failure_never_becomes_a_cache_row(self):
        cached_row, failure = judge_cache_candidate(
            judge_row(),
            {"__parse_error__": "timed out", "__raw_content__": None},
            base_url="http://127.0.0.1:18080/v1",
            model="msmu-judge",
        )
        self.assertIsNone(cached_row)
        self.assertIsNotNone(failure)
        self.assertIn("timed out", failure["error"])

    def test_schema_valid_response_can_be_cached_and_reused(self):
        row = judge_row()
        cached_row, failure = judge_cache_candidate(
            row,
            {"your_mark": 1},
            base_url="http://127.0.0.1:18080/v1",
            model="msmu-judge",
        )
        self.assertIsNone(failure)
        self.assertIsNotNone(cached_row)
        self.assertTrue(
            cache_entry_is_usable(
                row,
                cached_row,
                expected_cache_key=cached_row["cache_key"],
            )
        )

    def test_old_parse_error_cache_entry_is_retried(self):
        row = judge_row()
        entry = {
            "cache_key": "same-key",
            "judge": {"__parse_error__": "temporary 500"},
        }
        self.assertFalse(
            cache_entry_is_usable(row, entry, expected_cache_key="same-key")
        )

    def test_main_does_not_cache_failure_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                predictions="predictions.jsonl",
                output_dir=directory,
                dataset_root="MSMU",
                validation_report=None,
                base_url="http://127.0.0.1:18080/v1",
                model="msmu-judge",
                api_key="local",
                workers=1,
                retries=1,
            )
            with (
                patch.object(scorer, "parse_args", return_value=args),
                patch.object(
                    scorer,
                    "validated_predictions_for_scoring",
                    return_value=[judge_row()],
                ),
                patch.object(
                    scorer,
                    "call_chat",
                    return_value={"__parse_error__": "temporary timeout"},
                ),
                patch.object(
                    scorer,
                    "tqdm",
                    side_effect=lambda iterable, **_kwargs: iterable,
                ),
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(SystemExit, "Scoring blocked"):
                    scorer.main()

            output_dir = Path(directory)
            self.assertFalse((output_dir / "judge_cache.jsonl").exists())
            failures = [
                json.loads(line)
                for line in (output_dir / "judge_failures.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(failures), 1)
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary["publishable"])
            self.assertEqual(summary["num_judge_failures"], 1)
            self.assertNotIn("official_macro8_accuracy", summary)


class PublicationGateTest(unittest.TestCase):
    def test_unresolved_judge_failure_blocks_publication(self):
        publishable, checks, failures = publication_gate_status(
            validation_passed=True,
            num_samples=987,
            indices=list(range(987)),
            missing_official_types=[],
            judge_failure_count=1,
        )
        self.assertFalse(publishable)
        self.assertFalse(checks["judge_failures_zero"])
        self.assertEqual(failures, ["judge_failures_zero"])

    def test_complete_successful_run_is_publishable(self):
        publishable, checks, failures = publication_gate_status(
            validation_passed=True,
            num_samples=987,
            indices=list(range(987)),
            missing_official_types=[],
            judge_failure_count=0,
        )
        self.assertTrue(publishable)
        self.assertTrue(all(checks.values()))
        self.assertEqual(failures, [])


class OfficialQuantitativeLengthTest(unittest.TestCase):
    def score(self, response):
        return official_quant_score(
            "scale_estimation",
            {
                "answer_in_meters": [1.0, 1.0, 1.0],
                "response_in_meters": response,
            },
        )

    def test_scalar_response_fails_for_list_reference(self):
        score, details = self.score(1.0)
        self.assertEqual(score, 0.0)
        self.assertFalse(details["match_success"])

    def test_shorter_list_response_fails(self):
        score, details = self.score([1.0, 1.0])
        self.assertEqual(score, 0.0)
        self.assertFalse(details["match_success"])
        self.assertIn("shorter than the reference", details["error"])

    def test_equal_length_response_is_scored(self):
        score, details = self.score([1.0, 1.0, 1.0])
        self.assertEqual(score, 1.0)
        self.assertTrue(details["match_success"])

    def test_extra_trailing_values_are_ignored_like_official_scorer(self):
        score, details = self.score([1.0, 1.0, 1.0, 100.0])
        self.assertEqual(score, 1.0)
        self.assertTrue(details["match_success"])
        self.assertEqual(details["delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
