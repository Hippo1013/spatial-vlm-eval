from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.q_spatial.data import (
    DATASET_REVISION,
    EXPECTED_SPLIT_TYPE_COUNTS,
    OFFICIAL_TEST_SIZE,
    STANDARD_SYSTEM_PROMPT_SHA256,
)
from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILE_SEQUENCE, PROFILES
from spatial_vlm_eval.benchmarks.q_spatial.report import discover_results, render_markdown
from spatial_vlm_eval.benchmarks.q_spatial.score_results import discover_candidates
from spatial_vlm_eval.benchmarks.q_spatial.scorer import RESULT_KIND, SCORER_PROTOCOL


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(correct, total):
    return {"correct": correct, "total": total, "accuracy": correct / total}


def write_result(root: Path, profile_key: str, *, run_name=None, correct=True):
    profile = PROFILES[profile_key]
    run = root / (run_name or profile_key)
    predictions = run / "predictions.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(
        "".join(json.dumps({"index": index, "raw_prediction": "1 cm"}) + "\n" for index in range(271)),
        encoding="utf-8",
    )
    metadata = {
        "output": str(predictions.resolve()),
        "output_sha256": _sha256(predictions),
        "inference_protocol": profile.inference_protocol,
        "scorer_protocol": SCORER_PROTOCOL,
        "publishable_inference": True,
        "dataset": {
            "revision": DATASET_REVISION,
            "fingerprint": "test-q-spatial-fingerprint",
            "official_test_size": OFFICIAL_TEST_SIZE,
        },
        "prompt": {"system_prompt_sha256": STANDARD_SYSTEM_PROMPT_SHA256},
        "model": {
            "profile": profile.key,
            "model": profile.model,
            "model_revision": profile.revision,
            "input_profile": profile.input_profile,
            "comparison_group": profile.comparison_group,
            "inference_protocol": profile.inference_protocol,
            "decoding": profile.decoding,
            "seed_strategy": profile.seed_strategy,
            "system_role_supported": profile.system_role_supported,
        },
    }
    predictions.with_suffix(".jsonl.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    score_dir = run / "scores" / SCORER_PROTOCOL
    score_dir.mkdir(parents=True)
    scored_rows = score_dir / "scored_rows.jsonl"
    rows = []
    for canonical_type, count in EXPECTED_SPLIT_TYPE_COUNTS["QSpatial_scannet"].items():
        for _ in range(count):
            rows.append({
                "index": len(rows), "split": "QSpatial_scannet", "canonical_type": canonical_type,
                "success_delta_le_1_25": correct, "success_delta_le_2": correct,
                "legacy_success_delta_lt_1_25": correct, "legacy_success_delta_lt_2": correct,
            })
    for raw_type, count in EXPECTED_SPLIT_TYPE_COUNTS["QSpatial_plus"].items():
        for _ in range(count):
            rows.append({
                "index": len(rows), "split": "QSpatial_plus",
                "canonical_type": "object_width" if raw_type == "1d_horizontal" else raw_type,
                "success_delta_le_1_25": correct, "success_delta_le_2": correct,
                "legacy_success_delta_lt_1_25": correct, "legacy_success_delta_lt_2": correct,
            })
    scored_rows.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    value = int(correct)
    split_metrics = {
        split: {
            "delta_le_1_25": _metric(value * count, count),
            "delta_le_2": _metric(value * count, count),
            "legacy_delta_lt_1_25": _metric(value * count, count),
            "legacy_delta_lt_2": _metric(value * count, count),
        }
        for split, count in (("QSpatial_scannet", 170), ("QSpatial_plus", 101))
    }
    type_metrics = {
        key: {
            "delta_le_1_25": _metric(value * count, count),
            "delta_le_2": _metric(value * count, count),
        }
        for key, count in EXPECTED_SPLIT_TYPE_COUNTS["QSpatial_scannet"].items()
    }
    summary = {
        "result_kind": RESULT_KIND,
        "scorer_protocol": SCORER_PROTOCOL,
        "dataset": {
            "revision": DATASET_REVISION,
            "fingerprint": "test-q-spatial-fingerprint",
            "official_test_size": OFFICIAL_TEST_SIZE,
        },
        "inference": {
            "profile": profile.key,
            "model_revision": profile.revision,
            "input_profile": profile.input_profile,
            "comparison_group": profile.comparison_group,
            "inference_protocol": profile.inference_protocol,
            "decoding": profile.decoding,
            "seed_strategy": profile.seed_strategy,
        },
        "num_scored_rows": 271,
        "split_metrics": split_metrics,
        "scannet_type_metrics": type_metrics,
        "metrics": {
            "overall_delta_le_1_25": float(value),
            "overall_delta_le_2": float(value),
            "legacy_notebook_overall_delta_lt_2": float(value),
        },
        "num_main_vs_legacy_differences": 0,
        "artifacts": {
            "predictions": str(predictions.resolve()),
            "predictions_sha256": _sha256(predictions),
            "scored_rows": str(scored_rows.resolve()),
            "scored_rows_sha256": _sha256(scored_rows),
        },
    }
    summary_path = score_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (score_dir / "publication_gates.json").write_text(
        json.dumps({
            "passed": True,
            "scorer_protocol": SCORER_PROTOCOL,
            "summary": str(summary_path.resolve()),
            "gates": {"all": True},
        }),
        encoding="utf-8",
    )
    return summary_path


class QSpatialDiscoveryAndReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_directory_scoring_excludes_test_and_shards(self):
        write_result(self.root, "qwen3_vl_2b")
        for relative in ("test_runs/a/predictions.jsonl", "shards/worker-0/predictions.jsonl"):
            path = self.root / relative
            path.parent.mkdir(parents=True)
            path.write_text("", encoding="utf-8")
        candidates = discover_candidates(self.root)
        self.assertEqual([(item.state, item.predictions.name) for item in candidates], [("complete", "predictions.jsonl")])

    def test_report_is_gated_and_declares_both_completeness_counts(self):
        write_result(self.root, "qwen3_vl_2b", correct=False)
        write_result(self.root, "qwen3_vl_4b", correct=True)
        results = discover_results(self.root)
        markdown = render_markdown(results, generated_at="2026-08-04T00:00:00Z")
        self.assertIn("RGB 轨完整度：2/18", markdown)
        self.assertIn("全轨完整度：2/21", markdown)
        self.assertIn("Input track", markdown)
        self.assertIn("Comparison group", markdown)
        self.assertIn("**100.00**", markdown)
        self.assertIn("`internvl3_78b`", markdown)

    def test_duplicate_or_tampered_publishable_result_fails_closed(self):
        write_result(self.root, "qwen3_vl_2b", run_name="one")
        write_result(self.root, "qwen3_vl_2b", run_name="two")
        with self.assertRaisesRegex(ValueError, "Multiple publishable"):
            discover_results(self.root)
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        summary = write_result(self.root, "qwen3_vl_2b")
        predictions = summary.parents[2] / "predictions.jsonl"
        predictions.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prediction artifact hash mismatch"):
            discover_results(self.root)

    def test_all_registered_tracks_render_18_of_18_and_21_of_21(self):
        for key in PROFILE_SEQUENCE:
            write_result(self.root, key)
        markdown = render_markdown(discover_results(self.root))
        self.assertIn("RGB 轨完整度：18/18", markdown)
        self.assertIn("全轨完整度：21/21", markdown)
        self.assertIn("缺失 profile：无", markdown)


if __name__ == "__main__":
    unittest.main()
