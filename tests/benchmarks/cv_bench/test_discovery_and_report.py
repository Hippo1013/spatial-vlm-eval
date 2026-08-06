from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.cv_bench.data import (
    DATASET_REVISION,
    EXPECTED_SOURCE_COUNTS,
    EXPECTED_TASK_COUNTS,
    OFFICIAL_TEST_SIZE,
)
from spatial_vlm_eval.benchmarks.cv_bench.profiles import PROFILE_SEQUENCE, PROFILES
from spatial_vlm_eval.benchmarks.cv_bench.report import discover_results, render_markdown
from spatial_vlm_eval.benchmarks.cv_bench.score_results import discover_candidates
from spatial_vlm_eval.benchmarks.cv_bench.scorer import (
    LEGACY_SCORER_PROTOCOL_V2,
    RESULT_KIND,
    SCORER_PROTOCOL,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_summary(
    root: Path,
    profile_key: str,
    *,
    run_name: str | None = None,
    accuracy=1.0,
    metadata_scorer_protocol: str = SCORER_PROTOCOL,
):
    profile = PROFILES[profile_key]
    run = root / (run_name or profile_key)
    predictions = run / "predictions.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("", encoding="utf-8")
    metadata = {
        "output": str(predictions.resolve()),
        "output_sha256": _sha256(predictions),
        "scorer_protocol": metadata_scorer_protocol,
        "publishable_inference": True,
        "dataset": {
            "revision": DATASET_REVISION,
            "fingerprint": "test-dataset-fingerprint",
            "official_test_size": OFFICIAL_TEST_SIZE,
        },
        "model": {
            "profile": profile.key,
            "model_revision": profile.revision,
            "input_profile": profile.input_profile,
            "inference_protocol": profile.inference_protocol,
            "decoding": {**profile.decoding, "stream": False},
        },
    }
    predictions.with_suffix(".jsonl.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    score_dir = run / "scores" / SCORER_PROTOCOL
    score_dir.mkdir(parents=True, exist_ok=True)
    scored_rows = score_dir / "scored_rows.jsonl"
    rows = []
    for source, task, count in (
        ("ADE20K", "Count", 342),
        ("ADE20K", "Relation", 291),
        ("COCO", "Count", 446),
        ("COCO", "Relation", 359),
        ("Omni3D", "Depth", 600),
        ("Omni3D", "Distance", 600),
    ):
        for _ in range(count):
            rows.append(
                {
                    "index": len(rows),
                    "source": source,
                    "task": task,
                    "correct": bool(accuracy),
                }
            )
    scored_rows.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "result_kind": RESULT_KIND,
        "scorer_protocol": SCORER_PROTOCOL,
        "dataset": {
            "revision": DATASET_REVISION,
            "fingerprint": "test-dataset-fingerprint",
            "official_test_size": OFFICIAL_TEST_SIZE,
        },
        "inference": {
            "profile": profile.key,
            "model_revision": profile.revision,
            "input_profile": profile.input_profile,
            "inference_protocol": profile.inference_protocol,
            "declared_scorer_protocol": metadata_scorer_protocol,
            "decoding": {**profile.decoding, "stream": False},
        },
        "num_scored_rows": OFFICIAL_TEST_SIZE,
        "task_metrics": {
            "spatial_relationship": {
                "accuracy": accuracy,
                "correct": int(EXPECTED_TASK_COUNTS["Relation"] * accuracy),
                "total": EXPECTED_TASK_COUNTS["Relation"],
            },
            "object_count": {
                "accuracy": accuracy,
                "correct": int(EXPECTED_TASK_COUNTS["Count"] * accuracy),
                "total": EXPECTED_TASK_COUNTS["Count"],
            },
            "depth_order": {
                "accuracy": accuracy,
                "correct": int(EXPECTED_TASK_COUNTS["Depth"] * accuracy),
                "total": EXPECTED_TASK_COUNTS["Depth"],
            },
            "relative_distance": {
                "accuracy": accuracy,
                "correct": int(EXPECTED_TASK_COUNTS["Distance"] * accuracy),
                "total": EXPECTED_TASK_COUNTS["Distance"],
            },
        },
        "source_metrics": {
            key: {
                "accuracy": accuracy,
                "correct": int(total * accuracy),
                "total": total,
            }
            for key, total in EXPECTED_SOURCE_COUNTS.items()
        },
        "metrics": {
            "accuracy_2d": accuracy,
            "accuracy_3d": accuracy,
            "overall_accuracy": accuracy,
        },
        "artifacts": {
            "predictions": str(predictions.resolve()),
            "predictions_sha256": _sha256(predictions),
            "scored_rows": str(scored_rows.resolve()),
            "scored_rows_sha256": _sha256(scored_rows),
        },
    }
    summary_path = score_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    gates = {
        "passed": True,
        "scorer_protocol": SCORER_PROTOCOL,
        "summary": str(summary_path.resolve()),
        "gates": {"full": True},
    }
    (score_dir / "publication_gates.json").write_text(json.dumps(gates), encoding="utf-8")
    return score_dir / "summary.json"


class CVBenchDiscoveryAndReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_directory_scoring_discovery_excludes_test_and_shard_predictions(self):
        write_summary(self.root, "qwen3_vl_2b")
        for relative in ["test_runs/a/predictions.jsonl", "shards/worker-0/predictions.jsonl"]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        candidates = discover_candidates(self.root)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].state, "complete")

    def test_report_discovery_validates_gates_and_registry_provenance(self):
        write_summary(self.root, "qwen3_vl_2b", accuracy=0.0)
        write_summary(self.root, "qwen3_vl_4b", accuracy=1.0)
        results = discover_results(self.root)
        self.assertEqual([result.profile for result in results], ["qwen3_vl_2b", "qwen3_vl_4b"])
        markdown = render_markdown(results, generated_at="2026-08-03T00:00:00+00:00")
        self.assertIn("未完整（2/23）", markdown)
        self.assertIn("Qwen3-VL-4B（RGB）", markdown)
        self.assertIn("**100.00**", markdown)
        self.assertIn("缺失 profile", markdown)

    def test_report_accepts_explicit_v2_inference_metadata_compatibility(self):
        write_summary(
            self.root,
            "qwen3_vl_2b",
            metadata_scorer_protocol=LEGACY_SCORER_PROTOCOL_V2,
        )
        results = discover_results(self.root)
        self.assertEqual([result.profile for result in results], ["qwen3_vl_2b"])

    def test_duplicate_publishable_profile_fails_closed(self):
        write_summary(self.root, "qwen3_vl_2b", run_name="one")
        write_summary(self.root, "qwen3_vl_2b", run_name="two")
        with self.assertRaisesRegex(ValueError, "Multiple publishable"):
            discover_results(self.root)

    def test_post_score_artifact_or_formula_tampering_fails_closed(self):
        summary_path = write_summary(self.root, "qwen3_vl_2b")
        predictions = summary_path.parents[2] / "predictions.jsonl"
        predictions.write_text('{"index":0,"raw_prediction":"A"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prediction artifact hash mismatch"):
            discover_results(self.root)

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        summary_path = write_summary(self.root, "qwen3_vl_2b")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["metrics"]["overall_accuracy"] = 0.25
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Overall aggregation mismatch"):
            discover_results(self.root)

    def test_target_complete_label_requires_all_23_profiles(self):
        for key in PROFILE_SEQUENCE:
            write_summary(self.root, key)
        markdown = render_markdown(discover_results(self.root))
        self.assertIn("完整（23/23）", markdown)
        self.assertIn("缺失 profile：无", markdown)


if __name__ == "__main__":
    unittest.main()
