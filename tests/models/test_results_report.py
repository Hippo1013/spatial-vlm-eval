from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
HELPER_PATH = REPOSITORY / "scripts" / "msmu" / "_build_results_report.py"
SPEC = importlib.util.spec_from_file_location("msmu_results_report", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
results_report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = results_report
SPEC.loader.exec_module(results_report)


OFFICIAL_TYPES = [key for key, _header in results_report.OFFICIAL_METRICS]
PUBLICATION_GATES = (
    "prediction_validation_passed",
    "full_official_test_split",
    "all_official_types_present",
    "judge_failures_zero",
)
INDEX_ROWS = "".join(
    json.dumps({"index": index}) + "\n"
    for index in range(results_report.MSMU_OFFICIAL_TEST_SIZE)
)


def write_result(
    root: Path,
    *,
    run_name: str,
    profile: str,
    scorer_protocol: str,
    model: str,
    model_revision: str = "revision-sha",
    inference_protocol: str | None = None,
    input_profile: str = "question_only",
    accuracies: tuple[float, ...] | None = None,
) -> Path:
    resolved_inference_protocol = (
        inference_protocol or f"msmu_{profile}_question_only_v1"
    )
    prediction_dir = (
        root
        / run_name
        / model_revision
        / resolved_inference_protocol
        / scorer_protocol
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)
    predictions = prediction_dir / "predictions.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")

    metadata = {
        "schema_version": 1,
        "publishable_inference": True,
        "num_predictions": results_report.MSMU_OFFICIAL_TEST_SIZE,
        "inference_protocol": resolved_inference_protocol,
        "scorer_protocol": scorer_protocol,
        "model": {
            "model": model,
            "profile": profile,
            "model_revision": model_revision,
            "inference_protocol": resolved_inference_protocol,
            "input_profile": input_profile,
        },
    }
    (prediction_dir / "predictions.jsonl.metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    score_dir = prediction_dir / "scores" / scorer_protocol
    score_dir.mkdir(parents=True)
    values = accuracies or tuple(
        index / 10 for index in range(1, len(OFFICIAL_TYPES) + 1)
    )
    counts = {kind: 1 for kind in OFFICIAL_TYPES}
    counts[OFFICIAL_TYPES[-1]] += (
        results_report.MSMU_OFFICIAL_TEST_SIZE - len(OFFICIAL_TYPES)
    )
    summary = {
        "publishable": True,
        "status": "complete",
        "result_kind": results_report.EXPECTED_RESULT_KIND,
        "publication_gates": {name: True for name in PUBLICATION_GATES},
        "publication_gate_failures": [],
        "num_judge_failures": 0,
        "num_samples": results_report.MSMU_OFFICIAL_TEST_SIZE,
        "micro_accuracy": 0.5,
        "official_macro8_accuracy": sum(values) / len(values),
        "missing_official_types": [],
        "protocol": scorer_protocol,
        "official_types": {
            kind: {"count": counts[kind], "accuracy": accuracy}
            for kind, accuracy in zip(OFFICIAL_TYPES, values, strict=True)
        },
    }
    validation = {
        "passed": True,
        "allow_subset": False,
        "official_test_size": results_report.MSMU_OFFICIAL_TEST_SIZE,
        "num_prediction_rows": results_report.MSMU_OFFICIAL_TEST_SIZE,
        "num_unique_indices": results_report.MSMU_OFFICIAL_TEST_SIZE,
        "errors": [],
    }
    (score_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (score_dir / "prediction_validation.json").write_text(
        json.dumps(validation),
        encoding="utf-8",
    )
    (score_dir / "scored_rows.jsonl").write_text(INDEX_ROWS, encoding="utf-8")
    (score_dir / "judge_cache.jsonl").write_text(INDEX_ROWS, encoding="utf-8")
    (score_dir / "judge_failures.jsonl").write_text("", encoding="utf-8")
    (score_dir / "score.log").write_text("complete\n", encoding="utf-8")
    return score_dir / "summary.json"


class ResultsReportDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "results root with spaces"
        self.root.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_discovers_complete_results_across_scorer_protocols(self):
        write_result(
            self.root,
            run_name="current",
            profile="shared_profile",
            scorer_protocol="current-protocol",
            model="org/Model-A",
        )
        write_result(
            self.root,
            run_name="historical",
            profile="shared_profile",
            scorer_protocol="historical-protocol",
            model="org/Model-A",
        )

        results = results_report.discover_results(self.root)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.eligible for result in results))
        self.assertEqual(
            {result.scorer_protocol for result in results},
            {"current-protocol", "historical-protocol"},
        )

    def test_incomplete_canonical_artifact_is_listed_but_not_reportable(self):
        summary = write_result(
            self.root,
            run_name="broken",
            profile="broken_profile",
            scorer_protocol="protocol",
            model="org/Broken",
        )
        (summary.parent / "score.log").unlink()

        result = results_report.discover_results(self.root)[0]

        self.assertFalse(result.eligible)
        self.assertIn("missing canonical artifact score.log", result.reason)
        with self.assertRaisesRegex(
            results_report.ConfigurationError,
            "no matching score result is reportable",
        ):
            results_report.selected_results(
                [result],
                profiles=[],
                scorer_protocols=[],
            )

    def test_valid_result_is_reportable_when_sibling_summary_is_incomplete(self):
        valid = write_result(
            self.root,
            run_name="valid",
            profile="shared_profile",
            scorer_protocol="current",
            model="org/Shared",
        )
        invalid = write_result(
            self.root,
            run_name="invalid",
            profile="shared_profile",
            scorer_protocol="historical",
            model="org/Shared",
        )
        (invalid.parent / "score.log").unlink()
        results = results_report.discover_results(self.root)

        selected = results_report.selected_results(
            results,
            profiles=["shared_profile"],
            scorer_protocols=[],
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].summary_path, valid.resolve())

    def test_metadata_protocol_mismatch_is_not_reportable(self):
        summary = write_result(
            self.root,
            run_name="mismatch",
            profile="mismatch_profile",
            scorer_protocol="score-protocol",
            model="org/Mismatch",
        )
        metadata_path = (
            summary.parents[2] / "predictions.jsonl.metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["scorer_protocol"] = "another-protocol"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        result = results_report.discover_results(self.root)[0]

        self.assertFalse(result.eligible)
        self.assertIn("metadata scorer_protocol does not match", result.reason)


class ResultsReportRenderingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_markdown_contains_exact_title_and_one_official_order_table(self):
        write_result(
            self.root,
            run_name="model",
            profile="model_profile",
            scorer_protocol="protocol-v1",
            model="/models/Model-With-Path",
        )
        result = results_report.discover_results(self.root)[0]

        markdown = results_report.render_markdown([result])
        lines = markdown.splitlines()

        self.assertEqual(lines[0], "# MSMU-Bench评测结果")
        self.assertEqual(lines[1], "")
        self.assertEqual(lines[2], results_report.REPORT_NOTE)
        self.assertEqual(lines[3], "")
        self.assertEqual(
            lines[4],
            "| 模型名称 | 存在性 | 物体计数 | 尺度估计 | 空间定位 | "
            "相对位置 | 绝对距离 | 尺度比较 | 参照物估计 | 平均 |",
        )
        self.assertEqual(len(lines), 7)
        self.assertEqual(markdown.count("\n| ---"), 1)
        self.assertIn("| Model-With-Path |", lines[6])
        self.assertIn(
            "| 10.00 | 20.00 | 30.00 | 40.00 | 50.00 | 60.00 | "
            "70.00 | 80.00 | 45.00 |",
            lines[6],
        )
        self.assertNotIn("strict official", markdown)
        self.assertNotIn(results_report.EXPECTED_RESULT_KIND, markdown)
        self.assertNotIn("Profile", markdown)
        self.assertEqual(len(lines[6].strip("|").split("|")), 10)

    def test_specialized_tracks_use_explicit_input_configuration_names(self):
        write_result(
            self.root,
            run_name="ssr-fair",
            profile="ssr",
            scorer_protocol="protocol-v1",
            model="org/SSR-VLM-7B",
            input_profile="rgb_only",
        )
        write_result(
            self.root,
            run_name="ssr-native",
            profile="ssr_native",
            scorer_protocol="protocol-v1",
            model="org/SSR-VLM-7B + org/SSR-MIDI-7B",
            input_profile="depth_native",
        )
        write_result(
            self.root,
            run_name="spatialrgpt",
            profile="spatialrgpt",
            scorer_protocol="protocol-v1",
            model="org/SpatialRGPT-VILA1.5-8B",
            input_profile="rgb_only",
        )
        write_result(
            self.root,
            run_name="3dthinker-fair",
            profile="3dthinker",
            scorer_protocol="protocol-v1",
            model="org/3DThinker-Mindcube (MindCube-trained stage-1 checkpoint)",
            input_profile="question_only",
        )
        write_result(
            self.root,
            run_name="3dthinker-native",
            profile="3dthinker_native",
            scorer_protocol="protocol-v1",
            model="org/3DThinker-Mindcube (MindCube-trained stage-1 checkpoint)",
            input_profile="mental3d_native",
        )
        write_result(
            self.root,
            run_name="spatialbot-fair",
            profile="spatialbot",
            scorer_protocol="protocol-v1",
            model="org/SpatialBot-3B",
            input_profile="rgb_only",
        )
        write_result(
            self.root,
            run_name="spatialbot-native",
            profile="spatialbot_native",
            scorer_protocol="protocol-v1",
            model="org/SpatialBot-3B",
            input_profile="zoedepth_rgbd_native",
        )
        results = results_report.selected_results(
            results_report.discover_results(self.root),
            profiles=[
                "ssr",
                "ssr_native",
                "spatialrgpt",
                "3dthinker",
                "3dthinker_native",
                "spatialbot",
                "spatialbot_native",
            ],
            scorer_protocols=["protocol-v1"],
        )

        markdown = results_report.render_markdown(results)

        self.assertIn("| SSR-VLM-7B（RGB） |", markdown)
        self.assertIn(
            "| SSR-VLM-7B + SSR-MIDI-7B（RGB + 深度估计） |",
            markdown,
        )
        self.assertIn("| SpatialRGPT-VILA1.5-8B |", markdown)
        self.assertIn("| 3DThinker-Mindcube（RGB） |", markdown)
        self.assertIn(
            "| 3DThinker-Mindcube（RGB + Mental-3D 提示词） |",
            markdown,
        )
        self.assertIn("| SpatialBot-3B（RGB） |", markdown)
        self.assertIn(
            "| SpatialBot-3B（RGB + 深度估计） |",
            markdown,
        )
        self.assertNotIn("公平版", markdown)
        self.assertNotIn("原生版", markdown)
        self.assertNotIn("估计深度", markdown)

    def test_unmapped_dual_track_profile_fails_closed(self):
        write_result(
            self.root,
            run_name="fair",
            profile="future_specialized",
            scorer_protocol="protocol-v1",
            model="org/Future-Specialized",
        )
        write_result(
            self.root,
            run_name="native",
            profile="future_specialized_native",
            scorer_protocol="protocol-v1",
            model="org/Future-Specialized",
        )
        results = results_report.discover_results(self.root)

        with self.assertRaisesRegex(
            results_report.ConfigurationError,
            "missing an explicit presentation configuration",
        ):
            results_report.render_markdown(results)

    def test_concise_table_rejects_mixed_scorer_protocols(self):
        write_result(
            self.root,
            run_name="old",
            profile="model",
            scorer_protocol="old",
            model="org/Model",
        )
        write_result(
            self.root,
            run_name="new",
            profile="model",
            scorer_protocol="new",
            model="org/Model",
        )

        with self.assertRaisesRegex(
            results_report.ConfigurationError,
            "exactly one scorer protocol",
        ):
            results_report.render_markdown(
                results_report.discover_results(self.root)
            )

    def test_profile_filter_order_is_preserved_and_protocol_filter_is_exact(self):
        write_result(
            self.root,
            run_name="alpha-old",
            profile="alpha",
            scorer_protocol="old",
            model="org/Zeta",
        )
        write_result(
            self.root,
            run_name="alpha-new",
            profile="alpha",
            scorer_protocol="new",
            model="org/Zeta",
        )
        write_result(
            self.root,
            run_name="beta-new",
            profile="beta",
            scorer_protocol="new",
            model="org/Alpha",
        )
        results = results_report.discover_results(self.root)

        selected = results_report.selected_results(
            results,
            profiles=["alpha", "beta"],
            scorer_protocols=["new"],
        )

        self.assertEqual(
            [(result.profile, result.scorer_protocol) for result in selected],
            [("alpha", "new"), ("beta", "new")],
        )

    def test_unknown_filters_fail_closed(self):
        write_result(
            self.root,
            run_name="known",
            profile="known",
            scorer_protocol="known-protocol",
            model="org/Known",
        )
        results = results_report.discover_results(self.root)

        with self.assertRaisesRegex(
            results_report.ConfigurationError,
            "requested profile",
        ):
            results_report.selected_results(
                results,
                profiles=["missing"],
                scorer_protocols=[],
            )
        with self.assertRaisesRegex(
            results_report.ConfigurationError,
            "requested scorer protocol",
        ):
            results_report.selected_results(
                results,
                profiles=[],
                scorer_protocols=["missing"],
            )

    def test_atomic_write_replaces_existing_report(self):
        output = self.root / "report.md"
        output.write_text("old\n", encoding="utf-8")

        results_report.write_atomic(output, "new\n")

        self.assertEqual(output.read_text(encoding="utf-8"), "new\n")
        self.assertEqual(list(self.root.glob(".report.md.*.tmp")), [])


class PublicResultsReportEntryTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary = Path(self.temporary_directory.name)
        self.output_root = temporary / "manual output"
        self.results_root = self.output_root / "03_full987"
        self.results_root.mkdir(parents=True)
        self.dataset = temporary / "dataset"
        self.dataset.mkdir()
        self.server_env = temporary / "server.env"
        self.server_env.write_text(
            "\n".join(
                (
                    f'REPO_ROOT="{REPOSITORY}"',
                    f'DATASET_ROOT="{self.dataset}"',
                    f'MANUAL_TEST_OUTPUT_ROOT="{self.output_root}"',
                    f'LATENT_PYTHON="{sys.executable}"',
                )
            )
            + "\n",
            encoding="utf-8",
        )
        self.script = (
            REPOSITORY / "scripts" / "msmu" / "build_results_report.sh"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_script(self, *arguments, env_file: str | None = None):
        environment = dict(os.environ)
        environment["MSMU_SERVER_ENV"] = env_file or str(self.server_env)
        return subprocess.run(
            ["bash", str(self.script), *arguments],
            cwd=REPOSITORY,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_help_does_not_require_server_environment(self):
        result = self.run_script(
            "--help",
            env_file="/missing/server.env",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile", result.stdout)
        self.assertIn("--scorer-protocol", result.stdout)

    def test_list_does_not_write_report_and_build_uses_default_output(self):
        write_result(
            self.results_root,
            run_name="selected",
            profile="selected_profile",
            scorer_protocol=results_report.SCORER_PROTOCOL,
            model="org/Selected",
        )
        output = self.results_root / results_report.DEFAULT_OUTPUT_NAME
        self.assertEqual(results_report.DEFAULT_OUTPUT_NAME, "msmu-result.md")

        listed = self.run_script("--list")

        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("selected_profile", listed.stdout)
        self.assertFalse(output.exists())

        built = self.run_script("--profile", "selected_profile")

        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertTrue(output.is_file())
        self.assertIn(
            "# MSMU-Bench评测结果",
            output.read_text(encoding="utf-8"),
        )

    def test_relative_results_root_is_rejected(self):
        result = self.run_script("--results-root", "relative/path")

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be absolute", result.stderr)


if __name__ == "__main__":
    unittest.main()
