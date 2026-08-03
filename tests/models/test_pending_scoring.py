import contextlib
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[2]
HELPER_PATH = REPOSITORY / "scripts" / "msmu" / "_score_pending_results.py"
SPEC = importlib.util.spec_from_file_location("msmu_pending_scoring", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
pending_scoring = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pending_scoring
SPEC.loader.exec_module(pending_scoring)


OFFICIAL_TYPES = sorted(
    pending_scoring.OFFICIAL_QUANT_TYPES | pending_scoring.OFFICIAL_QUAL_TYPES
)


def prediction_path(root: Path, run_name: str, protocol: str | None = None) -> Path:
    selected_protocol = protocol or pending_scoring.SCORER_PROTOCOL
    path = (
        root
        / run_name
        / "revision"
        / "inference-protocol"
        / selected_protocol
        / "predictions.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def write_complete_score_artifacts(predictions: Path, score_dir: Path) -> None:
    score_dir.mkdir(parents=True, exist_ok=True)
    counts = {kind: 1 for kind in OFFICIAL_TYPES}
    counts[OFFICIAL_TYPES[-1]] += pending_scoring.MSMU_OFFICIAL_TEST_SIZE - len(
        OFFICIAL_TYPES
    )
    summary = {
        "publishable": True,
        "status": "complete",
        "result_kind": "official-compatible internal score",
        "publication_gates": {
            name: True for name in pending_scoring.EXPECTED_PUBLICATION_GATES
        },
        "publication_gate_failures": [],
        "num_judge_failures": 0,
        "num_samples": pending_scoring.MSMU_OFFICIAL_TEST_SIZE,
        "micro_accuracy": 0.5,
        "official_macro8_accuracy": 0.5,
        "missing_official_types": [],
        "protocol": pending_scoring.SCORER_PROTOCOL,
        "official_types": {
            kind: {"count": count, "accuracy": 0.5}
            for kind, count in counts.items()
        },
    }
    validation = {
        "passed": True,
        "allow_subset": False,
        "official_test_size": pending_scoring.MSMU_OFFICIAL_TEST_SIZE,
        "num_prediction_rows": pending_scoring.MSMU_OFFICIAL_TEST_SIZE,
        "num_unique_indices": pending_scoring.MSMU_OFFICIAL_TEST_SIZE,
        "errors": [],
        "predictions": str(predictions),
    }
    (score_dir / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (score_dir / "prediction_validation.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )
    rows = "".join(
        json.dumps({"index": index}) + "\n"
        for index in range(pending_scoring.MSMU_OFFICIAL_TEST_SIZE)
    )
    (score_dir / "scored_rows.jsonl").write_text(rows, encoding="utf-8")
    (score_dir / "judge_cache.jsonl").write_text(rows, encoding="utf-8")
    (score_dir / "judge_failures.jsonl").write_text("", encoding="utf-8")
    (score_dir / "score.log").write_text("complete\n", encoding="utf-8")


@contextlib.contextmanager
def models_server(payload: bytes, *, status: int = 200):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class PendingScoringStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_discovers_arbitrary_runs_and_excludes_other_protocols(self):
        included = prediction_path(self.root, "newly-added-run")
        excluded = prediction_path(
            self.root,
            "historical-run",
            protocol="different-scorer-protocol",
        )
        states = pending_scoring.discover_results(self.root)
        by_path = {state.predictions: state for state in states}
        self.assertEqual(by_path[included.resolve()].state, "new")
        self.assertTrue(by_path[included.resolve()].included)
        self.assertEqual(by_path[excluded.resolve()].state, "excluded_protocol")
        self.assertFalse(by_path[excluded.resolve()].included)

    def test_classifies_new_resume_retry_and_complete(self):
        new_prediction = prediction_path(self.root, "a-new")
        resume_prediction = prediction_path(self.root, "b-resume")
        retry_prediction = prediction_path(self.root, "c-retry")
        corrupt_prediction = prediction_path(self.root, "d-corrupt")
        complete_prediction = prediction_path(self.root, "e-complete")

        resume_dir = (
            resume_prediction.parent
            / "scores"
            / pending_scoring.SCORER_PROTOCOL
        )
        resume_dir.mkdir(parents=True)
        (resume_dir / "judge_cache.jsonl").write_text(
            json.dumps({"index": 0}) + "\n",
            encoding="utf-8",
        )

        retry_dir = (
            retry_prediction.parent / "scores" / pending_scoring.SCORER_PROTOCOL
        )
        write_complete_score_artifacts(retry_prediction, retry_dir)
        retry_summary = json.loads(
            (retry_dir / "summary.json").read_text(encoding="utf-8")
        )
        retry_summary["publishable"] = False
        (retry_dir / "summary.json").write_text(
            json.dumps(retry_summary), encoding="utf-8"
        )

        corrupt_dir = (
            corrupt_prediction.parent
            / "scores"
            / pending_scoring.SCORER_PROTOCOL
        )
        corrupt_dir.mkdir(parents=True)
        (corrupt_dir / "summary.json").write_text("{bad json", encoding="utf-8")

        complete_dir = (
            complete_prediction.parent
            / "scores"
            / pending_scoring.SCORER_PROTOCOL
        )
        write_complete_score_artifacts(complete_prediction, complete_dir)

        states = {
            state.predictions: state
            for state in pending_scoring.discover_results(self.root)
        }
        self.assertEqual(states[new_prediction.resolve()].state, "new")
        self.assertEqual(states[resume_prediction.resolve()].state, "resume")
        self.assertEqual(states[retry_prediction.resolve()].state, "retry")
        self.assertIn("publishable", states[retry_prediction.resolve()].reason)
        self.assertEqual(states[corrupt_prediction.resolve()].state, "retry")
        self.assertIn("summary.json is invalid", states[corrupt_prediction.resolve()].reason)
        self.assertEqual(states[complete_prediction.resolve()].state, "complete")

    def test_missing_canonical_artifact_forces_retry(self):
        predictions = prediction_path(self.root, "missing-artifact")
        score_dir = predictions.parent / "scores" / pending_scoring.SCORER_PROTOCOL
        write_complete_score_artifacts(predictions, score_dir)
        (score_dir / "judge_failures.jsonl").unlink()
        state = pending_scoring.classify_prediction(predictions)
        self.assertEqual(state.state, "retry")
        self.assertIn("missing canonical artifact", state.reason)

    def test_exact_prediction_selector_must_be_inside_results_root(self):
        selected = prediction_path(self.root, "selected")
        resolved = pending_scoring.resolve_selected_predictions(
            str(selected),
            self.root.resolve(),
        )
        self.assertEqual(resolved, selected.resolve())

        outside_root = self.root.parent / "outside"
        outside = prediction_path(outside_root, "outside")
        with self.assertRaisesRegex(
            pending_scoring.ConfigurationError,
            "outside results root",
        ):
            pending_scoring.resolve_selected_predictions(
                str(outside),
                self.root.resolve(),
            )


class JudgeReadinessTest(unittest.TestCase):
    def test_accepts_expected_model(self):
        payload = json.dumps({"data": [{"id": "expected-judge"}]}).encode()
        with models_server(payload) as base_url:
            pending_scoring.check_judge_ready(
                base_url=base_url,
                expected_model="expected-judge",
                api_key="local",
                timeout=2,
            )

    def test_rejects_bad_json_and_missing_model(self):
        with models_server(b"not-json") as base_url:
            with self.assertRaises(pending_scoring.JudgePreflightError):
                pending_scoring.check_judge_ready(
                    base_url=base_url,
                    expected_model="expected-judge",
                    api_key="local",
                    timeout=2,
                )
        payload = json.dumps({"data": [{"id": "another-judge"}]}).encode()
        with models_server(payload) as base_url:
            with self.assertRaisesRegex(
                pending_scoring.JudgePreflightError,
                "is not served",
            ):
                pending_scoring.check_judge_ready(
                    base_url=base_url,
                    expected_model="expected-judge",
                    api_key="local",
                    timeout=2,
                )

    def test_rejects_unreachable_endpoint(self):
        with self.assertRaises(pending_scoring.JudgePreflightError):
            pending_scoring.check_judge_ready(
                base_url="http://127.0.0.1:1/v1",
                expected_model="expected-judge",
                api_key="local",
                timeout=0.1,
            )


class SerialScoringExecutionTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary = Path(self.temporary_directory.name)
        self.root = temporary / "results root with spaces"
        self.root.mkdir()
        self.dataset = temporary / "dataset"
        self.dataset.mkdir()
        self.call_log = temporary / "calls.tsv"
        self.template = temporary / "complete-template"
        template_prediction = prediction_path(temporary, "template")
        write_complete_score_artifacts(template_prediction, self.template)
        self.fake_scorer = temporary / "fake-scorer.sh"
        self.fake_scorer.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf 'start\\t%s\\n' "$PREDICTIONS" >> "$CALL_LOG"
if [[ -n "${FAIL_MATCH:-}" && "$PREDICTIONS" == *"$FAIL_MATCH"* ]]; then
  exit 7
fi
if [[ "${INTERRUPT_SCORER:-0}" == "1" ]]; then
  exit 130
fi
if [[ "${INCOMPLETE_SCORER:-0}" == "1" ]]; then
  exit 0
fi
if [[ -n "${LATE_PREDICTION:-}" && ! -e "$LATE_PREDICTION" ]]; then
  mkdir -p "$(dirname "$LATE_PREDICTION")"
  printf '{}\\n' > "$LATE_PREDICTION"
fi
mkdir -p "$OUTPUT_DIR"
cp -R "$COMPLETE_TEMPLATE/." "$OUTPUT_DIR/"
printf 'end\\t%s\\n' "$PREDICTIONS" >> "$CALL_LOG"
""",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def environment(self, base_url: str) -> dict[str, str]:
        return {
            "MSMU_SCORE_RESULTS_ROOT": str(self.root),
            "DATASET_ROOT": str(self.dataset),
            "JUDGE_BASE_URL": base_url,
            "JUDGE_MODEL_NAME": "expected-judge",
            "JUDGE_PREFLIGHT_TIMEOUT_SECONDS": "2",
            "API_KEY": "secret-for-test",
            "CALL_LOG": str(self.call_log),
            "COMPLETE_TEMPLATE": str(self.template),
            "MANUAL_DRY_RUN": "0",
            "WORKERS": "16",
            "RETRIES": "4",
        }

    def run_main(self, base_url: str, extra_environment=None, arguments=None) -> int:
        environment = self.environment(base_url)
        if extra_environment:
            environment.update(extra_environment)
        with patch.dict(os.environ, environment, clear=False):
            return pending_scoring.main(
                arguments or [],
                repository=REPOSITORY,
                score_script=self.fake_scorer,
            )

    def test_multiple_pending_results_run_once_each_in_stable_serial_order(self):
        later = prediction_path(self.root, "z-run")
        earlier = prediction_path(self.root, "a-run")
        resume = prediction_path(self.root, "m-resume")
        resume_dir = resume.parent / "scores" / pending_scoring.SCORER_PROTOCOL
        resume_dir.mkdir(parents=True)
        (resume_dir / "judge_cache.jsonl").write_text(
            json.dumps({"index": 0}) + "\n",
            encoding="utf-8",
        )
        payload = json.dumps({"data": [{"id": "expected-judge"}]}).encode()
        with models_server(payload) as base_url:
            self.assertEqual(self.run_main(base_url), 0)

        lines = self.call_log.read_text(encoding="utf-8").splitlines()
        expected_paths = [earlier.resolve(), resume.resolve(), later.resolve()]
        self.assertEqual(
            lines,
            [
                item
                for path in expected_paths
                for item in (f"start\t{path}", f"end\t{path}")
            ],
        )
        for path in expected_paths:
            self.assertEqual(
                pending_scoring.classify_prediction(path).state,
                "complete",
            )
        control_dir = pending_scoring.batch_directory(self.root)
        for path in control_dir.rglob("*"):
            if path.is_file():
                self.assertNotIn(
                    "secret-for-test",
                    path.read_text(encoding="utf-8"),
                )

    def test_exact_prediction_selector_scores_only_that_result(self):
        selected = prediction_path(self.root, "selected")
        prediction_path(self.root, "other-pending")
        payload = json.dumps({"data": [{"id": "expected-judge"}]}).encode()
        with models_server(payload) as base_url:
            return_code = self.run_main(
                base_url,
                arguments=["--predictions", str(selected.resolve())],
            )
        self.assertEqual(return_code, 0)
        self.assertEqual(
            self.call_log.read_text(encoding="utf-8").splitlines(),
            [
                f"start\t{selected.resolve()}",
                f"end\t{selected.resolve()}",
            ],
        )

    def test_first_scorer_failure_stops_later_results(self):
        first = prediction_path(self.root, "a-fail")
        prediction_path(self.root, "z-never-started")
        payload = json.dumps({"data": [{"id": "expected-judge"}]}).encode()
        with models_server(payload) as base_url:
            return_code = self.run_main(
                base_url,
                extra_environment={"FAIL_MATCH": "a-fail"},
            )
        self.assertEqual(return_code, 1)
        self.assertEqual(
            self.call_log.read_text(encoding="utf-8").splitlines(),
            [f"start\t{first.resolve()}"],
        )

    def test_zero_exit_still_requires_complete_publication_artifacts(self):
        first = prediction_path(self.root, "a-incomplete")
        prediction_path(self.root, "z-never-started")
        payload = json.dumps({"data": [{"id": "expected-judge"}]}).encode()
        with models_server(payload) as base_url:
            return_code = self.run_main(
                base_url,
                extra_environment={"INCOMPLETE_SCORER": "1"},
            )
        self.assertEqual(return_code, 1)
        self.assertEqual(
            self.call_log.read_text(encoding="utf-8").splitlines(),
            [f"start\t{first.resolve()}"],
        )

    def test_scorer_interrupt_returns_130(self):
        prediction_path(self.root, "interrupt")
        payload = json.dumps({"data": [{"id": "expected-judge"}]}).encode()
        with models_server(payload) as base_url:
            return_code = self.run_main(
                base_url,
                extra_environment={"INTERRUPT_SCORER": "1"},
            )
        self.assertEqual(return_code, 130)

    def test_candidates_created_after_start_wait_for_the_next_batch(self):
        initial = prediction_path(self.root, "a-initial")
        late = (
            self.root
            / "z-late"
            / "revision"
            / "inference-protocol"
            / pending_scoring.SCORER_PROTOCOL
            / "predictions.jsonl"
        )
        payload = json.dumps({"data": [{"id": "expected-judge"}]}).encode()
        with models_server(payload) as base_url:
            return_code = self.run_main(
                base_url,
                extra_environment={"LATE_PREDICTION": str(late)},
            )
        self.assertEqual(return_code, 0)
        self.assertEqual(
            self.call_log.read_text(encoding="utf-8").splitlines(),
            [f"start\t{initial.resolve()}", f"end\t{initial.resolve()}"],
        )
        self.assertEqual(pending_scoring.classify_prediction(late).state, "new")

    def test_check_validates_paths_lock_and_expected_judge(self):
        payload = json.dumps({"data": [{"id": "expected-judge"}]}).encode()
        with models_server(payload) as base_url:
            with patch.dict(
                os.environ,
                self.environment(base_url),
                clear=False,
            ):
                self.assertEqual(pending_scoring.main(["--check"]), 0)

        wrong_payload = json.dumps({"data": [{"id": "wrong-judge"}]}).encode()
        with models_server(wrong_payload) as base_url:
            with patch.dict(
                os.environ,
                self.environment(base_url),
                clear=False,
            ):
                self.assertEqual(pending_scoring.main(["--check"]), 4)

    def test_lock_conflict_returns_4(self):
        prediction_path(self.root, "locked")
        lock_path = (
            pending_scoring.batch_directory(self.root)
            / "lock"
        )
        lock_path.parent.mkdir(parents=True)
        payload = json.dumps({"data": [{"id": "expected-judge"}]}).encode()
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with models_server(payload) as base_url:
                self.assertEqual(self.run_main(base_url), 4)

    def test_zero_candidates_succeeds_without_dataset_or_judge(self):
        environment = {
            "MSMU_SCORE_RESULTS_ROOT": str(self.root),
            "DATASET_ROOT": str(self.root / "missing-dataset"),
            "JUDGE_BASE_URL": "",
            "JUDGE_MODEL_NAME": "",
            "MANUAL_DRY_RUN": "0",
        }
        with patch.dict(os.environ, environment, clear=False):
            return_code = pending_scoring.main(
                [],
                repository=REPOSITORY,
                score_script=self.fake_scorer,
            )
        self.assertEqual(return_code, 0)
        self.assertFalse(self.call_log.exists())


class PublicScoringEntryTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary = Path(self.temporary_directory.name)
        self.output_root = temporary / "manual output"
        self.results_root = self.output_root / "03_full987"
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
                    'JUDGE_BASE_URL="http://127.0.0.1:1/v1"',
                    'JUDGE_MODEL_NAME="expected-judge"',
                )
            )
            + "\n",
            encoding="utf-8",
        )
        self.script = REPOSITORY / "scripts" / "msmu" / "score_pending_results.sh"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_script(self, *arguments, extra_environment=None):
        environment = dict(os.environ)
        environment.update(
            {
                "MSMU_SERVER_ENV": str(self.server_env),
                "MANUAL_DRY_RUN": "0",
            }
        )
        if extra_environment:
            environment.update(extra_environment)
        return subprocess.run(
            ["bash", str(self.script), *arguments],
            cwd=REPOSITORY,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_list_status_and_dry_run_never_require_judge(self):
        prediction_path(self.results_root, "arbitrary-run")
        listed = self.run_script("--list")
        status = self.run_script("--status")
        dry_run = self.run_script(
            extra_environment={"MANUAL_DRY_RUN": "1"},
        )
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertIn("arbitrary-run", listed.stdout)
        self.assertIn("new\t1", status.stdout)
        self.assertIn("dry-run order=1", dry_run.stdout)
        self.assertIn("no judge/scorer action was taken", dry_run.stdout)

    def test_predictions_filter_lists_only_the_exact_selected_result(self):
        selected = prediction_path(self.results_root, "selected-run")
        prediction_path(self.results_root, "other-run")
        listed = self.run_script(
            "--list",
            "--predictions",
            str(selected.resolve()),
        )
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("selected-run", listed.stdout)
        self.assertNotIn("other-run", listed.stdout)

    def test_results_root_override_must_be_absolute(self):
        result = self.run_script("--list", "--results-root", "relative/path")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be absolute", result.stderr)

    def test_help_does_not_require_server_environment(self):
        result = self.run_script(
            "--help",
            extra_environment={"MSMU_SERVER_ENV": "/missing/server.env"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--results-root", result.stdout)
        self.assertIn("--predictions", result.stdout)

    def test_new_scoring_sources_do_not_contain_current_model_names(self):
        source = self.script.read_text(encoding="utf-8") + HELPER_PATH.read_text(
            encoding="utf-8"
        )
        forbidden = (
            "llava_next_",
            "internvl3_",
            "qwen25_vl_",
            "ssr_native",
            "spatialrgpt",
            "3dthinker",
            "spatialbot",
            "gpt5",
            "gemini31pro",
        )
        for name in forbidden:
            with self.subTest(name=name):
                self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
