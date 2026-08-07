from __future__ import annotations

import os
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from spatial_vlm_eval.benchmarks.q_spatial import scheduled_batch, scheduled_watcher
from spatial_vlm_eval.benchmarks.q_spatial.data import DATASET_FILES, DATASET_REVISION
from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILE_SEQUENCE, PROFILES
from spatial_vlm_eval.benchmarks.q_spatial.scorer import LEGACY_SCORER_PROTOCOL_V1, SCORER_PROTOCOL


class QSpatialRuntimeAndScriptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[3]

    def test_public_list_contains_exact_registry_order(self):
        environment = {**os.environ, "QSPATIAL_ENV_FILE": "/dev/null"}
        environment.pop("QSPATIAL_PYTHON", None)
        environment.pop("PYTHON", None)
        environment["LATENT_PYTHON"] = sys.executable
        completed = subprocess.run(
            ["bash", "scripts/q_spatial/run_inference.sh", "--list"],
            cwd=self.repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), 21)
        self.assertTrue(lines[0].startswith("llava_next_mistral_7b\t"))
        self.assertTrue(lines[-1].startswith("spatialladder3b_rgb\t"))

    def test_dry_run_selection_uses_registry_order_without_data_or_keys(self):
        environment = {**os.environ, "QSPATIAL_ENV_FILE": "/dev/null"}
        environment.pop("QSPATIAL_PYTHON", None)
        environment.pop("PYTHON", None)
        environment["LATENT_PYTHON"] = sys.executable
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    "bash", "scripts/q_spatial/run_inference.sh", "--stage", "full",
                    "--models", "qwen3_vl_8b,llava_next_mistral_7b",
                    "--output-root", directory, "--dry-run",
                ],
                cwd=self.repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        lines = completed.stdout.splitlines()
        self.assertTrue(lines[0].startswith("full\tllava_next_mistral_7b\t"))
        self.assertTrue(lines[1].startswith("full\tqwen3_vl_8b\t"))

    def test_internvl78_four_gpu_controller_reuses_canonical_root_and_current_protocols(self):
        script = (
            self.repository
            / "scripts"
            / "q_spatial"
            / "run_internvl3_78b_evaluation.sh"
        )
        source = script.read_text(encoding="utf-8")
        for required in [
            'PROFILE="internvl3_78b"',
            "track_directory",
            "--stage test",
            "--stage full",
            "--predictions",
            "SCORER_PROTOCOL",
            "complete_result_errors",
            "build_results_report.sh",
            '_single_model_evaluation',
            'QSPATIAL_INTERNVL3_78B_GPU_IDS',
            "q-spatial-result.md",
            "全轨完整度：21/21",
        ]:
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertNotIn(
            'control_root="${QSPATIAL_OUTPUT_ROOT%/}/_internvl3_78b_evaluation"',
            source,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "q-spatial-outputs"
            completed = subprocess.run(
                ["bash", str(script), "--dry-run"],
                cwd=self.repository,
                env={
                    **os.environ,
                    "QSPATIAL_ENV_FILE": "/dev/null",
                    "QSPATIAL_PYTHON": sys.executable,
                    "QSPATIAL_OUTPUT_ROOT": str(output_root),
                    "QSPATIAL_VLLM": "/dry-run/vllm",
                    "INTERNVL3_78B_MODEL": "/dry-run/locked-internvl3-78b",
                },
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(output_root.exists())
        self.assertIn("full\tinternvl3_78b", completed.stdout)
        self.assertIn(PROFILES["internvl3_78b"].inference_protocol, completed.stdout)
        self.assertIn(SCORER_PROTOCOL, completed.stdout)
        self.assertIn("q-spatial-result.md", completed.stdout)
        self.assertIn("no GPU/service/inference/scoring/report action", completed.stdout)

        faq = subprocess.run(
            ["bash", str(script), "--faq"],
            cwd=self.repository,
            env={**os.environ, "QSPATIAL_ENV_FILE": "/dev/null"},
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("原 q-spatial-result.md 原地更新", faq.stdout)
        self.assertIn("当前 registry/SCORER_PROTOCOL", faq.stdout)
        self.assertIn("结束后恢复", faq.stdout)

    def test_shell_entrypoints_are_syntax_valid_and_server_is_non_destructive(self):
        for path in sorted((self.repository / "scripts" / "q_spatial").glob("*.sh")):
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)
        serve = (self.repository / "scripts" / "q_spatial" / "serve_vllm_profile.sh").read_text()
        self.assertIn("port ${port} is occupied", serve)
        self.assertIn("gpu_preflight.sh", serve)
        self.assertNotIn("kill ", serve)
        self.assertNotIn("pkill", serve)

    def test_schedule_covers_exactly_twenty_runnable_profiles_without_conflicts(self):
        self.assertEqual(scheduled_batch.schedule_errors(), [])
        jobs = scheduled_batch.SCHEDULE
        keys = [job.profile for job in jobs]
        self.assertEqual(len(keys), 20)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), set(PROFILE_SEQUENCE) - {"internvl3_78b"})
        self.assertEqual(
            [job.profile for job in scheduled_batch.jobs_for_lane("api")],
            ["gpt5_openrouter_non_zdr", "gemini31pro_openrouter_non_zdr"],
        )
        self.assertEqual(
            {job.gpu_ids for job in scheduled_batch.jobs_for_lane("dual")}, {(0, 1)}
        )
        self.assertEqual(
            {job.gpu_ids for job in scheduled_batch.jobs_for_lane("gpu0")}, {(0,)}
        )
        self.assertEqual(
            {job.gpu_ids for job in scheduled_batch.jobs_for_lane("gpu1")}, {(1,)}
        )
        self.assertEqual(scheduled_batch.LANE_PHASE_BARRIER, {"gpu0": "dual", "gpu1": "dual"})

    def test_llava_jobs_use_checkpoint_safe_context_length(self):
        for profile_key in ("llava_next_yi_34b", "llava_next_mistral_7b"):
            job = next(job for job in scheduled_batch.SCHEDULE if job.profile == profile_key)
            with patch.dict(
                os.environ, {"QSPATIAL_VLLM_MAX_MODEL_LEN": "32768"}, clear=False
            ):
                environment = scheduled_batch.environment_for_job(job)
            self.assertEqual(environment["QSPATIAL_VLLM_MAX_MODEL_LEN"], "4096")

        qwen_job = next(
            job for job in scheduled_batch.SCHEDULE if job.profile == "qwen3_vl_32b"
        )
        with patch.dict(
            os.environ, {"QSPATIAL_VLLM_MAX_MODEL_LEN": "32768"}, clear=False
        ):
            environment = scheduled_batch.environment_for_job(qwen_job)
        self.assertEqual(environment["QSPATIAL_VLLM_MAX_MODEL_LEN"], "32768")

    def test_schedule_dry_run_has_zero_external_calls_and_writes(self):
        environment = {**os.environ, "QSPATIAL_ENV_FILE": "/dev/null"}
        environment.pop("QSPATIAL_PYTHON", None)
        environment.pop("PYTHON", None)
        environment["LATENT_PYTHON"] = sys.executable
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            environment["QSPATIAL_OUTPUT_ROOT"] = str(output)
            completed = subprocess.run(
                ["bash", "scripts/q_spatial/run_scheduled_batch.sh", "--dry-run"],
                cwd=self.repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(output.exists())
        self.assertEqual(completed.stdout.count("dry-run stage=full phase="), 20)
        self.assertIn("no dataset/model/runner/GPU/API/scorer call", completed.stdout)

    def test_schedule_test_dry_run_never_lists_full_or_validator(self):
        environment = {**os.environ, "QSPATIAL_ENV_FILE": "/dev/null"}
        environment["QSPATIAL_PYTHON"] = sys.executable
        completed = subprocess.run(
            [
                "bash",
                "scripts/q_spatial/run_scheduled_batch.sh",
                "--stage",
                "test",
                "--dry-run",
            ],
            cwd=self.repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.count("dry-run stage=test phase="), 20)
        self.assertNotIn("full,validator", completed.stdout)

    def test_schedule_test_rejects_skip_completed(self):
        environment = {**os.environ, "QSPATIAL_ENV_FILE": "/dev/null"}
        environment["QSPATIAL_PYTHON"] = sys.executable
        completed = subprocess.run(
            [
                "bash",
                "scripts/q_spatial/run_scheduled_batch.sh",
                "--stage",
                "test",
                "--without-internvl78",
                "--with-paid-api",
                "--skip-completed",
            ],
            cwd=self.repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("applies only to --stage full", completed.stderr)

    def test_test_only_lane_runs_test_but_never_full(self):
        job = scheduled_batch.jobs_for_lane("api")[0]
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            return 0

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "QSPATIAL_OUTPUT_ROOT": directory,
                "QSPATIAL_PARQUET_ROOT": directory,
                "QSPATIAL_SCANNET_RGB_ROOT": directory,
                "QSPATIAL_PYTHON": sys.executable,
            },
            clear=False,
        ), patch.object(
            scheduled_batch, "jobs_for_lane", return_value=(job,)
        ), patch.object(
            scheduled_batch, "QSpatialTestContract", return_value=Mock()
        ), patch.object(
            scheduled_batch, "reusable_gate_errors", side_effect=[["missing"], []]
        ), patch.object(
            scheduled_batch, "_run_owned", side_effect=fake_run
        ), patch.object(
            scheduled_batch, "_signal_watcher"
        ):
            control = Path(directory) / "control"
            (control / "logs").mkdir(parents=True)
            result = scheduled_batch.run_lane(
                "api",
                run_id="test-run",
                repository_root=self.repository,
                control_root=control,
                skip_completed=False,
                stage="test",
            )
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index("--stage") + 1], "test")
        self.assertNotIn("full", calls[0])

    def test_formal_schedule_requires_both_explicit_authorizations(self):
        environment = {**os.environ, "QSPATIAL_ENV_FILE": "/dev/null"}
        environment["QSPATIAL_PYTHON"] = sys.executable
        for arguments in ([], ["--without-internvl78"], ["--with-paid-api"]):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    ["bash", "scripts/q_spatial/run_scheduled_batch.sh", *arguments],
                    cwd=self.repository,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("requires both", completed.stderr)

    def test_gate_reuse_and_complete_skip_are_fail_closed(self):
        class Contract:
            dataset_fingerprint = "dataset-fingerprint"

            def __len__(self):
                return 271

        profile = PROFILES["qwen3_vl_2b"]
        contract = Contract()
        binding_value = {"current": "binding"}
        with tempfile.TemporaryDirectory() as directory:
            track = Path(directory)
            prediction = track / "predictions.jsonl"
            prediction.write_text(
                "".join(json.dumps({"index": index, "raw_prediction": "1 cm"}) + "\n" for index in range(271)),
                encoding="utf-8",
            )
            validation = {
                "passed": True,
                "num_prediction_rows": 271,
                "dataset_fingerprint": contract.dataset_fingerprint,
            }
            (track / "prediction_validation.json").write_text(json.dumps(validation), encoding="utf-8")
            metadata = {
                "publishable_inference": True,
                "num_predictions": 271,
                "model": {"profile": profile.key, "model_revision": profile.revision},
                "inference_protocol": profile.inference_protocol,
                "scorer_protocol": LEGACY_SCORER_PROTOCOL_V1,
                "dataset": {
                    "revision": DATASET_REVISION,
                    "fingerprint": contract.dataset_fingerprint,
                    "files": {item.name: item.sha256 for item in DATASET_FILES},
                    "official_test_size": 271,
                },
                "binding": binding_value,
                "binding_digest": scheduled_batch._digest(binding_value),
                "output_sha256": scheduled_batch._file_digest(prediction),
            }
            prediction.with_suffix(".jsonl.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with (
                patch.object(scheduled_batch, "track_directory", return_value=track),
                patch.object(scheduled_batch, "resolve_configuration", return_value=object()),
                patch.object(scheduled_batch, "binding", return_value=binding_value),
                patch.object(scheduled_batch, "reusable_gate_errors", return_value=[]),
            ):
                self.assertEqual(
                    scheduled_batch.complete_result_errors(profile, contract, Path(directory)), []
                )
                metadata["model"]["model_revision"] = "wrong"
                prediction.with_suffix(".jsonl.metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                errors = scheduled_batch.complete_result_errors(profile, contract, Path(directory))
                self.assertIn("metadata model revision mismatch", errors)

    def test_owned_process_group_cleanup_targets_only_recorded_group(self):
        process = Mock()
        process.pid = 43210
        process.poll.return_value = 0
        process.wait.return_value = 0
        with (
            patch.object(scheduled_batch, "_group_alive", side_effect=[True, False, False]),
            patch.object(scheduled_batch.os, "killpg") as killpg,
        ):
            scheduled_batch._stop_owned_group(process, 1)
        killpg.assert_called_once_with(43210, scheduled_batch.signal.SIGTERM)
        process.poll.assert_called_once_with()

    def test_owned_service_port_release_waits_without_taking_over_listener(self):
        with patch.object(
            scheduled_batch, "_port_is_available", side_effect=[False, False, True]
        ), patch.object(scheduled_batch.time, "sleep") as sleep:
            scheduled_batch._wait_for_port_release(18101, 30)
        self.assertEqual(sleep.call_count, 2)

    def test_controller_failure_isolation_and_dual_barrier(self):
        class FakeProcess:
            def __init__(self, status):
                self.status = status
                self.pid = 10000 + status

            def wait(self, timeout=None):
                return self.status

            def poll(self):
                return self.status

        def exercise(statuses):
            started = []
            handles = []

            def fake_lane(lane, **_kwargs):
                started.append(lane)
                handle = io.StringIO()
                handles.append(handle)
                return FakeProcess(statuses[lane]), handle

            def fake_watcher(lane, **_kwargs):
                handle = io.StringIO()
                handles.append(handle)
                return FakeProcess(0), handle

            with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ,
                {"QSPATIAL_OUTPUT_ROOT": directory},
                clear=False,
            ), patch.object(
                scheduled_batch, "preflight", return_value={"passed": True, "errors": []}
            ), patch.object(
                scheduled_batch, "schedule_payload", return_value={"plan": "frozen"}
            ), patch.object(
                scheduled_batch, "_start_lane", side_effect=fake_lane
            ), patch.object(
                scheduled_batch, "_start_watcher", side_effect=fake_watcher
            ), patch.object(
                scheduled_batch, "_signal_watcher"
            ):
                result = scheduled_batch.run_controller(
                    self.repository, skip_completed=False, stage="test"
                )
            return result, started

        result, started = exercise({"dual": 0, "api": 1, "gpu0": 1, "gpu1": 0})
        self.assertEqual(result, 1)
        self.assertEqual(started, ["dual", "api", "gpu0", "gpu1"])
        result, started = exercise({"dual": 1, "api": 0})
        self.assertEqual(result, 1)
        self.assertEqual(started, ["dual", "api"])

    def test_health_watcher_is_read_only_event_driven_and_filters_start(self):
        source = (
            self.repository
            / "src/spatial_vlm_eval/benchmarks/q_spatial/scheduled_watcher.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"wait-for"', source)
        self.assertNotIn("SIGTERM", source)
        self.assertNotIn("SIGKILL", source)
        self.assertNotIn("score_results", source)
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory)
            (control / "active").mkdir()
            (control / "status.tsv").write_text(
                "run_id\tphase\tlane\tprofile\tstate\ttimestamp\tdetail\n"
                "run\tA\tapi\tgpt5\tSTART\tt\t-\n"
                "run\tA\tapi\tgpt5\tPASS\tt\tvalidated\n"
                "run\tA\tapi\t-\tCOMPLETE\tt\tpassed\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                result = scheduled_watcher.watch(control, lane="api", run_id="run")
        self.assertEqual(result, 0)
        self.assertNotIn("START", output.getvalue())
        self.assertIn("PASS lane=api", output.getvalue())
        self.assertIn("COMPLETE lane=api", output.getvalue())


if __name__ == "__main__":
    unittest.main()
