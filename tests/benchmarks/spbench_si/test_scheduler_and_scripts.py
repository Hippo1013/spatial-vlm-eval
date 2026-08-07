from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from spatial_vlm_eval.benchmarks.spbench_si import scheduled_batch, scheduled_watcher
from spatial_vlm_eval.benchmarks.spbench_si.profiles import PROFILE_SEQUENCE


class SPBenchSISchedulerScriptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.repository = Path(__file__).resolve().parents[3]

    def test_schedule_exact_twenty_api_serial_and_phase_barrier(self):
        self.assertEqual(scheduled_batch.schedule_errors(), [])
        keys = [job.profile for job in scheduled_batch.SCHEDULE]
        self.assertEqual(set(keys), set(PROFILE_SEQUENCE) - {"internvl3_78b"})
        self.assertEqual([job.profile for job in scheduled_batch.jobs_for_lane("dual")], [
            "internvl3_38b", "llava_next_yi_34b", "qwen3_vl_32b"
        ])
        self.assertEqual([job.profile for job in scheduled_batch.jobs_for_lane("api")], [
            "gpt5_openrouter_non_zdr", "gemini31pro_openrouter_non_zdr"
        ])
        self.assertEqual(scheduled_batch.LANE_PHASE_BARRIER, {"gpu0": "dual", "gpu1": "dual"})

    def test_public_list_and_dry_run_have_zero_model_or_api_calls(self):
        environment = {**os.environ, "SPBENCH_SI_ENV_FILE": "/dev/null", "LATENT_PYTHON": sys.executable}
        listed = subprocess.run(["bash", "scripts/spbench_si/run_inference.sh", "--list"], cwd=self.repository, env=environment, check=True, capture_output=True, text=True)
        self.assertEqual(len(listed.stdout.splitlines()), 21)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            environment["SPBENCH_SI_OUTPUT_ROOT"] = str(output)
            dry = subprocess.run(["bash", "scripts/spbench_si/run_scheduled_batch.sh", "--stage", "test", "--dry-run"], cwd=self.repository, env=environment, check=True, capture_output=True, text=True)
            self.assertFalse(output.exists())
        self.assertEqual(dry.stdout.count("no dataset/model/runner/GPU/API/scorer call"), 20)

    def test_formal_schedule_requires_explicit_missing_four_card_and_paid_api_ack(self):
        environment = {**os.environ, "SPBENCH_SI_ENV_FILE": "/dev/null", "LATENT_PYTHON": sys.executable}
        completed = subprocess.run(["bash", "scripts/spbench_si/run_scheduled_batch.sh"], cwd=self.repository, env=environment, check=False, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires both", completed.stderr)

    def test_full_job_reuses_or_builds_test_gate_before_full(self):
        job = scheduled_batch.jobs_for_lane("api")[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "status.tsv").write_text(
                "run_id\tphase\tlane\tprofile\tstate\ttimestamp\tdetail\n", encoding="utf-8"
            )
            with patch.object(scheduled_batch, "_run_owned", return_value=0) as run, patch.object(
                scheduled_batch, "_signal_watcher"
            ):
                status = scheduled_batch._run_job(
                    job, stage="full", repository_root=self.repository,
                    control_root=root, run_id="run",
                )
        self.assertEqual(status, 0)
        stages = [call.args[0][3] for call in run.call_args_list]
        self.assertEqual(stages, ["test", "full"])

    def test_shell_syntax_and_watcher_source_is_read_only(self):
        for path in sorted((self.repository / "scripts" / "spbench_si").glob("*.sh")):
            subprocess.run(["bash", "-n", str(path)], check=True)
        source = (self.repository / "src/spatial_vlm_eval/benchmarks/spbench_si/scheduled_watcher.py").read_text()
        self.assertIn('"wait-for"', source)
        self.assertNotIn("SIGTERM", source)
        self.assertNotIn("score_results", source)

    def test_env_file_cannot_override_scheduler_job_gpu_or_context(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "server.env"
            env_file.write_text(
                "CUDA_VISIBLE_DEVICES=0\nSPBENCH_SI_VLLM_MAX_MODEL_LEN=32768\n",
                encoding="utf-8",
            )
            environment = {
                **os.environ,
                "SPBENCH_SI_ENV_FILE": str(env_file),
                "CUDA_VISIBLE_DEVICES": "1",
                "SPBENCH_SI_VLLM_MAX_MODEL_LEN": "4096",
            }
            completed = subprocess.run(
                [
                    "bash", "-c",
                    "source scripts/spbench_si/_env.sh; "
                    "printf '%s %s' \"$CUDA_VISIBLE_DEVICES\" \"$SPBENCH_SI_VLLM_MAX_MODEL_LEN\"",
                ],
                cwd=self.repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.stdout, "1 4096")

    def test_internvl78_single_model_controller_uses_canonical_outputs(self):
        script = (
            self.repository
            / "scripts"
            / "spbench_si"
            / "run_internvl3_78b_evaluation.sh"
        )
        source = script.read_text(encoding="utf-8")
        for required in [
            'PROFILE="internvl3_78b"',
            "track_directory",
            "--stage test",
            "--stage full",
            "--predictions",
            "exact-target dual-protocol scoring",
            "build_results_report.sh",
            '_single_model_evaluation',
            "SPBENCH_SI_INTERNVL3_78B_GPU_IDS",
            "MIN_GPU_COUNT=4",
            'sock.bind(("127.0.0.1", int(sys.argv[1])))',
            'report="${SPBENCH_SI_OUTPUT_ROOT%/}/spbench-si-result.md"',
            'control_root="${SPBENCH_SI_OUTPUT_ROOT%/}/_single_model_evaluation"',
        ]:
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertNotIn("--skip-resource-blocked", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn(
            'control_root="${SPBENCH_SI_OUTPUT_ROOT%/}/_internvl3_78b_evaluation"',
            source,
        )
        self.assertNotIn("internvl3-78b-result.md", source)
        self.assertNotIn("SPBENCH_SI_INTERNVL3_78B_OUTPUT_ROOT", source)

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "must-not-exist"
            environment = {
                **os.environ,
                "SPBENCH_SI_ENV_FILE": "/dev/null",
                "LATENT_PYTHON": sys.executable,
                "SPBENCH_SI_OUTPUT_ROOT": str(output_root),
                "INTERNVL3_78B_MODEL": "/dry-run/locked-internvl3-78b",
            }
            completed = subprocess.run(
                ["bash", str(script), "--dry-run"],
                cwd=self.repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            faq = subprocess.run(
                ["bash", str(script), "--faq"],
                cwd=self.repository,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(output_root.exists())
        self.assertIn("full-1009", completed.stdout)
        self.assertIn("spbench-si-result.md", completed.stdout)
        self.assertIn("no GPU/service/inference/scoring/report action", completed.stdout)
        self.assertIn("Q: 迁移到四卡服务器后第一条命令是什么？", faq.stdout)
        self.assertIn("不会", faq.stdout)

    def test_watcher_only_reports_terminal_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.tsv").write_text(
                "run_id\tphase\tlane\tprofile\tstate\ttimestamp\tdetail\n"
                "run\tA\tapi\tgpt5\tSTART\tt\t-\n"
                "run\tA\tapi\tgpt5\tPASS\tt\tvalidated\n"
                "run\tA\tapi\t-\tCOMPLETE\tt\tpassed\n", encoding="utf-8"
            )
            messages, _cursor, complete = scheduled_watcher._events(root / "status.tsv", lane="api", run_id="run", start_line=0)
        self.assertTrue(complete)
        self.assertEqual(len(messages), 2)
        self.assertFalse(any("START" in message for message in messages))

    def test_port_release_requires_actual_bindability(self):
        fake_socket = MagicMock()
        fake_socket.__enter__.return_value = fake_socket
        fake_socket.bind.side_effect = OSError(98, "Address already in use")
        with patch.object(socket, "socket", return_value=fake_socket):
            self.assertFalse(scheduled_batch._port_is_available(18100))

    def test_service_readiness_uses_listener_probe(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(
            scheduled_batch, "_port_is_listening", side_effect=[False, True]
        ), patch.object(scheduled_batch.time, "sleep") as sleep:
            scheduled_batch._wait_for_service(18100, process, timeout=10)
        sleep.assert_called_once_with(2)

    def test_cleanup_failure_is_recorded_before_lane_returns(self):
        job = scheduled_batch.jobs_for_lane("dual")[0]
        process = MagicMock()
        process.pid = 12345
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "status.tsv").write_text(
                "run_id\tphase\tlane\tprofile\tstate\ttimestamp\tdetail\n",
                encoding="utf-8",
            )
            with patch.object(scheduled_batch.subprocess, "Popen", return_value=process), patch.object(
                scheduled_batch, "_port_is_available", return_value=True
            ), patch.object(scheduled_batch, "_wait_for_service"), patch.object(
                scheduled_batch, "_run_owned", return_value=0
            ), patch.object(scheduled_batch, "_stop_owned_group"), patch.object(
                scheduled_batch, "_wait_for_port_release", side_effect=ResourceWarning("still bound")
            ), patch.object(scheduled_batch, "_signal_watcher"):
                status = scheduled_batch._run_job(
                    job, stage="test", repository_root=self.repository,
                    control_root=root, run_id="run",
                )
            events = (root / "status.tsv").read_text(encoding="utf-8")
        self.assertEqual(status, 1)
        self.assertIn("\tFAIL\t", events)
        self.assertIn("still bound", events)


if __name__ == "__main__":
    unittest.main()
