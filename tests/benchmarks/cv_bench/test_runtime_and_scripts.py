from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.cv_bench.profiles import PROFILES
from spatial_vlm_eval.models.common.runtime import (
    GenerationResult,
    InferenceAdapter,
    run_recoverable_inference,
)
from spatial_vlm_eval.models.openai_compatible.client import APIRequestError


@dataclass(frozen=True)
class _Input:
    index: int
    image: object
    question: str


class _Contract:
    def __init__(self, root):
        self.dataset_root = Path(root)
        self.dataset_fingerprint = "fake-dataset"

    def __len__(self):
        return 1

    def model_input(self, index):
        return _Input(index, Image.new("RGB", (4, 4)), "question")

    def model_inputs(self, indices):
        return [self.model_input(index) for index in indices]

    def prediction_row(self, index, prediction):
        return {"index": index, "raw_prediction": prediction}


class _RetryAdapter(InferenceAdapter):
    supports_concurrency = False

    def __init__(self, *, fail_first):
        self.calls = 0
        self.fail_first = fail_first

    def metadata(self):
        return {
            "model": "fake",
            "model_revision": "rev",
            "backend": "openrouter",
            "profile": "fake",
            "inference_protocol": "protocol",
            "chat_template": "template",
            "image_processing": {"image_count": 1},
            "decoding": {"temperature": 0},
            "upstream": {},
        }

    def generate(self, model_input):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise APIRequestError("rate limited", status_code=429)
        return GenerationResult("A", metadata={"num_media_prompt": 1})


class _NonRetryableAdapter(_RetryAdapter):
    def generate(self, model_input):
        self.calls += 1
        raise APIRequestError("bad request", status_code=400)


class _PaidCompletionVerificationFailure(_RetryAdapter):
    def generate(self, model_input):
        self.calls += 1
        raise APIRequestError("metadata verification failed", retryable=False)


class CVBenchRuntimeAndScriptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[3]

    def test_public_inference_list_matches_23_profile_registry(self):
        completed = subprocess.run(
            ["bash", "scripts/cv_bench/run_inference.sh", "--list"],
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in completed.stdout.splitlines() if line]
        self.assertEqual(len(lines), 23)
        self.assertTrue(lines[0].startswith("llava_next_mistral_7b\t"))
        self.assertTrue(lines[-1].startswith("spatialladder3b_thinking\t"))

    def test_full_serial_controller_is_registry_driven_and_owns_vllm(self):
        script = self.repository / "scripts" / "cv_bench" / "run_full_serial.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("PROFILE_SEQUENCE", text)
        self.assertIn("--without-internvl78", text)
        self.assertIn("--skip-completed", text)
        self.assertIn("validate_predictions", text)
        self.assertIn("SKIP_COMPLETE", text)
        self.assertIn("serve_vllm_profile.sh", text)
        self.assertIn("_probe_openai_models.py", text)
        self.assertIn("flock -n", text)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1", text)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", text)
        self.assertNotIn("llava_next_mistral_7b,llava_next_yi_34b", text)

    def test_internvl78_single_model_controller_uses_canonical_outputs(self):
        script = (
            self.repository
            / "scripts"
            / "cv_bench"
            / "run_internvl3_78b_evaluation.sh"
        )
        text = script.read_text(encoding="utf-8")
        for required in [
            'PROFILE="internvl3_78b"',
            "track_directory",
            "--stage test",
            "--stage full",
            "--predictions",
            "build_results_report.sh",
            '_single_model_evaluation',
            'CVBENCH_INTERNVL3_78B_GPU_IDS',
        ]:
            with self.subTest(required=required):
                self.assertIn(required, text)
        self.assertNotIn(
            'control_root="${CVBENCH_OUTPUT_ROOT%/}/_internvl3_78b_evaluation"',
            text,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "cv-bench-outputs"
            completed = subprocess.run(
                ["bash", str(script), "--dry-run"],
                cwd=self.repository,
                env={
                    **os.environ,
                    "CVBENCH_ENV_FILE": "/dev/null",
                    "CVBENCH_PYTHON": sys.executable,
                    "CVBENCH_OUTPUT_ROOT": str(output_root),
                    "INTERNVL3_78B_MODEL": "/dry-run/locked-internvl3-78b",
                },
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(output_root.exists())
        self.assertIn('"stage": "full"', completed.stdout)
        self.assertIn("cv-bench-result.md", completed.stdout)

    def test_dry_run_preserves_registry_order_for_unsorted_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    "bash",
                    "scripts/cv_bench/run_inference.sh",
                    "--stage",
                    "full",
                    "--models",
                    "qwen3_vl_8b,llava_next_mistral_7b",
                    "--output-root",
                    directory,
                    "--dry-run",
                ],
                cwd=self.repository,
                check=True,
                capture_output=True,
                text=True,
            )
        lines = completed.stdout.splitlines()
        self.assertIn('"profile": "llava_next_mistral_7b"', lines[0])
        self.assertIn('"profile": "qwen3_vl_8b"', lines[1])

    def test_all_cvbench_shell_scripts_are_syntax_valid(self):
        for path in sorted((self.repository / "scripts" / "cv_bench").glob("*.sh")):
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_live_prediction_watcher_is_read_only_and_filters_debug_journals(self):
        script = self.repository / "scripts" / "cv_bench" / "watch_live_predictions.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn("CVBENCH_OUTPUT_ROOT", text)
        self.assertIn("status.tsv", text)
        self.assertIn("*.journal.jsonl", text)
        self.assertIn('"test_runs" not in path.parts', text)
        self.assertIn('"failed_attempts" not in path.parts', text)
        self.assertIn("--from-start", text)
        self.assertIn("--lane", text)
        self.assertIn('output_root / "_dual_lane" / lane', text)
        self.assertNotIn("tmux send-keys", text)
        self.assertNotIn("kill ", text)

        completed = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=self.repository,
            env={**os.environ, "CVBENCH_ENV_FILE": "/dev/null"},
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("read-only", completed.stdout)

    def test_live_prediction_watcher_prints_only_new_current_events(self):
        script = self.repository / "scripts" / "cv_bench" / "watch_live_predictions.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "_serial_full"
            logs = control / "logs"
            journal = root / "runs" / "test_profile" / "revision" / "protocol" / (
                "predictions.jsonl.journal.jsonl"
            )
            logs.mkdir(parents=True)
            journal.parent.mkdir(parents=True)
            (control / "status.tsv").write_text(
                "run_id\tprofile\tstate\ttimestamp\n"
                "test-run\ttest_profile\tSTART\t2026-08-04T00:00:00Z\n",
                encoding="utf-8",
            )
            journal.write_text(
                json.dumps(
                    {
                        "timestamp": "old",
                        "status": "success",
                        "index": 0,
                        "prediction": "OLD",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                ["bash", str(script)],
                cwd=self.repository,
                env={
                    **os.environ,
                    "CVBENCH_ENV_FILE": "/dev/null",
                    "CVBENCH_OUTPUT_ROOT": str(root),
                    "CVBENCH_PYTHON": sys.executable,
                    "CVBENCH_WATCH_POLL_SECONDS": "1",
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                time.sleep(1.2)
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "timestamp": "new",
                                "status": "success",
                                "index": 1,
                                "prediction": "NEW",
                            }
                        )
                        + "\n"
                    )
                time.sleep(1.2)
                (logs / "test-run.controller.log").write_text(
                    "[cv-bench-full-serial] COMPLETE\n", encoding="utf-8"
                )
                stdout, stderr = process.communicate(timeout=5)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn("index=1", stdout)
        self.assertIn("NEW", stdout)
        self.assertNotIn("index=0", stdout)
        self.assertNotIn("OLD", stdout)
        self.assertIn("serial inference COMPLETE", stdout)

    def test_live_prediction_watcher_follows_one_dual_lane(self):
        script = self.repository / "scripts" / "cv_bench" / "watch_live_predictions.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "_dual_lane" / "gpu1"
            journal = root / "runs" / "lane_profile" / "revision" / "protocol" / (
                "predictions.jsonl.journal.jsonl"
            )
            control.mkdir(parents=True)
            journal.parent.mkdir(parents=True)
            status = control / "status.tsv"
            status.write_text(
                "run_id\tprofile\tstate\ttimestamp\tdetail\n"
                "lane-run\tlane_profile\tSTART\t2026-08-05T00:00:00Z\tgpu=1\n",
                encoding="utf-8",
            )
            journal.write_text(
                json.dumps(
                    {
                        "timestamp": "old",
                        "status": "success",
                        "index": 0,
                        "prediction": "OLD",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                ["bash", str(script), "--lane", "gpu1"],
                cwd=self.repository,
                env={
                    **os.environ,
                    "CVBENCH_ENV_FILE": "/dev/null",
                    "CVBENCH_OUTPUT_ROOT": str(root),
                    "CVBENCH_PYTHON": sys.executable,
                    "CVBENCH_WATCH_POLL_SECONDS": "1",
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                time.sleep(1.2)
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "timestamp": "new",
                                "status": "success",
                                "index": 1,
                                "prediction": "NEW",
                            }
                        )
                        + "\n"
                    )
                time.sleep(1.2)
                with status.open("a", encoding="utf-8") as handle:
                    handle.write(
                        "lane-run\tlane_profile\tPASS\t2026-08-05T00:01:00Z\tvalidated\n"
                        "lane-run\t-\tCOMPLETE\t2026-08-05T00:01:01Z\tgpu=1\n"
                    )
                stdout, stderr = process.communicate(timeout=5)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn("lane=gpu1 active model: lane_profile", stdout)
        self.assertIn("index=1", stdout)
        self.assertIn("NEW", stdout)
        self.assertNotIn("index=0", stdout)
        self.assertNotIn("OLD", stdout)
        self.assertIn("CV-Bench lane=gpu1 COMPLETE", stdout)

    def test_live_prediction_watcher_rejects_unknown_lane(self):
        script = self.repository / "scripts" / "cv_bench" / "watch_live_predictions.sh"
        completed = subprocess.run(
            ["bash", str(script), "--lane", "gpu2"],
            cwd=self.repository,
            env={**os.environ, "CVBENCH_ENV_FILE": "/dev/null"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("gpu0 or gpu1", completed.stderr)

    def test_vllm_launcher_uses_registry_revision_tp_and_served_name(self):
        script = self.repository / "scripts" / "cv_bench" / "serve_vllm_profile.sh"
        for profile in [value for value in PROFILES.values() if value.group == "general_open"]:
            with self.subTest(profile=profile.key):
                environment = dict(os.environ)
                environment.update(
                    {
                        "CVBENCH_ENV_FILE": "/dev/null",
                        "CVBENCH_PYTHON": sys.executable,
                        "CVBENCH_VLLM": "/locked/vllm",
                        profile.model_path_env: "/locked/model",
                    }
                )
                gpu_ids = ",".join(
                    str(index) for index in range(profile.default_tensor_parallel_size)
                )
                completed = subprocess.run(
                    [
                        "bash",
                        str(script),
                        "--model",
                        profile.key,
                        "--gpu-ids",
                        gpu_ids,
                        "--port",
                        "18101",
                        "--dry-run",
                    ],
                    cwd=self.repository,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertIn(f"--revision {profile.revision}", completed.stdout)
                self.assertIn(
                    f"--served-model-name {profile.served_model_name}", completed.stdout
                )
                self.assertIn(
                    f"--tensor-parallel-size {profile.default_tensor_parallel_size}",
                    completed.stdout,
                )
                self.assertIn("--limit-mm-per-prompt.image 1", completed.stdout)

    def test_retry_backoff_and_resume_do_not_repeat_successful_paid_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.jsonl"
            first = _RetryAdapter(fail_first=True)
            with patch("spatial_vlm_eval.models.common.runtime.time.sleep") as sleep:
                run_recoverable_inference(
                    contract=_Contract(root),
                    adapter=first,
                    output=output,
                    target_indices=[0],
                    benchmark="CV-Bench",
                    split="test",
                    official_size=1,
                    scorer_protocol="scorer",
                    retries=1,
                )
            self.assertEqual(first.calls, 2)
            sleep.assert_called_once_with(1.0)

            resumed = _RetryAdapter(fail_first=False)
            run_recoverable_inference(
                contract=_Contract(root),
                adapter=resumed,
                output=output,
                target_indices=[0],
                benchmark="CV-Bench",
                split="test",
                official_size=1,
                scorer_protocol="scorer",
                retries=1,
            )
            self.assertEqual(resumed.calls, 0)

    def test_non_retryable_http_error_is_not_reissued(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = _NonRetryableAdapter(fail_first=False)
            with self.assertRaisesRegex(RuntimeError, "Inference incomplete"):
                run_recoverable_inference(
                    contract=_Contract(directory),
                    adapter=adapter,
                    output=Path(directory) / "predictions.jsonl",
                    target_indices=[0],
                    benchmark="CV-Bench",
                    split="test",
                    official_size=1,
                    scorer_protocol="scorer",
                    retries=3,
                )
            self.assertEqual(adapter.calls, 1)

    def test_paid_completion_verification_failure_is_not_reissued(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = _PaidCompletionVerificationFailure(fail_first=False)
            with self.assertRaisesRegex(RuntimeError, "Inference incomplete"):
                run_recoverable_inference(
                    contract=_Contract(directory),
                    adapter=adapter,
                    output=Path(directory) / "predictions.jsonl",
                    target_indices=[0],
                    benchmark="CV-Bench",
                    split="test",
                    official_size=1,
                    scorer_protocol="scorer",
                    retries=3,
                    retry_missing_passes=1,
                )
            self.assertEqual(adapter.calls, 1)


if __name__ == "__main__":
    unittest.main()
