import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


class VLLMLaunchConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[2]
        cls.script = cls.repository / "scripts" / "msmu" / "serve_vllm_profile.sh"

    def dry_run(self, profile):
        environment = dict(os.environ)
        for key in ["MODEL_REVISION", "TENSOR_PARALLEL_SIZE", "DTYPE", "SERVED_MODEL_NAME"]:
            environment.pop(key, None)
        environment.update({"PROFILE": profile, "MODEL_PATH": "/locked/snapshot", "DRY_RUN": "1"})
        return subprocess.run(
            ["bash", str(self.script)],
            cwd=self.repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_each_profile_has_locked_revision_tp_and_one_image_limit(self):
        expected = {
            "llava_next_mistral_7b": ("2424fdd47412fccc66d91719126b420e9fbd7065", "1"),
            "llava_next_yi_34b": ("84e4488fffae48f9da316ec31288b7c03f102ec7", "2"),
            "internvl3_8b": ("259a3b64a14623c0ec91a045cb43f7c5af5fa6af", "1"),
            "internvl3_38b": ("b2a05c0c325235f7530d8274c313a1d01082e069", "2"),
            "internvl3_78b": ("3aecc2b26fd0ea29ea9f41e0ecaf877a1351f356", "2"),
        }
        for profile, (revision, tensor_parallel) in expected.items():
            with self.subTest(profile=profile):
                output = self.dry_run(profile)
                self.assertIn(f"--revision {revision}", output)
                self.assertIn(f"--tensor-parallel-size {tensor_parallel}", output)
                self.assertIn("--dtype bfloat16", output)
                self.assertIn("--limit-mm-per-prompt.image 1", output)

    def test_internvl_78b_cannot_be_forced_without_dry_run(self):
        environment = dict(os.environ)
        for key in ["DRY_RUN", "MODEL_REVISION"]:
            environment.pop(key, None)
        environment.update({"PROFILE": "internvl3_78b", "MODEL_PATH": "/locked/snapshot"})
        completed = subprocess.run(
            ["bash", str(self.script)],
            cwd=self.repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 4)
        self.assertIn("not approved", completed.stderr)

    def test_pipeline_uses_a_separate_required_judge_endpoint(self):
        pipeline = (self.repository / "scripts" / "msmu" / "_run_model_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("JUDGE_BASE_URL:?", pipeline)
        self.assertIn('BASE_URL="${JUDGE_BASE_URL}"', pipeline)

    def test_documented_msmu_shell_paths_exist(self):
        documents = [
            self.repository / "README.md",
            self.repository / "docs" / "msmu-inference.md",
        ]
        references: set[str] = set()
        for document in documents:
            references.update(
                re.findall(
                    r"scripts/msmu/[A-Za-z0-9_.-]+\.sh",
                    document.read_text(encoding="utf-8"),
                )
            )
        self.assertTrue(references)
        missing = [reference for reference in sorted(references) if not (self.repository / reference).is_file()]
        self.assertEqual(missing, [])


class GPUPreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[2]
        cls.script = cls.repository / "scripts" / "msmu" / "gpu_preflight.sh"

    def run_preflight(
        self,
        *,
        free_mib: int,
        utilization: int,
        compute_pids: str,
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "nvidia-smi"
            fake.write_text(
                """#!/usr/bin/env bash
case "$*" in
  *--query-gpu=memory.free*) printf '%s\\n' "${FAKE_FREE_MIB}" ;;
  *--query-gpu=utilization.gpu*) printf '%s\\n' "${FAKE_UTILIZATION}" ;;
  *--query-compute-apps=pid*) printf '%s' "${FAKE_COMPUTE_PIDS}" ;;
  *) exit 2 ;;
esac
""",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{directory}:{environment['PATH']}",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "MIN_FREE_GPU_MIB": "30000",
                    "FAKE_FREE_MIB": str(free_mib),
                    "FAKE_UTILIZATION": str(utilization),
                    "FAKE_COMPUTE_PIDS": compute_pids,
                }
            )
            if extra_environment:
                environment.update(extra_environment)
            return subprocess.run(
                ["bash", str(self.script)],
                cwd=self.repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

    def test_idle_gpu_with_enough_memory_passes(self):
        completed = self.run_preflight(free_mib=40000, utilization=0, compute_pids="")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("compute_processes=0", completed.stdout)

    def test_existing_compute_process_fails_without_termination(self):
        completed = self.run_preflight(
            free_mib=40000,
            utilization=100,
            compute_pids="1234\n",
        )
        self.assertEqual(completed.returncode, 4)
        self.assertIn("already has compute process", completed.stderr)
        self.assertIn("left untouched", completed.stderr)

    def test_busy_gpu_requires_an_explicit_override(self):
        blocked = self.run_preflight(free_mib=40000, utilization=90, compute_pids="")
        self.assertEqual(blocked.returncode, 4)
        allowed = self.run_preflight(
            free_mib=40000,
            utilization=90,
            compute_pids="",
            extra_environment={"MAX_GPU_UTILIZATION_PERCENT": "100"},
        )
        self.assertEqual(allowed.returncode, 0, allowed.stderr)


class ManualTestPreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[2]
        cls.script = cls.repository / "scripts" / "msmu" / "prepare_manual_test.sh"

    def test_script_must_be_sourced(self):
        completed = subprocess.run(
            ["bash", str(self.script)],
            cwd=self.repository,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("must be sourced", completed.stderr)

    def test_sourcing_loads_config_and_creates_three_stage_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            base_output = temporary / "outputs"
            server_env = temporary / "server.env"
            server_env.write_text(
                "\n".join(
                    [
                        f"REPO_ROOT={self.repository}",
                        "DATASET_ROOT=/locked/msmu",
                        f"OUTPUT_ROOT={base_output}",
                        "BASE_MODEL=/locked/qwen-vl",
                        "BASE_MODEL_REVISION=locked-qwen-revision",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["MSMU_SERVER_ENV"] = str(server_env)
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        'source "$1" || exit $?; '
                        'printf "resolved=%s\\npwd=%s\\nqwen=%s@%s\\n" '
                        '"$OUTPUT_ROOT" "$PWD" "$QWEN_BASE_MODEL" "$QWEN_BASE_REVISION"'
                    ),
                    "bash",
                    str(self.script),
                ],
                cwd=temporary,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            manual_output = base_output / "manual-three-stage-v1"
            self.assertIn(f"resolved={manual_output}", completed.stdout)
            self.assertIn(f"pwd={self.repository}", completed.stdout)
            self.assertIn("qwen=/locked/qwen-vl@locked-qwen-revision", completed.stdout)
            for stage in ["01_canary", "02_smoke8", "03_full987"]:
                self.assertTrue((manual_output / stage).is_dir())


if __name__ == "__main__":
    unittest.main()
