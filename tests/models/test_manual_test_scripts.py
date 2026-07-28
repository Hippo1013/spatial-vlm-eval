import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ManualStageScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[2]
        cls.scripts = {
            stage: cls.repository / "scripts" / "msmu" / f"run_manual_stage{stage}.sh"
            for stage in (1, 2, 3)
        }

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary = Path(self.temporary_directory.name)
        self.output_root = temporary / "manual-output"
        self.server_env = temporary / "server.env"
        values = {
            "REPO_ROOT": self.repository,
            "DATASET_ROOT": temporary / "dataset",
            "OUTPUT_ROOT": temporary / "outputs",
            "MANUAL_TEST_OUTPUT_ROOT": self.output_root,
            "QWEN_BASE_MODEL": temporary / "qwen",
            "QWEN_BASE_REVISION": "qwen-revision",
            "QWEN_PEFT_CHECKPOINT": temporary / "run-a" / "checkpoint-100",
            "LLAVA_MISTRAL_7B_MODEL": temporary / "llava-mistral",
            "LLAVA_YI_34B_MODEL": temporary / "llava-yi",
            "INTERNVL3_8B_MODEL": temporary / "internvl-8b",
            "INTERNVL3_38B_MODEL": temporary / "internvl-38b",
            "INTERNVL3_78B_MODEL": temporary / "internvl-78b",
            "SPATIALRGPT_MODEL": temporary / "spatialrgpt",
            "THREEDTHINKER_MODEL": temporary / "3dthinker",
            "SPATIALBOT_MODEL": temporary / "spatialbot",
            "JUDGE_MODEL": temporary / "judge",
        }
        self.server_env.write_text(
            "".join(f'{key}="{value}"\n' for key, value in values.items()),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_stage(self, stage, *arguments, extra_environment=None):
        environment = dict(os.environ)
        for key in [
            "INDICES",
            "LIMIT",
            "MANUAL_API_BACKEND",
            "MANUAL_CUDA_VISIBLE_DEVICES",
            "MANUAL_JUDGE_BASE_URL",
            "MANUAL_JUDGE_CUDA_VISIBLE_DEVICES",
            "MANUAL_RUN_SLUG",
            "MSMU_SMOKE_INDICES",
            "RESOLVE_PATHS_ONLY",
            "SCORE_ONLY",
        ]:
            environment.pop(key, None)
        environment.update(
            {
                "MSMU_SERVER_ENV": str(self.server_env),
                "MANUAL_DRY_RUN": "1",
            }
        )
        if extra_environment:
            environment.update(extra_environment)
        return subprocess.run(
            ["bash", str(self.scripts[stage]), *arguments],
            cwd=self.repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_help_and_model_list_do_not_require_server_env(self):
        environment = dict(os.environ)
        environment["MSMU_SERVER_ENV"] = "/missing/server.env"
        help_result = subprocess.run(
            ["bash", str(self.scripts[1]), "--help"],
            cwd=self.repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        list_result = subprocess.run(
            ["bash", str(self.scripts[3]), "--list"],
            cwd=self.repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertEqual(list_result.returncode, 0, list_result.stderr)
        self.assertIn("MANUAL_DRY_RUN=1", help_result.stdout)
        self.assertIn("internvl3_78b", list_result.stdout)
        self.assertIn("judge", list_result.stdout)

    def test_stage1_vllm_has_separate_serve_and_canary_actions(self):
        serve = self.run_stage(1, "llava_next_mistral_7b", "serve")
        check = self.run_stage(1, "llava_next_mistral_7b", "check")
        self.assertEqual(serve.returncode, 0, serve.stderr)
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertIn("preflight_vllm_processor.sh", serve.stdout)
        self.assertIn("serve_llava_next.sh", serve.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", serve.stdout)
        self.assertIn("canary_vllm_vision.sh", check.stdout)
        self.assertIn("vision_canary.json", check.stdout)

    def test_stage1_direct_model_is_one_sample_and_never_scores(self):
        result = self.run_stage(1, "qwen25_vl_base")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gpu_preflight.sh", result.stdout)
        self.assertIn("LIMIT=1", result.stdout)
        self.assertIn("RUN_SCORE=0", result.stdout)
        self.assertIn("SCORE_ONLY=0", result.stdout)
        self.assertIn("01_canary/qwen25-vl-base", result.stdout)

    def test_stage1_api_uses_two_samples_and_backend_specific_slug(self):
        result = self.run_stage(
            1,
            "gpt5",
            extra_environment={"MANUAL_API_BACKEND": "openai"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LIMIT=2", result.stdout)
        self.assertIn("BACKEND=openai", result.stdout)
        self.assertIn("01_canary/gpt5-openai", result.stdout)

    def test_stage2_selects_benchmark_owned_indices_and_cannot_score(self):
        result = self.run_stage(2, "ssr_native")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("select_smoke_indices.sh", result.stdout)
        self.assertIn("selected_indices.json", result.stdout)
        self.assertIn("INDICES=benchmark-selected-8-indices", result.stdout)
        self.assertIn("RUN_SCORE=0", result.stdout)
        self.assertIn("02_smoke8/ssr-native", result.stdout)

    def test_stage3_separates_inference_from_score_only(self):
        inference = self.run_stage(3, "qwen25_vl_base", "infer")
        score = self.run_stage(3, "qwen25_vl_base", "score")
        self.assertEqual(inference.returncode, 0, inference.stderr)
        self.assertEqual(score.returncode, 0, score.stderr)
        self.assertIn("gpu_preflight.sh", inference.stdout)
        self.assertIn("RUN_SCORE=0", inference.stdout)
        self.assertIn("SCORE_ONLY=0", inference.stdout)
        self.assertNotIn("gpu_preflight.sh", score.stdout)
        self.assertIn("RUN_SCORE=1", score.stdout)
        self.assertIn("SCORE_ONLY=1", score.stdout)
        self.assertIn("03_full987/qwen25-vl-base", score.stdout)
        self.assertNotIn("LIMIT=", score.stdout)
        self.assertNotIn("INDICES=", score.stdout)

    def test_stage3_judge_runs_preflight_then_separate_port(self):
        result = self.run_stage(3, "judge", "serve")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gpu_preflight.sh", result.stdout)
        self.assertIn("serve_local_judge.sh", result.stdout)
        self.assertIn("PORT=18080", result.stdout)
        self.assertIn("03_full987/judge/vllm_serve.log", result.stdout)

    def test_internvl78_is_blocked_after_static_stage1(self):
        static_check = self.run_stage(1, "internvl3_78b", "check")
        smoke = self.run_stage(2, "internvl3_78b")
        full = self.run_stage(3, "internvl3_78b", "infer")
        self.assertEqual(static_check.returncode, 0, static_check.stderr)
        self.assertIn("DRY_RUN=1", static_check.stdout)
        self.assertEqual(smoke.returncode, 4)
        self.assertEqual(full.returncode, 4)
        self.assertIn("not approved", smoke.stderr)
        self.assertIn("not approved", full.stderr)

    def test_peft_slug_includes_parent_and_checkpoint_name(self):
        result = self.run_stage(1, "qwen25_vl_peft")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("qwen25-vl-peft-run-a-checkpoint-100", result.stdout)
        self.assertIn("CHECKPOINT=", result.stdout)

    def test_every_listed_tested_model_has_commands_for_each_approved_stage(self):
        vllm_models = {
            "llava_next_mistral_7b",
            "llava_next_yi_34b",
            "internvl3_8b",
            "internvl3_38b",
        }
        direct_models = {
            "gpt5",
            "gemini31pro",
            "qwen25_vl_base",
            "qwen25_vl_peft",
            "ssr",
            "ssr_native",
            "spatialrgpt",
            "3dthinker",
            "3dthinker_native",
            "spatialbot",
            "spatialbot_native",
        }
        for model in sorted(vllm_models):
            with self.subTest(stage=1, model=model, action="serve"):
                self.assertEqual(self.run_stage(1, model, "serve").returncode, 0)
            with self.subTest(stage=1, model=model, action="check"):
                self.assertEqual(self.run_stage(1, model, "check").returncode, 0)
        for model in sorted(direct_models):
            with self.subTest(stage=1, model=model):
                self.assertEqual(self.run_stage(1, model).returncode, 0)
        for stage, action in ((2, "run"), (3, "infer"), (3, "score")):
            for model in sorted(vllm_models | direct_models):
                with self.subTest(stage=stage, model=model, action=action):
                    self.assertEqual(self.run_stage(stage, model, action).returncode, 0)


class ScoreOnlyPathResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[2]

    def test_qwen_resolve_only_sets_output_without_running_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment.update(
                {
                    "BASE_MODEL": "/locked/qwen",
                    "BASE_MODEL_REVISION": "locked-revision",
                    "DATASET_ROOT": "/locked/msmu",
                    "OUTPUT_ROOT": directory,
                    "RUN_NAME": "03_full987/qwen",
                    "RESOLVE_PATHS_ONLY": "1",
                }
            )
            script = self.repository / "scripts" / "msmu" / "infer_qwen_peft.sh"
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; printf "output=%s\\n" "$OUTPUT"',
                    "bash",
                    str(script),
                ],
                cwd=self.repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("03_full987/qwen/locked-revision", completed.stdout)
            self.assertIn("predictions.jsonl", completed.stdout)
            self.assertNotIn("msmu-infer", completed.stdout)


if __name__ == "__main__":
    unittest.main()
