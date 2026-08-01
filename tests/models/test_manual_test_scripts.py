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
            "QWEN_32B_MODEL": temporary / "qwen-32b",
            "QWEN_32B_REVISION": "qwen-32b-revision",
            "QWEN_72B_MODEL": temporary / "qwen-72b",
            "QWEN_72B_REVISION": "qwen-72b-revision",
            "QWEN3_2B_MODEL": temporary / "qwen3-2b",
            "QWEN3_2B_REVISION": "qwen3-2b-revision",
            "QWEN3_4B_MODEL": temporary / "qwen3-4b",
            "QWEN3_4B_REVISION": "qwen3-4b-revision",
            "QWEN3_8B_MODEL": temporary / "qwen3-8b",
            "QWEN3_8B_REVISION": "qwen3-8b-revision",
            "QWEN3_32B_MODEL": temporary / "qwen3-32b",
            "QWEN3_32B_REVISION": "qwen3-32b-revision",
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
        self.assertIn("QWEN_VISION_CANARY=1", result.stdout)
        self.assertIn("01_canary/qwen25-vl-base", result.stdout)

    def test_large_qwen_profiles_use_distinct_models_and_device_maps(self):
        qwen32 = self.run_stage(1, "qwen25_vl_32b")
        qwen72 = self.run_stage(1, "qwen25_vl_72b")
        self.assertEqual(qwen32.returncode, 0, qwen32.stderr)
        self.assertEqual(qwen72.returncode, 0, qwen72.stderr)
        self.assertIn("PROFILE=qwen25_vl_32b", qwen32.stdout)
        self.assertIn("BASE_MODEL=", qwen32.stdout)
        self.assertIn("qwen-32b", qwen32.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", qwen32.stdout)
        self.assertIn("DEVICE_MAP=single", qwen32.stdout)
        self.assertIn("MIN_FREE_GPU_MIB=75000", qwen32.stdout)
        self.assertIn("PROFILE=qwen25_vl_72b", qwen72.stdout)
        self.assertIn("qwen-72b", qwen72.stdout)
        self.assertIn("CUDA_VISIBLE_DEVICES=0\\,1", qwen72.stdout)
        self.assertIn("DEVICE_MAP=balanced", qwen72.stdout)

    def test_qwen3_supplement_uses_four_distinct_single_gpu_profiles(self):
        for size in ("2b", "4b", "8b", "32b"):
            with self.subTest(size=size):
                result = self.run_stage(1, f"qwen3_vl_{size}")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"PROFILE=qwen3_vl_{size}", result.stdout)
                self.assertIn(f"qwen3-{size}", result.stdout)
                self.assertIn("CUDA_VISIBLE_DEVICES=0", result.stdout)
                self.assertIn("DEVICE_MAP=single", result.stdout)
                self.assertIn("run_qwen_peft_pipeline.sh", result.stdout)
                self.assertIn("QWEN_VISION_CANARY=1", result.stdout)
                if size == "32b":
                    self.assertIn("BATCH_SIZE=1", result.stdout)

    def test_qwen_stage2_does_not_repeat_the_stage1_vision_canary(self):
        result = self.run_stage(2, "qwen3_vl_4b")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("QWEN_VISION_CANARY", result.stdout)

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

    def test_internvl78_uses_four_gpus_across_manual_stages(self):
        stage1_serve = self.run_stage(1, "internvl3_78b", "serve")
        stage1_check = self.run_stage(1, "internvl3_78b", "check")
        stage2_serve = self.run_stage(2, "internvl3_78b", "serve")
        smoke = self.run_stage(2, "internvl3_78b")
        stage3_serve = self.run_stage(3, "internvl3_78b", "serve")
        full = self.run_stage(3, "internvl3_78b", "infer")

        stage_results = [stage1_serve, stage1_check, stage2_serve, smoke, stage3_serve, full]
        for result in stage_results:
            self.assertEqual(result.returncode, 0, result.stderr)
        for result in [stage1_serve, stage2_serve, stage3_serve]:
            self.assertIn("CUDA_VISIBLE_DEVICES=0\\,1\\,2\\,3", result.stdout)
            self.assertIn("serve_internvl3.sh", result.stdout)
        self.assertIn("canary_vllm_vision.sh", stage1_check.stdout)
        self.assertIn("02_smoke8/internvl3-78b-vllm", smoke.stdout)
        self.assertIn("03_full987/internvl3-78b-vllm", full.stdout)
        self.assertIn("RUN_SCORE=0", full.stdout)

    def test_qwen72_stage3_is_excluded_but_earlier_stages_remain_available(self):
        canary = self.run_stage(1, "qwen25_vl_72b")
        smoke = self.run_stage(2, "qwen25_vl_72b")
        full = self.run_stage(3, "qwen25_vl_72b", "infer")
        self.assertEqual(canary.returncode, 0, canary.stderr)
        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertEqual(full.returncode, 4)
        self.assertIn("70B+", full.stderr)

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
            "internvl3_78b",
        }
        direct_models = {
            "gpt5",
            "gemini31pro",
            "qwen25_vl_base",
            "qwen25_vl_32b",
            "qwen25_vl_peft",
            "qwen3_vl_2b",
            "qwen3_vl_4b",
            "qwen3_vl_8b",
            "qwen3_vl_32b",
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


class SerialStage3InferenceScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[2]
        cls.script = cls.repository / "scripts" / "msmu" / "run_stage3_serial_inference.sh"

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
            "QWEN_32B_MODEL": temporary / "qwen-32b",
            "QWEN_32B_REVISION": "qwen-32b-revision",
            "QWEN3_2B_MODEL": temporary / "qwen3-2b",
            "QWEN3_2B_REVISION": "qwen3-2b-revision",
            "QWEN3_4B_MODEL": temporary / "qwen3-4b",
            "QWEN3_4B_REVISION": "qwen3-4b-revision",
            "QWEN3_8B_MODEL": temporary / "qwen3-8b",
            "QWEN3_8B_REVISION": "qwen3-8b-revision",
            "QWEN3_32B_MODEL": temporary / "qwen3-32b",
            "QWEN3_32B_REVISION": "qwen3-32b-revision",
            "LLAVA_MISTRAL_7B_MODEL": temporary / "llava-mistral",
            "LLAVA_YI_34B_MODEL": temporary / "llava-yi",
            "INTERNVL3_8B_MODEL": temporary / "internvl-8b",
            "INTERNVL3_38B_MODEL": temporary / "internvl-38b",
            "SPATIALRGPT_MODEL": temporary / "spatialrgpt",
            "THREEDTHINKER_MODEL": temporary / "3dthinker",
            "SPATIALBOT_MODEL": temporary / "spatialbot",
        }
        self.server_env.write_text(
            "".join(f'{key}="{value}"\n' for key, value in values.items()),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def run_batch(self, *arguments, extra_environment=None):
        environment = dict(os.environ)
        for key in [
            "BATCH_CONTINUE_ON_ERROR",
            "BATCH_MODEL_ATTEMPTS",
            "BATCH_SKIP_COMPLETED",
            "BATCH_STALL_TIMEOUT_SECONDS",
            "MANUAL_CUDA_VISIBLE_DEVICES",
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
            ["bash", str(self.script), *arguments],
            cwd=self.repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_plan_has_exactly_thirteen_tracks_and_explicit_exclusions(self):
        result = self.run_batch("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        included = [line for line in result.stdout.splitlines() if "\t" not in line]
        self.assertEqual(
            included,
            [
                "llava_next_mistral_7b",
                "llava_next_yi_34b",
                "internvl3_8b",
                "internvl3_38b",
                "qwen25_vl_base",
                "qwen25_vl_32b",
                "ssr",
                "ssr_native",
                "spatialrgpt",
                "3dthinker",
                "3dthinker_native",
                "spatialbot",
                "spatialbot_native",
            ],
        )
        for excluded in [
            "gpt5",
            "gemini31pro",
            "qwen25_vl_72b",
            "internvl3_78b",
            "qwen25_vl_peft",
        ]:
            self.assertIn(f"excluded\t{excluded}\t", result.stdout)
        self.assertIn(
            "excluded\tinternvl3_78b\tseparate four-GPU manual supplement",
            result.stdout,
        )

    def test_dry_run_dispatches_every_included_track_without_scoring(self):
        result = self.run_batch()
        self.assertEqual(result.returncode, 0, result.stderr)
        dispatched = [
            line.split("model=", 1)[1].split(" ", 1)[0]
            for line in result.stdout.splitlines()
            if line.startswith("[msmu-batch] dry-run: model=")
        ]
        self.assertEqual(len(dispatched), 13)
        self.assertEqual(len(set(dispatched)), 13)
        self.assertNotIn("qwen25_vl_72b", dispatched)
        self.assertNotIn("internvl3_78b", dispatched)
        self.assertNotIn("qwen25_vl_peft", dispatched)
        self.assertIn("RUN_SCORE=0", result.stdout)
        self.assertNotIn("RUN_SCORE=1", result.stdout)
        self.assertIn("no GPU/API/judge action was taken", result.stdout)

    def test_qwen3_selector_dispatches_only_four_supplement_tracks(self):
        listed = self.run_batch("--qwen3", "--list")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(
            listed.stdout.splitlines(),
            [
                "qwen3_vl_2b",
                "qwen3_vl_4b",
                "qwen3_vl_8b",
                "qwen3_vl_32b",
            ],
        )

        result = self.run_batch("--qwen3")
        self.assertEqual(result.returncode, 0, result.stderr)
        dispatched = [
            line.split("model=", 1)[1].split(" ", 1)[0]
            for line in result.stdout.splitlines()
            if line.startswith("[msmu-batch] dry-run: model=")
        ]
        self.assertEqual(dispatched, listed.stdout.splitlines())
        self.assertNotIn(" serve", result.stdout)
        self.assertIn("plan=qwen3", result.stdout)
        self.assertIn("BATCH_SIZE=1", result.stdout)
        self.assertIn("no GPU/API/judge action was taken", result.stdout)

    def test_qwen3_status_uses_state_separate_from_default_plan(self):
        legacy = self.run_batch("--status")
        qwen3 = self.run_batch("--qwen3", "--status")
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        self.assertEqual(qwen3.returncode, 0, qwen3.stderr)
        self.assertIn(f"state\t{self.output_root}/03_full987/_serial_inference\n", legacy.stdout)
        self.assertIn(
            f"state\t{self.output_root}/03_full987/_serial_inference/qwen3\n",
            qwen3.stdout,
        )
        self.assertEqual(qwen3.stdout.count("pending\tqwen3_vl_"), 4)

    def test_invalid_watchdog_timeout_fails_before_dispatch(self):
        result = self.run_batch(
            extra_environment={"BATCH_STALL_TIMEOUT_SECONDS": "0"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be greater than zero", result.stderr)

    def test_script_contains_owned_group_watchdog_lock_and_resume_guards(self):
        script = self.script.read_text(encoding="utf-8")
        self.assertIn("BATCH_STALL_TIMEOUT_SECONDS", script)
        self.assertIn("predictions.jsonl.journal.jsonl", script)
        self.assertIn('setsid env "${child_env_cleanup[@]}"', script)
        self.assertIn('kill -TERM -- "-${pid}"', script)
        self.assertIn('kill -KILL -- "-${pid}"', script)
        self.assertIn("flock -n 9", script)
        self.assertIn("port 18081 is already occupied", script)
        self.assertIn('sock.connect_ex(("127.0.0.1", 18081))', script)
        self.assertIn("active_process.env", script)
        self.assertIn("BATCH_SKIP_COMPLETED", script)
        self.assertIn("-u NO_RESUME", script)


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

    def test_generic_qwen_wrapper_changes_only_profile_model_and_revision(self):
        script = self.repository / "scripts" / "msmu" / "infer_qwen_peft.sh"
        profiles = {
            "qwen3_vl_2b": (
                "89644892e4d85e24eaac8bacfd4f463576704203",
                "msmu_qwen3_vl_2b_question_only_deterministic_v1",
            ),
            "qwen3_vl_4b": (
                "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "msmu_qwen3_vl_4b_question_only_deterministic_v1",
            ),
            "qwen3_vl_8b": (
                "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "msmu_qwen3_vl_8b_question_only_deterministic_v1",
            ),
            "qwen3_vl_32b": (
                "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
                "msmu_qwen3_vl_32b_question_only_deterministic_v1",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for profile, (revision, protocol) in profiles.items():
                with self.subTest(profile=profile):
                    environment = dict(os.environ)
                    for name in ["CHECKPOINT", "CHECKPOINT_REVISION"]:
                        environment.pop(name, None)
                    environment.update(
                        {
                            "PROFILE": profile,
                            "BASE_MODEL": f"/locked/{profile}",
                            "BASE_MODEL_REVISION": revision,
                            "DATASET_ROOT": "/locked/msmu",
                            "OUTPUT_ROOT": directory,
                            "RUN_NAME": f"03_full987/{profile}",
                            "RESOLVE_PATHS_ONLY": "1",
                        }
                    )
                    completed = subprocess.run(
                        [
                            "bash",
                            "-c",
                            'source "$1"; printf "%s\\n" "$OUTPUT"',
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
                    self.assertIn(revision, completed.stdout)
                    self.assertIn(protocol, completed.stdout)

    def test_qwen3_wrapper_rejects_stray_peft_checkpoint(self):
        environment = dict(os.environ)
        environment.update(
            {
                "PROFILE": "qwen3_vl_2b",
                "BASE_MODEL": "/locked/qwen3",
                "BASE_MODEL_REVISION": "89644892e4d85e24eaac8bacfd4f463576704203",
                "DATASET_ROOT": "/locked/msmu",
                "CHECKPOINT": "/wrong/peft",
                "RESOLVE_PATHS_ONLY": "1",
            }
        )
        completed = subprocess.run(
            ["bash", "scripts/msmu/infer_qwen_peft.sh"],
            cwd=self.repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("does not accept CHECKPOINT", completed.stderr)


if __name__ == "__main__":
    unittest.main()
