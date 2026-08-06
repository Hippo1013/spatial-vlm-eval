from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path
from urllib.parse import unquote

from spatial_vlm_eval.benchmarks.cv_bench.data import DATASET_REVISION as CVBENCH_DATASET_REVISION
from spatial_vlm_eval.benchmarks.cv_bench.profiles import (
    PROFILE_SEQUENCE as CVBENCH_PROFILE_SEQUENCE,
)
from spatial_vlm_eval.benchmarks.cv_bench.scorer import (
    SCORER_PROTOCOL as CVBENCH_SCORER_PROTOCOL,
)
from spatial_vlm_eval.benchmarks.msmu.scorer import SCORER_PROTOCOL as MSMU_SCORER_PROTOCOL
from spatial_vlm_eval.models.profiles import CURRENT_TARGET_PROFILE_KEYS, PROFILES


class DocumentationConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = Path(__file__).resolve().parents[1]
        cls.docs = cls.repository / "docs"

    def markdown_documents(self) -> list[Path]:
        documents = [
            self.repository / "AGENTS.md",
            self.repository / "CHANGELOG.md",
            self.repository / "README.md",
            self.repository / "benchmark_paper" / "README.md",
        ]
        documents.extend(sorted(self.docs.rglob("*.md")))
        return documents

    def test_relative_markdown_links_resolve(self) -> None:
        link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
        missing: list[str] = []
        for document in self.markdown_documents():
            for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
                target = raw_target.strip()
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                if not target or target.startswith("#"):
                    continue
                if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                    continue
                local_target = unquote(target.split("#", 1)[0])
                resolved = (document.parent / local_target).resolve()
                if not resolved.exists():
                    missing.append(
                        f"{document.relative_to(self.repository)} -> {raw_target}"
                    )
        self.assertEqual(missing, [])

    def test_docs_index_lists_every_markdown_document(self) -> None:
        index = self.docs / "README.md"
        index_text = index.read_text(encoding="utf-8")
        unlisted = [
            path.relative_to(self.docs).as_posix()
            for path in sorted(self.docs.rglob("*.md"))
            if path != index
            and f"]({path.relative_to(self.docs).as_posix()})" not in index_text
        ]
        self.assertEqual(unlisted, [])

    def test_knowledge_base_stays_layered_and_bounded(self) -> None:
        agents_path = self.repository / "AGENTS.md"
        agents = agents_path.read_text(encoding="utf-8")
        self.assertLessEqual(len(agents.splitlines()), 300)
        self.assertLessEqual(len(agents.encode("utf-8")), 15 * 1024)

        oversized = [
            path.relative_to(self.repository).as_posix()
            for path in sorted(self.docs.rglob("*.md"))
            if len(path.read_text(encoding="utf-8").splitlines()) > 1500
        ]
        self.assertEqual(oversized, [])

        index = (self.docs / "README.md").read_text(encoding="utf-8")
        for required in [
            "## 信息层级与按需读取",
            "Agent 记忆",
            "## 日志生命周期",
            "未跟踪运行产物",
            "不要新建 `DEVLOG.md`",
        ]:
            with self.subTest(required=required):
                self.assertIn(required, index)

        gitignore = (self.repository / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/tmp/", gitignore.splitlines())
        self.assertFalse((self.repository / "DEVLOG.md").exists())

    def test_model_matrix_matches_current_target_profiles(self) -> None:
        matrix = (self.docs / "model-matrix.md").read_text(encoding="utf-8")
        msmu_section = matrix.split(
            "## MSMU 当前 18 条已完成目标 inference profile", 1
        )[1].split("## 完成状态", 1)[0]
        documented = re.findall(r"^\| `([^`]+)` \|", msmu_section, flags=re.MULTILINE)
        self.assertEqual(documented, list(CURRENT_TARGET_PROFILE_KEYS))
        count = re.search(
            r"^## MSMU 当前 (\d+) 条已完成目标 inference profile$",
            matrix,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(count)
        self.assertEqual(int(count.group(1)), len(CURRENT_TARGET_PROFILE_KEYS))
        self.assertEqual(
            len(set(CURRENT_TARGET_PROFILE_KEYS)), len(CURRENT_TARGET_PROFILE_KEYS)
        )
        self.assertTrue(set(CURRENT_TARGET_PROFILE_KEYS) <= set(PROFILES))
        self.assertNotIn("gpt5", CURRENT_TARGET_PROFILE_KEYS)
        self.assertNotIn("gemini31pro", CURRENT_TARGET_PROFILE_KEYS)
        self.assertFalse(
            any(
                PROFILES[key].family == "qwen25_vl"
                for key in CURRENT_TARGET_PROFILE_KEYS
            )
        )

        cvbench_section = matrix.split(
            "## CV-Bench 当前 23 条目标 inference profile", 1
        )[1].split("## MSMU 当前 18 条已完成目标 inference profile", 1)[0]
        documented_cvbench = re.findall(
            r"^\| `([^`]+)` \|", cvbench_section, flags=re.MULTILINE
        )
        self.assertEqual(documented_cvbench, list(CVBENCH_PROFILE_SEQUENCE))

    def test_cross_benchmark_scope_and_planned_sota_models_are_explicit(self) -> None:
        scope = (self.docs / "evaluation-scope.md").read_text(encoding="utf-8")
        matrix = (self.docs / "model-matrix.md").read_text(encoding="utf-8")
        readme = (self.repository / "README.md").read_text(encoding="utf-8")

        for benchmark in [
            "MSMU-Bench",
            "CV-Bench",
            "Q-Spatial Bench",
            "SPBench-SI",
        ]:
            with self.subTest(benchmark=benchmark):
                self.assertIn(benchmark, scope)
                self.assertIn(benchmark, readme)

        for model in [
            "RoboBrain2.5-8B-NV",
            "RoboBrain2.5-8B-MT",
            "HiSpatial-3B",
            "SpatialLadder-3B",
        ]:
            with self.subTest(model=model):
                self.assertIn(model, scope)
                self.assertIn(model, matrix)

        self.assertIn("共 19 个模型身份", scope)
        self.assertIn("**下一项待定**", scope)
        self.assertIn("不是本项目复现结果", scope)
        self.assertIn("CV-Bench 的 23 条目标轨由独立 registry 维护", scope)
        self.assertIn("22 条轨正在进行 full-2638 串行推理，尚未评分", scope)
        self.assertIn("/media/datasets/tangzecong/huggingface/", scope)
        self.assertIn("/media/datasets/lihaoran/huggingface/", scope)

    def test_stage3_runbook_matches_serial_script_plan(self) -> None:
        runbook = (self.docs / "msmu-stage3-full-eval.md").read_text(encoding="utf-8")
        match = re.search(
            r"本轮阶段三固定运行 (\d+) 条本地推理轨：\s*```text\n(.*?)\n```",
            runbook,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        expected_count = int(match.group(1))
        documented = [
            value
            for value in re.split(r"[,\s]+", match.group(2).strip())
            if value
        ]

        completed = subprocess.run(
            ["bash", "scripts/msmu/run_stage3_serial_inference.sh", "--list"],
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )
        scripted = [
            line
            for line in completed.stdout.splitlines()
            if line and "\t" not in line
        ]
        self.assertEqual(len(documented), expected_count)
        self.assertEqual(documented, scripted)

    def test_canonical_scorer_protocol_is_documented(self) -> None:
        protocol = (
            self.docs / "benchmarks" / "msmu" / "protocol.md"
        ).read_text(encoding="utf-8")
        agents = (self.repository / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn(MSMU_SCORER_PROTOCOL, protocol)
        self.assertIn(MSMU_SCORER_PROTOCOL, agents)

        cvbench_protocol = (
            self.docs / "benchmarks" / "cv_bench" / "protocol.md"
        ).read_text(encoding="utf-8")
        self.assertIn(CVBENCH_SCORER_PROTOCOL, cvbench_protocol)
        self.assertIn(CVBENCH_DATASET_REVISION, cvbench_protocol)
        self.assertIn("Overall = (2D + 3D) / 2", cvbench_protocol)
        self.assertIn("不是旧脚本的逐字节复刻", cvbench_protocol)

    def test_cvbench_runbook_and_config_match_public_entrypoints(self) -> None:
        runbook = (self.docs / "cv-bench-two-stage-runbook.md").read_text(
            encoding="utf-8"
        )
        commands = (self.docs / "cv-bench-commands.md").read_text(encoding="utf-8")
        internvl78 = (
            self.docs / "cv-bench-internvl3-78b-evaluation.md"
        ).read_text(encoding="utf-8")
        config = (
            self.repository / "configs" / "cv-bench-server.env.example"
        ).read_text(encoding="utf-8")
        readme = (self.repository / "README.md").read_text(encoding="utf-8")
        for required in [
            "run_inference.sh --stage test",
            "run_inference.sh --stage full",
            "score_results.sh --predictions",
            "build_results_report.sh",
            "23/23",
        ]:
            with self.subTest(required=required):
                self.assertIn(required, runbook)
                self.assertIn(required, commands)
        self.assertIn("CVBENCH_OUTPUT_ROOT", config)
        self.assertIn("/media/datasets/lihaoran", config)
        self.assertIn("/media/datasets/tangzecong", config)
        self.assertIn("CVBENCH_OUTPUT_ROOT", readme)
        for document in (runbook, commands, internvl78, readme):
            with self.subTest(document=document[:40]):
                self.assertIn("run_internvl3_78b_evaluation.sh", document)
        self.assertIn("只评分 internvl3_78b", internvl78)
        self.assertIn("cv-bench-result.md", internvl78)
        self.assertIn("新增 InternVL3-78B 一行", internvl78)

    def test_results_report_uses_one_protocol_and_concise_chinese_table(self) -> None:
        architecture = (self.docs / "architecture.md").read_text(encoding="utf-8")
        runbook = (
            self.docs / "msmu-stage3-scoring-commands.md"
        ).read_text(encoding="utf-8")
        helper = (
            self.repository / "scripts" / "msmu" / "_build_results_report.py"
        ).read_text(encoding="utf-8")
        for required in [
            "build_results_report.sh",
            "--profile",
            "--scorer-protocol",
        ]:
            with self.subTest(required=required):
                self.assertIn(required, runbook)
        for required in [
            "一次只允许一个 scorer protocol",
            "模型名称",
            "RGB + Mental-3D 提示词",
            "RGB + 深度估计",
            "canonical provenance",
        ]:
            with self.subTest(required=required):
                self.assertIn(required, architecture)
        self.assertIn('DEFAULT_OUTPUT_NAME = "msmu-result.md"', helper)
        self.assertIn('"模型名称"', helper)
        self.assertIn("REPORT_NOTE", helper)
        self.assertIn("RGB + Mental-3D 提示词", helper)
        self.assertIn("RGB + 深度估计", helper)
        self.assertNotIn("估计深度", helper)
        self.assertNotIn("公平版）", helper)
        self.assertNotIn("原生版）", helper)
        self.assertNotIn('"Model Revision"', helper)
        self.assertNotIn('"Scorer Protocol"', helper)
        self.assertNotIn("llava_next_", helper)
        self.assertNotIn("qwen3_vl_", helper)

    def test_single_model_entry_is_targeted_and_globally_reported(self) -> None:
        readme = (self.repository / "README.md").read_text(encoding="utf-8")
        inference = (self.docs / "msmu-inference.md").read_text(encoding="utf-8")
        stage3 = (self.docs / "msmu-stage3-full-eval.md").read_text(
            encoding="utf-8"
        )
        scoring = (self.docs / "msmu-stage3-scoring-commands.md").read_text(
            encoding="utf-8"
        )
        script = (
            self.repository / "scripts" / "msmu" / "run_model_evaluation.sh"
        ).read_text(encoding="utf-8")
        for document in (readme, inference, stage3):
            with self.subTest(document=document[:40]):
                self.assertIn("run_model_evaluation.sh", document)
                self.assertIn("全局", document)
        self.assertIn("--predictions", scoring)
        self.assertIn("--predictions", script)
        self.assertIn("build_results_report.sh", script)

    def test_generated_outputs_stay_outside_repository(self) -> None:
        for directory_name in ("output", "outputs"):
            with self.subTest(directory_name=directory_name):
                self.assertFalse((self.repository / directory_name).is_dir())

        gitignore = (self.repository / ".gitignore").read_text(encoding="utf-8")
        stage3 = (self.docs / "msmu-stage3-full-eval.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("/outputs/", gitignore)
        self.assertNotIn("$REPO_ROOT/outputs", stage3)
        self.assertIn("$OUTPUT_ROOT/_answer_audit", stage3)

    def test_server_storage_template_separates_new_downloads_from_legacy_assets(self) -> None:
        config = self.repository / "configs" / "msmu-server.env.example"
        variable_names = [
            "SERVER_STORAGE_ROOT",
            "LEGACY_STORAGE_ROOT",
            "REPO_ROOT",
            "OUTPUT_ROOT",
            "MANUAL_TEST_OUTPUT_ROOT",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "HF_ASSETS_CACHE",
            "DATA_DOWNLOAD_ROOT",
            "MODEL_ROOT",
            "CONDA_ENVS_PATH",
            "CONDA_PKGS_DIRS",
            "XDG_CACHE_HOME",
            "PIP_CACHE_DIR",
            "UV_CACHE_DIR",
            "TORCH_HOME",
            "UPSTREAM_ROOT",
            "CHECKPOINT_ROOT",
            "DATASET_ROOT",
            "JUDGE_MODEL",
            "LLAVA_MISTRAL_7B_MODEL",
        ]
        shell_names = " ".join(variable_names)
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'set -a; source "$1"; set +a; '
                f'for name in {shell_names}; do printf "%s=%s\\n" "$name" "${{!name}}"; done',
                "storage-template-test",
                str(config),
            ],
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )
        values = dict(line.split("=", 1) for line in completed.stdout.splitlines())
        new_root = "/media/datasets/lihaoran"
        legacy_root = "/media/datasets/tangzecong"

        self.assertEqual(values["SERVER_STORAGE_ROOT"], new_root)
        self.assertEqual(values["LEGACY_STORAGE_ROOT"], legacy_root)
        agents = (self.repository / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn(f"`{new_root}/`", agents)
        self.assertIn(f"`{legacy_root}/`", agents)
        self.assertIn("不得继续向其中下载新资产", agents)
        for name in [
            "REPO_ROOT",
            "OUTPUT_ROOT",
            "MANUAL_TEST_OUTPUT_ROOT",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "HF_ASSETS_CACHE",
            "DATA_DOWNLOAD_ROOT",
            "MODEL_ROOT",
            "CONDA_PKGS_DIRS",
            "XDG_CACHE_HOME",
            "PIP_CACHE_DIR",
            "UV_CACHE_DIR",
            "TORCH_HOME",
            "UPSTREAM_ROOT",
            "CHECKPOINT_ROOT",
        ]:
            with self.subTest(name=name):
                self.assertTrue(values[name].startswith(f"{new_root}/"))
        self.assertEqual(values["CONDA_ENVS_PATH"].split(":", 1)[0], f"{new_root}/conda/envs")
        for name in ["DATASET_ROOT", "JUDGE_MODEL", "LLAVA_MISTRAL_7B_MODEL"]:
            with self.subTest(name=name):
                self.assertTrue(values[name].startswith(f"{legacy_root}/"))

    def test_server_scripts_do_not_hardcode_storage_namespaces(self) -> None:
        hardcoded: list[str] = []
        scripts = self.repository / "scripts"
        for path in sorted(scripts.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                if "/media/datasets/" in path.read_text(encoding="utf-8"):
                    hardcoded.append(path.relative_to(self.repository).as_posix())
        self.assertEqual(hardcoded, [])

        old_project = "/media/datasets/tangzecong/latent_reasoning/spatial-vlm-eval"
        old_outputs = "/media/datasets/tangzecong/latent_reasoning/msmu-outputs"
        operational_docs = [
            self.repository / "README.md",
            self.docs / "msmu-all-model-test-commands.md",
            self.docs / "msmu-stage1-canary.md",
            self.docs / "msmu-stage2-smoke8.md",
            self.docs / "msmu-stage3-full-eval.md",
            self.docs / "msmu-stage3-scoring-commands.md",
        ]
        for document in operational_docs:
            text = document.read_text(encoding="utf-8")
            with self.subTest(document=document.name):
                self.assertNotIn(old_project, text)
                self.assertNotIn(old_outputs, text)

    def test_agents_routes_document_reading_and_updates(self) -> None:
        agents = (self.repository / "AGENTS.md").read_text(encoding="utf-8")
        for required in [
            "## 文档读取路由",
            "## 文档更新触发",
            "docs/README.md",
            "docs/evaluation-scope.md",
            "docs/troubleshooting/",
        ]:
            with self.subTest(required=required):
                self.assertIn(required, agents)

    def test_concise_report_protocol_policy_is_consistent(self) -> None:
        agents = (self.repository / "AGENTS.md").read_text(encoding="utf-8")
        matrix = (self.docs / "model-matrix.md").read_text(encoding="utf-8")
        inference = (self.docs / "msmu-inference.md").read_text(encoding="utf-8")
        adr = (
            self.docs / "decisions" / "0001-separate-inference-and-scorer-protocols.md"
        ).read_text(encoding="utf-8")
        for document in (agents, matrix, inference, adr):
            with self.subTest(document=document[:40]):
                self.assertRegex(document, r"一次只选择一个 scorer\s+protocol")
                self.assertIn("provenance", document)
                self.assertIn("模型名称", document)


if __name__ == "__main__":
    unittest.main()
