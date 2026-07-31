from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path
from urllib.parse import unquote

from spatial_vlm_eval.benchmarks.msmu.scorer import SCORER_PROTOCOL
from spatial_vlm_eval.models.profiles import PROFILES


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

    def test_model_matrix_matches_profile_registry(self) -> None:
        matrix = (self.docs / "model-matrix.md").read_text(encoding="utf-8")
        documented = set(re.findall(r"^\| `([^`]+)` \|", matrix, flags=re.MULTILINE))
        self.assertEqual(documented, set(PROFILES))
        count = re.search(r"^## 当前 (\d+) 个 inference profile$", matrix, flags=re.MULTILINE)
        self.assertIsNotNone(count)
        self.assertEqual(int(count.group(1)), len(PROFILES))

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
        self.assertIn(SCORER_PROTOCOL, protocol)
        self.assertIn(SCORER_PROTOCOL, agents)

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

    def test_agents_routes_document_reading_and_updates(self) -> None:
        agents = (self.repository / "AGENTS.md").read_text(encoding="utf-8")
        for required in [
            "## 文档读取路由",
            "## 文档更新触发",
            "docs/README.md",
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
