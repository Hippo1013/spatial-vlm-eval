from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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

    def test_shell_entrypoints_are_syntax_valid_and_server_is_non_destructive(self):
        for path in sorted((self.repository / "scripts" / "q_spatial").glob("*.sh")):
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)
        serve = (self.repository / "scripts" / "q_spatial" / "serve_vllm_profile.sh").read_text()
        self.assertIn("port ${port} is occupied", serve)
        self.assertIn("gpu_preflight.sh", serve)
        self.assertNotIn("kill ", serve)
        self.assertNotIn("pkill", serve)


if __name__ == "__main__":
    unittest.main()
