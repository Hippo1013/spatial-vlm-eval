from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = REPOSITORY / "scripts" / "msmu"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIRECTORY / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller_module = load_script("msmu_sota_controller", "_run_sota_supplement.py")
watcher_module = load_script("msmu_sota_watcher", "_sota_event_watcher.py")


class SotaSupplementControllerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.manual_root = Path(self.temporary_directory.name) / "manual"
        self.controller = controller_module.Controller(self.manual_root, REPOSITORY)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_frozen_lanes_score_order_and_report_scope(self):
        self.assertEqual(
            controller_module.LANES,
            {
                "gpu0": (
                    "robobrain25_8b_nv_rgb",
                    "hispatial3b_moge2_xyz",
                    "spatialladder3b_rgb",
                ),
                "gpu1": (
                    "robobrain25_8b_mt_rgb",
                    "spatialladder3b_thinking",
                ),
            },
        )
        self.assertEqual(
            controller_module.SCORE_ORDER,
            (
                "robobrain25_8b_nv_rgb",
                "robobrain25_8b_mt_rgb",
                "hispatial3b_moge2_xyz",
                "spatialladder3b_rgb",
                "spatialladder3b_thinking",
            ),
        )
        self.assertEqual(len(controller_module.SOTA_SUPPLEMENT_REPORT_PROFILE_KEYS), 23)

    def test_list_and_dry_run_are_read_only_and_show_two_lanes(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            controller_module.print_plan()
            controller_module.dry_run(self.controller, "0", "1")
        rendered = output.getvalue()
        self.assertIn("gpu0\t1\trobobrain25_8b_nv_rgb", rendered)
        self.assertIn("gpu1\t2\tspatialladder3b_thinking", rendered)
        self.assertIn("report\tbaseline18+main4+thinking1\t23", rendered)
        self.assertIn("no GPU/model/judge/scorer/report action", rendered)
        self.assertFalse(self.manual_root.exists())

    def test_check_smoke_selection_can_remain_read_only(self):
        selections = [
            {"index": index, "raw_type": f"type-{index}"}
            for index in range(8)
        ]
        with mock.patch.object(controller_module, "load_arrow_split", return_value=object()), mock.patch.object(
            controller_module, "select_type_covering_indices", return_value=selections
        ), mock.patch.dict(controller_module.os.environ, {"DATASET_ROOT": "/absolute/msmu"}):
            self.controller.select_smoke_indices(write_report=False)
        self.assertEqual(self.controller.smoke_indices, list(range(8)))
        self.assertFalse((self.manual_root / "02_smoke8" / "selected_indices.json").exists())

    def test_gpu_preflight_scopes_process_checks_to_only_selected_gpus(self):
        responses = [
            SimpleNamespace(
                returncode=0,
                stdout="0, 81920, 80000, 0\n1, 81920, 79000, 1\n2, 81920, 1000, 99\n",
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
        with mock.patch.object(controller_module.subprocess, "run", side_effect=responses) as run:
            controller_module.gpu_inventory(("0", "1"))
        process_commands = [call.args[0] for call in run.call_args_list[1:]]
        self.assertEqual([command[command.index("-i") + 1] for command in process_commands], ["0", "1"])
        self.assertTrue(all("2" not in command for command in process_commands))

    def test_existing_compute_process_on_selected_gpu_fails_closed(self):
        responses = [
            SimpleNamespace(returncode=0, stdout="0, 81920, 80000, 0\n1, 81920, 80000, 0\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="12345\n", stderr=""),
        ]
        with mock.patch.object(controller_module.subprocess, "run", side_effect=responses):
            with self.assertRaisesRegex(controller_module.ConfigurationError, "GPU 0"):
                controller_module.gpu_inventory(("0", "1"))

    def test_completed_scores_skip_judge_and_still_rebuild_exact_report(self):
        commands: list[list[str]] = []

        class FakeController(controller_module.Controller):
            def inspect_inference_artifacts(self, **_kwargs):
                return True

            def run_owned(self, command, *, environment, log_path):
                commands.append(command)
                if "build_results_report.sh" in " ".join(command) and "--check" not in command:
                    report = self.stage3_root / "msmu-result.md"
                    report.parent.mkdir(parents=True, exist_ok=True)
                    rows = ["| model |", "| --- |"] + [f"| row-{index} |" for index in range(23)]
                    report.write_text("\n".join(rows) + "\n", encoding="utf-8")
                return 0

        fake = FakeController(self.manual_root, REPOSITORY)
        score_helper = sys.modules.get("_score_pending_results") or load_script(
            "_score_pending_results", "_score_pending_results.py"
        )
        with mock.patch.object(score_helper, "complete_state_errors", return_value=[]), mock.patch.object(
            fake, "_score_pending_profiles"
        ) as score_pending:
            fake.score_and_report("0")
        score_pending.assert_not_called()
        self.assertEqual(len(commands), 2)
        self.assertIn("--check", commands[0])
        self.assertNotIn("judge", " ".join(" ".join(command) for command in commands))

    def test_lane_fault_is_recorded_as_a_real_watcher_event(self):
        status = self.controller.status.path
        self.controller.status.add("gpu0", "profile", "phase", "FAULT", "broken")
        event = watcher_module.latest_visible_event(status, "gpu0")
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event[1:], ("FAULT", "profile", "phase", "broken"))

    def test_status_table_has_only_frozen_fields(self):
        self.controller.status.add("gpu1", "profile", "phase", "PASS", "ok")
        with self.controller.status.path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(tuple(rows[0]), controller_module.STATUS_FIELDS)
        self.assertEqual(rows[0]["state"], "PASS")


class SotaSupplementShellEntryTest(unittest.TestCase):
    def test_family_inference_cannot_be_shadowed_by_controller_python(self):
        inference = (SCRIPT_DIRECTORY / "infer_sota_supplement.sh").read_text(
            encoding="utf-8"
        )
        entrypoint = (SCRIPT_DIRECTORY / "run_sota_supplement.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('PYTHON="${ROBOBRAIN25_PYTHON:?Set ROBOBRAIN25_PYTHON}"', inference)
        self.assertIn('PYTHON="${HISPATIAL_PYTHON:?Set HISPATIAL_PYTHON}"', inference)
        self.assertIn(
            'PYTHON="${SPATIALLADDER_PYTHON:?Set SPATIALLADDER_PYTHON}"', inference
        )
        self.assertNotIn('exec "${PYTHON}" "${SCRIPT_DIR}/_run_sota_supplement.py"', entrypoint)

    def test_list_and_dry_run_do_not_need_gpu_or_models(self):
        script = SCRIPT_DIRECTORY / "run_sota_supplement.sh"
        listed = subprocess.run(
            ["bash", str(script), "--list"],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn("gpu0\t1\trobobrain25_8b_nv_rgb", listed.stdout)


if __name__ == "__main__":
    unittest.main()
