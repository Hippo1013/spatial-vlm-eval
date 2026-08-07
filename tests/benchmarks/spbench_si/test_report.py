from __future__ import annotations

import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.spbench_si.profiles import PROFILE_SEQUENCE, PROFILES
from spatial_vlm_eval.benchmarks.spbench_si.report import ReportResult, render_markdown


def result(profile_key: str) -> ReportResult:
    task = {
        key: {"total": 1, "score_sum": 0.5, "mean": 0.5}
        for key in ("object_abs_distance", "object_size_estimation", "object_rel_distance", "object_rel_direction")
    }
    summary = {"task_metrics": task, "metrics": {"nq_macro": 0.5, "mcq_macro": 0.5, "overall_four_task_macro": 0.5}}
    audit = {"metrics": {"overall_four_task_macro": 0.4}, "num_main_vs_audit_differences": 3}
    return ReportResult(profile_key, Path(f"/{profile_key}/summary.json"), summary, audit)


class SPBenchSIReportTest(unittest.TestCase):
    def test_approved_twenty_of_twenty_one_report_is_explicit(self):
        results = [result(key) for key in PROFILE_SEQUENCE if key != "internvl3_78b"]
        rendered = render_markdown(results, generated_at="now")
        self.assertIn("暂行 20/21", rendered)
        self.assertIn("仅缺四卡 InternVL3-78B", rendered)
        self.assertIn("Upstream compatibility audit（非主分）", rendered)
        self.assertIn("RGB + MoGe-2 XYZ", rendered)

    def test_full_report_rebuilds_in_place_semantics_and_other_partial_states_fail(self):
        full = render_markdown([result(key) for key in PROFILE_SEQUENCE], generated_at="now")
        self.assertTrue(full.startswith("# SPBench-SI 评测结果\n"))
        self.assertIn("完整 21/21", full)
        with self.assertRaisesRegex(ValueError, "requires 21/21"):
            render_markdown([result(key) for key in PROFILE_SEQUENCE[:19]])


if __name__ == "__main__":
    unittest.main()
