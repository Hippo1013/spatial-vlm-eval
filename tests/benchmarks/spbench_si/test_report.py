from __future__ import annotations

import unittest
from pathlib import Path

from spatial_vlm_eval.benchmarks.spbench_si.profiles import PROFILE_SEQUENCE, PROFILES
from spatial_vlm_eval.benchmarks.spbench_si.report import ReportResult, render_markdown


def result(profile_key: str, score: float = 0.5) -> ReportResult:
    task = {
        key: {"total": 1, "score_sum": score, "mean": score}
        for key in ("object_abs_distance", "object_size_estimation", "object_rel_distance", "object_rel_direction")
    }
    summary = {"task_metrics": task, "metrics": {"nq_macro": score, "mcq_macro": score, "overall_four_task_macro": score}}
    audit = {"metrics": {"overall_four_task_macro": 0.4}, "num_main_vs_audit_differences": 3}
    return ReportResult(profile_key, Path(f"/{profile_key}/summary.json"), summary, audit)


class SPBenchSIReportTest(unittest.TestCase):
    def test_explicit_two_profile_exclusion_renders_nineteen_rows(self):
        excluded = ("internvl3_78b", "gemini31pro_openrouter_non_zdr")
        results = [result(key) for key in PROFILE_SEQUENCE if key not in excluded]
        rendered = render_markdown(
            results,
            generated_at="now",
            excluded_profiles=excluded,
        )
        self.assertIn("部分 19/21", rendered)
        self.assertIn("明确排除 2 条；未完成 0 条", rendered)
        self.assertIn("`internvl3_78b`", rendered)
        self.assertIn("`gemini31pro_openrouter_non_zdr`", rendered)
        self.assertIn("未纳入且未排除 profile：无", rendered)
        self.assertIn("主协议评分与 publication gates", rendered)
        self.assertNotIn("上游", rendered)
        self.assertNotIn("审计", rendered)
        self.assertNotIn("双协议", rendered)
        self.assertIn("RGB + MoGe-2 XYZ", rendered)
        self.assertIn("| SSR（RGB） |", rendered)
        self.assertIn("| SSR（RGB + DepthPro + MIDI + TOR10） |", rendered)
        self.assertIn("| SpatialBot-3B（RGB） |", rendered)
        self.assertIn("| SpatialBot-3B（RGB + ZoeDepth） |", rendered)
        self.assertIn("| HiSpatial-3B（RGB + MoGe-2 XYZ） |", rendered)
        self.assertEqual(rendered.count("SSR（RGB）"), 1)
        self.assertEqual(rendered.count("SSR（RGB + DepthPro + MIDI + TOR10）"), 1)
        self.assertNotIn("| 实际输入配置 |", rendered)
        self.assertNotIn("| Input |", rendered)
        self.assertNotIn("| InternVL3-78B（RGB） |", rendered)
        self.assertNotIn("| Gemini 3.1 Pro（RGB） |", rendered)
        self.assertNotIn("PackyAPI", rendered)
        self.assertNotIn("OpenRouter", rendered)
        self.assertIn("每列并列最高分均加粗", rendered)

    def test_each_metric_bolds_all_tied_maxima(self):
        low_key, high_key, tied_key = PROFILE_SEQUENCE[:3]
        rendered = render_markdown(
            [result(low_key, 0.7), result(high_key, 0.8), result(tied_key, 0.8)],
            generated_at="now",
        )
        low_line = next(line for line in rendered.splitlines() if PROFILES[low_key].display_name in line)
        high_line = next(line for line in rendered.splitlines() if PROFILES[high_key].display_name in line)
        tied_line = next(line for line in rendered.splitlines() if PROFILES[tied_key].display_name in line)
        self.assertNotIn("**70.00**", low_line)
        self.assertEqual(high_line.count("**80.00**"), 7)
        self.assertEqual(tied_line.count("**80.00**"), 7)

    def test_full_and_unexcluded_partial_reports_are_both_supported(self):
        full = render_markdown([result(key) for key in PROFILE_SEQUENCE], generated_at="now")
        self.assertTrue(full.startswith("# SPBench-SI 评测结果\n"))
        self.assertIn("完整 21/21", full)

        partial = render_markdown(
            [result(key) for key in PROFILE_SEQUENCE[:19]],
            generated_at="now",
        )
        self.assertIn("部分汇总 19/21", partial)
        self.assertIn("明确排除 0 条；未完成 2 条", partial)

    def test_empty_unknown_duplicate_and_overlapping_selections_fail(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            render_markdown([])
        with self.assertRaisesRegex(ValueError, "Unknown excluded"):
            render_markdown([result(PROFILE_SEQUENCE[0])], excluded_profiles=("unknown",))
        with self.assertRaisesRegex(ValueError, "Duplicate SPBench-SI result"):
            render_markdown([result(PROFILE_SEQUENCE[0]), result(PROFILE_SEQUENCE[0])])
        with self.assertRaisesRegex(ValueError, "still supplied"):
            render_markdown(
                [result("internvl3_78b")],
                excluded_profiles=("internvl3_78b",),
            )


if __name__ == "__main__":
    unittest.main()
