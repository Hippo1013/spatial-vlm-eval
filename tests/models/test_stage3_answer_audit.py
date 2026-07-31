import runpy
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "msmu"
    / "build_stage3_answer_audit.py"
)
MODULE = runpy.run_path(str(SCRIPT), run_name="stage3_answer_audit")


class Stage3AnswerAuditTest(unittest.TestCase):
    def test_sampling_is_deterministic_sorted_and_unique(self):
        select = MODULE["select_sample_indices"]
        first = select(population_size=987, sample_size=30, seed=20260730)
        second = select(population_size=987, sample_size=30, seed=20260730)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 30)
        self.assertEqual(len(set(first)), 30)

    def test_prediction_loader_requires_complete_unique_indices(self):
        load_rows = MODULE["load_prediction_rows"]
        with tempfile.TemporaryDirectory() as temporary:
            predictions = Path(temporary) / "predictions.jsonl"
            predictions.write_text(
                "\n".join(
                    [
                        '{"index":0,"question":"Q0","prediction":"A0"}',
                        '{"index":1,"question":"Q1","prediction":"A1"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_rows(predictions, expected_size=2)
            self.assertEqual(rows[0]["prediction"], "A0")
            predictions.write_text(
                '{"index":0,"question":"Q0","prediction":"A0"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not a complete 2-row result"):
                load_rows(predictions, expected_size=2)

    def test_markdown_uses_clickable_small_relative_images_and_full_text(self):
        model_spec = MODULE["ModelSpec"]("示例模型", "example")
        model_result = MODULE["ModelResult"](
            spec=model_spec,
            predictions_path=Path("predictions.jsonl"),
            rows_by_index={
                7: {
                    "question": "完整题干\n第二行",
                    "prediction": "完整回答，包含 ``` 标记",
                }
            },
        )
        markdown = MODULE["render_markdown"](
            [model_result],
            [7],
            {7: "audit-assets/msmu-0007.jpg"},
            seed=123,
            image_width=240,
        )
        self.assertTrue(markdown.startswith("# 答案抽查\n"))
        self.assertIn("## 示例模型", markdown)
        self.assertIn("### 1（MSMU index 7）", markdown)
        self.assertIn("完整题干\n第二行", markdown)
        self.assertIn("完整回答，包含 ``` 标记", markdown)
        self.assertIn(
            '<a href="audit-assets/msmu-0007.jpg">'
            '<img src="audit-assets/msmu-0007.jpg" alt="MSMU index 7" width="240">'
            "</a>",
            markdown,
        )


if __name__ == "__main__":
    unittest.main()
