from __future__ import annotations

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from spatial_vlm_eval.benchmarks.cv_bench.data import (
    CVBenchModelInput,
    DATASET_FILES,
    DATASET_REVISION,
    OFFICIAL_TEST_SIZE,
    QUESTION_EXTENSION,
    SMOKE8_INDICES,
)

from tests.benchmarks.cv_bench.helpers import official_fake_contract


class CVBenchDataContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.contract = official_fake_contract(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_locked_file_identity_and_split_size(self):
        self.assertEqual(DATASET_REVISION, "bc284db50d036958861cb60cdd7b77612052ce0d")
        self.assertEqual(len(self.contract), OFFICIAL_TEST_SIZE)
        self.assertEqual([item.rows for item in DATASET_FILES], [1438, 1200])
        for item in DATASET_FILES:
            self.assertRegex(item.sha256, r"^[0-9a-f]{64}$")

    def test_adapter_input_has_only_index_rgb_and_final_prompt(self):
        model_input = self.contract.model_input(0)
        self.assertEqual([field.name for field in fields(CVBenchModelInput)], ["index", "image", "question"])
        self.assertEqual(model_input.image.mode, "RGB")
        self.assertTrue(model_input.question.endswith(QUESTION_EXTENSION))
        for forbidden in ("answer", "task", "source", "choices", "bbox"):
            self.assertFalse(hasattr(model_input, forbidden))
        self.assertEqual(set(self.contract.prediction_row(0, "A")), {"index", "raw_prediction"})

    def test_dataset_manifest_locks_schema_counts_prompts_and_images(self):
        manifest = self.contract.dataset_manifest(include_images=True)
        self.assertEqual(manifest["split_counts"], {"2D": 1438, "3D": 1200})
        self.assertEqual(
            manifest["task_counts"],
            {"Count": 788, "Depth": 600, "Distance": 600, "Relation": 650},
        )
        self.assertRegex(manifest["prompt_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(manifest["image_pixel_manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_smoke8_covers_four_tasks_two_each_and_all_sources(self):
        rows = [self.contract.scoring_row(index) for index in SMOKE8_INDICES]
        counts = {task: sum(row["task"] == task for row in rows) for task in {row["task"] for row in rows}}
        self.assertEqual(counts, {"Count": 2, "Relation": 2, "Depth": 2, "Distance": 2})
        self.assertEqual({row["source"] for row in rows}, {"ADE20K", "COCO", "Omni3D"})


if __name__ == "__main__":
    unittest.main()
