from __future__ import annotations

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from spatial_vlm_eval.benchmarks.spbench_si.data import (
    EXPECTED_TASK_COUNTS,
    IMAGES_ARCHIVE_SHA256,
    IMAGES_ARCHIVE_SIZE_BYTES,
    MCQ_DIRECT_SUFFIX,
    NUMERIC_DIRECT_SUFFIX,
    OFFICIAL_IMAGE_COUNT,
    OFFICIAL_TEST_SIZE,
    PARQUET_SHA256,
    PARQUET_SIZE_BYTES,
    SMOKE8_INDICES,
    SPBenchSIModelInput,
    SYSTEM_PROMPT,
)
from spatial_vlm_eval.models.common.runtime import input_audit

from tests.benchmarks.spbench_si.helpers import small_contract


class SPBenchSIDataContractTest(unittest.TestCase):
    def test_locked_dataset_constants(self):
        self.assertEqual(OFFICIAL_TEST_SIZE, 1009)
        self.assertEqual(OFFICIAL_IMAGE_COUNT, 524)
        self.assertEqual(PARQUET_SIZE_BYTES, 24423)
        self.assertEqual(IMAGES_ARCHIVE_SIZE_BYTES, 49171512)
        self.assertEqual(PARQUET_SHA256, "72aa46f998212a0d0a9c93ea24107eea086425ccc610083ede35c6218050c9a4")
        self.assertEqual(IMAGES_ARCHIVE_SHA256, "bb53190a1eacf4268fb109b0d8e353c750908bdf33cad8a9221b187d81439461")
        self.assertEqual(EXPECTED_TASK_COUNTS, {
            "object_abs_distance": 149, "object_size_estimation": 463,
            "object_rel_distance": 91, "object_rel_direction": 306,
        })
        self.assertEqual(SMOKE8_INDICES, (4, 297, 306, 410, 460, 518, 918, 1008))

    def test_adapter_boundary_and_prompt_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = small_contract(Path(directory))
            numeric = contract.model_input(0)
            choice = contract.model_input(2)
        self.assertEqual([field.name for field in fields(SPBenchSIModelInput)], [
            "index", "image", "system_prompt", "user_prompt"
        ])
        self.assertEqual(numeric.system_prompt, SYSTEM_PROMPT)
        self.assertEqual(
            numeric.user_prompt,
            "Question: How far in meters?\n\n"
            "Please answer the question using a numerical value (e.g., 42 or 3.1) directly.",
        )
        self.assertEqual(
            choice.user_prompt,
            "Question: Which object is closer?\nOptions:\nA. chair\nB. table\n\n"
            "Please answer with the option's letter from the given choices "
            "(e.g., A, B, etc.) directly.",
        )
        self.assertEqual(choice.image.mode, "RGB")
        for forbidden in ("ground_truth", "question_type", "scene_name", "dataset", "row"):
            self.assertFalse(hasattr(choice, forbidden))
        self.assertFalse(hasattr(choice, "question"))
        audit = input_audit(
            choice,
            {"profile": "test", "inference_protocol": "test", "chat_template": "test"},
        )
        self.assertEqual(audit["question"], choice.user_prompt)

    def test_zip_is_directly_decoded_and_reference_set_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = small_contract(root)
            manifest = contract.dataset_manifest(include_images=True)
            self.assertEqual(manifest["images_archive"]["jpeg_count"], 4)
            self.assertEqual(manifest["images_archive"]["extraction"], "none; decoded directly from ZIP")
            self.assertEqual({tuple(item["size"]) for item in manifest["images"]}, {(512, 512), (640, 480)})
            with self.assertRaisesRegex(ValueError, "ZIP/reference mismatch"):
                small_contract(root / "extra", extra=True)


if __name__ == "__main__":
    unittest.main()
