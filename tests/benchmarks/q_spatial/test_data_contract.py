from __future__ import annotations

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from spatial_vlm_eval.benchmarks.q_spatial.data import (
    DATASET_FILES,
    DATASET_REVISION,
    EXPECTED_DISTINCT_IMAGE_COUNTS,
    EXPECTED_PARQUET_SCHEMA,
    EXPECTED_SCANNET_FRAMES,
    EXPECTED_SPLIT_TYPE_COUNTS,
    EXPECTED_SPLIT_UNIT_COUNTS,
    QSpatialModelInput,
    SCANNET_FILE_MANIFEST_SHA256,
    SMOKE8_INDICES,
    STANDARD_SYSTEM_PROMPT,
    STANDARD_SYSTEM_PROMPT_SHA256,
    UPSTREAM_COMMIT,
)

from tests.benchmarks.q_spatial.helpers import small_contract


class QSpatialDataContractTest(unittest.TestCase):
    def test_locked_public_assets_and_prompt_identity(self):
        self.assertEqual(DATASET_REVISION, "17b92e470d58fa46859ebd48ff35a1669828c9be")
        self.assertEqual(UPSTREAM_COMMIT, "ebe8137eae9781aaf7e29691ce8bc68b2a498a83")
        self.assertEqual([item.rows for item in DATASET_FILES], [170, 101])
        self.assertEqual([item.size_bytes for item in DATASET_FILES], [12_022, 129_408_418])
        self.assertEqual(EXPECTED_PARQUET_SCHEMA[-1], ("image", "struct<bytes: binary, path: string>"))
        self.assertRegex(SCANNET_FILE_MANIFEST_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(len(EXPECTED_SCANNET_FRAMES), 99)
        self.assertEqual(len({path.split("/", 1)[0] for path in EXPECTED_SCANNET_FRAMES}), 66)
        self.assertEqual(EXPECTED_DISTINCT_IMAGE_COUNTS, {"QSpatial_scannet": 99, "QSpatial_plus": 87})
        self.assertEqual(EXPECTED_SPLIT_UNIT_COUNTS["QSpatial_plus"], {"centimeter": 94, "meter": 7})
        self.assertEqual(EXPECTED_SPLIT_TYPE_COUNTS["QSpatial_plus"]["1d_horizontal"], 1)
        self.assertEqual(len(STANDARD_SYSTEM_PROMPT.encode()), 337)
        self.assertEqual(STANDARD_SYSTEM_PROMPT_SHA256, "b3da32feb428a7840ecaf1d08ef095b9cd72ff6ef34d5b2b05ec1c1599bb613c")
        self.assertEqual(SMOKE8_INDICES, (0, 1, 3, 9, 14, 205, 247, 250))

    def test_adapter_boundary_contains_only_safe_prompt_and_one_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = small_contract(Path(directory))
            value = contract.model_input(0)
            self.assertEqual(
                [field.name for field in fields(QSpatialModelInput)],
                ["index", "image", "system_prompt", "user_prompt"],
            )
            self.assertEqual(value.image.mode, "RGB")
            self.assertEqual(value.system_prompt, STANDARD_SYSTEM_PROMPT)
            self.assertEqual(value.user_prompt, "Question: How tall is the object?")
            for forbidden in ("answer", "answer_value", "answer_unit", "question_type", "split"):
                self.assertFalse(hasattr(value, forbidden))
            self.assertEqual(contract.prediction_row(0, "one meter"), {"index": 0, "raw_prediction": "one meter"})
            self.assertEqual(contract.scoring_row(1)["canonical_type"], "object_width")

    def test_two_explicit_roots_are_preserved_without_symlink_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = small_contract(root)
            manifest = contract.dataset_manifest(include_images=False)
            self.assertEqual(manifest["roots"]["parquet"], str((root / "parquet").absolute()))
            self.assertEqual(manifest["roots"]["scannet_rgb"], str((root / "scannet-images").absolute()))
            self.assertNotEqual(manifest["roots"]["parquet"], manifest["roots"]["scannet_rgb"])


if __name__ == "__main__":
    unittest.main()
