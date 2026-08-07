from __future__ import annotations

import unittest

from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILES
from spatial_vlm_eval.benchmarks.q_spatial.specialized_runner import _response
from spatial_vlm_eval.models.common.runtime import GenerationResult


class QSpatialSpecializedRunnerTest(unittest.TestCase):
    @staticmethod
    def _response_for(profile_key: str, metadata: dict[str, object]) -> dict[str, object]:
        profile = PROFILES[profile_key]
        return _response(
            {"index": 7},
            profile,
            profile.decoding,
            GenerationResult(
                text="\\scalar{2} \\distance_unit{meters}",
                metadata={"template_sha256": "a" * 64, **metadata},
            ),
            "system",
            "Question: q",
            "system\n\nQuestion: q",
        )

    def test_spatialbot_zoedepth_accepts_one_rgb_and_same_rgb_derived_depth(self):
        response = self._response_for(
            "spatialbot_zoedepth",
            {
                "num_model_image_tensors": 2,
                "input_rgb_count": 1,
                "derived_depth_count": 1,
                "depth_derived_from_same_rgb": True,
            },
        )
        generation = response["generation"]
        self.assertEqual(generation["source_rgb_count"], 1)
        self.assertEqual(generation["num_media_prompt"], 1)
        self.assertEqual(generation["num_model_image_tensors"], 2)
        self.assertEqual(generation["derived_depth_count"], 1)

    def test_spatialbot_zoedepth_rejects_incomplete_or_unbound_depth_evidence(self):
        valid = {
            "num_model_image_tensors": 2,
            "input_rgb_count": 1,
            "derived_depth_count": 1,
            "depth_derived_from_same_rgb": True,
        }
        invalid_cases = (
            {**valid, "num_model_image_tensors": 1},
            {**valid, "input_rgb_count": 2},
            {**valid, "derived_depth_count": 0},
            {**valid, "depth_derived_from_same_rgb": False},
        )
        for metadata in invalid_cases:
            with self.subTest(metadata=metadata), self.assertRaisesRegex(
                ValueError, "must prove one input RGB"
            ):
                self._response_for("spatialbot_zoedepth", metadata)

    def test_other_specialized_profiles_still_require_one_model_image_tensor(self):
        with self.assertRaisesRegex(ValueError, "exactly one model-bound image tensor"):
            self._response_for("spatialbot_rgb", {"num_model_image_tensors": 2})


if __name__ == "__main__":
    unittest.main()
