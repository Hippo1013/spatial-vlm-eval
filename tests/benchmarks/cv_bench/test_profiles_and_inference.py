from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.cv_bench.inference import (
    _digest,
    _vision_canary_spec,
    inspect_local_gpus,
    merge_prediction_shards,
    test_gate_errors,
)
from spatial_vlm_eval.benchmarks.cv_bench.processor_audit import validate_processor_audit
from spatial_vlm_eval.benchmarks.cv_bench.profiles import PROFILE_SEQUENCE, PROFILES
from spatial_vlm_eval.models.common.vision_canary import (
    RED_IMAGE_CANARY_PROTOCOL,
    VISION_CANARY_PROTOCOL,
)


class _Pixels:
    shape = (1, 3, 2, 2)

    def numel(self):
        return 12


class CVBenchProfilesAndInferenceTest(unittest.TestCase):
    def test_only_3dthinker_uses_the_minimum_red_image_canary(self):
        for key in PROFILE_SEQUENCE:
            protocol = _vision_canary_spec(PROFILES[key])[0]
            expected = RED_IMAGE_CANARY_PROTOCOL if PROFILES[key].family == "3dthinker" else VISION_CANARY_PROTOCOL
            self.assertEqual(protocol, expected, key)

    def test_target_registry_has_exactly_23_distinct_tracks(self):
        self.assertEqual(len(PROFILE_SEQUENCE), 23)
        self.assertEqual(len(set(PROFILE_SEQUENCE)), 23)
        self.assertEqual(set(PROFILE_SEQUENCE), set(PROFILES))
        self.assertEqual(PROFILES["internvl3_78b"].default_tensor_parallel_size, 4)
        self.assertEqual(PROFILES["gpt5_openrouter_non_zdr"].decoding["max_new_tokens"], 16384)
        self.assertIsNone(PROFILES["gpt5_openrouter_non_zdr"].decoding["temperature"])
        self.assertEqual(PROFILES["gemini31pro_openrouter_non_zdr"].decoding["temperature"], 0.0)
        self.assertNotIn("hispatial3b_rgb", PROFILES)
        self.assertIn("MoGe-2 XYZ", PROFILES["hispatial3b_moge2_xyz"].display_name)
        self.assertEqual(
            PROFILES["hispatial3b_moge2_xyz"].image_processing["derived_xyz_revision"],
            "b135031bae30b5ac2ae141a0e68717795ce38340",
        )
        self.assertEqual(
            PROFILES["hispatial3b_moge2_xyz"].image_processing[
                "derived_xyz_upstream_commit"
            ],
            "925b8ed835a7a9cdb7578ba15c658a0afc969030",
        )

    def test_specialized_sampling_and_prompt_profiles_are_separate(self):
        mental = PROFILES["3dthinker_mental3d"]
        self.assertTrue(mental.decoding["do_sample"])
        self.assertEqual(mental.decoding["temperature"], 0.7)
        self.assertEqual(mental.decoding["top_p"], 0.9)
        self.assertEqual(mental.decoding["max_new_tokens"], 2048)
        ladder = PROFILES["spatialladder3b_thinking"]
        self.assertIn("<think>", ladder.prompt_prefix)
        self.assertEqual(ladder.decoding["max_new_tokens"], 1024)

    def test_fixed_shard_merge_is_sorted_complete_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard0 = root / "zero.jsonl"
            shard1 = root / "one.jsonl"
            shard0.write_text('{"index":2,"raw_prediction":"C"}\n{"index":0,"raw_prediction":"A"}\n')
            shard1.write_text('{"index":3,"raw_prediction":"D"}\n{"index":1,"raw_prediction":"B"}\n')
            output = root / "predictions.jsonl"
            rows = merge_prediction_shards(
                [shard0, shard1], output, expected_indices=[0, 1, 2, 3]
            )
            self.assertEqual([row["index"] for row in rows], [0, 1, 2, 3])
            shard1.write_text('{"index":2,"raw_prediction":"D"}\n{"index":1,"raw_prediction":"B"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                merge_prediction_shards(
                    [shard0, shard1], output, expected_indices=[0, 1, 2]
                )

    def test_processor_audit_requires_one_family_placeholder_and_one_image_tensor(self):
        cases = {
            "llava_next_mistral_7b": "<image>",
            "internvl3_8b": "<IMG_CONTEXT>",
            "qwen3_vl_2b": "<|image_pad|>",
        }
        for key, placeholder in cases.items():
            with self.subTest(profile=key):
                report = validate_processor_audit(
                    profile=PROFILES[key],
                    rendered_prompt=f"user {placeholder} question",
                    encoded={"pixel_values": _Pixels(), "image_grid_thw": [[1, 2, 2]]},
                    image=Image.new("RGB", (4, 3)),
                )
                self.assertEqual(report["logical_image_placeholder_count"], 1)
        with self.assertRaisesRegex(ValueError, "expected one"):
            validate_processor_audit(
                profile=PROFILES["qwen3_vl_2b"],
                rendered_prompt="no image placeholder",
                encoded={"pixel_values": _Pixels()},
                image=Image.new("RGB", (4, 3)),
            )

    def test_internvl78_gpu_preflight_enumerates_four_80gb_devices_read_only(self):
        inventory = "\n".join(
            f"{index}, GPU-{index}, NVIDIA A800 80GB PCIe, 81920, 80000, 0"
            for index in range(4)
        )
        with (
            patch.dict(os.environ, {"CVBENCH_INTERNVL3_78B_GPU_IDS": "0,1,2,3"}),
            patch(
                "spatial_vlm_eval.benchmarks.cv_bench.inference.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch(
                "spatial_vlm_eval.benchmarks.cv_bench.inference.subprocess.run",
                side_effect=[
                    SimpleNamespace(stdout=inventory, returncode=0),
                    SimpleNamespace(stdout="", returncode=0),
                ],
            ),
        ):
            report = inspect_local_gpus(PROFILES["internvl3_78b"], "vllm")
        self.assertEqual(report["selected_gpu_ids"], [0, 1, 2, 3])
        self.assertEqual(len(report["inventory"]), 4)
        self.assertEqual(report["policy"], "read-only; no process is terminated")

    def test_test_gate_rejects_missing_bound_audit_artifacts(self):
        binding = {
            "dataset": {"fingerprint": "dataset"},
            "adapter": {"processor_audit": None},
        }
        binding_digest = _digest(binding)
        errors = test_gate_errors(
            {
                "passed": True,
                "binding": binding,
                "binding_digest": binding_digest,
                "smoke_indices": [0, 633, 342, 1080, 1438, 1442, 2038, 2042],
            },
            binding_digest,
        )
        self.assertIn("missing dataset_audit artifact", errors)
        self.assertIn("missing smoke prediction artifact", errors)


if __name__ == "__main__":
    unittest.main()
