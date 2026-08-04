from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.cv_bench.inference import (
    _adapter_digest_only_change,
    _cvbench_color_canary_specs,
    _digest,
    _request_timeout_seconds,
    _runtime_retry_policy,
    _strict_legacy_gate_migration_errors,
    inspect_local_gpus,
    merge_prediction_shards,
    profile_prompt,
    test_gate_errors,
)
from spatial_vlm_eval.benchmarks.cv_bench.processor_audit import validate_processor_audit
from spatial_vlm_eval.benchmarks.cv_bench.profiles import (
    DIRECT_ANSWER_SUFFIX,
    PROFILE_SEQUENCE,
    PROFILES,
)


class _Pixels:
    shape = (1, 3, 2, 2)

    def numel(self):
        return 12


class CVBenchProfilesAndInferenceTest(unittest.TestCase):
    def test_local_vllm_uses_long_timeout_and_deferred_missing_retry(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_request_timeout_seconds("vllm"), 600.0)
            self.assertEqual(
                _runtime_retry_policy("vllm"),
                {"retries": 0, "retry_missing_passes": 1},
            )
            self.assertEqual(_request_timeout_seconds("openrouter"), 180.0)
            self.assertEqual(
                _runtime_retry_policy("openrouter"),
                {"retries": 2, "retry_missing_passes": 1},
            )

    def test_local_vllm_timeout_and_retry_policy_are_independently_overridable(self):
        with patch.dict(
            os.environ,
            {
                "CVBENCH_VLLM_API_TIMEOUT": "720",
                "CVBENCH_VLLM_INFERENCE_RETRIES": "1",
                "CVBENCH_VLLM_RETRY_MISSING_PASSES": "2",
                "CVBENCH_INFERENCE_RETRIES": "9",
            },
            clear=True,
        ):
            self.assertEqual(_request_timeout_seconds("vllm"), 720.0)
            self.assertEqual(
                _runtime_retry_policy("vllm"),
                {"retries": 1, "retry_missing_passes": 2},
            )

    def test_adapter_digest_only_migration_rejects_any_protocol_change(self):
        old = {"adapter": {"adapter_digest": "old", "backend": "vllm"}, "profile": "p"}
        new = {"adapter": {"adapter_digest": "new", "backend": "vllm"}, "profile": "p"}
        self.assertTrue(_adapter_digest_only_change(old, new))
        new["adapter"]["backend"] = "transformers"
        self.assertFalse(_adapter_digest_only_change(old, new))

    def test_stricter_legacy_canary_can_migrate_without_model_reinvocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in {
                "dataset.json": {"dataset_fingerprint": "dataset"},
                "capacity.json": {"passed": True},
                "validation.json": {"passed": True},
                "input.json": {"passed": True},
                "canary.json": {
                    "passed": True,
                    "endpoints": [
                        {
                            "canary_protocol": (
                                "msmu_semantic_vision_canary_red_circle_top_left_"
                                "blue_square_bottom_right_antialiased512_v2"
                            ),
                            "request_image_count": 1,
                            "answer": "A red circle is top left and a blue square is bottom right.",
                            "generation": {"num_media_prompt": 1},
                        }
                    ],
                },
            }.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            prediction = root / "predictions.jsonl"
            prediction.write_text("{}\n", encoding="utf-8")
            prediction.with_suffix(".jsonl.metadata.json").write_text("{}", encoding="utf-8")
            profile = PROFILES["qwen3_vl_2b"]
            current_binding = {
                "dataset": {"fingerprint": "dataset"},
                "profile": {"key": profile.key},
                "adapter": {"backend": "vllm", "adapter_digest": "new", "processor_audit": None},
                "sharding": {"strategy": "fixed_modulo"},
            }
            gate = {
                "passed": True,
                "profile": profile.key,
                "binding": {
                    **current_binding,
                    "adapter": {**current_binding["adapter"], "adapter_digest": "old"},
                },
                "smoke_indices": [0, 633, 342, 1080, 1438, 1442, 2038, 2042],
                "dataset_audit": str(root / "dataset.json"),
                "capacity_probe": str(root / "capacity.json"),
                "smoke_validation": str(root / "validation.json"),
                "input_audit_gate": str(root / "input.json"),
                "vision_canary": str(root / "canary.json"),
                "smoke_predictions": str(prediction),
            }
            self.assertEqual(
                _strict_legacy_gate_migration_errors(gate, profile, current_binding),
                [],
            )
            canary = json.loads((root / "canary.json").read_text(encoding="utf-8"))
            canary["endpoints"][0]["answer"] = "I cannot see the image."
            (root / "canary.json").write_text(json.dumps(canary), encoding="utf-8")
            self.assertTrue(
                _strict_legacy_gate_migration_errors(gate, profile, current_binding)
            )

    def test_every_profile_uses_the_same_red_and_blue_color_canaries(self):
        specs = _cvbench_color_canary_specs()
        self.assertEqual([color for color, _ in specs], ["red", "blue"])
        for key in PROFILE_SEQUENCE:
            self.assertEqual([color for color, _ in _cvbench_color_canary_specs()], ["red", "blue"], key)

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

    def test_profile_prompt_keeps_direct_and_reasoning_instructions_disjoint(self):
        dataset_prompt = "Which side?\n(A) left\n(B) right"
        reasoning_keys = {"3dthinker_mental3d", "spatialladder3b_thinking"}
        for key in PROFILE_SEQUENCE:
            with self.subTest(profile=key):
                rendered = profile_prompt(dataset_prompt, PROFILES[key])
                self.assertIn(dataset_prompt, rendered)
                if key in reasoning_keys:
                    self.assertIn("<think>", rendered)
                    self.assertIn("<answer>", rendered)
                    self.assertNotIn(DIRECT_ANSWER_SUFFIX, rendered)
                    self.assertTrue(PROFILES[key].inference_protocol.endswith("_v2"))
                else:
                    self.assertEqual(rendered, f"{dataset_prompt}\n{DIRECT_ANSWER_SUFFIX}")

    def test_custom_profile_prompt_without_answer_tags_fails_closed(self):
        profile = SimpleNamespace(key="broken", prompt_prefix="Explain first.")
        with self.assertRaisesRegex(ValueError, "without answer tags"):
            profile_prompt("Question\n(A) yes\n(B) no", profile)

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
