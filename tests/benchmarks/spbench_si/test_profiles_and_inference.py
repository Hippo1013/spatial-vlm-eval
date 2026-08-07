from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.spbench_si.command_adapter import load_generation_manifest
from spatial_vlm_eval.benchmarks.spbench_si.inference import (
    ResolvedConfiguration,
    _capacity_candidates,
    _rotate_stale_test_artifacts,
    _spatialladder_batch_candidates,
    _vllm_runtime_version,
    binding,
    test_gate_errors,
)
from spatial_vlm_eval.benchmarks.spbench_si.processor_audit import validate_processor_audit
from spatial_vlm_eval.benchmarks.spbench_si.profiles import (
    DERIVED_PROFILE_KEYS,
    PROFILE_SEQUENCE,
    PROFILES,
    RGB_PROFILE_KEYS,
)


class _Pixels:
    shape = (1, 3, 4, 4)
    def numel(self):
        return 48


class SPBenchSIProfilesInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[3]

    def test_registry_is_exactly_21_with_18_rgb_and_three_derived(self):
        self.assertEqual(len(PROFILE_SEQUENCE), 21)
        self.assertEqual(len(PROFILES), 21)
        self.assertEqual(len(RGB_PROFILE_KEYS), 18)
        self.assertEqual(set(DERIVED_PROFILE_KEYS), {"ssr_native", "spatialbot_zoedepth", "hispatial3b_moge2_xyz"})
        self.assertEqual(PROFILES["internvl3_78b"].default_tensor_parallel_size, 4)
        self.assertNotIn("3dthinker_mental3d", PROFILES)
        self.assertNotIn("spatialladder3b_thinking", PROFILES)

    def test_locked_decoding_matches_spbench_plan(self):
        for key in ("llava_next_mistral_7b", "llava_next_yi_34b", "internvl3_8b", "internvl3_38b"):
            self.assertEqual(PROFILES[key].decoding["max_new_tokens"], 128)
            self.assertFalse(PROFILES[key].decoding["do_sample"])
        for key in ("qwen3_vl_2b", "qwen3_vl_4b", "qwen3_vl_8b", "qwen3_vl_32b"):
            decoding = PROFILES[key].decoding
            self.assertEqual(
                {name: decoding[name] for name in ("temperature", "top_p", "top_k", "presence_penalty", "max_new_tokens", "seed")},
                {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_new_tokens": 128, "seed": 3407},
            )
        self.assertEqual(PROFILES["spatialbot_rgb"].decoding["max_new_tokens"], 100)
        self.assertEqual(PROFILES["spatialladder3b_rgb"].image_processing["attention"], "flash_attention_2")
        self.assertTrue(PROFILES["spatialladder3b_rgb"].native_batch_probe)

    def test_generation_manifests_are_exact(self):
        required = [profile for profile in PROFILES.values() if profile.requires_runtime_generation_manifest]
        self.assertEqual([profile.key for profile in required], ["ssr_rgb", "ssr_native", "3dthinker_rgb", "spatialladder3b_rgb"])
        for profile in required:
            path = self.repository / "configs" / "spbench-si-generation" / f"{profile.key}.json"
            self.assertEqual(load_generation_manifest(profile, path), profile.decoding)

    def test_processor_audit_requires_exact_prompt_once_and_one_tensor(self):
        profile = PROFILES["qwen3_vl_8b"]
        report = validate_processor_audit(
            profile=profile,
            rendered_prompt="You are a helpful assistant. <|image_pad|> Question: q",
            encoded={"pixel_values": _Pixels(), "image_grid_thw": [[1, 2, 2]]},
            image=Image.new("RGB", (4, 3)),
            system_prompt="You are a helpful assistant.",
            user_prompt="Question: q",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["input_image_count"], 1)

    def test_capacity_orders_and_stale_gate_rotation(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_capacity_candidates(), (32, 16, 8, 4, 2, 1))
            self.assertEqual(_capacity_candidates("openrouter"), (8, 4, 2, 1))
            self.assertEqual(_spatialladder_batch_candidates(), (16, 8, 4, 2, 1))
        with tempfile.TemporaryDirectory() as directory:
            track = Path(directory)
            artifacts = track / "test_artifacts"
            artifacts.mkdir()
            (artifacts / "old.txt").write_text("old", encoding="utf-8")
            (track / "test_gate.json").write_text(json.dumps({"passed": True, "binding_digest": "old"}), encoding="utf-8")
            archived = _rotate_stale_test_artifacts(track, "new")
            self.assertIsNotNone(archived)
            self.assertFalse(artifacts.exists())
            self.assertEqual((archived / "old.txt").read_text(), "old")

    def test_vllm_runtime_is_exact_019_and_gate_bound(self):
        with patch.dict(os.environ, {"SPBENCH_SI_VLLM_RUNTIME_VERSION": "0.19.1"}, clear=True):
            self.assertEqual(_vllm_runtime_version(), "0.19.1")
            profile = PROFILES["qwen3_vl_8b"]
            configuration = ResolvedConfiguration(
                profile, "vllm", ("http://127.0.0.1:18101/v1",),
                profile.decoding, "a" * 64, None, {"passed": True},
            )
            value = binding(configuration, type("Contract", (), {"dataset_fingerprint": "d"})())
        self.assertEqual(value["runtime"]["vllm_runtime_version"], "0.19.1")
        with patch.dict(os.environ, {"SPBENCH_SI_VLLM_RUNTIME_VERSION": "0.20.0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires vLLM 0.19"):
                _vllm_runtime_version()

    def test_gate_binding_includes_prompt_dataset_capacity_and_batch(self):
        class Contract:
            dataset_fingerprint = "dataset"
        profile = PROFILES["spatialladder3b_rgb"]
        configuration = ResolvedConfiguration(profile, "upstream_transformers", (), profile.decoding, "a" * 64, "python runner.py", None)
        value = binding(configuration, Contract(), {"selected_gpu_ids": [1]})
        self.assertEqual(value["dataset"]["official_test_size"], 1009)
        self.assertEqual(value["runtime"]["selected_gpu_ids"], [1])
        self.assertEqual(value["capacity_candidates"], [16, 8, 4, 2, 1])
        gate = {
            "passed": True, "binding_digest": "digest", "vision_canary": {"passed": True},
            "smoke_validation": {"passed": True}, "input_audit_gate": {"passed": True},
            "processor_audit": {"passed": True}, "selected_capacity": 8,
        }
        self.assertEqual(test_gate_errors(gate, "digest"), [])
        self.assertIn("binding digest differs", test_gate_errors(gate, "other"))


if __name__ == "__main__":
    unittest.main()
