from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.q_spatial.command_adapter import (
    fold_system_user_prompt,
    load_generation_manifest,
)
from spatial_vlm_eval.benchmarks.q_spatial.data import QSpatialModelInput, STANDARD_SYSTEM_PROMPT
from spatial_vlm_eval.benchmarks.q_spatial.inference import (
    LLAVA_CONTINUATION,
    LLAVA_USER_FORMAT_SUFFIX,
    LlavaTwoStageAdapter,
    ResolvedConfiguration,
    ResourceBlockedError,
    _capacity_candidates,
    _rotate_stale_test_artifacts,
    _vllm_max_model_len,
    inspect_local_gpus,
    test_gate_errors,
)
from spatial_vlm_eval.benchmarks.q_spatial.profiles import (
    PROFILE_SEQUENCE,
    PROFILES,
    RGB_PROFILE_KEYS,
)
from spatial_vlm_eval.benchmarks.q_spatial.processor_audit import validate_processor_audit
from spatial_vlm_eval.models.common.runtime import GenerationResult


class _FakeOpenAI:
    def __init__(self):
        self.calls = []
        self.closed = False

    def metadata(self):
        return {"backend": "vllm"}

    def generate_messages(self, model_input, messages, *, max_tokens=None, continue_final_message=False):
        self.calls.append((model_input, messages, max_tokens, continue_final_message))
        number = len(self.calls)
        return GenerationResult(
            "first reasoning" if number == 1 else "2} \\distance_unit{meters}\"\"\"",
            {"num_media_prompt": 1, "call": number},
        )

    def close(self):
        self.closed = True


class _Pixels:
    shape = (1, 3, 4, 4)

    def numel(self):
        return 48


class QSpatialProfilesAndInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[3]

    def test_registry_is_exactly_21_tracks_and_18_rgb(self):
        self.assertEqual(len(PROFILE_SEQUENCE), 21)
        self.assertEqual(len(PROFILES), 21)
        self.assertEqual(len(RGB_PROFILE_KEYS), 18)
        self.assertEqual(PROFILE_SEQUENCE[:2], ("llava_next_mistral_7b", "llava_next_yi_34b"))
        self.assertEqual(PROFILES["internvl3_78b"].default_tensor_parallel_size, 4)
        self.assertEqual(PROFILES["qwen3_vl_32b"].default_tensor_parallel_size, 2)
        self.assertNotIn("3dthinker_mental3d", PROFILES)
        self.assertNotIn("spatialladder3b_thinking", PROFILES)

    def test_locked_sampling_decoding_and_seed_policies(self):
        for key in ("qwen3_vl_2b", "qwen3_vl_4b", "qwen3_vl_8b", "qwen3_vl_32b"):
            decoding = PROFILES[key].decoding
            self.assertEqual(
                {name: decoding[name] for name in ("temperature", "top_p", "top_k", "presence_penalty", "max_new_tokens", "seed")},
                {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_new_tokens": 1024, "seed": 3407},
            )
            self.assertEqual(PROFILES[key].seed_strategy, "per_request_fixed_base_seed")
        self.assertTrue(PROFILES["gpt5_openrouter_non_zdr"].provider_nondeterministic)
        self.assertEqual(PROFILES["gpt5_openrouter_non_zdr"].decoding["max_new_tokens"], 16384)
        self.assertIsNone(PROFILES["gpt5_openrouter_non_zdr"].decoding["temperature"])
        self.assertEqual(PROFILES["gemini31pro_openrouter_non_zdr"].decoding["temperature"], 0.0)

    def test_specialized_tracks_preserve_input_comparison_boundaries(self):
        self.assertEqual(PROFILES["ssr_native"].comparison_group, "rgb_derived_depth")
        self.assertEqual(PROFILES["spatialbot_zoedepth"].comparison_group, "rgb_derived_depth")
        self.assertEqual(PROFILES["hispatial3b_moge2_xyz"].comparison_group, "rgb_derived_xyz")
        self.assertEqual(PROFILES["hispatial3b_moge2_xyz"].image_processing["derived_xyz"], "MoGe-2")
        self.assertIn("forbidden", PROFILES["hispatial3b_moge2_xyz"].known_deviation)
        self.assertIn("disabled", PROFILES["3dthinker_rgb"].known_deviation)
        self.assertIn("disabled", PROFILES["spatialladder3b_rgb"].known_deviation)

    def test_llava_two_stage_reuses_one_image_and_preserves_both_outputs(self):
        base = _FakeOpenAI()
        adapter = LlavaTwoStageAdapter(base, PROFILES["llava_next_mistral_7b"])
        value = QSpatialModelInput(
            0,
            Image.new("RGB", (7, 5), "red"),
            STANDARD_SYSTEM_PROMPT,
            "Question: how far?",
        )
        result = adapter.generate(value)
        self.assertEqual(len(base.calls), 2)
        self.assertEqual([call[2] for call in base.calls], [512, 64])
        self.assertEqual([call[3] for call in base.calls], [False, True])
        self.assertIn(LLAVA_USER_FORMAT_SUFFIX, base.calls[0][1][0]["content"][1]["text"])
        self.assertIn(LLAVA_CONTINUATION, base.calls[1][1][-1]["content"])
        self.assertEqual(result.text, "first reasoning" + LLAVA_CONTINUATION + '2} \\distance_unit{meters}\"\"\"')
        self.assertTrue(result.metadata["same_image_in_both_calls"])
        self.assertEqual([call["media_count"] for call in result.metadata["model_calls"]], [1, 1])
        adapter.close()
        self.assertTrue(base.closed)

    def test_folded_prompt_is_lossless_and_bridge_manifest_is_exact(self):
        self.assertEqual(fold_system_user_prompt("system", "Question: q"), "system\n\nQuestion: q")
        profile = PROFILES["ssr_rgb"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps({
                    "profile": profile.key,
                    "model_revision": profile.revision,
                    "upstream_commit": profile.upstream_commit,
                    "decoding": profile.decoding,
                }),
                encoding="utf-8",
            )
            self.assertEqual(load_generation_manifest(profile, path), profile.decoding)
            value = json.loads(path.read_text())
            value["decoding"]["temperature"] = 0.2
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decoding mismatch"):
                load_generation_manifest(profile, path)

    def test_committed_generation_manifests_match_every_required_profile(self):
        required = [profile for profile in PROFILES.values() if profile.requires_runtime_generation_manifest]
        self.assertEqual([profile.key for profile in required], [
            "ssr_rgb", "ssr_native", "3dthinker_rgb", "spatialladder3b_rgb"
        ])
        for profile in required:
            with self.subTest(profile=profile.key):
                path = self.repository / "configs" / "q-spatial-generation" / f"{profile.key}.json"
                self.assertTrue(path.is_file())
                self.assertEqual(load_generation_manifest(profile, path), profile.decoding)

    def test_processor_audit_locks_roles_llava_suffix_and_one_tensor(self):
        qwen = PROFILES["qwen3_vl_8b"]
        qwen_report = validate_processor_audit(
            profile=qwen,
            rendered_prompt=f"{STANDARD_SYSTEM_PROMPT} user <|image_pad|> Question: q",
            encoded={"pixel_values": _Pixels(), "image_grid_thw": [[1, 2, 2]]},
            image=Image.new("RGB", (4, 3)),
            system_prompt=STANDARD_SYSTEM_PROMPT,
            user_prompt="Question: q",
        )
        self.assertTrue(qwen_report["system_role_supported"])
        self.assertEqual(qwen_report["input_image_count"], 1)
        self.assertFalse(qwen_report["llava_two_stage"]["enabled"])

        llava = PROFILES["llava_next_mistral_7b"]
        llava_report = validate_processor_audit(
            profile=llava,
            rendered_prompt=(
                f"<image> {STANDARD_SYSTEM_PROMPT}\n\nQuestion: q\n{LLAVA_USER_FORMAT_SUFFIX}"
            ),
            encoded={"pixel_values": _Pixels()},
            image=Image.new("RGB", (4, 3)),
            system_prompt=STANDARD_SYSTEM_PROMPT,
            user_prompt="Question: q",
        )
        self.assertFalse(llava_report["system_role_supported"])
        self.assertTrue(llava_report["llava_two_stage"]["enabled"])
        self.assertRegex(llava_report["final_stage_1_user_prompt_sha256"], r"^[0-9a-f]{64}$")

    def test_stale_test_gate_fails_closed(self):
        gate = {
            "passed": True,
            "binding_digest": "old",
            "binding": {"dataset": {"fingerprint": "dataset"}},
            "dataset_fingerprint": "dataset",
            "vision_canary": {"passed": True},
            "smoke_validation": {"passed": True},
            "input_audit_gate": {"passed": True},
            "processor_audit": {"passed": True},
            "selected_concurrency": 8,
        }
        self.assertEqual(test_gate_errors(gate, "new"), ["binding digest differs"])

    def test_stale_gate_artifacts_are_preserved_before_fresh_test(self):
        with tempfile.TemporaryDirectory() as directory:
            track = Path(directory)
            artifacts = track / "test_artifacts"
            artifacts.mkdir()
            (artifacts / "old.journal.jsonl").write_text("old", encoding="utf-8")
            (track / "test_gate.json").write_text(
                json.dumps({"passed": True, "binding_digest": "old"}), encoding="utf-8"
            )
            archived = _rotate_stale_test_artifacts(track, "new")
            self.assertIsNotNone(archived)
            assert archived is not None
            self.assertEqual((archived / "old.journal.jsonl").read_text(), "old")
            self.assertFalse((track / "test_artifacts").exists())
            self.assertFalse((track / "test_gate.json").exists())
            self.assertEqual(len(list(track.glob("test_gate.stale-old-*.json"))), 1)

    def test_capacity_candidates_are_descending_and_resource_block_is_typed(self):
        with patch.dict(os.environ, {"QSPATIAL_CAPACITY_CANDIDATES": "32,16,8,4,2,1"}):
            self.assertEqual(_capacity_candidates(), (32, 16, 8, 4, 2, 1))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_capacity_candidates("openrouter"), (8, 4, 2, 1))
        with patch.dict(os.environ, {"QSPATIAL_API_CAPACITY_CANDIDATES": "4,2,1"}):
            self.assertEqual(_capacity_candidates("openrouter"), (4, 2, 1))
        with patch.dict(os.environ, {"QSPATIAL_CAPACITY_CANDIDATES": "1,2"}):
            with self.assertRaises(ValueError):
                _capacity_candidates()
        self.assertTrue(issubclass(ResourceBlockedError, RuntimeError))

    def test_vllm_context_budget_is_positive_and_large_enough_for_qwen_smoke(self):
        configuration = ResolvedConfiguration(
            profile=PROFILES["qwen3_vl_32b"],
            backend="vllm",
            base_urls=("http://127.0.0.1:18101/v1",),
            decoding=dict(PROFILES["qwen3_vl_32b"].decoding),
            adapter_digest="digest",
            command=None,
            processor_audit=None,
        )
        with patch.dict(os.environ, {"QSPATIAL_VLLM_MAX_MODEL_LEN": "32768"}):
            self.assertEqual(_vllm_max_model_len(configuration), 32768)
        with patch.dict(os.environ, {"QSPATIAL_VLLM_MAX_MODEL_LEN": "0"}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                _vllm_max_model_len(configuration)

    def test_gpu_selection_count_and_busy_specialized_runner_fail_closed(self):
        inventory = "0, GPU-0, NVIDIA A800, 81920, 80000, 0\n1, GPU-1, NVIDIA A800, 81920, 80000, 0\n"
        completed_inventory = subprocess.CompletedProcess([], 0, inventory, "")
        empty_processes = subprocess.CompletedProcess([], 0, "", "")
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=[completed_inventory, empty_processes]),
            patch.dict(os.environ, {"QSPATIAL_QWEN3_VL_8B_GPU_IDS": "0"}, clear=True),
        ):
            report = inspect_local_gpus(PROFILES["qwen3_vl_8b"], "vllm")
        self.assertEqual(report["selected_gpu_ids"], [0])

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=[completed_inventory, empty_processes]),
            patch.dict(os.environ, {"QSPATIAL_QWEN3_VL_8B_GPU_IDS": "0,1"}, clear=True),
            self.assertRaisesRegex(ResourceBlockedError, "requires 1 explicit"),
        ):
            inspect_local_gpus(PROFILES["qwen3_vl_8b"], "vllm")

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=[completed_inventory, empty_processes]),
            patch.dict(os.environ, {"QSPATIAL_INTERNVL3_78B_GPU_IDS": "0,1"}, clear=True),
            self.assertRaisesRegex(ResourceBlockedError, "requires 4 explicit"),
        ):
            inspect_local_gpus(PROFILES["internvl3_78b"], "vllm")

        occupied = subprocess.CompletedProcess([], 0, "GPU-0, 123, python, 1024\n", "")
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", side_effect=[completed_inventory, occupied]),
            patch.dict(os.environ, {"QSPATIAL_SSR_RGB_GPU_IDS": "0"}, clear=True),
            self.assertRaisesRegex(ResourceBlockedError, "already has compute"),
        ):
            inspect_local_gpus(PROFILES["ssr_rgb"], "upstream_transformers")


if __name__ == "__main__":
    unittest.main()
