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
    _api_resume_binding_errors,
    _capacity_candidates,
    _digest,
    _rotate_stale_test_artifacts,
    _spatialladder_batch_candidates,
    _validated_openrouter_resume_seed,
    _vllm_runtime_version,
    binding,
    probe_capacity,
    resolve_configuration,
    test_gate_errors,
)
from spatial_vlm_eval.benchmarks.spbench_si.processor_audit import validate_processor_audit
from spatial_vlm_eval.benchmarks.spbench_si.profiles import (
    DERIVED_PROFILE_KEYS,
    PROFILE_SEQUENCE,
    PROFILES,
    RGB_PROFILE_KEYS,
)
from spatial_vlm_eval.models.common.runtime import GenerationResult
from spatial_vlm_eval.models.common.runtime import pixel_sha256

from .helpers import small_contract


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
        gemini = PROFILES["gemini31pro_openrouter_non_zdr"]
        self.assertEqual(gemini.display_name, "Gemini 3.1 Pro")
        self.assertEqual(gemini.default_backend, "openrouter")

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
        self.assertEqual(PROFILES["spatialladder3b_rgb"].image_processing["tokenizer_padding_side"], "left")
        self.assertEqual(
            PROFILES["spatialladder3b_rgb"].inference_protocol,
            "spbench_si_spatialladder3b_rgb_rgb_default_direct_folded_user_upstream_locked_v2",
        )
        self.assertTrue(PROFILES["spatialladder3b_rgb"].native_batch_probe)

    def test_spatialladder_capacity_probe_uses_unequal_prompts_and_proves_left_padding(self):
        class Adapter:
            profile = PROFILES["spatialladder3b_rgb"]

            def set_batch_size(self, value):
                self.batch_size = value

            def generate_batch(self, values):
                self.prompt_lengths = {len(value.user_prompt) for value in values}
                return [
                    GenerationResult(
                        "red" if value.image.getpixel((0, 0)) == (255, 0, 0) else "blue",
                        {"tokenizer_padding_side": "left"},
                    )
                    for value in values
                ]

        adapter = Adapter()
        with patch(
            "spatial_vlm_eval.benchmarks.spbench_si.inference._spatialladder_batch_candidates",
            return_value=(4,),
        ):
            report = probe_capacity(adapter, backend="upstream_transformers")
        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_capacity"], 4)
        self.assertEqual(report["tokenizer_padding_side"], "left")
        self.assertTrue(report["heterogeneous_prompt_lengths"])
        self.assertGreater(len(adapter.prompt_lengths), 1)

        original = adapter.generate_batch
        adapter.generate_batch = lambda values: [
            GenerationResult(result.text, {"tokenizer_padding_side": "right"})
            for result in original(values)
        ]
        with patch(
            "spatial_vlm_eval.benchmarks.spbench_si.inference._spatialladder_batch_candidates",
            return_value=(4,),
        ):
            with self.assertRaisesRegex(RuntimeError, "No stable SpatialLadder"):
                probe_capacity(adapter, backend="upstream_transformers")

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
            self.assertEqual(_capacity_candidates("packyapi"), (8, 4, 2, 1))
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
            "profile": "spatialladder3b_rgb", "passed": True,
            "binding_digest": "digest", "vision_canary": {"passed": True},
            "smoke_validation": {"passed": True}, "input_audit_gate": {"passed": True},
            "processor_audit": {"passed": True, "tokenizer_padding_side": "left"},
            "selected_capacity": 8, "capacity_probe": {
                "passed": True, "tokenizer_padding_side": "left",
                "heterogeneous_prompt_lengths": True,
            },
        }
        self.assertEqual(test_gate_errors(gate, "digest"), [])
        self.assertIn("binding digest differs", test_gate_errors(gate, "other"))
        gate["capacity_probe"]["tokenizer_padding_side"] = "right"
        self.assertIn(
            "SpatialLadder capacity probe did not prove left padding",
            test_gate_errors(gate, "digest"),
        )

    def test_packyapi_catalog_selects_only_exact_gemini31_and_keeps_profile_identity(self):
        profile = PROFILES["gemini31pro_openrouter_non_zdr"]
        environment = {
            "PACKYAPI_API_KEY": "secret",
            "PACKYAPI_BASE_URL": "https://www.packyapi.com/v1",
            "SPBENCH_SI_GEMINI31PRO_OPENROUTER_NON_ZDR_BACKEND": "packyapi",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch(
                "spatial_vlm_eval.benchmarks.spbench_si.inference.openai_compatible_model_ids",
                return_value=("gemini-3-pro-preview", "gemini-3.1-pro-preview"),
            ),
        ):
            configuration = resolve_configuration(profile)
        self.assertEqual(configuration.backend, "packyapi")
        self.assertEqual(configuration.served_model_name, "gemini-3.1-pro-preview")
        self.assertEqual(configuration.profile.key, "gemini31pro_openrouter_non_zdr")
        self.assertEqual(configuration.profile.model, "google/gemini-3.1-pro-preview-20260219")

        with (
            patch.dict(os.environ, environment, clear=True),
            patch(
                "spatial_vlm_eval.benchmarks.spbench_si.inference.openai_compatible_model_ids",
                return_value=("gemini-3-pro-preview", "gemini-2.5-pro"),
            ),
        ):
            alias_configuration = resolve_configuration(profile)
        self.assertEqual(alias_configuration.served_model_name, "gemini-3-pro-preview")

    def test_api_source_resume_allows_only_source_fields_to_change(self):
        class Contract:
            dataset_fingerprint = "dataset"

        profile = PROFILES["gemini31pro_openrouter_non_zdr"]
        old = ResolvedConfiguration(
            profile, "openrouter", ("https://openrouter.ai/api/v1",),
            profile.decoding, "a" * 64, None, None,
        )
        new = ResolvedConfiguration(
            profile, "packyapi", ("https://www.packyapi.com/v1",),
            profile.decoding, "b" * 64, None, None, "gemini-3.1-pro-preview",
        )
        old_binding = binding(old, Contract())
        new_binding = binding(new, Contract())
        gate = {
            "profile": profile.key,
            "passed": True,
            "binding": old_binding,
            "binding_digest": _digest(old_binding),
            "vision_canary": {"passed": True},
            "smoke_validation": {"passed": True},
            "input_audit_gate": {"passed": True},
            "processor_audit": {"passed": True},
            "selected_capacity": 8,
        }
        self.assertEqual(_api_resume_binding_errors(gate, new_binding, profile), [])
        changed = dict(new_binding)
        changed["decoding"] = {**profile.decoding, "temperature": 0.5}
        self.assertIn(
            "non-source binding field changed: decoding",
            _api_resume_binding_errors(gate, changed, profile),
        )

    def test_openrouter_resume_seed_revalidates_every_success_against_model_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = small_contract(root)
            profile = PROFILES["gemini31pro_openrouter_non_zdr"]
            model_input = contract.model_input(0)
            rgb_sha = pixel_sha256(model_input.image)
            template_sha = _digest({
                "chat_template": profile.chat_template,
                "system_transport": profile.system_transport,
                "system_prompt": model_input.system_prompt,
                "user_prompt": model_input.user_prompt,
                "image_pixel_sha256": rgb_sha,
                "media_count": 1,
            })
            event = {
                "schema_version": 1,
                "run_signature": "old-signature",
                "status": "success",
                "index": 0,
                "prediction": "2",
                "audit": {
                    "profile": profile.key,
                    "inference_protocol": profile.inference_protocol,
                    "chat_template": profile.chat_template,
                    "system_prompt": model_input.system_prompt,
                    "user_prompt": model_input.user_prompt,
                    "image_count": 1,
                    "image_pixel_sha256": rgb_sha,
                },
                "generation": {
                    "canonical_model": profile.model,
                    "provider": "Google AI Studio",
                    "num_media_prompt": 1,
                    "source_rgb_sha256": rgb_sha,
                    "template_sha256": template_sha,
                },
            }
            journal = root / "old.journal.jsonl"
            journal.write_text(json.dumps(event) + "\n", encoding="utf-8")
            seeds, provenance = _validated_openrouter_resume_seed(journal, profile, contract)
            self.assertEqual(seeds[0].text, "2")
            self.assertEqual(provenance["success_count"], 1)

            event["audit"]["user_prompt"] = "changed"
            journal.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audit user prompt"):
                _validated_openrouter_resume_seed(journal, profile, contract)


if __name__ == "__main__":
    unittest.main()
