from __future__ import annotations

import inspect
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from spatial_vlm_eval.models.common.runtime import GenerationResult
from spatial_vlm_eval.models.profiles import (
    CURRENT_TARGET_PROFILE_KEYS,
    PROFILES,
    SOTA_SUPPLEMENT_MAIN_PROFILE_KEYS,
    SOTA_SUPPLEMENT_PROFILE_KEYS,
    SOTA_SUPPLEMENT_REPORT_PROFILE_KEYS,
)
from spatial_vlm_eval.models.sota_spatial import common
from spatial_vlm_eval.models.sota_spatial.hispatial import (
    HiSpatialAdapter,
    MOGE2_REVISION,
)
from spatial_vlm_eval.models.sota_spatial.robobrain25 import RoboBrain25Adapter
from spatial_vlm_eval.models.sota_spatial.spatialladder import (
    SPATIALLADDER_GENERIC_SPECIAL_POST_PROMPT,
    SPATIALLADDER_MAX_PIXELS,
    SPATIALLADDER_MIN_PIXELS,
    SPATIALLADDER_THINKING_TEMPLATE,
    SpatialLadderAdapter,
    extract_last_complete_answer,
    prepare_spatialladder_config,
    prepare_spatialladder_processor,
    select_spatialladder_prediction,
    spatialladder_prompt,
)


class SotaSupplementProfileTest(unittest.TestCase):
    def test_four_main_profiles_are_promoted_and_thinking_remains_supplementary(self):
        self.assertEqual(
            SOTA_SUPPLEMENT_MAIN_PROFILE_KEYS,
            (
                "robobrain25_8b_nv_rgb",
                "robobrain25_8b_mt_rgb",
                "hispatial3b_moge2_xyz",
                "spatialladder3b_rgb",
            ),
        )
        self.assertEqual(
            SOTA_SUPPLEMENT_PROFILE_KEYS,
            (*SOTA_SUPPLEMENT_MAIN_PROFILE_KEYS, "spatialladder3b_thinking"),
        )
        self.assertEqual(len(CURRENT_TARGET_PROFILE_KEYS), 22)
        self.assertTrue(set(SOTA_SUPPLEMENT_MAIN_PROFILE_KEYS) <= set(CURRENT_TARGET_PROFILE_KEYS))
        self.assertNotIn("spatialladder3b_thinking", CURRENT_TARGET_PROFILE_KEYS)
        self.assertEqual(
            SOTA_SUPPLEMENT_REPORT_PROFILE_KEYS,
            (*CURRENT_TARGET_PROFILE_KEYS, "spatialladder3b_thinking"),
        )
        self.assertEqual(len(SOTA_SUPPLEMENT_REPORT_PROFILE_KEYS), 23)

    def test_locked_revisions_protocols_and_decoding_are_independent(self):
        expected_revisions = {
            "robobrain25_8b_nv_rgb": "3d77a19a3ddd8616b3979e03de56096edfb12ff6",
            "robobrain25_8b_mt_rgb": "01145b89a0fe49f78f5d677d25af7351088d7c7d",
            "hispatial3b_moge2_xyz": "75a5e3d65351d7602c492aa91533f62b8a252604",
            "spatialladder3b_rgb": "0819c3adf8827a2ea6c0348d49a23503ecb1f428",
            "spatialladder3b_thinking": "0819c3adf8827a2ea6c0348d49a23503ecb1f428",
        }
        profiles = [PROFILES[key] for key in SOTA_SUPPLEMENT_PROFILE_KEYS]
        self.assertEqual(
            {profile.key: profile.revision for profile in profiles},
            expected_revisions,
        )
        self.assertEqual(len({profile.inference_protocol for profile in profiles}), 5)
        self.assertNotIn(
            "question_only",
            PROFILES["spatialladder3b_thinking"].inference_protocol,
        )
        for key in ("robobrain25_8b_nv_rgb", "robobrain25_8b_mt_rgb"):
            profile = PROFILES[key]
            self.assertTrue(profile.do_sample)
            self.assertEqual((profile.temperature, profile.top_p, profile.max_new_tokens), (0.7, 0.8, 768))
        self.assertFalse(PROFILES["hispatial3b_moge2_xyz"].do_sample)
        self.assertEqual(PROFILES["hispatial3b_moge2_xyz"].max_new_tokens, 100)
        self.assertRegex(MOGE2_REVISION, r"^[0-9a-f]{40}$")
        for key, maximum in (("spatialladder3b_rgb", 128), ("spatialladder3b_thinking", 1024)):
            profile = PROFILES[key]
            self.assertTrue(profile.do_sample)
            self.assertEqual(
                (profile.temperature, profile.top_p, profile.repetition_penalty, profile.max_new_tokens),
                (0.01, 1.0, 1.05, maximum),
            )

    def test_adapter_constructor_boundaries_do_not_accept_benchmark_answers_or_types(self):
        forbidden = {"reference", "raw_type", "task_family", "history", "answer"}
        for adapter in (RoboBrain25Adapter, HiSpatialAdapter, SpatialLadderAdapter):
            parameters = set(inspect.signature(adapter.__init__).parameters)
            self.assertTrue(parameters.isdisjoint(forbidden), adapter.__name__)

    def test_adapter_digest_is_profile_specific_and_scope_independent(self):
        direct = common.adapter_source_digest("spatialladder3b_rgb")
        thinking = common.adapter_source_digest("spatialladder3b_thinking")
        self.assertRegex(direct, r"^[0-9a-f]{64}$")
        self.assertNotEqual(direct, thinking)
        with mock.patch(
            "spatial_vlm_eval.models.profiles.CURRENT_TARGET_PROFILE_KEYS",
            (*CURRENT_TARGET_PROFILE_KEYS, "robobrain25_8b_nv_rgb"),
        ):
            self.assertEqual(common.adapter_source_digest("spatialladder3b_rgb"), direct)


class SpatialLadderProtocolTest(unittest.TestCase):
    def test_direct_uses_original_question_and_thinking_uses_generic_special_template(self):
        question = "How far is the chair?"
        self.assertEqual(spatialladder_prompt("spatialladder3b_rgb", question), question)
        thinking = spatialladder_prompt("spatialladder3b_thinking", question)
        self.assertEqual(
            thinking,
            SPATIALLADDER_THINKING_TEMPLATE.format(question=question)
            + "\n"
            + SPATIALLADDER_GENERIC_SPECIAL_POST_PROMPT,
        )
        self.assertIn("<think>", thinking)
        self.assertIn("<answer>", thinking)
        self.assertNotIn("option's letter", thinking)
        self.assertNotIn("numerical value", thinking)

    def test_thinking_extracts_last_complete_answer_and_preserves_fallback(self):
        raw = "<answer>old</answer>\n<think>retry</think><answer>  final  </answer>"
        self.assertEqual(extract_last_complete_answer(raw), ("final", True))
        prediction, extracted, warnings = select_spatialladder_prediction(
            "spatialladder3b_thinking", raw, index=7
        )
        self.assertEqual((prediction, extracted, warnings), ("final", True, ()))
        incomplete = "<think>still reasoning</think><answer>unfinished"
        prediction, extracted, warnings = select_spatialladder_prediction(
            "spatialladder3b_thinking", incomplete, index=7
        )
        self.assertEqual(prediction, incomplete)
        self.assertFalse(extracted)
        self.assertEqual(len(warnings), 1)

    def test_tied_embeddings_left_padding_and_pixel_bounds_are_locked(self):
        config = SimpleNamespace(
            text_config=SimpleNamespace(tie_word_embeddings=True),
            tie_word_embeddings=False,
        )
        self.assertIs(prepare_spatialladder_config(config), config)
        self.assertTrue(config.tie_word_embeddings)
        config.text_config.tie_word_embeddings = False
        with self.assertRaisesRegex(ValueError, "tied"):
            prepare_spatialladder_config(config)
        processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
        self.assertIs(prepare_spatialladder_processor(processor), processor)
        self.assertEqual(processor.tokenizer.padding_side, "left")
        self.assertEqual(SPATIALLADDER_MIN_PIXELS, 16 * 28 * 28)
        self.assertEqual(SPATIALLADDER_MAX_PIXELS, 512 * 28 * 28)

    def test_native_batch_size_is_part_of_resume_identity(self):
        with mock.patch(
            "spatial_vlm_eval.models.sota_spatial.spatialladder.verify_hf_snapshot_revision",
            return_value=True,
        ), mock.patch(
            "spatial_vlm_eval.models.sota_spatial.spatialladder.verify_git_checkout",
            return_value=True,
        ):
            adapter = SpatialLadderAdapter(
                profile_key="spatialladder3b_rgb",
                model_path="/locked/model",
                upstream_root="/locked/upstream",
                batch_size=7,
            )
        self.assertEqual(adapter.metadata()["decoding"]["native_batch_size"], 7)
        self.assertTrue(adapter.metadata()["decoding"]["fixed_dataset_order"])

    def test_native_batch_probe_descends_and_proves_left_padding(self):
        class FakeAdapter:
            batch_size = 1

            def generate_batch(self, inputs):
                if self.batch_size == 4:
                    raise RuntimeError("simulated OOM")
                results = []
                for value in inputs:
                    red = value.image.convert("RGB").getpixel((0, 0))[0] > 200
                    results.append(
                        GenerationResult(
                            "red" if red else "blue",
                            {"tokenizer_padding_side": "left"},
                        )
                    )
                return results

        with mock.patch.dict(os.environ, {"SPATIALLADDER_BATCH_CANDIDATES": "4,2,1"}):
            report = common.probe_spatialladder_native_batch(FakeAdapter())
        self.assertEqual(report["selected_capacity"], 2)
        self.assertEqual(report["tokenizer_padding_side"], "left")
        self.assertTrue(report["heterogeneous_prompt_lengths"])
        self.assertEqual([item["candidate"] for item in report["attempts"]], [4, 2])


if __name__ == "__main__":
    unittest.main()
