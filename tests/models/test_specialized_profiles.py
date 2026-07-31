import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_vlm_eval.models.profiles import PROFILES
from spatial_vlm_eval.models.spatialbot.infer import (
    MIDAS_RELATIVE_POSITION_MODULE_COUNT,
    ZOEDEPTH_DERIVED_BUFFER_COUNT,
    encode_spatialbot_depth,
    install_legacy_timm_layers_alias,
    load_zoedepth_checkpoint_compat,
    meters_to_uint16_millimeters,
    patch_midas_relative_position_sizes,
    patch_zoedepth_resize_python_int,
    spatialbot_prompt,
)
from spatial_vlm_eval.models.spatialrgpt.infer import spatialrgpt_question
from spatial_vlm_eval.models.ssr.infer import (
    SSR_BASE_MODEL_ID,
    SSR_BASE_MODEL_REVISION,
    SSR_CLIP_MODEL_REVISION,
    SSR_DEPTHPRO_CHECKPOINT_SHA256,
    SSR_MAMBA_MODEL_REVISION,
    SSR_MIDI_LLM_MODEL_ID,
    SSR_MIDI_LLM_MODEL_REVISION,
    SSR_SIGLIP_MODEL_REVISION,
    _local_adapter_kwargs,
    _verify_file_sha256,
    _verify_local_hidden_size,
    ssr_component_switches,
    ssr_autoroot_entrypoint,
    ssr_image_views,
    ssr_question,
    tor_prefix,
)
from spatial_vlm_eval.models.three_d_thinker.infer import (
    MENTAL_3D_CONTROL_PROMPT,
    ThreeDThinkerAdapter,
    ensure_processor_chat_template,
    extract_last_complete_answer,
    three_d_thinker_prompt,
)


class ProfileRegistryTest(unittest.TestCase):
    def test_registry_contains_exactly_twenty_one_unique_protocols(self):
        self.assertEqual(len(PROFILES), 21)
        protocols = [profile.inference_protocol for profile in PROFILES.values()]
        self.assertEqual(len(protocols), len(set(protocols)))
        self.assertTrue(all(protocol.startswith("msmu_") for protocol in protocols))

    def test_locked_specialized_weight_revisions(self):
        self.assertIn("7bcb4636", PROFILES["ssr"].revision)
        self.assertEqual(
            PROFILES["spatialrgpt"].revision,
            "64df7902f82b5053f5a53455095805e6de3a1f87",
        )
        self.assertEqual(
            PROFILES["3dthinker"].revision,
            "69a70411605f86ec69bada0a625bb96ddee995d9",
        )
        self.assertEqual(
            PROFILES["spatialbot"].revision,
            "41d3b52c642058dfb087885bec0b8e37e0e67f8d",
        )
        self.assertFalse(PROFILES["internvl3_78b"].deployable_on_two_a800_80gb)
        self.assertEqual(PROFILES["qwen25_vl_32b"].default_tensor_parallel_size, 1)
        self.assertEqual(PROFILES["qwen25_vl_72b"].default_tensor_parallel_size, 2)
        for key in ["qwen3_vl_2b", "qwen3_vl_4b", "qwen3_vl_8b", "qwen3_vl_32b"]:
            self.assertEqual(PROFILES[key].default_tensor_parallel_size, 1)

    def test_ssr_auxiliary_stack_is_locked_to_the_checkpoint_dimensions(self):
        self.assertEqual(SSR_BASE_MODEL_ID, "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(SSR_MIDI_LLM_MODEL_ID, "Qwen/Qwen2.5-7B")
        self.assertNotIn("3B", SSR_MIDI_LLM_MODEL_ID)
        for revision in [
            SSR_BASE_MODEL_REVISION,
            SSR_CLIP_MODEL_REVISION,
            SSR_SIGLIP_MODEL_REVISION,
            SSR_MAMBA_MODEL_REVISION,
            SSR_MIDI_LLM_MODEL_REVISION,
        ]:
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
        self.assertRegex(SSR_DEPTHPRO_CHECKPOINT_SHA256, r"^[0-9a-f]{64}$")

    def test_ssr_depthpro_checkpoint_hash_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "depth_pro.pt"
            checkpoint.write_bytes(b"locked-depthpro-checkpoint")
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.assertEqual(
                _verify_file_sha256(
                    str(checkpoint),
                    expected=expected,
                    label="DepthPro",
                ),
                expected,
            )
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                _verify_file_sha256(
                    str(checkpoint),
                    expected="0" * 64,
                    label="DepthPro",
                )

    def test_ssr_dimension_preflight_accepts_top_level_and_nested_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"hidden_size": 3584}', encoding="utf-8")
            self.assertTrue(
                _verify_local_hidden_size(
                    str(root),
                    expected=3584,
                    label="top-level",
                    nested_key="text_config",
                )
            )
            (root / "config.json").write_text(
                '{"vision_config": {"hidden_size": 1152}}',
                encoding="utf-8",
            )
            self.assertTrue(
                _verify_local_hidden_size(
                    str(root),
                    expected=1152,
                    label="nested",
                    nested_key="vision_config",
                )
            )
            with self.assertRaisesRegex(ValueError, "expected 1024"):
                _verify_local_hidden_size(
                    str(root),
                    expected=1024,
                    label="nested",
                    nested_key="vision_config",
                )

    def test_shell_wrappers_use_the_registered_protocol_ids(self):
        repository = Path(__file__).resolve().parents[2]
        shell_text = "\n".join(
            path.read_text(encoding="utf-8") for path in (repository / "scripts" / "msmu").glob("infer_*.sh")
        )
        for profile in PROFILES.values():
            self.assertIn(profile.inference_protocol, shell_text)


class SpecializedProfileSwitchTest(unittest.TestCase):
    def test_spatialbot_midas_relative_position_sizes_are_python_ints(self):
        class FakeAttention:
            def __init__(self):
                self.received = None

            def _get_rel_pos_bias(self, window_size):
                self.received = window_size
                return window_size

        class FakeModel:
            def __init__(self):
                self.attention = [
                    FakeAttention() for _ in range(MIDAS_RELATIVE_POSITION_MODULE_COUNT)
                ]

            def modules(self):
                return iter(self.attention)

        model = FakeModel()
        self.assertEqual(
            patch_midas_relative_position_sizes(model),
            MIDAS_RELATIVE_POSITION_MODULE_COUNT,
        )
        output = model.attention[0]._get_rel_pos_bias((np.int64(7), np.int64(9)))
        self.assertEqual(output, (7, 9))
        self.assertTrue(all(type(value) is int for value in model.attention[0].received))

    def test_spatialbot_installs_legacy_timm_alias_without_overwriting(self):
        existing = object()
        legacy_layers = object()
        legacy_norm_act = object()
        modules = {"timm.layers": existing}
        install_legacy_timm_layers_alias(modules, legacy_layers, legacy_norm_act)
        self.assertIs(modules["timm.layers"], existing)
        self.assertIs(modules["timm.layers.norm_act"], legacy_norm_act)

    def test_spatialbot_zoedepth_resize_casts_numpy_scalar_to_python_int(self):
        class FakeResize:
            def constrain_to_multiple_of(self, value):
                return np.int64(value)

        patch_zoedepth_resize_python_int(FakeResize)
        self.assertIs(type(FakeResize().constrain_to_multiple_of(384)), int)
        original = FakeResize.constrain_to_multiple_of
        patch_zoedepth_resize_python_int(FakeResize)
        self.assertIs(FakeResize.constrain_to_multiple_of, original)

    def test_spatialbot_zoedepth_only_ignores_expected_derived_buffers(self):
        test_case = self

        class FakeTorch:
            @staticmethod
            def load(_path, map_location):
                test_case.assertEqual(map_location, "cpu")
                state = {"model.weight": "weight"}
                for index in range(ZOEDEPTH_DERIVED_BUFFER_COUNT):
                    state[
                        f"module.core.blocks.{index}.attn.relative_position_index"
                    ] = index
                return {"model": state}

        class FakeModel:
            def state_dict(_self):
                return {"model.weight": "weight"}

            def load_state_dict(_self, state, strict):
                test_case.assertTrue(strict)
                test_case.assertEqual(state, {"model.weight": "weight"})

        ignored = load_zoedepth_checkpoint_compat(FakeModel(), Path("checkpoint.pt"), FakeTorch())
        self.assertEqual(len(ignored), ZOEDEPTH_DERIVED_BUFFER_COUNT)

        class WrongTorch:
            @staticmethod
            def load(_path, map_location):
                return {"model": {"model.weight": "weight"}}

        with self.assertRaisesRegex(RuntimeError, "expected 24, got 0"):
            load_zoedepth_checkpoint_compat(FakeModel(), Path("checkpoint.pt"), WrongTorch())

        class LegacyModel:
            def state_dict(_self):
                state = {"model.weight": "old"}
                for index in range(ZOEDEPTH_DERIVED_BUFFER_COUNT):
                    state[f"core.blocks.{index}.attn.relative_position_index"] = index
                return state

            def load_state_dict(_self, state, strict):
                test_case.assertTrue(strict)
                test_case.assertEqual(len(state), ZOEDEPTH_DERIVED_BUFFER_COUNT + 1)

        self.assertEqual(
            load_zoedepth_checkpoint_compat(
                LegacyModel(), Path("checkpoint.pt"), FakeTorch()
            ),
            [],
        )

    def test_ssr_adapter_offline_flag_uses_transformers_adapter_kwargs(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                _local_adapter_kwargs(directory, "local-unspecified"),
                {"adapter_kwargs": {"local_files_only": True}},
            )
        self.assertEqual(
            _local_adapter_kwargs("locked/remote-id", "abc123"),
            {
                "revision": "abc123",
                "adapter_kwargs": {"local_files_only": True},
            },
        )

    def test_ssr_autoroot_is_anchored_to_upstream_and_restored(self):
        original = sys.argv[0]
        with tempfile.TemporaryDirectory() as directory:
            upstream = Path(directory)
            entrypoint = upstream / "infer.py"
            entrypoint.write_text("# upstream entrypoint\n", encoding="utf-8")
            with ssr_autoroot_entrypoint(upstream):
                self.assertEqual(sys.argv[0], str(entrypoint.resolve()))
            self.assertEqual(sys.argv[0], original)

            with self.assertRaisesRegex(RuntimeError, "probe"):
                with ssr_autoroot_entrypoint(upstream):
                    raise RuntimeError("probe")
            self.assertEqual(sys.argv[0], original)

        with self.assertRaisesRegex(FileNotFoundError, "entrypoint is missing"):
            with ssr_autoroot_entrypoint(Path(directory)):
                pass

    def test_ssr_fair_and_native_switches_cannot_cross(self):
        question = "How far is the chair?"
        self.assertEqual(ssr_question("ssr", question), question)
        native = ssr_question("ssr_native", question)
        self.assertTrue(native.startswith(tor_prefix(10)))
        self.assertEqual(native.count("<tor>"), 10)
        self.assertEqual(
            ssr_component_switches("ssr"),
            {"depthpro": False, "midi": False, "tor_count": 0, "model_image_tensor_count": 1},
        )
        self.assertTrue(ssr_component_switches("ssr_native")["depthpro"])

    def test_ssr_native_auxiliaries_keep_the_original_rgb_resolution(self):
        source = Image.new("RGBA", (641, 359), (10, 20, 30, 255))
        original, vlm = ssr_image_views(source)
        self.assertEqual(original.mode, "RGB")
        self.assertEqual(original.size, (641, 359))
        self.assertEqual(vlm.mode, "RGB")
        self.assertEqual(vlm.size, (256, 256))

    def test_spatialrgpt_has_only_one_image_and_no_region_or_depth_tokens(self):
        prompt = spatialrgpt_question("Where is it?", use_image_start_end=False)
        self.assertEqual(prompt.count("<image>"), 1)
        self.assertNotIn("<mask>", prompt)
        self.assertNotIn("<depth>", prompt)

    def test_3dthinker_control_prompt_and_last_answer_extraction(self):
        question = "Is it left?"
        self.assertEqual(three_d_thinker_prompt("3dthinker", question), question)
        native = three_d_thinker_prompt("3dthinker_native", question)
        self.assertTrue(native.endswith(MENTAL_3D_CONTROL_PROMPT))
        extracted, ok = extract_last_complete_answer(
            "<answer>first</answer> trailing <answer> final answer </answer>"
        )
        self.assertTrue(ok)
        self.assertEqual(extracted, "final answer")
        fallback, ok = extract_last_complete_answer("unfinished <answer>raw response")
        self.assertFalse(ok)
        self.assertEqual(fallback, "unfinished <answer>raw response")

    def test_3dthinker_uses_the_checkpoint_tokenizer_template_when_needed(self):
        class Tokenizer:
            chat_template = "checkpoint-native-template"

        class Processor:
            chat_template = None
            tokenizer = Tokenizer()

        processor = Processor()
        self.assertEqual(
            ensure_processor_chat_template(processor),
            "checkpoint-native-template",
        )
        self.assertEqual(processor.chat_template, "checkpoint-native-template")

        processor.chat_template = "processor-template"
        self.assertEqual(ensure_processor_chat_template(processor), "processor-template")

        processor.chat_template = None
        processor.tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, "no native chat template"):
            ensure_processor_chat_template(processor)

    def test_3dthinker_locks_the_slow_official_image_processor(self):
        adapter = object.__new__(ThreeDThinkerAdapter)
        adapter.profile = PROFILES["3dthinker"]
        adapter.upstream_root = Path("/verified/upstream")
        adapter.model_path = "/verified/model"
        adapter.model_revision = PROFILES["3dthinker"].revision
        adapter.model_snapshot_revision_verified = True
        adapter.upstream_commit_verified = True
        adapter.device_map = "auto"
        self.assertFalse(adapter.metadata()["image_processing"]["processor_use_fast"])

    def test_spatialbot_depth_is_mm_uint16_and_officially_packed(self):
        depth_m = np.array([[0.0, 0.031, 0.032], [1.024, 2.5, 65.535]])
        depth_mm = meters_to_uint16_millimeters(depth_m)
        self.assertEqual(depth_mm.dtype, np.uint16)
        expected = np.array([[0, 31, 32], [1024, 2500, 65535]], dtype=np.uint16)
        np.testing.assert_array_equal(depth_mm, expected)
        encoded = encode_spatialbot_depth(depth_mm)
        reconstructed = (
            (encoded[:, :, 0].astype(np.uint16) // 4) * 1024
            + (encoded[:, :, 1].astype(np.uint16) // 8) * 32
            + encoded[:, :, 2].astype(np.uint16) // 8
        )
        np.testing.assert_array_equal(reconstructed, expected)
        self.assertEqual(spatialbot_prompt("spatialbot", "Q").count("<image>"), 1)
        native = spatialbot_prompt("spatialbot_native", "Q")
        self.assertEqual(native.count("<image 1>"), 1)
        self.assertEqual(native.count("<image 2>"), 1)


if __name__ == "__main__":
    unittest.main()
