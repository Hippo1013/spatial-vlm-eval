import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

from spatial_vlm_eval.models.common.runtime import GenerationResult
from spatial_vlm_eval.models.profiles import PROFILES
from spatial_vlm_eval.models.qwen3_vl.infer import (
    QWEN3_IMAGE_MAX_PIXELS,
    QWEN3_IMAGE_MIN_PIXELS,
    QWEN3_MAX_NEW_TOKENS,
    Qwen3VLAdapter,
)


class Qwen3ProtocolLockTest(unittest.TestCase):
    REVISIONS: ClassVar[dict[str, str]] = {
        "qwen3_vl_2b": "89644892e4d85e24eaac8bacfd4f463576704203",
        "qwen3_vl_4b": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "qwen3_vl_8b": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "qwen3_vl_32b": "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
    }

    def adapter(
        self,
        profile_key="qwen3_vl_2b",
        *,
        revision=None,
        max_new_tokens=QWEN3_MAX_NEW_TOKENS,
        min_pixels=QWEN3_IMAGE_MIN_PIXELS,
        max_pixels=QWEN3_IMAGE_MAX_PIXELS,
    ):
        return Qwen3VLAdapter(
            profile_key=profile_key,
            base_model="local-model",
            base_model_revision=revision or self.REVISIONS[profile_key],
            batch_size=1,
            max_new_tokens=max_new_tokens,
            image_min_pixels=min_pixels,
            image_max_pixels=max_pixels,
            device_map="single",
        )

    def test_four_instruct_profiles_have_locked_independent_identities(self):
        self.assertEqual(set(self.REVISIONS), set(PROFILES) & set(self.REVISIONS))
        protocols = set()
        for key, revision in self.REVISIONS.items():
            with self.subTest(profile=key):
                profile = PROFILES[key]
                self.assertEqual(profile.family, "qwen3_vl")
                self.assertEqual(profile.revision, revision)
                self.assertIn("Instruct", profile.model)
                self.assertNotIn("Thinking", profile.model)
                protocols.add(profile.inference_protocol)
        self.assertEqual(len(protocols), 4)

    def test_qwen3_uses_32_pixel_equal_visual_token_budget(self):
        self.assertEqual(QWEN3_IMAGE_MIN_PIXELS, 16 * 32 * 32)
        self.assertEqual(QWEN3_IMAGE_MAX_PIXELS, 144 * 32 * 32)
        adapter = self.adapter()
        self.assertEqual(adapter.max_new_tokens, 192)

    def test_protocol_changing_overrides_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "new inference protocol"):
            self.adapter(max_new_tokens=193)
        with self.assertRaisesRegex(ValueError, "new inference protocol"):
            self.adapter(min_pixels=12544)

    def test_revision_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "locked to revision"):
            self.adapter(revision="0" * 40)

    def test_metadata_declares_no_implicit_system_prompt(self):
        adapter = self.adapter()

        class Processor:
            chat_template = "native Qwen3-VL template without a default system message"

        adapter.processor = Processor()
        metadata = adapter.metadata()
        self.assertIsNone(metadata["system_prompt"])
        self.assertEqual(metadata["image_processing"]["min_pixels"], 16384)
        self.assertEqual(metadata["image_processing"]["max_pixels"], 147456)
        self.assertEqual(metadata["decoding"]["do_sample"], False)
        self.assertEqual(metadata["decoding"]["num_beams"], 1)
        self.assertEqual(
            metadata["image_processing"]["processor_size_fields"],
            {"shortest_edge": 16384, "longest_edge": 147456},
        )
        self.assertEqual(metadata["model_loading"]["model_class"], "Qwen3VLForConditionalGeneration")

    def test_official_qwen3_model_class_is_explicit(self):
        self.assertEqual(Qwen3VLAdapter.MODEL_CLASS_NAME, "Qwen3VLForConditionalGeneration")
        self.assertFalse(Qwen3VLAdapter.TRUST_REMOTE_CODE)
        self.assertEqual(Qwen3VLAdapter.PIXEL_CONFIG_STYLE, "size_edges")

    def test_semantic_vision_canary_checks_two_spatial_shapes_in_one_image(self):
        adapter = self.adapter()
        adapter._ensure_processor = lambda: adapter._runtime_versions.update(
            {"transformers": "test"}
        )

        def fake_generate(model_inputs):
            self.assertEqual(len(model_inputs), 1)
            model_input = model_inputs[0]
            self.assertEqual(model_input.image.mode, "RGB")
            self.assertEqual(model_input.image.size, (512, 512))
            self.assertEqual(model_input.image.getpixel((128, 128)), (255, 0, 0))
            self.assertEqual(model_input.image.getpixel((384, 384)), (0, 0, 255))
            adapter._runtime_versions["torch"] = "test"
            return [
                GenerationResult(
                    text="A red circle is at top-left and a blue square is at bottom-right.",
                    metadata={"num_model_image_tensors": 1},
                )
            ]

        adapter.generate_batch = fake_generate
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "vision_canary.json"
            report = adapter.run_vision_canary(report_path)
            self.assertTrue(report["passed"])
            self.assertEqual(report["request_count"], 1)
            self.assertIn("red circle", report["answer"])
            self.assertIn("blue square", report["answer"])
            self.assertTrue(report_path.is_file())
            self.assertEqual(adapter._runtime_versions, {"transformers": "test"})

    def test_32b_locks_single_gpu_and_batch_size_one(self):
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            Qwen3VLAdapter(
                profile_key="qwen3_vl_32b",
                base_model="local-model",
                base_model_revision=self.REVISIONS["qwen3_vl_32b"],
                batch_size=2,
                max_new_tokens=QWEN3_MAX_NEW_TOKENS,
                image_min_pixels=QWEN3_IMAGE_MIN_PIXELS,
                image_max_pixels=QWEN3_IMAGE_MAX_PIXELS,
                device_map="single",
            )
        with self.assertRaisesRegex(ValueError, "single-GPU"):
            Qwen3VLAdapter(
                profile_key="qwen3_vl_8b",
                base_model="local-model",
                base_model_revision=self.REVISIONS["qwen3_vl_8b"],
                batch_size=8,
                max_new_tokens=QWEN3_MAX_NEW_TOKENS,
                image_min_pixels=QWEN3_IMAGE_MIN_PIXELS,
                image_max_pixels=QWEN3_IMAGE_MAX_PIXELS,
                device_map="balanced",
            )


if __name__ == "__main__":
    unittest.main()
