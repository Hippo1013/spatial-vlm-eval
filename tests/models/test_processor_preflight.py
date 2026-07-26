import unittest

from spatial_vlm_eval.models.openai_compatible.processor_preflight import (
    processor_messages,
    validate_processor_contract,
)


class _PixelValues:
    shape = (1, 3, 8, 8)

    def __init__(self, size=192):
        self.size = size

    def numel(self):
        return self.size


class ProcessorPreflightTest(unittest.TestCase):
    def test_llava_has_one_image_placeholder_and_nonempty_pixels(self):
        report = validate_processor_contract(
            profile_key="llava_next_mistral_7b",
            rendered_prompt="USER: <image> What is this? ASSISTANT:",
            encoded={"pixel_values": _PixelValues()},
        )
        self.assertEqual(report["logical_model_placeholder"], "<image>")
        self.assertEqual(report["logical_model_placeholder_count"], 1)
        self.assertGreater(report["pixel_values_numel"], 0)

    def test_internvl_has_one_logical_img_context_and_nonempty_pixels(self):
        report = validate_processor_contract(
            profile_key="internvl3_8b",
            rendered_prompt="<|im_start|>user\n<IMG_CONTEXT>\nWhat is this?<|im_end|>",
            encoded={"pixel_values": _PixelValues()},
        )
        self.assertEqual(report["logical_model_placeholder"], "<IMG_CONTEXT>")
        self.assertEqual(report["logical_model_placeholder_count"], 1)

    def test_prompt_builder_uses_one_structured_image_before_native_template(self):
        expected = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What color is the square?"},
                ],
            }
        ]
        self.assertEqual(processor_messages("llava_next_mistral_7b"), expected)
        self.assertEqual(processor_messages("internvl3_8b"), expected)

    def test_cross_family_placeholder_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "IMG_CONTEXT"):
            validate_processor_contract(
                profile_key="internvl3_8b",
                rendered_prompt="<image>",
                encoded={"pixel_values": _PixelValues()},
            )

    def test_missing_or_empty_pixel_values_fails(self):
        for encoded in [{}, {"pixel_values": _PixelValues(0)}]:
            with self.subTest(encoded=encoded), self.assertRaisesRegex(ValueError, "pixel_values"):
                validate_processor_contract(
                    profile_key="llava_next_mistral_7b",
                    rendered_prompt="<image>",
                    encoded=encoded,
                )


if __name__ == "__main__":
    unittest.main()
