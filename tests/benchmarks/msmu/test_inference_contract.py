import unittest
from dataclasses import fields

from PIL import Image

from spatial_vlm_eval.benchmarks.msmu.data import (
    MSMUTestContract,
    QwenGenerationCollator,
)


def source_row(index: int = 0) -> dict:
    return {
        "image": Image.new("RGB", (4 + index, 3), (10 + index, 20, 30)),
        "type": "width",
        "conversations": {
            "from": ["human", "gpt", "human", "gpt"],
            "value": [
                " <image>\nHow wide is the table? ",
                "SECRET REFERENCE: one metre.",
                "SECOND QUESTION MUST NOT LEAK",
                "SECOND ANSWER MUST NOT LEAK",
            ],
        },
    }


class _Tokenizer:
    padding_side = "right"
    pad_token_id = None
    pad_token = None
    eos_token = "<eos>"


class _ImageProcessor:
    min_pixels = None
    max_pixels = None


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = _ImageProcessor()
        self.messages = []
        self.images = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages.append(messages)
        return "<image>" + messages[0]["content"][1]["text"]

    def __call__(self, *, text, images, **_kwargs):
        self.images = images
        return {"input_ids": text, "pixel_values": images}


class InferenceInputContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = MSMUTestContract(
            "MSMU",
            dataset=[source_row()],
            require_official_size=False,
        )

    def test_model_input_has_only_index_rgb_and_clean_first_question(self):
        model_input = self.contract.model_input(0)
        self.assertEqual([item.name for item in fields(model_input)], ["index", "image", "question"])
        self.assertFalse(hasattr(model_input, "reference"))
        self.assertFalse(hasattr(model_input, "raw_type"))
        self.assertFalse(hasattr(model_input, "task_family"))
        self.assertFalse(hasattr(model_input, "conversations"))
        self.assertEqual(model_input.question, "How wide is the table?")
        self.assertEqual(model_input.image.mode, "RGB")
        self.assertNotIn("SECRET", model_input.question)
        self.assertNotIn("SECOND", model_input.question)

    def test_benchmark_owned_writer_reattaches_exact_provenance(self):
        row = self.contract.prediction_row(0, "about one metre")
        self.assertEqual(
            list(row),
            ["index", "raw_type", "task_family", "question", "reference", "prediction"],
        )
        self.assertEqual(row["raw_type"], "width")
        self.assertEqual(row["task_family"], "scale_estimation")
        self.assertEqual(row["reference"], "SECRET REFERENCE: one metre.")
        self.assertEqual(row["prediction"], "about one metre")

    def test_qwen_collator_receives_no_reference_or_history(self):
        processor = _Processor()
        collator = QwenGenerationCollator(processor, image_min_pixels=10, image_max_pixels=20)
        batch = collator([self.contract.model_input(0)])
        self.assertEqual(batch["indices"], [0])
        self.assertEqual(set(batch), {"model_inputs", "indices"})
        rendered_messages = processor.messages[0]
        self.assertEqual(len(rendered_messages), 1)
        self.assertEqual(rendered_messages[0]["role"], "user")
        self.assertEqual(rendered_messages[0]["content"][0], {"type": "image"})
        self.assertEqual(
            rendered_messages[0]["content"][1],
            {"type": "text", "text": "How wide is the table?"},
        )
        self.assertNotIn("SECRET", str(rendered_messages))
        self.assertNotIn("SECOND", str(rendered_messages))
        self.assertEqual(len(processor.images), 1)

    def test_requires_exactly_one_dataset_image_placeholder(self):
        invalid = source_row()
        invalid["conversations"]["value"][0] = "No placeholder"
        contract = MSMUTestContract("MSMU", dataset=[invalid], require_official_size=False)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            contract.model_input(0)


if __name__ == "__main__":
    unittest.main()
