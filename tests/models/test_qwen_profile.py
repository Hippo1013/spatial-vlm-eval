import unittest

from spatial_vlm_eval.models.qwen25_vl.peft_infer import QwenPeftAdapter


class QwenProtocolLockTest(unittest.TestCase):
    def adapter(self, *, max_new_tokens=192, min_pixels=12544, max_pixels=112896):
        return QwenPeftAdapter(
            base_model="local-model",
            base_model_revision="revision",
            checkpoint=None,
            checkpoint_revision=None,
            batch_size=1,
            max_new_tokens=max_new_tokens,
            image_min_pixels=min_pixels,
            image_max_pixels=max_pixels,
        )

    def test_canonical_decoding_and_pixel_profile_is_accepted(self):
        adapter = self.adapter()
        self.assertEqual(adapter.max_new_tokens, 192)

    def test_protocol_changing_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "new inference protocol"):
            self.adapter(max_new_tokens=193)
        with self.assertRaisesRegex(ValueError, "new inference protocol"):
            self.adapter(min_pixels=1)


if __name__ == "__main__":
    unittest.main()
