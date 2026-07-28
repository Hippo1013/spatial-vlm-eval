import unittest

from spatial_vlm_eval.models.qwen25_vl.peft_infer import QwenPeftAdapter


class QwenProtocolLockTest(unittest.TestCase):
    def adapter(self, *, max_new_tokens=192, min_pixels=12544, max_pixels=112896):
        return QwenPeftAdapter(
            profile_key="qwen25_vl_7b",
            base_model="local-model",
            base_model_revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
            checkpoint=None,
            checkpoint_revision=None,
            batch_size=1,
            max_new_tokens=max_new_tokens,
            image_min_pixels=min_pixels,
            image_max_pixels=max_pixels,
            device_map="single",
        )

    def test_canonical_decoding_and_pixel_profile_is_accepted(self):
        adapter = self.adapter()
        self.assertEqual(adapter.max_new_tokens, 192)

    def test_protocol_changing_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "new inference protocol"):
            self.adapter(max_new_tokens=193)
        with self.assertRaisesRegex(ValueError, "new inference protocol"):
            self.adapter(min_pixels=1)

    def test_72b_requires_balanced_loading(self):
        with self.assertRaisesRegex(ValueError, "requires balanced two-GPU"):
            QwenPeftAdapter(
                profile_key="qwen25_vl_72b",
                base_model="local-model",
                base_model_revision="89c86200743eec961a297729e7990e8f2ddbc4c5",
                checkpoint=None,
                checkpoint_revision=None,
                batch_size=1,
                max_new_tokens=192,
                image_min_pixels=12544,
                image_max_pixels=112896,
                device_map="single",
            )

    def test_profile_revision_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "locked to revision"):
            QwenPeftAdapter(
                profile_key="qwen25_vl_32b",
                base_model="local-model",
                base_model_revision="0" * 40,
                checkpoint=None,
                checkpoint_revision=None,
                batch_size=1,
                max_new_tokens=192,
                image_min_pixels=12544,
                image_max_pixels=112896,
                device_map="single",
            )


if __name__ == "__main__":
    unittest.main()
