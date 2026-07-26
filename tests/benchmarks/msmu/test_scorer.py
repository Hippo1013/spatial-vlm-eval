import unittest

from spatial_vlm_eval.benchmarks.msmu.scorer import official_quant_score


class OfficialQuantitativeLengthTest(unittest.TestCase):
    def score(self, response):
        return official_quant_score(
            "scale_estimation",
            {
                "answer_in_meters": [1.0, 1.0, 1.0],
                "response_in_meters": response,
            },
        )

    def test_scalar_response_fails_for_list_reference(self):
        score, details = self.score(1.0)
        self.assertEqual(score, 0.0)
        self.assertFalse(details["match_success"])

    def test_shorter_list_response_fails(self):
        score, details = self.score([1.0, 1.0])
        self.assertEqual(score, 0.0)
        self.assertFalse(details["match_success"])
        self.assertIn("shorter than the reference", details["error"])

    def test_equal_length_response_is_scored(self):
        score, details = self.score([1.0, 1.0, 1.0])
        self.assertEqual(score, 1.0)
        self.assertTrue(details["match_success"])

    def test_extra_trailing_values_are_ignored_like_official_scorer(self):
        score, details = self.score([1.0, 1.0, 1.0, 100.0])
        self.assertEqual(score, 1.0)
        self.assertTrue(details["match_success"])
        self.assertEqual(details["delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
