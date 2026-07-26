import unittest

from spatial_vlm_eval.benchmarks.msmu.smoke_indices import (
    OFFICIAL_TYPE_ORDER,
    select_type_covering_indices,
)


class SmokeIndexSelectionTest(unittest.TestCase):
    def test_selects_one_deterministic_row_per_official_type(self):
        rows = [
            {"type": "height"},
            {"type": "width"},
            {"type": "distance"},
            {"type": "count"},
            {"type": "position"},
            {"type": "refer_two_objects"},
            {"type": "left/right"},
            {"type": "taller_two_object"},
            {"type": "zero"},
        ]
        selected = select_type_covering_indices(rows)
        self.assertEqual([item["official_type"] for item in selected], list(OFFICIAL_TYPE_ORDER))
        self.assertEqual([item["index"] for item in selected], [0, 2, 3, 4, 5, 6, 7, 8])

    def test_missing_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            select_type_covering_indices([{"type": "width"}])


if __name__ == "__main__":
    unittest.main()
