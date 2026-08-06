from __future__ import annotations

from PIL import Image

from spatial_vlm_eval.benchmarks.q_spatial.data import QSpatialTestContract


class FakeDataset:
    column_names = [
        "question",
        "answer_value",
        "answer_unit",
        "question_type",
        "image_path",
        "image",
    ]

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def small_contract(root):
    rgb_root = root / "scannet-images"
    image_path = rgb_root / "scene0015_00" / "color" / "0.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), (20, 40, 60)).save(image_path)
    scan = {
        "question": "How tall is the object?",
        "answer_value": 1.0,
        "answer_unit": "meter",
        "question_type": "object_height",
        "image_path": "QSpatial_scannet/images/scene0015_00/color/0.jpg",
        "image": None,
    }
    plus = {
        "question": "How wide is it?",
        "answer_value": 25.0,
        "answer_unit": "centimeter",
        "question_type": "1d_horizontal",
        "image_path": "QSpatial_plus/images/example.jpg",
        "image": Image.new("RGB", (5, 9), (1, 2, 3)),
    }
    return QSpatialTestContract(
        root / "parquet",
        rgb_root,
        split_datasets={
            "QSpatial_scannet": FakeDataset([scan]),
            "QSpatial_plus": FakeDataset([plus]),
        },
        require_official_size=False,
        verify_files=False,
    )


class OfficialScoringContract:
    dataset_fingerprint = "synthetic-q-spatial-fingerprint"

    def __init__(self):
        scan_counts = (
            ("object_width", 23),
            ("object_height", 22),
            ("horizontal_distance", 60),
            ("vertical_distance", 29),
            ("direct_distance", 36),
        )
        self.rows = []
        for canonical_type, count in scan_counts:
            self.rows.extend(
                {
                    "index": len(self.rows),
                    "split": "QSpatial_scannet",
                    "raw_type": canonical_type,
                    "canonical_type": canonical_type,
                    "answer_value": "1",
                    "answer_unit": "centimeter",
                }
                for _ in range(count)
            )
        for raw_type, count in (("horizontal_distance", 98), ("vertical_distance", 2), ("1d_horizontal", 1)):
            canonical_type = "object_width" if raw_type == "1d_horizontal" else raw_type
            self.rows.extend(
                {
                    "index": len(self.rows),
                    "split": "QSpatial_plus",
                    "raw_type": raw_type,
                    "canonical_type": canonical_type,
                    "answer_value": "1",
                    "answer_unit": "centimeter",
                }
                for _ in range(count)
            )

    def __len__(self):
        return len(self.rows)

    def scoring_row(self, index):
        return dict(self.rows[index])
