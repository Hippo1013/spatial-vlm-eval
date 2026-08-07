from __future__ import annotations

import io
import zipfile
from pathlib import Path

from PIL import Image

from spatial_vlm_eval.benchmarks.spbench_si.data import SPBenchSITestContract


def rows() -> list[dict]:
    return [
        {
            "id": 1, "dataset": "scannet", "scene_name": "scene0001_00",
            "question_type": "object_abs_distance", "question": "How far in meters?",
            "ground_truth": "2", "options": None, "images": ["100.jpg"],
        },
        {
            "id": 2, "dataset": "scannet", "scene_name": "scene0001_00",
            "question_type": "object_size_estimation", "question": "How large in centimeters?",
            "ground_truth": "80", "options": None, "images": ["200.jpg"],
        },
        {
            "id": 3, "dataset": "scannet", "scene_name": "scene0002_00",
            "question_type": "object_rel_distance", "question": "Which object is closer?",
            "ground_truth": "B", "options": ["A. chair", "B. table"], "images": ["300.jpg"],
        },
        {
            "id": 4, "dataset": "scannet", "scene_name": "scene0002_00",
            "question_type": "object_rel_direction", "question": "Which direction?",
            "ground_truth": "C", "options": ["A. left-front", "B. left-back", "C. right-front", "D. right-back"],
            "images": ["400.jpg"],
        },
    ]


def _jpeg(size: tuple[int, int], color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_archive(root: Path, *, extra: bool = False) -> Path:
    archive = root / "images.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        specifications = [
            ("root/scene0001_00/100.jpg", (512, 512), "red"),
            ("root/scene0001_00/200.jpg", (640, 480), "blue"),
            ("root/scene0002_00/300.jpg", (512, 512), "green"),
            ("root/scene0002_00/400.jpg", (640, 480), "yellow"),
        ]
        if extra:
            specifications.append(("root/scene0002_00/extra.jpg", (10, 10), "black"))
        for name, size, color in specifications:
            handle.writestr(name, _jpeg(size, color))
    return archive


def small_contract(root: Path, *, extra: bool = False) -> SPBenchSITestContract:
    root.mkdir(parents=True, exist_ok=True)
    parquet = root / "synthetic.parquet"
    parquet.touch()
    return SPBenchSITestContract(
        parquet,
        make_archive(root, extra=extra),
        rows=rows(),
        require_official_size=False,
        verify_files=False,
        verify_images=True,
    )
