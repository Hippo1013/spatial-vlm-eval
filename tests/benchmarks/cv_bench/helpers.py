from __future__ import annotations

from PIL import Image

from spatial_vlm_eval.benchmarks.cv_bench.data import CVBenchTestContract


class FakeDataset:
    column_names = [
        "type",
        "task",
        "image",
        "question",
        "choices",
        "answer",
        "prompt",
        "filename",
        "source",
        "source_dataset",
        "source_filename",
        "target_class",
        "target_size",
        "bbox",
    ]

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_row(split_type, task, source, *, answer="(A)", choices=None, color=(10, 20, 30)):
    choices = list(choices or ["left", "right"])
    question = f"Synthetic {task} question?"
    options = "\n".join(f"({chr(65 + index)}) {choice}" for index, choice in enumerate(choices))
    prompt = f"{question} Select from the following choices.\n{options}"
    return {
        "type": split_type,
        "task": task,
        "image": Image.new("RGB", (8, 6), color),
        "question": question,
        "choices": choices,
        "answer": answer,
        "prompt": prompt,
        "filename": f"{split_type}/{task}.png",
        "source": source,
        "source_dataset": f"{source} source",
        "source_filename": "source.png",
        "target_class": None,
        "target_size": None,
        "bbox": None,
    }


def official_fake_contract(tmp_path):
    rows_2d = []
    rows_2d.extend(make_row("2D", "Count", "ADE20K") for _ in range(342))
    rows_2d.extend(make_row("2D", "Relation", "ADE20K") for _ in range(291))
    rows_2d.extend(make_row("2D", "Count", "COCO") for _ in range(446))
    rows_2d.extend(make_row("2D", "Relation", "COCO") for _ in range(359))
    rows_3d = [make_row("3D", "Depth", "Omni3D") for _ in range(600)]
    rows_3d.extend(make_row("3D", "Distance", "Omni3D") for _ in range(600))
    return CVBenchTestContract(
        tmp_path,
        split_datasets={"2D": FakeDataset(rows_2d), "3D": FakeDataset(rows_3d)},
        require_official_size=True,
        verify_files=False,
    )
