"""Run native Qwen2.5-VL/PEFT inference on the MSMU-Bench test split.

Example:
  CUDA_VISIBLE_DEVICES=0 python -m spatial_vlm_eval.models.qwen25_vl.peft_infer \
    --base-model /models/Qwen2.5-VL-7B-Instruct \
    --dataset-root /datasets/MSMU \
    --checkpoint /checkpoints/adapter \
    --output /outputs/msmu/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import peft
import transformers
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ...benchmarks.msmu.data import MSMUArrowDataset, QwenGenerationCollator, task_family


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", default=None, help="PEFT adapter checkpoint; omit for base.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--image-min-pixels", type=int, default=12544)
    parser.add_argument("--image-max-pixels", type=int, default=112896)
    parser.add_argument(
        "--run-metadata",
        default=None,
        help="Defaults to <output>.metadata.json.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    dataset = MSMUArrowDataset(args.dataset_root, split="test", limit=args.limit)
    collator = QwenGenerationCollator(
        processor,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        device_map={"": 0},
    )
    if args.checkpoint:
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="MSMU-Bench generation"):
            inputs = {
                key: value.to(model.device, non_blocking=True)
                for key, value in batch["model_inputs"].items()
            }
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for index, raw_type, question, reference, prediction in zip(
                batch["indices"],
                batch["raw_types"],
                batch["questions"],
                batch["references"],
                predictions,
            ):
                rows.append(
                    {
                        "index": int(index),
                        "raw_type": raw_type,
                        "task_family": task_family(raw_type),
                        "question": question,
                        "reference": reference,
                        "prediction": prediction.strip(),
                    }
                )

    rows.sort(key=lambda row: row["index"])
    output_path = Path(args.output).resolve()
    write_jsonl(output_path, rows)
    metadata_path = (
        Path(args.run_metadata).resolve()
        if args.run_metadata
        else output_path.with_suffix(output_path.suffix + ".metadata.json")
    )
    metadata = {
        "protocol": "msmu_qwen25_vl_deterministic_v1",
        "base_model": str(args.base_model),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": "test",
        "output": str(output_path),
        "num_predictions": len(rows),
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "chat_template": processor.chat_template,
        "system_prompt": (
            "You are a helpful assistant."
            if "You are a helpful assistant." in str(processor.chat_template)
            else None
        ),
        "image_processing": {
            "min_pixels": args.image_min_pixels,
            "max_pixels": args.image_max_pixels,
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        },
        "batch_size": args.batch_size,
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} predictions to {args.output}")
    print(f"Wrote run metadata to {metadata_path}")


if __name__ == "__main__":
    main()
