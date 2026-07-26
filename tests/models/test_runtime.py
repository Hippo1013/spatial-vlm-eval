import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from spatial_vlm_eval.benchmarks.msmu.data import MSMUTestContract
from spatial_vlm_eval.models.common.runtime import (
    GenerationResult,
    InferenceAdapter,
    PredictionJournal,
    input_audit,
    run_msmu_inference,
    select_target_indices,
)


def source_row(index: int) -> dict:
    return {
        "image": Image.new("RGB", (3, 2), (index, 20, 40)),
        "type": "width",
        "conversations": {
            "from": ["human", "gpt"],
            "value": [f"<image>\nQuestion {index}?", f"reference secret {index}"],
        },
    }


class FakeAdapter(InferenceAdapter):
    supports_concurrency = True

    def __init__(self, failures=None, empty_indices=None):
        self.failures = dict(failures or {})
        self.empty_indices = set(empty_indices or [])
        self.calls = []

    def metadata(self):
        return {
            "model": "fake/model",
            "model_revision": "abc123",
            "backend": "fake",
            "profile": "question_only",
            "inference_protocol": "msmu_fake_question_only_v1",
            "chat_template": "one user message",
            "image_processing": {"image_count": 1},
            "decoding": {"max_new_tokens": 192, "do_sample": False},
            "upstream": {"commit": "abc123"},
        }

    def generate(self, model_input):
        self.calls.append(model_input.index)
        remaining = self.failures.get(model_input.index, 0)
        if remaining:
            self.failures[model_input.index] = remaining - 1
            raise RuntimeError("temporary network failure with Bearer top-secret")
        text = "" if model_input.index in self.empty_indices else f"answer {model_input.index}"
        warning = ("model returned an empty text completion",) if not text else ()
        return GenerationResult(text, {"request_id": f"req-{model_input.index}"}, warning)


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.contract = MSMUTestContract(
            "MSMU",
            dataset=[source_row(0), source_row(1), source_row(2)],
            require_official_size=False,
        )

    def test_atomic_finalization_schema_audit_and_empty_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            adapter = FakeAdapter(empty_indices={1})
            metadata = run_msmu_inference(
                contract=self.contract,
                adapter=adapter,
                output=output,
                target_indices=[2, 0, 1],
                workers=2,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["index"] for row in rows], [0, 1, 2])
            self.assertTrue(all(len(row) == 6 for row in rows))
            self.assertEqual(rows[1]["prediction"], "")
            self.assertEqual(metadata["empty_prediction_indices"], [1])
            self.assertTrue(metadata["dataset"]["is_subset"])
            self.assertFalse(metadata["publishable_inference"])
            journal_rows = [
                json.loads(line)
                for line in Path(metadata["journal"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(journal_rows), 3)
            for event in journal_rows:
                audit = event["audit"]
                self.assertEqual(audit["image_count"], 1)
                self.assertEqual(len(audit["image_pixel_sha256"]), 64)
                self.assertNotIn("reference", audit)
                self.assertNotIn("base64", json.dumps(audit))

    def test_failures_leave_only_journal_then_resume_missing_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            journal = output.with_suffix(".jsonl.journal.jsonl")
            first = FakeAdapter(failures={1: 5})
            with self.assertRaisesRegex(RuntimeError, "No prediction JSONL was finalized"):
                run_msmu_inference(
                    contract=self.contract,
                    adapter=first,
                    output=output,
                    target_indices=[0, 1],
                    retries=1,
                )
            self.assertFalse(output.exists())
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([event["status"] for event in events], ["success", "failure", "failure"])
            self.assertNotIn("top-secret", json.dumps(events))
            self.assertIn("[REDACTED]", json.dumps(events))

            second = FakeAdapter()
            run_msmu_inference(
                contract=self.contract,
                adapter=second,
                output=output,
                target_indices=[0, 1],
                retries=0,
            )
            self.assertEqual(second.calls, [1])
            self.assertTrue(output.exists())

    def test_retry_success_is_not_converted_to_empty_text(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter(failures={0: 1})
            output = Path(directory) / "predictions.jsonl"
            run_msmu_inference(
                contract=self.contract,
                adapter=adapter,
                output=output,
                target_indices=[0],
                retries=1,
            )
            row = json.loads(output.read_text())
            self.assertEqual(row["prediction"], "answer 0")
            self.assertEqual(adapter.calls, [0, 0])

    def test_duplicate_success_in_journal_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter()
            metadata = adapter.metadata()
            audit = input_audit(self.contract.model_input(0), metadata)
            journal = PredictionJournal(Path(directory) / "journal.jsonl", "signature", resume=True)
            result = GenerationResult("answer")
            journal.append_success(
                model_input=self.contract.model_input(0), attempt=1, audit=audit, result=result
            )
            journal.append_success(
                model_input=self.contract.model_input(0), attempt=2, audit=audit, result=result
            )
            with self.assertRaisesRegex(ValueError, "Duplicate successful"):
                journal.successful_results({0})

    def test_resume_signature_covers_full_adapter_provenance(self):
        class VariantAdapter(FakeAdapter):
            def __init__(self, component, **kwargs):
                super().__init__(**kwargs)
                self.component = component

            def metadata(self):
                metadata = super().metadata()
                metadata["component_revision"] = self.component
                return metadata

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            with self.assertRaises(RuntimeError):
                run_msmu_inference(
                    contract=self.contract,
                    adapter=VariantAdapter("revision-a", failures={1: 5}),
                    output=output,
                    target_indices=[0, 1],
                    retries=0,
                )
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                run_msmu_inference(
                    contract=self.contract,
                    adapter=VariantAdapter("revision-b"),
                    output=output,
                    target_indices=[0, 1],
                    retries=0,
                )

    def test_debug_index_parser_rejects_duplicates_and_applies_limit(self):
        self.assertEqual(select_target_indices(10, indices="1,3-5", limit=3), [1, 3, 4])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_target_indices(10, indices="1,1")

    def test_output_journal_and_metadata_paths_cannot_collide(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            with self.assertRaisesRegex(ValueError, "distinct"):
                run_msmu_inference(
                    contract=self.contract,
                    adapter=FakeAdapter(),
                    output=output,
                    target_indices=[0],
                    journal_path=output,
                )


if __name__ == "__main__":
    unittest.main()
