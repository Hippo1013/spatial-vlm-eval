from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from spatial_vlm_eval.benchmarks.cv_bench.command_adapter import (
    UpstreamCommandAdapter,
    load_generation_manifest,
)
from spatial_vlm_eval.benchmarks.cv_bench.data import CVBenchModelInput
from spatial_vlm_eval.benchmarks.cv_bench.profiles import PROFILES


RUNNER_SOURCE = r'''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request.get("action") == "close":
        break
    response = {
        "index": request["index"],
        "profile": request["profile"],
        "model_revision": request["model_revision"],
        "inference_protocol": request["inference_protocol"],
        "decoding": request["decoding"],
        "raw_prediction": json.dumps(request, sort_keys=True),
        "generation": {
            "num_model_image_tensors": 1,
            "template_sha256": "0" * 64,
        },
    }
    print(json.dumps(response), flush=True)
'''


class CVBenchCommandAdapterTest(unittest.TestCase):
    def test_bridge_request_contains_one_image_and_no_scoring_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / "runner.py"
            runner.write_text(RUNNER_SOURCE, encoding="utf-8")
            profile = PROFILES["spatialrgpt_rgb"]
            adapter = UpstreamCommandAdapter(
                profile=profile,
                command=f"{sys.executable} -u {runner}",
                adapter_digest="a" * 64,
                decoding=profile.decoding,
            )
            result = adapter.generate(
                CVBenchModelInput(
                    index=7,
                    image=Image.new("RGB", (9, 5), (1, 2, 3)),
                    question="Question and choices only",
                )
            )
            adapter.close()
            request = json.loads(result.text)
            self.assertEqual(request["index"], 7)
            self.assertEqual(request["prompt"], "Question and choices only")
            self.assertEqual(request["image"]["count"], 1)
            self.assertEqual(request["image"]["mode"], "RGB")
            self.assertTrue(request["image"]["png_data_uri"].startswith("data:image/png;base64,"))
            for forbidden in ("answer", "gold", "task", "source", "choices"):
                self.assertNotIn(forbidden, request)
            self.assertEqual(result.metadata["num_model_image_tensors"], 1)

    def test_unresolved_generation_defaults_require_bound_manifest(self):
        profile = PROFILES["ssr_rgb"]
        with self.assertRaisesRegex(ValueError, "requires a locked runtime generation manifest"):
            load_generation_manifest(profile, None)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "generation.json"
            manifest.write_text(
                json.dumps(
                    {
                        "profile": profile.key,
                        "model_revision": profile.revision,
                        "upstream_commit": profile.upstream_commit,
                        "decoding": {
                            "do_sample": False,
                            "max_new_tokens": 128,
                            "seed": 42,
                        },
                    }
                ),
                encoding="utf-8",
            )
            resolved = load_generation_manifest(profile, manifest)
            self.assertFalse(resolved["do_sample"])
            self.assertEqual(resolved["max_new_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
