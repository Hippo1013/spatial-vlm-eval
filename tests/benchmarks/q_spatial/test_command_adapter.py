from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.q_spatial.command_adapter import UpstreamCommandAdapter
from spatial_vlm_eval.benchmarks.q_spatial.data import QSpatialModelInput, STANDARD_SYSTEM_PROMPT
from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILES


class _InputPipe:
    def __init__(self):
        self.lines = []
        self.closed = False

    def write(self, value):
        self.lines.append(value)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class _OutputPipe:
    def __init__(self, input_pipe):
        self.input_pipe = input_pipe
        self.closed = False

    def readline(self):
        request = json.loads(self.input_pipe.lines[-1])
        folded = request["system_prompt"] + "\n\n" + request["user_prompt"]
        sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
        return json.dumps({
            "index": request["index"],
            "profile": request["profile"],
            "model_revision": request["model_revision"],
            "inference_protocol": request["inference_protocol"],
            "decoding": request["decoding"],
            "system_role_supported": False,
            "raw_prediction": "2 meters",
            "generation": {
                "num_model_image_tensors": 1,
                "template_sha256": "a" * 64,
                "system_prompt_sha256": sha(request["system_prompt"]),
                "user_prompt_sha256": sha(request["user_prompt"]),
                "folded_prompt_sha256": sha(folded),
            },
        }) + "\n"

    def close(self):
        self.closed = True


class _Process:
    def __init__(self):
        self.stdin = _InputPipe()
        self.stdout = _OutputPipe(self.stdin)
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class QSpatialCommandAdapterTest(unittest.TestCase):
    def test_bridge_sends_separate_prompts_one_image_and_no_scoring_fields(self):
        process = _Process()
        profile = PROFILES["spatialrgpt_rgb"]
        with patch("subprocess.Popen", return_value=process):
            adapter = UpstreamCommandAdapter(
                profile=profile,
                command="python runner.py",
                adapter_digest="b" * 64,
                decoding=profile.decoding,
            )
            result = adapter.generate(QSpatialModelInput(
                9,
                Image.new("RGB", (4, 3), "blue"),
                STANDARD_SYSTEM_PROMPT,
                "Question: How far?",
            ))
            adapter.close()
        request = json.loads(process.stdin.lines[0])
        self.assertEqual(request["system_prompt"], STANDARD_SYSTEM_PROMPT)
        self.assertEqual(request["user_prompt"], "Question: How far?")
        self.assertEqual(request["image"]["count"], 1)
        self.assertEqual(request["image"]["mode"], "RGB")
        for forbidden in ("answer", "answer_value", "answer_unit", "question_type", "split"):
            self.assertNotIn(forbidden, request)
        self.assertEqual(result.text, "2 meters")
        self.assertEqual(result.metadata["num_model_image_tensors"], 1)


if __name__ == "__main__":
    unittest.main()
