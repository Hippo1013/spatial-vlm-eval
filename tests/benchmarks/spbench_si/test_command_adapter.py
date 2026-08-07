from __future__ import annotations

import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.spbench_si.command_adapter import UpstreamCommandAdapter
from spatial_vlm_eval.benchmarks.spbench_si.data import SPBenchSIModelInput, SYSTEM_PROMPT
from spatial_vlm_eval.benchmarks.spbench_si.profiles import PROFILES
from spatial_vlm_eval.benchmarks.spbench_si.specialized_runner import _prepare_spatialladder_config


class _Input:
    def __init__(self): self.lines = []; self.closed = False
    def write(self, value): self.lines.append(value)
    def flush(self): pass
    def close(self): self.closed = True


class _Output:
    def __init__(self, source): self.source = source; self.closed = False
    def readline(self):
        request = json.loads(self.source.lines[-1])
        folded = request["system_prompt"] + "\n\n" + request["user_prompt"]
        sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
        return json.dumps({
            "index": request["index"], "profile": request["profile"],
            "model_revision": request["model_revision"], "inference_protocol": request["inference_protocol"],
            "decoding": request["decoding"], "system_role_supported": False,
            "raw_prediction": "B", "generation": {
                "num_model_image_tensors": 1, "source_rgb_count": 1,
                "source_rgb_sha256": request["image"]["pixel_sha256"],
                "template_sha256": "a" * 64, "system_prompt_sha256": sha(request["system_prompt"]),
                "user_prompt_sha256": sha(request["user_prompt"]), "folded_prompt_sha256": sha(folded),
            },
        }) + "\n"
    def close(self): self.closed = True


class _Process:
    def __init__(self):
        self.stdin = _Input(); self.stdout = _Output(self.stdin); self.returncode = None
    def poll(self): return self.returncode
    def wait(self, timeout=None): self.returncode = 0; return 0
    def terminate(self): self.returncode = -15
    def kill(self): self.returncode = -9


class SPBenchSICommandAdapterTest(unittest.TestCase):
    def test_spatialladder_preserves_tied_qwen25_text_config(self):
        text_config = SimpleNamespace(tie_word_embeddings=True)
        config = SimpleNamespace(text_config=text_config, tie_word_embeddings=False)
        self.assertIs(_prepare_spatialladder_config(config), config)
        self.assertTrue(config.tie_word_embeddings)
        text_config.tie_word_embeddings = False
        with self.assertRaisesRegex(ValueError, "tied text output embeddings"):
            _prepare_spatialladder_config(config)

    def test_bridge_request_has_only_safe_fields_and_same_rgb_evidence(self):
        process = _Process()
        profile = PROFILES["spatialrgpt_rgb"]
        with patch("subprocess.Popen", return_value=process):
            adapter = UpstreamCommandAdapter(profile=profile, command="python runner.py", adapter_digest="b" * 64, decoding=profile.decoding)
            result = adapter.generate(SPBenchSIModelInput(2, Image.new("RGB", (4, 3), "blue"), SYSTEM_PROMPT, "Question: q"))
            adapter.close()
        request = json.loads(process.stdin.lines[0])
        self.assertEqual(request["image"]["count"], 1)
        self.assertEqual(result.text, "B")
        for forbidden in ("ground_truth", "question_type", "scene", "scene_name", "dataset", "row"):
            self.assertNotIn(forbidden, request)


if __name__ == "__main__":
    unittest.main()
