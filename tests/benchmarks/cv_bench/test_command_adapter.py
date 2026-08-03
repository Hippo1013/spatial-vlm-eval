from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from spatial_vlm_eval.benchmarks.cv_bench.command_adapter import (
    UpstreamCommandAdapter,
    load_generation_manifest,
)
from spatial_vlm_eval.benchmarks.cv_bench.data import CVBenchModelInput
from spatial_vlm_eval.benchmarks.cv_bench.profiles import PROFILES
from spatial_vlm_eval.benchmarks.cv_bench.specialized_runner import (
    MOGE2_CHECKPOINT_FILENAME,
    MOGE2_UTILS3D_COMMIT,
    _build_backend,
    _prepare_spatialladder_config,
    adapter_digest,
)
from spatial_vlm_eval.models.spatialbot.infer import ZOEDEPTH_REVISION


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
    def test_moge_checkpoint_filename_is_locked(self):
        self.assertEqual(MOGE2_CHECKPOINT_FILENAME, "model.pt")
        self.assertEqual(
            MOGE2_UTILS3D_COMMIT, "3fab839f0be9931dac7c8488eb0e1600c236e183"
        )

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

    def test_committed_generation_manifests_resolve_locked_defaults(self):
        repository = Path(__file__).resolve().parents[3]
        manifest_root = repository / "configs" / "cv-bench-generation"
        for key in (
            "ssr_rgb",
            "ssr_native",
            "3dthinker_rgb",
            "spatialladder3b_rgb",
            "spatialladder3b_thinking",
        ):
            with self.subTest(profile=key):
                resolved = load_generation_manifest(
                    PROFILES[key], manifest_root / f"{key}.json"
                )
                self.assertIn("do_sample", resolved)
                self.assertGreater(resolved["max_new_tokens"], 0)
                self.assertEqual(resolved["seed"], 42)

    def test_specialized_adapter_digest_is_explicit_and_family_sensitive(self):
        spatialrgpt = adapter_digest(PROFILES["spatialrgpt_rgb"])
        spatialbot = adapter_digest(PROFILES["spatialbot_rgb"])
        self.assertRegex(spatialrgpt, r"^[0-9a-f]{64}$")
        self.assertRegex(spatialbot, r"^[0-9a-f]{64}$")
        self.assertNotEqual(spatialrgpt, spatialbot)

    def test_spatialladder_runner_propagates_tied_qwen25_text_config(self):
        text_config = SimpleNamespace(
            tie_word_embeddings=True, to_dict=lambda: {"hidden_size": 2048}
        )
        config = SimpleNamespace(text_config=text_config, tie_word_embeddings=False)
        self.assertIs(_prepare_spatialladder_config(config), config)
        self.assertTrue(config.tie_word_embeddings)

        text_config.tie_word_embeddings = False
        with self.assertRaisesRegex(ValueError, "tied text output embeddings"):
            _prepare_spatialladder_config(config)

        class FlatConfig(SimpleNamespace):
            sub_configs = {"vision_config": object}

        flat = FlatConfig(
            text_config={"tie_word_embeddings": True}, tie_word_embeddings=False
        )
        _prepare_spatialladder_config(flat)
        self.assertTrue(flat.tie_word_embeddings)
        self.assertFalse(hasattr(flat, "text_config"))

    def test_spatialbot_zoedepth_runner_binds_upstream_revision(self):
        profile = PROFILES["spatialbot_zoedepth"]
        environment = {
            profile.model_path_env: "/locked/spatialbot",
            "SPATIALBOT_UPSTREAM_ROOT": "/locked/spatialbot-upstream",
            "SPATIALBOT_SIGLIP_MODEL": "/locked/siglip",
            "SPATIALBOT_MIDAS_ROOT": "/locked/midas",
            "ZOEDEPTH_ROOT": "/locked/zoedepth-upstream",
            "ZOEDEPTH_CHECKPOINT": "/locked/zoedepth.pt",
        }
        with patch.dict(os.environ, environment, clear=False), patch(
            "spatial_vlm_eval.models.spatialbot.infer.SpatialBotAdapter"
        ) as adapter:
            _build_backend(profile, dict(profile.decoding))
        self.assertEqual(adapter.call_args.kwargs["zoedepth_revision"], ZOEDEPTH_REVISION)
        self.assertEqual(adapter.call_args.kwargs["siglip_model"], "/locked/siglip")
        self.assertEqual(adapter.call_args.kwargs["midas_root"], "/locked/midas")


if __name__ == "__main__":
    unittest.main()
