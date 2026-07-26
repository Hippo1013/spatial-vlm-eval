import tempfile
import unittest
from pathlib import Path

from spatial_vlm_eval.models.common.provenance import verify_hf_snapshot_revision


class LocalRevisionVerificationTest(unittest.TestCase):
    def test_hf_snapshot_hash_is_verified_or_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = "a" * 40
            snapshot = Path(directory) / "models--org--name" / "snapshots" / expected
            snapshot.mkdir(parents=True)
            self.assertTrue(verify_hf_snapshot_revision(snapshot, expected, "model"))
            with self.assertRaisesRegex(ValueError, "expected"):
                verify_hf_snapshot_revision(snapshot, "b" * 40, "model")

    def test_ordinary_local_directory_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(verify_hf_snapshot_revision(directory, "a" * 40, "model"))

    def test_hf_local_dir_metadata_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = "a" * 40
            metadata = Path(directory) / ".cache" / "huggingface" / "download" / "config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(f"{expected}\nblob-id\n123.0\n", encoding="utf-8")
            self.assertTrue(verify_hf_snapshot_revision(directory, expected, "model"))

    def test_hf_local_dir_mixed_or_wrong_revisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_root = Path(directory) / ".cache" / "huggingface" / "download"
            metadata_root.mkdir(parents=True)
            (metadata_root / "config.json.metadata").write_text(
                f"{'a' * 40}\nblob-id\n123.0\n",
                encoding="utf-8",
            )
            (metadata_root / "tokenizer.json.metadata").write_text(
                f"{'b' * 40}\nblob-id\n123.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "local-dir revisions"):
                verify_hf_snapshot_revision(directory, "a" * 40, "model")


if __name__ == "__main__":
    unittest.main()
