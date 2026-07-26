"""Small, fail-closed checks for local model and upstream revisions."""

from __future__ import annotations

import subprocess
from pathlib import Path


def verify_hf_snapshot_revision(path_or_id: str | Path, expected: str, label: str) -> bool:
    """Validate a Hugging Face snapshot path or ``local_dir`` download metadata.

    ``snapshot_download`` embeds the commit in ``snapshots/<sha>``. Hugging Face's
    ``local_dir`` mode instead writes one ``.metadata`` sidecar per downloaded file
    under ``.cache/huggingface/download``; the first line is the resolved commit.
    A model id or an ordinary directory with neither form of evidence returns
    ``False`` so an unverifiable custom copy cannot be mistaken for a locked one.
    """

    path = Path(path_or_id).expanduser()
    if not path.exists():
        return False
    resolved = path.resolve()
    parts = resolved.parts
    if "snapshots" not in parts:
        metadata_root = resolved / ".cache" / "huggingface" / "download"
        metadata_files = sorted(metadata_root.rglob("*.metadata")) if metadata_root.is_dir() else []
        if not metadata_files:
            return False
        revisions: set[str] = set()
        for metadata_path in metadata_files:
            with metadata_path.open("r", encoding="utf-8") as handle:
                revision = handle.readline().strip()
            if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
                raise ValueError(f"{label} has malformed Hugging Face metadata: {metadata_path}")
            revisions.add(revision)
        if revisions != {expected}:
            found = ", ".join(sorted(revisions))
            raise ValueError(f"{label} local-dir revisions are [{found}], expected {expected}")
        return True
    position = parts.index("snapshots")
    if position + 1 >= len(parts):
        raise ValueError(f"{label} has a malformed Hugging Face snapshot path: {path}")
    actual = parts[position + 1]
    if actual != expected:
        raise ValueError(f"{label} snapshot revision is {actual}, expected {expected}")
    return True


def verify_git_checkout(path: str | Path, expected: str, label: str) -> bool:
    """Require the expected HEAD when ``path`` is a Git checkout.

    Source archives without ``.git`` are permitted but reported as unverified in
    run metadata. Git command failures are fatal because silently continuing would
    record provenance known to be unreliable.
    """

    root = Path(path).expanduser().resolve()
    if not (root / ".git").exists():
        return False
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != expected:
        raise ValueError(f"{label} checkout is at {actual}, expected {expected}")
    return True
