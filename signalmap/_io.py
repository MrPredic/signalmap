"""Small shared IO helpers."""
import os


def ensure_parent(path: str) -> None:
    """Create the parent directory of an output file if it doesn't exist.

    SignalMap's default output paths live under gitignored `data/` and
    `artifacts/` dirs that a fresh `git clone` / sdist install does not have.
    Every command that writes a file calls this first so a fresh checkout never
    crashes with a FileNotFoundError (or torch's "Parent directory ... does not
    exist") after doing real work.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
