"""Fresh-checkout guard: SignalMap's default outputs live under gitignored
`data/` and `artifacts/` dirs. Every writer must create its parent dir so the
README quickstart works on a clean `git clone` (L3 fresh-checkout replay found
benchmark/universal/train crashing with FileNotFoundError here)."""
import os

import numpy as np

from signalmap._io import ensure_parent


def test_ensure_parent_creates_missing_dir(tmp_path):
    p = tmp_path / "data" / "sub" / "x.parquet"
    assert not p.parent.exists()
    ensure_parent(str(p))
    assert p.parent.is_dir()


def test_ensure_parent_noop_for_bare_filename(tmp_path):
    # a plain filename (no dir component) must not raise
    ensure_parent("x.parquet")


def test_build_pdm_benchmark_creates_parent(tmp_path):
    """The `signalmap benchmark` path: build_pdm_benchmark writes into a
    (fresh-checkout-missing) directory."""
    from signalmap.synth import build_pdm_benchmark

    out = tmp_path / "data" / "_benchmark.parquet"
    assert not out.parent.exists()
    n = build_pdm_benchmark(str(out), normal=2, faults=1, seed=1)
    assert out.exists()
    assert n == 3
