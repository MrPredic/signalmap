"""Multichannel ingest: turn real synchronized recordings (CSV / NPY) into the
``dict[str, np.ndarray]`` channel form that causal discovery consumes. This is
the bridge from the single-channel frame pipeline to CCM on real sensor arrays.
"""
from __future__ import annotations

import numpy as np
import pytest

from signalmap.multichannel import load_channels


def test_load_channels_csv_named(tmp_path):
    p = tmp_path / "rec.csv"
    p.write_text("A,B,C\n1,2,3\n4,5,6\n7,8,9\n")
    ch = load_channels(p)
    assert set(ch) == {"A", "B", "C"}
    assert np.allclose(ch["A"], [1, 4, 7])
    assert np.allclose(ch["C"], [3, 6, 9])
    assert all(len(v) == 3 for v in ch.values())


def test_load_channels_select_columns(tmp_path):
    p = tmp_path / "rec.csv"
    p.write_text("A,B,C\n1,2,3\n4,5,6\n")
    ch = load_channels(p, columns=["A", "C"])
    assert set(ch) == {"A", "C"}


def test_load_channels_drops_nan_rows_keeping_sync(tmp_path):
    p = tmp_path / "rec.csv"
    p.write_text("A,B\n1,2\n,5\n7,8\n")  # middle row has a missing A
    ch = load_channels(p)
    # The whole row is dropped so channels stay aligned and equal length.
    assert np.allclose(ch["A"], [1, 7])
    assert np.allclose(ch["B"], [2, 8])


def test_load_channels_requires_two_channels(tmp_path):
    p = tmp_path / "one.csv"
    p.write_text("A\n1\n2\n")
    with pytest.raises(ValueError):
        load_channels(p)


def test_load_channels_npy(tmp_path):
    p = tmp_path / "rec.npy"
    np.save(p, np.array([[1.0, 2.0], [3.0, 4.0]]))
    ch = load_channels(p, columns=["x", "y"])
    assert np.allclose(ch["x"], [1, 3])
    assert np.allclose(ch["y"], [2, 4])


def test_csv_to_causal_graph_end_to_end(tmp_path):
    # A real-file path that recovers the directed chain proves the bridge works.
    from signalmap.causal import causal_graph
    from signalmap.causal_discover import _chain

    ch = _chain(n=1500, b_ab=0.1, b_bc=0.1, seed=0)
    p = tmp_path / "chain.csv"
    header = ",".join(ch)
    rows = np.column_stack([ch[k] for k in ch])
    lines = [header] + [",".join(f"{v:.8f}" for v in row) for row in rows]
    p.write_text("\n".join(lines) + "\n")

    loaded = load_channels(p)
    edges = causal_graph(loaded, E=3, tau=1, min_strength=0.3)
    found = {(e["cause"], e["effect"]) for e in edges}
    assert ("A", "B") in found and ("B", "C") in found
    assert ("C", "B") not in found and ("B", "A") not in found
