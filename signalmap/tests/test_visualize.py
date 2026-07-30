"""The latent-map HTML: self-contained, and it must not fall over on tiny datasets."""
import json

import numpy as np
import pytest
import torch

from signalmap import visualize
from signalmap.model import SpectralAutoencoder

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SR = 16_000


def _dataset(path, n_rows):
    rows = []
    for i in range(n_rows):
        t = np.arange(512) / SR
        sig = 300 * np.sin(2 * np.pi * (200 + 37 * i) * t)
        rows.append(np.clip(sig, -2048, 2047).astype("<i2").tobytes())
    pq.write_table(pa.table({
        "label": [f"mat{i}" for i in range(n_rows)],
        "sr_hz": [SR] * n_rows,
        "samples": rows,
    }), path)
    return str(path)


def _model(path):
    torch.save(SpectralAutoencoder(n_bins=visualize.N_BINS, latent_dim=32).state_dict(),
               str(path))
    return str(path)


def _points(html_path):
    html = open(html_path).read()
    start = html.index("const data = ") + len("const data = ")
    return json.loads(html[start:html.index("\n", start)].rstrip(";"))


def test_build_writes_one_self_contained_point_per_recording(tmp_path):
    out = tmp_path / "art" / "map.html"      # nested: dir must be created for us
    visualize.build(_dataset(tmp_path / "ds.parquet", 12),
                    _model(tmp_path / "m.pt"), str(out))
    pts = _points(out)
    assert len(pts) == 12
    assert {p["label"] for p in pts} == {f"mat{i}" for i in range(12)}
    assert all(np.isfinite([p["x"], p["y"], p["score"]]).all() for p in pts)
    html = open(out).read()                    # no external fetches in the asset
    assert "src=" not in html and "href=" not in html and "fetch(" not in html


def test_build_survives_a_single_recording(tmp_path):
    # A 1-row dataset is what a user's very first capture looks like. The 2D
    # projection has no second component there, and must not raise.
    out = tmp_path / "one.html"
    visualize.build(_dataset(tmp_path / "one.parquet", 1),
                    _model(tmp_path / "m.pt"), str(out))
    pts = _points(out)
    assert len(pts) == 1
    assert np.isfinite([pts[0]["x"], pts[0]["y"]]).all()


def test_build_survives_two_recordings(tmp_path):
    out = tmp_path / "two.html"
    visualize.build(_dataset(tmp_path / "two.parquet", 2),
                    _model(tmp_path / "m.pt"), str(out))
    assert len(_points(out)) == 2


def test_at_least_one_point_is_flagged_so_the_asset_is_never_empty(tmp_path):
    out = tmp_path / "flag.html"
    visualize.build(_dataset(tmp_path / "ds.parquet", 8),
                    _model(tmp_path / "m.pt"), str(out))
    assert sum(p["anomaly"] for p in _points(out)) >= 1


def test_projection_falls_back_to_pca_without_umap():
    vecs = np.random.default_rng(0).standard_normal((20, 32))
    coords = visualize._project_2d(vecs)
    assert coords.shape == (20, 2)
    assert np.isfinite(coords).all()
