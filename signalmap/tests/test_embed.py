"""Embedder + anomaly scoring: the scoring core every entry point routes through."""
import numpy as np

from signalmap.dsp import raw_to_features
from signalmap.embed import Embedder, anomaly_score
from signalmap.model import SpectralAutoencoder

N_BINS = 256
SR = 16_000


def _feat(amp: float, freq: float = 440.0, n: int = 512):
    t = np.arange(n) / SR
    return raw_to_features((amp * np.sin(2 * np.pi * freq * t)).astype(np.float32),
                           SR, n_bins=N_BINS)


def _embedder():
    return Embedder(SpectralAutoencoder(n_bins=N_BINS, latent_dim=32))


def test_embed_returns_latent_vector_and_carries_frame_identity():
    emb, _ = _embedder().embed(_feat(300.0), node_id=7, seq=42, ts_us=123)
    assert emb.vector.shape == (32,)
    assert (emb.node_id, emb.seq, emb.ts_us) == (7, 42, 123)
    assert emb.recon_error >= 0.0
    assert emb.energy_rms > 0.0


def test_first_frame_has_no_energy_baseline_so_z_is_zero():
    # One sample cannot be an outlier of itself — the running baseline must not
    # invent a deviation on frame 1 (that would flag every stream's first frame).
    _emb, z = _embedder().embed(_feat(300.0), 0, 0, 0)
    assert z == 0.0


def test_constant_energy_stream_never_accumulates_a_deviation():
    e = _embedder()
    zs = [e.embed(_feat(300.0), 0, i, 0)[1] for i in range(8)]
    assert max(zs) < 1e-6


def test_energy_outlier_scores_above_the_baseline_frames():
    e = _embedder()
    quiet = [e.embed(_feat(300.0), 0, i, 0)[1] for i in range(10)]
    _emb, loud = e.embed(_feat(4000.0), 0, 10, 0)
    assert loud > max(quiet)


def test_anomaly_score_is_monotone_in_each_factor():
    base = anomaly_score(0.5, 0.5, 0.5)
    assert anomaly_score(1.0, 0.5, 0.5) > base
    assert anomaly_score(0.5, 1.0, 0.5) > base
    assert anomaly_score(0.5, 0.5, 1.0) > base


def test_anomaly_score_stays_positive_when_a_single_factor_is_zero():
    # The epsilons exist so one zero factor cannot erase the other evidence.
    assert anomaly_score(0.0, 0.7, 0.7) > 0.0
    assert anomaly_score(0.7, 0.0, 0.7) > 0.0
    assert anomaly_score(0.7, 0.7, 0.0) > 0.0
