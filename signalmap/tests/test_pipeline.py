import struct

import numpy as np

from signalmap.dsp import raw_to_features
from signalmap.frame import GapTracker, decode
from signalmap.model import SpectralAutoencoder


def _raw(node_id, seq, samples, sr=16000):
    hdr = struct.pack("<HBBIIQIHH", 0x5247, 1, 0, node_id, seq, 0, sr, len(samples), 0)
    return hdr + np.asarray(samples, dtype="<i2").tobytes()


def test_pipeline_propagates_energy():
    """Raw energy must survive end-to-end (it is signal, not nuisance)."""
    from signalmap.core import Pipeline
    from signalmap.models import AutoencoderModel
    from signalmap.sources import SimulatorSource
    from signalmap.transforms import FFTTransform

    captured = []

    class CaptureSink:
        def emit(self, r):
            captured.append(r)

        def close(self):
            pass

    Pipeline(SimulatorSource(count=5, seed=1), FFTTransform(),
             AutoencoderModel(), [CaptureSink()]).run()

    assert len(captured) == 5
    for r in captured:
        assert r.meta["energy_rms"] is not None
        assert r.meta["energy_rms"] >= 0.0
    assert any(r.meta["energy_rms"] > 0 for r in captured)


def test_frame_roundtrip():
    s = np.array([0, 100, -100, 2047, -2048], dtype=np.int16)
    f = decode(_raw(7, 3, s))
    assert not f.is_spectrum
    assert f.node_id == 7 and f.seq == 3 and f.n == 5
    np.testing.assert_array_equal(f.payload, s)


def test_sensor_class_roundtrip():
    from signalmap.frame import encode_raw
    s = np.array([1, 2, 3, 4], dtype=np.int16)
    f = decode(encode_raw(node_id=5, seq=2, ts_us=0, sr_hz=8000, samples=s, sensor_class=6))
    assert f.sensor_class == 6
    assert f.node_id == 5
    np.testing.assert_array_equal(f.payload, s)


def test_bad_magic_raises():
    buf = bytearray(_raw(1, 0, [1, 2, 3]))
    buf[0:2] = (0x0000).to_bytes(2, "little")
    try:
        decode(bytes(buf))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_gap_tracker_detects_loss():
    g = GapTracker()
    assert g.check(1, 0) == 0
    assert g.check(1, 1) == 0
    assert g.check(1, 4) == 2  # frames 2,3 lost


def test_dsp_preserves_energy_difference():
    sr = 16000
    t = np.arange(512) / sr
    quiet = (100 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    loud = (1800 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    fq = raw_to_features(quiet, sr)
    fl = raw_to_features(loud, sr)
    # Shape normalized => similar mag norm; energy must stay distinct.
    assert fl.energy_rms > 10 * fq.energy_rms


def test_model_embed_shapes():
    m = SpectralAutoencoder(n_bins=256, latent_dim=32)
    import torch

    x = torch.randn(4, 256)
    z, err = m.embed(x)
    assert z.shape == (4, 32)
    assert err.shape == (4,)
