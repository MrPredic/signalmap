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


def test_roc_auc_matches_known_values():
    from signalmap.benchmark import roc_auc
    # perfect separation
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    assert roc_auc(s, y) == 1.0
    # inverted = 0.0
    assert roc_auc(-s, y) == 0.0
    # partial overlap = 0.75
    s2 = np.array([0.1, 0.2, 0.3, 0.4])
    y2 = np.array([0, 1, 0, 1])
    assert abs(roc_auc(s2, y2) - 0.75) < 1e-9
    # ties across classes = 0.5
    s3 = np.array([0.5, 0.5, 0.5, 0.5])
    y3 = np.array([0, 1, 0, 1])
    assert abs(roc_auc(s3, y3) - 0.5) < 1e-9


def test_ingest_wav_roundtrip(tmp_path):
    import wave

    from signalmap.frame import decode
    from signalmap.ingest import ingest
    from signalmap.sources import ReplaySource

    wav = tmp_path / "tone.wav"
    sr = 16000
    t = np.arange(sr) / sr  # 1 second -> ~31 frames of 512
    sig = (3000 * np.sin(2 * np.pi * 440 * t)).astype("<i2")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(sig.tobytes())

    out = tmp_path / "real.parquet"
    n = ingest(str(wav), "tone", str(out), sr=sr, column=0, sensor_class=3)
    assert n == sr // 512

    frames = list(ReplaySource(str(out)).frames())
    assert len(frames) == n
    f0 = frames[0]
    assert not f0.is_spectrum and f0.sr_hz == sr and f0.sensor_class == 3
    assert f0.payload.dtype == np.int16 and len(f0.payload) == 512
    assert np.max(np.abs(f0.payload)) > 1000  # real signal energy preserved


def test_benchmark_detects_synthetic_fault(tmp_path):
    """The pipeline must separate a homogeneous-normal vs fault set unsupervised."""
    from signalmap.benchmark import run
    from signalmap.synth import build_pdm_benchmark

    ds = tmp_path / "pdm.parquet"
    build_pdm_benchmark(str(ds), normal=120, faults=20, seed=3)
    res = run(str(ds), epochs=25, anomaly_label="ANOMALY")  # AE needs ~25 ep to converge here
    assert res["auc"] > 0.9
    assert res["auc_recon"] > 0.8


def test_mutual_information_independent_vs_dependent():
    from signalmap.coupling import mutual_information
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    indep = rng.standard_normal(4000)
    dep = np.tanh(x) + 0.05 * rng.standard_normal(4000)  # nonlinear dependence
    assert mutual_information(x, indep) < 0.05
    assert mutual_information(x, dep) > 0.3


def test_partial_correlation_removes_confound():
    from signalmap.coupling import _safe_corr, partial_correlation
    rng = np.random.default_rng(1)
    z = rng.standard_normal(4000)
    x = z + 0.3 * rng.standard_normal(4000)   # both driven by z, not each other
    y = z + 0.3 * rng.standard_normal(4000)
    assert _safe_corr(x, y) > 0.7              # looks coupled...
    assert abs(partial_correlation(x, y, z)) < 0.1   # ...but vanishes given z


def test_discover_keeps_real_coupling_rejects_confound():
    """Headline rigor test: the apparatus must NOT lie."""
    from signalmap.discover import evaluate
    v = evaluate(seed=0)
    assert v["true_coupling_found"], "missed the genuine heat->acoustic coupling"
    assert v["confound_rejected"], "fell for the temp-driven confound"
    # the confounded pair looks coupled raw, but collapses after confound removal
    assert v["confound_raw_corr"] > 0.4
    assert v["confound_adj_corr"] < 0.15


def test_detector_alerts_on_fault_not_healthy(tmp_path):
    """The product surface: fit on healthy, alert on faults, stay quiet on healthy."""
    import pyarrow.parquet as pq

    from signalmap.detector import Detector
    from signalmap.dsp import raw_to_features
    from signalmap.synth import build_pdm_benchmark

    ds = tmp_path / "pdm.parquet"
    build_pdm_benchmark(str(ds), normal=200, faults=40, seed=5)
    t = pq.read_table(str(ds))
    labels = t.column("label").to_pylist()
    srs = t.column("sr_hz").to_pylist()
    blobs = t.column("samples").to_pylist()

    def feat(i):
        f = raw_to_features(np.frombuffer(blobs[i], dtype="<i2").astype(np.float32), srs[i], 256)
        return f.mag, f.energy_rms

    healthy = [i for i, l in enumerate(labels) if l == "normal"]
    fault = [i for i, l in enumerate(labels) if "ANOMALY" in l]

    train = healthy[:150]
    det = Detector.fit(np.stack([feat(i)[0] for i in train]),
                       np.array([feat(i)[1] for i in train]),
                       epochs=30, threshold=4.0, seed=0)

    # save/load roundtrip must preserve behavior
    p = tmp_path / "det.pt"
    det.save(str(p))
    det = Detector.load(str(p))

    held_healthy = healthy[150:]
    h_rate = sum(det.score(*feat(i)).alert for i in held_healthy) / len(held_healthy)
    f_rate = sum(det.score(*feat(i)).alert for i in fault) / len(fault)
    assert h_rate < 0.15, f"too many false alarms on healthy: {h_rate:.0%}"
    assert f_rate > 0.8, f"missed faults: only {f_rate:.0%} caught"


def test_model_embed_shapes():
    m = SpectralAutoencoder(n_bins=256, latent_dim=32)
    import torch

    x = torch.randn(4, 256)
    z, err = m.embed(x)
    assert z.shape == (4, 32)
    assert err.shape == (4,)
