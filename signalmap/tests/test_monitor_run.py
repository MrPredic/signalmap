"""`monitor.run`: the streaming half of the fit->monitor USP.

The spec/bank backend is covered in test_fit_monitor_spec.py; this covers the
frame-stream path, including the accounting a user reads off the summary line.
"""
import numpy as np
import pytest

from signalmap import monitor
from signalmap.detector import Detector
from signalmap.dsp import raw_to_features
from signalmap.frame import Frame

SR = 16_000
N_BINS = 256


def _raw(amp, freq=300.0, n=512, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    sig = amp * np.sin(2 * np.pi * freq * t) + rng.standard_normal(n) * 5
    return np.clip(sig, -2048, 2047).astype(np.int16)


def _frame(samples, seq=0, is_spectrum=False, payload=None):
    p = samples if payload is None else payload
    return Frame(is_spectrum=is_spectrum, node_id=1, seq=seq, ts_us=seq * 1000,
                 sr_hz=SR, n=len(p), payload=p)


@pytest.fixture(scope="module")
def healthy_detector():
    feats, energies = [], []
    for i in range(24):
        f = raw_to_features(_raw(300.0, seed=i).astype(np.float32), SR, N_BINS)
        feats.append(f.mag)
        energies.append(f.energy_rms)
    return Detector.fit(np.stack(feats), np.array(energies), n_bins=N_BINS,
                        epochs=8, threshold=4.0)


def test_healthy_frames_do_not_alert(healthy_detector, capsys):
    frames = [_frame(_raw(300.0, seed=100 + i), seq=i) for i in range(10)]
    res = monitor.run(healthy_detector, frames, quiet=True)
    assert res["n"] == 10
    assert res["alerts"] == 0
    assert res["rate"] == 0.0
    assert "10 frames · 0 alerts" in capsys.readouterr().out


def test_severity_buckets_sum_to_the_frame_count(healthy_detector):
    frames = [_frame(_raw(300.0, seed=200 + i), seq=i) for i in range(6)]
    frames += [_frame(_raw(4000.0, freq=1700.0, seed=i), seq=50 + i) for i in range(4)]
    res = monitor.run(healthy_detector, frames, quiet=True)
    assert res["ok"] + res["warn"] + res["alarm"] == res["n"] == 10


def test_spectrum_frames_are_skipped_not_scored(healthy_detector):
    raw = [_frame(_raw(300.0, seed=i), seq=i) for i in range(4)]
    spec = [_frame(None, seq=10 + i, is_spectrum=True,
                   payload=np.zeros(N_BINS, dtype=np.float32)) for i in range(3)]
    assert monitor.run(healthy_detector, raw + spec, quiet=True)["n"] == 4


def test_empty_stream_reports_zero_rate_instead_of_dividing_by_zero(healthy_detector):
    res = monitor.run(healthy_detector, [], quiet=True)
    assert res == {"n": 0, "alerts": 0, "rate": 0.0, "ok": 0, "warn": 0, "alarm": 0}


def test_quiet_suppresses_per_frame_lines_but_keeps_the_summary(healthy_detector, capsys):
    frames = [_frame(_raw(4000.0, freq=1700.0, seed=i), seq=i) for i in range(6)]
    monitor.run(healthy_detector, frames, quiet=True)
    quiet_out = capsys.readouterr().out
    monitor.run(healthy_detector, frames, quiet=False)
    loud_out = capsys.readouterr().out

    assert "⚠" not in quiet_out
    assert "frames ·" in quiet_out
    assert len(loud_out) >= len(quiet_out)


def test_detector_round_trips_through_disk_without_changing_verdicts(healthy_detector, tmp_path):
    path = tmp_path / "nested" / "det.pt"
    healthy_detector.save(str(path))
    reloaded = Detector.load(str(path))

    frames = [_frame(_raw(a, seed=i), seq=i) for i, a in enumerate([300.0, 4000.0, 300.0])]
    before = monitor.run(healthy_detector, frames, quiet=True)
    after = monitor.run(reloaded, frames, quiet=True)
    assert before == after
