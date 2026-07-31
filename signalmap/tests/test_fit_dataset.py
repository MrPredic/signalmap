"""`signalmap fit --dataset healthy.parquet` — the spectral half of fit->monitor.

The spec/bank half lives in test_fit_monitor_spec.py. This one pins the label
filtering, because a wrong filter silently fits the detector on faulty frames
and there is no later signal that anything went wrong.
"""
import numpy as np
import pytest

from signalmap import monitor
from signalmap.detector import Detector

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SR = 16_000


def _dataset(path, rows):
    """rows: list of (label, amplitude)."""
    blobs = []
    for _label, amp in rows:
        t = np.arange(512) / SR
        sig = amp * np.sin(2 * np.pi * 300 * t)
        blobs.append(np.clip(sig, -2048, 2047).astype("<i2").tobytes())
    pq.write_table(pa.table({
        "label": [r[0] for r in rows],
        "sr_hz": [SR] * len(rows),
        "samples": blobs,
    }), path)
    return str(path)


def _healthy_and_faulty(n=12):
    return ([("normal", 300.0)] * n) + ([("IR007_fault", 3500.0)] * n)


def test_fit_writes_a_detector_and_reports_the_frame_count(tmp_path, capsys):
    ds = _dataset(tmp_path / "ds.parquet", [("normal", 300.0)] * 10)
    out = tmp_path / "nested" / "det.pt"     # nested: dir must be created for us
    det = monitor.fit_from_dataset(ds, str(out), epochs=3)

    assert out.exists()
    assert isinstance(det, Detector)
    assert "10 healthy frames" in capsys.readouterr().out


def test_healthy_label_selects_only_matching_rows(tmp_path, capsys):
    ds = _dataset(tmp_path / "mixed.parquet", _healthy_and_faulty(12))
    monitor.fit_from_dataset(ds, str(tmp_path / "det.pt"),
                             healthy_label="normal", epochs=3)
    assert "12 healthy frames" in capsys.readouterr().out


def test_healthy_label_matching_is_case_insensitive_and_substring(tmp_path, capsys):
    ds = _dataset(tmp_path / "mixed.parquet", _healthy_and_faulty(12))
    monitor.fit_from_dataset(ds, str(tmp_path / "det.pt"),
                             healthy_label="NORMAL", epochs=3)
    assert "12 healthy frames" in capsys.readouterr().out


def test_no_healthy_label_uses_every_row(tmp_path, capsys):
    ds = _dataset(tmp_path / "mixed.parquet", _healthy_and_faulty(12))
    monitor.fit_from_dataset(ds, str(tmp_path / "det.pt"), epochs=3)
    assert "24 healthy frames" in capsys.readouterr().out


def test_a_label_that_matches_nothing_fails_loudly(tmp_path):
    ds = _dataset(tmp_path / "mixed.parquet", _healthy_and_faulty(4))
    with pytest.raises(SystemExit) as exc:
        monitor.fit_from_dataset(ds, str(tmp_path / "det.pt"),
                                 healthy_label="does-not-exist", epochs=1)
    assert "does-not-exist" in str(exc.value)


def test_fitting_on_healthy_only_separates_the_faulty_rows(tmp_path):
    # The end-to-end claim of `fit`: trained on healthy alone, the detector
    # scores the faulty frames higher. Without this the label filter is decorative.
    ds = _dataset(tmp_path / "mixed.parquet", _healthy_and_faulty(16))
    det = monitor.fit_from_dataset(ds, str(tmp_path / "det.pt"),
                                   healthy_label="normal", epochs=25)

    from signalmap.dsp import raw_to_features

    def score(amp):
        t = np.arange(512) / SR
        sig = np.clip(amp * np.sin(2 * np.pi * 300 * t), -2048, 2047).astype(np.float32)
        f = raw_to_features(sig, SR, det.n_bins)
        return det.score(f.mag, f.energy_rms).score

    assert score(3500.0) > score(300.0)
