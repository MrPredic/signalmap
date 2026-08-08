import numpy as np

from heart_method_families import (
    cardiac_features,
    extract_beats,
    resample_fixed_fs,
)


def test_resample_uses_time_not_sample_count():
    x = np.sin(2 * np.pi * 1.0 * np.arange(1280) / 128.0)
    y = resample_fixed_fs(x, 128.0, 250.0)
    assert 2490 <= len(y) <= 2510


def test_beat_extraction_and_features_are_finite():
    fs = 250.0
    t = np.arange(int(12 * fs)) / fs
    x = np.zeros_like(t)
    for p in np.arange(0.8, 12.0, 1.0):
        i = int(p * fs)
        x += np.exp(-0.5 * ((t - p) / 0.025) ** 2)
    beats, rr = extract_beats(x, fs)
    assert len(beats) >= 8
    assert len(rr) >= 7
    for family in ("morphology", "hrv", "wavelet"):
        vals = cardiac_features(x, fs, family)
        assert vals.size > 0
        assert np.isfinite(vals).all()
