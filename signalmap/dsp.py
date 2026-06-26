"""Backend DSP — the bias-critical transform.

Rules that protect the project's goal (discovering UNKNOWN effects):
  * No frequency cropping: we keep the full rfft band. Throwing away "boring"
    bins is exactly how you miss an unexpected resonance.
  * Amplitude is SIGNAL, not nuisance: we normalize the spectral SHAPE for the
    model, but carry the raw energy (RMS / peak) as a separate scalar so the
    "extreme energy conversion" outliers stay detectable.
  * Phase is kept as an optional second channel — transients (triboelectricity,
    thermal shock) often live in phase, not in the stationary magnitude.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalFeatures:
    mag: np.ndarray      # log1p magnitude, shape (n_bins,) — model input (shape)
    phase: np.ndarray    # phase in radians, shape (n_bins,)
    energy_rms: float    # raw RMS amplitude (NOT normalized away)
    energy_peak: float   # raw peak amplitude
    sr_hz: int
    n_bins: int


def raw_to_features(samples: np.ndarray, sr_hz: int, n_bins: int = 256) -> SignalFeatures:
    """int16 raw block -> spectral features. `samples` are centered ADC counts."""
    x = samples.astype(np.float64)

    # Raw energy first — before any normalization touches the signal.
    energy_rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    energy_peak = float(np.max(np.abs(x))) if x.size else 0.0

    # Window to reduce spectral leakage (Hann is shape-preserving, not a filter).
    if x.size:
        win = np.hanning(x.size)
        xw = x * win
    else:
        xw = x

    spec = np.fft.rfft(xw)
    mag = np.abs(spec)
    phase = np.angle(spec)

    # Resample magnitude/phase to a fixed bin count so the model sees a constant
    # input width regardless of frame length / sample rate.
    mag = _resample(mag, n_bins)
    phase = _resample(phase, n_bins)

    # Normalize SHAPE only: log compress, then unit-norm. Energy is preserved
    # separately above, so this does not erase high-performance outliers.
    mag = np.log1p(mag)
    norm = np.linalg.norm(mag)
    if norm > 0:
        mag = mag / norm

    return SignalFeatures(
        mag=mag.astype(np.float32),
        phase=phase.astype(np.float32),
        energy_rms=energy_rms,
        energy_peak=energy_peak,
        sr_hz=sr_hz,
        n_bins=n_bins,
    )


def _resample(a: np.ndarray, n: int) -> np.ndarray:
    if a.size == n:
        return a
    if a.size == 0:
        return np.zeros(n, dtype=a.dtype)
    xp = np.linspace(0.0, 1.0, a.size)
    xq = np.linspace(0.0, 1.0, n)
    return np.interp(xq, xp, a)
