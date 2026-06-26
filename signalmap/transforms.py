"""Built-in Transforms — Frame to feature vector. Bias-conscious by contract."""
from __future__ import annotations

import numpy as np

from .core import register
from .dsp import raw_to_features
from .frame import Frame


@register("transform", "fft")
class FFTTransform:
    """Full-band rfft magnitude (shape-normalized), resampled to a fixed bin
    count so every sample rate shares one latent space. Raw energy is preserved
    separately in meta — NOT normalized away."""

    def __init__(self, n_bins: int = 256):
        self.n_bins = n_bins
        self.last_energy = 0.0

    def __call__(self, frame: Frame) -> np.ndarray:
        feat = raw_to_features(frame.payload.astype(np.float32), frame.sr_hz, self.n_bins)
        self.last_energy = feat.energy_rms
        return feat.mag
