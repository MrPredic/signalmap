"""Built-in Models — feature to (embedding, anomaly score)."""
from __future__ import annotations

import numpy as np

from .core import register
from .model import SpectralAutoencoder  # the nn.Module lives here


@register("model", "autoencoder")
class AutoencoderModel:
    """Conv-AE embedder. Reconstruction error = anomaly score. Loads trained
    weights if given; otherwise runs untrained (warns — scores are noise)."""

    def __init__(self, n_bins: int = 256, latent_dim: int = 32, weights: str | None = None):
        import torch
        self.torch = torch
        self.net = SpectralAutoencoder(n_bins=n_bins, latent_dim=latent_dim).eval()
        if weights:
            self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        else:
            print("  [model] no weights -> untrained AE; scores are not meaningful yet")

    def process(self, feature: np.ndarray) -> tuple[np.ndarray, float]:
        x = self.torch.from_numpy(np.asarray(feature, dtype=np.float32)).unsqueeze(0)
        z, err = self.net.embed(x)
        return z.squeeze(0).numpy(), float(err.item())
