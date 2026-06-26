"""Conv-Autoencoder over spectral magnitude -> latent embedding.

The reconstruction error doubles as an anomaly score: a material whose
signature the model reconstructs poorly is, by definition, novel relative to
everything seen so far. That is exactly the discovery signal we want.

Lightweight on purpose (runs on CPU). Swap to TF-C contrastive later for
phase-invariant robustness — the embedding interface stays the same.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectralAutoencoder(nn.Module):
    def __init__(self, n_bins: int = 256, latent_dim: int = 32) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.latent_dim = latent_dim

        # Encoder: 1D conv stack over the frequency axis.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),  # n_bins/2
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # n_bins/4
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # n_bins/8
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * (n_bins // 8), latent_dim),
        )

        self.decoder_in = nn.Linear(latent_dim, 64 * (n_bins // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, 7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_bins) -> z: (B, latent_dim)
        return self.encoder(x.unsqueeze(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_in(z).view(z.size(0), 64, self.n_bins // 8)
        return self.decoder(h).squeeze(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, per-sample reconstruction error)."""
        recon, z = self.forward(x)
        err = torch.mean((recon - x) ** 2, dim=1)
        return z, err


def reconstruction_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean((recon - x) ** 2)
