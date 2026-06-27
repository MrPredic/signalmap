"""Deployable unsupervised anomaly detector — the product surface.

The real use case: you have a machine (or any signal source) running normally.
You DON'T have fault labels. You `fit` a detector on healthy data, and from then
on it scores live frames and raises alerts when something deviates — works the
same on vibration, acoustics, current, any sensor (that is the breadth).

Score = robust z-score against the HEALTHY baseline, on two independent signals:
  * reconstruction error  -> the spectral shape is unfamiliar (novel pattern)
  * raw energy            -> the amplitude is abnormal (too high/low)
We take the max: alert if EITHER fires. Thresholds are robust (median + MAD), so
they don't need tuning per sensor — the zero-config promise.

A portable detector (weights + baselines + threshold) saves/loads as one file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import SpectralAutoencoder, reconstruction_loss

MAD_TO_SIGMA = 1.4826  # makes MAD a consistent estimator of std for normal data


@dataclass
class Score:
    score: float       # robust z, max over the two channels
    z_recon: float
    z_energy: float
    alert: bool
    severity: str      # ok | warn | alarm


class Detector:
    def __init__(self, net: SpectralAutoencoder, med_r: float, mad_r: float,
                 med_e: float, mad_e: float, threshold: float, n_bins: int):
        self.net = net.eval()
        self.med_r, self.mad_r = med_r, mad_r
        self.med_e, self.mad_e = med_e, mad_e
        self.threshold = threshold
        self.n_bins = n_bins

    # --- training on healthy data only ---------------------------------------
    @classmethod
    def fit(cls, features: np.ndarray, energies: np.ndarray, n_bins: int = 256,
            latent_dim: int = 32, epochs: int = 40, threshold: float = 4.0,
            seed: int = 0) -> "Detector":
        torch.manual_seed(seed)
        x = torch.from_numpy(np.asarray(features, dtype=np.float32))
        net = SpectralAutoencoder(n_bins=n_bins, latent_dim=latent_dim)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True)
        net.train()
        for _ in range(epochs):
            for (b,) in loader:
                opt.zero_grad()
                recon, _ = net(b)
                reconstruction_loss(recon, b).backward()
                opt.step()
        net.eval()

        with torch.no_grad():
            _z, err = net.embed(x)
        recon = err.numpy()
        energies = np.asarray(energies, dtype=float)

        med_r, mad_r = _robust_baseline(recon)
        med_e, mad_e = _robust_baseline(energies)
        return cls(net, med_r, mad_r, med_e, mad_e, threshold, n_bins)

    # --- online scoring -------------------------------------------------------
    def score(self, feature: np.ndarray, energy: float) -> Score:
        with torch.no_grad():
            _z, err = self.net.embed(
                torch.from_numpy(np.asarray(feature, dtype=np.float32)).unsqueeze(0))
        recon = float(err.item())
        z_r = (recon - self.med_r) / self.mad_r            # only HIGH recon is anomalous
        z_e = abs(energy - self.med_e) / self.mad_e        # high OR low energy is anomalous
        s = float(max(z_r, z_e))
        if s >= 2 * self.threshold:
            sev = "alarm"
        elif s >= self.threshold:
            sev = "warn"
        else:
            sev = "ok"
        return Score(s, float(z_r), float(z_e), s >= self.threshold, sev)

    # --- persistence ----------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.net.state_dict(),
            "n_bins": self.n_bins, "latent_dim": self.net.latent_dim,
            "med_r": self.med_r, "mad_r": self.mad_r,
            "med_e": self.med_e, "mad_e": self.mad_e,
            "threshold": self.threshold,
        }, path)

    @classmethod
    def load(cls, path: str) -> "Detector":
        d = torch.load(path, map_location="cpu")
        net = SpectralAutoencoder(n_bins=d["n_bins"], latent_dim=d["latent_dim"])
        net.load_state_dict(d["state_dict"])
        return cls(net, d["med_r"], d["mad_r"], d["med_e"], d["mad_e"],
                   d["threshold"], d["n_bins"])


def _robust_baseline(a: np.ndarray) -> tuple[float, float]:
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * MAD_TO_SIGMA
    return med, (mad if mad > 1e-9 else 1.0)
