"""Embedding + anomaly scoring + (optional) Qdrant upsert.

Combined anomaly score = recon_error * kNN_distance * energy_z
  * recon_error  -> "the model has never seen a shape like this"
  * kNN_distance -> "this point sits far from existing clusters"
  * energy_z     -> "this carries unusually high raw energy" (the payoff signal)

High-performance outliers — unexpected energy conversion in odd materials —
rank to the top.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dsp import SignalFeatures
from .model import SpectralAutoencoder


@dataclass
class Embedding:
    node_id: int
    seq: int
    ts_us: int
    vector: np.ndarray      # latent_dim
    recon_error: float
    energy_rms: float
    energy_peak: float


class Embedder:
    """Wraps the AE plus a running energy baseline for z-scoring."""

    def __init__(self, model: SpectralAutoencoder) -> None:
        self.model = model.eval()
        # Welford running stats for energy_rms baseline.
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def _update_energy(self, e: float) -> float:
        self._n += 1
        delta = e - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (e - self._mean)
        std = (self._m2 / self._n) ** 0.5 if self._n > 1 else 1.0
        return (e - self._mean) / std if std > 0 else 0.0

    def embed(self, feat: SignalFeatures, node_id: int, seq: int, ts_us: int) -> tuple[Embedding, float]:
        x = torch.from_numpy(feat.mag).float().unsqueeze(0)
        z, err = self.model.embed(x)
        emb = Embedding(
            node_id=node_id,
            seq=seq,
            ts_us=ts_us,
            vector=z.squeeze(0).numpy(),
            recon_error=float(err.item()),
            energy_rms=feat.energy_rms,
            energy_peak=feat.energy_peak,
        )
        energy_z = self._update_energy(feat.energy_rms)
        return emb, abs(energy_z)


def anomaly_score(recon_error: float, knn_distance: float, energy_z: float) -> float:
    # Small epsilons keep any single zero factor from collapsing the product.
    return (recon_error + 1e-6) * (knn_distance + 1e-6) * (energy_z + 1e-3)


# --- Optional Qdrant sink (lazy import so torch-only users need no qdrant) ----
def qdrant_upsert(client, collection: str, emb: Embedding, score: float) -> None:
    from qdrant_client.models import PointStruct

    point_id = (emb.node_id << 32) | (emb.seq & 0xFFFF_FFFF)
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=point_id,
                vector=emb.vector.tolist(),
                payload={
                    "node_id": emb.node_id,
                    "seq": emb.seq,
                    "ts_us": emb.ts_us,
                    "recon_error": emb.recon_error,
                    "energy_rms": emb.energy_rms,
                    "energy_peak": emb.energy_peak,
                    "anomaly_score": score,
                },
            )
        ],
    )
