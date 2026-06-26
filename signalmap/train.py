"""Train the Conv-AE so the latent space (and recon-error anomaly score) is real.

Bias-free training: the model sees only spectral SHAPE (dsp.raw_to_features.mag).
Labels in the dataset are ignored here — they exist only for later human
verification of clustering. CPU-friendly; a few thousand frames train in
minutes, no GPU, no cloud.

    # from a recorded mic dataset:
    python -m backend.train --dataset data/dataset.parquet --epochs 30
    # or bootstrap purely synthetic (no mic yet):
    python -m backend.train --synthetic 2000 --epochs 30
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dsp import raw_to_features
from .model import SpectralAutoencoder, reconstruction_loss

N_BINS = 256


def _features_from_raw(samples: np.ndarray, sr: int) -> np.ndarray:
    return raw_to_features(samples.astype(np.float32), sr, n_bins=N_BINS).mag


def load_dataset(path: str) -> np.ndarray:
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    sr = t.column("sr_hz").to_pylist()
    blobs = t.column("samples").to_pylist()
    feats = []
    for raw_bytes, s in zip(blobs, sr):
        samples = np.frombuffer(raw_bytes, dtype="<i2")
        feats.append(_features_from_raw(samples, s))
    return np.stack(feats)


def synthetic_dataset(n: int, sr: int = 16000, frame_n: int = 512) -> np.ndarray:
    rng = np.random.default_rng(0)
    feats = []
    for _ in range(n):
        t = np.arange(frame_n) / sr
        f = rng.uniform(100, 3000)
        amp = rng.uniform(80, 500)
        sig = amp * np.sin(2 * np.pi * f * t)
        if rng.random() < 0.3:  # some harmonics / noise variety
            sig += 0.4 * amp * np.sin(2 * np.pi * 2 * f * t)
        sig += rng.standard_normal(frame_n) * 20
        samples = np.clip(sig, -2048, 2047).astype(np.int16)
        feats.append(_features_from_raw(samples, sr))
    return np.stack(feats)


def train(feats: np.ndarray, epochs: int, out: str) -> None:
    x = torch.from_numpy(feats).float()
    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True)
    model = SpectralAutoencoder(n_bins=N_BINS, latent_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"training on {len(x)} frames, {epochs} epochs (CPU)")
    for ep in range(epochs):
        total = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            recon, _ = model(batch)
            loss = reconstruction_loss(recon, batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  recon_loss={total / len(x):.6f}")

    torch.save(model.state_dict(), out)
    print(f"saved -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train SignalMap Conv-AE")
    p.add_argument("--dataset", help="parquet of recorded frames")
    p.add_argument("--synthetic", type=int, help="N synthetic frames instead")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", default="artifacts/model.pt")
    args = p.parse_args()

    if args.dataset:
        feats = load_dataset(args.dataset)
    elif args.synthetic:
        feats = synthetic_dataset(args.synthetic)
    else:
        raise SystemExit("provide --dataset PATH or --synthetic N")
    train(feats, args.epochs, args.out)


if __name__ == "__main__":
    main()
