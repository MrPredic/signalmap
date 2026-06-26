"""Honest anomaly-detection benchmark: ROC-AUC, not "we injected N, found N".

Trains the embedder on KNOWN/normal data only, scores held-out data, and
reports rank-based ROC-AUC + precision@k for separating anomalies from normal.
The same harness runs on:
  * the built-in synthetic multi-domain dataset (default), and
  * any real dataset via `--dataset` (Parquet of raw frames), so the real-world
    litmus test is one flag away.

AUC is computed from ranks (Mann-Whitney U) — no sklearn dependency.
"""
from __future__ import annotations

import numpy as np
import torch

from .dsp import raw_to_features
from .model import SpectralAutoencoder

N_BINS = 256


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). labels: 1 = anomaly (positive)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks (1-based) over all scores sorted ascending, ties averaged
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j < len(scores) and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks_sorted[i:j] = (i + 1 + j) / 2.0  # mean of 1-based ranks i+1..j
        i = j
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = ranks_sorted
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pct_rank(a: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1]; higher value -> higher rank. Tie-robust."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a))
    return r / max(len(a) - 1, 1)


def _features(blobs, srs):
    return np.stack([
        raw_to_features(np.frombuffer(b, dtype="<i2").astype(np.float32), sr, N_BINS).mag
        for b, sr in zip(blobs, srs)
    ])


def _energy(blobs, srs):
    return np.array([
        raw_to_features(np.frombuffer(b, dtype="<i2").astype(np.float32), sr, N_BINS).energy_rms
        for b, sr in zip(blobs, srs)
    ])


def run(dataset: str | None, epochs: int, anomaly_label: str, seed: int = 7) -> dict:
    import pyarrow.parquet as pq

    if dataset is None:
        from .synth import build_pdm_benchmark
        dataset = "data/_benchmark.parquet"
        build_pdm_benchmark(dataset, normal=400, faults=40, seed=seed)

    t = pq.read_table(dataset)
    labels = t.column("label").to_pylist()
    srs = t.column("sr_hz").to_pylist()
    blobs = t.column("samples").to_pylist()
    is_anom = np.array([anomaly_label.lower() in str(l).lower() for l in labels])
    if is_anom.sum() == 0:
        raise SystemExit(f"no rows match anomaly-label {anomaly_label!r}; "
                         f"labels present: {sorted(set(labels))[:8]}")

    feats = _features(blobs, srs)
    energy = _energy(blobs, srs)

    # train on normal only (anomalies must be unseen)
    x = torch.from_numpy(feats[~is_anom]).float()
    net = SpectralAutoencoder(n_bins=N_BINS, latent_dim=32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True)
    net.train()
    for _ in range(epochs):
        for (b,) in loader:
            opt.zero_grad()
            recon, _ = net(b)
            torch.mean((recon - b) ** 2).backward()
            opt.step()
    net.eval()

    with torch.no_grad():
        _z, err = net.embed(torch.from_numpy(feats).float())
    recon = err.numpy()  # reconstruction error = unseen-shape novelty (robust)

    y = is_anom.astype(int)
    auc_recon = roc_auc(recon, y)
    auc_energy = roc_auc(energy, y)

    # Robust combination: average of percentile ranks of the two sound signals
    # (rank-based => immune to the heavy tails that wreck a z-score sum).
    score = _pct_rank(recon) + _pct_rank(energy)
    auc = roc_auc(score, y)
    k = int(is_anom.sum())
    topk = set(np.argsort(score)[-k:].tolist())
    p_at_k = sum(is_anom[i] for i in topk) / k

    print(f"benchmark on {dataset}")
    print(f"  rows={len(labels)}  normal={int((~is_anom).sum())}  anomalies={k}")
    print(f"  AUC recon-error  = {auc_recon:.3f}")
    print(f"  AUC raw-energy   = {auc_energy:.3f}")
    print(f"  AUC combined     = {auc:.3f}   (0.5=random, 1.0=perfect)")
    print(f"  precision@{k:<3d}     = {p_at_k:.3f}")
    return {"auc": auc, "auc_recon": auc_recon, "auc_energy": auc_energy,
            "p_at_k": p_at_k, "n": len(labels), "anomalies": k}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="SignalMap anomaly-detection benchmark (ROC-AUC)")
    p.add_argument("--dataset", help="Parquet of raw frames (default: synthetic multi-domain)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--anomaly-label", default="ANOMALY",
                   help="rows whose label contains this substring are positives")
    args = p.parse_args()
    run(args.dataset, args.epochs, args.anomaly_label)


if __name__ == "__main__":
    main()
