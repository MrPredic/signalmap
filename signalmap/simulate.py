"""Hardware-free end-to-end proof of the vertical slice.

Generates synthetic raw frames (mix of 'ordinary' signals plus a few injected
high-energy / odd-spectrum anomalies), runs them through frame -> dsp -> embed,
and prints the ranked anomaly scores. No ESP32, no broker required.

    python -m backend.simulate
"""
from __future__ import annotations

import numpy as np

from .dsp import raw_to_features
from .embed import Embedder, anomaly_score
from .frame import decode
from .model import SpectralAutoencoder

SR = 16_000
N = 512


def make_raw_frame(node_id: int, seq: int, samples: np.ndarray) -> bytes:
    import struct

    hdr = struct.pack(
        "<HBBIIQIHH", 0x5247, 1, 0, node_id, seq, seq * 1000, SR, len(samples), 0
    )
    return hdr + samples.astype("<i2").tobytes()


def ordinary(seq: int) -> np.ndarray:
    t = np.arange(N) / SR
    f = 200 + (seq % 5) * 50
    sig = 300 * np.sin(2 * np.pi * f * t) + np.random.randn(N) * 20
    return np.clip(sig, -2048, 2047).astype(np.int16)


def anomaly(seq: int) -> np.ndarray:
    """High raw energy + broadband odd spectrum = the thing we hunt."""
    t = np.arange(N) / SR
    sig = 1800 * np.sin(2 * np.pi * 1300 * t) + 1500 * np.sign(np.sin(2 * np.pi * 90 * t))
    sig += np.random.randn(N) * 200
    return np.clip(sig, -2048, 2047).astype(np.int16)


def main() -> None:
    np.random.seed(0)
    model = SpectralAutoencoder(n_bins=256, latent_dim=32)
    embedder = Embedder(model)

    results = []
    for seq in range(40):
        samples = anomaly(seq) if seq in (12, 27, 33) else ordinary(seq)
        frame = decode(make_raw_frame(node_id=1, seq=seq, samples=samples))
        feat = raw_to_features(frame.payload.astype(np.float32), frame.sr_hz)
        emb, energy_z = embedder.embed(feat, frame.node_id, frame.seq, frame.ts_us)
        score = anomaly_score(emb.recon_error, knn_distance=1.0, energy_z=energy_z)
        results.append((seq, emb.energy_rms, energy_z, score))

    print("seq  e_rms     e_z    score   (injected anomalies: 12,27,33)")
    for seq, e, ez, sc in sorted(results, key=lambda r: r[3], reverse=True)[:8]:
        flag = "  <== ANOMALY" if seq in (12, 27, 33) else ""
        print(f"{seq:3d}  {e:7.1f}  {ez:5.2f}  {sc:6.3f}{flag}")


if __name__ == "__main__":
    main()
