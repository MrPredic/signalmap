"""Fit a detector on healthy data, then monitor a signal source for anomalies.

    signalmap fit --dataset healthy.parquet --out artifacts/detector.pt
    signalmap monitor --source replay --dataset stream.parquet --detector artifacts/detector.pt

Sensor-agnostic: the exact same two commands work for vibration, acoustics,
current, any modality — that is the breadth story.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .detector import Detector, Score
from .dsp import raw_to_features
from .frame import Frame


def fit_from_dataset(dataset: str, out: str, healthy_label: str = "",
                     n_bins: int = 256, epochs: int = 40, threshold: float = 4.0) -> Detector:
    import pyarrow.parquet as pq

    t = pq.read_table(dataset)
    labels = t.column("label").to_pylist()
    srs = t.column("sr_hz").to_pylist()
    blobs = t.column("samples").to_pylist()

    keep = [i for i, l in enumerate(labels)
            if healthy_label.lower() in str(l).lower()] if healthy_label else list(range(len(labels)))
    if not keep:
        raise SystemExit(f"no rows match healthy-label {healthy_label!r}")

    feats, energies = [], []
    for i in keep:
        f = raw_to_features(np.frombuffer(blobs[i], dtype="<i2").astype(np.float32), srs[i], n_bins)
        feats.append(f.mag)
        energies.append(f.energy_rms)

    det = Detector.fit(np.stack(feats), np.array(energies), n_bins=n_bins,
                       epochs=epochs, threshold=threshold)
    det.save(out)
    print(f"fitted detector on {len(keep)} healthy frames "
          f"(threshold z={threshold}) -> {out}")
    return det


def run(detector: Detector, frames: Iterable[Frame], quiet: bool = False) -> dict:
    n = alerts = 0
    by_sev = {"ok": 0, "warn": 0, "alarm": 0}
    for fr in frames:
        if fr.is_spectrum:
            continue
        feat = raw_to_features(fr.payload.astype(np.float32), fr.sr_hz, detector.n_bins)
        s: Score = detector.score(feat.mag, feat.energy_rms)
        n += 1
        by_sev[s.severity] += 1
        if s.alert:
            alerts += 1
            if not quiet:
                print(f"  ⚠ {s.severity.upper():5s} node={fr.node_id} seq={fr.seq:>5} "
                      f"score={s.score:5.1f}σ (recon {s.z_recon:+.1f}, energy {s.z_energy:+.1f})")
    rate = alerts / n if n else 0.0
    # summary always prints; `quiet` only suppresses the per-frame alert lines
    print(f"  {n} frames · {alerts} alerts ({rate:.0%}) · "
          f"ok={by_sev['ok']} warn={by_sev['warn']} alarm={by_sev['alarm']}")
    return {"n": n, "alerts": alerts, "rate": rate, **by_sev}
