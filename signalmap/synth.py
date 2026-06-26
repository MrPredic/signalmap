"""Universal multi-domain signal synthesis.

SignalMap is a *sensor-agnostic* TinyML platform: any transducer (mic, piezo,
photodiode, coil, thermistor, Hall, salvaged DVD pickup, ...) becomes raw int16
frames on the same wire. This module models the characteristic physics of each
sensor DOMAIN so we can validate the whole pipeline by simulation — without any
hardware — across heterogeneous signals.

CRITICAL: SensorClass is METADATA. It is NEVER fed to the model. We keep it only
to *verify afterwards* whether the unsupervised latent space organizes itself by
physics on its own. That self-organization is the platform's core claim.

The `anomaly()` generator simulates a "novel effect": unexpected broadband
energy + subharmonic conversion that does not match any domain's baseline — the
kind of thing nobody designed a sensor to look for. These are what must rank as
outliers regardless of which domain they appear in.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


class SensorClass(IntEnum):
    ACOUSTIC = 0    # mic, speaker-as-mic
    VIBRATION = 1   # piezo, HDD/DVD voice-coil seismometer
    OPTICAL = 2     # LED-as-photodiode, PV cell, CMOS
    RF = 3          # coil/SDR, modulated carriers
    THERMAL = 4     # thermistor/thermopile, near-DC drift
    TRIBO = 5       # triboelectric: raindrop/friction impulse events
    MAGNETIC = 6    # Hall/reed, rotation-periodic
    CAPACITIVE = 7  # touch/proximity, step plateaus
    UNKNOWN = 255


# Each domain has its own natural sample rate — the DSP resamples every spectrum
# to a fixed bin count, so wildly different rates all share one latent space.
_SR = {
    SensorClass.ACOUSTIC: 44100,
    SensorClass.VIBRATION: 2000,
    SensorClass.OPTICAL: 5000,
    SensorClass.RF: 48000,
    SensorClass.THERMAL: 50,
    SensorClass.TRIBO: 8000,
    SensorClass.MAGNETIC: 4000,
    SensorClass.CAPACITIVE: 200,
}

N = 512  # samples per frame
_CLIP = (-2048, 2047)


def _i16(sig: np.ndarray) -> np.ndarray:
    return np.clip(sig, *_CLIP).astype(np.int16)


def generate(cls: SensorClass, rng: np.random.Generator) -> np.ndarray:
    """One raw frame for a sensor domain, with natural physics + noise."""
    sr = _SR[cls]
    t = np.arange(N) / sr

    if cls == SensorClass.ACOUSTIC:
        f = rng.uniform(200, 4000)
        a = rng.uniform(150, 450)
        sig = a * np.sin(2 * np.pi * f * t) + 0.3 * a * np.sin(2 * np.pi * 2 * f * t)
        sig += rng.standard_normal(N) * 15

    elif cls == SensorClass.VIBRATION:
        f = rng.uniform(5, 80)
        a = rng.uniform(200, 600)
        sig = a * np.sin(2 * np.pi * f * t)
        if rng.random() < 0.4:  # damped transient ring
            k = rng.integers(0, N // 2)
            env = np.exp(-(np.arange(N) - k).clip(0) / 40.0)
            sig += a * env * np.sin(2 * np.pi * f * 6 * t)
        sig += rng.standard_normal(N) * 20

    elif cls == SensorClass.OPTICAL:
        dc = rng.uniform(200, 900)              # ambient light bias (DC = signal!)
        flick = rng.choice([100.0, 120.0])      # mains flicker
        sig = dc + 60 * np.sin(2 * np.pi * flick * t) + rng.standard_normal(N) * 8
        if rng.random() < 0.3:                  # shadow/photon spike
            sig[rng.integers(0, N)] += rng.uniform(300, 800)

    elif cls == SensorClass.RF:
        fc = rng.uniform(8000, 20000)           # carrier near Nyquist
        fm = rng.uniform(50, 400)
        env = 1 + 0.7 * np.sin(2 * np.pi * fm * t)
        sig = 400 * env * np.sin(2 * np.pi * fc * t) + rng.standard_normal(N) * 25

    elif cls == SensorClass.THERMAL:
        slope = rng.uniform(-2, 2)              # slow drift, near-DC
        sig = 500 + slope * np.arange(N) + 30 * np.sin(2 * np.pi * 0.5 * t)
        sig += rng.standard_normal(N) * 4

    elif cls == SensorClass.TRIBO:
        sig = rng.standard_normal(N) * 10       # quiet baseline...
        for _ in range(rng.integers(1, 5)):     # ...with sparse sharp impulses
            k = rng.integers(0, N)
            sig[k:k + 3] += rng.uniform(800, 1900) * rng.choice([-1, 1])

    elif cls == SensorClass.MAGNETIC:
        f = rng.uniform(10, 200)                # rotation -> near-square Hall
        sig = 500 * np.sign(np.sin(2 * np.pi * f * t)) + rng.standard_normal(N) * 20

    elif cls == SensorClass.CAPACITIVE:
        sig = np.full(N, rng.uniform(100, 300))  # plateaus with step changes
        for _ in range(rng.integers(1, 4)):
            k = rng.integers(0, N)
            sig[k:] += rng.uniform(200, 700) * rng.choice([-1, 1])
        sig += rng.standard_normal(N) * 6

    else:
        sig = rng.standard_normal(N) * 200

    return _i16(sig)


def anomaly(rng: np.random.Generator) -> np.ndarray:
    """A 'novel effect': extreme broadband energy + subharmonic conversion that
    matches no domain baseline. Varied per draw so the anomalies are genuine
    lone outliers, not a tight cluster. Must surface as outliers everywhere."""
    sr = int(rng.choice([8000, 16000, 24000]))
    t = np.arange(N) / sr
    f1 = rng.uniform(900, 3500)
    f2 = rng.uniform(40, 130)
    a = rng.uniform(1400, 1950)
    sig = a * np.sin(2 * np.pi * f1 * t)
    sig += rng.uniform(0.6, 1.0) * a * np.sign(np.sin(2 * np.pi * f2 * t))  # subharmonic
    sig += rng.standard_normal(N) * rng.uniform(150, 300)                    # broadband
    return _i16(sig)


def build_pdm_benchmark(out: str, normal: int = 400, faults: int = 40,
                        seed: int = 11) -> int:
    """A realistic single-regime predictive-maintenance benchmark: one healthy
    machine vibration regime ('normal') vs. a bearing-like fault signature
    (periodic decaying impulses + harmonics). Unlike the 8-domain universal set,
    'normal' here is homogeneous, so reconstruction error is a meaningful signal
    — the proper sanity check that mirrors real CWRU data."""
    import time

    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    sr = 12000
    t = np.arange(N) / sr
    labels, srs, ts, blobs = [], [], [], []

    def healthy():
        f = 30 * (1 + rng.uniform(-0.03, 0.03))
        a = rng.uniform(250, 350)
        sig = a * np.sin(2 * np.pi * f * t) + 0.3 * a * np.sin(2 * np.pi * 3 * f * t)
        return sig + rng.standard_normal(N) * 25

    def faulty():
        sig = healthy()
        bpfo = rng.uniform(100, 140)                 # bearing defect frequency
        period = max(int(sr / bpfo), 1)
        for k in range(0, N, period):
            env = np.exp(-(np.arange(N) - k).clip(0) / 25.0)
            sig += rng.uniform(150, 280) * env * np.sin(2 * np.pi * 2000 * t)
        return sig

    for i in range(normal + faults):
        fault = i >= normal
        s = _i16(faulty() if fault else healthy())
        labels.append("ANOMALY_fault" if fault else "normal")
        srs.append(sr); ts.append(int(time.time() * 1e6)); blobs.append(s.tobytes())

    pq.write_table(pa.table({"label": labels, "sensor_class": [1] * len(labels),
                             "sr_hz": srs, "ts_us": ts, "samples": blobs}), out)
    return normal + faults


def build_dataset(out: str, per_class: int = 60, anomalies: int = 8, seed: int = 7) -> int:
    """Write a heterogeneous multi-domain Parquet dataset (raw frames +
    sensor_class metadata + label). Returns row count."""
    import time

    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    classes, labels, srs, ts, blobs = [], [], [], [], []

    domains = [c for c in SensorClass if c != SensorClass.UNKNOWN]
    for cls in domains:
        for _ in range(per_class):
            s = generate(cls, rng)
            classes.append(int(cls)); labels.append(cls.name.lower())
            srs.append(_SR[cls]); ts.append(int(time.time() * 1e6)); blobs.append(s.tobytes())

    for _ in range(anomalies):
        host = rng.choice(domains)              # appears within a random domain
        s = anomaly(rng)
        classes.append(int(host)); labels.append("ANOMALY_novel_effect")
        srs.append(_SR[host]); ts.append(int(time.time() * 1e6)); blobs.append(s.tobytes())

    table = pa.table({
        "label": labels, "sensor_class": classes,
        "sr_hz": srs, "ts_us": ts, "samples": blobs,
    })
    pq.write_table(table, out)
    return table.num_rows
