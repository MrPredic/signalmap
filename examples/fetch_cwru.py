"""Reproduce the headline real-data result: download real CWRU bearing
vibration data, convert it to SignalMap frames, and write a Parquet dataset you
can immediately `fit` and `monitor`.

    python examples/fetch_cwru.py            # -> data/cwru_real.parquet
    signalmap fit     --dataset data/cwru_real.parquet --healthy-label normal --out det.pt
    signalmap monitor --source replay --dataset data/cwru_real.parquet --detector det.pt --quiet
    # expected: ~238/238 faults caught, ~0.2% false alarms

Data: Case Western Reserve University Bearing Data Center, via the open mirror
github.com/zerothphase/CWRU (12 kHz drive-end accelerometer). Needs scipy
(`pip install scipy`) to read the .mat files.

Note on scaling: both files are DC-centered and divided by a SHARED peak, so the
relative amplitude between healthy and faulty is preserved (raw amplitude is
signal — see the bias-free principle).
"""
from __future__ import annotations

import io
import os
import time
import urllib.request

import numpy as np

BASE = "https://raw.githubusercontent.com/zerothphase/CWRU/master/Data/12k_DE/"
FILES = [("Normal_1.mat", "normal"), ("IR007_1.mat", "ANOMALY_inner_race_fault")]
SR = 12000
FRAME_N = 512
OUT = "data/cwru_real.parquet"


def _load_de_signal(filename: str) -> np.ndarray:
    import scipy.io as sio
    raw = urllib.request.urlopen(BASE + filename, timeout=60).read()
    m = sio.loadmat(io.BytesIO(raw))
    key = next(k for k in m if k.endswith("DE_time"))
    return m[key].ravel().astype(np.float64)


def main() -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("needs pyarrow: pip install pyarrow")
    try:
        import scipy.io  # noqa: F401
    except ImportError:
        raise SystemExit("needs scipy to read .mat files: pip install scipy")

    print("downloading real CWRU bearing data ...")
    signals = {label: (_load_de_signal(fn) - 0.0) for fn, label in FILES}
    for label, s in signals.items():
        signals[label] = s - s.mean()                       # DC-center
    peak = max(np.max(np.abs(s)) for s in signals.values())  # SHARED scale

    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    for label, s in signals.items():
        s16 = np.clip(s / peak * 2047.0, -2048, 2047).astype(np.int16)
        nf = len(s16) // FRAME_N
        frames = [s16[i * FRAME_N:(i + 1) * FRAME_N] for i in range(nf)]
        table = pa.table({
            "label": [label] * nf, "sensor_class": [1] * nf, "sr_hz": [SR] * nf,
            "ts_us": [int(time.time() * 1e6) + i for i in range(nf)],
            "samples": [f.astype("<i2").tobytes() for f in frames],
        })
        if os.path.exists(OUT):
            table = pa.concat_tables([pq.read_table(OUT), table])
        pq.write_table(table, OUT)
        print(f"  {label:28s} {len(s):>7} samples -> {nf} frames")
    total = pq.read_table(OUT).num_rows
    print(f"\nwrote {total} frames -> {OUT}")
    print("next:")
    print("  signalmap fit     --dataset data/cwru_real.parquet --healthy-label normal --out det.pt")
    print("  signalmap monitor --source replay --dataset data/cwru_real.parquet --detector det.pt --quiet")


if __name__ == "__main__":
    main()
