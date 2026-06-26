"""Convert real recordings (WAV / CSV / NPY) into raw-frame Parquet datasets.

This is how real-world signals enter the platform without any hardware: point
it at a file someone recorded, and it slices the raw samples into v1 frames you
can `replay`, `train`, `benchmark`, and `map`. No filtering is applied — the
samples enter exactly as stored.

WAV uses the Python stdlib (no dependency). CSV/NPY use numpy. Multi-column CSV:
the chosen column (default 0) is taken as the signal.

    signalmap ingest-file rec.wav --label motor_ok --out data/real.parquet
    signalmap ingest-file fault.csv --label motor_fault --column 1 --sr 12000 \
        --out data/real.parquet            # appends if the file exists
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

FRAME_N = 512


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        width = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise SystemExit(f"unsupported WAV sample width {width*8} bit")
    a = np.frombuffer(raw, dtype=dtype)
    if ch > 1:
        a = a[::ch]  # first channel
    return a.astype(np.float64), sr


def _load_csv(path: str, column: int) -> np.ndarray:
    a = np.genfromtxt(path, delimiter=",", skip_header=0)
    if a.ndim == 2:
        a = a[:, column]
    return a.astype(np.float64)


def _to_int16(a: np.ndarray) -> np.ndarray:
    """Scale arbitrary real-valued samples into int16 range WITHOUT changing
    their shape/spectrum: a single global gain + DC-center. Documented bias."""
    a = a - np.mean(a)
    peak = np.max(np.abs(a)) or 1.0
    return np.clip(a / peak * 2047.0, -2048, 2047).astype(np.int16)


def ingest(path: str, label: str, out: str, sr: int, column: int,
           sensor_class: int) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        samples, sr = _load_wav(path)
    elif ext == ".npy":
        samples = np.load(path).astype(np.float64).ravel()
    elif ext in (".csv", ".txt"):
        samples = _load_csv(path, column)
    else:
        raise SystemExit(f"unsupported file type {ext!r} (use .wav/.csv/.npy)")

    s16 = _to_int16(samples)
    n_full = len(s16) // FRAME_N
    if n_full == 0:
        raise SystemExit(f"file too short: {len(s16)} samples < {FRAME_N}")
    frames = [s16[i * FRAME_N:(i + 1) * FRAME_N] for i in range(n_full)]

    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "label": [label] * n_full,
        "sensor_class": [sensor_class] * n_full,
        "sr_hz": [sr] * n_full,
        "ts_us": [int(time.time() * 1e6) + i for i in range(n_full)],
        "samples": [f.astype("<i2").tobytes() for f in frames],
    })
    if os.path.exists(out):
        table = pa.concat_tables([pq.read_table(out), table])
    pq.write_table(table, out)
    print(f"  {path} -> {n_full} frames @ {sr} Hz, label={label} -> {out} "
          f"(total {table.num_rows})")
    return n_full


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest a real recording into frames")
    p.add_argument("path")
    p.add_argument("--label", required=True)
    p.add_argument("--out", default="data/real.parquet")
    p.add_argument("--sr", type=int, default=16000, help="sample rate (CSV/NPY; WAV reads its own)")
    p.add_argument("--column", type=int, default=0, help="column index for multi-col CSV")
    p.add_argument("--sensor-class", type=int, default=0)
    args = p.parse_args()
    ingest(args.path, args.label, args.out, args.sr, args.column, args.sensor_class)


if __name__ == "__main__":
    main()
