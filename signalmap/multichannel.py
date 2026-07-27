"""Load synchronized multivariate recordings into channel form.

Causal discovery (``causal.py``) and coupling discovery (``coupling.py``) both
consume ``dict[str, np.ndarray]`` of equal-length, time-aligned channels. The
existing ``ingest`` path is single-channel-per-file (frames for the anomaly
pipeline); this module is the multichannel bridge for CCM on real sensor arrays.

Supported inputs (stdlib + numpy only):
  * ``.csv`` — first row is the header of channel names.
  * ``.npy`` — 2-D array (samples x channels); names via ``columns``.

Rows containing any missing/NaN value are dropped as a unit, so channels stay
aligned and equal length (never interpolated — same honesty rule as frames).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_MISSING = {"", "nan", "na", "null", "none"}


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows: list[list[float]] = []
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            rows.append([np.nan if c.strip().lower() in _MISSING else float(c)
                         for c in row])
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return [h.strip() for h in header], np.asarray(rows, float)


def load_channels(path, columns: list[str] | None = None,
                  dropna: bool = True) -> dict[str, np.ndarray]:
    """Return time-aligned channels from a CSV or NPY recording.

    ``columns`` selects/renames channels (required to name NPY columns). Raises
    ValueError for unknown formats, fewer than two channels, or empty data.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path}: NPY must be 2-D (samples x channels)")
        names = columns or [f"ch{i}" for i in range(arr.shape[1])]
        if len(names) != arr.shape[1]:
            raise ValueError("columns length does not match NPY channel count")
        data = {n: arr[:, i].astype(float) for i, n in enumerate(names)}
    elif suffix == ".csv":
        header, arr = _read_csv(path)
        wanted = columns or header
        missing = [c for c in wanted if c not in header]
        if missing:
            raise ValueError(f"{path}: columns not found: {missing}")
        data = {c: arr[:, header.index(c)] for c in wanted}
    else:
        raise ValueError(f"{path}: unsupported format (use .csv or .npy)")

    if dropna and data:
        mat = np.column_stack(list(data.values()))
        keep = ~np.isnan(mat).any(axis=1)
        data = {k: v[keep] for k, v in data.items()}

    if len(data) < 2:
        raise ValueError("need at least two channels for causal discovery")
    return data
