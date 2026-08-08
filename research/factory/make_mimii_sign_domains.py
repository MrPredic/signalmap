#!/usr/bin/env python3
"""Build the four MIMII/DCASE2020-Task2 id_00 domain banks for the sign study.

    nice -n 19 .venv/bin/python research/factory/make_mimii_sign_domains.py

Sources, both already on disk:
  * valve  — data/mimii/valve_id00/{train,test}/*.wav (fetched 2026-07-20)
  * fan, pump, slider — data/mimii_raw/dev_data_<machine>.zip, read streaming
    so the ~3.2 GB of archives are never expanded on disk.

Output: data/signdomains/mimii_<machine>_id00/ per DOMAIN_BANK_CONTRACT.md.
Deterministic: files in sorted order, fixed decode path, no sampling, no seed.

Decode path, stated exactly because the manifest must let a reviewer redo it:
16-bit PCM mono is read with the stdlib `wave` module and reinterpreted as
int16, then widened to float64 WITHOUT scaling — the stored values are the
raw PCM integers. This matches the existing valve bank built by
dcase_valve_adapter.py, so the numbers stay comparable to the 2026-07-21
readout that this study takes as its trigger.

Each recording is stored as exactly K*W = 20480 samples (prereg AMENDMENT 1):
the readout consumes precisely that prefix, so storing more cannot change any
result, and storing float64 of a full 10 s clip would cost 8x the archive.
"""
import hashlib
import io
import json
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "mimii_raw"
VALVE_WAV = ROOT / "data" / "mimii" / "valve_id00"
OUT = ROOT / "data" / "signdomains"

K, W = 20, 1024
N = K * W
SOURCE_URL = ("https://zenodo.org/record/3678171 (DCASE2020 Task 2 development "
              "set, derived from the MIMII dataset, https://zenodo.org/record/3384388)")
LICENSE = "CC BY-SA 4.0 (MIMII / DCASE2020 Task 2 development dataset)"


def decode(raw_bytes):
    """16-bit PCM mono -> float64 of the raw integer sample values."""
    with wave.open(io.BytesIO(raw_bytes), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"expected 16-bit mono, got {w.getsampwidth() * 8}-bit "
                             f"{w.getnchannels()}ch")
        frames = w.readframes(w.getnframes())
        fs = w.getframerate()
    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    return arr, fs


def emit(dest, stem, arr):
    if arr.size < N:
        return False
    np.save(dest / f"{stem}.npy", np.ascontiguousarray(arr[:N], dtype=np.float64))
    return True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_zip(zip_path, machine):
    """Yield (split, stem, wav_bytes) for id_00 members, in sorted order."""
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist()
                       if n.endswith(".wav") and "_id_00_" in n
                       and n.startswith((f"{machine}/train/", f"{machine}/test/")))
        for name in names:
            split = "train" if name.startswith(f"{machine}/train/") else "test"
            yield split, Path(name).stem, zf.read(name)


def iter_dir(root):
    """Yield (split, stem, wav_bytes) from an already-extracted valve tree."""
    for split in ("train", "test"):
        for path in sorted((root / split).glob("*_id_00_*.wav")):
            yield split, path.stem, path.read_bytes()


def build(domain, source):
    dest = OUT / domain
    (dest / "fit").mkdir(parents=True, exist_ok=True)
    (dest / "eval").mkdir(parents=True, exist_ok=True)
    for old in list((dest / "fit").glob("*.npy")) + list((dest / "eval").glob("*.npy")):
        old.unlink()

    counts = {"fit": 0, "normal": 0, "anomaly": 0, "too_short": 0}
    fs_seen = set()
    for split, stem, raw in source:
        arr, fs = decode(raw)
        fs_seen.add(fs)
        if split == "train":
            # the DCASE development train split is all-normal by construction
            if not stem.startswith("normal"):
                raise ValueError(f"unexpected non-normal train clip: {stem}")
            ok = emit(dest / "fit", stem, arr)
            counts["fit"] += ok
        else:
            label = "anomaly" if stem.startswith("anomaly") else "normal"
            ok = emit(dest / "eval", f"{label}_{stem}", arr)
            counts[label] += ok
        counts["too_short"] += (not ok)

    files = {}
    for path in sorted(dest.rglob("*.npy")):
        files[str(path.relative_to(dest))] = sha256(path)
    (dest / "manifest.json").write_text(json.dumps({
        "domain": domain, "source_url": SOURCE_URL, "license": LICENSE,
        "modality": "acoustic (single microphone, machine operating sound)",
        "fs_hz": sorted(fs_seen)[0] if len(fs_seen) == 1 else sorted(fs_seen),
        "channel": "0 = the only channel; source clips are 16 kHz mono 16-bit PCM",
        "anomaly_mapping": ("the source's own filename prefix: clips named "
                            "anomaly_id_00_* are the dataset's labelled anomalies, "
                            "normal_id_00_* its labelled normal operation; the "
                            "train split is all-normal by dataset construction"),
        "decode": ("stdlib wave -> int16 -> float64, raw PCM integers, no scaling "
                   "(identical to dcase_valve_adapter.py)"),
        "subset_rule": (f"every id_00 clip of this machine type, sorted by name; "
                        f"train -> fit/, test -> eval/; first {N} samples stored"),
        "n_fit": counts["fit"], "n_eval_normal": counts["normal"],
        "n_eval_anomaly": counts["anomaly"],
        "skipped_too_short": counts["too_short"],
        "files": files,
    }, indent=2, sort_keys=True) + "\n")
    return counts


def main():
    jobs = [("mimii_valve_id00", lambda: iter_dir(VALVE_WAV))]
    for machine in ("fan", "pump", "slider"):
        zip_path = RAW / f"dev_data_{machine}.zip"
        if zip_path.exists():
            jobs.append((f"mimii_{machine}_id00",
                         lambda z=zip_path, m=machine: iter_zip(z, m)))
        else:
            print(f"! {zip_path} missing -> mimii_{machine}_id00 not-obtained")

    for domain, source in jobs:
        print(f"[{domain}] building ...", flush=True)
        counts = build(domain, source())
        print(f"[{domain}] fit={counts['fit']} normal={counts['normal']} "
              f"anomaly={counts['anomaly']} short={counts['too_short']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
