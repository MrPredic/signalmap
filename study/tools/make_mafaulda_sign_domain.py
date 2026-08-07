#!/usr/bin/env python3
"""Build the mafaulda domain bank for the sign study.

    nice -n 19 .venv/bin/python study/tools/make_mafaulda_sign_domain.py

Source: MAFAULDA (Machinery Fault Database, UFRJ), fetched by
study/tools/fetch_remaining_domains.sh into data/raw_domains/mafaulda/.
Each sequence is 5 s at 50 kHz (250000 samples), eight columns.

Rules fixed in prereg AMENDMENT 7 before any number existed:
  column   the first CONDITION-measuring column in the source's order.
           Column 1 is a tachometer reference, so this is column 2,
           underhang accelerometer axial (index 1, zero-based).
  split    fit = first 20 healthy by sorted name, eval normal = remaining 29.
           The source holds only 49 healthy sequences, one short of the
           contract's 20+30; the shortfall is reported, not padded.
  anomaly  every imbalance sequence of every severity (6g..35g), sorted.

Only the first 20480 rows of each CSV are parsed — that is what the readout
consumes — so 5.7 GB of archives never get fully decoded.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_domains" / "mafaulda"
DEST = ROOT / "data" / "signdomains" / "mafaulda"

N = 20 * 1024
COL = 1  # zero-based: source column 2, underhang accelerometer axial
N_FIT = 20
SOURCE_URL = "https://www02.smt.ufrj.br/~offshore/mfs/page_01.html"
LICENSE = "MAFAULDA, UFRJ — free for research use with attribution"


def read_head(zf, name):
    """Parse column COL from the first N rows without decoding the whole file."""
    vals = np.empty(N, dtype=np.float64)
    got = 0
    with zf.open(name) as raw:
        for line in io.TextIOWrapper(raw, encoding="ascii", newline=""):
            parts = line.split(",")
            if len(parts) <= COL:
                continue
            try:
                vals[got] = float(parts[COL])
            except ValueError:
                continue
            got += 1
            if got == N:
                return vals
    return None  # too short for the contract


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    (DEST / "fit").mkdir(parents=True, exist_ok=True)
    (DEST / "eval").mkdir(parents=True, exist_ok=True)
    for old in list((DEST / "fit").glob("*.npy")) + list((DEST / "eval").glob("*.npy")):
        old.unlink()

    counts = {"fit": 0, "normal": 0, "anomaly": 0, "too_short": 0}

    with zipfile.ZipFile(RAW / "normal.zip") as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        print(f"normal.zip: {len(names)} healthy sequences", flush=True)
        for i, name in enumerate(names):
            arr = read_head(zf, name)
            if arr is None:
                counts["too_short"] += 1
                continue
            stem = Path(name).stem.replace(".", "_")
            if counts["fit"] < N_FIT:
                np.save(DEST / "fit" / f"{stem}.npy", arr)
                counts["fit"] += 1
            else:
                np.save(DEST / "eval" / f"normal_{stem}.npy", arr)
                counts["normal"] += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(names)}", flush=True)

    with zipfile.ZipFile(RAW / "imbalance.zip") as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        print(f"imbalance.zip: {len(names)} fault sequences", flush=True)
        for i, name in enumerate(names):
            arr = read_head(zf, name)
            if arr is None:
                counts["too_short"] += 1
                continue
            p = Path(name)
            stem = f"{p.parent.name}_{p.stem}".replace(".", "_")
            np.save(DEST / "eval" / f"anomaly_{stem}.npy", arr)
            counts["anomaly"] += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(names)}", flush=True)

    files = {str(p.relative_to(DEST)): sha256(p) for p in sorted(DEST.rglob("*.npy"))}
    (DEST / "manifest.json").write_text(json.dumps({
        "domain": "mafaulda", "source_url": SOURCE_URL, "license": LICENSE,
        "modality": "accelerometer on a rotating machinery fault simulator",
        "fs_hz": 50000,
        "channel": ("source column 2 (zero-based index 1), underhang bearing "
                    "accelerometer axial — the first condition-measuring column; "
                    "column 1 is a tachometer reference. Prereg AMENDMENT 7."),
        "anomaly_mapping": ("the source's own directory split: normal/ is healthy, "
                            "imbalance/<weight>/ are the labelled imbalance faults, "
                            "all severities 6g-35g included without selection"),
        "subset_rule": (f"healthy sorted by filename: first {N_FIT} to fit/, the "
                        f"remaining to eval/normal_ (source holds only 49, one "
                        f"short of the contract's 20+30 — reported, not padded); "
                        f"every imbalance sequence to eval/anomaly_; first {N} "
                        f"samples of each"),
        "contract_shortfall": ("eval_normal is 29, one below the contract minimum "
                               "of 30; only 49 distinct healthy sequences exist"),
        "n_fit": counts["fit"], "n_eval_normal": counts["normal"],
        "n_eval_anomaly": counts["anomaly"],
        "skipped_too_short": counts["too_short"],
        "files": files,
    }, indent=2, sort_keys=True) + "\n")
    print(f"fit={counts['fit']} normal={counts['normal']} "
          f"anomaly={counts['anomaly']} short={counts['too_short']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
