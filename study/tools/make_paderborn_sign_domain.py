#!/usr/bin/env python3
"""Build the paderborn_kat domain bank for the sign study.

    nice -n 19 .venv/bin/python study/tools/make_paderborn_sign_domain.py

Source: Paderborn University KAt-DataCenter bearing data, fetched by
study/tools/fetch_remaining_domains.sh into data/raw_domains/paderborn/.

Rules, all fixed in prereg AMENDMENT 6 before any number existed:
  channel  first channel in the source's own order that meets the 20480-sample
           minimum -> phase_current_1 (channel 0, `force`, has only 16001)
  split    by BEARING CODE, not by run, so no physical bearing is in both
           halves: fit = K001-K003, eval normal = K004-K005,
           eval anomaly = KA01, KA03 (outer race), KI01, KI03 (inner race)
  runs     every run of every operating point, sorted by filename

Values are stored raw: no scaling, no filtering, no detrend. Windowing and
z-normalisation belong to the readout.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw_domains" / "paderborn"
EX = RAW / "ex"
DEST = ROOT / "data" / "signdomains" / "paderborn_kat"

N = 20 * 1024
CHANNEL_NAME = "phase_current_1"
FIT_CODES = ["K001", "K002", "K003"]
EVAL_NORMAL_CODES = ["K004", "K005"]
EVAL_ANOMALY_CODES = ["KA01", "KA03", "KI01", "KI03"]
SOURCE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
LICENSE = ("Paderborn University KAt-DataCenter bearing dataset, free for "
           "scientific use with citation of Lessmeier et al. (PHM Europe 2016)")


def extract(code):
    """Unpack one .rar once; unar is already present on this machine."""
    out = EX / code
    if out.is_dir() and any(out.glob("*.mat")):
        return out
    EX.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unar", "-q", "-o", str(EX), str(RAW / f"{code}.rar")],
                   check=True, capture_output=True, timeout=900)
    return out


def channel(mat_path):
    """Return the named channel's samples as float64, or None if too short."""
    m = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    key = next(k for k in m if not k.startswith("__"))
    Y = m[key][0, 0].Y
    for i in range(Y.shape[1]):
        ch = Y[0, i]
        name = str(ch.Name[0]) if hasattr(ch, "Name") else ""
        if name == CHANNEL_NAME:
            data = np.asarray(ch.Data, dtype=np.float64).reshape(-1)
            return data if data.size >= N else None
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(codes, dest_dir, prefix, op=None):
    written, skipped = 0, 0
    for code in codes:
        src = extract(code)
        for mat in sorted(src.glob(f"{op}_*.mat" if op else "*.mat")):
            data = channel(mat)
            if data is None:
                skipped += 1
                continue
            stem = f"{prefix}{mat.stem}" if prefix else mat.stem
            np.save(dest_dir / f"{stem}.npy",
                    np.ascontiguousarray(data[:N], dtype=np.float64))
            written += 1
        print(f"  {code}: {written} written so far", flush=True)
    return written, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--op", default=None,
                    help="restrict to one operating point, e.g. N15_M07_F10")
    ap.add_argument("--domain", default="paderborn_kat")
    args = ap.parse_args()
    global DEST
    DEST = ROOT / "data" / "signdomains" / args.domain
    (DEST / "fit").mkdir(parents=True, exist_ok=True)
    (DEST / "eval").mkdir(parents=True, exist_ok=True)
    for old in list((DEST / "fit").glob("*.npy")) + list((DEST / "eval").glob("*.npy")):
        old.unlink()

    print("fit (healthy K001-K003) ...", flush=True)
    n_fit, s1 = emit(FIT_CODES, DEST / "fit", "", args.op)
    print("eval normal (healthy K004-K005) ...", flush=True)
    n_norm, s2 = emit(EVAL_NORMAL_CODES, DEST / "eval", "normal_", args.op)
    print("eval anomaly (KA01, KA03, KI01, KI03) ...", flush=True)
    n_anom, s3 = emit(EVAL_ANOMALY_CODES, DEST / "eval", "anomaly_", args.op)

    files = {str(p.relative_to(DEST)): sha256(p) for p in sorted(DEST.rglob("*.npy"))}
    (DEST / "manifest.json").write_text(json.dumps({
        "domain": args.domain, "source_url": SOURCE_URL, "license": LICENSE,
        "modality": "motor phase current on a bearing test rig",
        "fs_hz": 64000,
        "channel": (f"{CHANNEL_NAME} — the first channel in the source's own "
                    f"order (force, phase_current_1, phase_current_2, speed, "
                    f"temp_2_bearing_module, torque, vibration_1) that meets "
                    f"the 20480-sample minimum; force has only 16001. Prereg "
                    f"AMENDMENT 6."),
        "anomaly_mapping": (f"the source's own bearing codes: {EVAL_ANOMALY_CODES} "
                            f"are damaged (KA = outer race, KI = inner race); "
                            f"K001-K005 are undamaged"),
        "subset_rule": (f"fit = {FIT_CODES}, eval normal = {EVAL_NORMAL_CODES}, "
                        f"eval anomaly = {EVAL_ANOMALY_CODES}; every run of "
                        f"every operating point, sorted by filename; split by "
                        f"bearing code so no bearing appears in both halves; "
                        f"first {N} samples stored"
                        + (f"; restricted to operating point {args.op} "
                           f"(prereg AMENDMENT 8)" if args.op else "")),
        "n_fit": n_fit, "n_eval_normal": n_norm, "n_eval_anomaly": n_anom,
        "skipped_too_short": s1 + s2 + s3,
        "files": files,
    }, indent=2, sort_keys=True) + "\n")
    print(f"fit={n_fit} normal={n_norm} anomaly={n_anom} skipped={s1 + s2 + s3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
