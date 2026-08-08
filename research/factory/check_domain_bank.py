#!/usr/bin/env python3
"""Class-blind acceptance check for a domain bank.

    python research/factory/check_domain_bank.py data/signdomains/<domain> [--json]

Enforces DOMAIN_BANK_CONTRACT.md and nothing else. Every check here is
deliberately blind to the normal/anomaly label: it counts files, verifies
shape/dtype/finiteness/length, recomputes sha256, and proves fit/eval
disjointness. It never compares the two classes — a bank that was tuned
against a separation metric would silently invalidate the sign measurement
in PREREG_SIGN_IDENTIFIABILITY.md.

Exit 0 = bank accepted. Exit 1 = violations listed, bank not accepted.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

K, W = 20, 1024
MIN_SAMPLES = K * W
MIN_FIT = 20
MIN_EVAL_PER_CLASS = 30


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def window_digest(arr):
    """Digest of the exact windows the readout will use — catches an eval
    recording that repeats a fit recording's samples under a new filename."""
    block = np.ascontiguousarray(arr[: MIN_SAMPLES], dtype=np.float64)
    return hashlib.sha256(block.tobytes()).hexdigest()


def check_array(path, problems):
    """Load one recording and check the contract's per-file promises."""
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - report, never crash the run
        problems.append(f"{path}: unreadable ({exc})")
        return None
    if arr.ndim != 1:
        problems.append(f"{path}: ndim={arr.ndim}, contract requires 1-D")
        return None
    if arr.dtype != np.float64:
        problems.append(f"{path}: dtype={arr.dtype}, contract requires float64")
    if arr.size < MIN_SAMPLES:
        problems.append(f"{path}: {arr.size} samples < {MIN_SAMPLES} required")
        return None
    if not np.all(np.isfinite(arr)):
        problems.append(f"{path}: contains NaN/Inf")
        return None
    if float(np.std(arr)) <= 0.0:
        problems.append(f"{path}: constant signal (std == 0)")
        return None
    return arr


def check_bank(root):
    """Return (problems, stats) for one domain bank directory."""
    root = Path(root)
    problems, stats = [], {}
    if not root.is_dir():
        return [f"{root}: not a directory"], stats

    fit = sorted((root / "fit").glob("*.npy"))
    ev_norm = sorted((root / "eval").glob("normal_*.npy"))
    ev_anom = sorted((root / "eval").glob("anomaly_*.npy"))
    stray = [p for p in sorted((root / "eval").glob("*.npy"))
             if not p.name.startswith(("normal_", "anomaly_"))]
    for p in stray:
        problems.append(f"{p}: eval file without normal_/anomaly_ prefix")

    stats.update(n_fit=len(fit), n_eval_normal=len(ev_norm), n_eval_anomaly=len(ev_anom))
    if len(fit) < MIN_FIT:
        problems.append(f"fit/: {len(fit)} recordings < {MIN_FIT} required")
    if len(ev_norm) < MIN_EVAL_PER_CLASS:
        problems.append(f"eval/normal_: {len(ev_norm)} < {MIN_EVAL_PER_CLASS} required")
    if len(ev_anom) < MIN_EVAL_PER_CLASS:
        problems.append(f"eval/anomaly_: {len(ev_anom)} < {MIN_EVAL_PER_CLASS} required")

    fit_digests, eval_digests = {}, {}
    for path in fit:
        arr = check_array(path, problems)
        if arr is not None:
            fit_digests.setdefault(window_digest(arr), []).append(path.name)
    for path in ev_norm + ev_anom:
        arr = check_array(path, problems)
        if arr is not None:
            eval_digests.setdefault(window_digest(arr), []).append(path.name)

    for digest, names in fit_digests.items():
        if len(names) > 1:
            problems.append(f"fit/: duplicate recordings share samples: {names}")
        if digest in eval_digests:
            problems.append(
                f"LEAKAGE: fit {names} and eval {eval_digests[digest]} share the "
                f"first {MIN_SAMPLES} samples")
    for digest, names in eval_digests.items():
        if len(names) > 1:
            problems.append(f"eval/: duplicate recordings share samples: {names}")

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        problems.append("manifest.json missing")
        return problems, stats
    try:
        manifest = json.loads(manifest_path.read_text())
    except ValueError as exc:
        problems.append(f"manifest.json: invalid JSON ({exc})")
        return problems, stats

    for field in ("domain", "source_url", "license", "modality", "fs_hz",
                  "anomaly_mapping", "files"):
        if not manifest.get(field):
            problems.append(f"manifest.json: missing/empty field '{field}'")
    if manifest.get("domain") and manifest["domain"] != root.name:
        problems.append(f"manifest.json: domain '{manifest['domain']}' != dir '{root.name}'")

    listed = manifest.get("files") or {}
    on_disk = {str(p.relative_to(root)) for p in fit + ev_norm + ev_anom}
    missing, extra = on_disk - set(listed), set(listed) - on_disk
    for name in sorted(missing):
        problems.append(f"manifest.json: file on disk not listed: {name}")
    for name in sorted(extra):
        problems.append(f"manifest.json: listed file not on disk: {name}")
    for name in sorted(on_disk & set(listed)):
        actual = sha256_file(root / name)
        if actual != listed[name]:
            problems.append(f"{name}: sha256 mismatch (manifest {listed[name][:12]}…, "
                            f"actual {actual[:12]}…)")
    return problems, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bank", help="path to data/signdomains/<domain>")
    ap.add_argument("--json", action="store_true", help="machine-readable result")
    args = ap.parse_args(argv)

    problems, stats = check_bank(args.bank)
    if args.json:
        print(json.dumps({"bank": args.bank, "accepted": not problems,
                          "stats": stats, "problems": problems}, indent=2))
    else:
        print(f"bank: {args.bank}")
        print(f"  fit={stats.get('n_fit', 0)} "
              f"eval_normal={stats.get('n_eval_normal', 0)} "
              f"eval_anomaly={stats.get('n_eval_anomaly', 0)}")
        if problems:
            print(f"REJECTED — {len(problems)} violation(s):")
            for p in problems:
                print(f"  - {p}")
        else:
            print("ACCEPTED — contract satisfied")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
