#!/usr/bin/env python3
"""Is the valve AUC inversion a real effect or the degenerate-feature artefact?

    nice -n 19 .venv/bin/python study/tools/degenerate_feature_probe.py

`window()` z-normalises every window, so `std(id(id(x)))` is exactly 1.0 by
construction. Its healthy MAD therefore collapses onto the 1e-12 guard floor
in `DistilledDetector.fit`, and last-bit rounding differences get divided by
1e-12 — the feature contributes a first-order term to max|z| that tracks the
input amplitude's floating-point signature rather than the signal.

The valve readout of 2026-07-21 (the 2026-07 valve preregistration) reported
AUC 0.2642, CI [0.1929, 0.3360] using a 9-program spec that contains exactly
that feature. This probe re-scores the SAME frozen bank two ways — with the
spec as shipped, and with the degenerate feature removed — and reports both
AUCs with bootstrap CIs. Nothing else changes: same bank, same windows, same
aggregation, same seed.

Reads only; writes a report and no receipts. It decides which of two claims
is true, and both are publishable:
  * inversion survives  -> the sign finding is real, the artefact is separate
  * inversion vanishes  -> the headline was numerical, and we say so
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from signalmap.distill import DistilledDetector, FeatureSpec, window  # noqa: E402

sys.path.insert(0, str(ROOT / "study" / "tools"))
from sign_identifiability_readout import auc, boot_ci  # noqa: E402

BANK = ROOT / "data" / "mimii" / "valve_id00_bank"
OUT = ROOT / "study" / "degenerate_feature_probe.md"

SHIPPED = ["acf1(id(id(x)))", "crest(id(id(x)))", "lcross(id(id(x)))",
           "meanabs(id(id(x)))", "peakcv(id(id(x)))", "runcv(id(id(x)))",
           "runmean(id(id(x)))", "std(id(id(x)))", "zcr(id(id(x)))"]
DEGENERATE = "std(id(id(x)))"
K = 20
GUARD = 1e-12


def load_windows(path):
    arr = np.load(path, allow_pickle=False)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return window(np.ascontiguousarray(arr[: K * 1024], dtype=np.float64))[:K]


def score_bank(programs, fit_wins, eval_items):
    spec = FeatureSpec(programs=list(programs), premium=[], window=1024)
    det = DistilledDetector.fit(spec, fit_wins, envelope=3.0)
    mad = np.asarray(det.mad, dtype=float)
    degenerate = [p for p, m in zip(programs, mad) if m <= GUARD * 1.000001]
    y = np.array([lab for _, lab, _ in eval_items])
    s = np.array([float(np.mean([det.score(w) for w in wins]))
                  for _, _, wins in eval_items])
    observed = auc(y, s)
    lo, hi, _ = boot_ci(y, s)
    return {"programs": list(programs), "degenerate_on_guard_floor": degenerate,
            "auc": observed, "ci_lo": lo, "ci_hi": hi,
            "threshold": float(det.threshold),
            "direction": ("inverted" if hi < 0.5 else
                          "aligned" if lo > 0.5 else "undetermined")}


def main():
    t0 = time.time()
    if not BANK.is_dir():
        raise SystemExit(f"{BANK} missing")

    fit_paths = sorted((BANK / "train").glob("*.npy"))
    ev_paths = sorted((BANK / "test").glob("*.npy"))
    if not fit_paths or not ev_paths:
        raise SystemExit(f"bank empty: {len(fit_paths)} fit, {len(ev_paths)} eval")

    print(f"loading {len(fit_paths)} fit + {len(ev_paths)} eval recordings ...", flush=True)
    fit_wins = []
    for p in fit_paths:
        fit_wins.extend(load_windows(p))
    eval_items = [(p.name, 1 if p.name.startswith("anomaly") else 0, load_windows(p))
                  for p in ev_paths]
    n_anom = sum(lab for _, lab, _ in eval_items)
    print(f"{len(fit_wins)} healthy windows | eval {len(eval_items)} clips "
          f"({n_anom} anomaly, {len(eval_items) - n_anom} normal)", flush=True)

    reduced = [p for p in SHIPPED if p != DEGENERATE]
    results = {}
    for label, programs in (("shipped_9", SHIPPED), ("without_degenerate_8", reduced)):
        print(f"[{label}] scoring ...", flush=True)
        results[label] = score_bank(programs, fit_wins, eval_items)
        r = results[label]
        print(f"[{label}] AUC={r['auc']:.4f} CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] "
              f"-> {r['direction']} | degenerate={r['degenerate_on_guard_floor']}",
              flush=True)

    b = results["without_degenerate_8"]
    verdict = ("the inversion SURVIVES removal of the degenerate feature"
               if b["direction"] == "inverted" else
               "the inversion DEPENDS on the degenerate feature")
    lines = [
        "# Does the valve AUC inversion depend on a feature that is constant by construction?",
        "",
        f"Bank: `{BANK.relative_to(ROOT)}` (frozen by the 2026-07 valve preregistration).",
        f"Fit windows: {len(fit_wins)} | eval clips: {len(eval_items)} "
        f"({n_anom} anomaly, {len(eval_items) - n_anom} normal). Seed 0, K={K}.",
        "",
        "| spec | AUC | 95% CI | direction | features on the 1e-12 guard floor |",
        "|---|---|---|---|---|",
    ]
    for label in ("shipped_9", "without_degenerate_8"):
        r = results[label]
        lines.append(f"| {label} | {r['auc']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | "
                     f"{r['direction']} | {r['degenerate_on_guard_floor'] or 'none'} |")
    lines += [
        "",
        f"**Finding: {verdict}.**",
        "",
        "Reference: the 2026-07-21 readout reported AUC 0.2642, CI "
        "[0.1929, 0.3360] on this bank with the shipped 9-program spec.",
        "",
        f"Reproduce: `nice -n 19 .venv/bin/python "
        f"study/tools/degenerate_feature_probe.py` ({round(time.time() - t0, 1)}s)",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
