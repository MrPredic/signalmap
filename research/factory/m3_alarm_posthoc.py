#!/usr/bin/env python3
"""M3 redone in the unit the product actually alarms in — POST-HOC.

    nice -n 19 .venv/bin/python research/factory/m3_alarm_posthoc.py

NOT PREREGISTERED. Prereg AMENDMENT 3 records why: the first M3 run compared
the detector's window-calibrated threshold against a per-recording mean, which
is structurally silent, and those zeros had already been seen when the mistake
was found. This rerun is therefore reported as post-hoc and does not count as
evidence for or against H3. It is here because "the shipped detector never
fires" is a claim worth getting right either way.

`DistilledDetector.alert(w)` is `score(w) >= threshold`, evaluated per WINDOW.
Two rates are reported per domain:

  window level     fraction of all eval windows that alarm
  recording level  fraction of recordings with at least one alarming window
                   -- the operational rule, since an operator sees a machine,
                   not a window

The number that matters is the GAP between the anomaly rate and the normal
rate. A gap at or below zero means the alarm carries no usable signal in that
domain, whatever the AUC says.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "factory"))

from signalmap.distill import DistilledDetector, FeatureSpec  # noqa: E402

from sign_identifiability_readout import (  # noqa: E402
    BANKS, ENVELOPE, N_BOOT, SEED, SPEC_PROGRAMS, W, load_domain,
)

OUT = ROOT / "research" / "factory" / "logs" / "m3_alarm_posthoc.md"


def boot_gap(flags_a, flags_n, seed=SEED):
    """Bootstrap the anomaly-minus-normal alarm-rate gap over recordings."""
    rng = np.random.default_rng(seed)
    a, n = np.asarray(flags_a, float), np.asarray(flags_n, float)
    if a.size == 0 or n.size == 0:
        return float("nan"), float("nan")
    gaps = [rng.choice(a, a.size, replace=True).mean()
            - rng.choice(n, n.size, replace=True).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)


def run(name, root):
    fit_wins, ev, leaked = load_domain(root)
    spec = FeatureSpec(programs=list(SPEC_PROGRAMS), premium=[], window=W)
    det = DistilledDetector.fit(spec, fit_wins, envelope=ENVELOPE)

    per_class = {0: {"win": [], "rec": []}, 1: {"win": [], "rec": []}}
    for _, lab, wins in ev:
        flags = [det.score(w) >= det.threshold for w in wins]   # det.alert()
        per_class[lab]["win"].extend(flags)
        per_class[lab]["rec"].append(any(flags))

    out = {"domain": name, "threshold": float(det.threshold),
           "leaked": leaked, "n_eval": len(ev)}
    for level in ("win", "rec"):
        a = np.asarray(per_class[1][level], float)
        n = np.asarray(per_class[0][level], float)
        lo, hi = boot_gap(a, n)
        out[level] = {"anomaly": float(a.mean()) if a.size else float("nan"),
                      "normal": float(n.mean()) if n.size else float("nan"),
                      "gap": float(a.mean() - n.mean()) if a.size and n.size
                      else float("nan"),
                      "ci_lo": lo, "ci_hi": hi, "n_anomaly": int(a.size),
                      "n_normal": int(n.size)}
    return out


def main():
    rows = []
    for root in sorted(p for p in BANKS.iterdir() if (p / "manifest.json").exists()):
        print(f"[{root.name}] ...", flush=True)
        r = run(root.name, root)
        rows.append(r)
        print(f"[{root.name}] window {r['win']['anomaly']:.3f}/{r['win']['normal']:.3f} "
              f"| recording {r['rec']['anomaly']:.3f}/{r['rec']['normal']:.3f} "
              f"gap {r['rec']['gap']:+.3f}", flush=True)

    lines = [
        "# M3 — does the shipped alarm fire? (POST-HOC, not preregistered)",
        "",
        "Prereg AMENDMENT 3: the preregistered M3 was invalid (window-calibrated "
        "threshold compared against a recording mean) and its zeros had been seen "
        "before the error was found. This rerun uses `score(w) >= threshold` per "
        "window, exactly as `DistilledDetector.alert()` does, and is reported as "
        "post-hoc. It is not evidence for or against H3.",
        "",
        "| domain | threshold | window a/n | recording a/n | recording gap [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["rec"]["gap"]):
        lines.append(
            f"| {r['domain']} | {r['threshold']:.2f} | "
            f"{r['win']['anomaly']:.3f} / {r['win']['normal']:.3f} | "
            f"{r['rec']['anomaly']:.3f} / {r['rec']['normal']:.3f} | "
            f"{r['rec']['gap']:+.3f} [{r['rec']['ci_lo']:+.3f}, "
            f"{r['rec']['ci_hi']:+.3f}] |")
    silent = [r["domain"] for r in rows
              if r["win"]["anomaly"] == 0.0 and r["win"]["normal"] == 0.0]
    useless = [r["domain"] for r in rows if r["rec"]["ci_hi"] <= 0]
    lines += [
        "",
        f"**Never fires at all** (no window alarms in either class): "
        f"{silent or 'none'}.",
        f"**Alarm gap not above zero** (CI upper bound <= 0): {useless or 'none'}.",
        "",
        "Reproduce: `nice -n 19 .venv/bin/python "
        "research/factory/m3_alarm_posthoc.py`",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(rows, indent=2)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
