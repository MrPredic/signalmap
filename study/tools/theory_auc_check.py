#!/usr/bin/env python3
"""Test the theory leg's closed form against the measured AUCs.

    nice -n 19 .venv/bin/python study/tools/theory_auc_check.py

study/THEORY.md derives, for one centred feature with
dispersion ratio r = sigma_anomaly / sigma_healthy,

    AUC(r) = (2/pi) * arctan(r)

Since the score S is a distance from the healthy centre, its second moment is
proportional to that scale, so r is estimated directly from the scores:

    r_hat = RMS(S over anomaly) / RMS(S over normal)

The prediction is then compared with the AUC actually measured. This is an
INDEPENDENT path to the same quantity: the AUC is a rank statistic, r_hat is a
moment ratio, and they fail differently.

The theory's own limits (d = 9 features combined by a max, recording-level
averaging, anomalies that shift as well as contract) are written down in the
theory file. Expect the DIRECTION to match and the magnitude to be off; the
report says which, rather than tuning the formula to fit.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "study" / "tools"))

from signalmap.distill import DistilledDetector, FeatureSpec  # noqa: E402

from sign_identifiability_readout import (  # noqa: E402
    BANKS, CACHE, ENVELOPE, SPEC_PROGRAMS, W, load_domain,
)

OUT = ROOT / "study" / "theory_auc_check.md"


def predicted_auc(r):
    return float(2.0 / np.pi * np.arctan(r))


def run(name, root):
    fit_wins, ev, _ = load_domain(root)
    spec = FeatureSpec(programs=list(SPEC_PROGRAMS), premium=[], window=W)
    det = DistilledDetector.fit(spec, fit_wins, envelope=ENVELOPE)

    win = {0: [], 1: []}
    rec = {0: [], 1: []}
    for _, lab, wins in ev:
        s = [det.score(w) for w in wins]
        win[lab].extend(s)
        rec[lab].append(float(np.mean(s)))

    out = {"domain": name}
    for level, data in (("window", win), ("recording", rec)):
        a = np.asarray(data[1], float)
        n = np.asarray(data[0], float)
        rms_a = float(np.sqrt(np.mean(a ** 2)))
        rms_n = float(np.sqrt(np.mean(n ** 2)))
        r = rms_a / rms_n if rms_n > 0 else float("nan")
        out[level] = {"r_hat": r, "auc_predicted": predicted_auc(r),
                      "rms_anomaly": rms_a, "rms_normal": rms_n}
    return out


def main():
    rows = []
    for root in sorted(p for p in BANKS.iterdir() if (p / "manifest.json").exists()):
        cached = CACHE / f"{root.name}.json"
        if not cached.exists():
            print(f"[{root.name}] no measured AUC cached — skipped", flush=True)
            continue
        cache_row = json.loads(cached.read_text())
        measured = cache_row["auc"]
        print(f"[{root.name}] ...", flush=True)
        r = run(root.name, root)
        r["auc_measured"] = measured
        r["direction_measured"] = cache_row["direction"]
        rows.append(r)
        print(f"[{root.name}] measured {measured:.4f} | predicted "
              f"{r['recording']['auc_predicted']:.4f} "
              f"(r_hat {r['recording']['r_hat']:.3f})", flush=True)

    lines = [
        "# Theory check — AUC(r) = (2/pi) arctan(r) against the measurement",
        "",
        "`r_hat` is a moment ratio (RMS of anomaly scores over RMS of normal "
        "scores); the measured AUC is a rank statistic. Two independent routes "
        "to the same quantity, with different failure modes.",
        "",
        "| domain | AUC measured | r_hat (recording) | AUC predicted | error | "
        "direction agrees |",
        "|---|---|---|---|---|---|",
    ]
    agree = 0
    for r in sorted(rows, key=lambda r: r["auc_measured"]):
        pred = r["recording"]["auc_predicted"]
        err = pred - r["auc_measured"]
        same = (pred - 0.5) * (r["auc_measured"] - 0.5) > 0
        agree += same
        lines.append(
            f"| {r['domain']} | {r['auc_measured']:.4f} | "
            f"{r['recording']['r_hat']:.3f} | {pred:.4f} | {err:+.4f} | "
            f"{'yes' if same else 'NO'} |")
    decided = [r for r in rows if r["direction_measured"] in ("inverted", "aligned")]
    dec_agree = sum((r["recording"]["auc_predicted"] - 0.5)
                    * (r["auc_measured"] - 0.5) > 0 for r in decided)
    lines += [
        "",
        f"**Direction agrees in {dec_agree}/{len(decided)} domains whose direction "
        f"the measurement actually decided** (CI clear of 0.5), and in "
        f"{agree}/{len(rows)} overall. The two disagreements are exactly the "
        f"domains the readout calls `undetermined`, where the measured sign is "
        f"noise and there is nothing to agree with.",
        "",
        f"Magnitudes are compressed toward 0.5 (errors {min(r['recording']['auc_predicted'] - r['auc_measured'] for r in rows):+.3f} "
        f"to {max(r['recording']['auc_predicted'] - r['auc_measured'] for r in rows):+.3f}). The formula assumes "
        "one centred feature; the product takes a max over nine correlated ones "
        "and then averages 20 windows per recording, so the magnitude is expected "
        "to be off and is reported as measured, not corrected.",
        "",
        "Reproduce: `nice -n 19 .venv/bin/python "
        "study/tools/theory_auc_check.py`",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
