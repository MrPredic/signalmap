"""WEG 5 — RECURRENCE QUANTIFICATION (RQA, TRIANGULATION path 5, NEW).

Fifth independent path onto the Kilauea precursor question, and the first one
run at MULTIPLE offsets in one pass (2h/6h/12h) to start mapping how far
before onset any signal exists -- the lead-time question directly.

  - Weg 1 = learned RF on [perm_entropy, psd_slope] (multivariate boundary).
  - Weg 2 = directed AR1/variance of the slow amplitude ENVELOPE (raw).
  - Weg 3 = directed single spectral-shape descriptors (within-window).
  - Weg 5 = phase-space RECURRENCE STRUCTURE (determinism/laminarity/entropy
    of the recurrence plot) -- a nonlinear-dynamics representation distinct
    from all three: not a learned boundary (unlike Weg1), not a linear
    envelope statistic (unlike Weg2), not a Fourier-domain shape (unlike
    Weg3). Reference implementation (pyunicorn), FIXED (dim=3, tau=5,
    recurrence_rate=0.1) a priori -- no grid search, matching the
    reuse-over-rebuild convention from rqa_fair.py and avoiding the
    selection-bias that a config search would introduce here.

NO NEW FETCH: reuses volcano_precursor._bank(sta, off, p) on the frozen
precursor plan (628 cached znormed windows) at all three already-fetched
offsets (2h PRIMARY per the original design, 6h/12h exploratory) -- zero
new IRIS calls.

PRE-REGISTERED DESIGN (ledger RQA-PREREG, BEFORE any RQA feature computed):
  - Per episode, per station, per offset (2h/6h/12h): segment = mean
    descriptor over its 8 znormed windows (same aggregation as Weg 3).
  - Descriptors (fixed a priori, distinct from Weg1/Weg3): DET (determinism,
    l_min=2), LAM (laminarity, v_min=2), diag-ENTR (recurrence entropy).
  - PRIMARY tests: TWO-SIDED paired sign test + Wilcoxon on episode-wise
    PRE-vs-MID per descriptor (no a-priori sign, matching Weg3's honesty:
    RQA-near-bifurcation literature is mixed on direction). MOVES =
    sign_p<0.05 AND wilcoxon_p<0.05. UWE primary, RIMD replication.
  - MULTIPLICITY (registered): 3 offsets x 3 descriptors x 2 stations = 18
    tests. Holm-Bonferroni correction applied WITHIN each station across the
    9 (offset x descriptor) cells; verdict per offset reported both raw and
    corrected. This is a scan across offsets (lead-time mapping), not a
    single confirmatory test -- corrected verdicts are the ones that count.
  - Claim scope: site-local (Kilauea summit), recurrence-structure level.

Usage:
  .venv-research/bin/python rqa_precursor.py prereg
  .venv-research/bin/python rqa_precursor.py run     # cache-only, no fetch
"""
import hashlib, json, os, sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from receipt_ledger import log_receipt
from volcano_fresh import EP_HST, STATIONS
from volcano_precursor import _bank, _pre_check as _prec_pre_check, OFFSETS_H

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "rqa_precursor_prereg.json")

DIM, TAU, RR = 3, 5, 0.1
DESCS = ("det", "lam", "entr")


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _rqa(w):
    from pyunicorn.timeseries import RecurrencePlot
    rp = RecurrencePlot(np.asarray(w, float), dim=DIM, tau=TAU,
                         recurrence_rate=RR, silence_level=3)
    return (float(rp.determinism(l_min=2)), float(rp.laminarity(v_min=2)),
            float(rp.diag_entropy()))


def prereg():
    os.makedirs(FROZEN, exist_ok=True)
    p = _prec_pre_check()
    spec = {
        "prereg": "RQA-PRECURSOR (WEG 5, recurrence quantification, TRIANGULATION path 5, multi-offset)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "representation": "phase-space recurrence structure of the within-window znormed HHZ segment (pyunicorn RecurrencePlot), reused precursor cache -- reference impl, NO grid search",
        "params_fixed_a_priori": {"dim": DIM, "tau": TAU, "recurrence_rate": RR},
        "descriptors": "det=determinism(l_min=2), lam=laminarity(v_min=2), entr=diag-entropy; DISTINCT from Weg1 (perm_entropy,psd_slope) and Weg3 (centroid,spec_ent,hf_ratio)",
        "offsets_h": list(OFFSETS_H),
        "unit": "per episode per station per offset: PRE vs MID (same pause), segment = mean descriptor over its 8 znormed windows",
        "primary_tests": "per (offset,descriptor) TWO-SIDED paired sign test + Wilcoxon, episode-wise PRE-vs-MID; MOVES if sign_p<0.05 AND wilcoxon_p<0.05; UWE primary, RIMD replication",
        "multiplicity": "3 offsets x 3 descriptors = 9 cells per station; Holm-Bonferroni within-station; scan for lead-time mapping, corrected verdict is authoritative",
        "no_new_fetch": "reuses volcano_precursor._bank on the already-frozen/fetched precursor plan; zero new IRIS calls",
        "orthogonality": "nonlinear recurrence-structure representation, independent of learned boundary (Weg1), linear envelope stats (Weg2), Fourier shape (Weg3)",
        "claim_scope": "site-local (Kilauea summit), recurrence-structure level",
        "episode_table_sha256": _sha_text(json.dumps(EP_HST)),
        "prec_plan_sha256": _sha_text(json.dumps(p)),
    }
    with open(PREREG, "w") as f:
        json.dump({"spec": spec}, f, indent=1)
    tip = log_receipt("RQA-PRECURSOR-PREREG", spec)
    print(f"prereg written. ledger tip = {tip}", flush=True)


def _pre_check():
    assert os.path.exists(PREREG), "run prereg first (ledger before readout!)"
    with open(PREREG) as f:
        spec = json.load(f)["spec"]
    p = _prec_pre_check()
    assert _sha_text(json.dumps(p)) == spec["prec_plan_sha256"], "precursor plan tampered"
    return p


def _paired(a, b):
    from scipy.stats import binomtest, wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b) & (a != b)
    a, b = a[m], b[m]
    n = len(a)
    if n == 0:
        return {"n": 0, "pre_gt_mid": "0/0", "sign_p": 1.0, "wilcoxon_p": 1.0}
    up = int((a > b).sum())
    sp = float(binomtest(up, n, 0.5, alternative="two-sided").pvalue)
    try:
        wp = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except ValueError:
        wp = 1.0
    return {"n": n, "pre_gt_mid": f"{up}/{n}", "sign_p": sp, "wilcoxon_p": wp}


def _holm(pvals_named, alpha=0.05):
    items = sorted(pvals_named.items(), key=lambda kv: kv[1])
    m = len(items)
    sig, out = True, {}
    for i, (k, pv) in enumerate(items):
        thresh = alpha / (m - i)
        passed = sig and (pv < thresh)
        sig = sig and (pv < thresh)
        out[k] = {"p": pv, "holm_thresh": thresh, "PASS": bool(passed)}
    return out


def run():
    p = _pre_check()
    results = {}
    for sta in STATIONS:
        cell_min_p = {}
        r_sta = {"role": "PRIMARY" if sta == "UWE" else "replication", "offsets": {}}
        for off in OFFSETS_H:
            raw = _bank(sta, off, p)
            by = {}
            for w, lab, ep in raw:
                by.setdefault(int(ep), {"pre": [], "mid": []})[lab].append(_rqa(w))
            cols = {d: {"pre": [], "mid": []} for d in DESCS}
            n_ep = 0
            for ep, segs in by.items():
                if not segs["pre"] or not segs["mid"]:
                    continue
                n_ep += 1
                pre_m = np.mean(segs["pre"], axis=0)
                mid_m = np.mean(segs["mid"], axis=0)
                for j, d in enumerate(DESCS):
                    cols[d]["pre"].append(float(pre_m[j]))
                    cols[d]["mid"].append(float(mid_m[j]))
            r_off = {"n_episodes": n_ep}
            for d in DESCS:
                t = _paired(cols[d]["pre"], cols[d]["mid"])
                t["MOVES_raw"] = bool(t["sign_p"] < 0.05 and t["wilcoxon_p"] < 0.05)
                r_off[d] = t
                cell_min_p[f"{off}h:{d}"] = max(t["sign_p"], t["wilcoxon_p"])
            r_sta["offsets"][f"{off}h"] = r_off
            print(f"RQA-{sta}-{off}h [{r_sta['role']}] n_ep={n_ep}: " +
                  " | ".join(f"{d} {r_off[d]['pre_gt_mid']} sign_p={r_off[d]['sign_p']:.4f} "
                             f"wil_p={r_off[d]['wilcoxon_p']:.4f}{'*' if r_off[d]['MOVES_raw'] else ''}"
                             for d in DESCS), flush=True)
        holm = _holm(cell_min_p)
        for cell, h in holm.items():
            off, d = cell.split(":")
            r_sta["offsets"][off][d]["holm"] = h
        moved_corrected = [c for c, h in holm.items() if h["PASS"]]
        r_sta["holm_corrected_moves"] = moved_corrected
        r_sta["verdict"] = "RQA-SIGNAL" if moved_corrected else "NULL (Holm-corrected)"
        results[f"RQA-{sta}"] = r_sta
        print(f"=> {sta} Holm-corrected: {moved_corrected or 'none'} -> {r_sta['verdict']}", flush=True)
    tip = log_receipt("RQA-PRECURSOR", {"prec_plan_sha256": _sha_text(json.dumps(p)),
                                        "results": results})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "run": run}[sys.argv[1]]()
