"""WEG 3 — SPECTRAL WANDERING (TRIANGULATION path 3).

Third independent path onto the -12h Kilauea precursor claim. Representation =
the WITHIN-WINDOW power spectrum SHAPE; error mode = amplitude confound (which
is neutralised by construction here, see below).

  - Weg 1 = learned RF on [perm_entropy, psd_slope] (a multivariate boundary).
  - Weg 2 = directed AR1/variance of the slow amplitude ENVELOPE (raw).
  - Weg 3 = directed PAIRED test on single interpretable SPECTRAL descriptors
    that "wander" toward onset, on the SAME znormed windows Weg 1 sees.

Descriptors chosen DISTINCT from Weg 1's two features (perm_entropy is ordinal/
temporal, psd_slope is the 1/f exponent):
  - centroid   : spectral centroid Sum(f*P)/Sum(P) in Hz  (energy migration)
  - spec_ent   : normalised spectral entropy of P          (broadening)
  - hf_ratio   : power above the segment-median frequency / total (band shift)

AMPLITUDE-CONFOUND: handled by construction. The reused cache stores z-normed
windows (_znorm detrends + unit-scales), so every descriptor is a pure SHAPE
statistic, amplitude-invariant -- exactly the control the plan asks for. NO new
fetch: reuses volcano_precursor._bank(off=12) on the frozen precursor plan.

Honesty note (registered): Weg 3 shares the within-window spectral REPRESENTATION
with Weg 1's psd_slope feature, so it is NOT fully representation-orthogonal to
Weg 1 (it IS to Weg 2, envelope dynamics on raw amplitude). Its error mode still
differs (single directed descriptor, cannot overfit vs learned boundary). The
consilience shuffle-null is what exposes any shared artifact.

PRE-REGISTERED DESIGN (ledger SPECTRAL-PREREG, BEFORE any descriptor computed):
  - Per episode, per station (UWE primary, RIMD replication): PRE(-12h) vs MID
    (same pause), each segment = mean descriptor over its 8 znormed windows.
  - PRIMARY (per descriptor): TWO-SIDED paired sign test + Wilcoxon on episode-
    wise PRE-vs-MID (no a-priori sign for "wandering"); a descriptor "moves" if
    sign_p<0.05 AND wilcoxon_p<0.05. Per-episode signed direction is retained
    for the consilience layer. UWE primary, RIMD replication.
  - Reported: which descriptors move, the consistent direction, medians.
  - Claim scope: site-local (Kilauea summit), spectral-shape level.

Usage:
  ../../.venv-research/bin/python spectral.py prereg
  ../../.venv-research/bin/python spectral.py run     # cache-only, no fetch
"""
import hashlib, json, os, sys
from datetime import datetime

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from receipt_ledger import log_receipt
from volcano_fresh import EP_HST, STATIONS, WIN_PER_SEG
from volcano_precursor import _bank, _pre_check as _prec_pre_check
from harvest3_loaders import W

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "spectral_prereg.json")
FS = 100.0  # HHZ sampling rate


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _descriptors(w):
    """Amplitude-invariant spectral-shape descriptors of one znormed window."""
    f, P = welch(np.asarray(w, float), fs=FS, nperseg=min(256, len(w)))
    f, P = f[1:], P[1:]                       # drop DC
    Ps = P.sum() + 1e-30
    centroid = float((f * P).sum() / Ps)
    p = P / Ps
    spec_ent = float(-(p * np.log(p + 1e-30)).sum() / np.log(len(p)))
    fmed = float(np.median(f))
    hf_ratio = float(P[f > fmed].sum() / Ps)
    return centroid, spec_ent, hf_ratio


DESCS = ("centroid", "spec_ent", "hf_ratio")


def prereg():
    os.makedirs(FROZEN, exist_ok=True)
    p = _prec_pre_check()  # verify frozen precursor plan hash before reuse
    spec = {
        "prereg": "SPECTRAL (WEG 3, spectral-wandering, TRIANGULATION path 3)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "representation": "within-window power spectrum SHAPE (Welch) on the reused znormed precursor cache; amplitude-invariant by construction",
        "descriptors": "centroid=Sum(f*P)/Sum(P) Hz, spec_ent=normalised spectral entropy, hf_ratio=power above segment-median-freq / total; DISTINCT from Weg1 (perm_entropy, psd_slope)",
        "unit": "per episode per station: PRE(-12h) vs MID (same pause), segment = mean descriptor over its 8 znormed windows",
        "primary_tests": "per descriptor TWO-SIDED paired sign test + Wilcoxon on episode-wise PRE-vs-MID; MOVES if sign_p<0.05 AND wilcoxon_p<0.05; per-episode signed direction retained for consilience; UWE primary, RIMD replication",
        "amplitude_confound": "neutralised by construction (znormed windows -> pure shape statistics)",
        "orthogonality_honesty": "shares within-window spectral representation with Weg1 psd_slope (NOT fully orthogonal to Weg1); orthogonal to Weg2 (raw envelope dynamics); consilience shuffle-null exposes shared artifacts",
        "claim_scope": "site-local (Kilauea summit), spectral-shape level",
        "episode_table_sha256": _sha_text(json.dumps(EP_HST)),
        "prec_plan_sha256": _sha_text(json.dumps(p)),
    }
    with open(PREREG, "w") as f:
        json.dump({"spec": spec}, f, indent=1)
    tip = log_receipt("SPECTRAL-PREREG", spec)
    print(f"prereg written. ledger tip = {tip}", flush=True)


def _pre_check():
    assert os.path.exists(PREREG), "run prereg first (ledger before readout!)"
    with open(PREREG) as f:
        spec = json.load(f)["spec"]
    p = _prec_pre_check()
    assert _sha_text(json.dumps(p)) == spec["prec_plan_sha256"], "precursor plan tampered"
    return p


def _paired(a, b):
    """Two-sided paired sign test + Wilcoxon; returns direction (PRE>MID share)."""
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


def run():
    p = _pre_check()
    results = {}
    for sta in STATIONS:
        raw = _bank(sta, 12, p)  # cache-only znormed windows at -12h offset
        # aggregate per episode: mean descriptor over the 8 windows of each segment
        by = {}  # ep -> {"pre":[desc per window], "mid":[...]}
        for w, lab, ep in raw:
            by.setdefault(ep, {"pre": [], "mid": []})[lab].append(_descriptors(w))
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
        r = {"role": "PRIMARY" if sta == "UWE" else "replication", "n_episodes": n_ep}
        moved = []
        for d in DESCS:
            t = _paired(cols[d]["pre"], cols[d]["mid"])
            t["MOVES"] = bool(t["sign_p"] < 0.05 and t["wilcoxon_p"] < 0.05)
            t["median_pre"] = float(np.median(cols[d]["pre"])) if cols[d]["pre"] else None
            t["median_mid"] = float(np.median(cols[d]["mid"])) if cols[d]["mid"] else None
            r[d] = t
            if t["MOVES"]:
                moved.append(d)
        r["descriptors_moved"] = moved
        r["spectral_verdict"] = "WANDER" if moved else "NULL"
        results[f"SPECTRAL-{sta}"] = r
        print(f"SPECTRAL-{sta} [{r['role']}] n_ep={n_ep}: " +
              " | ".join(f"{d} {r[d]['pre_gt_mid']} sign_p={r[d]['sign_p']:.4f} "
                         f"wil_p={r[d]['wilcoxon_p']:.4f}{'*' if r[d]['MOVES'] else ''}"
                         for d in DESCS) +
              f" => {r['spectral_verdict']} {moved}", flush=True)
    tip = log_receipt("SPECTRAL", {"prec_plan_sha256": _sha_text(json.dumps(p)),
                                   "results": results})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "run": run}[sys.argv[1]]()
