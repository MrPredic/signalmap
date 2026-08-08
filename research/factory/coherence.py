"""WEG 4 — CROSS-STATION COHERENCE (TRIANGULATION path 4, NEW, spatial).

Fourth independent path onto the Kilauea precursor question, and the first
SPATIAL/RELATIONAL one -- all prior paths (Weg1/2/3/5) are univariate: one
station's signal judged against itself (PRE vs MID). Weg 4 asks a different
question entirely: do UWE and RIMD's amplitude envelopes become MORE (or
less) COUPLED shortly before an episode, vs mid-pause? Error mode is
structurally orthogonal to all univariate paths -- a coupling artifact
cannot be produced by any single-station statistic being noisy or drifting.

Physical motivation (soft prior only, NOT a directed hypothesis -- literature
does not establish a sign for pre-eruptive summit-station coherence at
Kilauea specifically, so this is registered TWO-SIDED like Weg 3): system-
wide pressurization before fountaining could plausibly increase inter-
station coupling (shared source mechanism) or decrease it (localized venting
noise decorrelating the two channels). Both directions are informative.

Statistic: magnitude-squared coherence (Welch cross-spectral, scipy) between
the UWE and RIMD detrended RMS envelope at the SAME absolute UTC segment --
the two stations' envelopes ARE time-aligned (same t0/t1 fetch), unlike the
peak-picked znormed windows used by Weg1/3/5 (independent per-station peak
selection there breaks cross-station alignment, which is why coherence is
NOT computed on that cache). Descriptor = mean coherence over the full
resolvable band (envelope Nyquist ~0.10 Hz, i.e. periods 10s-1800s).

NO NEW FETCH: reuses the already-fetched csd_env cache (EARLY/MID/PRE-12h,
pause_phase.plan(), 264 files) -- zero new IRIS calls.

PRE-REGISTERED DESIGN (ledger COHERENCE-PREREG, BEFORE any coherence computed):
  - Per episode, station-PAIR (UWE,RIMD) jointly: coherence(PRE-envelope,
    PRE-envelope) at EARLY, MID, and PRE(-12h) phase points (pause_phase
    plan, hash-checked).
  - Descriptor: mean_coh = mean magnitude-squared coherence over all Welch
    frequency bins (nperseg=min(64,len), noverlap=len//2).
  - PRIMARY test: TWO-SIDED paired sign test + Wilcoxon, episode-wise
    PRE-vs-MID mean_coh. MOVES = sign_p<0.05 AND wilcoxon_p<0.05.
  - CONTROL (gradient disambiguation, same interpretation map as CSD/Weg2):
    EARLY vs MID, two-sided. EARLY~MID AND PRE moves -> late-pause-specific;
    EARLY moves same direction -> monotone gradient downgrade.
  - Single station-pair only (UWE-RIMD is the only pair with both stations
    fetched at every phase point) -- no UWE/RIMD primary-replication split
    here; this path is inherently one joint test, reported as such.
  - Claim scope: site-local (Kilauea summit), inter-station coupling level.

Usage:
  .venv-research/bin/python coherence.py prereg
  .venv-research/bin/python coherence.py run     # cache-only, no fetch
"""
import hashlib, json, os, sys

import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from receipt_ledger import log_receipt
from volcano_fresh import EP_HST
import csd, pause_phase

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "coherence_prereg.json")
FS_ENV = 100.0 / 512  # envelope sample rate: raw fs=100Hz, hop=512


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _mean_coh(a, b):
    from scipy.signal import coherence
    n = min(len(a), len(b))
    if n < 8:
        return float("nan")
    nper = min(64, n)
    f, Cxy = coherence(a[:n], b[:n], fs=FS_ENV, nperseg=nper, noverlap=nper // 2)
    return float(np.mean(Cxy))


def prereg():
    os.makedirs(FROZEN, exist_ok=True)
    p = pause_phase.plan()
    n_valid = sum(1 for x in p if x["early"] and x["mid"] and x["pre"])
    spec = {
        "prereg": "COHERENCE (WEG 4, cross-station UWE-RIMD envelope coherence, TRIANGULATION path 4, spatial)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "representation": "magnitude-squared coherence (Welch) between UWE and RIMD detrended RMS envelope, same absolute UTC segment (time-aligned by construction, unlike peak-picked znormed windows)",
        "descriptor": "mean_coh = mean coherence over all resolvable Welch freq bins, nperseg=min(64,len), noverlap=len//2, fs_env=100/512 Hz",
        "phase_points": "EARLY/MID/PRE(-12h) from pause_phase.plan() (hash-checked, reused, no new fetch)",
        "primary_test": "TWO-SIDED paired sign test + Wilcoxon, episode-wise PRE-vs-MID mean_coh; MOVES if sign_p<0.05 AND wilcoxon_p<0.05 (no directed sign a priori -- literature does not establish direction for this system)",
        "control": "EARLY vs MID two-sided; early~mid AND pre-moves -> late-pause-specific; early moves same-dir -> monotone gradient downgrade",
        "structure": "single joint UWE-RIMD test (spatial method needs both stations at once); no primary/replication split",
        "no_new_fetch": "reuses csd_env cache (pause_phase plan, 264 files); zero new IRIS calls",
        "orthogonality": "spatial/relational (inter-station coupling), structurally independent of all univariate paths Weg1/2/3/5 (single-station statistics)",
        "claim_scope": "site-local (Kilauea summit), inter-station coupling level",
        "episode_table_sha256": _sha_text(json.dumps(EP_HST)),
        "plan_sha256": _sha_text(json.dumps(p)),
        "n_valid_triples": n_valid,
    }
    with open(PREREG, "w") as f:
        json.dump({"spec": spec, "plan": p}, f, indent=1)
    tip = log_receipt("COHERENCE-PREREG", spec)
    print(f"prereg written, n_valid_triples={n_valid}. ledger tip = {tip}", flush=True)


def _pre_check():
    assert os.path.exists(PREREG), "run prereg first (ledger before readout!)"
    with open(PREREG) as f:
        pre = json.load(f)
    assert _sha_text(json.dumps(pre["plan"])) == pre["spec"]["plan_sha256"], "plan tampered"
    return pre["plan"]


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


def run():
    p = _pre_check()
    cols = {"early": [], "mid": [], "pre": []}
    n_ep = 0
    for x in p:
        if not (x["early"] and x["mid"] and x["pre"]):
            continue
        envs = {}
        ok = True
        for k in ("early", "mid", "pre"):
            eu = csd._fetch_env("UWE", x[k])
            er = csd._fetch_env("RIMD", x[k])
            if eu is None or er is None:
                ok = False
                break
            envs[k] = _mean_coh(eu, er)
        if not ok:
            continue
        n_ep += 1
        for k in ("early", "mid", "pre"):
            cols[k].append(envs[k])
    rise = _paired(cols["pre"], cols["mid"])
    rise["MOVES"] = bool(rise["sign_p"] < 0.05 and rise["wilcoxon_p"] < 0.05)
    ctrl = _paired(cols["early"], cols["mid"])
    ctrl["MOVES"] = bool(ctrl["sign_p"] < 0.05 and ctrl["wilcoxon_p"] < 0.05)
    verdict = "COHERENCE-SHIFT" if rise["MOVES"] else "NULL"
    r = {
        "n_episodes": n_ep,
        "pre_vs_mid": rise, "early_vs_mid": ctrl,
        "median": {"early": float(np.nanmedian(cols["early"])) if cols["early"] else None,
                   "mid": float(np.nanmedian(cols["mid"])) if cols["mid"] else None,
                   "pre": float(np.nanmedian(cols["pre"])) if cols["pre"] else None},
        "verdict": verdict,
    }
    print(f"COHERENCE UWE-RIMD n_ep={n_ep}: PRE-vs-MID {rise['pre_gt_mid']} "
          f"sign_p={rise['sign_p']:.4f} wil_p={rise['wilcoxon_p']:.4f} "
          f"{'*MOVES*' if rise['MOVES'] else ''} | EARLY-vs-MID {ctrl['pre_gt_mid']} "
          f"sign_p={ctrl['sign_p']:.4f} wil_p={ctrl['wilcoxon_p']:.4f} "
          f"=> {verdict}", flush=True)
    tip = log_receipt("COHERENCE", {"plan_sha256": _sha_text(json.dumps(p)), "results": r})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "run": run}[sys.argv[1]]()
