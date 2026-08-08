"""WEG 2 — CRITICAL SLOWING DOWN (theory anchor, directed, TRIANGULATION path 2).

Independent second path onto the -12h Kilauea precursor claim, with error
modes ORTHOGONAL to Weg 1 (volcano_precursor's learned texture classifier):

  - Weg 1 = learned RF on within-window texture (perm_entropy + psd_slope).
  - Weg 2 = a DIRECTED dynamical-systems statistic, no learning, no selection:
    the Scheffer/Dakos early-warning signature. Approaching a bifurcation the
    slow state variable's lag-1 autocorrelation (AR1) AND variance MUST rise.

Slow variable = the DETRENDED ROLLING-RMS AMPLITUDE ENVELOPE of the raw HHZ
segment (W=1024, hop=512, same envelope _fetch uses to find its RMS peak).
This is a different representation (envelope dynamics) than Weg 1 (within-
window texture) and Weg 3 (within-window 1/f spectrum) -> orthogonal.

Why a fresh RAW fetch (not the 628 cached precursor windows): the cache is
z-normed per window (_znorm detrends + scales to unit variance), which by
construction DESTROYS the variance signal (var==1) and the envelope. AR1 is
scale-invariant and would survive, but variance-rise -- half of the CSD
signature -- would not. So CSD needs raw amplitude. NO new selection: the
EARLY/MID/PRE(-12h) segment TIMES are reused verbatim from the already-frozen
precursor + pause_phase plans (deterministic from the committed EP_HST table).

PRE-REGISTERED DESIGN (ledger CSD-PREREG, written BEFORE any envelope loaded):
  - Per episode, per station (UWE primary, RIMD replication), three pause-phase
    points at the SAME clock: EARLY (first >= prev_end+24h), MID (mid-pause),
    PRE (-12h before onset). Times pinned by pause_phase.plan() (hash-checked).
  - Per segment two statistics of the detrended RMS envelope: ar1 = lag-1
    autocorrelation, logvar = log10(variance).
  - PRIMARY directed tests (1-sided GREATER, paired within episode across eps):
      AR1-rise : AR1(PRE)  > AR1(MID)
      VAR-rise : logVar(PRE) > logVar(MID)
    Each judged by exact binomial SIGN test (P(PRE>MID)=0.5 under H0) AND
    Wilcoxon signed-rank (1-sided). PASS(signal) = sign_p<0.05 AND wilcoxon_p
    <0.05. UWE = primary bank, RIMD = replication (same tests).
  - CONTROL (gradient disambiguation, reuses pause_phase interpretation map):
    EARLY vs MID, two-sided sign test. EARLY~MID AND PRE>MID -> late-pause-
    specific precursor (CSD reading survives). EARLY>MID same direction ->
    monotone pause-phase gradient (downgrade). EARLY<MID -> non-monotone flag.
  - AMPLITUDE-CONFOUND guard: AR1 (dimensionless envelope shape) reported
    separately from logVar (magnitude). VAR-rise WITHOUT AR1-rise = flagged
    amplitude confound, not a CSD verdict.
  - Claim scope: site-local (Kilauea summit), envelope-dynamics level.

Usage:
  ../../.venv-research/bin/python csd.py prereg
  nice -n 19 ../../.venv-research/bin/python csd.py fetch   # checkpointed
  nice -n 19 ../../.venv-research/bin/python csd.py run
"""
import hashlib, json, os, sys, urllib.request
from datetime import datetime, timedelta

import numpy as np
from scipy.signal import detrend

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from receipt_ledger import log_receipt
from volcano_fresh import EP_HST, STATIONS, WIN_PER_SEG, _utc
from harvest3_loaders import W
import pause_phase

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "csd_prereg.json")
CACHE = "<local-path>/signalmap/data/volcano/csd_env"
HOP = 512  # same envelope hop as volcano_precursor._fetch


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def plan():
    """Deterministic EARLY/MID/PRE times, reused from pause_phase (hash-checked)."""
    return pause_phase.plan()


def prereg():
    os.makedirs(FROZEN, exist_ok=True)
    p = plan()
    n_valid = sum(1 for x in p if x["early"] and x["mid"] and x["pre"])
    spec = {
        "prereg": "CSD (WEG 2, critical-slowing-down early-warning, directed, TRIANGULATION path 2)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slow_variable": "detrended rolling-RMS amplitude envelope of raw HHZ 30-min segment (W=1024, hop=512)",
        "statistics": "per segment: ar1=lag-1 autocorr of detrended envelope, logvar=log10(var of detrended envelope)",
        "primary_tests": "1-sided GREATER paired within episode: AR1(PRE)>AR1(MID) and logVar(PRE)>logVar(MID); each exact binomial sign test AND Wilcoxon signed-rank; PASS(signal)=sign_p<0.05 AND wilcoxon_p<0.05; UWE primary, RIMD replication",
        "control": "EARLY vs MID two-sided sign test; early~mid AND pre>mid -> late-pause-specific precursor; early>mid same-dir -> monotone gradient downgrade; early<mid -> non-monotone flag",
        "confound_guard": "AR1 (envelope shape, dimensionless) reported separately from logVar (magnitude); VAR-rise without AR1-rise = amplitude confound, not a CSD verdict",
        "orthogonality": "directed dynamical-systems statistic on the slow envelope, no learning/no selection; error mode independent of Weg 1 learned within-window texture and Weg 3 within-window spectrum",
        "reuse_note": "segment TIMES pinned by pause_phase.plan() (hash-checked, deterministic from committed EP_HST); fresh RAW fetch because znormed cache destroys variance signal",
        "claim_scope": "site-local (Kilauea summit), envelope-dynamics level",
        "episode_table_sha256": _sha_text(json.dumps(EP_HST)),
        "plan_sha256": _sha_text(json.dumps(p)),
        "n_valid_triples": n_valid,
    }
    with open(PREREG, "w") as f:
        json.dump({"spec": spec, "plan": p}, f, indent=1)
    tip = log_receipt("CSD-PREREG", spec)
    print(f"prereg written, n_valid_triples={n_valid}. ledger tip = {tip}", flush=True)


def _pre_check():
    assert os.path.exists(PREREG), "run prereg first (ledger before load!)"
    with open(PREREG) as f:
        pre = json.load(f)
    assert _sha_text(json.dumps(pre["plan"])) == pre["spec"]["plan_sha256"], "plan tampered"
    return pre["plan"]


def _fetch_env(sta, t0_iso):
    """Raw 30-min HHZ -> detrended rolling-RMS envelope. Cached (checkpointed)."""
    key = f"{sta}_{t0_iso.replace(':', '')}.npz"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        z = np.load(path)
        return z["env"] if z["env"].size else None
    t0 = datetime.fromisoformat(t0_iso)
    t1 = t0 + timedelta(minutes=30)
    u = (f"https://service.iris.edu/irisws/timeseries/1/query?net=HV&sta={sta}"
         f"&loc=--&cha=HHZ&starttime={t0.strftime('%Y-%m-%dT%H:%M:%S')}"
         f"&endtime={t1.strftime('%Y-%m-%dT%H:%M:%S')}&format=ascii1")
    try:
        d = urllib.request.urlopen(u, timeout=180).read().decode()
        x = np.array(d.strip().split("\n")[1:], float)
        assert len(x) >= (WIN_PER_SEG + 2) * W, f"only {len(x)} samples"
    except Exception as e:
        print(f"  {sta} {t0_iso}: FETCH-FAIL {e}", flush=True)
        np.savez_compressed(path, env=np.array([]))  # checkpoint the failure too
        return None
    env = np.array([x[i:i + W].std() for i in range(0, len(x) - W, HOP)])
    np.savez_compressed(path, env=env)
    return env


def fetch():
    p = _pre_check()
    os.makedirs(CACHE, exist_ok=True)
    todo = [(sta, x[k]) for sta in STATIONS for x in p
            for k in ("early", "mid", "pre") if x[k]]
    for i, (sta, t) in enumerate(todo):
        _fetch_env(sta, t)
        if (i + 1) % 25 == 0:
            print(f"[checkpoint] {i + 1}/{len(todo)} envelopes", flush=True)
    print(f"fetch complete: {len(todo)} envelopes (incl. cached/failed)", flush=True)


def _stats(env):
    """CSD statistics of the detrended slow envelope."""
    e = detrend(np.asarray(env, float))
    ar1 = float(np.corrcoef(e[:-1], e[1:])[0, 1]) if len(e) > 2 else float("nan")
    logvar = float(np.log10(e.var() + 1e-30))
    return ar1, logvar


def _paired(a, b, alt):
    """Paired PRE-vs-control test: exact sign test + Wilcoxon signed-rank."""
    from scipy.stats import binomtest, wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b) & (a != b)
    a, b = a[m], b[m]
    n = len(a)
    if n == 0:
        return {"n": 0, "concordant": "0/0", "sign_p": 1.0, "wilcoxon_p": 1.0}
    conc = int((a > b).sum())
    sign_alt = "greater" if alt == "greater" else "two-sided"
    sp = float(binomtest(conc, n, 0.5, alternative=sign_alt).pvalue)
    try:
        wp = float(wilcoxon(a, b, alternative=alt).pvalue)
    except ValueError:
        wp = 1.0
    return {"n": n, "concordant": f"{conc}/{n}", "sign_p": sp, "wilcoxon_p": wp}


def run():
    p = _pre_check()
    results = {}
    for sta in STATIONS:
        per = {"ar1": {"early": [], "mid": [], "pre": []},
               "logvar": {"early": [], "mid": [], "pre": []}}
        n_ep = 0
        for x in p:
            if not (x["early"] and x["mid"] and x["pre"]):
                continue
            envs = {k: _fetch_env(sta, x[k]) for k in ("early", "mid", "pre")}
            if any(v is None for v in envs.values()):
                continue
            n_ep += 1
            for k in ("early", "mid", "pre"):
                a, lv = _stats(envs[k])
                per["ar1"][k].append(a)
                per["logvar"][k].append(lv)
        role = "PRIMARY" if sta == "UWE" else "replication"
        r = {"role": role, "n_episodes": n_ep}
        for stat in ("ar1", "logvar"):
            pre_v, mid_v, early_v = per[stat]["pre"], per[stat]["mid"], per[stat]["early"]
            rise = _paired(pre_v, mid_v, "greater")          # PRE > MID (directed)
            ctrl = _paired(early_v, mid_v, "two-sided")       # EARLY vs MID (gradient)
            rise["PASS"] = bool(rise["sign_p"] < 0.05 and rise["wilcoxon_p"] < 0.05)
            r[stat] = {"pre_gt_mid": rise, "early_vs_mid": ctrl,
                       "median": {"early": float(np.nanmedian(early_v)) if early_v else None,
                                  "mid": float(np.nanmedian(mid_v)) if mid_v else None,
                                  "pre": float(np.nanmedian(pre_v)) if pre_v else None}}
        ar1_pass = r["ar1"]["pre_gt_mid"]["PASS"]
        var_pass = r["logvar"]["pre_gt_mid"]["PASS"]
        r["csd_verdict"] = ("CSD-CONFIRMED" if ar1_pass and var_pass else
                            "AR1-ONLY" if ar1_pass else
                            "VAR-ONLY-CONFOUND-FLAG" if var_pass else "NULL")
        results[f"CSD-{sta}"] = r
        print(f"CSD-{sta} [{role}] n_ep={n_ep}: "
              f"AR1 pre>mid {r['ar1']['pre_gt_mid']['concordant']} "
              f"sign_p={r['ar1']['pre_gt_mid']['sign_p']:.4f} "
              f"wil_p={r['ar1']['pre_gt_mid']['wilcoxon_p']:.4f} -> {'PASS' if ar1_pass else 'null'} | "
              f"VAR pre>mid {r['logvar']['pre_gt_mid']['concordant']} "
              f"sign_p={r['logvar']['pre_gt_mid']['sign_p']:.4f} "
              f"wil_p={r['logvar']['pre_gt_mid']['wilcoxon_p']:.4f} -> {'PASS' if var_pass else 'null'} "
              f"=> {r['csd_verdict']}", flush=True)
    tip = log_receipt("CSD", {"plan_sha256": _sha_text(json.dumps(p)), "results": results})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "fetch": fetch, "run": run,
     "plan": lambda: print(json.dumps(plan(), indent=1))}[sys.argv[1]]()
