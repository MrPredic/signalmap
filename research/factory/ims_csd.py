"""ZEIT-FAKTOR / CRITICAL SLOWING DOWN on IMS bearing RUN-TO-FAILURE.

Context: the Kilauea -12h precursor claim was honestly DOWNGRADED (TRIANGULATION.md):
Weg-2 CSD (csd.py) came out NULL there, and consilience diverged. But CSD (the
Scheffer/Dakos early-warning signature) is a THEORY: approaching a bifurcation the
slow state variable's lag-1 autocorrelation (AR1) AND variance MUST rise. The volcano
precursor was a speculative place to expect it. Mechanical run-to-failure is the
domain where CSD is textbook-EXPECTED. So this is the honest next test of the SAME
theory anchor on a family where it should appear -- the user's pre-decided fallback
(memory: "falls Vulkan-Timing scheitert -> Zeit bei anderen Familien: IMS-RUL, ...").

Directed, learning-free, selection-free -- error mode orthogonal to any learned
classifier. Reuses csd.py's ar1/logvar statistic and the prereg-freeze discipline;
the Delta is the run-to-failure trajectory loader + a rolling-window CSD trend
(Kendall's tau) with a Fourier-phase-randomized surrogate null (standard Dakos EWS).

DATA: IMS/NASA (Rexnord ZA-2115), 20kHz 1s snapshots every ~10min, run to failure.
  1st_test 2156 files/8ch (2 axes/bearing), 2nd_test 984/4ch, 4th_test(=documented
  test3) 6324/4ch. Ground-truth failed bearings (retro_loaders.IMS_FAILED, 0-idx):
  1st b2 (inner race), 1st b3 (roller), 2nd b0 (outer race), 4th/test3 b2 (outer).
  Healthy bearings of the SAME rigs = the specificity control (same run duration,
  same operating drift, did NOT fail).

PRE-REGISTERED DESIGN (ledger IMS-CSD-PREREG, frozen BEFORE any RMS loaded):
  - Slow variable HI[t] = RMS amplitude of the 1s snapshot for one bearing channel,
    over the chronological run (t = snapshot index). test1 uses axis x (col 2*b),
    pinned a priori to avoid axis selection; test2/3 single axis (col b).
  - Detrend HI with a Gaussian filter (sigma = 0.05*N) -> residual (CSD is about the
    FLUCTUATION dynamics, not the degradation trend in the mean).
  - Rolling window wl = round(0.5*N), step = round(0.005*N): per window compute
    ar1 = lag-1 autocorr and logvar = log10(var) of the residual -> two rolling series.
  - Directed statistic per bearing: Kendall tau of ar1 vs time and of logvar vs time.
    CSD predicts tau > 0 (rising toward failure).
  - Significance per bearing: Fourier-phase-randomized surrogate of the residual
    (preserves power spectrum, destroys the trend) x N_surr=500; 1-sided p =
    (1 + #surr tau >= obs) / (1 + N_surr) for ar1 and var separately.
  - PRIMARY verdict per FAILED bearing: PASS = tau_ar1>0 AND p_ar1<0.05 AND
    tau_var>0 AND p_var<0.05. Headline = CSD-positive rate among failed bearings.
  - SPECIFICITY control: healthy bearings should NOT be CSD-positive at the failed
    rate; Fisher exact on (CSD+ / CSD-) x (failed / healthy). Failed>>healthy =
    CSD tracks approach-to-failure, not generic rig aging.
  - CONFOUND guard: AR1-rise (dimensionless) reported separately from var-rise
    (magnitude); var-rise alone could be amplitude drift -> CONFIRMED needs BOTH.
  - Claim scope: bearing run-to-failure, fluctuation-dynamics level. n_failed small
    (4) -> per-bearing surrogate significance is the primary evidence, not a paired
    across-bearing test.

USAGE (order enforces prereg-before-readout):
  python ims_csd.py prereg   # freeze design + ledger (commit this BEFORE readout)
  python ims_csd.py fetch    # build cached per-snapshot RMS trajectories
  python ims_csd.py run      # CSD taus + surrogates + verdict + ledger
"""
import hashlib, json, os, sys
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau, fisher_exact

from receipt_ledger import log_receipt
from retro_loaders import _ims_files, IMS_FAILED

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
CACHE = os.path.join(HERE, "cache", "ims_csd")


def _prereg_path(ind):  # rms keeps the original committed filename
    return os.path.join(FROZEN, "ims_csd_prereg.json" if ind == "rms"
                        else f"ims_csd_{ind}_prereg.json")


def _result_path(ind):
    return os.path.join(HERE, "logs", "ims_csd_result.json" if ind == "rms"
                        else f"ims_csd_{ind}_result.json")

TESTS = ("1st_test", "2nd_test", "4th_test")  # 4th_test == documented test 3
# health indicators cached per snapshot; each is a separately-pre-registered CSD run.
# rms = broadband amplitude (rig-global); kurt = impulsivity = failure-specific bearing
# defect marker (orthogonal readout to disambiguate "CSD absent" from "rms too global").
INDICATORS = ("rms", "kurt")
PARAMS = {
    "health_indicator": "per-snapshot RMS amplitude",
    "axis_pin": "test1 uses col 2*b (axis x); test2/3 col b",
    "detrend_gaussian_sigma_frac": 0.05,
    "rolling_window_frac": 0.5,
    "rolling_step_frac": 0.005,
    "n_surrogate": 500,
    "alpha": 0.05,
    "surrogate": "fourier phase-randomization of residual",
    "seed": 20260708,
}


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _col(test, b):
    return 2 * b if test == "1st_test" else b


def _bearings():
    """Deterministic (test, b, col, label) list for all 4 bearings x 3 rigs."""
    out = []
    for test in TESTS:
        for b in range(4):
            lab = "failed" if (test, b) in IMS_FAILED else "healthy"
            out.append({"test": test, "b": b, "col": _col(test, b), "label": lab})
    return out


def _file_manifest():
    """Frozen provenance: per-test chronological basenames (can't change post hoc)."""
    return {t: [os.path.basename(f) for f in _ims_files(t)] for t in TESTS}


_HI_DESC = {"rms": "per-snapshot RMS amplitude (broadband, rig-global)",
            "kurt": "per-snapshot kurtosis (impulsivity, failure-specific bearing-defect marker)"}


def prereg(ind="rms"):
    assert ind in INDICATORS
    os.makedirs(FROZEN, exist_ok=True)
    bearings = _bearings()
    manifest = {t: len(v) for t, v in _file_manifest().items()}
    spec = {
        "prereg": f"IMS-CSD-{ind.upper()} (critical-slowing-down early-warning on bearing run-to-failure, directed, theory-anchor)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "domain": "IMS/NASA Rexnord ZA-2115, 20kHz 1s snapshots ~every 10min, run to failure",
        "indicator": ind,
        "slow_variable": f"HI[t] = {_HI_DESC[ind]} of one bearing channel over the chronological run",
        "params": PARAMS,
        "statistic": "Kendall tau of rolling ar1(t) and rolling logvar(t) of the Gaussian-detrended HI residual",
        "significance": "per bearing, Fourier-phase-randomized surrogate of residual x N_surr; 1-sided p=(1+#surr>=obs)/(1+N)",
        "primary": "per FAILED bearing PASS = tau_ar1>0 & p_ar1<alpha & tau_var>0 & p_var<alpha; headline = CSD+ rate among failed",
        "specificity": "Fisher exact (CSD+/CSD-) x (failed/healthy); failed>>healthy = tracks approach-to-failure not rig aging",
        "confound_guard": "ar1-rise (dimensionless) reported separately from var-rise (magnitude); CONFIRMED needs BOTH",
        "orthogonality": "directed dynamical-systems statistic, no learning/no selection; error mode independent of learned classifiers",
        "claim_scope": "bearing run-to-failure, fluctuation-dynamics level; n_failed=4 -> per-bearing surrogate p is primary",
        "failed_ground_truth": sorted([f"{t}:b{b}" for (t, b) in IMS_FAILED if t in TESTS]),
        "bearings_sha256": _sha_text(json.dumps(bearings)),
        "manifest_sha256": _sha_text(json.dumps(_file_manifest())),
        "n_files_per_test": manifest,
    }
    with open(_prereg_path(ind), "w") as f:
        json.dump({"spec": spec, "bearings": bearings}, f, indent=1)
    tip = log_receipt(f"IMS-CSD-{ind.upper()}-PREREG", spec)
    print(f"prereg[{ind}] written. failed={spec['failed_ground_truth']} files={manifest}")
    print(f"ledger tip = {tip}")


def _pre_check(ind="rms"):
    assert os.path.exists(_prereg_path(ind)), "run prereg first (ledger before load!)"
    with open(_prereg_path(ind)) as f:
        pre = json.load(f)
    assert _sha_text(json.dumps(pre["bearings"])) == pre["spec"]["bearings_sha256"], "bearings tampered"
    assert _sha_text(json.dumps(_file_manifest())) == pre["spec"]["manifest_sha256"], "file manifest changed since freeze"
    return pre


def _snapshot_indicators(x):
    """Per-channel (rms, kurt) of a 1s snapshot array [samples, channels]."""
    m = x.mean(axis=0)
    d = x - m
    var = (d ** 2).mean(axis=0)
    rms = np.sqrt((x ** 2).mean(axis=0))
    kurt = (d ** 4).mean(axis=0) / (var ** 2 + 1e-30)
    return rms, kurt


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    for test in TESTS:
        out = os.path.join(CACHE, f"{test}_hi.npz")
        if os.path.exists(out):
            print(f"{test}: cached")
            continue
        files = _ims_files(test)
        nc = 8 if test == "1st_test" else 4
        rms = np.empty((len(files), nc)); kurt = np.empty((len(files), nc))
        for i, f in enumerate(files):
            rms[i], kurt[i] = _snapshot_indicators(np.loadtxt(f))
            if (i + 1) % 500 == 0:
                print(f"  {test} {i + 1}/{len(files)}", flush=True)
        np.savez_compressed(out, rms=rms, kurt=kurt)
        print(f"{test}: wrote {rms.shape} rms+kurt trajectory", flush=True)
    print("fetch complete")


def _rolling_taus(res):
    """Kendall tau of rolling ar1 and rolling logvar of a (detrended) residual."""
    N = len(res)
    wl = max(20, int(round(PARAMS["rolling_window_frac"] * N)))
    step = max(1, int(round(PARAMS["rolling_step_frac"] * N)))
    idx = list(range(0, N - wl + 1, step))
    ar1 = np.empty(len(idx))
    lv = np.empty(len(idx))
    for j, i in enumerate(idx):
        w = res[i:i + wl]
        w = w - w.mean()
        d = w[:-1]
        ar1[j] = np.corrcoef(d, w[1:])[0, 1] if d.std() > 0 else 0.0
        lv[j] = np.log10(w.var() + 1e-30)
    tt = np.arange(len(idx))
    return float(kendalltau(tt, ar1)[0]), float(kendalltau(tt, lv)[0])


def _phase_surrogate(res, rng):
    F = np.fft.rfft(res)
    ph = rng.uniform(0, 2 * np.pi, F.shape[0])
    ph[0] = 0.0
    if len(res) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(res))


def _csd_bearing(hi):
    sigma = max(2.0, PARAMS["detrend_gaussian_sigma_frac"] * len(hi))
    res = hi - gaussian_filter1d(hi, sigma=sigma)
    t_ar1, t_var = _rolling_taus(res)
    rng = np.random.default_rng(PARAMS["seed"])
    ns = PARAMS["n_surrogate"]
    ge_ar1 = ge_var = 0
    for _ in range(ns):
        s = _phase_surrogate(res, rng)
        s_ar1, s_var = _rolling_taus(s)
        ge_ar1 += s_ar1 >= t_ar1
        ge_var += s_var >= t_var
    p_ar1 = (1 + ge_ar1) / (1 + ns)
    p_var = (1 + ge_var) / (1 + ns)
    return t_ar1, t_var, float(p_ar1), float(p_var)


def run(ind="rms"):
    pre = _pre_check(ind)
    a = PARAMS["alpha"]
    rows = []
    for be in pre["bearings"]:
        hi = np.load(os.path.join(CACHE, f"{be['test']}_hi.npz"))[ind][:, be["col"]]
        t_ar1, t_var, p_ar1, p_var = _csd_bearing(hi)
        ar1_ok = t_ar1 > 0 and p_ar1 < a
        var_ok = t_var > 0 and p_var < a
        row = {**{k: be[k] for k in ("test", "b", "label")},
               "n": int(len(hi)), "tau_ar1": round(t_ar1, 3), "p_ar1": round(p_ar1, 4),
               "tau_var": round(t_var, 3), "p_var": round(p_var, 4),
               "ar1_rise": ar1_ok, "var_rise": var_ok, "csd_pass": bool(ar1_ok and var_ok)}
        rows.append(row)
        print(f"{be['test']:9s} b{be['b']} {be['label']:7s} N={row['n']:5d} "
              f"tau_ar1={t_ar1:+.3f}(p={p_ar1:.3f}) tau_var={t_var:+.3f}(p={p_var:.3f}) "
              f"{'CSD+' if row['csd_pass'] else '.'}", flush=True)

    failed = [r for r in rows if r["label"] == "failed"]
    healthy = [r for r in rows if r["label"] == "healthy"]
    fp = sum(r["csd_pass"] for r in failed)
    hp = sum(r["csd_pass"] for r in healthy)
    odds, fisher_p = fisher_exact([[fp, len(failed) - fp], [hp, len(healthy) - hp]],
                                  alternative="greater")
    verdict = ("CSD-CONFIRMED" if fp >= 3 and fisher_p < 0.05 else
               "CSD-PARTIAL" if fp >= 2 else "CSD-NULL")
    summary = {
        "csd_pass_failed": f"{fp}/{len(failed)}",
        "csd_pass_healthy": f"{hp}/{len(healthy)}",
        "fisher_p_failed_gt_healthy": round(float(fisher_p), 4),
        "ar1_rise_failed": f"{sum(r['ar1_rise'] for r in failed)}/{len(failed)}",
        "var_rise_failed": f"{sum(r['var_rise'] for r in failed)}/{len(failed)}",
        "verdict": verdict,
        "indicator": ind,
        "note": "n_failed=4 caps power; per-bearing surrogate p is the primary evidence",
    }
    print(f"\n== SUMMARY [{ind}] ==")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    rp = _result_path(ind)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    tip = log_receipt(f"IMS-CSD-{ind.upper()}-RESULT", summary)
    print(f"ledger tip = {tip}")


if __name__ == "__main__":
    fn, ind = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "rms")
    {"prereg": lambda: prereg(ind), "fetch": fetch, "run": lambda: run(ind)}[fn]()
