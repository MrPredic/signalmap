"""CONSILIENCE LAYER — the actual novelty of the triangulation.

Four paths onto the -12h Kilauea precursor produce, per episode, a signed
PRE-minus-MID score. The claim is NOT "each path has good mean accuracy" but:

  (a) AGREEMENT: do INDEPENDENT paths flag the SAME episodes at -12h?
      -> rank correlation of per-episode scores between path pairs.
  (b) CONVERGENCE-UNDER-REAL vs SPREAD-UNDER-SHUFFLE: permuting the episode
      correspondence MUST collapse the agreement. Agreement that survives the
      shuffle is a shared artifact, not a shared signal.

Decision (registered): independent path-pairs whose agreement is significant
under REAL episode-alignment AND collapses under the permutation null ->
CONSILIENCE upgrade (first evidence-chain point). Divergence -> diagnose +
HONEST downgrade (a valid result; [[feedback-multi-method-triangulation]]).

Per-episode signed scores (all PRE - MID, same pause, same clock; UWE primary,
RIMD replication), reusing each path's own frozen machinery, no re-selection:
  weg1_texture : LOGO(episode) RF(lean duo) pre-probability(PRE) - (MID)
  weg2_ar1     : AR1(PRE)    - AR1(MID)    of detrended RMS envelope
  weg2_var     : logVar(PRE) - logVar(MID) of detrended RMS envelope
  weg3_centroid: centroid(PRE) - centroid(MID)     (spectral shape)
  weg3_specent : spec_ent(PRE) - spec_ent(MID)
  weg3_hf      : hf_ratio(PRE) - hf_ratio(MID)

Independence for the headline test: weg1_texture (learned, znormed within-
window) vs weg2_ar1/weg2_var (directed, raw envelope dynamics) are the most
orthogonal pair; weg3 shares representation with weg1 (registered in spectral.py)
so weg1<->weg3 agreement is reported but NOT counted as independent corroboration.

PRE-REGISTERED (ledger CONSILIENCE-PREREG, BEFORE any score computed):
  - Metric = Spearman rho of per-episode scores over the COMMON valid episode
    set (both members finite). Null = permute one vector's episode order,
    n_perm=10000, two-sided p = share of |rho_perm| >= |rho_obs|.
  - Headline independent pair = weg1_texture vs weg2_ar1 (and vs weg2_var).
  - PASS(pair) = perm two-sided p < 0.05 (agreement beyond episode-shuffle).
  - CONSILIENCE verdict = at least one INDEPENDENT (weg1<->weg2) pair PASS,
    same sign on UWE and RIMD. Else honest divergence readout.

Usage:
  ../../.venv-research/bin/python consilience.py prereg
  ../../.venv-research/bin/python consilience.py run   # needs csd_env cache ready
"""
import hashlib, itertools, json, os, sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from receipt_ledger import log_receipt
from volcano_fresh import STATIONS
from feature_forge import lean_baseline
from volcano_precursor import _bank, _pre_check as _prec_pre_check
import csd, spectral

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "consilience_prereg.json")

INDEP_PAIRS = [("weg1_texture", "weg2_ar1"), ("weg1_texture", "weg2_var")]


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prereg():
    os.makedirs(FROZEN, exist_ok=True)
    p = _prec_pre_check()
    spec = {
        "prereg": "CONSILIENCE (per-episode agreement + shuffle-null across triangulation paths)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scores": "per episode PRE-MID: weg1_texture=LOGO RF pre-prob delta; weg2_ar1/weg2_var=envelope AR1/logVar delta; weg3_centroid/specent/hf=spectral-shape delta",
        "metric": "Spearman rho of per-episode scores over common valid episodes; permutation null shuffles one vector's episode order (n_perm=10000), two-sided p = share |rho_perm|>=|rho_obs|",
        "independent_pairs": INDEP_PAIRS,
        "weg3_caveat": "weg3 shares representation with weg1 -> weg1<->weg3 reported, NOT counted as independent corroboration",
        "pass_rule": "PASS(pair)=perm p<0.05; CONSILIENCE = >=1 independent (weg1<->weg2) pair PASS with same sign on UWE and RIMD; else honest divergence readout",
        "prec_plan_sha256": _sha_text(json.dumps(p)),
    }
    with open(PREREG, "w") as f:
        json.dump({"spec": spec}, f, indent=1)
    tip = log_receipt("CONSILIENCE-PREREG", spec)
    print(f"prereg written. ledger tip = {tip}", flush=True)


def _pre_check():
    assert os.path.exists(PREREG), "run prereg first (ledger before readout!)"
    with open(PREREG) as f:
        spec = json.load(f)["spec"]
    p = _prec_pre_check()
    assert _sha_text(json.dumps(p)) == spec["prec_plan_sha256"], "precursor plan tampered"
    return p


def _weg1_scores(sta, p):
    """Per-episode LOGO RF pre-probability(PRE) - pre-probability(MID)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    raw = _bank(sta, 12, p)
    L = lean_baseline(raw)
    y = np.array([r[1] for r in raw])
    g = np.array([r[2] for r in raw])
    out = {}
    for ep in np.unique(g):
        tr, te = g != ep, g == ep
        if not (set(y[te]) >= {"pre", "mid"}):
            continue
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(L[tr], y[tr])
        pre_col = list(clf.classes_).index("pre")
        proba = clf.predict_proba(L[te])[:, pre_col]
        yt = y[te]
        out[int(ep)] = float(proba[yt == "pre"].mean() - proba[yt == "mid"].mean())
    return out


def _weg2_scores(sta, plan_csd):
    """Per-episode envelope AR1 and logVar deltas PRE - MID."""
    ar1, var = {}, {}
    for x in plan_csd:
        if not (x["mid"] and x["pre"]):
            continue
        ep = pre = mid = None
        pe, me = csd._fetch_env(sta, x["pre"]), csd._fetch_env(sta, x["mid"])
        if pe is None or me is None:
            continue
        a_p, v_p = csd._stats(pe)
        a_m, v_m = csd._stats(me)
        ar1[int(x["ep"])] = a_p - a_m
        var[int(x["ep"])] = v_p - v_m
    return ar1, var


def _weg3_scores(sta, p):
    """Per-episode spectral-shape descriptor deltas PRE - MID."""
    raw = _bank(sta, 12, p)
    by = {}
    for w, lab, ep in raw:
        by.setdefault(int(ep), {"pre": [], "mid": []})[lab].append(spectral._descriptors(w))
    out = {d: {} for d in spectral.DESCS}
    for ep, segs in by.items():
        if not segs["pre"] or not segs["mid"]:
            continue
        dp = np.mean(segs["pre"], axis=0) - np.mean(segs["mid"], axis=0)
        for j, d in enumerate(spectral.DESCS):
            out[d][ep] = float(dp[j])
    return out


def _spearman_perm(sa, sb, n_perm=10000, seed=0):
    from scipy.stats import spearmanr
    eps = sorted(set(sa) & set(sb))
    a = np.array([sa[e] for e in eps])
    b = np.array([sb[e] for e in eps])
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 5:
        return {"n": n, "rho": None, "perm_p": 1.0}
    rho = float(spearmanr(a, b).statistic)
    rng = np.random.default_rng(seed)
    hits = sum(abs(float(spearmanr(a, rng.permutation(b)).statistic)) >= abs(rho) - 1e-12
               for _ in range(n_perm))
    return {"n": n, "rho": rho, "perm_p": (hits + 1) / (n_perm + 1)}


def run():
    p = _pre_check()
    plan_csd = csd.plan()
    results, sign_by_pair = {}, {}
    for sta in STATIONS:
        S = {"weg1_texture": _weg1_scores(sta, p)}
        ar1, var = _weg2_scores(sta, plan_csd)
        S["weg2_ar1"], S["weg2_var"] = ar1, var
        w3 = _weg3_scores(sta, p)
        S["weg3_centroid"], S["weg3_specent"], S["weg3_hf"] = \
            w3["centroid"], w3["spec_ent"], w3["hf_ratio"]
        pairs = {}
        for a, b in itertools.combinations(S, 2):
            r = _spearman_perm(S[a], S[b])
            key = f"{a}<->{b}"
            pairs[key] = r
            if (a, b) in INDEP_PAIRS or (b, a) in INDEP_PAIRS:
                sign_by_pair.setdefault(key, {})[sta] = (
                    None if r["rho"] is None else np.sign(r["rho"]), r["perm_p"])
        results[sta] = {"n_scores": {k: len(v) for k, v in S.items()}, "pairs": pairs}
        print(f"== {sta} ({'PRIMARY' if sta=='UWE' else 'replication'}) ==", flush=True)
        for a, b in INDEP_PAIRS:
            k = f"{a}<->{b}"
            r = pairs[k]
            print(f"  [INDEP] {k}: n={r['n']} rho={r['rho']} perm_p={r['perm_p']:.4f}"
                  f"{'  *PASS' if r['perm_p']<0.05 else ''}", flush=True)
        for k, r in pairs.items():
            if "weg1" in k and "weg3" in k:
                print(f"  [shared-rep] {k}: rho={r['rho']} perm_p={r['perm_p']:.4f}", flush=True)
    # verdict: >=1 independent pair PASS with same sign on both stations
    consilient = []
    for key, byst in sign_by_pair.items():
        u, ri = byst.get("UWE"), byst.get("RIMD")
        if u and ri and u[1] < 0.05 and ri[1] < 0.05 and u[0] == ri[0] and u[0] is not None:
            consilient.append(key)
    verdict = "CONSILIENCE" if consilient else "DIVERGENCE (honest downgrade)"
    results["verdict"] = {"consilient_independent_pairs": consilient, "verdict": verdict}
    print(f"\n=> {verdict}  {consilient}", flush=True)
    tip = log_receipt("CONSILIENCE", {"prec_plan_sha256": _sha_text(json.dumps(p)),
                                      "results": results})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "run": run}[sys.argv[1]]()
