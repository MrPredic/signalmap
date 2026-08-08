"""WS3 Phase 1 (CONFIRMATORY) — isolate the MOX drift-invariant fingerprint.

The #11 forge selected, as top-1 in ALL 39/39 leave-one-day-out folds, the single
cross-sensor program  shape:logratio(32,37):specflat(diff(env(x))).  Four more
cross-sensor logratio/chdiff programs were selected in all 39 folds too.

This readout tests those programs IN ISOLATION with the features FIXED A PRIORI
(no per-fold F-stat selection) -> a strictly leak-free, stronger test than the
forge itself: if a fixed handful of named cross-sensor ratios still clears chance
under LODO, the fingerprint is a real, interpretable, drift-invariant property.

Discipline: `freeze` appends the prereg to the ledger + git BEFORE `run`.
Reuses mox_drift.{load_mox_mc, combiners, _znorm, _rf, _ci, _sha, log_receipt}
and feature_forge.programs UNCHANGED.

    python mox_fingerprint_isolated.py freeze     # prereg -> ledger tip (commit!)
    python mox_fingerprint_isolated.py run         # readout -> ledger receipt
"""
import sys
import numpy as np
import mox_drift as mx
from feature_forge import programs
from receipt_ledger import log_receipt

# forge winners: full "norm:combiner:program" names, each selected in ALL 39/39
# LODO folds (from mox_fingerprint.py extraction). Note mixed norm modes and
# distinct unary programs; all combiners are CROSS-sensor logratio/chdiff.
TOP1 = "shape:logratio(32,37):specflat(diff(env(x)))"
STABLE5 = [
    "shape:logratio(32,37):specflat(diff(env(x)))",
    "level:chdiff(16,57):crest(env(env(x)))",
    "shape:logratio(32,50):speccent(sq(sq(x)))",
    "shape:chdiff(32,34):crest(rollstd(rank(x)))",
    "shape:logratio(18,46):std(sq(diff2(x)))",
]

PREREG = {
    "physics": "#11 MOX drift-invariance -> NAMED fingerprint isolation (WS3 Phase 1)",
    "dataset": "Zenodo 10.5281/zenodo.15681119; data/mox/raw Day 1..39 (12 months)",
    "claim": ("A FIXED, a-priori set of cross-sensor log-ratio programs — top-1 in "
              "all 39/39 LODO folds of the frozen #11 forge — discriminates the 3 "
              "analytes drift-invariantly. PRIMARY = the single top-1 program "
              "shape:logratio(32,37):specflat(diff(env(x))). SECONDARY = the 5 "
              "all-fold-stable cross-sensor programs together."),
    "features": ("FIXED A PRIORI (no per-fold selection) -> zero selection leak. "
                 "combiner logratio(i,j) on the (62-ch) window -> shape-norm -> "
                 "unary program; 62 channels are 31 duplicate sensor-pairs "
                 "(adjacent r=0.9997), all winning pairs are CROSS-sensor."),
    "split": "leave-one-DAY-out (39 groups); StandardScaler+RF(150,rs=0); no reselection",
    "metric": "LODO accuracy vs chance 0.333; bootstrap CI(10k, seed0)",
    "controls": "label-shuffle-within-day NULL (expect ~chance); per-analyte feature means",
    "verdict_rule": ("DISCOVERY-CONFIRMED iff PRIMARY CI-lo > 0.333 AND "
                     "shuffle-null ~ chance; SECONDARY reported alongside"),
    "novelty": ("dataset authors document sensor DRIFT; the drift-INVARIANCE of "
                "these named cross-sensor ratios is undocumented"),
    "no_reselection": True,
}


def _prog_by_name(name):
    for p in programs():
        if p[0] == name:
            return p
    raise KeyError(name)


def _comb_by_name(name):
    for cn, cf in mx.combiners():
        if cn == name:
            return cf
    raise KeyError(name)


def _feat(raw, names):
    """(N, len(names)) matrix; each col = one fixed forge program, reproducing
    build_features' norm handling EXACTLY: shape -> _znorm, level -> raw signal."""
    from feature_forge import run_prog
    cols = []
    for full in names:
        nm, cn, pn = full.split(":", 2)
        cols.append((nm, _comb_by_name(cn), _prog_by_name(pn)))
    F = np.empty((len(raw), len(names)))
    for w, (X, _, _) in enumerate(raw):
        for k, (nm, cf, prog) in enumerate(cols):
            sig = cf(np.asarray(X, float))
            sig = mx._znorm(sig) if nm == "shape" else np.ascontiguousarray(sig, float)
            F[w, k] = run_prog(prog, sig)
    return np.nan_to_num(F)


def _lodo(F, y, g):
    folds = {}
    for hold in np.unique(g):
        tr = g != hold
        if len(set(y[tr])) < 2 or (~tr).sum() == 0:
            continue
        clf = mx._rf(); clf.fit(F[tr], y[tr])
        folds[hold] = float((clf.predict(F[~tr]) == y[~tr]).mean())
    fa = list(folds.values())
    lo, hi = mx._ci(fa)
    return float(np.mean(fa)), (lo, hi), len(fa)


def run():
    raw = mx.load_mox_mc()
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    chance = 1 / len(set(y))
    print(f"bank {len(raw)} windows, {len(set(g))} days, chance {chance:.3f}", flush=True)

    F1 = _feat(raw, [TOP1])
    acc1, ci1, n1 = _lodo(F1, y, g)
    F5 = _feat(raw, STABLE5)
    acc5, ci5, n5 = _lodo(F5, y, g)

    # shuffle-within-day NULL on the PRIMARY feature
    rng = np.random.default_rng(0); ysh = y.copy()
    for d in np.unique(g):
        idx = np.where(g == d)[0]; ysh[idx] = rng.permutation(y[idx])
    accn, cin, _ = _lodo(F1, ysh, g)

    # interpretability: per-analyte mean of the PRIMARY scalar feature
    means = {a: float(F1[y == a, 0].mean()) for a in sorted(set(y))}
    stds = {a: float(F1[y == a, 0].std()) for a in sorted(set(y))}

    print(f"\nPRIMARY  logratio(32,37):specflat(diff(env(x)))  "
          f"LODO {acc1:.3f} CI[{ci1[0]:.3f},{ci1[1]:.3f}]  ({n1} folds)", flush=True)
    print(f"SECONDARY 5 stable cross-sensor progs           "
          f"LODO {acc5:.3f} CI[{ci5[0]:.3f},{ci5[1]:.3f}]", flush=True)
    print(f"SHUFFLE-NULL (primary)                          "
          f"LODO {accn:.3f} CI[{cin[0]:.3f},{cin[1]:.3f}]  (expect ~{chance:.3f})", flush=True)
    print(f"per-analyte primary-feature mean+-sd: "
          f"{ {a: (round(means[a],3), round(stds[a],3)) for a in means} }", flush=True)
    verdict = ("DISCOVERY-CONFIRMED" if ci1[0] > chance and cin[1] >= chance - 0.02
               and accn < ci1[0] else "NOT CONFIRMED")
    print(f"VERDICT: {verdict}", flush=True)

    log_receipt("MOX-FINGERPRINT", {
        "primary": {"acc": acc1, "ci": list(ci1), "n_folds": n1,
                    "prog": "shape:logratio(32,37):specflat(diff(env(x)))"},
        "secondary5": {"acc": acc5, "ci": list(ci5)},
        "shuffle_null": {"acc": accn, "ci": list(cin)},
        "per_analyte_mean": means, "per_analyte_sd": stds,
        "chance": chance, "verdict": verdict,
        "spec_sha256": mx._sha(PREREG)})


def freeze():
    tip = log_receipt("MOX-FINGERPRINT-PREREG", {**PREREG, "spec_sha256": mx._sha(PREREG)})
    print("FROZEN. ledger tip:", tip)
    print("spec_sha256:", mx._sha(PREREG))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    (freeze if cmd == "freeze" else run)()
