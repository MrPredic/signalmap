"""Physics #11 — CHEMICAL-SENSING DRIFT-INVARIANCE (MOX e-nose, 12 months).

Independent-physics generalization test (Prio 2b(c)). Dataset =
Zenodo 10.5281/zenodo.15681119 (Sci Data Oct 2025): 62 SnO2-nanowire sensors,
~700 exposures across 39 sessions ("Day N") spanning 12 months, 3 analytes
(Diacetyl / EtOH / Phenylethanol) at several concentrations. The authors
DOCUMENT sensor drift; they do NOT test whether a channel-pattern signature for
analyte identity SURVIVES the drift. That untested invariance is our question.

Why this is the right generalization probe:
  - New physics (chemical selectivity) with a genuine, previously-unseen holdout
    (data/mox, SHA-frozen, never inspected for label-separability before freeze).
  - Reuses the VALIDATED multichannel grammar (gas_multichannel.py): the e-nose
    cross-channel logratio/chdiff combiner that turned the GAS single-channel
    double-null into a CI-fest LODO win. Same machinery, new substrate = a fair
    reuse-over-rebuild test, not a bespoke hack.
  - The split IS the invariance test: leave-one-DAY-out (LODO over 39 time
    groups). A held-out day is a future/past time point the model never saw ->
    clearing chance there = a drift-invariant analyte signature.

Discipline: FROZEN_SPEC below is committed to the hash-chained ledger (freeze
mode) and git BEFORE any readout run. No re-selection after readout; a single
run; concentration/replicate are nuisance (pooled into the analyte label), never
a target. An honest NULL is a real result (generalization boundary), not a fail.

  freeze : append FROZEN_SPEC (+sha) to LEDGER, print tip (do this + commit first)
  run    : build bank, LODO readout, receipts, ledger the result
"""
import glob, hashlib, os, re, sys, time
import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import programs, run_prog, lean_baseline
from receipt_ledger import log_receipt

MOX = "<local-path>/signalmap/data/mox/raw"
CACHE = "<local-path>/signalmap/research/factory/cache/mox_mc.npz"
W, C_GATE = 256, 50           # W = resampled response length; gate as validated
N_CH = 62
ANALYTES = ("Diacetyl", "EtOH", "Phenylethanol")

FROZEN_SPEC = {
    "physics": "#11 chemical-sensing drift-invariance (MOX e-nose)",
    "dataset": "Zenodo 10.5281/zenodo.15681119; local data/mox/raw/Day 1..39",
    "question": ("Does a drift-invariant cross-channel selectivity signature "
                 "discriminate the 3 analytes across 12 months, i.e. survive "
                 "leave-one-DAY-out over 39 time groups? Authors document drift; "
                 "invariance untested -> discovery."),
    "window": ("per exposure CSV parse R1..R62[Ohm], (62,T) -> linear-resample "
               "to (62,W=256); label = filename analyte prefix; concentration & "
               "replicate = nuisance (pooled); group = Day chronological ordinal"),
    "grammar": ("reuse multichannel: combiners ch_i(62)|logratio(i,j)|chdiff(i,j)"
                "|mean62|std62 x norm{shape,level} x unary grammar v2.1 dedup; "
                "sampled to capacity gate C=50 x n_days"),
    "split_primary": ("leave-one-DAY-out (LODO, 39 mixed-label time groups); "
                      "per-fold F-stat prescreen top-5; RF(150); leak-free"),
    "baseline": "single-channel lean_baseline (perm-entropy + psd-slope, R1 z-norm)",
    "primary_metric": ("MC-forge LODO accuracy vs chance 0.333; champion = "
                       "chance-gated paired mc-lean CI"),
    "secondary": "stability top1-freq; label-shuffle NULL control (expect ~chance)",
    "verdict_rule": ("DISCOVERY iff MC-forge CI-lo > 0.333 AND champion != NULL "
                     "AND stability top1-freq >= 0.5 AND shuffle-null ~ chance; "
                     "else honest NULL (generalization boundary)"),
    "no_reselection": True,
}


def _znorm(seg):
    s = detrend(np.ascontiguousarray(seg, float))
    return (s - s.mean()) / (s.std() + 1e-12)


def _day_ord(name):
    m = re.search(r"Day\s*(\d+)", name)
    return int(m.group(1)) if m else 10**6


def load_mox_mc(rebuild=False):
    """(C=62, W) resampled resistance windows. group = Day ordinal, mixed labels.
    Cached to npz (700 CSV parse is slow)."""
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        return [(z["X"][i], str(z["y"][i]), int(z["g"][i]))
                for i in range(len(z["y"]))]
    days = sorted(glob.glob(f"{MOX}/Day*"), key=_day_ord)
    Xs, ys, gs = [], [], []
    for gid, d in enumerate(days):
        for fp in sorted(glob.glob(f"{d}/*.csv")):
            base = os.path.basename(fp)
            analyte = next((a for a in ANALYTES if base.startswith(a)), None)
            if analyte is None:
                continue
            try:
                df = pd.read_csv(fp, sep=r"\s+", engine="c")
            except Exception:
                df = pd.read_csv(fp, sep=r"\s+", engine="python")
            cols = [c for c in df.columns if re.fullmatch(r"R\d+\[Ohm\]", str(c))]
            if len(cols) != N_CH:
                continue
            m = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)  # (T,62)
            m = m[~np.isnan(m).any(1)]
            if len(m) < 8:
                continue
            t_old = np.linspace(0, 1, len(m))
            t_new = np.linspace(0, 1, W)
            win = np.stack([np.interp(t_new, t_old, m[:, c]) for c in range(N_CH)])  # (62,W)
            Xs.append(win.astype(np.float32)); ys.append(analyte); gs.append(gid)
    X = np.array(Xs); y = np.array(ys); g = np.array(gs)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, g=g)
    return [(X[i], str(y[i]), int(g[i])) for i in range(len(y))]


def combiners():
    cs = [(f"ch{i}", lambda X, i=i: X[i]) for i in range(N_CH)]
    for i in range(N_CH):
        for j in range(i + 1, N_CH):
            cs.append((f"logratio({i},{j})", lambda X, i=i, j=j:
                       np.log(np.abs(X[i]) + 1e-9) - np.log(np.abs(X[j]) + 1e-9)))
            cs.append((f"chdiff({i},{j})", lambda X, i=i, j=j: X[i] - X[j]))
    cs.append((f"mean{N_CH}", lambda X: X.mean(0)))
    cs.append((f"std{N_CH}", lambda X: X.std(0)))
    return cs


def build_features(raw, n_grp, seed=0):
    progs = programs(); cs = combiners()
    space = [(ci, pi, nm) for ci in range(len(cs))
             for pi in range(len(progs)) for nm in ("shape", "level")]
    budget = min(len(space), C_GATE * n_grp)
    rng = np.random.default_rng(seed)
    picks = [space[k] for k in rng.choice(len(space), budget, replace=False)]
    names = [f"{nm}:{cs[ci][0]}:{progs[pi][0]}" for ci, pi, nm in picks]
    t0 = time.time()
    F = np.empty((len(raw), budget))
    for w, (X, _, _) in enumerate(raw):
        cache = {}
        for k, (ci, pi, nm) in enumerate(picks):
            key = (ci, nm)
            if key not in cache:
                sig = cs[ci][1](X)
                cache[key] = _znorm(sig) if nm == "shape" else np.ascontiguousarray(sig, float)
            F[w, k] = run_prog(progs[pi], cache[key])
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"features: {budget}/{len(space)} sampled (gate C={C_GATE}x{n_grp}) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return F, names


def _ci(a, seed=0):
    a = np.array(a, float); rng = np.random.default_rng(seed)
    m = rng.choice(a, (10000, len(a)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _rf():
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(150, random_state=0, n_jobs=-1))


def _fstat(F, y):
    from sklearn.feature_selection import f_classif
    s, _ = f_classif(F, y)
    return np.nan_to_num(s, nan=0.0)


def lodo(raw, F, names, lean_F, y, g, tag, k=5):
    days = np.unique(g); chance = 1 / len(set(y))
    folds, lean_folds, sel_log = {}, {}, {}
    for hold in days:
        tr = g != hold
        if len(set(y[tr])) < 2 or (~tr).sum() == 0:
            continue
        sel = list(np.argsort(-_fstat(F[tr], y[tr]))[:k])
        clf = _rf(); clf.fit(F[tr][:, sel], y[tr])
        folds[hold] = float((clf.predict(F[~tr][:, sel]) == y[~tr]).mean())
        sel_log[hold] = [names[j] for j in sel]
        clf = _rf(); clf.fit(lean_F[tr], y[tr])
        lean_folds[hold] = float((clf.predict(lean_F[~tr]) == y[~tr]).mean())
    fa = list(folds.values()); la = list(lean_folds.values())
    lo, hi = _ci(fa)
    d = np.array([folds[r] - lean_folds[r] for r in folds])
    rng = np.random.default_rng(0)
    dm = rng.choice(d, (10000, len(d)), replace=True).mean(1)
    dlo, dhi = float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))
    tops = [s[0] for s in sel_log.values() if s]
    t1 = max(tops.count(t) for t in set(tops)) / len(sel_log) if tops else 0.0
    if lo <= chance and _ci(la)[0] <= chance:
        champ = "NULL (nobody clears chance)"
    elif dlo > 0: champ = "mc-forge"
    elif dhi < 0: champ = "lean"
    else: champ = "tie->lean"
    print(f"\n===== {tag} (LODO, {len(folds)} days, chance {chance:.3f}) =====", flush=True)
    print(f"MC-FORGE: {np.mean(fa):.3f}  CI [{lo:.3f},{hi:.3f}]   "
          f"lean(single-ch): {np.mean(la):.3f}  CI {_ci(la)}", flush=True)
    print(f"PAIRED mc-lean: {d.mean():+.3f}  CI [{dlo:+.3f},{dhi:+.3f}]", flush=True)
    print(f"STABILITY top1-freq {t1:.2f} "
          f"({'STABLE' if t1 >= 0.5 else 'UNSTABLE'})", flush=True)
    print(f"CHAMPION (chance-gated): {champ}", flush=True)
    return {"tag": tag, "mc_forge": float(np.mean(fa)), "lean": float(np.mean(la)),
            "ci": [lo, hi], "lean_ci": list(_ci(la)), "paired_ci": [dlo, dhi],
            "top1_freq": t1, "champion": champ, "chance": chance,
            "n_days": len(folds), "gate": C_GATE, "W": W}


def main_run():
    raw = load_mox_mc()
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    n_grp = len(set(g))
    print(f"bank: {len(raw)} windows, {n_grp} days, classes={sorted(set(y))}, "
          f"per-class {[int((y==a).sum()) for a in sorted(set(y))]}", flush=True)
    F, names = build_features(raw, n_grp)
    lean_F = lean_baseline([(_znorm(X[0]), c, gid) for X, c, gid in raw])  # single-ch R1

    real = lodo(raw, F, names, lean_F, y, g, "MOX-DRIFT-id")

    # label-shuffle NULL: permute analyte within each day, expect ~chance
    rng = np.random.default_rng(0)
    ysh = y.copy()
    for d in np.unique(g):
        idx = np.where(g == d)[0]
        ysh[idx] = rng.permutation(y[idx])
    null = lodo(raw, F, names, lean_F, ysh, g, "MOX-DRIFT-SHUFFLE-NULL")

    log_receipt("MOX-DRIFT", {"real": real, "shuffle_null": null,
                              "spec_sha256": _sha(FROZEN_SPEC)})


def _sha(spec):
    import json
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def freeze():
    tip = log_receipt("MOX-DRIFT-PREREG", {**FROZEN_SPEC, "spec_sha256": _sha(FROZEN_SPEC)})
    print("FROZEN. ledger tip:", tip)
    print("spec_sha256:", _sha(FROZEN_SPEC))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "freeze":
        freeze()
    else:
        main_run()
