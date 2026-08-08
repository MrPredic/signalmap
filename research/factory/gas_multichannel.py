"""Multi-channel grammar — attack on the GAS double-null (RESTART Prio B6,
formulated as a SEARCH DIMENSION, not a one-off hack).

Why: GAS-id was an honest null in 5 single-channel variants; boundary analysis
said the missing information is the PATTERN ACROSS the 8 MOX channels. E-nose
literature reads arrays via between-channel (log-)ratios — a concentration-
robust gas signature invisible to any single-channel pipeline. So the grammar
gets a 4th slot: CHANNEL COMBINER before the unary program chain:
  combiners: ch_i (8) | logratio(i,j) (28) | diff(i,j) (28) | mean8 | std8  = 66
  norm modes: shape (z-norm, ratio dynamics) | level (raw, fold-internal scaler
              — ratio LEVEL is the classic array signature; single-channel level
              was falsified, multi-channel level is exactly the open hypothesis)
  programs:   existing grammar v2.1 (dedup) on the combined 1-D signal
Search space 66 x ~1150 x 2 -> sampled to the capacity gate C=50 x n_rec.
Timescale: pool=25 (4 Hz), the CI-best setting from the timescale cascade.

Receipts as always: nested selection (never sees held-out rec), stability,
champion vs single-channel lean, chance gate, paired CI, ledger entry.
Success criterion (RESTART): GAS-id-LODO (leave-one-DEVICE-out) > chance CI-fest.
Run: nice -n 19 ../../.venv-research/bin/python gas_multichannel.py
"""
import glob, time
import numpy as np
from scipy.signal import detrend
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import programs, run_prog, lean_baseline
from forge_nested import anova_rank, inner_logo
from harvest2_loaders import GAS_LAB, BASE
from receipt_ledger import log_receipt

W, POOL, C_GATE = 1024, 25, 50


def _znorm(seg):
    s = detrend(np.ascontiguousarray(seg, float))
    return (s - s.mean()) / (s.std() + 1e-12)


def load_gas_mc(files_per_rec=4, win_per_file=3, pool=POOL, rep="R1"):
    """(C=8, W) raw multi-channel windows @ 4 Hz. Same recording layout as
    load_gas (rec = unit x gas, 20 recs) so results are comparable.
    rep='R2': the second concentration repetition — NEVER used in any search
    session, i.e. a frozen fresh holdout (pre-registration substitute)."""
    raw = []
    rec_ids = {}
    for unit in ("B1", "B2", "B3", "B4", "B5"):
        for gcode, gas in GAS_LAB.items():
            files = sorted(glob.glob(f"{BASE}/gas/data1/{unit}_{gcode}_F*_{rep}.txt"))
            picks = [files[i] for i in
                     np.linspace(0, len(files) - 1, min(files_per_rec, len(files))).astype(int)]
            gid = rec_ids.setdefault(f"{unit}-{gas}", len(rec_ids))
            for fp in picks:
                m = np.loadtxt(fp, usecols=range(1, 9))          # (T, 8)
                n = (len(m) // pool) * pool
                m = m[:n].reshape(-1, pool, 8).mean(1)           # 100 Hz -> 4 Hz
                nw = min(win_per_file, max(len(m) // W, 1))
                for o in np.linspace(0, len(m) - W, nw).astype(int):
                    raw.append((np.ascontiguousarray(m[o:o + W].T), gas, gid, unit))
    return raw


def combiners():
    cs = [(f"ch{i}", lambda X, i=i: X[i]) for i in range(8)]
    for i in range(8):
        for j in range(i + 1, 8):
            cs.append((f"logratio({i},{j})",
                       lambda X, i=i, j=j: np.log(np.abs(X[i]) + 1e-9) - np.log(np.abs(X[j]) + 1e-9)))
            cs.append((f"chdiff({i},{j})", lambda X, i=i, j=j: X[i] - X[j]))
    cs.append(("mean8", lambda X: X.mean(0)))
    cs.append(("std8", lambda X: X.std(0)))
    return cs


def build_features(raw, n_rec, seed=0):
    """Sample (combiner, program, norm) triples to the capacity budget and
    precompute the feature matrix. Combiner outputs are cached per window."""
    progs = programs()                       # grammar v2.1, dedup
    cs = combiners()
    space = [(ci, pi, nm) for ci in range(len(cs))
             for pi in range(len(progs)) for nm in ("shape", "level")]
    budget = min(len(space), C_GATE * n_rec)
    rng = np.random.default_rng(seed)
    picks = [space[k] for k in rng.choice(len(space), budget, replace=False)]
    names = [f"{nm}:{cs[ci][0]}:{progs[pi][0]}" for ci, pi, nm in picks]

    t0 = time.time()
    F = np.empty((len(raw), budget))
    for w, (X, _, _, _) in enumerate(raw):
        cache = {}
        for k, (ci, pi, nm) in enumerate(picks):
            key = (ci, nm)
            if key not in cache:
                sig = cs[ci][1](X)
                cache[key] = _znorm(sig) if nm == "shape" else np.ascontiguousarray(sig, float)
            F[w, k] = run_prog(progs[pi], cache[key])
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"features: {budget}/{len(space)} sampled (gate C={C_GATE}x{n_rec}) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return F, names


def _ci(a, seed=0):
    a = np.array(a, float); rng = np.random.default_rng(seed)
    m = rng.choice(a, (10000, len(a)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _rf():
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(150, random_state=0, n_jobs=-1))


def _receipts(name, folds, lean_folds, sel_log, chance):
    fa = list(folds.values()); la = list(lean_folds.values())
    lo, hi = _ci(fa)
    d = np.array([folds[r] - lean_folds[r] for r in folds])
    rng = np.random.default_rng(0)
    dm = rng.choice(d, (10000, len(d)), replace=True).mean(1)
    dlo, dhi = np.percentile(dm, 2.5), np.percentile(dm, 97.5)
    print(f"MC-FORGE: {np.mean(fa):.3f}  CI [{lo:.3f},{hi:.3f}]   "
          f"lean(single-ch): {np.mean(la):.3f}", flush=True)
    print(f"PAIRED mc-lean: {d.mean():+.3f}  CI [{dlo:+.3f},{dhi:+.3f}]", flush=True)
    tops = [s[0] for s in sel_log.values() if s]
    t1 = max(tops.count(t) for t in set(tops)) / len(sel_log) if tops else 0.0
    print(f"STABILITY: top1-freq {t1:.2f} "
          f"({'STABLE' if t1 >= 0.5 else 'UNSTABLE -> do not trust'})", flush=True)
    if lo <= chance and _ci(la)[0] <= chance:
        champ = "NULL (nobody clears chance)"
    elif dlo > 0: champ = "mc-forge"
    elif dhi < 0: champ = "lean"
    else: champ = "tie->lean"
    print(f"CHAMPION (chance-gated): {champ}", flush=True)
    res = {"mc_forge": float(np.mean(fa)), "lean": float(np.mean(la)),
           "ci": (lo, hi), "paired_ci": (float(dlo), float(dhi)),
           "top1_freq": t1, "champion": champ, "chance": chance,
           "pool": POOL, "gate": C_GATE}
    log_receipt(name, res)
    return res


def gas_id_logo(raw, F, names, lean_F):
    """rec = unit x gas (single label per rec) -> nested greedy like gauntlet."""
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    recs = np.unique(g); chance = 1 / len(set(y))
    print(f"\n===== GAS-id MULTI-CHANNEL (LOGO, {len(recs)} recs, chance {chance:.3f}) =====",
          flush=True)
    folds, lean_folds, sel_log = {}, {}, {}
    for hold in recs:
        tr = g != hold
        order = anova_rank(F[tr], y[tr], g[tr])
        sel, best = [], 0.0
        for j in order[:30]:
            acc = inner_logo(F[tr], y[tr], g[tr], sel + [j])
            if acc > best + 0.005:
                sel, best = sel + [j], acc
            if len(sel) == 5: break
        clf = _rf(); clf.fit(F[tr][:, sel] if sel else lean_F[tr], y[tr])
        folds[hold] = float((clf.predict(F[~tr][:, sel] if sel else lean_F[~tr]) == y[~tr]).mean())
        sel_log[hold] = [names[j] for j in sel]
        clf = _rf(); clf.fit(lean_F[tr], y[tr])
        lean_folds[hold] = float((clf.predict(lean_F[~tr]) == y[~tr]).mean())
        print(f"  hold {hold}: mc={folds[hold]:.3f} lean={lean_folds[hold]:.3f} "
              f"sel={sel_log[hold][:3]}", flush=True)
    return _receipts("GAS-id-MC (LOGO)", folds, lean_folds, sel_log, chance)


def gas_id_lodo(raw, F, names, lean_F, k=5):
    """group = DEVICE (mixed labels) -> per-fold prescreen top-k (gauntlet_mixed v2)."""
    y = np.array([r[1] for r in raw]); u = np.array([r[3] for r in raw])
    chance = 1 / len(set(y))
    print(f"\n===== GAS-id MULTI-CHANNEL (LODO over 5 devices, chance {chance:.3f}) =====",
          flush=True)
    from gauntlet_mixed import _fstat
    folds, lean_folds, sel_log = {}, {}, {}
    for hold in np.unique(u):
        tr = u != hold
        sel = list(np.argsort(-_fstat(F[tr], y[tr]))[:k])
        clf = _rf(); clf.fit(F[tr][:, sel], y[tr])
        folds[hold] = float((clf.predict(F[~tr][:, sel]) == y[~tr]).mean())
        sel_log[hold] = [names[j] for j in sel]
        clf = _rf(); clf.fit(lean_F[tr], y[tr])
        lean_folds[hold] = float((clf.predict(lean_F[~tr]) == y[~tr]).mean())
        print(f"  hold {hold}: mc={folds[hold]:.3f} lean={lean_folds[hold]:.3f} "
              f"sel={sel_log[hold][:3]}", flush=True)
    return _receipts("GAS-id-MC (LODO)", folds, lean_folds, sel_log, chance)


if __name__ == "__main__":
    raw = load_gas_mc()
    n_rec = len(set(r[2] for r in raw))
    print(f"bank: {len(raw)} windows, {n_rec} recs, 5 devices, pool={POOL} (4 Hz)", flush=True)
    F, names = build_features(raw, n_rec)
    lean_F = lean_baseline([( _znorm(X[1]), c, gid) for X, c, gid, _ in raw])  # single-ch baseline
    gas_id_logo(raw, F, names, lean_F)
    gas_id_lodo(raw, F, names, lean_F)
