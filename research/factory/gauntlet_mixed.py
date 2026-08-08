"""Gauntlet for MIXED-LABEL groups (a recording contains several classes):
wearables (subject has many activities), battery-LOCO (cell spans SOH stages).
Honesty comes from the CLASSIFIER metric (leave-one-group-out over the true unit
= subject/patient/cell).

v2 (speed + honesty fix): selection is per-fold PRESCREEN -> FIXED TOP-K.
  - v1 ran greedy-forward where every candidate was scored by a full inner LOGO,
    per outer fold: O(folds x candidates x folds x RF) — minutes-to-hours on
    20+ groups (HAR had to be killed).
  - v1 also computed the prescreen ANOVA-F on ALL windows (test included) — a
    selection leak. v2 ranks on TRAIN windows only, inside each fold.
  - v2 takes the top-k F-stat features directly (no greedy, no inner CV):
    one RF fit per fold. Selection never sees the held-out group -> still nested.
Window-level ANOVA-F has pseudoreplication (windows within a group correlate);
it is only a selection heuristic — honesty comes from the held-out-group score.
group_perm_p is NOT computed here: with labels varying inside a group, the
recording-label permutation scheme does not apply (see RESULTS.md CALCE note).

Usage: from gauntlet_mixed import gauntlet_mixed
Self-test (speed regression, CALCE-LOCO + synthetic 20 groups):
  nice -n 19 .venv-research/bin/python research/factory/gauntlet_mixed.py
"""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from feature_forge import programs, run_prog, lean_baseline

C_GATE = 50

def _ci(a, seed=0):
    a = np.array(a); rng = np.random.default_rng(seed)
    m = rng.choice(a, (10000, len(a)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def _fold_accs(X, y, g):
    accs = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(X[tr], y[tr]); accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return accs

def _fstat(F, y):
    """Window-level ANOVA-F per feature (vectorized; selection heuristic only)."""
    gm = F.mean(0); b = np.zeros(F.shape[1]); w = np.zeros(F.shape[1]) + 1e-12
    for c in np.unique(y):
        fc = F[y == c]
        b += len(fc) * (fc.mean(0) - gm) ** 2
        w += ((fc - fc.mean(0)) ** 2).sum(0)
    return np.where(F.std(0) > 1e-9, b / w, -1.0)

def gauntlet_mixed(name, raw, k=5):
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    recs = np.unique(g); chance = 1 / len(set(y))
    print(f"\n===== {name} (MIXED-LABEL LOGO): {len(raw)} win, {len(recs)} groups, "
          f"{len(set(y))} cls, chance={chance:.3f} =====", flush=True)

    L = lean_baseline(raw)
    lean_folds = _fold_accs(L, y, g)
    lo, hi = _ci(lean_folds)
    print(f"LEAN: {np.mean(lean_folds):.3f}  CI [{lo:.3f},{hi:.3f}]", flush=True)

    full = programs()
    budget = min(len(full), C_GATE * len(recs))
    idx = (np.random.default_rng(0).choice(len(full), budget, replace=False)
           if budget < len(full) else np.arange(len(full)))
    progs = [full[i] for i in idx]
    t0 = time.time()
    F = np.array([[run_prog(p, s) for p in progs] for s, _, _ in raw])
    print(f"gate {budget}/{len(full)} programs; features [{time.time()-t0:.0f}s]", flush=True)

    # NESTED: per fold, rank on TRAIN windows only -> fixed top-k -> one RF fit
    forge_folds, sel_log = [], {}
    for tr, te in LeaveOneGroupOut().split(F, y, g):
        sel = list(np.argsort(-_fstat(F[tr], y[tr]))[:k])
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(F[tr][:, sel], y[tr])
        forge_folds.append(float((clf.predict(F[te][:, sel]) == y[te]).mean()))
        sel_log[str(g[te][0])] = [progs[j][0] for j in sel]
    for hold, names in sel_log.items():
        print(f"  hold {hold}: acc={forge_folds[list(sel_log).index(hold)]:.3f}  sel={names}", flush=True)
    lo, hi = _ci(forge_folds)
    d = np.array(forge_folds) - np.array(lean_folds)
    rng = np.random.default_rng(0); dm = rng.choice(d, (10000, len(d)), replace=True).mean(1)
    dlo, dhi = np.percentile(dm, 2.5), np.percentile(dm, 97.5)
    dt = time.time() - t0
    print(f"FORGE nested (top-{k} prescreen): {np.mean(forge_folds):.3f}  CI [{lo:.3f},{hi:.3f}]  [{dt:.0f}s]", flush=True)
    print(f"PAIRED forge-lean: {d.mean():+.3f}  CI [{dlo:+.3f},{dhi:+.3f}]  "
          f"forge>lean {int((d>0).sum())}/{len(d)}", flush=True)

    # RECEIPT 1 — selection stability across folds (same diagnostic as gauntlet.py)
    names_per_fold = list(sel_log.values())
    top1 = [s[0] for s in names_per_fold if s]
    top1_freq = max(top1.count(t) for t in set(top1)) / len(names_per_fold) if top1 else 0.0
    sets = [set(s) for s in names_per_fold]
    jac = [len(a & b) / max(len(a | b), 1) for i, a in enumerate(sets) for b in sets[i + 1:]]
    stab = float(np.mean(jac)) if jac else 0.0
    print(f"STABILITY: top1-freq {top1_freq:.2f}, mean-jaccard {stab:.2f} "
          f"({'STABLE' if top1_freq >= 0.5 else 'UNSTABLE -> do not trust forge'})", flush=True)

    # RECEIPT 2 — champion rule with chance gate (same semantics as gauntlet.py)
    lean_lo = _ci(lean_folds)[0]
    if lo <= chance and lean_lo <= chance:
        champion = "NULL (nobody clears chance)"
    elif dlo > 0:
        champion = "forge"
    elif dhi < 0:
        champion = "lean"
    else:
        champion = "tie->lean (default to cheaper)"
    print(f"CHAMPION (paired CI, chance-gated): {champion}", flush=True)

    result = {"lean": float(np.mean(lean_folds)), "forge": float(np.mean(forge_folds)),
              "chance": chance, "seconds": dt, "paired_ci": (float(dlo), float(dhi)),
              "stability": stab, "top1_freq": top1_freq, "champion": champion}
    try:  # tamper-evident hash-chain entry; never let logging kill a run
        from receipt_ledger import log_receipt
        log_receipt(name, result)
    except Exception as e:
        print(f"[ledger skipped: {e}]", flush=True)
    return result

# ---------------- speed-regression self-test ----------------
def _bank_synth20(n_groups=20, win=12, n_cls=3, seed=0):
    """Synthetic mixed-label bank at the 20-group scale that killed v1."""
    rng = np.random.default_rng(seed)
    raw = []
    for gid in range(n_groups):
        for w in range(win):
            cls = w % n_cls
            s = rng.standard_normal(1024)
            s[::3] += cls * 0.8
            raw.append(((s - s.mean()) / s.std(), f"c{cls}", gid))
    return raw

if __name__ == "__main__":
    t0 = time.time()
    r20 = gauntlet_mixed("SYNTH-20G (speed regression)", _bank_synth20())
    assert r20["seconds"] < 300, f"SPEED REGRESSION: 20-group forge took {r20['seconds']:.0f}s (>300s)"
    assert r20["forge"] > r20["chance"] + 0.20, f"planted signal not recovered ({r20['forge']:.3f})"

    from retro_loaders import load_calce
    raw = [(s, c, gid // 3) for s, c, gid in load_calce()]   # group = CELL (LOCO)
    rc = gauntlet_mixed("CALCE-soh-LOCO (speed regression)", raw)
    assert rc["seconds"] < 300, f"SPEED REGRESSION: CALCE-LOCO forge took {rc['seconds']:.0f}s (>300s)"
    assert rc["forge"] > rc["chance"] + 0.15, f"CALCE-LOCO forge degraded to {rc['forge']:.3f}"
    print(f"\n=== SPEED REGRESSION PASS: synth20 {r20['seconds']:.0f}s, "
          f"CALCE-LOCO {rc['seconds']:.0f}s (both <300s), total {time.time()-t0:.0f}s ===", flush=True)
