"""Standardized new-domain gauntlet. For a bank of (window, label, recording):
1. LEAN baseline (perm3+psd_slope), LOGO + group-perm-p
2. Nested forge WITH capacity gate (budget = C * n_recordings, fixed-seed subset)
3. Bootstrap CI over recordings + paired forge-lean diff
Every number leakage-free (LOGO over recordings). Fold accs appended to CSV.
Usage: from gauntlet import gauntlet; gauntlet("NAME", raw)
CONVENTION (Jul 4): n_perm=60 is fine for exploration; any FINAL bank verdict
(RESULTS.md table row) must be run with n_perm=200 (p floor 0.005, stable
ranking around the 0.05 line).
"""
import time, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import programs, run_prog, lean_baseline, logo_scores, group_perm_p
from forge_nested import anova_rank, inner_logo

C_GATE = 50
FOLDS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "logs", "gauntlet_folds.csv")

def _ci(a, seed=0):
    a = np.array(a); rng = np.random.default_rng(seed)
    m = rng.choice(a, (10000, len(a)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def gauntlet(name, raw, n_perm=60, mode="texture"):
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    recs = np.unique(g); chance = 1 / len(set(y))
    print(f"\n===== {name}: {len(raw)} windows, {len(recs)} recordings, "
          f"{len(set(y))} classes, chance={chance:.3f} =====", flush=True)

    # 1. lean
    t0 = time.time()
    L = lean_baseline(raw)
    lean_folds = {}
    for hold in recs:
        tr = g != hold
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(L[tr], y[tr])
        lean_folds[hold] = float((clf.predict(L[~tr]) == y[~tr]).mean())
    lean_acc = float(np.mean(list(lean_folds.values())))
    lo, hi = _ci(list(lean_folds.values()))
    print(f"LEAN: {lean_acc:.3f}  CI [{lo:.3f},{hi:.3f}]  [{time.time()-t0:.0f}s]", flush=True)
    p = group_perm_p(L, y, g, lean_acc, n_perm=n_perm)
    print(f"LEAN group-perm-p = {p:.3f} ({n_perm} perms)", flush=True)

    # 2. nested forge with capacity gate
    #    mode="texture" (default): stationarity guard ON (validated: ECN 0.469->0.689);
    #    mode="phase": guard OFF (cumsum carries information in aligned windows).
    full = programs(texture_guard=(mode == "texture"))
    budget = min(len(full), C_GATE * len(recs))
    idx = np.random.default_rng(0).choice(len(full), budget, replace=False) \
        if budget < len(full) else np.arange(len(full))
    progs = [full[i] for i in idx]
    t0 = time.time()
    F = np.array([[run_prog(p_, s) for p_ in progs] for s, _, _ in raw])
    print(f"gate: {budget}/{len(full)} programs (C={C_GATE} x {len(recs)} recs); "
          f"features [{time.time()-t0:.0f}s]", flush=True)
    forge_folds = {}; sel_log = {}
    for hold in recs:
        tr = g != hold
        Ftr, ytr, gtr = F[tr], y[tr], g[tr]
        order = anova_rank(Ftr, ytr, gtr)
        sel, best = [], 0.0
        for j in order[:30]:
            acc = inner_logo(Ftr, ytr, gtr, sel + [j])
            if acc > best + 0.005:
                sel, best = sel + [j], acc
            if len(sel) == 5: break
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(Ftr[:, sel], ytr)
        forge_folds[hold] = float((clf.predict(F[~tr][:, sel]) == y[~tr]).mean())
        sel_log[hold] = [progs[j][0] for j in sel]
        print(f"  hold rec {hold}: acc={forge_folds[hold]:.3f}  sel={sel_log[hold]}", flush=True)
    fa = list(forge_folds.values())
    lo, hi = _ci(fa)
    d = np.array([forge_folds[r] - lean_folds[r] for r in recs])
    rng = np.random.default_rng(0)
    dm = rng.choice(d, (10000, len(d)), replace=True).mean(1)
    print(f"FORGE nested: {np.mean(fa):.3f}  CI [{lo:.3f},{hi:.3f}]", flush=True)
    dlo, dhi = np.percentile(dm, 2.5), np.percentile(dm, 97.5)
    print(f"PAIRED forge-lean: {d.mean():+.3f}  CI [{dlo:+.3f},{dhi:+.3f}]  "
          f"forge>lean {int((d>0).sum())}/{len(d)}", flush=True)

    # RECEIPT 1 — selection stability across folds (validated diagnostic: stable
    # winners CALCE 6/6 / SEIS 15/16 vs anti-learning VALVE-id fully unstable).
    names_per_fold = list(sel_log.values())
    top1 = [s[0] for s in names_per_fold if s]
    top1_freq = max(top1.count(t) for t in set(top1)) / len(names_per_fold) if top1 else 0.0
    sets = [set(s) for s in names_per_fold]
    jac = [len(a & b) / max(len(a | b), 1) for i, a in enumerate(sets) for b in sets[i + 1:]]
    stab = float(np.mean(jac)) if jac else 0.0
    print(f"STABILITY: top1-freq {top1_freq:.2f}, mean-jaccard {stab:.2f} "
          f"({'STABLE' if top1_freq >= 0.5 else 'UNSTABLE -> do not trust forge'})", flush=True)

    # RECEIPT 2 — champion rule: forge NEVER silently replaces lean; the paired
    # CI decides (validated: forge can be CI-solidly WORSE, VALVE-id). CHANCE
    # GATE first: a champion needs its CI lower bound above chance — otherwise
    # the verdict is NULL (found on GAS-LEVEL: paired CI crowned forge while
    # both sat below chance).
    lean_lo = _ci(list(lean_folds.values()))[0]
    if lo <= chance and lean_lo <= chance:
        champion = "NULL (nobody clears chance)"
    elif dlo > 0:
        champion = "forge"
    elif dhi < 0:
        champion = "lean"
    else:
        champion = "tie->lean (default to cheaper)"
    print(f"CHAMPION (paired CI, chance-gated): {champion}", flush=True)

    # RECEIPT 3 — classifier robustness: refit the champion's folds with a
    # LINEAR model (LogReg). A real signal survives the model swap (level may
    # drop); a signal that only RF finds is flagged, not failed — RF can
    # legitimately exploit nonlinear structure, but the receipt makes model-
    # dependence visible instead of silent.
    champ_lean = champion.startswith(("lean", "tie", "NULL"))
    alt_folds = []
    for hold in recs:
        tr = g != hold
        sel = [j for j, p_ in enumerate(progs) if p_[0] in sel_log[hold]]
        if champ_lean or not sel:
            Xtr, Xte = L[tr], L[~tr]
        else:
            Xtr, Xte = F[tr][:, sel], F[~tr][:, sel]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=0))
        clf.fit(Xtr, y[tr])
        alt_folds.append(float((clf.predict(Xte) == y[~tr]).mean()))
    alt = float(np.mean(alt_folds)); alt_lo = _ci(alt_folds)[0]
    print(f"CLF-ROBUSTNESS (LogReg on champion features): {alt:.3f}  "
          f"({'HOLDS' if alt_lo > chance else 'RF-ONLY -> flag model-dependence'})",
          flush=True)

    new = not os.path.exists(FOLDS_CSV)
    with open(FOLDS_CSV, "a") as f:
        if new: f.write("bank,rec,lean,forge\n")
        for r in recs:
            f.write(f"{name},{r},{lean_folds[r]:.4f},{forge_folds[r]:.4f}\n")
    result = {"lean": lean_acc, "forge": float(np.mean(fa)), "perm_p": p, "chance": chance,
              "paired_ci": (float(dlo), float(dhi)), "stability": stab,
              "top1_freq": top1_freq, "champion": champion,
              "clf_robust": alt, "clf_robust_holds": bool(alt_lo > chance)}
    try:  # tamper-evident hash-chain entry; never let logging kill a run
        from receipt_ledger import log_receipt
        log_receipt(name, result)
    except Exception as e:
        print(f"[ledger skipped: {e}]", flush=True)
    return result
