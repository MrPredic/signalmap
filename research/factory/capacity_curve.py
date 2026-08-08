"""Capacity curve: nested forge accuracy on ECN as a function of grammar-subset
size. Turns the 3-point capacity rule (v1 ok, v2 collapse) into a curve — the
core evidence for the distill capacity gate (budget = C * n_recordings).
Feature matrix (196 win x 2128 progs) is computed ONCE; each (size, seed) run
does nested selection on a random program subset. Checkpointed to CSV per run.
Run: nice -n 19 .venv-research/bin/python research/factory/capacity_curve.py
"""
import os, time, csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import programs, run_prog, load_ecn
from forge_nested import anova_rank, inner_logo

SIZES = [125, 250, 500, 700, 1092, 1500, 2128]
SEEDS = [0, 1, 2]
CKPT = "research/factory/logs/capacity_curve.csv"

def nested_subset(F, y, g, names, top=30, kmax=5):
    accs = []
    for hold in np.unique(g):
        tr = g != hold
        Ftr, ytr, gtr = F[tr], y[tr], g[tr]
        order = anova_rank(Ftr, ytr, gtr)
        sel, best = [], 0.0
        for j in order[:top]:
            acc = inner_logo(Ftr, ytr, gtr, sel + [j])
            if acc > best + 0.005:
                sel, best = sel + [j], acc
            if len(sel) == kmax: break
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(Ftr[:, sel], ytr)
        accs.append(float((clf.predict(F[~tr][:, sel]) == y[~tr]).mean()))
    return float(np.mean(accs))

if __name__ == "__main__":
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            done = {(int(r["size"]), int(r["seed"])) for r in csv.DictReader(f)}
    else:
        with open(CKPT, "w") as f:
            f.write("size,seed,nested_acc,secs\n")
    raw = load_ecn()
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    progs = programs()
    print(f"grammar={len(progs)}, computing full feature matrix once...", flush=True)
    t0 = time.time()
    Ffull = np.array([[run_prog(p, s) for p in progs] for s, _, _ in raw])
    print(f"features done [{time.time()-t0:.0f}s]", flush=True)
    for size in SIZES:
        for seed in (SEEDS if size < len(progs) else [0]):
            if (size, seed) in done: continue
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(progs), size, replace=False) if size < len(progs) \
                else np.arange(len(progs))
            t1 = time.time()
            acc = nested_subset(Ffull[:, idx], y, g, [progs[i][0] for i in idx])
            secs = time.time() - t1
            with open(CKPT, "a") as f:
                f.write(f"{size},{seed},{acc:.4f},{secs:.0f}\n")
            print(f"size={size:4d} seed={seed}: nested={acc:.3f} [{secs:.0f}s]", flush=True)
    print("capacity curve complete", flush=True)
