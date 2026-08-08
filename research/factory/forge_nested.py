"""Nested validation of the forge: kills the selection bias. For every held-out
recording, ranking + greedy selection run ONLY on the remaining recordings; the
held-out recording is scored by a model that never influenced the choice.
The resulting accuracy is the honest forge number (vs the biased upper bound).
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/forge_nested.py
"""
import numpy as np, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from feature_forge import programs, run_prog, load_ecn, load_cwru, lean_baseline

def anova_rank(F, y, g):
    recs = np.unique(g)
    M = np.array([F[g == r].mean(0) for r in recs]); yr = np.array([y[g == r][0] for r in recs])
    Ms = (M - M.mean(0)) / (M.std(0) + 1e-12)
    classes = np.unique(yr)
    b = np.zeros(F.shape[1]); w = np.zeros(F.shape[1]) + 1e-12
    for c in classes:
        mc = Ms[yr == c]
        b += len(mc) * (mc.mean(0) - Ms.mean(0)) ** 2
        w += ((mc - mc.mean(0)) ** 2).sum(0)
    f = (b / max(len(classes) - 1, 1)) / (w / max(len(recs) - len(classes), 1))
    f[np.std(F, 0) <= 1e-9] = -1
    return np.argsort(-f)

def inner_logo(F, y, g, cols, trees=60):
    return cross_val_score(
        make_pipeline(StandardScaler(), RandomForestClassifier(trees, random_state=0, n_jobs=-1)),
        F[:, cols], y, groups=g, cv=LeaveOneGroupOut()).mean()

def nested(name, raw, top=30, kmax=5):
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    progs = programs()
    F = np.array([[run_prog(p, s) for p in progs] for s, _, _ in raw])
    accs = []; t0 = time.time()
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
        print(f"  hold rec {hold}: acc={accs[-1]:.3f}  sel={[progs[j][0] for j in sel]}", flush=True)
    lean = cross_val_score(
        make_pipeline(StandardScaler(), RandomForestClassifier(150, random_state=0, n_jobs=-1)),
        lean_baseline(raw), y, groups=g, cv=LeaveOneGroupOut()).mean()
    print(f"\n{name}: NESTED forged = {np.mean(accs):.3f} (biased was reported earlier)  "
          f"vs LEAN {lean:.3f}   [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    print("=== ECN nested ==="); nested("ECN", load_ecn())
    print("\n=== CWRU nested ==="); nested("CWRU", load_cwru(), top=20)
