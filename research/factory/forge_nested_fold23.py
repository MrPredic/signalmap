"""One-off: compute the missing CWRU nested fold (hold rec 23) for grammar v2.
Same logic as forge_nested.nested(), restricted to a single held-out recording.
Run: .venv-research/bin/python research/factory/forge_nested_fold23.py
"""
import numpy as np, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import programs, run_prog, load_cwru
from forge_nested import anova_rank, inner_logo

HOLD = 23

if __name__ == "__main__":
    raw = load_cwru()
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    progs = programs()
    print(f"grammar size = {len(progs)}", flush=True)
    t0 = time.time()
    F = np.array([[run_prog(p, s) for p in progs] for s, _, _ in raw])
    print(f"features done [{time.time()-t0:.0f}s]", flush=True)
    tr = g != HOLD
    Ftr, ytr, gtr = F[tr], y[tr], g[tr]
    order = anova_rank(Ftr, ytr, gtr)
    sel, best = [], 0.0
    for j in order[:20]:
        acc = inner_logo(Ftr, ytr, gtr, sel + [j])
        if acc > best + 0.005:
            sel, best = sel + [j], acc
        if len(sel) == 5: break
    clf = make_pipeline(StandardScaler(),
                        RandomForestClassifier(150, random_state=0, n_jobs=-1))
    clf.fit(Ftr[:, sel], ytr)
    acc = float((clf.predict(F[~tr][:, sel]) == y[~tr]).mean())
    print(f"  hold rec {HOLD}: acc={acc:.3f}  sel={[progs[j][0] for j in sel]}", flush=True)
    print(f"done [{time.time()-t0:.0f}s]", flush=True)
