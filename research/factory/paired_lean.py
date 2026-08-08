"""Paired forge-vs-lean test: compute per-fold LEAN accuracies with the exact
outer protocol of forge_nested (train on all-but-one recording, score held-out),
then pair them with the forge fold accuracies already in the logs and bootstrap
the per-fold difference. Closes the 'gains not CI-solid unpaired' gap.
Run: .venv-research/bin/python research/factory/paired_lean.py
"""
import re, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import load_ecn, load_cwru, lean_baseline
from mfpt_run import load_mfpt

LOG = "research/factory/logs/"

def forge_folds(path, sec=None):
    out, name, cur = {}, None, {}
    for line in open(path):
        m = re.match(r"=== (\S+) nested ===", line)
        if m:
            if name: out[name] = cur
            name, cur = m.group(1), {}
        m = re.search(r"hold rec (\S+): acc=([\d.]+)", line)
        if m: cur[m.group(1)] = float(m.group(2))
    out[name or "ALL"] = cur
    return out[sec] if sec else out[name or "ALL"]

def lean_folds(raw):
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    F = lean_baseline(raw)
    accs = {}
    for hold in np.unique(g):
        tr = g != hold
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(F[tr], y[tr])
        accs[str(hold)] = float((clf.predict(F[~tr]) == y[~tr]).mean())
    return accs

def paired(name, forge, lean):
    common = sorted(set(forge) & set(lean))
    d = np.array([forge[r] - lean[r] for r in common])
    rng = np.random.default_rng(0)
    means = rng.choice(d, (10000, len(d)), replace=True).mean(1)
    print(f"{name}: paired forge-lean = {d.mean():+.3f}  "
          f"95% CI [{np.percentile(means, 2.5):+.3f}, {np.percentile(means, 97.5):+.3f}]  "
          f"n={len(d)}  folds forge>lean: {(d > 0).sum()}/{len(d)}", flush=True)

if __name__ == "__main__":
    t0 = time.time()
    paired("ECN v1", forge_folds(LOG + "v1_nested_ecn_cwru.log", "ECN"), lean_folds(load_ecn()))
    print(f"[{time.time()-t0:.0f}s]", flush=True)
    paired("MFPT v1", forge_folds(LOG + "mfpt_full.log", "MFPT"), lean_folds(load_mfpt()))
    print(f"[{time.time()-t0:.0f}s]", flush=True)
    paired("CWRU v1", forge_folds(LOG + "v1_nested_ecn_cwru.log", "CWRU"), lean_folds(load_cwru()))
    print(f"done [{time.time()-t0:.0f}s]", flush=True)
