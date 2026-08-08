"""Bootstrap CIs over recordings for all nested fold accuracies already in the
logs. Each held-out recording is one independent unit; resample folds with
replacement, 10000 draws, report mean + 95% percentile CI.
Run: .venv-research/bin/python research/factory/bootstrap_cis.py
"""
import re
import numpy as np

LOG = "research/factory/logs/"

def folds(path):
    sections, cur, name = {}, [], None
    for line in open(path):
        m = re.match(r"=== (\S+) nested ===", line)
        if m:
            if name: sections[name] = cur
            name, cur = m.group(1), []
        m = re.search(r"hold rec \S+: acc=([\d.]+)", line)
        if m: cur.append(float(m.group(1)))
    if name: sections[name] = cur
    return sections or {"ALL": cur}

def ci(acc, n=10000, seed=0):
    acc = np.array(acc); rng = np.random.default_rng(seed)
    means = rng.choice(acc, (n, len(acc)), replace=True).mean(1)
    return acc.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

banks = {}
for name, a in folds(LOG + "v1_nested_ecn_cwru.log").items():
    banks[f"{name} v1"] = a
for name, a in folds(LOG + "v2_nested_ecn_cwru.log").items():
    banks[f"{name} v2"] = a
banks["MFPT v1"] = [a for s in folds(LOG + "mfpt_full.log").values() for a in s]
banks["NULL v2"] = folds(LOG + "forge_null_1422.log").get("ALL", [])

for name, a in banks.items():
    if not a: continue
    m, lo, hi = ci(a)
    print(f"{name:10s} n={len(a):2d}  mean={m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
