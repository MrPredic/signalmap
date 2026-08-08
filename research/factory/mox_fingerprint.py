"""WS3 Phase 1 (EXPLORATORY interpretation) — extract the stable forge winners
on MOX-DRIFT and resolve them to concrete sensor channels / channel-pairs.

Reuses mox_drift.build_features / _fstat / _rf UNCHANGED (same seed=0 selection
as the frozen #11 verdict). This step is interpretation of already-selected
programs, not a new readout -> no prereg needed here. The prereg + isolated
out-of-sample validation of the named fingerprint is the NEXT step.

    python mox_fingerprint.py            # build (cache) F, aggregate winners
"""
import os, re, sys, collections
import numpy as np
import mox_drift as mx
from feature_forge import programs

F_CACHE = "cache/mox_F.npz"


def _build_or_load_F(raw, n_grp):
    names = None
    if os.path.exists(F_CACHE):
        z = np.load(F_CACHE, allow_pickle=True)
        if len(z["F"]) == len(raw):
            return z["F"], list(z["names"])
    F, names = mx.build_features(raw, n_grp)   # seed=0 -> identical picks to #11
    os.makedirs(os.path.dirname(F_CACHE), exist_ok=True)
    np.savez_compressed(F_CACHE, F=F, names=np.array(names, dtype=object))
    return F, names


def _selected_per_fold(raw, F, names, y, g, k=5):
    """Repeat the EXACT LODO prescreen from mox_drift.lodo, capturing the top-k
    F-stat-selected program names for every held-out day."""
    sel_log = {}
    for hold in np.unique(g):
        tr = g != hold
        if len(set(y[tr])) < 2 or (~tr).sum() == 0:
            continue
        sel = list(np.argsort(-mx._fstat(F[tr], y[tr]))[:k])
        sel_log[hold] = [names[j] for j in sel]
    return sel_log


def _parse_combiner(name):
    """name = 'norm:combiner:program'. Return (combiner_kind, channels tuple)."""
    combiner = name.split(":", 2)[1]
    m = re.match(r"logratio\((\d+),(\d+)\)", combiner)
    if m:
        return "logratio", (int(m.group(1)), int(m.group(2)))
    m = re.match(r"chdiff\((\d+),(\d+)\)", combiner)
    if m:
        return "chdiff", (int(m.group(1)), int(m.group(2)))
    m = re.match(r"ch(\d+)$", combiner)
    if m:
        return "ch", (int(m.group(1)),)
    return combiner, ()   # mean62 / std62


def main():
    raw = mx.load_mox_mc()
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    n_grp = len(set(g))
    F, names = _build_or_load_F(raw, n_grp)
    print(f"F {F.shape}  ({n_grp} days, {len(set(y))} analytes)", flush=True)

    sel = _selected_per_fold(raw, F, names, y, g)
    nfold = len(sel)

    top1 = collections.Counter(v[0] for v in sel.values() if v)
    topk = collections.Counter(n for v in sel.values() for n in v)

    print(f"\n=== TOP-1 selected program across {nfold} LODO folds ===")
    for name, c in top1.most_common(8):
        kind, ch = _parse_combiner(name)
        print(f"  {c:3d}/{nfold}  [{kind} {ch}]  {name}")

    print(f"\n=== combiner KIND frequency among all top-5 selections ===")
    kinds = collections.Counter(_parse_combiner(n)[0] for v in sel.values() for n in v)
    for kind, c in kinds.most_common():
        print(f"  {kind:10s} {c}")

    print(f"\n=== channel-PAIR frequency (logratio/chdiff) among top-5 ===")
    pairs = collections.Counter()
    singles = collections.Counter()
    for v in sel.values():
        for n in v:
            kind, ch = _parse_combiner(n)
            if len(ch) == 2:
                pairs[tuple(sorted(ch))] += 1
            elif len(ch) == 1:
                singles[ch[0]] += 1
    for pr, c in pairs.most_common(12):
        print(f"  pair {pr}  x{c}")
    print(f"  (single channels among top-5: {dict(singles.most_common(6))})")

    print(f"\n=== channel PARTICIPATION (any pair/single, top-5) ===")
    part = collections.Counter()
    for v in sel.values():
        for n in v:
            _, ch = _parse_combiner(n)
            for c in ch:
                part[c] += 1
    for c, n in part.most_common(12):
        print(f"  ch{c}: {n}")


if __name__ == "__main__":
    main()
