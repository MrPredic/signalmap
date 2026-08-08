"""GRIDFREQ location disentanglement (RESTART Prio A2). Original caveat: grid,
location, recorder and campaign period are inseparable (one CSV per grid). Fix:
the OSF DB has SECOND locations for 4 of the 6 grids — different site, recorder
and campaign period, same synchronous area:
  PT01 (Lisbon 2018)   -> FR01 (CE)        GB02 (2019)    -> GB01
  US_TX02 (2019)       -> US_TX01 (ERCOT)  ZA02 (2025)    -> ZA01
Test: train the 6-way grid classifier on the ORIGINAL bank only, predict the
second-location windows; correct = the paired grid label. The model has never
seen the site, device or period -> accuracy here is grid-physics, not location.
Both directions (B->A trains on a bank with the 4 grids swapped to their second
locations). Features: lean duo + the bank's stable forge champion, fixed.
Run: .venv-research/bin/python research/factory/gridfreq_disentangle.py
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import lean_baseline, programs, run_prog
from harvest2_loaders import load_gridfreq, GRIDS

PAIR = {"PT01": "FR01", "GB02": "GB01", "US_TX02": "US_TX01", "ZA02": "ZA01"}
CHAMPION = "specflat(clip(diff(x)))"  # stable winner 0.83 top1-freq (HARVEST2)


def feats(raw, champ):
    L = lean_baseline(raw)
    P = np.array([[run_prog(champ, s)] for s, _, _ in raw])
    return {"lean": L, "lean+champ": np.hstack([L, P])}


def evaluate(tag, Ftr, ytr, Fte, yte, gte):
    clf = make_pipeline(StandardScaler(),
                        RandomForestClassifier(150, random_state=0, n_jobs=-1))
    clf.fit(Ftr, ytr)
    pred = clf.predict(Fte)
    recs = np.unique(gte)
    accs = {r: float((pred[gte == r] == yte[gte == r]).mean()) for r in recs}
    a = np.array(list(accs.values()))
    rng = np.random.default_rng(0)
    m = rng.choice(a, (10000, len(a)), replace=True).mean(1)
    lo, hi = np.percentile(m, 2.5), np.percentile(m, 97.5)
    print(f"{tag}: {a.mean():.3f}  CI [{lo:.3f},{hi:.3f}]  (chance 0.167)", flush=True)
    for grid in sorted(set(yte)):
        mask = yte == grid
        sub = pred[mask]
        acc = float((sub == grid).mean())
        wrong = {l: int((sub == l).sum()) for l in set(sub) if l != grid}
        print(f"  {grid}: {acc:.3f}  misses->{wrong}", flush=True)


if __name__ == "__main__":
    champ = {p[0]: p for p in programs()}[CHAMPION]
    raw_a = load_gridfreq()                                  # original 6 grids
    raw_b = load_gridfreq(grids=tuple(PAIR))                 # 4 second locations
    ya = np.array([r[1] for r in raw_a]); ga = np.array([r[2] for r in raw_a])
    yb = np.array([PAIR[r[1]] for r in raw_b]); gb = np.array([r[2] for r in raw_b])
    print(f"A(original): {len(raw_a)} win/{len(set(ga))} recs; "
          f"B(2nd loc): {len(raw_b)} win/{len(set(gb))} recs", flush=True)
    Fa, Fb = feats(raw_a, champ), feats(raw_b, champ)

    for fs in ("lean", "lean+champ"):
        evaluate(f"A->B [{fs}]", Fa[fs], ya, Fb[fs], yb, gb)

    # B->A: train on the swapped bank (2nd locations + untouched SE01/IS01),
    # test only the 4 swapped grids' original recordings.
    raw_rest = load_gridfreq(grids=("SE01", "IS01"))
    yr = np.array([r[1] for r in raw_rest]); gr = np.array([r[2] for r in raw_rest])
    Fr = feats(raw_rest, champ)
    mask_a = np.isin(ya, list(PAIR.values()))
    for fs in ("lean", "lean+champ"):
        Ftr = np.vstack([Fb[fs], Fr[fs]]); ytr = np.concatenate([yb, yr])
        evaluate(f"B->A [{fs}]", Ftr, ytr, Fa[fs][mask_a], ya[mask_a], ga[mask_a])
