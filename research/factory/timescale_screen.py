"""Timescale screen — receipt derived from the GAS discovery (method thesis #3:
window timescale is a search dimension; GAS-id went 0.158@10s -> 0.346@256s
once windows spanned the slow chemistry). Cheap proxy for any existing bank:
mean-pool the ALREADY-CUT windows by p in {1,4,16} (window shrinks 1024 ->
256/64 samples but spans the same time; features tolerate short windows) and
re-run the lean LOGO. A MONOTONE RISE with pooling flags a timescale mismatch:
the honest fix is then to re-cut the bank from the SOURCE signal with pooling
(keeping W=1024) like GAS did — this screen only cheaply points there, it does
not replace the rebuild. Flat or falling = the bank's null is not a windowing
artifact on this axis.

Run: .venv-research/bin/python timescale_screen.py   (all remaining nulls)
"""
import numpy as np
from feature_forge import lean_baseline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

POOLS = (1, 4, 16)


def _pool(w, p):
    if p == 1: return w
    n = (len(w) // p) * p
    return w[:n].reshape(-1, p).mean(1)


def _logo(F, y, g):
    accs = []
    for r in np.unique(g):
        tr = g != r
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(F[tr], y[tr])
        accs.append(float((clf.predict(F[~tr]) == y[~tr]).mean()))
    return float(np.mean(accs))


def timescale_screen(name, raw):
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    chance = 1 / len(set(y))
    accs = {}
    for p in POOLS:
        pooled = [(_pool(np.asarray(s, float), p), c, gid) for s, c, gid in raw]
        accs[p] = _logo(lean_baseline(pooled), y, g)
    a = [accs[p] for p in POOLS]
    rising = a[0] < a[1] < a[2] and a[2] > chance + 0.05
    print(f"{name}: " + "  ".join(f"pool{p}={accs[p]:.3f}" for p in POOLS)
          + f"  (chance {chance:.3f})  "
          + ("-> TIMESCALE MISMATCH: rebuild bank with source-level pooling"
             if rising else "-> flat/falling: null not a window-timescale artifact"),
          flush=True)
    return accs, rising


if __name__ == "__main__":
    from retro_loaders import load_hydraulic, load_dcase_valve, load_dcase_pump, load_ims
    banks = [
        ("HYD-accumulator", lambda: load_hydraulic(target="accumulator")),
        ("DCASE-valve-anomaly", lambda: load_dcase_valve(task="anomaly", full_coverage=True)),
        ("DCASE-pump-anomaly", lambda: load_dcase_pump(task="anomaly")),
        ("DCASE-pump-id", lambda: load_dcase_pump(task="id")),
        ("IMS-fault", load_ims),
    ]
    for name, fn in banks:
        try:
            timescale_screen(name, fn())
        except Exception as e:
            print(f"{name}: SKIP ({e})", flush=True)
