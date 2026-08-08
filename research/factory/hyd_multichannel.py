"""HYD multi-sensor — second testbed for the channel-combiner slot (Jul 3,
dogfoods the feature_forge core API from step 1; first was GAS, id-LODO 0.621).

Targets = the two remaining HYD nulls (single-sensor PS1 texture):
  accumulator (4 cls, lean 0.275 n.s.), valve texture (4 cls, lean 0.192).
Hypothesis (same shape as GAS): the state lives in the PATTERN ACROSS the six
pressure sensors (PS1-PS6, all 100 Hz, 6000 samples/cycle) — e.g. pressure
drops BETWEEN circuit points — invisible to any single channel.

Bank: same cycle selection as load_hydraulic (stable cycles, per_class=6,
evenly spread); window = (6, 1024) at 4 aligned offsets per cycle; recording =
cycle (LOGO). Capacity gate C=50 x n_rec over the sampled mc space. Receipts:
nested selection, stability, champion vs single-channel lean (PS1), chance
gate, paired CI, ledger. Same-rig-adjacency caveat applies as before.
Run: nice -n 19 ../../.venv-research/bin/python hyd_multichannel.py [accumulator|valve]
"""
import os, sys, time
import numpy as np
from scipy.signal import detrend
from feature_forge import mc_programs, run_mc_prog, lean_baseline
from forge_nested import anova_rank, inner_logo
from gas_multichannel import _receipts, _rf
from receipt_ledger import log_receipt

W, C_GATE, HYD = 1024, 50, "<local-path>/signalmap/data/hydraulic"
CACHE = f"{HYD}/ps_stack.npy"


def _znorm(seg):
    s = detrend(np.ascontiguousarray(seg, float))
    return (s - s.mean()) / (s.std() + 1e-12)


def load_hyd_mc(target, per_class=6, win_per_cycle=4):
    if os.path.exists(CACHE):
        sig = np.load(CACHE)                       # (2205, 6, 6000)
    else:
        sig = np.stack([np.loadtxt(f"{HYD}/PS{i}.txt") for i in range(1, 7)], 1)
        np.save(CACHE, sig)
    prof = np.loadtxt(f"{HYD}/profile.txt")
    col = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}[target]
    stable = prof[:, 4] == 0
    labels = prof[:, col]
    raw, gid = [], 0
    for cls in np.unique(labels):
        cand = np.where((labels == cls) & stable)[0]
        picks = cand[np.linspace(0, len(cand) - 1, per_class).astype(int)]
        for c in picks:
            for o in np.linspace(0, sig.shape[2] - W, win_per_cycle).astype(int):
                raw.append((np.ascontiguousarray(sig[c, :, o:o + W], float),
                            f"{target}{cls:g}", gid))
            gid += 1
    return raw


def run(target):
    raw = load_hyd_mc(target)
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    recs = np.unique(g); chance = 1 / len(set(y))
    print(f"\n===== HYD-{target} MULTI-CHANNEL: {len(raw)} win, {len(recs)} recs, "
          f"chance {chance:.3f} =====", flush=True)
    space = mc_programs(6)
    budget = min(len(space), C_GATE * len(recs))
    rng = np.random.default_rng(0)
    picks = [space[k] for k in rng.choice(len(space), budget, replace=False)]
    names = [m[0] for m in picks]
    t0 = time.time()
    F = np.empty((len(raw), budget))
    for w, (X, _, _) in enumerate(raw):
        cache = {}
        F[w] = [run_mc_prog(m, X, cache) for m in picks]
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"features {budget}/{len(space)} (gate C={C_GATE}x{len(recs)}) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    lean_F = lean_baseline([(_znorm(X[0]), c, gid) for X, c, gid in raw])  # PS1 baseline

    folds, lean_folds, sel_log = {}, {}, {}
    for hold in recs:
        tr = g != hold
        order = anova_rank(F[tr], y[tr], g[tr])
        sel, best = [], 0.0
        for j in order[:30]:
            acc = inner_logo(F[tr], y[tr], g[tr], sel + [j])
            if acc > best + 0.005:
                sel, best = sel + [j], acc
            if len(sel) == 5: break
        clf = _rf(); clf.fit(F[tr][:, sel] if sel else lean_F[tr], y[tr])
        folds[hold] = float((clf.predict(F[~tr][:, sel] if sel else lean_F[~tr]) == y[~tr]).mean())
        sel_log[hold] = [names[j] for j in sel]
        clf = _rf(); clf.fit(lean_F[tr], y[tr])
        lean_folds[hold] = float((clf.predict(lean_F[~tr]) == y[~tr]).mean())
    return _receipts(f"HYD-{target}-MC (LOGO)", folds, lean_folds, sel_log, chance)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["accumulator", "valve"]
    for t in targets:
        run(t)
