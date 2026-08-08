"""GAS-MC hardening on the frozen R2 holdout (RESTART Prio B4).

The channel-combiner result (GAS-id-LODO 0.621 CI[0.449,0.738]) was found by
SEARCHING on repetition R1. R2 (same devices, same gases, second concentration
repetition) was never touched by any search session -> it is a pre-registration
substitute. Rules here: the combiner FAMILY is frozen from R1 (top-k by group-
safe F-stat on ALL of R1, before R2 is ever loaded); on R2 there is NO
re-selection of any kind, only classifier refits inside folds.

Receipts:
  A. R2-LODO (frozen family): the original claim's protocol on fresh data.
  B. Cross-rep transfer: train on R1 devices != d, test on R2 device d —
     held-out device AND held-out repetition simultaneously (hardest cut).
Both with device-fold bootstrap CIs + chance gate; n=5 devices is small and is
reported as such. Ledger entry at the end.
Run: nice -n 19 ../../.venv-research/bin/python gas_r2_holdout.py
"""
import numpy as np
from gas_multichannel import load_gas_mc, build_features, _rf, _ci, POOL
from gauntlet_mixed import _fstat
from receipt_ledger import log_receipt

K = 5

if __name__ == "__main__":
    # ---- 1. freeze the family on R1 (selection never sees R2)
    r1 = load_gas_mc(rep="R1")
    n_rec = len(set(r[2] for r in r1))
    F1, names = build_features(r1, n_rec)
    y1 = np.array([r[1] for r in r1]); u1 = np.array([r[3] for r in r1])
    sel = list(np.argsort(-_fstat(F1, y1))[:K])
    print(f"frozen family (selected on ALL R1, k={K}): "
          f"{[names[j] for j in sel]}", flush=True)

    # ---- 2. fresh holdout
    r2 = load_gas_mc(rep="R2")
    F2, names2 = build_features(r2, len(set(r[2] for r in r2)))
    assert names2 == names, "feature space must be identical (same seed/gate)"
    y2 = np.array([r[1] for r in r2]); u2 = np.array([r[3] for r in r2])
    chance = 1 / len(set(y2))
    print(f"R2 bank: {len(r2)} windows, {len(set(r[2] for r in r2))} recs, "
          f"{len(set(u2))} devices, chance {chance:.3f}", flush=True)

    # ---- A: R2-LODO, frozen family
    folds_a, folds_b = {}, {}
    for d in np.unique(u2):
        clf = _rf(); clf.fit(F2[u2 != d][:, sel], y2[u2 != d])
        folds_a[d] = float((clf.predict(F2[u2 == d][:, sel]) == y2[u2 == d]).mean())
        clf = _rf(); clf.fit(F1[u1 != d][:, sel], y1[u1 != d])
        folds_b[d] = float((clf.predict(F2[u2 == d][:, sel]) == y2[u2 == d]).mean())
        print(f"  hold {d}: R2-LODO={folds_a[d]:.3f}  R1->R2={folds_b[d]:.3f}",
              flush=True)
    for tag, folds in (("A R2-LODO frozen-family", folds_a),
                       ("B train-R1 -> test-R2 (cross-rep+cross-device)", folds_b)):
        fa = list(folds.values()); lo, hi = _ci(fa)
        verdict = "CI-fest > chance" if lo > chance else \
                  ("above chance, CI n.s." if np.mean(fa) > chance else "NULL")
        print(f"{tag}: {np.mean(fa):.3f}  CI [{lo:.3f},{hi:.3f}]  "
              f"(chance {chance:.3f}, n=5 devices) -> {verdict}", flush=True)
        log_receipt(f"GAS-id-MC {tag}",
                    {"acc": float(np.mean(fa)), "ci": (lo, hi), "chance": chance,
                     "folds": {str(k): v for k, v in folds.items()},
                     "frozen_family": [names[j] for j in sel],
                     "pool": POOL, "k": K, "verdict": verdict})
