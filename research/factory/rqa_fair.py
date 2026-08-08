"""RQA fair rematch (RESTART Prio A4, mandatory before public). The original
moat head-to-heads used a homemade RQA on the first 400 samples with fixed
(m=3, tau=5) and only 2 features — three handicaps. Fair version:
  - reference implementation (pyunicorn 0.9.0 RecurrencePlot)
  - FULL 1024-sample window
  - 6 standard measures: RR, DET, LAM, diag-ENTR, L_mean, TT
  - small (dim, tau) grid, best config wins -> selection bias FAVOURS RQA,
    so a surviving lean win is conservative.
Same protocol as always: LOGO over recordings, RF(150), per-fold accs saved ->
paired lean-RQA CI. Cost = feature ms/window (same machine, same loop).
Checkpointed: one process per (bank, dim, tau), appends to logs/rqa_fair.csv.
Run:  rqa_fair.py <ecn|cwru> <dim> <tau>       (one config, ~20s ECN / ~5min CWRU)
      rqa_fair.py <ecn|cwru> lean              (lean reference row)
      rqa_fair.py report                       (paired comparison from the CSV)
"""
import os, sys, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import load_ecn, load_cwru, lean_baseline

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "rqa_fair.csv")
GRID = [(3, 5), (3, 10), (5, 5), (5, 10)]


def rqa_feats(raw, dim, tau):
    from pyunicorn.timeseries import RecurrencePlot
    t0 = time.time()
    F = []
    for s, _, _ in raw:
        rp = RecurrencePlot(np.asarray(s, float), dim=dim, tau=tau,
                            recurrence_rate=0.1, silence_level=3)
        F.append([rp.recurrence_rate(), rp.determinism(l_min=2),
                  rp.laminarity(v_min=2), rp.diag_entropy(),
                  rp.average_diaglength(l_min=2), rp.trapping_time(v_min=2)])
    return np.array(F), (time.time() - t0) * 1000 / len(raw)


def logo_folds(F, y, g):
    accs = {}
    for r in np.unique(g):
        tr = g != r
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(F[tr], y[tr])
        accs[r] = float((clf.predict(F[~tr]) == y[~tr]).mean())
    return accs


def load_bank(bank):
    if bank == "ecn": return load_ecn()
    if bank == "cwru": return load_cwru()
    if bank == "mfpt":
        from mfpt_run import load_mfpt
        return load_mfpt()
    # coh/RQA-screen flag verification (Jul 12): same loader calls as the screen
    if bank == "calce":
        from retro_loaders import load_calce
        return load_calce()
    if bank == "hydcooler":
        from retro_loaders import load_hydraulic
        return load_hydraulic(target="cooler")
    raise ValueError(bank)


def run(bank, tag, feat_fn):
    raw = load_bank(bank)
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    F, ms = feat_fn(raw)
    accs = logo_folds(F, y, g)
    acc = float(np.mean(list(accs.values())))
    new = not os.path.exists(CSV)
    with open(CSV, "a") as f:
        if new: f.write("bank,config,rec,acc,ms_per_win\n")
        for r, a in accs.items():
            f.write(f"{bank},{tag},{r},{a:.4f},{ms:.2f}\n")
    print(f"{bank} {tag}: LOGO {acc:.3f}  @ {ms:.1f} ms/win", flush=True)


def report():
    import csv
    rows = list(csv.DictReader(open(CSV)))
    for bank in ("ecn", "cwru", "mfpt", "calce", "hydcooler"):
        cfgs = {}
        for r in rows:
            if r["bank"] != bank: continue
            cfgs.setdefault(r["config"], {})[r["rec"]] = (float(r["acc"]), float(r["ms_per_win"]))
        if "lean" not in cfgs: continue
        lean = cfgs["lean"]
        print(f"\n===== {bank.upper()} (lean {np.mean([v[0] for v in lean.values()]):.3f} "
              f"@ {list(lean.values())[0][1]:.1f} ms/win) =====", flush=True)
        best, bacc = None, -1
        for cfg, folds in sorted(cfgs.items()):
            if cfg == "lean": continue
            acc = np.mean([v[0] for v in folds.values()])
            print(f"  RQA {cfg}: {acc:.3f} @ {list(folds.values())[0][1]:.0f} ms/win", flush=True)
            if acc > bacc: best, bacc = cfg, acc
        if best is None: continue
        common = sorted(set(lean) & set(cfgs[best]))
        d = np.array([lean[r][0] - cfgs[best][r][0] for r in common])
        rng = np.random.default_rng(0)
        dm = rng.choice(d, (10000, len(d)), replace=True).mean(1)
        lo, hi = np.percentile(dm, 2.5), np.percentile(dm, 97.5)
        cost = list(cfgs[best].values())[0][1] / max(list(lean.values())[0][1], 1e-9)
        print(f"  BEST RQA = {best}; paired lean-RQA {d.mean():+.3f} CI [{lo:+.3f},{hi:+.3f}]  "
              f"lean {cost:.0f}x cheaper", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    elif sys.argv[2] == "lean":
        def lean_timed(raw):
            t0 = time.time()
            F = lean_baseline(raw)
            return F, (time.time() - t0) * 1000 / len(raw)
        run(sys.argv[1], "lean", lean_timed)
    else:
        dim, tau = int(sys.argv[2]), int(sys.argv[3])
        run(sys.argv[1], f"m{dim}t{tau}", lambda raw: rqa_feats(raw, dim, tau))
