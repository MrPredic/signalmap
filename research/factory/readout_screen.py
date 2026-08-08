"""Readout screen — ONE cheap screening matrix over the four readout families
(method theses #1-#4 turned into a standard receipt). For every existing bank
of already-cut (window, label, recording) triples it evaluates window-level
PROXIES of each source-level readout choice and re-runs the lean LOGO:

  family     variants                     proxy of (source-level fix)
  ---------  ---------------------------  ------------------------------------
  sampling   even_sub vs peak_sub         where windows are cut in the clip
             (same-length subwindow at    (loader sampling='peak', cf. DCASE-
             center vs at max rolling     valve 0.238->0.875)
             energy)
  timescale  pool1(base)/pool4/pool16     window span vs process timescale
                                          (source-level pooling, cf. IMS 0.755,
                                          GAS 0.158@10s->0.346@256s)
  norm       level = lean + amplitude     shape-only z-norm at load DESTROYS
             receipts (log std, log ptp)  level; degenerate cell = the bank
                                          threw level away at readout
  window     integ = lean on cumsum(x)    phase-aligned scale-preserving
                                          windows (cf. HYD-valve PHASE +0.383)

Flags are PAIRED over recording folds (bootstrap CI of the per-fold acc diff
> 0), not thresholds. A flag does NOT change any claim — it points to the
honest fix: rebuild the bank from the SOURCE signal with that readout and put
it through bank_audit + gauntlet. The matrix itself is the product argument:
standard readout tests one cell of twelve.

Run: nice -n 19 .venv-research/bin/python readout_screen.py   (checkpointed:
logs/readout_screen.csv, safe to re-run; per-bank, per-variant rows)
"""
import csv, os, time
import numpy as np
from feature_forge import lean_baseline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "logs", "readout_screen.csv")
VARIANTS = ("base", "pool4", "pool16", "even_sub", "peak_sub", "level", "integ")


# ------------------------------------------------------------- variant builders
def _pool(w, p):
    n = (len(w) // p) * p
    return w[:n].reshape(-1, p).mean(1)


def _sub(w, mode):
    """Same-length subwindow (len//4, >=64): center (even) vs max rolling-std
    (peak). Same rule for every class -> no procedural confound."""
    L = max(64, len(w) // 4)
    if len(w) <= L: return w
    if mode == "even":
        o = (len(w) - L) // 2
    else:
        step = max(1, L // 4)
        offs = range(0, len(w) - L + 1, step)
        o = max(offs, key=lambda i: w[i:i + L].std())
    return w[o:o + L]


def _variant_features(raw, v):
    if v == "base":
        return lean_baseline(raw)
    if v in ("pool4", "pool16"):
        p = int(v[4:])
        return lean_baseline([(_pool(np.asarray(s, float), p), c, g) for s, c, g in raw])
    if v in ("even_sub", "peak_sub"):
        return lean_baseline([(_sub(np.asarray(s, float), v[:4]), c, g) for s, c, g in raw])
    if v == "level":
        L = lean_baseline(raw)
        amp = np.array([[np.log(np.std(s) + 1e-12), np.log(np.ptp(s) + 1e-12)]
                        for s, _, _ in raw])
        return np.hstack([L, amp])
    if v == "integ":
        out = []
        for s, c, g in raw:
            x = np.cumsum(np.asarray(s, float) - np.mean(s))
            out.append((x / (x.std() + 1e-12), c, g))
        return lean_baseline(out)
    raise ValueError(v)


# ------------------------------------------------------------------ evaluation
def _logo_folds(F, y, g):
    folds = {}
    for r in np.unique(g):
        tr = g != r
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(F[tr], y[tr])
        folds[str(r)] = float((clf.predict(F[~tr]) == y[~tr]).mean())
    return folds


def _paired_ci(fa, fb, seed=0):
    """Bootstrap CI of mean(fa - fb) over recording folds (paired)."""
    recs = sorted(fa)
    d = np.array([fa[r] - fb[r] for r in recs])
    m = np.random.default_rng(seed).choice(d, (10000, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _done():
    """(bank, variant) -> fold dict from checkpoint CSV (resume-safe flags)."""
    if not os.path.exists(CSV): return {}
    out = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            out[(r["bank"], r["variant"])] = dict(
                (k, float(v)) for k, v in
                (kv.rsplit(":", 1) for kv in r["folds"].split(";")))
    return out


def _append(row):
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bank", "variant", "acc", "chance",
                                          "n_win", "n_rec", "secs", "folds"])
        if new: w.writeheader()
        w.writerow(row)


def screen_bank(name, raw, done):
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    chance = 1 / len(set(y))
    n_win, n_rec = len(raw), len(np.unique(g))
    # level degeneracy: loader z-normed every window -> level destroyed at load
    stds = np.array([np.std(s) for s, _, _ in raw])
    level_dead = stds.std() / (stds.mean() + 1e-12) < 1e-6
    folds = {}
    for v in VARIANTS:
        if (name, v) in done:
            folds[v] = done[(name, v)]
            continue
        t0 = time.time()
        F = _variant_features(raw, v)
        folds[v] = _logo_folds(F, y, g)
        acc = float(np.mean(list(folds[v].values())))
        _append({"bank": name, "variant": v, "acc": f"{acc:.4f}",
                 "chance": f"{chance:.4f}", "n_win": n_win, "n_rec": n_rec,
                 "secs": f"{time.time()-t0:.0f}",
                 "folds": ";".join(f"{r}:{a:.3f}" for r, a in folds[v].items())})
        print(f"  {name}/{v}: {acc:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    _flags(name, folds, chance, level_dead)


def _flags(name, folds, chance, level_dead=None):
    """Paired fold CIs against the family-correct reference. TIMESCALE uses the
    BEST pool level, not only pool16 (fix Jul 4: IMS peaks at pool4 then falls
    again — a pool16-only rule missed the one timescale artifact we already
    knew about)."""
    acc = {v: float(np.mean(list(folds[v].values()))) for v in VARIANTS}
    a1, a4, a16 = acc["base"], acc["pool4"], acc["pool16"]
    rising = a1 < a4 < a16 and a16 > chance + 0.05
    bp = "pool4" if a4 >= a16 else "pool16"
    dts, lots, _ = _paired_ci(folds[bp], folds["base"])
    dpk, lopk, _ = _paired_ci(folds["peak_sub"], folds["even_sub"])
    dlv, lolv, _ = _paired_ci(folds["level"], folds["base"])
    din, loin, _ = _paired_ci(folds["integ"], folds["base"])
    flags = []

    def grade(d, lo, v):
        """Screen = cheap pointer, so a false negative costs a discovery:
        CI-lo>0 -> strong flag; else d>=0.10 above-chance -> WEAK flag
        (paired CI n.s. with few recordings; only the source-level rebuild
        through bank_audit+gauntlet decides)."""
        if lo > 0 and acc[v] > chance + 0.05: return "strong"
        if d >= 0.10 and acc[v] > chance + 0.05: return "weak"
        return None

    gts = grade(dts, lots, bp)
    if rising or gts:
        flags.append(f"TIMESCALE{' WEAK' if gts == 'weak' and not rising else ''} "
                     f"({bp}-base +{dts:.3f}, CI-lo {lots:+.3f}, "
                     f"{'monotone' if rising else 'peaked'})")
    gpk = grade(dpk, lopk, "peak_sub")
    if gpk:
        flags.append(f"SAMPLING{' WEAK' if gpk == 'weak' else ''} "
                     f"(peak-even +{dpk:.3f}, CI-lo {lopk:+.3f})")
    glv = grade(dlv, lolv, "level")
    if glv:
        flags.append(f"NORM/LEVEL{' WEAK' if glv == 'weak' else ''} "
                     f"(+{dlv:.3f}, CI-lo {lolv:+.3f})"
                     + ("" if level_dead is None else
                        " [level destroyed at load -> crest only]" if level_dead else
                        " [confound-prone: chance-gate at rebuild]"))
    gin = grade(din, loin, "integ")
    if gin:
        flags.append(f"WINDOW/PHASE{' WEAK' if gin == 'weak' else ''} "
                     f"(integ-base +{din:.3f}, CI-lo {loin:+.3f})")
    print(f"{name}: base={a1:.3f} pool4={a4:.3f} pool16={a16:.3f} "
          f"even={acc['even_sub']:.3f} peak={acc['peak_sub']:.3f} "
          f"level={acc['level']:.3f}{'(dead)' if level_dead else ''} "
          f"integ={acc['integ']:.3f}  (chance {chance:.3f})", flush=True)
    print(f"{name}: " + ("; ".join(flags) if flags else "no readout flag "
          "(null/claim robust on all four proxy axes)"), flush=True)


def replay_flags():
    """Recompute the flag matrix from the checkpoint CSV alone (no bank loads).
    level_dead is not persisted -> level flags print without the annotation."""
    done, meta = _done(), {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            meta[r["bank"]] = float(r["chance"])
    for name, chance in meta.items():
        if all((name, v) in done for v in VARIANTS):
            _flags(name, {v: done[(name, v)] for v in VARIANTS}, chance)


# ------------------------------------------------------------------- bank list
def banks():
    from feature_forge import load_ecn, load_cwru
    from mfpt_run import load_mfpt
    from retro_loaders import (load_hydraulic, load_dcase_valve, load_dcase_pump,
                               load_ims, load_calce, load_seismic_depth)
    from harvest2_loaders import load_geomag, load_gridfreq, load_gas
    from harvest3_loaders import load_volcano
    return [
        ("ECN", load_ecn),
        ("CWRU", load_cwru),
        ("MFPT", load_mfpt),
        ("CALCE-soh", load_calce),
        ("HYD-cooler", lambda: load_hydraulic(target="cooler")),
        ("HYD-valve", lambda: load_hydraulic(target="valve")),
        ("HYD-accumulator", lambda: load_hydraulic(target="accumulator")),
        ("DCASE-valve-id", lambda: load_dcase_valve(task="id")),
        ("DCASE-valve-anomaly", lambda: load_dcase_valve(task="anomaly")),
        ("DCASE-pump-id", lambda: load_dcase_pump(task="id")),
        ("DCASE-pump-anomaly", lambda: load_dcase_pump(task="anomaly")),
        ("IMS-fault", load_ims),
        ("SEIS-depth", load_seismic_depth),
        ("GEOMAG-storm", load_geomag),
        ("GRIDFREQ-area", load_gridfreq),
        ("GAS-id", load_gas),
        ("VOLCANO-eruption", load_volcano),
    ]


if __name__ == "__main__":
    import sys
    if "--flags" in sys.argv:
        replay_flags(); sys.exit(0)
    done = _done()
    for name, fn in banks():
        if all((name, v) in done for v in VARIANTS):
            print(f"{name}: all variants checkpointed, skip", flush=True)
            continue
        try:
            raw = fn()
        except Exception as e:
            print(f"{name}: SKIP ({type(e).__name__}: {e})", flush=True)
            continue
        try:
            screen_bank(name, raw, done)
        except Exception as e:
            print(f"{name}: FAIL ({type(e).__name__}: {e})", flush=True)
