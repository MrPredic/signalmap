"""Coherence + RQA screen — TWO NEW screening families over the existing bank
set (sibling of readout_screen.py, same paired-LOGO-CI contract, same flag
semantics). Both families are LEAN-AUGMENTATIONS (lean + extra features), not
replacements, evaluated with the identical fold structure as the lean-only
baseline so the comparison is paired per recording.

  family      applies to            extra features (appended to lean 2-vec)
  ----------  ---------------------  ---------------------------------------
  rqa         ALL 17 readout_screen  6 standard RQA measures (pyunicorn
              banks (single-channel  RecurrencePlot, same convention as
              windows by construc-   rqa_fair.py): RR, DET, LAM, diag-ENTR,
              tion of _win/_push)    L_mean, TT. Fixed (dim=3, tau=5,
                                     recurrence_rate=0.1) -- rqa_fair's
                                     cheapest grid point, NOT grid-searched
                                     per bank (this is a cheap proxy over 17
                                     banks, not a method thesis; rqa_fair.py
                                     remains the place for a maximized,
                                     selection-bias-conservative verdict).
                                     Deterministic point cap MAX_PTS=2000/
                                     window (documented; every bank here
                                     uses W=1024 <= cap today, so the cap is
                                     currently a no-op safety net, not an
                                     active subsample -- kept explicit so a
                                     future longer-window bank is covered).

  coherence   only GENUINELY multi- 3 summary stats (mean/std/max) of the
              channel banks: GAS-id  pairwise magnitude-squared coherence
              (8 MOX channels, via   (scipy.signal.coherence, Welch) over
              gas_multichannel.      all C(n_ch,2) channel pairs. Convention
              load_gas_mc) and       COPIED from coherence.py's _mean_coh
              HYD-cooler/valve/      (Weg-4 volcano cross-station path):
              accumulator (6 PS     nperseg=min(64,len), noverlap=len//2.
              sensors, via           fs is left at scipy's default (Cxy
              hyd_multichannel.      values do not depend on fs, only the
              load_hyd_mc)           returned frequency axis does -- so this
                                     is numerically identical to coherence.
                                     py's fs_env convention, just generic).

CHANNEL-COUNT CHECK (per task): every loader reachable from readout_screen.
banks() emits 1-D single-channel windows by construction (_win/_push/_readout
in feature_forge.py / retro_loaders.py / harvest2_loaders.py all operate on
1-D signals before windowing) -- confirmed by reading those loaders, not
assumed. Genuine multi-channel raw (window.ndim==2, i.e. (n_ch, W)) only
exists via the two dedicated loaders above; _mc_bank() asserts ndim==2 and
n_ch>=2 at load time and SKIPs (does not fabricate a result) if that ever
stops being true.

lean baseline for the coherence family is the SAME single-channel null
already established as the reference in prior work: GAS channel=1 (matches
load_gas's default / gas_multichannel's lean_F), HYD channel=PS1/index-0
(matches hyd_multichannel's lean_F) -- so a flag here means the coherence
summary genuinely adds over the known single-channel null, not a different
baseline.

Flags are PAIRED over recording folds (bootstrap CI of the per-fold acc diff
aug-lean > 0), identical semantics to readout_screen.py: CI-lo>0 -> strong,
d>=0.10 above chance -> weak, else no flag. A flag does NOT change any claim
-- it points to the honest next step: rebuild the bank at SOURCE level with
that readout through bank_audit + gauntlet.

Run: nice -n 19 .venv-research/bin/python coherence_rqa_screen.py
     (checkpointed: logs/coherence_rqa_screen.csv, safe to re-run)
     --smoke   one small bank per family only (fast sanity pass)
     --flags   recompute + rewrite COHERENCE_RQA_SCREEN.md from the CSV alone
"""
import csv, os, sys, time
import numpy as np
from feature_forge import lean_baseline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from receipt_ledger import log_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "logs", "coherence_rqa_screen.csv")
MD = os.path.join(HERE, "COHERENCE_RQA_SCREEN.md")

RQA_DIM, RQA_TAU, RQA_RR = 3, 5, 0.1
MAX_PTS = 2000  # deterministic per-window point cap (see module docstring)


# ------------------------------------------------------------- RQA family
def _subsample(s, max_pts=MAX_PTS):
    s = np.asarray(s, float)
    if len(s) <= max_pts:
        return s
    idx = np.linspace(0, len(s) - 1, max_pts).astype(int)
    return s[idx]


def rqa_row(s):
    from pyunicorn.timeseries import RecurrencePlot
    x = _subsample(s)
    rp = RecurrencePlot(x, dim=RQA_DIM, tau=RQA_TAU, recurrence_rate=RQA_RR,
                        silence_level=3)
    return [rp.recurrence_rate(), rp.determinism(l_min=2), rp.laminarity(v_min=2),
            rp.diag_entropy(), rp.average_diaglength(l_min=2),
            rp.trapping_time(v_min=2)]


def rqa_features(raw):
    return np.array([rqa_row(s) for s, _, _ in raw])


# ------------------------------------------------------- coherence family
def _znorm(seg):
    from scipy.signal import detrend
    s = detrend(np.ascontiguousarray(seg, float))
    return (s - s.mean()) / (s.std() + 1e-12)


def _mean_coh(a, b):
    """Copied convention from coherence.py's _mean_coh (see module docstring)."""
    from scipy.signal import coherence
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    nper = min(64, n)
    f, Cxy = coherence(a[:n], b[:n], nperseg=nper, noverlap=nper // 2)
    return float(np.mean(Cxy))


def coherence_features(X):
    """X: (n_ch, W). 3 summary stats (mean/std/max) of pairwise mean-coherence
    over all channel pairs -- fixed-size augmentation regardless of n_ch."""
    n_ch = X.shape[0]
    pairs = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
    vals = np.array([_mean_coh(X[i], X[j]) for i, j in pairs])
    return [float(vals.mean()), float(vals.std()), float(vals.max())]


def coherence_feature_matrix(raw_mc):
    out = []
    for r in raw_mc:
        X = r[0]
        assert X.ndim == 2 and X.shape[0] >= 2, \
            f"expected (n_ch>=2, W) window, got shape {np.shape(X)}"
        out.append(coherence_features(X))
    return np.array(out)


# ------------------------------------------------------------------ LOGO/CI
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
    recs = sorted(fa)
    d = np.array([fa[r] - fb[r] for r in recs])
    m = np.random.default_rng(seed).choice(d, (10000, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def grade(d, lo, acc, chance):
    if lo > 0 and acc > chance + 0.05:
        return "strong"
    if d >= 0.10 and acc > chance + 0.05:
        return "weak"
    return None


# --------------------------------------------------------------- checkpoint
def _done():
    if not os.path.exists(CSV):
        return {}
    out = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            out[(r["bank"], r["family"], r["kind"])] = dict(
                (k, float(v)) for k, v in
                (kv.rsplit(":", 1) for kv in r["folds"].split(";")))
    return out


def _append(row):
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bank", "family", "kind", "acc", "chance",
                                          "n_win", "n_rec", "secs", "folds"])
        if new:
            w.writeheader()
        w.writerow(row)


def _get_or_compute(bank, family, kind, done, chance, n_win, n_rec, build_fn):
    key = (bank, family, kind)
    if key in done:
        return done[key]
    t0 = time.time()
    folds = build_fn()
    acc = float(np.mean(list(folds.values())))
    _append({"bank": bank, "family": family, "kind": kind, "acc": f"{acc:.4f}",
             "chance": f"{chance:.4f}", "n_win": n_win, "n_rec": n_rec,
             "secs": f"{time.time()-t0:.0f}",
             "folds": ";".join(f"{r}:{a:.3f}" for r, a in folds.items())})
    print(f"  {bank}/{family}/{kind}: {acc:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    return folds


# ------------------------------------------------------------------ runners
RESULTS = []  # rows for the final report table


def _screen(bank, family, raw_or_mc, y, g, chance, lean_fn, aug_fn, done):
    n_win, n_rec = len(y), len(np.unique(g))
    lean_folds = _get_or_compute(bank, family, "lean", done, chance, n_win, n_rec,
                                  lambda: _logo_folds(lean_fn(), y, g))
    aug_folds = _get_or_compute(bank, family, "aug", done, chance, n_win, n_rec,
                                 lambda: _logo_folds(aug_fn(), y, g))
    acc_lean = float(np.mean(list(lean_folds.values())))
    acc_aug = float(np.mean(list(aug_folds.values())))
    d, lo, hi = _paired_ci(aug_folds, lean_folds)
    flag = grade(d, lo, acc_aug, chance)
    print(f"{bank}/{family}: lean={acc_lean:.3f} aug={acc_aug:.3f} "
          f"diff={d:+.3f} CI [{lo:+.3f},{hi:+.3f}] chance={chance:.3f} "
          f"-> {flag or 'no flag'}", flush=True)
    RESULTS.append({"bank": bank, "family": family, "n_win": n_win, "n_rec": n_rec,
                    "chance": chance, "acc_lean": acc_lean, "acc_aug": acc_aug,
                    "diff": d, "ci_lo": lo, "ci_hi": hi, "flag": flag})


def run_rqa_bank(name, raw, done):
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    chance = 1 / len(set(y))
    _screen(name, "rqa", raw, y, g, chance,
            lean_fn=lambda: lean_baseline(raw),
            aug_fn=lambda: np.hstack([lean_baseline(raw), rqa_features(raw)]),
            done=done)


def run_coherence_gas(done):
    from gas_multichannel import load_gas_mc
    raw = load_gas_mc()  # (X(8,W), gas, gid, unit)
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    chance = 1 / len(set(y))
    print(f"GAS-id-MC: {len(raw)} windows, n_ch={raw[0][0].shape[0]}, "
          f"{len(np.unique(g))} recs", flush=True)
    lean_raw = [(_znorm(X[1]), c, gid) for X, c, gid, _ in raw]
    _screen("GAS-id", "coherence", raw, y, g, chance,
            lean_fn=lambda: lean_baseline(lean_raw),
            aug_fn=lambda: np.hstack([lean_baseline(lean_raw),
                                      coherence_feature_matrix(raw)]),
            done=done)


def run_coherence_hyd(target, done):
    from hyd_multichannel import load_hyd_mc
    raw = load_hyd_mc(target)  # (X(6,W), label, gid)
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    chance = 1 / len(set(y))
    print(f"HYD-{target}-MC: {len(raw)} windows, n_ch={raw[0][0].shape[0]}, "
          f"{len(np.unique(g))} recs", flush=True)
    lean_raw = [(_znorm(X[0]), c, gid) for X, c, gid in raw]
    _screen(f"HYD-{target}", "coherence", raw, y, g, chance,
            lean_fn=lambda: lean_baseline(lean_raw),
            aug_fn=lambda: np.hstack([lean_baseline(lean_raw),
                                      coherence_feature_matrix(raw)]),
            done=done)


# --------------------------------------------------------------------- report
def write_report():
    lines = [
        "# Coherence + RQA Screen",
        "",
        "Two new screening families over the existing bank set, same "
        "paired-LOGO-CI contract as readout_screen.py: lean vs lean+RQA (all "
        "banks) and lean vs lean+coherence-summary (genuinely multi-channel "
        "banks only: GAS-id 8ch, HYD-cooler/valve/accumulator 6ch). A flag "
        "does not change any claim -- it points to a source-level rebuild "
        "candidate for bank_audit + gauntlet.",
        "",
        f"RQA config: dim={RQA_DIM}, tau={RQA_TAU}, recurrence_rate={RQA_RR} "
        f"(rqa_fair.py's cheapest grid point, fixed not searched -- screen is "
        f"a cheap proxy, not a maximized method thesis). Point cap "
        f"MAX_PTS={MAX_PTS}/window (no-op today: every bank uses W=1024).",
        "",
        "| Bank | Family | n_win | n_rec | chance | lean | lean+aug | "
        "diff | CI | Flag |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in RESULTS:
        lines.append(
            f"| {r['bank']} | {r['family']} | {r['n_win']} | {r['n_rec']} | "
            f"{r['chance']:.3f} | {r['acc_lean']:.3f} | {r['acc_aug']:.3f} | "
            f"{r['diff']:+.3f} | [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] | "
            f"{r['flag'] or '-'} |")
    n_flags = sum(1 for r in RESULTS if r["flag"])
    lines += ["", f"{len(RESULTS)} bank x family cells screened, {n_flags} flagged."]
    with open(MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {MD}", flush=True)
    tip = log_receipt("COH-RQA-SCREEN", {"n_cells": len(RESULTS), "n_flags": n_flags,
                                          "rqa_config": {"dim": RQA_DIM, "tau": RQA_TAU,
                                                        "rr": RQA_RR, "max_pts": MAX_PTS},
                                          "rows": RESULTS})
    print(f"ledger tip = {tip}", flush=True)


# --------------------------------------------------------------------- main
def all_banks():
    from readout_screen import banks
    return banks()


HYD_TARGETS = ("cooler", "valve", "accumulator")


def main(smoke=False):
    done = _done()
    banks = all_banks()
    if smoke:
        banks = [b for b in banks if b[0] == "GEOMAG-storm"]
    for name, fn in banks:
        try:
            raw = fn()
        except Exception as e:
            print(f"{name}: SKIP load ({type(e).__name__}: {e})", flush=True)
            continue
        try:
            run_rqa_bank(name, raw, done)
        except Exception as e:
            print(f"{name}/rqa: FAIL ({type(e).__name__}: {e})", flush=True)

    hyd_targets = ("cooler",) if smoke else HYD_TARGETS
    for target in hyd_targets:
        try:
            run_coherence_hyd(target, done)
        except Exception as e:
            print(f"HYD-{target}/coherence: FAIL ({type(e).__name__}: {e})", flush=True)
    if not smoke:
        try:
            run_coherence_gas(done)
        except Exception as e:
            print(f"GAS-id/coherence: FAIL ({type(e).__name__}: {e})", flush=True)

    write_report()


if __name__ == "__main__":
    if "--flags" in sys.argv:
        done = _done()
        # rebuild RESULTS purely from checkpoint pairs present in the CSV
        keys = {(b, f) for (b, f, k) in done}
        chances = {}
        with open(CSV) as fh:
            for r in csv.DictReader(fh):
                chances[(r["bank"], r["family"])] = (float(r["chance"]), int(r["n_win"]),
                                                      int(r["n_rec"]))
        for b, f in sorted(keys):
            if (b, f, "lean") not in done or (b, f, "aug") not in done:
                continue
            chance, n_win, n_rec = chances[(b, f)]
            lean_folds, aug_folds = done[(b, f, "lean")], done[(b, f, "aug")]
            acc_lean = float(np.mean(list(lean_folds.values())))
            acc_aug = float(np.mean(list(aug_folds.values())))
            d, lo, hi = _paired_ci(aug_folds, lean_folds)
            flag = grade(d, lo, acc_aug, chance)
            RESULTS.append({"bank": b, "family": f, "n_win": n_win, "n_rec": n_rec,
                            "chance": chance, "acc_lean": acc_lean, "acc_aug": acc_aug,
                            "diff": d, "ci_lo": lo, "ci_hi": hi, "flag": flag})
        write_report()
        sys.exit(0)
    main(smoke="--smoke" in sys.argv)
