"""Coherence fair rematch (flag verification Jul 13, sibling of rqa_fair.py).
The coh/RQA screen flagged HYD-cooler x coherence (+0.153) and GAS-id x
coherence (+0.147), but its coherence family was an explicit cheap proxy with
three handicaps:
  - pair structure collapsed to 3 summary stats (mean/std/max over all pairs)
  - frequency structure collapsed to one broadband mean per pair
  - single Welch config (nperseg=64)
Fair version (same fairness logic as rqa_fair.py):
  - per-PAIR features: all C(n_ch,2) pairs kept separate
  - per-BAND features: magnitude-squared coherence averaged in n_bands equal
    frequency bands (band=1 reproduces the screen's broadband convention)
  - small (nperseg, n_bands) grid, best config wins -> selection bias FAVOURS
    coherence, so a surviving lean win is conservative; a confirmed flag is
    mildly optimistic (documented, same trade as the RQA fair path).
Two kinds per config, preregistered BEFORE readout (ledger COH-FAIR-PREREG):
  PRIMARY  aug   = lean 2-vec + coherence feats vs lean  (the screen's actual
                   flagged comparison: "coherence ADDS over single-channel lean")
  SECONDARY alone = coherence feats only vs lean          (family-frontier
                   semantics, comparable to the RQA fair rows)
Decision rule (frozen): flag CONFIRMED iff best-aug paired (aug-lean) bootstrap
CI-lo > 0 AND best-aug acc clears chance+0.05. Stability receipt = jackknife
over folds: frequency of the same aug config staying best. No perm-p in this
path (identical to the A4/rqa_fair methodology of the existing frontier rows).
Same protocol as always: LOGO over recordings, RF(150) seed 0, per-fold accs
saved -> paired CI. Cost = feature ms/window (same machine, same loop).
Checkpointed: one row set per (bank, config, kind) in logs/coherence_fair.csv.
Run:  coherence_fair.py <hydcooler|gasid> lean          (lean reference row)
      coherence_fair.py <hydcooler|gasid> <nper> <bands> (one config, both kinds)
      coherence_fair.py all                              (everything, in order)
      coherence_fair.py report                           (verdicts from the CSV)
"""
import os, sys, time
import numpy as np
from scipy.signal import coherence, detrend
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import lean_baseline
from receipt_ledger import log_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "logs", "coherence_fair.csv")
GRID = [(64, 1), (128, 2), (256, 4), (256, 8)]
BANKS = ("hydcooler", "gasid")


def _znorm(seg):
    s = detrend(np.ascontiguousarray(seg, float))
    return (s - s.mean()) / (s.std() + 1e-12)


def load_bank(bank):
    """EXACT same loader calls + lean convention as coherence_rqa_screen.py."""
    if bank == "hydcooler":
        from hyd_multichannel import load_hyd_mc
        raw = load_hyd_mc("cooler")                       # (X(6,W), label, gid)
        mc = [(X, c, gid) for X, c, gid in raw]
        lean_raw = [(_znorm(X[0]), c, gid) for X, c, gid in raw]   # PS1
    elif bank == "gasid":
        from gas_multichannel import load_gas_mc
        raw = load_gas_mc()                               # (X(8,W), gas, gid, unit)
        mc = [(X, c, gid) for X, c, gid, _ in raw]
        lean_raw = [(_znorm(X[1]), c, gid) for X, c, gid, _ in raw]  # ch 1
    else:
        raise ValueError(bank)
    for X, _, _ in mc:
        assert X.ndim == 2 and X.shape[0] >= 2, f"bad window shape {np.shape(X)}"
    return mc, lean_raw


def coh_feats(mc, nper, n_bands):
    t0 = time.time()
    F = []
    for X, _, _ in mc:
        row = []
        n_ch = X.shape[0]
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                a = np.asarray(X[i], float); b = np.asarray(X[j], float)
                n = min(len(a), len(b))
                if n < 8:
                    row += [0.0] * n_bands
                    continue
                npe = min(nper, n)
                _, Cxy = coherence(a[:n], b[:n], nperseg=npe, noverlap=npe // 2)
                row += [float(np.mean(c)) for c in np.array_split(Cxy, n_bands)]
        F.append(row)
    return np.array(F), (time.time() - t0) * 1000 / len(mc)


def logo_folds(F, y, g):
    accs = {}
    for r in np.unique(g):
        tr = g != r
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(F[tr], y[tr])
        accs[str(r)] = float((clf.predict(F[~tr]) == y[~tr]).mean())
    return accs


def _append_rows(bank, tag, accs, ms, chance):
    new = not os.path.exists(CSV)
    with open(CSV, "a") as f:
        if new:
            f.write("bank,config,rec,acc,ms_per_win,chance\n")
        for r, a in accs.items():
            f.write(f"{bank},{tag},{r},{a:.4f},{ms:.2f},{chance:.4f}\n")


def _done_tags(bank):
    if not os.path.exists(CSV):
        return set()
    import csv as _csv
    return {r["config"] for r in _csv.DictReader(open(CSV)) if r["bank"] == bank}


def run_lean(bank):
    if "lean" in _done_tags(bank):
        print(f"{bank} lean: checkpointed, skip", flush=True); return
    mc, lean_raw = load_bank(bank)
    y = np.array([r[1] for r in mc]); g = np.array([r[2] for r in mc])
    chance = 1 / len(set(y))
    t0 = time.time()
    F = lean_baseline(lean_raw)
    ms = (time.time() - t0) * 1000 / len(mc)
    accs = logo_folds(F, y, g)
    _append_rows(bank, "lean", accs, ms, chance)
    print(f"{bank} lean: LOGO {np.mean(list(accs.values())):.3f} @ {ms:.1f} ms/win "
          f"({len(mc)} win, {len(np.unique(g))} recs, chance {chance:.3f})", flush=True)


def run_config(bank, nper, n_bands):
    tags = {f"c{nper}b{n_bands}_alone", f"c{nper}b{n_bands}_aug"}
    if tags <= _done_tags(bank):
        print(f"{bank} c{nper}b{n_bands}: checkpointed, skip", flush=True); return
    mc, lean_raw = load_bank(bank)
    y = np.array([r[1] for r in mc]); g = np.array([r[2] for r in mc])
    chance = 1 / len(set(y))
    Fc, ms = coh_feats(mc, nper, n_bands)
    Fl = lean_baseline(lean_raw)
    for kind, F in (("alone", Fc), ("aug", np.hstack([Fl, Fc]))):
        tag = f"c{nper}b{n_bands}_{kind}"
        if tag in _done_tags(bank):
            continue
        accs = logo_folds(F, y, g)
        _append_rows(bank, tag, accs, ms, chance)
        print(f"{bank} {tag}: LOGO {np.mean(list(accs.values())):.3f} "
              f"@ {ms:.1f} ms/win ({F.shape[1]} feats)", flush=True)


# ---------------------------------------------------------------------- report
def _bootstrap_ci(d, seed=0):
    m = np.random.default_rng(seed).choice(d, (10000, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report():
    import csv as _csv
    rows = list(_csv.DictReader(open(CSV)))
    out = {}
    for bank in BANKS:
        cfgs, chance = {}, None
        for r in rows:
            if r["bank"] != bank:
                continue
            cfgs.setdefault(r["config"], {})[r["rec"]] = (float(r["acc"]),
                                                          float(r["ms_per_win"]))
            chance = float(r["chance"])
        if "lean" not in cfgs:
            continue
        lean = cfgs["lean"]
        lean_acc = float(np.mean([v[0] for v in lean.values()]))
        print(f"\n===== {bank.upper()} (lean {lean_acc:.3f} @ "
              f"{list(lean.values())[0][1]:.1f} ms/win, chance {chance:.3f}) =====",
              flush=True)
        mean_of = lambda c: float(np.mean([v[0] for v in cfgs[c].values()]))
        aug_cfgs = sorted(c for c in cfgs if c.endswith("_aug"))
        alone_cfgs = sorted(c for c in cfgs if c.endswith("_alone"))
        for c in alone_cfgs + aug_cfgs:
            print(f"  {c}: {mean_of(c):.3f} @ {list(cfgs[c].values())[0][1]:.0f} ms/win",
                  flush=True)

        res = {"bank": bank, "chance": chance, "lean": lean_acc,
               "lean_ms": list(lean.values())[0][1]}
        for label, pool in (("aug", aug_cfgs), ("alone", alone_cfgs)):
            if not pool:
                continue
            best = max(pool, key=mean_of)
            recs = sorted(set(lean) & set(cfgs[best]))
            d = np.array([cfgs[best][r][0] - lean[r][0] for r in recs])
            lo, hi = _bootstrap_ci(d)
            acc = mean_of(best)
            acc_lo, _ = _bootstrap_ci(np.array([cfgs[best][r][0] for r in recs]))
            ms = list(cfgs[best].values())[0][1]
            res[label] = {"best": best, "acc": acc, "acc_ci_lo": acc_lo,
                          "paired": float(d.mean()), "ci": (lo, hi),
                          "ms": ms, "cost_x": ms / max(res["lean_ms"], 1e-9)}
            print(f"  BEST {label} = {best}: {acc:.3f}  paired {label}-lean "
                  f"{d.mean():+.3f} CI [{lo:+.3f},{hi:+.3f}]", flush=True)
        # stability: jackknife over folds -- same aug config stays best?
        if aug_cfgs:
            best = res["aug"]["best"]
            recs = sorted(set.intersection(*(set(cfgs[c]) for c in aug_cfgs)))
            wins = sum(max(aug_cfgs, key=lambda c: np.mean(
                [cfgs[c][r][0] for r in recs if r != drop])) == best
                       for drop in recs)
            res["stability_top1"] = wins / len(recs)
            print(f"  STABILITY (config jackknife): {wins}/{len(recs)} = "
                  f"{wins/len(recs):.2f} "
                  f"({'STABLE' if wins/len(recs) >= 0.5 else 'UNSTABLE -> do not trust'})",
                  flush=True)
        # frozen decision rule (COH-FAIR-PREREG)
        a = res.get("aug")
        confirmed = bool(a and a["ci"][0] > 0 and a["acc"] > chance + 0.05)
        gated = bool(a and a["acc_ci_lo"] > chance)
        res["chance_gate_aug_ci_lo>chance"] = gated
        res["verdict"] = "FLAG CONFIRMED" if confirmed else "FLAG NOT CONFIRMED"
        print(f"  CHANCE-GATE: best-aug acc CI-lo {a['acc_ci_lo']:.3f} "
              f"{'>' if gated else '<='} chance {chance:.3f}", flush=True)
        print(f"  VERDICT: {res['verdict']}", flush=True)
        out[bank] = res
    tip = log_receipt("COH-FAIR", {"grid": GRID, "results": out})
    print(f"\nledger tip = {tip}", flush=True)


def prereg():
    tip = log_receipt("COH-FAIR-PREREG", {
        "grid": GRID, "banks": list(BANKS),
        "loaders": "identical calls to coherence_rqa_screen.py "
                   "(load_hyd_mc('cooler') lean=PS1; load_gas_mc() lean=ch1)",
        "primary": "best-aug (grid best-wins, selection favours coherence) "
                   "paired aug-lean bootstrap CI-lo > 0 AND aug acc > chance+0.05 "
                   "-> FLAG CONFIRMED",
        "secondary": "best-alone vs lean, family-frontier semantics (RQA-fair analog)",
        "receipts": "paired CI (10k bootstrap, seed 0) + config-jackknife "
                    "stability + chance-gate (acc CI-lo > chance) + this ledger "
                    "entry BEFORE any readout; no perm-p (A4/rqa_fair methodology)",
        "model": "LOGO over recordings, RF(150) seed 0, StandardScaler"})
    print(f"prereg ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    elif sys.argv[1] == "prereg":
        prereg()
    elif sys.argv[1] == "all":
        for bank in BANKS:
            run_lean(bank)
            for nper, nb in GRID:
                run_config(bank, nper, nb)
    elif sys.argv[2] == "lean":
        run_lean(sys.argv[1])
    else:
        run_config(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
