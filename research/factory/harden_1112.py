"""Confirmation / hardening pass for Physics #11 (MOX-DRIFT) and #12 (BATTERY-
TRANSFER + SOH) — push toward maximal confidence that data and results are sound.

Beyond the per-bank receipts already logged (LODO, chance-gated CI, within-group
shuffle-null, within-group perm-p, selection stability), this adds the checks a
mixed-label LODO bank still needs, all label-agnostic or model-swap (they can
only LOWER confidence — the honest direction, never re-selection):

  PROVENANCE   SHA-256 of the raw dataset files -> ledger (tamper-evident freeze;
               confirms the exact bytes the banks were derived from).
  HARNESS      audit.synthetic_controls(): pure noise -> chance, planted -> found
               (the pipeline itself neither invents nor loses signal).
  WINDOW-PROV  exact + near-identical (|r|>0.999) windows shared across DIFFERENT
               groups -> must be 0 (the leak LOGO/LODO cannot catch: duplicated
               measurements split across folds). Label-agnostic.
  LINEAR-SWAP  refit each bank's champion (per-fold F-stat top-5) with a LINEAR
               model (LogReg) -> a real signal survives the model swap.
  NOISE-PLACEBO rerun the WHOLE forge pipeline (feature build + prescreen + LODO)
               on Gaussian noise of matched shape & group structure -> chance.
               Stronger than label-shuffle: exercises feature build + selection.

Run: nice -n 19 ../../.venv-research/bin/python harden_1112.py
"""
import hashlib, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from receipt_ledger import log_receipt
import audit
import mox_drift as mx
import battery_pipeline as bt

R = {}


def _sha(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def provenance():
    files = {
        "mox/Dataset.zip": f"{mx.__dict__.get('MOX','')}".replace("/raw", "") + "/Dataset.zip",
        "battery/csv.zip": None, "battery/FCC.xlsx": None,
    }
    import glob
    csvzip = glob.glob(f"{bt.DATA}/**/csv.zip", recursive=True)
    fcc = [x for x in glob.glob(f"{bt.DATA}/**/*.xlsx", recursive=True) if "FCC" in os.path.basename(x)]
    paths = {"mox_Dataset.zip": "<local-path>/signalmap/data/mox/Dataset.zip",
             "battery_csv.zip": csvzip[0] if csvzip else None,
             "battery_FCC.xlsx": fcc[0] if fcc else None}
    prov = {}
    for name, p in paths.items():
        if p and os.path.exists(p):
            prov[name] = {"sha256": _sha(p), "bytes": os.path.getsize(p)}
            print(f"PROVENANCE {name}: {prov[name]['sha256'][:16]}... ({prov[name]['bytes']} B)", flush=True)
        else:
            prov[name] = {"sha256": None, "note": "missing"}
    R["provenance"] = prov


def window_provenance(tag, raw):
    """0 exact + 0 near-identical windows across different groups (leak test)."""
    g = np.array([r[2] for r in raw])
    hs, dup = {}, 0
    for i, (s, _, _) in enumerate(raw):
        h = hashlib.md5(np.round(np.asarray(s, float).ravel(), 6).tobytes()).hexdigest()
        if h in hs and hs[h] != g[i]:
            dup += 1
        hs.setdefault(h, g[i])
    X = np.array([np.asarray(s, float).ravel() for s, _, _ in raw])
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    C = (Z @ Z.T) / Z.shape[1]
    near = int(((np.abs(np.triu(C, 1)) > 0.999) & (g[:, None] != g[None, :])).sum())
    ok = dup == 0 and near == 0
    print(f"[{'PASS' if ok else 'FAIL'}] window-provenance[{tag}]: {dup} exact + {near} "
          f"near-identical (|r|>0.999) across different groups (must be 0)", flush=True)
    R.setdefault(tag, {})["window_provenance"] = {"exact": dup, "near": near, "pass": ok}
    return ok


def _linsvc():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def linear_swap(tag, F, y, g, chance, fstat, k=5):
    """Champion (per-fold F-stat top-k) refit with LINEAR model; LODO mean + CI."""
    accs = []
    for h in np.unique(g):
        tr = g != h
        if len(set(y[tr])) < 2 or (~tr).sum() == 0:
            continue
        s = list(np.argsort(-fstat(F[tr], y[tr]))[:k])
        c = _linsvc(); c.fit(F[tr][:, s], y[tr])
        accs.append(float((c.predict(F[~tr][:, s]) == y[~tr]).mean()))
    accs = np.array(accs)
    rng = np.random.default_rng(0)
    m = rng.choice(accs, (10000, len(accs)), replace=True).mean(1)
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    ok = lo > chance
    print(f"[{'PASS' if ok else 'WEAK'}] linear-swap[{tag}]: LogReg LODO {accs.mean():.3f} "
          f"CI[{lo:.3f},{hi:.3f}] (chance {chance:.3f}; survives model swap iff CI-lo>chance)", flush=True)
    R.setdefault(tag, {})["linear_swap"] = {"acc": float(accs.mean()), "ci": [lo, hi],
                                            "chance": chance, "pass": ok}
    return ok


def noise_placebo_mox(raw):
    """MOX MC pipeline on matched-shape Gaussian noise -> chance."""
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    rng = np.random.default_rng(1)
    noise = [(rng.standard_normal(X.shape).astype(np.float32), c, gid) for X, c, gid in raw]
    F, names = mx.build_features(noise, len(set(g)))
    r = mx.lodo(noise, F, names,
                mx.lean_baseline([(mx._znorm(X[0]), c, gid) for X, c, gid in noise]),
                y, g, "MOX-NOISE-PLACEBO")
    ok = r["ci"][0] <= r["chance"] + 0.02
    R.setdefault("mox", {})["noise_placebo"] = {"forge": r["mc_forge"], "ci": r["ci"],
                                                "chance": r["chance"], "pass": ok}
    print(f"[{'PASS' if ok else 'FAIL'}] noise-placebo[mox]: forge {r['mc_forge']:.3f} "
          f"CI{r['ci']} (must be ~chance {r['chance']:.3f})", flush=True)
    return ok


def noise_placebo_batt(raw):
    from feature_forge import programs, run_prog
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    rng = np.random.default_rng(1)
    noise = [(rng.standard_normal(len(X)).astype(np.float32), c, gid) for X, c, gid in raw]
    progs = programs()
    F = np.nan_to_num(np.array([[run_prog(p, bt._znorm(X)) for p in progs] for X, _, _ in noise]))
    names = [p[0] for p in progs]
    lean = bt.lean_baseline([(bt._znorm(X), c, gid) for X, c, gid in noise])
    r = bt._loco(noise, F, names, lean, y, g, "BATTERY-NOISE-PLACEBO")
    ok = r["ci"][0] <= r["chance"] + 0.02
    R.setdefault("battery", {})["noise_placebo"] = {"forge": r["forge"], "ci": r["ci"],
                                                    "chance": r["chance"], "pass": ok}
    print(f"[{'PASS' if ok else 'FAIL'}] noise-placebo[battery]: forge {r['forge']:.3f} "
          f"CI{r['ci']} (must be ~chance {r['chance']:.3f})", flush=True)
    return ok


if __name__ == "__main__":
    t0 = time.time()
    print("== PROVENANCE =="); provenance()
    print("\n== HARNESS SYNTHETIC CONTROLS =="); audit.synthetic_controls()
    R["synthetic_controls"] = {n: bool(p) for n, p, _ in audit.P}

    print("\n== #11 MOX-DRIFT ==")
    mox_raw = mx.load_mox_mc()
    my = np.array([r[1] for r in mox_raw]); mg = np.array([r[2] for r in mox_raw])
    window_provenance("mox", mox_raw)
    Fm, nm = mx.build_features(mox_raw, len(set(mg)))
    linear_swap("mox", Fm, my, mg, 1 / len(set(my)), mx._fstat)
    noise_placebo_mox(mox_raw)

    print("\n== #12 BATTERY-TRANSFER (aging) ==")
    bank = bt._load_bank()
    by = np.array([r[1] for r in bank]); bg = np.array([r[2] for r in bank])
    window_provenance("battery", bank)
    Fb, nb = bt._feature_matrix(bank, tag="aging")
    linear_swap("battery", Fb, by, bg, 1 / len(set(by)), bt._fstat)
    noise_placebo_batt(bank)

    n_fail = sum(1 for v in R.get("mox", {}).values() if isinstance(v, dict) and v.get("pass") is False)
    n_fail += sum(1 for v in R.get("battery", {}).values() if isinstance(v, dict) and v.get("pass") is False)
    n_fail += sum(1 for p in R["synthetic_controls"].values() if not p)
    R["n_fail"] = n_fail
    print(f"\n== HARDEN COMPLETE: {n_fail} failed checks [{time.time()-t0:.0f}s] ==", flush=True)
    log_receipt("HARDEN-1112", R)
