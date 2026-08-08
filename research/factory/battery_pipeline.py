"""Physics #12 — DEVICE-TRANSFER GENERALIZATION on fresh battery EIS.

Closes the honest gap MOX-DRIFT (#11) left open: MOX had ONE device, so only
time-drift invariance was testable, never DEVICE transfer. Here the split IS the
device-transfer test: leave-one-CELL-out (LOCO) over 22 NMC/graphite cells from
a FRESH dataset (OSF j2sn4, DOI 10.17605/OSF.IO/J2SN4; Data-in-Brief 2025/26).
A held-out cell is a device the model never saw -> clearing chance there = a
cell-transferable signature that survives cell-to-cell impedance variation.

Input modality is NEW for us: electrochemical impedance spectra (EIS), not cycle
curves -> also a modality-generalization of the CALCE-soh cell-LOCO win.

Robustness / "nothing lost" design (unattended, reboot-safe):
  download() : recursive OSF mirror, skips files already present at matching size
               -> an interruption loses at most the single in-flight file.
  build()    : per-cell feature cache cache/battery/<cell>.npz, skips done cells
               -> a reboot mid-build loses at most the current cell.
  readout()  : runs ONLY if EIS auto-detect AND labels parsed; else writes
               NEEDS_REVIEW (never fabricates). Writes DONE on success.
  main()     : no-ops if DONE exists; otherwise download -> build -> readout,
               each stage idempotent. A launchd watchdog (RunAtLoad + 30 min)
               re-fires this after any crash/reboot until DONE.

Discipline: FROZEN_SPEC is committed to the hash-chained ledger + git BEFORE any
readout. PRIMARY label = aging-state (fresh vs aged), filename-derived (always
parseable -> deterministic unattended verdict); SOH-tertile is SECONDARY/
exploratory if the capacity file parses. group = CELL (LOCO) = device transfer.
Single run, no re-selection; SOC/day = within-cell nuisance.

  freeze : append FROZEN_SPEC (+sha) to LEDGER, print tip (do + commit first)
  download / build / readout / run(all)
"""
import glob, hashlib, json, os, re, sys, time, urllib.request
import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from feature_forge import programs, run_prog, lean_baseline
from receipt_ledger import log_receipt

ROOT = "<local-path>/signalmap"
DATA = f"{ROOT}/data/battery_eis"
CACHEDIR = f"{ROOT}/research/factory/cache/battery"
MARK = f"{ROOT}/research/factory/cache/battery_markers"
OSF_ROOT = "https://api.osf.io/v2/nodes/j2sn4/files/osfstorage/"
W_EIS, C_GATE = 64, 50           # W_EIS = resampled spectrum length
FRESH_DAYS, AGED_DAYS = {0, 10, 20}, {40, 70, 90}

FROZEN_SPEC = {
    "physics": "#12 device-transfer generalization (battery EIS, fresh cells)",
    "dataset": "OSF j2sn4 (DOI 10.17605/OSF.IO/J2SN4); 22 NMC/graphite cells, EIS @5 SOC",
    "closes_gap": ("MOX-DRIFT #11 had 1 device -> only time-drift testable; here "
                   "leave-one-CELL-out = genuine device transfer, on NEW EIS modality"),
    "window": ("per EIS CSV auto-detect freq + Re/Im (or |Z|); X = |Z| vs freq, "
               "sorted, interp to W=64; SOC & day = within-cell nuisance"),
    "label_primary": ("aging-state 2-class: fresh day in {0,10,20} vs aged {40,70,90}; "
                      "filename-derived (deterministic, unattended-safe)"),
    "label_secondary_exploratory": "SOH-tertile from capacity file, if it parses",
    "grammar": "multichannel not needed (single EIS spectrum); unary grammar v2.1 dedup",
    "split_primary": ("leave-one-CELL-out (LOCO over ~22 cells) = DEVICE TRANSFER; "
                      "per-fold F-stat prescreen top-5; RF(150); leak-free"),
    "baseline": "lean_baseline (perm-entropy + psd-slope on z-normed spectrum)",
    "primary_metric": ("forge LOCO accuracy vs chance 0.5; champion = chance-gated "
                       "paired forge-lean CI; group-perm-p on lean; shuffle-null"),
    "verdict_rule": ("DEVICE-TRANSFER PASS iff forge OR lean CI-lo > 0.5 AND "
                     "perm-p < 0.05 AND shuffle-null ~ chance; else honest NULL "
                     "(device-transfer boundary)"),
    "no_reselection": True,
}


# ---------- download (resumable OSF mirror) ----------
def _get_json(url, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "signalmap/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if a == tries - 1:
                raise
            time.sleep(3 * (a + 1))


def _dl_file(url, dest, size, tries=4):
    if os.path.exists(dest) and (size is None or abs(os.path.getsize(dest) - size) <= 2):
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "signalmap/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 16)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, dest)
            return "get"
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(3 * (a + 1))


def download():
    stack = [(OSF_ROOT, "")]
    got = skip = 0
    while stack:
        url, rel = stack.pop()
        while url:
            d = _get_json(url)
            for it in d.get("data", []):
                a = it["attributes"]
                child_rel = os.path.join(rel, a["name"])
                if a["kind"] == "folder":
                    stack.append((it["relationships"]["files"]["links"]["related"]["href"], child_rel))
                else:
                    dl = it["links"].get("download")
                    if dl:
                        r = _dl_file(dl, os.path.join(DATA, child_rel), a.get("size"))
                        got += (r == "get"); skip += (r == "skip")
            url = d.get("links", {}).get("next")
    print(f"download: {got} fetched, {skip} already present", flush=True)
    _mark("DOWNLOADED")


# ---------- EIS parsing (auto-detect) ----------
def _read_eis(fp):
    """Return |Z| spectrum (1D, len W_EIS) or None if undetectable."""
    for sep in (",", ";", r"\s+", "\t"):
        try:
            df = pd.read_csv(fp, sep=sep, engine="python" if sep == r"\s+" else "c")
            if df.shape[1] >= 2:
                break
        except Exception:
            df = None
    if df is None or df.shape[1] < 2:
        return None
    cols = {str(c).lower(): c for c in df.columns}
    def find(*keys):
        for k in keys:
            for lc, c in cols.items():
                if k in lc:
                    return c
        return None
    fcol = find("freq", "hz")
    zabs = find("|z|", "zmod", "magn")
    re_c = find("rez", "zre", "z're", "real", "z'")
    im_c = find("imz", "zim", "z''", "imag", 'z"')
    if zabs is not None:
        z = pd.to_numeric(df[zabs], errors="coerce").to_numpy(float)
    elif re_c is not None and im_c is not None:
        rr = pd.to_numeric(df[re_c], errors="coerce").to_numpy(float)
        ii = pd.to_numeric(df[im_c], errors="coerce").to_numpy(float)
        z = np.sqrt(rr ** 2 + ii ** 2)
    else:
        return None
    if fcol is not None:
        f = pd.to_numeric(df[fcol], errors="coerce").to_numpy(float)
        m = np.isfinite(f) & np.isfinite(z)
        f, z = f[m], z[m]
        order = np.argsort(f)
        f, z = f[order], z[order]
    else:
        z = z[np.isfinite(z)]
    if len(z) < 8:
        return None
    t_old, t_new = np.linspace(0, 1, len(z)), np.linspace(0, 1, W_EIS)
    return np.interp(t_new, t_old, z).astype(np.float32)


def _cell_day(fp):
    b = os.path.basename(fp)
    cm = re.search(r"_S(\d+)_convert", b) or re.search(r"[/\\]S(\d+)[/\\]", fp)  # cell
    dm = re.search(r"_(\d+)d", b)                           # day #d
    return (int(cm.group(1)) if cm else None,
            int(dm.group(1)) if dm else None)


def _unzip():
    """Extract EIS csv.zip (idempotent). EIS data ships bundled, not as loose
    OSF files -> extract before the loader can glob it."""
    import zipfile
    for zp in glob.glob(f"{DATA}/**/csv.zip", recursive=True):
        out = os.path.join(os.path.dirname(zp), "extracted")
        if os.path.isdir(out) and glob.glob(f"{out}/**/*.csv", recursive=True):
            continue
        os.makedirs(out, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                if m.startswith("__MACOSX") or os.path.basename(m).startswith("._"):
                    continue
                z.extract(m, out)
        print(f"unzip: {zp} -> {out}", flush=True)


def build():
    os.makedirs(CACHEDIR, exist_ok=True)
    _unzip()
    files = [f for f in glob.glob(f"{DATA}/**/*.csv", recursive=True)
             if ("convert" in f.lower() or "acr" in os.path.basename(f).lower())
             and "__MACOSX" not in f and not os.path.basename(f).startswith("._")]
    per_cell = {}
    for fp in files:
        cell, day = _cell_day(fp)
        if cell is None or day is None:
            continue
        per_cell.setdefault(cell, []).append((fp, day))
    n_ok = n_skip = n_drop = 0
    for cell, items in sorted(per_cell.items()):
        out = f"{CACHEDIR}/cell{cell}.npz"
        if os.path.exists(out):
            n_skip += 1
            continue
        Xs, ds = [], []
        for fp, day in items:
            z = _read_eis(fp)
            if z is not None:
                Xs.append(z); ds.append(day)
            else:
                n_drop += 1   # spectrum failed to parse -> account for it LOUDLY
        if Xs:
            np.savez_compressed(out, X=np.array(Xs), day=np.array(ds), cell=cell)
            n_ok += 1
    print(f"build: {n_ok} cells cached, {n_skip} already done, "
          f"{len(per_cell)} cells seen, {len(files)} eis csv, {n_drop} unparseable "
          f"(dropped){' <-- REVIEW: silent EIS loss' if n_drop else ''}", flush=True)
    _mark("BUILT")


# ---------- readout ----------
def _ci(a, seed=0):
    a = np.array(a, float); rng = np.random.default_rng(seed)
    m = rng.choice(a, (10000, len(a)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _rf():
    return make_pipeline(StandardScaler(), RandomForestClassifier(150, random_state=0, n_jobs=-1))


def _fstat(F, y):
    from sklearn.feature_selection import f_classif
    return np.nan_to_num(f_classif(F, y)[0], nan=0.0)


def _znorm(s):
    s = detrend(np.ascontiguousarray(s, float))
    return (s - s.mean()) / (s.std() + 1e-12)


def _load_bank():
    raw = []
    for fp in sorted(glob.glob(f"{CACHEDIR}/cell*.npz")):
        z = np.load(fp)
        cell = int(z["cell"])
        for i in range(len(z["X"])):
            day = int(z["day"][i])
            if day in FRESH_DAYS:
                lab = "fresh"
            elif day in AGED_DAYS:
                lab = "aged"
            else:
                continue
            raw.append((z["X"][i], lab, cell))
    return raw


def _loco(raw, F, names, lean_F, y, g, tag, k=5):
    cells = np.unique(g); chance = 1 / len(set(y))
    folds, lean_folds, sel = {}, {}, {}
    for h in cells:
        tr = g != h
        if len(set(y[tr])) < 2 or len(set(y[~tr])) < 1 or (~tr).sum() == 0:
            continue
        s = list(np.argsort(-_fstat(F[tr], y[tr]))[:k])
        c = _rf(); c.fit(F[tr][:, s], y[tr])
        folds[h] = float((c.predict(F[~tr][:, s]) == y[~tr]).mean())
        sel[h] = [names[j] for j in s]
        c = _rf(); c.fit(lean_F[tr], y[tr])
        lean_folds[h] = float((c.predict(lean_F[~tr]) == y[~tr]).mean())
    fa = list(folds.values()); la = list(lean_folds.values())
    lo, hi = _ci(fa); llo, lhi = _ci(la)
    d = np.array([folds[r] - lean_folds[r] for r in folds])
    rng = np.random.default_rng(0)
    dm = rng.choice(d, (10000, len(d)), replace=True).mean(1)
    dlo, dhi = float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))
    tops = [s[0] for s in sel.values() if s]
    t1 = max(tops.count(t) for t in set(tops)) / len(sel) if tops else 0.0
    best_lo = max(lo, llo)
    champ = "NULL (nobody clears chance)" if best_lo <= chance else (
        "forge" if dlo > 0 else "lean" if dhi < 0 else "tie->lean")
    print(f"\n===== {tag} (LOCO, {len(folds)} cells, chance {chance:.3f}) =====", flush=True)
    print(f"FORGE: {np.mean(fa):.3f} CI[{lo:.3f},{hi:.3f}]  lean: {np.mean(la):.3f} CI[{llo:.3f},{lhi:.3f}]", flush=True)
    print(f"PAIRED forge-lean: {d.mean():+.3f} CI[{dlo:+.3f},{dhi:+.3f}]", flush=True)
    print(f"STABILITY top1 {t1:.2f}  CHAMPION(chance-gated): {champ}", flush=True)
    return {"tag": tag, "forge": float(np.mean(fa)), "lean": float(np.mean(la)),
            "ci": [lo, hi], "lean_ci": [llo, lhi], "paired_ci": [dlo, dhi],
            "top1_freq": t1, "champion": champ, "chance": chance, "n_cells": len(folds)}


def _bank_hash(raw):
    """Content hash of the bank's spectra (X only; labels don't affect features)."""
    h = hashlib.sha256()
    for X, _, _ in raw:
        h.update(np.ascontiguousarray(X, np.float32).tobytes())
    return h.hexdigest()[:16]


def _feature_matrix(raw, tag="aging"):
    """Build (+cache) the forge feature matrix for the bank. Cache key = tag
    (filename) AND content hash of X -> len(raw) alone can collide two banks of
    equal length OR silently return a stale matrix when X content changed."""
    fcache = f"{CACHEDIR}/F_{tag}.npz"
    bh = _bank_hash(raw)
    if os.path.exists(fcache):
        z = np.load(fcache, allow_pickle=True)
        cached_h = str(z["bank_hash"]) if "bank_hash" in z.files else None
        if len(z["F"]) == len(raw) and cached_h == bh:
            return z["F"], list(z["names"])
    progs = programs()
    F = np.nan_to_num(np.array([[run_prog(p, _znorm(X)) for p in progs] for X, _, _ in raw]))
    names = [p[0] for p in progs]
    np.savez_compressed(fcache, F=F, names=np.array(names, dtype=object), bank_hash=bh)
    return F, names


def _forge_loco_mean(F, y, g, k=5):
    """forge LOCO accuracy mean only (fast path for permutation test)."""
    accs = []
    for h in np.unique(g):
        tr = g != h
        if len(set(y[tr])) < 2 or (~tr).sum() == 0:
            continue
        s = list(np.argsort(-_fstat(F[tr], y[tr]))[:k])
        c = _rf(); c.fit(F[tr][:, s], y[tr])
        accs.append((c.predict(F[~tr][:, s]) == y[~tr]).mean())
    return float(np.mean(accs))


def permtest(M=200):
    """Within-cell label-permutation p for forge LOCO (frozen verdict needs
    perm-p<0.05). p = (#{perm >= observed}+1)/(M+1)."""
    raw = _load_bank()
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    F, _ = _feature_matrix(raw)
    obs = _forge_loco_mean(F, y, g)
    rng = np.random.default_rng(0); hits = 0
    for m in range(M):
        ysh = y.copy()
        for c in np.unique(g):
            idx = np.where(g == c)[0]; ysh[idx] = rng.permutation(y[idx])
        if _forge_loco_mean(F, ysh, g) >= obs:
            hits += 1
        if (m + 1) % 25 == 0:
            print(f"  perm {m+1}/{M}: hits={hits}", flush=True)
    p = (hits + 1) / (M + 1)
    print(f"PERM-P (forge LOCO, within-cell, M={M}): obs={obs:.3f}, p={p:.4f}", flush=True)
    log_receipt("BATTERY-TRANSFER-PERMP", {"obs": obs, "p": p, "M": M,
                                           "spec_sha256": _sha(FROZEN_SPEC)})
    return p


FROZEN_SPEC_SOH = {
    "physics": "#12 SECONDARY — SOH-tertile from EIS, device-transfer (battery)",
    "parent": "extends BATTERY-TRANSFER-PREREG (spec 170dcad4); finer continuous "
              "SOH target instead of coarse fresh/aged; comparable to CALCE-soh win",
    "dataset": "OSF j2sn4; same EIS bank (935 spectra, 24 cells)",
    "label": ("SOH = C_discharge(cell,day)/C(cell,0) from FCC 'Discharge capacity' "
              "(duplicate day-rows averaged, deterministic); tertile lo/mid/hi via "
              "GLOBAL 33/67 quantiles over all (cell, EIS-day) SOH"),
    "window": "same |Z| vs freq spectra as primary; group = CELL",
    "split": "leave-one-CELL-out (device transfer); F-stat prescreen top-5; RF(150)",
    "primary_metric": ("forge LOCO vs chance 0.333; chance-gated paired CI champion; "
                       "within-cell perm-p (M=200); shuffle-null"),
    "verdict_rule": ("PASS iff forge OR lean CI-lo > 0.333 AND perm-p < 0.05 AND "
                     "shuffle-null ~ chance; else honest NULL"),
    "no_reselection": True,
}


def _soh_map():
    """SOH = C_discharge(cell, day) / C_discharge(cell, 0), from the RAW-capacity
    block ONLY. The FCC sheet stacks two blocks per column: raw discharge capacity
    (Ah, days 0..90) then a pre-normalized SOH block (<=1, same days, after a "SOH"
    marker row). We use only the raw block and normalize ourselves -> single source
    of truth. The old code did groupby("day").mean() over BOTH blocks, averaging
    raw-Ah (~5.4) with normalized-SOH (~1.0); that is only *accidentally* correct
    (the common factor cancels) and silently corrupts any (cell,day) whose two
    blocks have mismatched NaN masks. Fails loud on unexpected structure."""
    import glob as _g
    f = [x for x in _g.glob(f"{DATA}/**/*.xlsx", recursive=True)
         if "FCC" in os.path.basename(x)][0]
    d = pd.read_excel(f, sheet_name="Discharge capacity", header=1)
    d = d.rename(columns={d.columns[0]: "day"})
    d["day"] = pd.to_numeric(d["day"], errors="coerce")
    d = d.dropna(subset=["day"]).reset_index(drop=True)
    # Keep only the FIRST block: stop at the day-reset that begins the SOH block.
    days = d["day"].tolist()
    end = next((i for i in range(1, len(days)) if days[i] <= days[i - 1]), len(days))
    raw = d.iloc[:end]
    cells = [c for c in raw.columns if str(c).startswith("S") and str(c)[1:].isdigit()]
    if 0 not in set(raw["day"]) or raw["day"].duplicated().any():
        raise ValueError(f"_soh_map: unexpected FCC raw block days={list(raw['day'])}")
    C = raw.set_index("day")[cells].astype(float)
    if not (C.loc[0] > 2).any():   # raw block must be Ah-scale (~5), not the <=1 SOH block
        raise ValueError("_soh_map: first block is not raw-capacity scale (day-0 <= 2)")
    SOH = C.div(C.loc[0])
    return {(int(c[1:]), int(day)): float(SOH.loc[day, c])
            for day in SOH.index for c in cells
            if np.isfinite(SOH.loc[day, c])}


def _load_bank_soh():
    m = _soh_map()
    items = []
    for fp in sorted(glob.glob(f"{CACHEDIR}/cell*.npz")):
        z = np.load(fp); cell = int(z["cell"])
        for i in range(len(z["X"])):
            day = int(z["day"][i]); s = m.get((cell, day))
            if s is not None and np.isfinite(s):
                items.append((z["X"][i], s, cell))
    sohs = np.array([it[1] for it in items])
    q1, q2 = np.nanquantile(sohs, [1 / 3, 2 / 3])
    def lab(s):
        return "lo" if s <= q1 else "mid" if s <= q2 else "hi"
    return [(X, lab(s), c) for X, s, c in items], (float(q1), float(q2))


def readout_soh():
    raw, cuts = _load_bank_soh()
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    print(f"SOH bank: {len(raw)} spectra, {len(set(g))} cells, cuts={cuts}, "
          f"per-class {[(c, int((y==c).sum())) for c in sorted(set(y))]}", flush=True)
    F, names = _feature_matrix(raw, tag="soh")
    lean_F = lean_baseline([(_znorm(X), c, gid) for X, c, gid in raw])
    real = _loco(raw, F, names, lean_F, y, g, "BATTERY-SOH-tertile")
    rng = np.random.default_rng(0); ysh = y.copy()
    for c in np.unique(g):
        idx = np.where(g == c)[0]; ysh[idx] = rng.permutation(y[idx])
    null = _loco(raw, F, names, lean_F, ysh, g, "BATTERY-SOH-SHUFFLE-NULL")
    # within-cell perm-p
    obs = _forge_loco_mean(F, y, g); hits = 0; M = 200
    for m2 in range(M):
        ys = y.copy()
        for c in np.unique(g):
            idx = np.where(g == c)[0]; ys[idx] = rng.permutation(y[idx])
        if _forge_loco_mean(F, ys, g) >= obs:
            hits += 1
    p = (hits + 1) / (M + 1)
    print(f"PERM-P (SOH forge LOCO, M={M}): obs={obs:.3f}, p={p:.4f}", flush=True)
    log_receipt("BATTERY-SOH", {"real": real, "shuffle_null": null, "perm_p": p,
                                "cuts": cuts, "spec_sha256": _sha(FROZEN_SPEC_SOH)})


def freeze_soh():
    tip = log_receipt("BATTERY-SOH-PREREG",
                      {**FROZEN_SPEC_SOH, "spec_sha256": _sha(FROZEN_SPEC_SOH)})
    print("FROZEN SOH. ledger tip:", tip, "\nspec_sha256:", _sha(FROZEN_SPEC_SOH))


def readout():
    raw = _load_bank()
    if len(raw) < 20 or len(set(r[2] for r in raw)) < 4 or len(set(r[1] for r in raw)) < 2:
        _mark("NEEDS_REVIEW", f"bank too small/degenerate: {len(raw)} windows, "
              f"{len(set(r[2] for r in raw))} cells, labels {set(r[1] for r in raw)}")
        print("NEEDS_REVIEW: bank degenerate; not running readout", flush=True)
        return
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    print(f"bank: {len(raw)} spectra, {len(set(g))} cells, "
          f"per-class {[int((y==c).sum()) for c in sorted(set(y))]}", flush=True)
    progs = programs()
    F = np.array([[run_prog(p, _znorm(X)) for p in progs] for X, _, _ in raw])
    F = np.nan_to_num(F)
    names = [p[0] for p in progs]
    lean_F = lean_baseline([(_znorm(X), c, gid) for X, c, gid in raw])
    real = _loco(raw, F, names, lean_F, y, g, "BATTERY-TRANSFER-aging")
    rng = np.random.default_rng(0)
    ysh = y.copy()
    for c in np.unique(g):
        idx = np.where(g == c)[0]; ysh[idx] = rng.permutation(y[idx])
    null = _loco(raw, F, names, lean_F, ysh, g, "BATTERY-TRANSFER-SHUFFLE-NULL")
    log_receipt("BATTERY-TRANSFER", {"real": real, "shuffle_null": null,
                                     "spec_sha256": _sha(FROZEN_SPEC)})
    _mark("DONE")


# ---------- markers / orchestration ----------
def _mark(name, note=""):
    os.makedirs(MARK, exist_ok=True)
    with open(f"{MARK}/{name}", "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ") + " " + note + "\n")


def _has(name):
    return os.path.exists(f"{MARK}/{name}")


def _sha(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def freeze():
    tip = log_receipt("BATTERY-TRANSFER-PREREG", {**FROZEN_SPEC, "spec_sha256": _sha(FROZEN_SPEC)})
    print("FROZEN. ledger tip:", tip, "\nspec_sha256:", _sha(FROZEN_SPEC))


def main():
    if _has("DONE"):
        print("DONE already; no-op", flush=True); return
    if _has("NEEDS_REVIEW"):
        print("NEEDS_REVIEW set; halting for manual loader check", flush=True); return
    if not _has("DOWNLOADED"):
        download()
    build()
    readout()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"freeze": freeze, "download": download, "build": build,
     "readout": readout, "permtest": permtest,
     "freeze_soh": freeze_soh, "readout_soh": readout_soh,
     "run": main}.get(cmd, main)()
