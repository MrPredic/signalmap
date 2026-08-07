"""signalmap distill — automatic per-domain feature distillation with a receipt.

USP core = the CAPACITY GATE: the search-space budget scales with the number of
recordings (budget = min(len(grammar), C * n_recordings), C=50 default). Neither
catch22 (a fixed 22) nor hctsa (7700+, ungated) does this. Below the collapse
zone (validated on ECN/MFPT/CWRU, see research/factory/DISTILL_DESIGN.md) the
distilled feature set generalises; above it, selection noise wins.

Pipeline:
  1. ingest a bank of recordings, window to 1024 (detrend + z-norm)
  2. enumerate the compositional grammar (pool o transform2 o transform1),
     order DETERMINISTICALLY by complexity (transform depth, then pool cost),
     cut at the capacity budget -> cheap programs are preferred, cut reproducible
  3. rank survivors by recording-level ANOVA-F, greedy-select inside the honest
     LOGO metric (kmax=5, +0.005 threshold)
  4. emit a human-readable gauntlet REPORT ("receipt") + a deployable spec.json

The grammar primitives are copied (minimally) from research/factory/feature_forge.py
so the shipped package stays self-contained (numpy-only for featurisation);
sklearn/scipy are imported lazily only for the distillation gauntlet itself.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

W = 1024  # window length (spec §2.1)

# ------------------------------------------------------------------ grammar
# series -> series transforms (copied from feature_forge grammar v2.1)
def t_id(x): return x
def t_diff(x): return np.diff(x)
def t_diff2(x): return np.diff(x, 2)
def t_abs(x): return np.abs(x)
def t_sign(x): return np.sign(x)
def t_rank(x): return np.argsort(np.argsort(x)).astype(float) / len(x)
def t_sq(x): return x * x
def t_tanh(x): return np.tanh(x)
def t_rollstd(x, w=16):
    c = np.cumsum(np.concatenate([[0.0], x])); c2 = np.cumsum(np.concatenate([[0.0], x * x]))
    n = len(x) - w + 1
    if n < 8: return x
    m = (c[w:] - c[:-w]) / w; m2 = (c2[w:] - c2[:-w]) / w
    return np.sqrt(np.maximum(m2 - m * m, 0.0))
def t_env(x):
    a = np.abs(x); k = np.ones(8) / 8.0
    return np.convolve(a, k, mode="valid")
def t_cumsum(x):
    c = np.cumsum(x - x.mean()); return c / (np.abs(c).max() + 1e-12)
def t_clip(x):
    lo, hi = np.quantile(x, [0.05, 0.95]); return np.clip(x, lo, hi)

TRANSFORMS: dict[str, Callable] = {
    "id": t_id, "diff": t_diff, "diff2": t_diff2, "abs": t_abs, "sign": t_sign,
    "rank": t_rank, "sq": t_sq, "tanh": t_tanh, "rollstd": t_rollstd,
    "env": t_env, "cumsum": t_cumsum, "clip": t_clip}

# series -> scalar pools
def p_std(x): return np.std(x)
def p_meanabs(x): return np.mean(np.abs(x))
def p_iqr90(x): return np.quantile(x, 0.95) - np.quantile(x, 0.05)
def p_hent(x):
    h, _ = np.histogram(x, bins=16); p = h / max(h.sum(), 1)
    p = p[p > 0]; return float(-(p * np.log(p)).sum())
def p_zcr(x):
    s = np.sign(x - np.median(x)); return float(np.mean(s[1:] != s[:-1]))
def p_acf1(x):
    v = x - x.mean(); d = (v * v).sum()
    return float((v[1:] * v[:-1]).sum() / d) if d > 0 else 0.0
def _runs(x):
    s = np.sign(x - np.median(x)); idx = np.flatnonzero(s[1:] != s[:-1])
    return np.diff(np.concatenate([[0], idx + 1, [len(s)]]))
def p_runmean(x): r = _runs(x); return float(r.mean())
def p_runcv(x): r = _runs(x); return float(r.std() / (r.mean() + 1e-12))
def p_peakcv(x):
    pk = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))
    if len(pk) < 3: return 0.0
    iv = np.diff(pk); return float(iv.std() / (iv.mean() + 1e-12))
def p_specratio(x):
    P = np.abs(np.fft.rfft(x)) ** 2; m = len(P) // 4
    return float(P[m:].sum() / (P[:m].sum() + 1e-12))
def p_speccent(x):
    P = np.abs(np.fft.rfft(x)) ** 2; f = np.arange(len(P))
    return float((f * P).sum() / (P.sum() + 1e-12) / len(P))
def p_crest(x):
    ma = np.mean(np.abs(x)); return float(np.max(np.abs(x)) / (ma + 1e-12))
def p_ordtrans(x):
    a, b, c = x[:-2], x[1:-1], x[2:]
    pat = (a < b).astype(int) * 2 + (b < c).astype(int)
    T = np.zeros((4, 4)); np.add.at(T, (pat[:-1], pat[1:]), 1.0)
    p = (T / max(T.sum(), 1)).ravel(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())
def p_lcross(x):
    up = np.flatnonzero((x[:-1] < 1.0) & (x[1:] >= 1.0))
    if len(up) < 3: return 0.0
    iv = np.diff(up); return float(iv.std() / (iv.mean() + 1e-12))
def p_acflag(x):
    v = x - x.mean(); d = (v * v).sum() + 1e-12
    ac = np.array([(v[k:] * v[:-k]).sum() / d for k in range(2, 65)])
    return float((np.argmax(ac) + 2) / 64.0)
def p_specflat(x):
    P = np.abs(np.fft.rfft(x)) ** 2 + 1e-20
    return float(np.exp(np.mean(np.log(P))) / P.mean())

POOLS: dict[str, Callable] = {
    "std": p_std, "meanabs": p_meanabs, "iqr90": p_iqr90, "hent": p_hent,
    "zcr": p_zcr, "acf1": p_acf1, "runmean": p_runmean, "runcv": p_runcv,
    "peakcv": p_peakcv, "specratio": p_specratio, "speccent": p_speccent,
    "crest": p_crest, "ordtrans": p_ordtrans, "lcross": p_lcross,
    "acflag": p_acflag, "specflat": p_specflat}

# ------------------------------------------------------- complexity ordering
# The gate must cut DETERMINISTICALLY and prefer cheap programs (spec §1):
# primary key = transform depth (# non-id transforms), secondary = pool cost,
# tertiary = transform cost, then name for a total, reproducible order.
_TRANSFORM_COST = {
    "id": 0, "diff": 1, "diff2": 1, "abs": 1, "sign": 1, "sq": 1, "tanh": 1,
    "clip": 2, "rank": 2, "rollstd": 2, "env": 2, "cumsum": 2}
_POOL_COST = {  # 1 = O(n) reduction, 2 = sort/hist, 3 = fft / lag-loop
    "std": 1, "meanabs": 1, "acf1": 1, "zcr": 1, "crest": 1, "runmean": 1,
    "runcv": 1, "peakcv": 1, "lcross": 1, "iqr90": 2, "hent": 2, "rank": 2,
    "ordtrans": 2, "specratio": 3, "speccent": 3, "specflat": 3, "acflag": 3}


@dataclass(frozen=True)
class Program:
    """A distillable feature: name = pool(t2(t1(x)))."""
    name: str
    t1: str
    t2: str
    pool: str

    def complexity(self) -> tuple:
        depth = (self.t1 != "id") + (self.t2 != "id")
        tcost = _TRANSFORM_COST[self.t1] + _TRANSFORM_COST[self.t2]
        return (depth, _POOL_COST[self.pool], tcost, self.name)

    def __call__(self, s: np.ndarray) -> float:
        try:
            v = POOLS[self.pool](TRANSFORMS[self.t2](TRANSFORMS[self.t1](s)))
            return float(v) if np.isfinite(v) else 0.0
        except Exception:
            return 0.0


def enumerate_programs() -> list[Program]:
    """Full grammar, ordered by complexity (cheap first). id is canonical only
    as the inner slot (matches feature_forge canonicalisation)."""
    out: list[Program] = []
    for n1 in TRANSFORMS:
        for n2 in TRANSFORMS:
            if n2 == "id" and n1 != "id":
                continue
            for npn in POOLS:
                out.append(Program(f"{npn}({n2}({n1}(x)))", n1, n2, npn))
    out.sort(key=lambda p: p.complexity())
    return out


def budget_for(n_recordings: int, grammar_size: int, C: int = 50) -> int:
    """Capacity gate (spec §1): budget = min(len(grammar), C * n_recordings)."""
    return int(min(grammar_size, C * max(n_recordings, 0)))


def gate(programs: list[Program], n_recordings: int, C: int = 50) -> list[Program]:
    """Cut the complexity-ordered grammar at the capacity budget. Deterministic:
    the same (grammar, n_recordings, C) always yields the same prefix, and the
    cheapest programs survive."""
    return programs[:budget_for(n_recordings, len(programs), C)]


# ---------------------------------------------------------------- bank ingest
def _mat_signal(m: dict) -> np.ndarray:
    keys = [k for k in m if not k.startswith("__")]
    de = [k for k in keys if k.endswith("DE_time")]
    tk = de or [k for k in keys if k.endswith("_time")]
    if tk:
        return np.asarray(m[tk[0]]).ravel().astype(float)
    # fall back to the largest numeric array
    arrs = [(np.asarray(m[k]).size, k) for k in keys if hasattr(m[k], "size")]
    return np.asarray(m[max(arrs)[1]]).ravel().astype(float)


def _read_recording(path: str, column: int = 0) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        import scipy.io as sio
        return _mat_signal(sio.loadmat(path))
    if ext == ".npy":
        return np.load(path).astype(float).ravel()
    if ext in (".csv", ".txt"):
        return _read_text_signal(path, column, "," if ext == ".csv" else None)
    raise SystemExit(f"unsupported recording type {ext!r} (use .mat/.csv/.txt/.npy)")


MAX_SKIPPED_ROW_FRACTION = 0.05


def _read_text_signal(path: str, column: int, delimiter: str | None) -> np.ndarray:
    """Tolerant line parser for real-world .txt/.csv recordings: eats a UTF-8
    BOM, CRLF endings, and leading header lines; skips (and counts) ragged or
    non-finite rows AFTER data starts, failing closed when more than
    MAX_SKIPPED_ROW_FRACTION of data rows lack the requested column — a signal
    that silently lost 98% of its samples is worse than an error."""
    def _has_float(toks: list[str]) -> bool:
        for t in toks:
            try:
                float(t)
                return True
            except ValueError:
                continue
        return False

    vals: list[float] = []
    skipped = 0
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            toks = line.split(delimiter) if delimiter else line.split()
            try:
                v = float(toks[column])
            except (IndexError, ValueError):
                # A genuine header/comment line has NO numeric field anywhere and
                # appears before any data -> skip it uncounted. A row that carries
                # a number in another column but not `column` is a data-row gap and
                # MUST count, so the fail-closed fraction can't be dodged by putting
                # the missing rows first (order-independent guarantee).
                if not vals and not skipped and not _has_float(toks):
                    continue
                skipped += 1
                continue
            if np.isfinite(v):
                vals.append(v)
            else:
                skipped += 1
    if not vals:
        raise SystemExit(f"{path}: no numeric data in column {column}")
    frac = skipped / (len(vals) + skipped)
    if frac > MAX_SKIPPED_ROW_FRACTION:
        raise SystemExit(
            f"{path}: column {column} missing/unreadable on {skipped} of "
            f"{len(vals) + skipped} data rows ({frac:.0%}) — refusing to load a "
            f"mostly-empty signal (try a different --column)")
    if skipped:
        print(f"[ingest] {path}: skipped {skipped} unreadable row(s), "
              f"kept {len(vals)}", flush=True)
    return np.array(vals, float)


def _label(path: str, rule: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    if rule == "prefix":
        return stem.split("_")[0]
    if rule == "stem":
        return stem
    if rule == "parent":
        return os.path.basename(os.path.dirname(path))
    raise SystemExit(f"unknown --label-by rule {rule!r} (use prefix/stem/parent)")


def window(x: np.ndarray) -> list[np.ndarray]:
    """Slice into 1024-sample windows, each detrended + z-normed (spec §2.1).
    2-D input (C, n) is cut synchronously into (C, 1024) windows with the exact
    1-D semantics applied per channel; the 1-D path is byte-identical."""
    from scipy.signal import detrend
    x = np.asarray(x)
    if x.ndim == 2:
        out2: list[np.ndarray] = []
        for k in range(x.shape[1] // W):
            rows = []
            for ch in range(x.shape[0]):
                s = detrend(np.ascontiguousarray(x[ch, k * W:(k + 1) * W], float))
                rows.append((s - s.mean()) / (s.std() + 1e-12))
            out2.append(np.stack(rows))
        return out2
    out = []
    for k in range(len(x) // W):
        s = detrend(np.ascontiguousarray(x[k * W:(k + 1) * W], float))
        out.append((s - s.mean()) / (s.std() + 1e-12))
    return out


def _primary(w: np.ndarray) -> np.ndarray:
    """Primary-channel convention: base grammar (and 1-channel premium
    families) always see channel 0 of a multi-channel window."""
    return w[0] if w.ndim == 2 else w


def _n_channels(bank: "Bank") -> int:
    return bank.windows[0].shape[0] if bank.windows[0].ndim == 2 else 1


@dataclass
class Bank:
    windows: list[np.ndarray]
    y: np.ndarray          # class label per window
    g: np.ndarray          # recording id per window
    classes: list[str]
    n_recordings: int
    channels: list[str] = field(default_factory=list)   # empty = 1-channel

    @property
    def chance(self) -> float:
        return 1.0 / len(self.classes)


def _read_mc_recording(path: str, channel_axis: int | None) -> tuple[list[str], np.ndarray]:
    """One multi-channel recording -> (channel names, (C, n) array).
    .npy must be 2-D; orientation heuristic: the LONGER axis is time (override
    with channel_axis = the axis holding channels). .csv needs a header row of
    channel names (multichannel.load_channels semantics: NaN rows drop as a
    unit, synchrony preserved, per-column hardening)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise SystemExit(f"{path}: multichannel .npy must be 2-D, got "
                             f"shape {arr.shape}")
        if channel_axis is None:
            lo, hi = min(arr.shape), max(arr.shape)
            if lo / hi > 0.5:                     # near-square: heuristic is a guess
                import warnings
                warnings.warn(
                    f"{path}: near-square 2-D array {arr.shape}; the longer-axis "
                    f"channel heuristic is ambiguous — pass channel_axis to be "
                    f"explicit (guessing axis {int(np.argmin(arr.shape))} = channels)",
                    UserWarning, stacklevel=2)
            ax = int(np.argmin(arr.shape))
        else:
            ax = channel_axis
        X = (arr if ax == 0 else arr.T).astype(float)
        return [f"ch{i}" for i in range(X.shape[0])], X
    if ext == ".csv":
        from signalmap.multichannel import load_channels
        try:
            chans = load_channels(path)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        return list(chans), np.stack([chans[c] for c in chans])
    raise SystemExit(f"unsupported multichannel recording type {ext!r} "
                     f"(use .csv with a header or 2-D .npy)")


def load_bank(path: str, label_by: str = "prefix", column: int = 0,
              pattern: str = "*", multichannel: bool = False,
              channel_axis: int | None = None) -> Bank:
    """A bank = a directory of recordings (one file per recording). Each file is
    windowed to 1024; its class comes from --label-by. With multichannel=True,
    recordings are (C, n) arrays cut to synchronous (C, 1024) windows; all
    recordings must agree on the channel layout (fail-closed)."""
    if os.path.isdir(path):
        files = sorted(f for f in glob.glob(os.path.join(path, "**", pattern), recursive=True)
                       if os.path.splitext(f)[1].lower() in (".mat", ".npy", ".csv", ".txt")
                       and os.path.isfile(f))
    else:
        files = sorted(glob.glob(path))
    if not files:
        raise SystemExit(f"no recordings under {path!r}")
    wins, ys, gs = [], [], []
    channels: list[str] = []
    for gid, f in enumerate(files):
        if multichannel:
            names, x = _read_mc_recording(f, channel_axis)
            if not channels:
                channels = names
            elif names != channels:
                raise SystemExit(
                    f"{f}: channel layout {names} does not match the bank's "
                    f"{channels} — all recordings must share one channel layout")
        else:
            x = _read_recording(f, column)
        w = window(x)
        lab = _label(f, label_by)
        wins.extend(w); ys.extend([lab] * len(w)); gs.extend([gid] * len(w))
    if not wins:
        raise SystemExit("no windows produced (recordings shorter than 1024?)")
    classes = sorted(set(ys))
    return Bank(wins, np.array(ys), np.array(gs), classes, len(files), channels)


# --------------------------------------------------------------- gauntlet
def _feature_matrix(bank: Bank, programs: list[Program]) -> np.ndarray:
    return np.array([[p(_primary(s)) for p in programs] for s in bank.windows], float)


def _anova_f(F: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Recording-level ANOVA-F (ranking on recording means avoids window
    pseudo-replication)."""
    recs = np.unique(g)
    M = np.array([F[g == r].mean(0) for r in recs])
    yr = np.array([y[g == r][0] for r in recs])
    keep = M.std(0) > 1e-9
    Ms = (M - M.mean(0)) / (M.std(0) + 1e-12)
    classes = np.unique(yr)
    between = np.zeros(F.shape[1]); within = np.zeros(F.shape[1]) + 1e-12
    for c in classes:
        mc = Ms[yr == c]
        between += len(mc) * (mc.mean(0) - Ms.mean(0)) ** 2
        within += ((mc - mc.mean(0)) ** 2).sum(0)
    f = (between / max(len(classes) - 1, 1)) / (within / max(len(recs) - len(classes), 1))
    return np.where(keep, f, -1.0)


def _logo_mean(F, y, g, trees=100, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return cross_val_score(
        make_pipeline(StandardScaler(),
                      RandomForestClassifier(trees, random_state=seed, n_jobs=-1)),
        F, y, groups=g, cv=LeaveOneGroupOut()).mean()


def _logo_folds(F, y, g, trees=100, seed=0) -> dict:
    """Per-held-out-recording LOGO accuracy (the paired-CI unit)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    accs = {}
    for r in np.unique(g):
        tr = g != r
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(trees, random_state=seed, n_jobs=-1))
        clf.fit(F[tr], y[tr])
        accs[r] = float((clf.predict(F[~tr]) == y[~tr]).mean())
    return accs


def _paired_ci(challenger: dict, champion: dict, n_boot=10000, seed=0):
    """Bootstrap 95% CI of the per-recording paired difference (challenger −
    champion). Positive lower bound = CI-solid win."""
    common = sorted(set(challenger) & set(champion))
    d = np.array([challenger[r] - champion[r] for r in common])
    rng = np.random.default_rng(seed)
    means = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _greedy_select(F, y, g, order, kmax=5, thr=0.005, cand=60, trees=100):
    """Forward selection INSIDE the honest LOGO metric (feature_forge logic)."""
    sel, best = [], 0.0
    for j in order[:cand]:
        acc = _logo_mean(F[:, sel + [j]], y, g, trees)
        if acc > best + thr:
            sel, best = sel + [j], acc
        if len(sel) == kmax:
            break
    return sel, best


def _nested_logo(F, y, g, kmax=5, thr=0.005, cand=40, trees=100, seed=0):
    """True nested LOGO (spec §2.3): for each held-out recording, rank + greedy-
    select on the REST only, then predict the held-out recording. No selection
    ever sees the test recording -> the honesty anchor for the deploy spec."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    recs = np.unique(g)
    correct = total = 0
    for r in recs:
        tr = g != r; te = g == r
        Ftr, ytr, gtr = F[tr], y[tr], g[tr]
        order = np.argsort(-_anova_f(Ftr, ytr, gtr))
        # greedy on train groups via inner GroupKFold (no test leakage)
        n_in = min(4, len(np.unique(gtr)))
        sel, best = [], 0.0
        for j in order[:cand]:
            trial = sel + [j]
            gk = GroupKFold(n_splits=n_in)
            accs = []
            for itr, iva in gk.split(Ftr, ytr, gtr):
                clf = make_pipeline(StandardScaler(),
                                    RandomForestClassifier(trees, random_state=seed, n_jobs=-1))
                clf.fit(Ftr[itr][:, trial], ytr[itr])
                accs.append(clf.score(Ftr[iva][:, trial], ytr[iva]))
            acc = float(np.mean(accs))
            if acc > best + thr:
                sel, best = trial, acc
            if len(sel) == kmax:
                break
        if not sel:
            sel = [int(order[0])]
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(trees, random_state=seed, n_jobs=-1))
        clf.fit(Ftr[:, sel], ytr)
        pred = clf.predict(F[te][:, sel])
        correct += int((pred == y[te]).sum()); total += int(te.sum())
    return correct / max(total, 1)


def _group_perm_p(F, y, g, obs, n_perm=60, trees=100, seed=0):
    rng = np.random.default_rng(seed)
    recs = np.unique(g); rc = {r: y[g == r][0] for r in recs}
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation([rc[r] for r in recs])
        ymap = dict(zip(recs, perm))
        if _logo_mean(F, np.array([ymap[r] for r in g]), g, trees) >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def _lean_baseline(bank: Bank) -> np.ndarray:
    """perm-entropy + psd-slope reference features (the honest baseline to beat)."""
    from scipy.signal import welch
    try:
        import antropy as ant
        pe = lambda s: ant.perm_entropy(s, order=3, normalize=True)
    except Exception:  # pragma: no cover - antropy present in research venv
        def pe(s):  # ordinal-pattern entropy fallback
            a, b, c = s[:-2], s[1:-1], s[2:]
            pat = (a < b).astype(int) * 2 + (b < c).astype(int)
            h = np.bincount(pat, minlength=4) / max(len(pat), 1)
            h = h[h > 0]; return float(-(h * np.log(h)).sum() / np.log(4))

    def slope(s):
        f, P = welch(s, nperseg=min(256, len(s))); f, P = f[1:], P[1:]
        return np.polyfit(np.log(f), np.log(P + 1e-20), 1)[0]
    return np.array([[pe(_primary(s)), slope(_primary(s))] for s in bank.windows])


@dataclass
class DistillResult:
    spec: "FeatureSpec"
    report: str
    nested_acc: float
    chance: float
    lean_acc: float
    forged_acc: float
    p_forged: float
    null_acc: float
    budget: int
    grammar_total: int
    cost_ms: float
    passed: bool
    premium_receipts: list = field(default_factory=list)


def distill(bank: Bank, C: int = 50, kmax: int = 5, thr: float = 0.005,
            n_perm: int = 60, trees: int = 100, cand: int = 60, seed: int = 0,
            null_check: bool = True, premium: tuple = (),
            family_families: set[str] | None = None) -> DistillResult:
    """Run the full distillation gauntlet and return a deployable spec + receipt."""
    if bank.n_recordings < 2:
        raise SystemExit(
            f"distill needs >=2 recordings for the leave-one-group-out gauntlet, "
            f"got {bank.n_recordings} — each recording is one held-out group, and "
            f"an honest accuracy receipt requires at least one to hold out")
    all_progs = enumerate_programs()
    if family_families is not None:
        from .qualification import filter_programs
        all_progs = filter_programs(all_progs, set(family_families))
        if not all_progs:
            raise SystemExit(
                "qualification routed zero base-grammar programs; "
                "re-qualify the source or use an explicit family set")
    kept = gate(all_progs, bank.n_recordings, C)
    budget = len(kept)

    F = _feature_matrix(bank, kept)
    order = np.argsort(-_anova_f(F, bank.y, bank.g))

    # deploy selection: greedy on the FULL bank (reported with the nested anchor)
    sel, forged_acc = _greedy_select(F, bank.y, bank.g, order, kmax, thr, cand, trees)
    if not sel:
        sel = [int(order[0])]
        forged_acc = _logo_mean(F[:, sel], bank.y, bank.g, trees)
    sel_names = [kept[j].name for j in sel]

    # honesty anchor: nested LOGO (selection never sees the test recording)
    nested_acc = _nested_logo(F, bank.y, bank.g, kmax, thr, min(cand, 40), trees, seed)
    # group-permutation p on the deploy selection
    p_forged = _group_perm_p(F[:, sel], bank.y, bank.g, forged_acc, n_perm, trees, seed)

    # NULL self-test: shuffle labels across recordings -> must fall to chance
    null_acc = float("nan")
    if null_check:
        rng = np.random.default_rng(seed + 1)
        recs = np.unique(bank.g); rc = {r: bank.y[bank.g == r][0] for r in recs}
        perm = rng.permutation([rc[r] for r in recs]); ymap = dict(zip(recs, perm))
        y_null = np.array([ymap[r] for r in bank.g])
        null_acc = _nested_logo(F, y_null, bank.g, kmax, thr, min(cand, 40), trees, seed)

    # lean baseline + cost
    lean_acc = _logo_mean(_lean_baseline(bank), bank.y, bank.g, trees)
    import time
    t0 = time.time()
    probe = bank.windows[:min(300, len(bank.windows))]
    for s in probe:
        for j in sel:
            kept[j](_primary(s))
    cost_ms = 1000 * (time.time() - t0) / max(len(probe), 1)

    # premium families: cost-receipted champion rule (paired CI over recordings;
    # a family enters the deploy spec ONLY on a CI-solid win — never silently)
    premium_receipts: list[dict] = []
    spec_premium: list[str] = []
    if premium:
        from signalmap.premium import PREMIUM_FAMILIES
        n_ch = _n_channels(bank)
        for name in premium:
            fam = PREMIUM_FAMILIES[name]
            if fam.needs_channels > n_ch:
                raise SystemExit(
                    f"premium family {name!r} needs {fam.needs_channels} "
                    f"channels, but the bank has {n_ch} — load a multi-channel "
                    f"bank (CSV with header / 2-D .npy) to use this family")
        base_folds = _logo_folds(F[:, sel], bank.y, bank.g, trees, seed)
        base_acc = float(np.mean(list(base_folds.values())))
        for name in premium:
            fam = PREMIUM_FAMILIES[name]
            fam_in = _primary if fam.needs_channels == 1 else (lambda w: w)
            t0 = time.time()
            P = np.nan_to_num(np.array([fam.featurize(fam_in(s)) for s in bank.windows],
                                       float), nan=0.0, posinf=0.0, neginf=0.0)
            fam_cost = 1000 * (time.time() - t0) / max(len(bank.windows), 1)
            aug_folds = _logo_folds(np.hstack([F[:, sel], P]), bank.y, bank.g,
                                    trees, seed)
            delta, lo, hi = _paired_ci(aug_folds, base_folds)
            included = lo > 0
            premium_receipts.append({
                "family": name, "base_acc": base_acc,
                "aug_acc": float(np.mean(list(aug_folds.values()))),
                "delta": delta, "ci_lo": lo, "ci_hi": hi,
                "cost_ms": float(fam_cost), "base_cost_ms": float(cost_ms),
                "cost_note": fam.cost_note, "included": included})
            if included:
                spec_premium.append(name)

    passed = (nested_acc > bank.chance + 0.05 and p_forged <= 0.05 and
              (np.isnan(null_acc) or null_acc <= bank.chance + 0.10))

    spec = FeatureSpec(
        programs=sel_names, classes=bank.classes, window=W, budget_c=C,
        n_recordings=bank.n_recordings, nested_acc=float(nested_acc),
        forged_acc=float(forged_acc), chance=float(bank.chance),
        premium=spec_premium, channels=list(bank.channels))
    report = _render_report(bank, sel_names, budget, len(all_progs), C, nested_acc,
                            forged_acc, lean_acc, p_forged, null_acc, cost_ms, passed,
                            premium_receipts, family_families)
    return DistillResult(spec, report, float(nested_acc), float(bank.chance),
                         float(lean_acc), float(forged_acc), float(p_forged),
                         float(null_acc), budget, len(all_progs), float(cost_ms), passed,
                         premium_receipts)


def _render_report(bank, sel, budget, total, C, nested, forged, lean, pf, null_acc,
                   cost_ms, passed, premium_receipts=(), family_families=None) -> str:
    per_rec = budget / max(bank.n_recordings, 1)
    L = ["# signalmap distill — gauntlet receipt", ""]
    L += [f"- verdict: {'PASS' if passed else 'FAIL'}",
          f"- classes ({len(bank.classes)}): {', '.join(bank.classes)}",
          f"- recordings: {bank.n_recordings} · windows: {len(bank.windows)} · "
          f"chance: {bank.chance:.3f}", ""]
    L += ["## capacity gate",
          f"- grammar total: {total}",
          f"- budget = min({total}, {C}*{bank.n_recordings}) = {budget} "
          f"({per_rec:.1f} programs/recording)", ""]
    if family_families is not None:
        L += ["## qualification routing",
              f"- compatible base families: {', '.join(sorted(family_families))}",
              "- incompatible families were excluded before feature computation", ""]
    L += ["## accuracy receipts (LOGO over recordings)",
          f"- nested LOGO (honesty anchor): {nested:.3f}",
          f"- deploy selection (biased UB): {forged:.3f}",
          f"- lean baseline (permEnt+psdSlope): {lean:.3f} "
          f"({forged - lean:+.3f} vs lean)",
          f"- group-permutation p: {pf:.3f}",
          f"- NULL self-test (labels shuffled): "
          f"{'n/a' if np.isnan(null_acc) else f'{null_acc:.3f} (want ~chance {bank.chance:.3f})'}",
          f"- cost: {cost_ms:.3f} ms/window ({len(sel)} programs)", ""]
    if premium_receipts:
        L += ["## premium families (cost-receipted champion rule)",
              "The PASS/FAIL verdict above gates the BASE selection only "
              "(nested LOGO on grammar programs). Premium families are judged "
              "separately by the champion rule below — a bank can honestly be "
              "base-FAIL with a premium family INCLUDED when the signal lives "
              "only in the premium features."]
        for r in premium_receipts:
            verdict = "INCLUDED" if r["included"] else "EXCLUDED"
            ratio = r["cost_ms"] / max(r["base_cost_ms"], 1e-9)
            L += [f"- {r['family']}: base {r['base_acc']:.3f} -> augmented "
                  f"{r['aug_acc']:.3f} (paired {r['delta']:+.3f}, "
                  f"95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]) — **{verdict}**"
                  f"{'' if r['included'] else ' (not CI-solid over base)'}",
                  f"  cost: {r['cost_ms']:.2f} ms/window vs base {r['base_cost_ms']:.3f} "
                  f"ms/window (~{ratio:.0f}x); {r['cost_note']}"]
        L += [""]
    L += ["## deploy spec (selected programs)"]
    L += [f"- {n}" for n in sel]
    L += ["", "Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend."]
    return "\n".join(L)


# ------------------------------------------------------------- deploy surface
def _ensure_parent(path: str) -> None:
    """Create the output file's parent dir if it doesn't exist — the README uses
    `--out artifacts/spec.json`, and a fresh checkout has no artifacts/. Never
    discard a completed gauntlet to a missing directory."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


@dataclass
class FeatureSpec:
    """The deployable output: distilled program strings + metadata. `fit`/`monitor`
    load this as the feature backend (via DistilledDetector)."""
    programs: list[str]
    classes: list[str] = field(default_factory=list)
    window: int = W
    budget_c: int = 50
    n_recordings: int = 0
    nested_acc: float = 0.0
    forged_acc: float = 0.0
    chance: float = 0.0
    premium: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)

    def _compiled(self) -> list[Program]:
        by_name = {p.name: p for p in enumerate_programs()}
        return [by_name[n] for n in self.programs]

    def featurize(self, w: np.ndarray) -> np.ndarray:
        """Raw 1024-window -> distilled feature vector. Windows must be produced
        the same way as at distill time (use `window()`), i.e. 1024, detrended,
        z-normed. Multi-channel specs (`channels` set) take (C, 1024) windows;
        base programs see only channel 0, premium families with needs_channels
        >= 2 see the full window. Channel-count mismatches fail closed.
        Premium-family features (champion-rule winners only) are appended after
        the grammar programs."""
        w = np.asarray(w, float)
        want = len(self.channels)
        if want:
            got = w.shape[0] if w.ndim == 2 else 1
            if w.ndim != 2 or got != want:
                raise ValueError(
                    f"spec expects {want} channels {self.channels}, "
                    f"got a window with {got} (shape {w.shape}) — windows must "
                    f"be produced like at distill time (window() on the same "
                    f"channel layout)")
        elif w.ndim != 1:
            raise ValueError(
                f"1-channel spec got a multi-channel window (shape {w.shape})")
        base = _primary(w)
        v = [float(p(base)) for p in self._compiled()]
        if self.premium:
            from signalmap.premium import PREMIUM_FAMILIES
            for name in self.premium:
                fam = PREMIUM_FAMILIES[name]
                if fam.needs_channels > max(want, 1):
                    raise ValueError(
                        f"premium family {name!r} needs {fam.needs_channels} "
                        f"channels, spec has {max(want, 1)}")
                fam_w = base if fam.needs_channels == 1 else w
                v.extend(np.nan_to_num(fam.featurize(fam_w),
                                       nan=0.0, posinf=0.0, neginf=0.0))
        return np.array(v, float)

    def save(self, path: str) -> None:
        _ensure_parent(path)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureSpec":
        with open(path) as fh:
            return cls(**json.load(fh))


MAD_TO_SIGMA = 1.4826


def _rank_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC with tie-averaged ranks; keeps sklearn off this path."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=np.float64)
    ss = y_score[order]
    i = 0
    while i < ss.size:
        j = i
        while j + 1 < ss.size and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


@dataclass
class DirectionVerdict:
    """Outcome of trying to identify which way anomaly lies."""
    identified: bool
    sign: int | None
    auc: float | None
    ci_lo: float | None
    ci_hi: float | None
    n_anchors: int
    reason: str


@dataclass
class Decision:
    """What the detector is willing to say about one window."""
    verdict: str          # ALARM | QUIET | REFUSED
    score: float
    reason: str


@dataclass
class DistilledDetector:
    """fit-on-healthy / monitor deployment for a distilled spec — the same product
    surface as signalmap fit/monitor, but the feature backend is spec.json instead
    of the spectral autoencoder. Robust per-feature z-score (max over features);
    the alert threshold is calibrated from the healthy envelope at fit time, so it
    self-scales to features of arbitrary magnitude (unlike the fixed-σ spectral
    detector, whose recon error is already O(1))."""
    spec: FeatureSpec
    med: np.ndarray
    mad: np.ndarray
    threshold: float = 4.0
    degenerate: np.ndarray | None = None
    direction: int | None = None

    @classmethod
    def fit(cls, spec: FeatureSpec, healthy_windows: list[np.ndarray],
            threshold: float | None = None, envelope: float = 3.0) -> "DistilledDetector":
        X = np.array([spec.featurize(w) for w in healthy_windows], float)
        med = np.median(X, 0)
        mad = np.median(np.abs(X - med), 0) * MAD_TO_SIGMA
        # A feature that does not vary at all on healthy data cannot be
        # standardised: dividing its rounding noise by the guard floor put it
        # twelve orders of magnitude above every real feature and captured the
        # score. Such features are excluded and named instead.
        # A feature can be exactly constant on healthy data for two very
        # different reasons, and the guard floor has to tell them apart.
        #
        # Legitimate: a discrete feature (acflag-style) whose healthy value
        # never moves. A shift of 1e-3 on it is real and must alert.
        # Spurious: a feature that is constant BY CONSTRUCTION -- window()
        # z-normalises every window, so std(x) is exactly 1.0 for any input
        # whatsoever. What is left is float accumulation noise, ~1e-12
        # relative here, and an absolute 1e-12 floor divided it up into the
        # dominant term of max|z|. Measured: two such features on the MAFAULDA
        # bank drove the self-calibrated threshold to 9.67e8.
        #
        # A floor of one part per billion of the feature's own magnitude
        # separates them: accumulation noise lands near z ~ 1e-3 and stays
        # quiet, while any physically meaningful change is many orders above
        # it and still alerts.
        floor = np.maximum(1e-12, 1e-9 * np.abs(med))
        degenerate = mad <= floor
        mad = np.where(degenerate, floor, mad)
        det = cls(spec, med, mad, 4.0, degenerate)
        if threshold is None:
            hs = np.array([det.score(w) for w in healthy_windows])
            # alert above `envelope`x the healthy 99th-percentile score (min 1e-6)
            threshold = float(max(np.quantile(hs, 0.99) * envelope, 1e-6))
        det.threshold = threshold
        return det

    def score(self, w: np.ndarray) -> float:
        z = np.abs(self.spec.featurize(w) - self.med) / self.mad
        return float(np.max(z)) if z.size else 0.0

    # ---------------------------------------------------------- direction
    # A healthy-only fit fixes the standardising map, and so a threshold, but
    # it never observes an anomaly. Whether anomalies score HIGHER or LOWER is
    # a property of the unobserved alternative, and measurement says it goes
    # both ways: MIMII valve 0.2786, MIMII slider 0.8254, Paderborn phase
    # current 0.3172, each with a bootstrap CI clear of 0.5 under one frozen
    # recipe. So `direction` starts None and `decide` refuses until labelled
    # anchors settle it.

    def calibrate_direction(self, windows, labels, groups=None,
                            n_boot: int = 2000, seed: int = 0) -> "DirectionVerdict":
        """Identify the sign from labelled anchors (1 = anomalous, 0 = normal).

        Sets ``self.direction`` to +1 (anomalies score higher), -1 (anomalies
        score lower), or leaves it None when the anchors do not decide it. The
        criterion is a bootstrap CI that clears 0.5 — a point estimate on the
        wrong side of chance is the very failure this guards against.

        ``groups`` gives the recording each window came from, and the
        bootstrap then resamples RECORDINGS. Pass it whenever windows share a
        recording: twenty windows cut from one signal are not twenty
        independent observations, and treating them as such lets chance
        structure in a single recording look like a direction. Which CI comes
        out wider depends on the case — near the boundary the grouped one can
        be narrower — but the recording is the honest inferential unit, and
        `n_anchors` then reports recordings rather than windows.
        """
        y = np.asarray(labels, dtype=int)
        if y.size < 4 or len(windows) != y.size:
            return DirectionVerdict(False, None, None, None, None, int(y.size),
                                    "too few anchors: need at least 4, with "
                                    "both classes present")
        if np.unique(y).size < 2:
            return DirectionVerdict(False, None, None, None, None, int(y.size),
                                    "anchors must contain both classes")

        s = np.array([self.score(w) for w in windows], dtype=float)
        observed = _rank_auc(y, s)
        rng = np.random.default_rng(seed)
        vals = []
        if groups is None:
            for _ in range(n_boot):
                idx = rng.integers(0, y.size, y.size)
                if np.unique(y[idx]).size < 2:
                    continue
                vals.append(_rank_auc(y[idx], s[idx]))
        else:
            g = np.asarray(groups)
            uniq = np.unique(g)
            if uniq.size < 4:
                return DirectionVerdict(False, None, observed, None, None,
                                        int(uniq.size),
                                        "too few anchor recordings: need at "
                                        "least 4 distinct groups")
            members = [np.flatnonzero(g == u) for u in uniq]
            for _ in range(n_boot):
                pick = rng.integers(0, uniq.size, uniq.size)
                idx = np.concatenate([members[k] for k in pick])
                if np.unique(y[idx]).size < 2:
                    continue
                vals.append(_rank_auc(y[idx], s[idx]))
        if not vals:
            return DirectionVerdict(False, None, observed, None, None,
                                    int(y.size),
                                    "bootstrap could not resample both classes")
        n_units = int(y.size if groups is None else np.unique(groups).size)
        lo, hi = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
        if lo > 0.5:
            self.direction = 1
            return DirectionVerdict(True, 1, observed, lo, hi, n_units,
                                    "anomalies score higher than healthy")
        if hi < 0.5:
            self.direction = -1
            return DirectionVerdict(True, -1, observed, lo, hi, n_units,
                                    "anomalies score lower than healthy")
        self.direction = None
        return DirectionVerdict(False, None, observed, lo, hi, n_units,
                                "the anchors do not separate: the CI covers "
                                "0.5, so the direction stays unknown")

    @property
    def inverse_threshold(self) -> float:
        """Cut for an inverted direction: the healthy envelope mirrored.

        The robust z lives on a ratio scale, so the low-side cut divides by
        the same factor the high-side cut multiplies by.
        """
        return float(self.threshold / 9.0)

    def decide(self, w: np.ndarray) -> "Decision":
        """ALARM / QUIET / REFUSED — the honest decision surface.

        REFUSED is not a failure mode. It is the correct answer when the
        direction was never identified: the score is informative about
        distance and silent about which way anomaly lies.
        """
        s = self.score(w)
        if self.direction is None:
            return Decision("REFUSED", s,
                            "direction unknown: a healthy-only fit cannot tell "
                            "whether anomalies score higher or lower. Supply "
                            "labelled anchors to calibrate_direction().")
        cut = self.threshold if self.direction == 1 else self.inverse_threshold
        hit = (s >= cut) if self.direction == 1 else (s <= cut)
        return Decision("ALARM" if hit else "QUIET", s,
                        f"direction {self.direction:+d}, score {s:.4g} "
                        f"vs cut {cut:.4g}")

    def alert(self, w: np.ndarray) -> bool:
        return self.score(w) >= self.threshold

    def save(self, path: str) -> None:
        _ensure_parent(path)
        with open(path, "w") as fh:
            json.dump({"spec": asdict(self.spec), "med": self.med.tolist(),
                       "mad": self.mad.tolist(), "threshold": self.threshold,
                       "degenerate": (None if self.degenerate is None
                                      else [bool(x) for x in self.degenerate]),
                       "direction": self.direction},
                      fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "DistilledDetector":
        with open(path) as fh:
            d = json.load(fh)
        deg = d.get("degenerate")
        return cls(FeatureSpec(**d["spec"]), np.array(d["med"]), np.array(d["mad"]),
                   d["threshold"], None if deg is None else np.array(deg, dtype=bool),
                   d.get("direction"))


# ------------------------------------------------------------------ CLI entry
def run_cli(bank_path: str, label_by: str, budget_c: int, out: str,
            report_out: str | None = None, pattern: str = "*", column: int = 0,
            n_perm: int = 60, trees: int = 100, premium: tuple = (),
            multichannel: bool = False,
            channel_axis: int | None = None,
            family_families: set[str] | None = None) -> DistillResult:
    bank = load_bank(bank_path, label_by, column, pattern, multichannel, channel_axis)
    res = distill(bank, C=budget_c, n_perm=n_perm, trees=trees, premium=premium,
                  family_families=family_families)
    res.spec.save(out)
    rpath = report_out or (os.path.splitext(out)[0] + "_report.md")
    with open(rpath, "w") as fh:
        fh.write(res.report)
    print(res.report)
    from .run_receipts import emit_distill_receipts
    receipts = emit_distill_receipts(res, bank_path=bank_path, spec_path=out,
                                     report_path=rpath)
    print(f"\nspec  -> {out}\nreport -> {rpath}")
    for p in receipts:
        print(f"receipt -> {p}")
    return res
