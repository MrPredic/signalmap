"""FEATURE FORGE — generate genuinely NEW method instances, not hyperparameters.

A compositional grammar over cheap O(n)/O(n log n) primitives:
    program = pool ∘ transform2 ∘ transform1
Every program is a feature no one has published *as an instance* (the CLASS —
generated feature construction — has prior art: hctsa's fixed 7k library,
catch22's fixed 22, ROCKET's random kernels, GP feature construction). Our
twist: unbounded grammar + edge-cost axis + the honest gauntlet (recording-level
LOGO, group permutation p, greedy selection INSIDE the honest metric).

Test bed = the retro banks with known answers: ECN chemicals (6-class) and CWRU
fault types (6-class). Question answered: can forged programs BEAT the lean
hand-picked baseline (perm3+psd_slope) at comparable cost?
Also closes the ECN debt: group-perm p for the lean baseline.
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/feature_forge.py
"""
import numpy as np, glob, os, time, warnings, antropy as ant
import scipy.io as sio
from scipy.signal import detrend, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
warnings.filterwarnings("ignore")

# ---------------- grammar: series->series transforms ----------------
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
def t_env(x):  # rectified + 8-point smoothed envelope
    a = np.abs(x); k = np.ones(8) / 8.0
    return np.convolve(a, k, mode="valid")
def t_cumsum(x):  # integrated profile (DFA-style view)
    c = np.cumsum(x - x.mean()); return c / (np.abs(c).max() + 1e-12)
def t_clip(x):    # winsorize: kill outliers, keep texture
    lo, hi = np.quantile(x, [0.05, 0.95]); return np.clip(x, lo, hi)
TRANSFORMS = {"id": t_id, "diff": t_diff, "diff2": t_diff2, "abs": t_abs,
              "sign": t_sign, "rank": t_rank, "sq": t_sq, "tanh": t_tanh,
              "rollstd": t_rollstd, "env": t_env, "cumsum": t_cumsum, "clip": t_clip}

# ---------------- grammar: series->scalar pools ----------------
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
def p_ordtrans(x):  # entropy of ordinal-pattern TRANSITIONS (order 3) — dynamics, not distribution
    a, b, c = x[:-2], x[1:-1], x[2:]
    pat = (a < b).astype(int) * 2 + (b < c).astype(int)   # 4 coarse ordinal states
    T = np.zeros((4, 4)); np.add.at(T, (pat[:-1], pat[1:]), 1.0)
    p = (T / max(T.sum(), 1)).ravel(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())
def p_lcross(x):    # CV of intervals between up-crossings of +1 sigma
    up = np.flatnonzero((x[:-1] < 1.0) & (x[1:] >= 1.0))
    if len(up) < 3: return 0.0
    iv = np.diff(up); return float(iv.std() / (iv.mean() + 1e-12))
def p_acflag(x):    # dominant memory: argmax of acf over lags 2..64
    v = x - x.mean(); d = (v * v).sum() + 1e-12
    ac = np.array([(v[l:] * v[:-l]).sum() / d for l in range(2, 65)])
    return float((np.argmax(ac) + 2) / 64.0)
def p_specflat(x):  # spectral flatness: tonal (0) vs noisy (1)
    P = np.abs(np.fft.rfft(x)) ** 2 + 1e-20
    return float(np.exp(np.mean(np.log(P))) / P.mean())
POOLS = {"std": p_std, "meanabs": p_meanabs, "iqr90": p_iqr90, "hent": p_hent,
         "zcr": p_zcr, "acf1": p_acf1, "runmean": p_runmean, "runcv": p_runcv,
         "peakcv": p_peakcv, "specratio": p_specratio, "speccent": p_speccent,
         "crest": p_crest, "ordtrans": p_ordtrans, "lcross": p_lcross,
         "acflag": p_acflag, "specflat": p_specflat}

def _probe_bank(n=48, seed=0):
    """Synthetic probe signals (white/AR(1)/tonal/bursty) for grammar dedup —
    data-independent, so canonicalization can never leak bank information."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    sigs = []
    for i in range(n):
        k = i % 4
        if k == 0:
            s = rng.standard_normal(1024)
        elif k == 1:
            s = lfilter([1.0], [1.0, -0.9], rng.standard_normal(1024))
        elif k == 2:
            t = np.arange(1024); f = rng.uniform(0.01, 0.3, 2)
            s = (np.sin(2 * np.pi * f[0] * t) + 0.5 * np.sin(2 * np.pi * f[1] * t)
                 + 0.3 * rng.standard_normal(1024))
        else:
            s = (rng.standard_normal(1024) * (rng.random(1024) < 0.05) * 5.0
                 + 0.3 * rng.standard_normal(1024))
        s = s - s.mean()
        sigs.append(s / (s.std() + 1e-12))
    return sigs

_DEDUP_IDX = None

def _dedup_indices(out):
    """Grammar v2.1 canonicalization (audit finding B1): the raw product grammar
    contains algebraic identity clones (diff∘cumsum ≈ id, rank∘tanh ≡ rank,
    abs∘env ≡ env, …) — on ECN only 13/30 of the top-ranked programs were
    effectively unique, burning selection budget. Empirical fix: evaluate all
    programs on synthetic probes, keep the FIRST program of every clone cluster
    (insertion order puts the simplest form first), drop probe-constants.
    Clone criterion is DUAL: |r|>0.999 OR >=95% bit-equal values — the latter
    catches discrete-valued features (acflag/ordtrans argmax outputs) where a
    single flipped probe breaks the correlation but the program is still a
    clone (e.g. acflag(diff(cumsum(x))) vs acflag(x): 99% equal, r 0.998).
    Cached; deterministic."""
    global _DEDUP_IDX
    if _DEDUP_IDX is None:
        cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "cache")
        os.makedirs(cdir, exist_ok=True)
        cache = os.path.join(cdir, f"grammar_dedup_v2_{len(out)}.npy")
        if os.path.exists(cache):
            _DEDUP_IDX = np.load(cache)
        else:
            P = np.array([[run_prog(p, s) for p in out] for s in _probe_bank(96)])
            sd = P.std(0)
            Z = (P - P.mean(0)) / (sd + 1e-12)
            C = np.abs(Z.T @ Z / len(P))
            keep = []
            for j in range(len(out)):
                if sd[j] <= 1e-9: continue
                if keep:
                    K = np.array(keep)
                    eq = (P[:, [j]] == P[:, K]).mean(0)
                    if (C[j, K] > 0.999).any() or (eq >= 0.95).any(): continue
                keep.append(j)
            _DEDUP_IDX = np.array(keep)
            np.save(cache, _DEDUP_IDX)
    return _DEDUP_IDX

def programs(dedup=True, texture_guard=False):
    """Grammar v2.1. dedup: drop identity clones (audit finding B1).
    texture_guard: additionally drop cumsum-chain programs — the STATIONARITY
    GUARD for texture-mode (phase-agnostic, per-window z-normed) banks, where
    cumsum ⇒ random-walk 1/f² spectra ⇒ high in-sample F but zero generalization
    (the v2 ECN collapse). VALIDATED intervention on ECN nested: full v2 0.469
    (collapse) -> dedup 0.485 -> dedup+guard 0.689 ≈ curated v1 0.709.
    Keep texture_guard=False for phase-aligned windows: there cumsum programs
    carry real information (HYD-valve PHASE, RESULTS.md)."""
    out = []
    for n1, f1 in TRANSFORMS.items():
        for n2, f2 in TRANSFORMS.items():
            if n2 == "id" and n1 != "id": continue      # canonical: id only as inner slot
            for np_, fp in POOLS.items():
                out.append((f"{np_}({n2}({n1}(x)))", f1, f2, fp))
    if dedup:
        out = [out[j] for j in _dedup_indices(out)]
    if texture_guard:
        out = [p for p in out if "cumsum" not in p[0]]
    return out

def run_prog(prog, s):
    _, f1, f2, fp = prog
    try:
        v = fp(f2(f1(s)))
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0

# ------- multichannel grammar: slot 4 = channel combiner (validated on GAS:
# id-LODO 0.621 CI-fest after 5 single-channel nulls; gas_multichannel.py) -----
def combiners(n_ch):
    """(name, fn) reducing a (C, W) window to one 1-D signal. logratio/chdiff =
    the e-nose-style cross-channel pattern; mean/std = array aggregates."""
    cs = [(f"ch{i}", lambda X, i=i: X[i]) for i in range(n_ch)]
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            cs.append((f"logratio({i},{j})", lambda X, i=i, j=j:
                       np.log(np.abs(X[i]) + 1e-9) - np.log(np.abs(X[j]) + 1e-9)))
            cs.append((f"chdiff({i},{j})", lambda X, i=i, j=j: X[i] - X[j]))
    cs.append((f"mean{n_ch}", lambda X: X.mean(0)))
    cs.append((f"std{n_ch}", lambda X: X.std(0)))
    return cs

def mc_programs(n_ch, norm_modes=("shape", "level"), **prog_kw):
    """Full multichannel space: combiner x unary grammar x norm mode.
    Entry = (name, comb_idx, comb_fn, prog, norm). Sample to the capacity gate
    BEFORE building features (space is ~150k for 8 channels)."""
    cs, ps = combiners(n_ch), programs(**prog_kw)
    return [(f"{nm}:{cn}:{p[0]}", ci, cf, p, nm)
            for ci, (cn, cf) in enumerate(cs) for p in ps for nm in norm_modes]

def run_mc_prog(mcp, X, cache=None):
    """cache: dict shared across calls on the SAME window (combiner reuse)."""
    _, ci, cf, prog, nm = mcp
    try:
        key = (ci, nm)
        if cache is None or key not in (cache or {}):
            sig = np.asarray(cf(np.asarray(X, float)), float)
            if nm == "shape":
                sig = detrend(np.ascontiguousarray(sig))
                sig = (sig - sig.mean()) / (sig.std() + 1e-12)
            if cache is not None:
                cache[key] = sig
        else:
            sig = cache[key]
        return run_prog(prog, sig)
    except Exception:
        return 0.0

# ---------------- data: the two retro banks ----------------
W = 1024
def _win(x, cls, gid, raw):
    for k in range(len(x) // W):
        s = detrend(np.ascontiguousarray(x[k * W:(k + 1) * W], float))
        raw.append(((s - s.mean()) / (s.std() + 1e-12), cls, gid))

def load_ecn():
    BASE = "<local-path>/signalmap/data/ecn/Electrochemical Noise Data/Types of Corrosive Substances"
    raw = []
    for gid, f in enumerate(sorted(glob.glob(BASE + "/*/*.txt"))):
        v = []
        for ln in open(f, encoding="utf-8-sig", errors="ignore").readlines()[1:]:
            p = ln.split()
            if len(p) >= 2:
                try: v.append(float(p[1]))
                except ValueError: pass
        x = np.array(v)
        if len(x) >= 3 * W: _win(x, os.path.basename(os.path.dirname(f)), gid, raw)
    return raw

def load_cwru():
    CACHE = "<local-path>/signalmap/data/cwru_mats"
    raw = []
    for gid, p in enumerate(sorted(glob.glob(CACHE + "/*.mat"))):
        m = sio.loadmat(p)
        key = next(k for k in m if k.endswith("DE_time"))
        cls = os.path.basename(p).split("_")[0]
        _win(m[key].ravel().astype(float), cls, gid, raw)
    return raw

# ---------------- honest gauntlet ----------------
def logo_scores(X, y, g, trees=150):
    return cross_val_score(
        make_pipeline(StandardScaler(), RandomForestClassifier(trees, random_state=0, n_jobs=-1)),
        X, y, groups=g, cv=LeaveOneGroupOut())

def lean_baseline(raw):
    def psd_slope(x):
        f, P = welch(x, nperseg=min(256, len(x))); f, P = f[1:], P[1:]
        return np.polyfit(np.log(f), np.log(P + 1e-20), 1)[0]
    return np.array([[ant.perm_entropy(s, order=3, normalize=True), psd_slope(s)]
                     for s, _, _ in raw])

def group_perm_p(X, y, g, obs, n_perm=60, trees=100, seed=0):
    rng = np.random.default_rng(seed)
    recs = np.unique(g); rc = {r: y[g == r][0] for r in recs}
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation([rc[r] for r in recs])
        ymap = dict(zip(recs, perm))
        if logo_scores(X, np.array([ymap[r] for r in g]), g, trees).mean() >= obs: hits += 1
    return (hits + 1) / (n_perm + 1)

def forge(name, raw):
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    print(f"\n================ {name}: {len(raw)} windows, {len(set(g))} recordings, "
          f"chance={1/len(set(y)):.3f} ================", flush=True)
    progs = programs()
    t0 = time.time()
    F = np.array([[run_prog(p, s) for p in progs] for s, _, _ in raw])
    print(f"forged {len(progs)} programs x {len(raw)} windows in {time.time()-t0:.0f}s", flush=True)

    # drop degenerate + near-duplicate programs
    keep = np.std(F, 0) > 1e-9
    # group-level ANOVA-F on recording means (window-pseudoreplication-safe ranking)
    recs = np.unique(g)
    M = np.array([F[g == r].mean(0) for r in recs]); yr = np.array([y[g == r][0] for r in recs])
    Ms = (M - M.mean(0)) / (M.std(0) + 1e-12)
    classes = np.unique(yr)
    between = np.zeros(F.shape[1]); within = np.zeros(F.shape[1]) + 1e-12
    for c in classes:
        mc = Ms[yr == c]
        between += len(mc) * (mc.mean(0) - Ms.mean(0)) ** 2
        within += ((mc - mc.mean(0)) ** 2).sum(0)
    fstat = np.where(keep, (between / max(len(classes) - 1, 1)) /
                     (within / max(len(recs) - len(classes), 1)), -1)
    order = np.argsort(-fstat)

    # greedy forward selection INSIDE the honest metric (LOGO), max 5 features
    sel, best = [], 0.0
    for j in order[:60]:
        cand = sel + [j]
        acc = logo_scores(F[:, cand], y, g).mean()
        if acc > best + 0.005:
            sel, best = cand, acc
            print(f"  + {progs[j][0]:32s} -> LOGO {best:.3f}", flush=True)
        if len(sel) == 5: break

    # cost of the selected set
    t0 = time.time()
    for s, _, _ in raw[:300]:
        for j in sel: run_prog(progs[j], s)
    cost = 1000 * (time.time() - t0) / 300

    Xl = lean_baseline(raw)
    sl = logo_scores(Xl, y, g)
    print(f"\n  LEAN baseline (perm3+psd_slope): LOGO {sl.mean():.3f} (std {sl.std():.3f})")
    print(f"  FORGED set ({len(sel)} programs):      LOGO {best:.3f} @ {cost:.2f} ms/win")
    pf = group_perm_p(F[:, sel], y, g, best)
    pl = group_perm_p(Xl, y, g, sl.mean())
    print(f"  group-perm p: forged={pf:.3f}  lean={pl:.3f}  (60 perms)")
    return {"forged_acc": best, "lean_acc": float(sl.mean()), "sel": [progs[j][0] for j in sel],
            "cost_ms": cost, "p_forged": pf, "p_lean": pl}

if __name__ == "__main__":
    r1 = forge("ECN chemicals", load_ecn())
    r2 = forge("CWRU fault types", load_cwru())
    print("\n=== FORGE VERDICT ===")
    for n, r in [("ECN", r1), ("CWRU", r2)]:
        d = r["forged_acc"] - r["lean_acc"]
        print(f"{n}: forged {r['forged_acc']:.3f} vs lean {r['lean_acc']:.3f} ({d:+.3f}) "
              f"@ {r['cost_ms']:.2f}ms  p={r['p_forged']:.3f}")
        print(f"   selected: {r['sel']}")
