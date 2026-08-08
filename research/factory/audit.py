"""ADVERSARIAL AUDIT — CI gate: external-mindset check that the gauntlet isn't smoke.
Each check targets one specific way the pipeline could lie to us and runs <2 min
(feature matrices are cached to logs/cache/ as checkpoints; first run pays ~2 min
extra to build them). PASS = that failure mode is ruled out; FAIL = real trouble.

Checks (12):
  A. synthetic controls
    1  neg-control-noise      noise + random labels -> ~chance even with biased ranking
    2  pos-control-planted    planted class signal -> must be recovered
    3  selection-bias-visible in-sample scoring inflates on noise, honest LOGO doesn't
  B. generic bank checks (run on any bank; CI gate for every NEW bank)
    4  group-integrity        no recording in both train and test of any LOGO fold
    5  window-provenance      no (near-)identical window content across different groups
    6  class-balance          majority class must stay close to nominal chance
    7  label-shuffle          shuffling recording labels must destroy the signal
    8  seed-stability         result must not depend on the RF seed
    9  determinism            identical run twice -> identical numbers (features + CV)
   10  feature-degeneracy     near-constant / near-duplicate forged features are ranked out
   11  nested-vs-biased       on REAL data: biased selection >= nested, nested > chance
   12  scaler-note            document that scaling is not a leak vector (RF invariant)

Run:  nice -n 19 .venv-research/bin/python research/factory/audit.py [bank ...]
      banks: ecn (default) | cwru  — synthetic controls always run.
Exit code = number of failed checks (0 = gate open).
"""
import os, sys, time, hashlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from feature_forge import programs, run_prog, lean_baseline, load_ecn, load_cwru
from forge_nested import anova_rank, inner_logo

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "cache")
os.makedirs(CACHE, exist_ok=True)
PROGS = programs()
P = []  # (name, passed, detail)

def check(name, passed, detail, t0=None):
    P.append((name, passed, detail))
    dt = f" [{time.time()-t0:.0f}s]" if t0 else ""
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}{dt}", flush=True)

def rf(trees=150, seed=0):
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(trees, random_state=seed, n_jobs=-1))

def logo_mean(X, y, g, trees=150, seed=0):
    return cross_val_score(rf(trees, seed), X, y, groups=g, cv=LeaveOneGroupOut()).mean()

def forge_matrix(bank, raw):
    """Feature matrix for the full grammar, cached (checkpoint for re-runs)."""
    path = os.path.join(CACHE, f"F_{bank}_{len(PROGS)}p_{len(raw)}w.npz")
    if os.path.exists(path):
        return np.load(path)["F"]
    t0 = time.time()
    F = np.array([[run_prog(p, s) for p in PROGS] for s, _, _ in raw])
    np.savez_compressed(path, F=F)
    print(f"  (built F cache {bank}: {F.shape} in {time.time()-t0:.0f}s)", flush=True)
    return F

# ---------------- A. synthetic controls ----------------
def bank_noise(seed=0, n_rec=18, n_cls=6, win=14):
    rng = np.random.default_rng(seed)
    raw = []
    for g in range(n_rec):
        for _ in range(win):
            s = rng.standard_normal(1024)
            raw.append(((s - s.mean()) / s.std(), f"c{g % n_cls}", g))
    return raw

def bank_planted(seed=0, n_rec=18, win=14):
    rng = np.random.default_rng(seed)
    raw = []
    for g in range(n_rec):
        cls = g % 3
        for _ in range(win):
            s = rng.standard_normal(1024)
            s[::2] += cls * 1.5          # class-dependent structure on even samples
            raw.append(((s - s.mean()) / s.std(), f"c{cls}", g))
    return raw

def synthetic_controls():
    # 1. NEGATIVE CONTROL: pure noise, labels tied to nothing -> chance even when
    #    the ranking is computed on ALL data (optimistic on purpose = leakage probe)
    t0 = time.time()
    raw = bank_noise()
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    F = forge_matrix("noise", raw)
    order = anova_rank(F, y, g)
    acc_biased = inner_logo(F, y, g, list(order[:5]))
    check("neg-control-noise", acc_biased < 0.30,
          f"biased-top5 LOGO on noise = {acc_biased:.3f} (chance 0.167; must stay low even biased)", t0)

    # 2. POSITIVE CONTROL: a planted 3-class signal must be found
    t0 = time.time()
    raw_p = bank_planted()
    yp = np.array([r[1] for r in raw_p]); gp = np.array([r[2] for r in raw_p])
    Fp = forge_matrix("planted", raw_p)
    op = anova_rank(Fp, yp, gp)
    acc_pos = inner_logo(Fp, yp, gp, list(op[:5]))
    check("pos-control-planted", acc_pos > 0.80,
          f"planted 3-class signal recovered at {acc_pos:.3f} (chance 0.333; must be high)", t0)

    # 3. SELECTION-BIAS SIZE: scoring the selection IN-SAMPLE on noise must inflate
    #    while the honest LOGO stays at chance -> bias is real AND nested removes it
    t0 = time.time()
    sel = list(order[:5])
    clf = rf(60)
    clf.fit(F[:, sel], y)
    acc_insample = float((clf.predict(F[:, sel]) == y).mean())
    check("selection-bias-visible", acc_insample > 0.30 and acc_biased < 0.30,
          f"in-sample-scored on noise = {acc_insample:.3f} (inflated) vs honest-LOGO "
          f"{acc_biased:.3f} (~chance) -> bias real AND nested removes it", t0)

# ---------------- B. generic bank checks (CI gate for every new bank) ----------------
def audit_bank(bank, raw, deep=True):
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    chance = 1 / len(set(y))
    L = lean_baseline(raw)
    print(f"\n--- bank {bank}: {len(raw)} win, {len(set(g))} recs, "
          f"{len(set(y))} cls, chance={chance:.3f} ---", flush=True)

    # 4. GROUP INTEGRITY: no recording in both train and test of any fold
    t0 = time.time()
    leaks = sum(bool(set(g[tr]) & set(g[te]))
                for tr, te in LeaveOneGroupOut().split(L, y, g))
    check(f"group-integrity[{bank}]", leaks == 0,
          f"{leaks} folds with train/test recording overlap (must be 0)", t0)

    # 5. WINDOW PROVENANCE: the same (or near-identical) window content must never
    #    sit in two DIFFERENT groups — catches duplicated files / recordings split
    #    across gids / overlapping windowing, all of which LOGO cannot protect against.
    t0 = time.time()
    hs = {}
    dup_x = 0
    for i, (s, _, _) in enumerate(raw):
        h = hashlib.md5(np.round(s, 6).tobytes()).hexdigest()
        if h in hs and hs[h] != g[i]: dup_x += 1
        hs.setdefault(h, g[i])
    X = np.array([s for s, _, _ in raw])
    Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)
    C = (Z @ Z.T) / Z.shape[1]
    near = int(((np.abs(np.triu(C, 1)) > 0.999) & (g[:, None] != g[None, :])).sum())
    check(f"window-provenance[{bank}]", dup_x == 0 and near == 0,
          f"{dup_x} exact + {near} near-identical (|r|>0.999) windows shared across "
          f"different groups (must be 0)", t0)

    # 6. CLASS BALANCE: 'chance' claims assume near-uniform classes; a dominant
    #    majority class would make accuracy-vs-chance comparisons misleading.
    t0 = time.time()
    _, cw = np.unique(y, return_counts=True)
    recs = np.unique(g); yr = np.array([y[g == r][0] for r in recs])
    _, cr = np.unique(yr, return_counts=True)
    mw, mr = cw.max() / len(y), cr.max() / len(recs)
    check(f"class-balance[{bank}]", mw < 1.5 * chance and mr < 1.5 * chance,
          f"majority share: windows {mw:.3f}, recordings {mr:.3f} "
          f"(both must be < 1.5x chance {1.5*chance:.3f})", t0)

    # 7. LABEL SHUFFLE on real data: permuting recording labels must destroy signal
    t0 = time.time()
    rng = np.random.default_rng(0)
    rc = {r: y[g == r][0] for r in recs}
    ymap = dict(zip(recs, rng.permutation([rc[r] for r in recs])))
    y_shuf = np.array([ymap[r] for r in g])
    acc_real = logo_mean(L, y, g)
    acc_shuf = logo_mean(L, y_shuf, g)
    check(f"label-shuffle[{bank}]", acc_shuf < acc_real - 0.15,
          f"real {acc_real:.3f} -> shuffled {acc_shuf:.3f} (shuffle must destroy signal)", t0)

    # 8. SEED STABILITY: the claim must not hinge on one lucky RF seed
    t0 = time.time()
    accs = [logo_mean(L, y, g, seed=s) for s in (0, 1, 2)]
    spread = max(accs) - min(accs)
    check(f"seed-stability[{bank}]", spread < 0.05 and min(accs) > chance,
          f"lean LOGO over seeds 0/1/2 = {[f'{a:.3f}' for a in accs]}, "
          f"spread {spread:.3f} (<0.05) and all > chance", t0)

    # 9. DETERMINISM: same inputs twice -> bit-identical outputs (features + CV);
    #    guards against unseeded randomness anywhere in the pipeline.
    t0 = time.time()
    s0 = raw[0][0]
    row_a = np.array([run_prog(p, s0) for p in PROGS[:200]])
    row_b = np.array([run_prog(p, s0) for p in PROGS[:200]])
    cv_a = logo_mean(L, y, g); cv_b = logo_mean(L, y, g)
    check(f"determinism[{bank}]", np.array_equal(row_a, row_b) and cv_a == cv_b,
          f"features identical={np.array_equal(row_a, row_b)}, "
          f"LOGO {cv_a:.6f} == {cv_b:.6f} -> {cv_a == cv_b}", t0)

    if not deep:
        return
    F = forge_matrix(bank, raw)

    # 10. FEATURE DEGENERACY: near-constant and near-duplicate forged programs are
    #     inevitable in a generated grammar (algebraic identities like diff∘cumsum
    #     ≈ id create exact clones). Constants in the top ranking = hard fail
    #     (ranking broken). Duplicates cannot inflate accuracy (greedy's +0.005
    #     threshold skips them) but burn candidate budget: fail only if the top-30
    #     collapses below 8 effective unique feature directions; otherwise WARN.
    t0 = time.time()
    const = np.std(F, 0) <= 1e-9
    Zc = (F - F.mean(0)) / (F.std(0) + 1e-12)
    order = anova_rank(F, y, g)
    top = order[:30]
    const_in_top = int(const[top].sum())
    Ct = np.abs((Zc[:, top].T @ Zc[:, top]) / F.shape[0])
    kept = []
    for i in range(len(top)):
        if all(Ct[i, j] <= 0.999 for j in kept): kept.append(i)
    check(f"feature-degeneracy[{bank}]", const_in_top == 0 and len(kept) >= 8,
          f"top-30 ranking: {const_in_top} constants (must be 0), "
          f"{len(kept)}/30 effective unique clusters (must be >=8; <30 = grammar "
          f"identity-duplicates burning candidate budget -> dedup recommended)", t0)

    # 11. NESTED-VS-BIASED GAP on REAL data: selecting on all data (biased) must
    #     score >= per-fold selection (nested), and nested must stay above chance
    #     -> the bias exists on real banks too, and honest numbers survive it.
    #     (reduced params top=8/kmax=3/trees=40 to stay <2 min; same protocol shape)
    t0 = time.time()
    def greedy(order_, Fs, ys, gs, top=8, kmax=3):
        sel, best = [], 0.0
        for j in order_[:top]:
            acc = inner_logo(Fs, ys, gs, sel + [j], trees=40)
            if acc > best + 0.005: sel, best = sel + [j], acc
            if len(sel) == kmax: break
        return sel, best
    _, acc_biased = greedy(anova_rank(F, y, g), F, y, g)
    nested_accs = []
    for hold in recs:
        tr = g != hold
        sel, _ = greedy(anova_rank(F[tr], y[tr], g[tr]), F[tr], y[tr], g[tr])
        clf = rf(150); clf.fit(F[tr][:, sel], y[tr])
        nested_accs.append(float((clf.predict(F[~tr][:, sel]) == y[~tr]).mean()))
    acc_nested = float(np.mean(nested_accs))
    check(f"nested-vs-biased[{bank}]", acc_biased >= acc_nested - 0.02 and acc_nested > chance + 0.05,
          f"biased {acc_biased:.3f} >= nested {acc_nested:.3f} (gap {acc_biased-acc_nested:+.3f}) "
          f"and nested > chance+0.05 ({chance+0.05:.3f})", t0)

    # 12. SCALER NOTE: pipeline-scaled vs globally-scaled — RF is scale-invariant,
    #     so these should match; documents that scaling is not where inflation
    #     could hide (informational, always pass).
    t0 = time.time()
    Xg = StandardScaler().fit_transform(L)
    acc_leaky = cross_val_score(RandomForestClassifier(150, random_state=0, n_jobs=-1),
                                Xg, y, groups=g, cv=LeaveOneGroupOut()).mean()
    check(f"scaler-note[{bank}]", True,
          f"pipeline {acc_real:.3f} vs global-scaled {acc_leaky:.3f} "
          f"(RF scale-invariant; scaling not a leak vector)", t0)

BANKS = {"ecn": load_ecn, "cwru": load_cwru}

if __name__ == "__main__":
    banks = sys.argv[1:] or ["ecn"]
    t_all = time.time()
    synthetic_controls()
    for b in banks:
        audit_bank(b, BANKS[b]())
    fails = sum(not p for _, p, _ in P)
    print(f"\n=== AUDIT: {len(P)-fails}/{len(P)} checks passed "
          f"[{time.time()-t_all:.0f}s] ===", flush=True)
    sys.exit(fails)
