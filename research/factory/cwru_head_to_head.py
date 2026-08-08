"""Second-domain replication of the efficiency moat: CWRU bearing vibration.
Same design as ecn_head_to_head.py — lean O(n log n) features vs heavy RQA/wavelet,
identical leakage-free task: classify FAULT TYPE from 1024-sample windows,
Leave-One-RECORDING-Out (each class has 4 load replicates -> honest LOGO).
Adds what ECN lacked: per-fold std + group-label permutation p-value for OURS.
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/cwru_head_to_head.py
"""
import numpy as np, os, io, time, urllib.request, warnings, antropy as ant, pywt
import scipy.io as sio
from scipy.signal import detrend, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
warnings.filterwarnings("ignore")

BASE = "https://raw.githubusercontent.com/zerothphase/CWRU/master/Data/12k_DE/"
CACHE = "<local-path>/signalmap/data/cwru_mats"
CLASSES = {  # 6 classes x 4 load replicates = 24 recordings
    "IR007": ["IR007_0.mat","IR007_1.mat","IR007_2.mat","IR007_3.mat"],
    "IR021": ["IR021_0.mat","IR021_1.mat","IR021_2.mat","IR021_3.mat"],
    "B007":  ["B007_0.mat","B007_1.mat","B007_2.mat","B007_3.mat"],
    "B021":  ["B021_0.mat","B021_1.mat","B021_2.mat","B021_3.mat"],
    "OR007": ["OR007@6_0.mat","OR007@6_1.mat","OR007@6_2.mat","OR007@6_3.mat"],
    "OR021": ["OR021@6_0.mat","OR021@6_1.mat","OR021@6_2.mat","OR021@6_3.mat"],
}

def fetch(fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, fn.replace("@","_"))
    if not os.path.exists(p):
        urllib.request.urlretrieve(BASE + urllib.request.quote(fn), p)
    m = sio.loadmat(p)
    key = next(k for k in m if k.endswith("DE_time"))
    return m[key].ravel().astype(float)

# ---- identical feature sets to ECN head-to-head ----
def perm3(x): return ant.perm_entropy(x, order=3, normalize=True)
def psd_slope(x):
    f, P = welch(x, nperseg=min(256, len(x))); f, P = f[1:], P[1:]
    return np.polyfit(np.log(f), np.log(P + 1e-20), 1)[0]
def rqa(x, m=3, tau=5, rr=0.1, sub=400):
    x = x[:sub]; n = len(x) - (m - 1) * tau
    emb = np.stack([x[i * tau:i * tau + n] for i in range(m)], 1)
    D = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1))
    thr = np.quantile(D, rr); R = (D < thr).astype(int)
    rec_rate = R.mean()
    diagpts = 0; tot = R.sum()
    for k in range(1, n):
        d = np.diag(R, k); run = 0
        for val in d:
            if val: run += 1
            else:
                if run >= 2: diagpts += run
                run = 0
        if run >= 2: diagpts += run
    return rec_rate, 2 * diagpts / (tot + 1e-9)
def wavelet_entropy(x, wav="db4", lvl=5):
    c = pywt.wavedec(x, wav, level=lvl)
    e = np.array([np.sum(ci ** 2) for ci in c]); p = e / (e.sum() + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

W = 1024
raw = []
gid = 0
for cls, files in CLASSES.items():
    for fn in files:
        try:
            x = fetch(fn)
        except Exception as e:
            print(f"skip {fn}: {e}"); continue
        for k in range(len(x) // W):
            s = detrend(np.ascontiguousarray(x[k * W:(k + 1) * W], float))
            s = (s - s.mean()) / (s.std() + 1e-12)
            raw.append((s, cls, gid))
        gid += 1
y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
chance = 1 / len(set(y))
print(f"recordings={len(set(g))} windows={len(raw)} chance={chance:.3f}\n", flush=True)

def logo_scores(X, labels, trees=200):
    return cross_val_score(
        make_pipeline(StandardScaler(), RandomForestClassifier(trees, random_state=0, n_jobs=-1)),
        X, labels, groups=g, cv=LeaveOneGroupOut())
def build(fns):
    t0 = time.time(); cols = []
    for s, _, _ in raw:
        row = []
        for fn in fns:
            r = fn(s); row += list(r) if isinstance(r, tuple) else [r]
        cols.append(row)
    return np.array(cols), 1000 * (time.time() - t0) / len(raw)

print(f"{'method':28s} {'LOGO_acc':>8s} {'fold_std':>8s} {'cost_ms/win':>11s}", flush=True)
res = {}; Xlean = None
for name, fns in {
    "OURS lean (perm3+psd_slope)": [perm3, psd_slope],
    "HEAVY rqa (recrate+determ)": [rqa],
    "HEAVY wavelet_entropy": [wavelet_entropy],
    "HEAVY rqa+wavelet (combo)": [rqa, wavelet_entropy],
}.items():
    X, ms = build(fns); sc = logo_scores(X, y); res[name] = (sc.mean(), ms)
    if name.startswith("OURS"): Xlean = X
    print(f"{name:28s} {sc.mean():8.3f} {sc.std():8.3f} {ms:11.3f}", flush=True)

ours, heavy = res["OURS lean (perm3+psd_slope)"], res["HEAVY rqa+wavelet (combo)"]
print(f"\nOURS acc={ours[0]:.3f} @ {ours[1]:.2f}ms  vs HEAVY acc={heavy[0]:.3f} @ {heavy[1]:.2f}ms")
print(f"-> accuracy gap = {ours[0]-heavy[0]:+.3f}   speed advantage = {heavy[1]/max(ours[1],1e-6):.0f}x cheaper", flush=True)

# ---- group-label permutation test for OURS (the gate ECN's head-to-head skipped) ----
rng = np.random.default_rng(0)
rec_ids = np.unique(g); rec_cls = {r: y[g == r][0] for r in rec_ids}
obs = res["OURS lean (perm3+psd_slope)"][0]
NP = 100; hits = 0
for i in range(NP):
    perm = rng.permutation([rec_cls[r] for r in rec_ids])
    ymap = dict(zip(rec_ids, perm))
    yp = np.array([ymap[r] for r in g])
    if logo_scores(Xlean, yp, trees=100).mean() >= obs: hits += 1
    if (i + 1) % 20 == 0: print(f"  perm {i+1}/{NP}", flush=True)
print(f"\ngroup-permutation p-value (OURS, {NP} perms) = {(hits+1)/(NP+1):.3f}")
