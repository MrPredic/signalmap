"""Third domain: MFPT bearing (DIFFERENT test rig than CWRU) via MathWorks mirror.
3 classes (baseline / inner / outer race fault), 20 recordings, LOGO.
sr-confound killed: baseline is 97656 Hz, faults 48828 Hz -> decimate baseline
by 2 so every window sees the same timebase (else the classifier learns sr).
Runs the full stack: lean-vs-heavy head-to-head + forge + NESTED forge.
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/mfpt_run.py
"""
import numpy as np, glob, os, time, warnings, antropy as ant, pywt
import scipy.io as sio
from scipy.signal import detrend, welch, decimate
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from feature_forge import forge, group_perm_p, lean_baseline, W
from forge_nested import nested
warnings.filterwarnings("ignore")

def load_mfpt():
    raw = []
    for gid, f in enumerate(sorted(glob.glob("<local-path>/signalmap/data/mfpt/*/*.mat"))):
        b = sio.loadmat(f)["bearing"]
        x = b["gs"][0, 0].ravel().astype(float)
        sr = int(b["sr"][0, 0].ravel()[0])
        if sr == 97656: x = decimate(x, 2)          # harmonize to 48828 Hz
        name = os.path.basename(f)
        cls = ("baseline" if "baseline" in name else
               "inner" if "Inner" in name else "outer")
        for k in range(len(x) // W):
            s = detrend(np.ascontiguousarray(x[k * W:(k + 1) * W], float))
            raw.append(((s - s.mean()) / (s.std() + 1e-12), cls, gid))
    return raw

# heavy competitors (identical impls to ecn/cwru head-to-head)
def rqa(x, m=3, tau=5, rr=0.1, sub=400):
    x = x[:sub]; n = len(x) - (m - 1) * tau
    emb = np.stack([x[i * tau:i * tau + n] for i in range(m)], 1)
    D = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1))
    thr = np.quantile(D, rr); R = (D < thr).astype(int)
    diagpts = 0; tot = R.sum()
    for k in range(1, n):
        d = np.diag(R, k); run = 0
        for val in d:
            if val: run += 1
            else:
                if run >= 2: diagpts += run
                run = 0
        if run >= 2: diagpts += run
    return R.mean(), 2 * diagpts / (tot + 1e-9)
def went(x, wav="db4", lvl=5):
    c = pywt.wavedec(x, wav, level=lvl)
    e = np.array([np.sum(ci ** 2) for ci in c]); p = e / (e.sum() + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

if __name__ == "__main__":
    raw = load_mfpt()
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    print(f"MFPT: {len(raw)} windows, {len(set(g))} recordings, chance={1/len(set(y)):.3f}", flush=True)

    def logo(X, trees=200):
        return cross_val_score(make_pipeline(StandardScaler(),
                               RandomForestClassifier(trees, random_state=0, n_jobs=-1)),
                               X, y, groups=g, cv=LeaveOneGroupOut()).mean()
    def build(fns):
        t0 = time.time(); cols = []
        for s, _, _ in raw:
            row = []
            for fn in fns:
                r = fn(s); row += list(r) if isinstance(r, tuple) else [r]
            cols.append(row)
        return np.array(cols), 1000 * (time.time() - t0) / len(raw)

    Xl = lean_baseline(raw); t0 = time.time(); _ = lean_baseline(raw[:200])
    ms_lean = 1000 * (time.time() - t0) / 200
    print(f"LEAN  perm3+psd_slope  LOGO={logo(Xl):.3f} @ {ms_lean:.2f}ms", flush=True)
    for nm, fns in [("HEAVY rqa", [rqa]), ("HEAVY wavelet", [went]), ("HEAVY rqa+wav", [rqa, went])]:
        X, ms = build(fns)
        print(f"{nm:22s} LOGO={logo(X):.3f} @ {ms:.2f}ms", flush=True)
    pl = group_perm_p(Xl, y, g, logo(Xl))
    print(f"lean group-perm p = {pl:.3f}", flush=True)

    forge("MFPT (biased UB)", raw)
    print("\n=== MFPT nested ===", flush=True)
    nested("MFPT", raw, top=20)
