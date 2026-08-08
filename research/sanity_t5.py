"""T5 signal-vs-noise via surrogate-data tests. Label-free, threshold-free (z vs null).
L1 shuffled+maxACF -> any temporal structure. L2 IAAFT+time-reversal -> nonlinear determinism.
Run: <local-path>/signalmap/.venv-research/bin/python research/sanity_t5.py
"""
import numpy as np
from scipy.integrate import solve_ivp
rng = np.random.default_rng(0)

# ---- test signals ----
N = 4000
def lorenz(t, s):
    x, y, z = s
    return [10*(y-x), x*(28-z)-y, x*y-8/3*z]
t = np.arange(0, 20, 0.002)
lx = solve_ivp(lorenz, (t[0], t[-1]), [1., 1., 1.], t_eval=t, rtol=1e-9, atol=1e-9).y[0][:N]
white = rng.standard_normal(N)
# 1/f (pink) noise: linear stochastic, correlated but NOT nonlinear
f = np.fft.rfftfreq(N, 1.0); f[0] = f[1]
spec = (rng.standard_normal(len(f)) + 1j*rng.standard_normal(len(f))) / np.sqrt(f)
pink = np.fft.irfft(spec, N); pink = (pink - pink.mean())/pink.std()
sine = np.sin(np.linspace(0, 80*np.pi, N)) + 0.1*rng.standard_normal(N)
signals = {"white": white, "pink_1/f": pink, "noisy_sine": sine, "lorenz_x": lx}

# ---- statistics ----
def max_acf(x, maxlag=200):
    x = (x - x.mean())/x.std(); n = len(x)
    ac = np.correlate(x, x, 'full')[n-1:n-1+maxlag]/n
    return np.max(np.abs(ac[1:]))

def pred_err(x, m=3, tau=5, h=5, theiler=20):
    # nonlinear prediction error on delay embedding; LOW for deterministic flows,
    # HIGH for phase-randomized surrogates with identical spectrum. (Kantz-Schreiber)
    x = (x - x.mean())/x.std()
    n = len(x) - (m-1)*tau - h
    emb = np.stack([x[i*tau:i*tau+n] for i in range(m)], axis=1)  # (n,m)
    fut = x[(m-1)*tau + h:(m-1)*tau + h + n]
    errs = []
    for i in range(0, n, 3):  # subsample for speed
        d = np.sum((emb - emb[i])**2, axis=1)
        d[max(0, i-theiler):i+theiler+1] = np.inf  # exclude temporal neighbors
        j = np.argmin(d)
        errs.append((fut[i] - fut[j])**2)
    return np.sqrt(np.mean(errs))

def iaaft(x, iters=100):  # surrogate: preserve power spectrum + amplitude dist, randomize phase
    amp = np.abs(np.fft.rfft(x)); sortx = np.sort(x)
    s = rng.permutation(x)
    for _ in range(iters):
        ph = np.angle(np.fft.rfft(s))
        s = np.fft.irfft(amp*np.exp(1j*ph), len(x))
        s = sortx[np.argsort(np.argsort(s))]
    return s

def ztest(x, stat, surrogate, n=40, lower=False):
    # lower=True: signal when real statistic is a LOW outlier vs null (e.g. prediction error)
    real = abs(stat(x))
    null = np.array([abs(stat(surrogate(x))) for _ in range(n)])
    z = (null.mean() - real)/(null.std() + 1e-12) if lower else (real - null.mean())/(null.std() + 1e-12)
    return real, z

print(f"{'signal':12s} | L1 z(maxACF,shuf) | L2 z(trev,IAAFT) | verdict")
print("-"*66)
for name, x in signals.items():
    _, z1 = ztest(x, max_acf, lambda a: rng.permutation(a))
    _, z2 = ztest(x, pred_err, iaaft, lower=True)
    if z1 < 3:
        v = "UNSTRUCTURED (noise)"
    elif z2 < 3:
        v = "linear-correlated"
    else:
        v = "NONLINEAR-deterministic"
    print(f"{name:12s} | {z1:16.1f} | {z2:15.1f} | {v}")

# sanity expectations
print("\nExpected: white=noise, pink=linear-correlated, sine/lorenz=structured (sine periodic->L1 high),")
print("lorenz -> NONLINEAR-deterministic. FP-control: white L1 z must be < 3.")
