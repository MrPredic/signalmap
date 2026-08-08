"""Method-factory harness: benchmark systems + causal delay embedding + HONEST multi-metric
evaluation. A method only implements fit(x,dt) and free_run(context,H). Everything else
(multi-start forecast, sparsity, boundedness, attractor-statistics) is measured here uniformly.

Metrics are deliberately plural: pointwise forecast can FAIL while attractor-statistics PASS
(that is how we rescue 'dirty gold' -- via the right metric, never by lowering thresholds).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score

# ---------- benchmark systems: single observable x(t), standardized ----------
def _traj(system, x0, T, dt):
    t = np.arange(0, T, dt)
    sol = solve_ivp(system, (t[0], t[-1]), x0, t_eval=t, rtol=1e-9, atol=1e-9).y
    x = sol[0]; return (x - x.mean())/x.std()

SYSTEMS = {
    "lorenz":  (lambda t,s:[10*(s[1]-s[0]), s[0]*(28-s[2])-s[1], s[0]*s[1]-8/3*s[2]], [1.,1.,1.]),
    "rossler": (lambda t,s:[-s[1]-s[2], s[0]+0.2*s[1], 0.2+s[2]*(s[0]-5.7)], [1.,1.,1.]),
    "thomas":  (lambda t,s:[np.sin(s[1])-0.19*s[0], np.sin(s[2])-0.19*s[1], np.sin(s[0])-0.19*s[2]], [1.,1.,1.]),
}
def get_observable(name, T=80, dt=0.01):
    system, x0 = SYSTEMS[name]; return _traj(system, x0, T, dt), dt

# ---------- shared causal delay embedding (known at time t) ----------
def embed_causal(x, tau, d):
    # z[t] = [x[t], x[t-tau], ..., x[t-(d-1)tau]]  defined for t >= (d-1)*tau
    off = (d-1)*tau
    return np.stack([x[off - i*tau : len(x) - i*tau] for i in range(d)], axis=1), off

# ---------- metrics ----------
HORIZON = 150
def multistart_forecast(method, x, split, K=10, H=HORIZON):
    starts = np.linspace(split, len(x)-H-1, K).astype(int)
    r2 = []
    for s0 in starts:
        try:
            pred = method.free_run(x[:s0], H)
            n = min(len(pred), H)
            r2.append(r2_score(x[s0:s0+n], pred[:n]) if n >= H//2 else np.nan)
        except Exception:
            r2.append(np.nan)
    r2 = np.array(r2)
    return dict(mean=np.nanmean(r2), std=np.nanstd(r2), min=np.nanmin(r2))

def attractor_stats(method, x, split, longH=2000):
    # rescue-metric: does the long free-run reproduce the value-distribution + autocorrelation,
    # even if pointwise forecast fails? lower dist = better. bounded = didn't blow up.
    try:
        sim = method.free_run(x[:split], longH)
    except Exception:
        return dict(bounded=False, wass=np.nan, acf_l2=np.nan)
    if not np.all(np.isfinite(sim)) or np.max(np.abs(sim)) > 20*np.std(x):
        return dict(bounded=False, wass=np.nan, acf_l2=np.nan)
    true = x[split:split+longH]
    def acf(v, L=100):
        v=(v-v.mean())/(v.std()+1e-12); n=len(v)
        return np.correlate(v,v,'full')[n-1:n-1+L]/n
    wass = wasserstein_distance(sim, true)
    acf_l2 = float(np.sqrt(np.mean((acf(sim)-acf(true))**2)))
    return dict(bounded=True, wass=float(wass), acf_l2=acf_l2)

def assay(method, x, split, H=HORIZON):
    # CHEAP first-pass: single free-run, quick R2 on first H//3 + boundedness. Ranks where to
    # spend full multi-start compute. Prospecting: sample+assay before you dig.
    try:
        pred = method.free_run(x[:split], H)
        if not np.all(np.isfinite(pred)): return -9.9
        n = min(len(pred), H//3)
        return r2_score(x[split:split+n], pred[:n])
    except Exception:
        return -9.9
