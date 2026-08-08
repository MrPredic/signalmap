"""M7: stress the M4 winner (delay tau=10, poly2, STLSQ threshold=0.5 -> sparse+predictive).
Verify honestly: (a) multi-start forecast R2 mean/std, (b) transfer to Rossler,
(c) long-horizon boundedness (does the learned sparse ODE stay on an attractor, not blow up).
"""
import numpy as np, pysindy as ps
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score
from m_common import HORIZON

def obs_of(system, T=80, dt=0.01):
    t=np.arange(0,T,dt)
    sol=solve_ivp(system,(t[0],t[-1]),[1.,1.,1.],t_eval=t,rtol=1e-9,atol=1e-9).y
    x=sol[0]; return (x-x.mean())/x.std(), dt
lorenz=lambda t,s:[10*(s[1]-s[0]), s[0]*(28-s[2])-s[1], s[0]*s[1]-8/3*s[2]]
rossler=lambda t,s:[-s[1]-s[2], s[0]+0.2*s[1], 0.2+s[2]*(s[0]-5.7)]

def fit_and_test(name, system, tau=10, d=3, th=0.5):
    x,dt=obs_of(system)
    n=len(x)-(d-1)*tau
    Z=np.stack([x[i*tau:i*tau+n] for i in range(d)],axis=1)
    split=int(0.6*n)
    m=ps.SINDy(optimizer=ps.STLSQ(threshold=th),
               feature_library=ps.PolynomialLibrary(degree=2),
               differentiation_method=ps.SmoothedFiniteDifference())
    m.fit(Z[:split], t=dt)
    nnz=int(np.count_nonzero(m.coefficients()))
    # (a) multi-start forecast
    starts=np.linspace(split, n-HORIZON-1, 10).astype(int)
    r2s=[]
    for s0 in starts:
        try:
            sim=m.simulate(Z[s0], t=np.arange(HORIZON)*dt)
            r2s.append(r2_score(Z[s0:s0+HORIZON,0], sim[:len(Z[s0:s0+HORIZON,0]),0]))
        except Exception: r2s.append(np.nan)
    r2s=np.array(r2s);
    # (c) boundedness: long free-run
    try:
        long=m.simulate(Z[split], t=np.arange(2000)*dt)
        bounded = np.all(np.isfinite(long)) and np.max(np.abs(long)) < 10*np.max(np.abs(Z))
    except Exception:
        bounded=False
    print(f"{name:9s} nnz={nnz:2d}  forecast_R2 mean={np.nanmean(r2s):.3f} std={np.nanstd(r2s):.3f} "
          f"min={np.nanmin(r2s):.3f}  long-run bounded={bounded}")
    return np.nanmean(r2s), nnz, bounded

print("M7 verify winner (sparse delay-SINDy, threshold=0.5):")
fit_and_test("Lorenz", lorenz)
fit_and_test("Rossler", rossler)
print("\nDone-criterion recap: sparse(nnz<=12) + mean forecast_R2>0.6 + bounded, reproducibly.")
