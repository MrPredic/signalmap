"""M1: SINDy on a 3D delay embedding of the single observable (smoothed derivatives).
Does a SPARSE nonlinear law in delay coords predict? Multi-tau sweep, keep best.
"""
import numpy as np, pysindy as ps
from m_common import get_obs, HORIZON, score

x, dt = get_obs()
def embed(x, tau, d=3):
    n=len(x)-(d-1)*tau
    return np.stack([x[i*tau:i*tau+n] for i in range(d)],axis=1)

split=6000
best=None
for tau in [5,8,10,12,15,20]:
    Z=embed(x,tau); Ztr=Z[:split]
    model=ps.SINDy(optimizer=ps.STLSQ(threshold=0.05),
                   feature_library=ps.PolynomialLibrary(degree=2),
                   differentiation_method=ps.SmoothedFiniteDifference())
    try:
        model.fit(Ztr, t=dt)
        z0=Z[split]
        sim=model.simulate(z0, t=np.arange(HORIZON)*dt)
        r2=score(Z[split:split+HORIZON,0], sim[:,0])
        nnz=int(np.count_nonzero(model.coefficients()))
    except Exception as e:
        r2=float('nan'); nnz=-1
    print(f"tau={tau:2d} nnz={nnz:3d} forecast_R2={r2:.3f}")
    if best is None or (r2==r2 and (np.isnan(best[0]) or r2>best[0])): best=(r2,tau,nnz)
print(f"\nM1 delay-SINDy BEST: tau={best[1]} nnz={best[2]} forecast_R2={best[0]:.3f}")
