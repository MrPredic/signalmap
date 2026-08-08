"""M4: follow the delay-SINDy winner toward SPARSITY. Sweep STLSQ threshold at tau=10;
find the sparsest model that still keeps forecast_R2>0.6. Answers: is sparse+predictive
achievable in delay coords, or is prediction inherently dense here?
"""
import numpy as np, pysindy as ps
from m_common import get_obs, HORIZON, score
x, dt = get_obs()
tau=10; d=3; split=6000
n=len(x)-(d-1)*tau
Z=np.stack([x[i*tau:i*tau+n] for i in range(d)],axis=1)
print(f"{'thresh':>7s} {'nnz':>4s} {'forecast_R2':>12s}")
for th in [0.02,0.05,0.1,0.2,0.3,0.5,0.8,1.2]:
    m=ps.SINDy(optimizer=ps.STLSQ(threshold=th),
               feature_library=ps.PolynomialLibrary(degree=2),
               differentiation_method=ps.SmoothedFiniteDifference())
    try:
        m.fit(Z[:split], t=dt)
        sim=m.simulate(Z[split], t=np.arange(HORIZON)*dt)
        r2=score(Z[split:split+HORIZON,0], sim[:,0]); nnz=int(np.count_nonzero(m.coefficients()))
    except Exception:
        r2=float('nan'); nnz=-1
    flag=" <-- sparse & predictive" if (r2==r2 and r2>0.6 and 0<nnz<=12) else ""
    print(f"{th:7.2f} {nnz:4d} {r2:12.3f}{flag}")
