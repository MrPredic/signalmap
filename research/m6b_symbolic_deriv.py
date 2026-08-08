"""M6b: symbolic SINDy done right. Fit a CLOSED-FORM expression for the derivative of each
delay coord (3 gplearn regressors), assemble an ODE, integrate, score. Avoids the one-step
identity trap of M6. Yields human-readable law + forecast R2.
"""
import numpy as np, pysindy as ps
from gplearn.genetic import SymbolicRegressor
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score
from m_common import get_obs, HORIZON
x, dt = get_obs()
tau=10; d=3; split=6000
n=len(x)-(d-1)*tau
Z=np.stack([x[i*tau:i*tau+n] for i in range(d)],axis=1)
dZ=ps.SmoothedFiniteDifference()._differentiate(Z, dt)   # smoothed derivatives
regs=[]
for k in range(d):
    est=SymbolicRegressor(population_size=1500, generations=15, stopping_criteria=1e-4,
        function_set=('add','sub','mul'), parsimony_coefficient=0.002,
        max_samples=0.9, verbose=0, random_state=k, n_jobs=1)
    est.fit(Z[:split], dZ[:split,k]); regs.append(est)
    print(f"  dz{k}/dt = {est._program}   (len {est._program.length_})")
def f(t,z):
    zz=np.array(z)[None,:]; return np.array([r.predict(zz)[0] for r in regs])
try:
    sim=solve_ivp(f,(0,HORIZON*dt),Z[split],t_eval=np.arange(HORIZON)*dt,rtol=1e-6,atol=1e-9,max_step=dt).y.T
    r2=r2_score(Z[split:split+len(sim),0], sim[:,0]) if len(sim)>=HORIZON//2 else float('nan')
except Exception:
    r2=float('nan')
print(f"M6b symbolic-derivative: forecast_R2={r2:.3f}")
