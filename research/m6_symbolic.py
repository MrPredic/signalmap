"""M6: symbolic regression (gplearn). Search a CLOSED-FORM one-step map
x(t+1) = f(x(t), x(t-tau), x(t-2tau)); iterate it HORIZON steps; report forecast R2 + formula.
Directly targets the 'sparse closed form' that is the open part of T1.
"""
import numpy as np
from gplearn.genetic import SymbolicRegressor
from m_common import get_obs, HORIZON, score
x, dt = get_obs()
tau=10; d=3; split=6000
n=len(x)-(d-1)*tau
Z=np.stack([x[i*tau:i*tau+n] for i in range(d)],axis=1)   # cols: x(t),x(t+tau),x(t+2tau)
# predict next value of leading coord: target = Z[t+1,0]
Xtr=Z[:split-1]; ytr=Z[1:split,0]
est=SymbolicRegressor(population_size=2000, generations=20, stopping_criteria=1e-4,
    p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, p_point_mutation=0.1,
    function_set=('add','sub','mul','div','sin','cos'), parsimony_coefficient=0.001,
    max_samples=0.9, verbose=0, random_state=0, n_jobs=1)
est.fit(Xtr, ytr)
# iterate closed-form map from split
z=Z[split].copy().astype(float); pred=[z[0]]
for _ in range(HORIZON-1):
    nx=est.predict(z[None,:])[0]
    z=np.array([nx, z[0], z[1]])   # shift delay window (approx: new leading, old ones slide)
    pred.append(nx)
r2=score(Z[split:split+HORIZON,0], np.array(pred))
print(f"M6 symbolic (gplearn): forecast_R2={r2:.3f}")
print("  program:", est._program)
print("  length:", est._program.length_)
