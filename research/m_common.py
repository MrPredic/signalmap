"""Shared benchmark task for the T1 method battery: single observable x(t) of Lorenz.
Metric: short-horizon forecast R2 of x over ~1.5 Lyapunov times from a held-out start.
"""
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score

def get_obs(dt=0.01, T=80):
    def lorenz(t,s): x,y,z=s; return [10*(y-x), x*(28-z)-y, x*y-8/3*z]
    t=np.arange(0,T,dt)
    sol=solve_ivp(lorenz,(t[0],t[-1]),[1.,1.,1.],t_eval=t,rtol=1e-9,atol=1e-9).y
    x=sol[0]; x=(x-x.mean())/x.std()
    return x, dt

HORIZON=150  # steps ~ 1.5 Lyapunov times (lambda~0.9, dt=0.01)
def score(true, pred):
    n=min(len(true),len(pred),HORIZON)
    if n<HORIZON//2: return float('nan')
    return r2_score(true[:n], pred[:n])
