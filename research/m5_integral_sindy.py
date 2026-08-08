"""M5: integral (weak-form) SINDy on delay coords -- avoid derivative estimation entirely.
Fit z(t)-z(0) = (cumtrapz Theta) Xi, then sequential-threshold for sparsity. Simulate & score.
If this beats M1's dense fit at similar R2, the 26 dense terms were derivative-noise artifacts.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from numpy.linalg import lstsq
from m_common import get_obs, HORIZON, score
x, dt = get_obs()
tau=10; d=3; split=6000
n=len(x)-(d-1)*tau
Z=np.stack([x[i*tau:i*tau+n] for i in range(d)],axis=1)

def theta(Z):
    T=[np.ones(len(Z))]+[Z[:,i] for i in range(d)]
    for i in range(d):
        for j in range(i,d): T.append(Z[:,i]*Z[:,j])
    return np.stack(T,1)
names=['1']+[f'z{i}' for i in range(d)]+[f'z{i}z{j}' for i in range(d) for j in range(i,d)]

Th=theta(Z[:split])
G=cumulative_trapezoid(Th, dx=dt, axis=0, initial=0)      # integral of library
Y=Z[:split]-Z[0]
Xi,_,_,_=lstsq(G, Y, rcond=None)
# sequential thresholding
for _ in range(10):
    small=np.abs(Xi)<0.05
    Xi[small]=0
    for k in range(d):
        big=~small[:,k]
        if big.any(): Xi[big,k]=lstsq(G[:,big], Y[:,k], rcond=None)[0]
nnz=int(np.count_nonzero(Xi))

def f(t,z):
    zt=np.array(z)[None,:]; return (theta(zt)@Xi).ravel()
try:
    sim=solve_ivp(f,(0,HORIZON*dt),Z[split],t_eval=np.arange(HORIZON)*dt,rtol=1e-6,atol=1e-9,max_step=dt).y.T
    r2=score(Z[split:split+HORIZON,0], sim[:,0])
except Exception as e:
    r2=float('nan')
print(f"M5 integral-SINDy: tau={tau} nnz={nnz} forecast_R2={r2:.3f}")
active=[f'{names[i]}->z{k}' for k in range(d) for i in range(len(names)) if abs(Xi[i,k])>1e-9]
print("  active terms:", active[:20])
