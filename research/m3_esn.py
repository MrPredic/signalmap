"""M3: Echo-State Reservoir. NOT sparse/interpretable, but the ACHIEVABILITY CEILING:
how predictable is single-observable Lorenz at all over the horizon? Calibrates whether
T1's failure is about sparse-law-extraction (if ESN succeeds) or intrinsic (if ESN fails too).
"""
import numpy as np
from m_common import get_obs, HORIZON, score
rng=np.random.default_rng(0)

x, dt = get_obs()
split=6000
N=400; sr=0.95; leak=0.3; reg=1e-6
Win=(rng.standard_normal((N,1)))*0.5
W=rng.standard_normal((N,N)); W*=sr/np.max(np.abs(np.linalg.eigvals(W)))
def run(u_seq, s0=None):
    s=np.zeros(N) if s0 is None else s0.copy(); S=[]
    for u in u_seq:
        s=(1-leak)*s+leak*np.tanh(Win@[u]+W@s); S.append(s.copy())
    return np.array(S), s
S,s_last=run(x[:split])
# ridge readout one-step-ahead
Xr=S[:-1]; Y=x[1:split]
Wout=np.linalg.solve(Xr.T@Xr+reg*np.eye(N), Xr.T@Y)
# free-run from split
s=s_last.copy(); u=x[split-1]; pred=[]
for _ in range(HORIZON):
    s=(1-leak)*s+leak*np.tanh(Win@[u]+W@s); u=s@Wout; pred.append(u)
r2=score(x[split:split+HORIZON], np.array(pred))
print(f"M3 ESN ceiling: N={N} forecast_R2={r2:.3f}")
print("  (if high: observable IS predictable -> T1 gap is sparse-law extraction, not intrinsic)")
