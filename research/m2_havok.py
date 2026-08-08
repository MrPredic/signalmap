"""M2: HAVOK / Hankel-DMD (Brunton 2017). Delay-embed -> SVD -> linear Koopman model
on dominant modes with the last mode as intermittent forcing. Established single-observable
reconstruction. Reports forecast R2 of x and whether a low-rank LINEAR model suffices.
"""
import numpy as np
from numpy.linalg import svd, lstsq
from m_common import get_obs, HORIZON, score

x, dt = get_obs()
q=100          # Hankel delays
r=11           # retained modes (r-1 state + 1 forcing)
split=6000
H=np.stack([x[i:i+len(x)-q] for i in range(q)],axis=0)   # (q, m)
U,S,Vt=svd(H[:, :split-q], full_matrices=False)
V=Vt[:r].T                                                # (t, r) time-delay coords
# discrete linear model on first r-1 modes, forced by mode r
Vs=V[:-1]; Vn=V[1:]
A,_,_,_=lstsq(Vs, Vn, rcond=None)                         # V_{k+1} = V_k A
# simulate from split
# reconstruct full V over all time using same U,S projection for continuation start
Hall=np.stack([x[i:i+len(x)-q] for i in range(q)],axis=0)
Vall=(np.diag(1/S[:r])@U[:,:r].T@Hall).T                  # project all onto modes
v=Vall[split].copy(); traj=[v.copy()]
for _ in range(HORIZON):
    v=v@A; traj.append(v.copy())
traj=np.array(traj)
# x is recovered from mode reconstruction: H ~ U S V^T ; row 0 of H is x
recon_x=(U[:,:r]@np.diag(S[:r])@traj.T)[0]
true_x=x[split:split+HORIZON]
r2=score(true_x, recon_x)
print(f"M2 HAVOK: q={q} r={r} forecast_R2={r2:.3f}")
print(f"  (linear Koopman model on {r-1}+1 delay modes; forcing mode captured but here free-run)")
