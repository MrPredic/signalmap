"""M8: honest multi-start re-evaluation of the prediction 'winners' (their single-start
numbers are suspect after M7). Same 10-start protocol for HAVOK and dense delay-SINDy on Lorenz.
Report mean/std/min forecast R2 -> the trustworthy number.
"""
import numpy as np, pysindy as ps
from numpy.linalg import svd, lstsq
from sklearn.metrics import r2_score
from m_common import get_obs, HORIZON
x, dt = get_obs()

def multistart(predict_fn, split, n):
    starts=np.linspace(split, n-HORIZON-1, 10).astype(int)
    r2=[]
    for s0 in starts:
        try: r2.append(predict_fn(s0))
        except Exception: r2.append(np.nan)
    return np.array(r2)

# ---- dense delay-SINDy (threshold 0.05) ----
tau=10; d=3; n=len(x)-(d-1)*tau
Z=np.stack([x[i*tau:i*tau+n] for i in range(d)],axis=1); split=6000
m=ps.SINDy(optimizer=ps.STLSQ(threshold=0.05), feature_library=ps.PolynomialLibrary(degree=2),
           differentiation_method=ps.SmoothedFiniteDifference()); m.fit(Z[:split],t=dt)
def sindy_pred(s0):
    sim=m.simulate(Z[s0], t=np.arange(HORIZON)*dt); return r2_score(Z[s0:s0+HORIZON,0], sim[:,0])
r=multistart(sindy_pred, split, n)
print(f"dense delay-SINDy (nnz={int(np.count_nonzero(m.coefficients()))}): "
      f"R2 mean={np.nanmean(r):.3f} std={np.nanstd(r):.3f} min={np.nanmin(r):.3f}")

# ---- HAVOK ----
q=100; rmod=11
H=np.stack([x[i:i+len(x)-q] for i in range(q)],axis=0)
U,S,Vt=svd(H[:, :split-q], full_matrices=False)
Vall=(np.diag(1/S[:rmod])@U[:,:rmod].T@H).T
Vs=Vall[:split-q-1]; Vn=Vall[1:split-q]; A,_,_,_=lstsq(Vs,Vn,rcond=None)
def havok_pred(s0):
    v=Vall[s0].copy(); traj=[v.copy()]
    for _ in range(HORIZON): v=v@A; traj.append(v.copy())
    recon=(U[:,:rmod]@np.diag(S[:rmod])@np.array(traj).T)[0]
    return r2_score(x[s0:s0+HORIZON], recon[:HORIZON])
# HAVOK start indices are into the delay/mode timeline; keep within range
rh=multistart(havok_pred, split, len(Vall)-1)
print(f"HAVOK (linear, rank {rmod}): R2 mean={np.nanmean(rh):.3f} std={np.nanstd(rh):.3f} min={np.nanmin(rh):.3f}")
print("\nTrustworthy = multi-start mean. Single-start numbers from M1/M2 were optimistic.")
