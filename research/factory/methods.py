"""Method bank. A method = a class with .fit(x,dt) and .free_run(context,H)->H obs values,
plus .nnz (interpretability, None if n/a) and .name. Add a new method = ~15 lines + append
to REGISTRY. That is the whole 'factory': breadth becomes a quantity game, not hand-work.

TEMPLATE (copy this):
    class MyMethod:
        name = "my_method"; nnz = None
        def fit(self, x, dt): ...            # learn from the training observable
        def free_run(self, context, H):      # context = observable up to start; return H future obs
            ...
    REGISTRY.append(MyMethod())
"""
import numpy as np, pysindy as ps
from numpy.linalg import svd, lstsq
from harness import embed_causal

class HAVOK:
    name = "havok_linear"; nnz = None
    def __init__(self, q=100, r=11): self.q=q; self.r=r
    def fit(self, x, dt):
        q,r=self.q,self.r
        H=np.stack([x[i:i+len(x)-q] for i in range(q)],axis=0)
        U,S,_=svd(H, full_matrices=False)
        self.U,self.S=U[:,:r],S[:r]
        V=(np.diag(1/self.S)@self.U.T@H).T
        self.A,_,_,_=lstsq(V[:-1], V[1:], rcond=None)
    def free_run(self, context, Hn):
        win=context[-self.q:]
        v=np.diag(1/self.S)@self.U.T@win
        out=[]
        for _ in range(Hn):
            v=v@self.A
            out.append((self.U@np.diag(self.S)@v)[-1])   # latest entry = current x
        return np.array(out)

def _library(kind, degree):
    if kind=='poly':    return ps.PolynomialLibrary(degree=degree)
    if kind=='fourier': return ps.FourierLibrary(n_frequencies=degree)
    if kind=='poly+fourier':
        return ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=degree), ps.FourierLibrary(n_frequencies=2)])
    raise ValueError(kind)

class DelaySINDy:
    def __init__(self, threshold=0.05, tau=10, d=3, degree=2, lib='poly'):
        self.th=threshold; self.tau=tau; self.d=d; self.degree=degree; self.lib=lib
        self.name=f"sindy_{lib}{degree}_tau{tau}_d{d}_th{threshold}"; self.nnz=None
    def fit(self, x, dt):
        Z,_=embed_causal(x, self.tau, self.d)
        self.m=ps.SINDy(optimizer=ps.STLSQ(threshold=self.th),
                        feature_library=_library(self.lib, self.degree),
                        differentiation_method=ps.SmoothedFiniteDifference())
        self.m.fit(Z, t=dt); self.dt=dt
        self.nnz=int(np.count_nonzero(self.m.coefficients()))
    def free_run(self, context, Hn):
        tau,d=self.tau,self.d
        z0=np.array([context[-1-i*tau] for i in range(d)])
        sim=self.m.simulate(z0, t=np.arange(Hn)*self.dt)
        return sim[:,0]

REGISTRY = [
    HAVOK(),
    DelaySINDy(threshold=0.05),   # dense
    DelaySINDy(threshold=0.5),    # sparse
]
