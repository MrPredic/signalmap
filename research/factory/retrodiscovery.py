"""Retrodiscovery seed (register B). Take laws whose closed form is KNOWN, generate data,
and measure: (1) does a method recover the EXACT symbolic form (not just forecast)?
(2) the DATA-EFFICIENCY = minimum samples / noise tolerance to recover it correctly.
Efficiency is the value metric: 'reach the known answer with a fraction of the effort'.

This is ground-truth by construction -> no fool's gold ambiguity. It is the honest bridge
from 'test many methods' to 'find real gold', and independently valuable (cheaper paths to
known results). No web needed: laws encoded here.
"""
import numpy as np, pysindy as ps
from scipy.integrate import solve_ivp

# ---- ground-truth laws: (name, rhs, x0, analytic active terms {(*eq*, term): coef}) ----
LAWS = {
    "damped_oscillator": dict(
        rhs=lambda t,s:[s[1], -4.0*s[0] - 0.3*s[1]],           # x'=v ; v'=-w^2 x - c v
        x0=[1.0, 0.0],
        truth={(0,'x1'):1.0, (1,'x0'):-4.0, (1,'x1'):-0.3}),
    "van_der_pol": dict(
        rhs=lambda t,s:[s[1], 2.0*(1-s[0]**2)*s[1]-s[0]],       # v'=mu(1-x^2)v - x
        x0=[0.5, 0.0],
        truth={(0,'x1'):1.0, (1,'x0'):-1.0, (1,'x1'):2.0, (1,'x0^2 x1'):-2.0}),
    "cubic_oscillator": dict(
        rhs=lambda t,s:[-0.1*s[0]**3+2.0*s[1]**3, -2.0*s[0]**3-0.1*s[1]**3],
        x0=[1.0, 0.0],
        truth={(0,'x0^3'):-0.1,(0,'x1^3'):2.0,(1,'x0^3'):-2.0,(1,'x1^3'):-0.1}),
    "lotka_volterra": dict(
        rhs=lambda t,s:[1.5*s[0]-1.0*s[0]*s[1], -3.0*s[1]+1.0*s[0]*s[1]],
        x0=[5.0, 3.0],
        truth={(0,'x0'):1.5,(0,'x0 x1'):-1.0,(1,'x1'):-3.0,(1,'x0 x1'):1.0}),
}

def recover(rhs, x0, T, dt, noise=0.0, seed=0):
    rng=np.random.default_rng(seed)
    t=np.arange(0,T,dt); X=solve_ivp(rhs,(t[0],t[-1]),x0,t_eval=t,rtol=1e-10,atol=1e-12).y.T
    X=X+noise*rng.standard_normal(X.shape)*X.std(0)
    m=ps.SINDy(optimizer=ps.STLSQ(threshold=0.1),
               feature_library=ps.PolynomialLibrary(degree=3),
               differentiation_method=ps.SmoothedFiniteDifference())
    m.fit(X,t=dt); names=m.get_feature_names(); C=m.coefficients()
    return names, C

def matches(names, C, truth, tol=0.15):
    # exact-form match: every truth term present & correct, and no spurious large term
    for (eq,term),val in truth.items():
        if term not in names or abs(C[eq][names.index(term)]-val) > tol*abs(val)+0.05:
            return False
    truth_idx={(eq,names.index(term)) for (eq,term) in truth}
    for eq in range(C.shape[0]):
        for j in range(C.shape[1]):
            if (eq,j) not in truth_idx and abs(C[eq][j])>0.1:
                return False
    return True

def partial_obs_recover(rhs, x0, dt=0.005, T=40, tau=10, d=4, th=0.05):
    # observe ONLY coord 0; delay-embed; recover a PREDICTIVE law; multi-start honest R2.
    from harness import multistart_forecast
    from methods import DelaySINDy
    t=np.arange(0,T,dt); X=solve_ivp(rhs,(t[0],t[-1]),x0,t_eval=t,rtol=1e-10,atol=1e-12).y[0]
    x=(X-X.mean())/(X.std()+1e-12)
    split=int(0.6*len(x))
    m=DelaySINDy(threshold=th,tau=tau,d=d)
    m.fit(x[:split],dt)
    return multistart_forecast(m,x,split), m.nnz

if __name__=="__main__":
    dt=0.005
    print("=== FULL-OBSERVATION retrodiscovery (exact symbolic form + efficiency) ===")
    for name,L in LAWS.items():
        names,C=recover(L['rhs'],L['x0'],T=40,dt=dt)
        ok=matches(names,C,L['truth'])
        minT=None
        for T in [2,4,6,8,10,15,20,30]:
            n,c=recover(L['rhs'],L['x0'],T=T,dt=dt)
            if matches(n,c,L['truth']): minT=T; break
        maxnoise=0.0
        for nz in [0.0,0.005,0.01,0.02,0.05,0.1]:
            n,c=recover(L['rhs'],L['x0'],T=40,dt=dt,noise=nz)
            if matches(n,c,L['truth']): maxnoise=nz
            else: break
        print(f"  {name:18s} exact_form={str(ok):5s}  min_data={int(minT/dt) if minT else '-':>5} samp  noise_tol<={maxnoise}")

    print("\n=== PARTIAL-OBSERVATION retrodiscovery (single observable -> predictive law) ===")
    for name,L in LAWS.items():
        try:
            fc,nnz=partial_obs_recover(L['rhs'],L['x0'])
            verdict="RECOVERED" if fc['mean']>0.6 else ("weak" if fc['mean']>0 else "FAIL")
            print(f"  {name:18s} nnz={nnz:2d}  multistart_R2 mean={fc['mean']:6.3f} min={fc['min']:6.3f}  -> {verdict}")
        except Exception as e:
            print(f"  {name:18s} err {type(e).__name__}")
    print("\nFull-obs = SINDy home turf (easy). Partial-obs = the real frontier + the honest test.")
