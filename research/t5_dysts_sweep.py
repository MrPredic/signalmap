"""T5 robustness sweep: does the surrogate detector reliably flag nonlinear determinism
across many chaotic systems, while keeping noise/linear controls below threshold?
Run: <local-path>/signalmap/.venv-research/bin/python research/t5_dysts_sweep.py
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)

def max_acf(x, maxlag=200):
    x = (x-x.mean())/(x.std()+1e-12); n=len(x)
    ac = np.correlate(x, x, 'full')[n-1:n-1+maxlag]/n
    return np.max(np.abs(ac[1:]))

def pred_err(x, m=3, tau=5, h=5, theiler=20, step=4):
    x=(x-x.mean())/(x.std()+1e-12); n=len(x)-(m-1)*tau-h
    if n < 100: return np.nan
    emb=np.stack([x[i*tau:i*tau+n] for i in range(m)],axis=1)
    fut=x[(m-1)*tau+h:(m-1)*tau+h+n]; errs=[]
    for i in range(0,n,step):
        d=np.sum((emb-emb[i])**2,axis=1)
        d[max(0,i-theiler):i+theiler+1]=np.inf
        errs.append((fut[i]-fut[np.argmin(d)])**2)
    return np.sqrt(np.mean(errs))

def iaaft(x, iters=60):
    amp=np.abs(np.fft.rfft(x)); sx=np.sort(x); s=rng.permutation(x)
    for _ in range(iters):
        ph=np.angle(np.fft.rfft(s)); s=np.fft.irfft(amp*np.exp(1j*ph),len(x))
        s=sx[np.argsort(np.argsort(s))]
    return s

def detect(x, n_sur=30):
    z1=(max_acf(x)-np.mean([max_acf(rng.permutation(x)) for _ in range(n_sur)]))/(np.std([max_acf(rng.permutation(x)) for _ in range(n_sur)])+1e-12)
    r=pred_err(x); null=np.array([pred_err(iaaft(x)) for _ in range(n_sur)])
    z2=(np.nanmean(null)-r)/(np.nanstd(null)+1e-12)
    return z1, z2

N=3000
# --- nonlinear chaotic systems from dysts (should be flagged nonlinear: L2 z>3) ---
from dysts.flows import (Lorenz, Rossler, Chua, Duffing, Chen, HyperRossler,
    Halvorsen, Thomas, DoubleGyre, RabinovichFabrikant, Aizawa, ShimizuMorioka,
    NoseHoover, Dadras, GenesioTesi, SprottA, SprottB, WangSun)
systems=[Lorenz,Rossler,Chua,Duffing,Chen,HyperRossler,Halvorsen,Thomas,
    RabinovichFabrikant,Aizawa,ShimizuMorioka,NoseHoover,Dadras,GenesioTesi,
    SprottA,SprottB,WangSun,DoubleGyre]

print("=== NONLINEAR SYSTEMS (want L1>3 structured, L2>3 nonlinear) ===")
nl_hits=0; nl_tot=0
for S in systems:
    try:
        traj=S().make_trajectory(N, resample=True)
        x=np.asarray(traj)[:,0]
        if not np.all(np.isfinite(x)) or x.std()<1e-9:
            print(f"  {S.__name__:18s} skipped (degenerate)"); continue
        z1,z2=detect(x); nl_tot+=1; nl_hits+=int(z2>3)
        print(f"  {S.__name__:18s} L1={z1:7.1f} L2={z2:5.1f} {'NL' if z2>3 else 'linear?'}")
    except Exception as e:
        print(f"  {S.__name__:18s} err {type(e).__name__}")

# --- controls (should NOT be flagged nonlinear) ---
print("\n=== CONTROLS (want L2<3; white also L1<3) ===")
white=rng.standard_normal(N)
f=np.fft.rfftfreq(N); f[0]=f[1]
pink=np.fft.irfft((rng.standard_normal(len(f))+1j*rng.standard_normal(len(f)))/np.sqrt(f),N)
ar1=np.zeros(N);
for i in range(1,N): ar1[i]=0.9*ar1[i-1]+rng.standard_normal()
sine=np.sin(np.linspace(0,60*np.pi,N))+0.1*rng.standard_normal(N)
ctrl={"white":white,"pink_1/f":pink,"AR(1)":ar1,"sine":sine}
fp=0
for k,x in ctrl.items():
    z1,z2=detect(x); fp+=int(z2>3)
    print(f"  {k:10s} L1={z1:7.1f} L2={z2:5.1f} {'FALSE-POS!' if z2>3 else 'ok'}")

print(f"\n=== SUMMARY: nonlinear detection {nl_hits}/{nl_tot}, control false-positives {fp}/{len(ctrl)} ===")
