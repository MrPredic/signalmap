"""Drill #1: within-record non-stationarity. Standard ECN collapses a 24h recording to ONE
stationary noise-resistance Rn. If the record actually has temporal REGIMES (corrosion evolving),
that structure is discarded. We test it WITHIN a single record (no cross-recording leakage).

Honest gate: is the regime structure real, or an artifact? Compare the temporal autocorrelation
of the feature trajectory to a shuffled-segment NULL (destroys time order). z>>3 = real evolution.
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/ecn_nonstationarity.py
"""
import numpy as np, glob, os, warnings, antropy as ant
import ruptures as rpt
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

BASE="<local-path>/signalmap/data/ecn/Electrochemical Noise Data/Corrosion Types"
files={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(BASE+"/*.txt")}

S=80   # segments across the record
def seg_features(V,I):
    L=len(I)//S; feats=[]
    for k in range(S):
        Vs=np.ascontiguousarray(V[k*L:(k+1)*L]); Is=np.ascontiguousarray(I[k*L:(k+1)*L])
        rn=np.std(Vs)/(np.std(Is)+1e-15)
        pe=ant.perm_entropy(Is,normalize=True)
        feats.append([rn, pe, np.std(Is)])
    return np.array(feats)   # (S,3): local Rn, perm-entropy, current-std over TIME

def lag1(v):
    v=(v-v.mean())/(v.std()+1e-12); return np.mean(v[:-1]*v[1:])

print(f"{'record':12s} {'#regimes':>9s} {'Rn_CoV':>8s} {'PE_range':>9s} {'traj_autocorr_z(vs shuffle)':>28s}")
for name,f in files.items():
    a=np.loadtxt(f,skiprows=1); V,I=a[:,0],a[:,1]
    F=seg_features(V,I)
    # change points on standardized multivariate feature trajectory
    Fs=(F-F.mean(0))/(F.std(0)+1e-12)
    bkps=rpt.Pelt(model="rbf",min_size=3).fit(Fs).predict(pen=8)
    n_reg=len(bkps)              # includes final endpoint => #regimes
    rn_cov=np.std(F[:,0])/(np.abs(np.mean(F[:,0]))+1e-12)
    pe_range=F[:,1].max()-F[:,1].min()
    # honest surrogate gate: temporal autocorr of local-Rn trajectory vs shuffled-time null
    real_ac=lag1(F[:,0])
    null=np.array([lag1(rng.permutation(F[:,0])) for _ in range(500)])
    z=(real_ac-null.mean())/(null.std()+1e-12)
    print(f"{name:12s} {n_reg:9d} {rn_cov:8.2f} {pe_range:9.3f} {z:28.1f}")
print("\nStandard ECN reports ONE Rn per record. High Rn_CoV + regimes + autocorr_z>>3 = real")
print("temporal evolution the single-number summary discards (structure standard misses, no leakage).")
