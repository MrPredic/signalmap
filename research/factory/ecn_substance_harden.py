"""Hardening drill: 18 substance recordings (6 chemicals x 3 concentrations) = REAL replicates.
Fixes the leakage caveat via LEAVE-ONE-RECORDING-OUT CV (whole recordings held out, not windows).
If our complexity features classify a HELD-OUT recording's chemical above chance AND above a
plain amplitude baseline, the structure is real and generalizes -- not recording-memorization.
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/ecn_substance_harden.py
"""
import numpy as np, glob, os, warnings, antropy as ant
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
rng=np.random.default_rng(0)

BASE="<local-path>/signalmap/data/ecn/Electrochemical Noise Data/Types of Corrosive Substances"
def load_potential(path):
    vals=[]
    with open(path, encoding="utf-8-sig", errors="ignore") as fh:
        next(fh,None)
        for line in fh:
            p=line.split()
            if len(p)>=2:
                try: vals.append(float(p[1]))
                except ValueError: pass
    return np.array(vals)

W=1024
def feats(x, kind):
    x=np.ascontiguousarray(x,dtype=float)
    if kind=="amplitude":                      # plain baseline (what a naive summary keeps)
        return [np.std(x), np.mean(np.abs(x-x.mean())), np.ptp(x)]
    return [ant.perm_entropy(x,normalize=True), ant.sample_entropy(x),
            ant.detrended_fluctuation(x), kurtosis(x), skew(x)]      # OURS: complexity

def lag1(v): v=(v-v.mean())/(v.std()+1e-12); return np.mean(v[:-1]*v[1:])

Xa,Xo,ychem,groups=[],[],[],[]
files=sorted(glob.glob(BASE+"/*/*.txt")); nonstat=[]
for gid,f in enumerate(files):
    chem=os.path.basename(os.path.dirname(f))
    x=load_potential(f)
    if len(x)<3*W: continue
    # non-stationarity per recording (z vs shuffled-time null on rolling std)
    S=40; L=len(x)//S; roll=np.array([np.std(x[k*L:(k+1)*L]) for k in range(S)])
    null=np.array([lag1(rng.permutation(roll)) for _ in range(300)])
    z=(lag1(roll)-null.mean())/(null.std()+1e-12); nonstat.append((chem,z))
    for k in range(len(x)//W):
        seg=x[k*W:(k+1)*W]
        Xa.append(feats(seg,"amplitude")); Xo.append(feats(seg,"complexity"))
        ychem.append(chem); groups.append(gid)
Xa,Xo,ychem,groups=np.array(Xa),np.array(Xo),np.array(ychem),np.array(groups)
print(f"recordings={len(set(groups))}  chemicals={sorted(set(ychem))}  windows={len(ychem)}\n")

def logo_acc(X):
    from sklearn.pipeline import make_pipeline
    s=cross_val_score(make_pipeline(StandardScaler(),RandomForestClassifier(300,random_state=0)),
                      X,ychem,groups=groups,cv=LeaveOneGroupOut()).mean()
    return s
chance=1/len(set(ychem))
print("Leave-one-RECORDING-out chemical classification (leakage-free):")
print(f"  chance level            = {chance:.3f}")
print(f"  AMPLITUDE baseline      = {logo_acc(Xa):.3f}")
print(f"  OURS (complexity)       = {logo_acc(Xo):.3f}")
print(f"  OURS + amplitude        = {logo_acc(np.hstack([Xa,Xo])):.3f}")

real=[c for c,z in nonstat if z>3]
print(f"\nWithin-record non-stationarity: {len(real)}/{len(nonstat)} recordings show real regimes (z>3)")
print("  ->", {c:round(z,1) for c,z in nonstat})
