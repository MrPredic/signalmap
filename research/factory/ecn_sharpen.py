"""Sharpen the ECN thesis: kill the artifact-confound + add real statistics.
(1) sampling-rate check across recordings (a confound if inconsistent).
(2) does accuracy SURVIVE per-window detrend+zscore? If yes, it is NOT a DC/scale/drift artifact.
(3) label-permutation test -> p-value for the leave-one-recording-out accuracy.
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/ecn_sharpen.py
"""
import numpy as np, glob, os, warnings, antropy as ant
from scipy.signal import detrend
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
rng=np.random.default_rng(0)

BASE="<local-path>/signalmap/data/ecn/Electrochemical Noise Data/Types of Corrosive Substances"
def load(path):
    t,v=[],[]
    with open(path,encoding="utf-8-sig",errors="ignore") as fh:
        next(fh,None)
        for line in fh:
            p=line.split()
            if len(p)>=2:
                try: t.append(float(p[0])); v.append(float(p[1]))
                except ValueError: pass
    return np.array(t),np.array(v)

W=1024
def cfeat(x):
    x=np.ascontiguousarray(x,dtype=float)
    return [ant.perm_entropy(x,normalize=True), ant.sample_entropy(x),
            ant.detrended_fluctuation(x), ant.spectral_entropy(x,sf=1.0,method='welch',normalize=True)]

files=sorted(glob.glob(BASE+"/*/*.txt"))
# (1) sampling check
dts=[]
for f in files:
    t,_=load(f)
    if len(t)>10: dts.append(np.median(np.diff(t)))
print(f"(1) sampling dt across recordings: min={min(dts):.4f} max={max(dts):.4f} "
      f"-> {'CONSISTENT' if max(dts)/min(dts)<1.5 else 'INCONSISTENT (confound risk!)'}")

Xraw,Xnorm,y,g=[],[],[],[]
for gid,f in enumerate(files):
    chem=os.path.basename(os.path.dirname(f)); _,x=load(f)
    if len(x)<3*W: continue
    for k in range(len(x)//W):
        seg=x[k*W:(k+1)*W]
        segN=detrend(seg); segN=(segN-segN.mean())/(segN.std()+1e-12)   # kill DC/drift/scale
        Xraw.append(cfeat(seg)); Xnorm.append(cfeat(segN)); y.append(chem); g.append(gid)
Xraw,Xnorm,y,g=np.array(Xraw),np.array(Xnorm),np.array(y),np.array(g)
nchem=len(set(y)); chance=1/nchem

def logo(X,yy):
    Xs=StandardScaler().fit_transform(X)
    return cross_val_score(RandomForestClassifier(200,random_state=0),Xs,yy,groups=g,cv=LeaveOneGroupOut()).mean()

acc_raw=logo(Xraw,y); acc_norm=logo(Xnorm,y)
print(f"(2) leave-one-recording-out accuracy (chance={chance:.3f}):")
print(f"    complexity on RAW window            = {acc_raw:.3f}")
print(f"    complexity on DETREND+ZSCORE window = {acc_norm:.3f}  "
      f"({'SURVIVES -> not a DC/scale/drift artifact' if acc_norm>chance+0.15 else 'collapses -> was artifact'})")

# (3) permutation test on the artifact-robust (normalized) version
print("(3) label-permutation significance (200 perms, group-preserving) ...")
obs=acc_norm; null=[]
uniq=np.array(sorted(set(g)))
for _ in range(200):
    # shuffle chemical labels at the RECORDING level (respects grouping)
    gl={gid:lab for gid,lab in zip(uniq, rng.permutation([y[g==gid][0] for gid in uniq]))}
    yp=np.array([gl[gi] for gi in g])
    null.append(logo(Xnorm,yp))
null=np.array(null); p=(np.sum(null>=obs)+1)/(len(null)+1)
print(f"    observed={obs:.3f}  null_mean={null.mean():.3f}  p-value={p:.4f}  "
      f"({'SIGNIFICANT' if p<0.05 else 'not significant'})")
