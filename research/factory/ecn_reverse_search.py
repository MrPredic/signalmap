"""Reverse-search: INVENT the leanest method that matches the heavy published ECN pipelines.
Big teams use wavelet-entropy / recurrence plots / Stockwell+Shannon / sample-entropy (all costly).
We search a broad bank of CHEAP features for the MINIMAL one(s) that reach the same leakage-free
discrimination -- and flag features competitors overlook. Efficiency = same result, fraction of cost.
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/ecn_reverse_search.py
"""
import numpy as np, glob, os, time, warnings, antropy as ant
from scipy.signal import detrend, welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
warnings.filterwarnings("ignore")

BASE="<local-path>/signalmap/data/ecn/Electrochemical Noise Data/Types of Corrosive Substances"
def load(p):
    v=[]
    for ln in open(p,encoding="utf-8-sig",errors="ignore").readlines()[1:]:
        s=ln.split()
        if len(s)>=2:
            try: v.append(float(s[1]))
            except: pass
    return np.array(v)

# ---- CHEAP feature bank (each: name -> fn). detrend+zscore first (kill amplitude confound). ----
def hjorth(x):
    dx=np.diff(x); ddx=np.diff(dx)
    a=np.var(x); m=np.sqrt(np.var(dx)/(a+1e-12))
    c=np.sqrt(np.var(ddx)/(np.var(dx)+1e-12))/(m+1e-12)
    return m, c                                   # mobility, complexity (O(n), classic, cheap)
def psd_slope(x):
    f,P=welch(x,nperseg=min(256,len(x))); f,P=f[1:],P[1:]
    return np.polyfit(np.log(f),np.log(P+1e-20),1)[0]        # 1/f exponent (cheap)
def zcr(x): return np.mean(np.abs(np.diff(np.sign(x)))>0)    # zero-crossing rate (O(n), trivial)

FEATS = {
 "hjorth_mobility": lambda x: hjorth(x)[0],
 "hjorth_complexity": lambda x: hjorth(x)[1],
 "psd_slope_1f": psd_slope,
 "zero_cross_rate": zcr,
 "rms_succ_diff": lambda x: np.sqrt(np.mean(np.diff(x)**2)),
 "perm_entropy_o3": lambda x: ant.perm_entropy(x,order=3,normalize=True),
 "perm_entropy_o5": lambda x: ant.perm_entropy(x,order=5,normalize=True),
 "spectral_entropy": lambda x: ant.spectral_entropy(x,sf=1.0,method='welch',normalize=True),
 "kurtosis": lambda x: kurtosis(x),
 # --- the EXPENSIVE ones the big teams lean on (baseline to MATCH cheaply) ---
 "sample_entropy_SLOW": lambda x: ant.sample_entropy(x),
 "dfa_SLOW": lambda x: ant.detrended_fluctuation(x),
}

W=1024
recs=sorted(glob.glob(BASE+"/*/*.txt"))
segs=[]  # (features_dict_values later), y, group
raw=[]
for gid,f in enumerate(recs):
    x=load(f); chem=os.path.basename(os.path.dirname(f))
    if len(x)<3*W: continue
    for k in range(len(x)//W):
        s=x[k*W:(k+1)*W]; s=detrend(np.ascontiguousarray(s,float)); s=(s-s.mean())/(s.std()+1e-12)
        raw.append((s,chem,gid))
y=np.array([r[1] for r in raw]); g=np.array([r[2] for r in raw])
chance=1/len(set(y))
print(f"recordings={len(set(g))} windows={len(raw)} chance={chance:.3f}\n")

def logo(X):
    if X.ndim==1: X=X[:,None]
    return cross_val_score(make_pipeline(StandardScaler(),RandomForestClassifier(200,random_state=0)),
                           X,y,groups=g,cv=LeaveOneGroupOut()).mean()

# compute each feature over all windows + time cost
print(f"{'feature':20s} {'LOGO_acc':>8s} {'cost_ms/win':>11s}")
results={}
for name,fn in FEATS.items():
    t0=time.time(); col=np.array([fn(s) for s,_,_ in raw]); ms=1000*(time.time()-t0)/len(raw)
    acc=logo(col); results[name]=(acc,ms,col)
    print(f"{name:20s} {acc:8.3f} {ms:11.3f}")

# rank cheap (non-SLOW) vs expensive
cheap={k:v for k,v in results.items() if "SLOW" not in k}
best_cheap=max(cheap,key=lambda k:cheap[k][0])
slow_best=max([k for k in results if "SLOW" in k],key=lambda k:results[k][0])
print(f"\nBEST CHEAP single feature : {best_cheap}  acc={cheap[best_cheap][0]:.3f}  cost={cheap[best_cheap][1]:.3f} ms/win")
print(f"BEST EXPENSIVE (big-team)  : {slow_best}  acc={results[slow_best][0]:.3f}  cost={results[slow_best][1]:.3f} ms/win")
spd=results[slow_best][1]/max(cheap[best_cheap][1],1e-6)
print(f"-> speed ratio (expensive/cheap) = {spd:.0f}x  | accuracy gap = {results[slow_best][0]-cheap[best_cheap][0]:+.3f}")

# greedy: cheapest 2-feature combo
names=list(cheap); best2=(0,None)
for i in range(len(names)):
    for j in range(i+1,len(names)):
        X=np.stack([cheap[names[i]][2],cheap[names[j]][2]],1); a=logo(X)
        if a>best2[0]: best2=(a,(names[i],names[j]))
print(f"\nBEST CHEAP 2-feature combo : {best2[1]}  acc={best2[0]:.3f}")
print("If cheap ~matches expensive at a fraction of cost -> the efficiency-moat is demonstrated.")
