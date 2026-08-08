"""Rigorous efficiency head-to-head: our LEAN features vs the big teams' HEAVY methods,
on the SAME leakage-free ECN task. Heavy = RQA (recurrence rate + determinism, the survey's
'best' method, O(n^2)) and wavelet-entropy (PyWavelets). Measure accuracy AND cost -> ratio.
Claim we want to earn: cheap ~matches heavy at a fraction of the compute.
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/ecn_head_to_head.py
"""
import numpy as np, glob, os, time, warnings, antropy as ant, pywt
from scipy.signal import detrend, welch
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

# ---- OUR lean features (O(n)/O(n log n)) ----
def perm3(x): return ant.perm_entropy(x,order=3,normalize=True)
def psd_slope(x):
    f,P=welch(x,nperseg=min(256,len(x))); f,P=f[1:],P[1:]
    return np.polyfit(np.log(f),np.log(P+1e-20),1)[0]

# ---- HEAVY competitor methods ----
def rqa(x, m=3, tau=5, rr=0.1, sub=400):        # RQA: recurrence-rate + determinism (O(n^2))
    x=x[:sub]; n=len(x)-(m-1)*tau
    emb=np.stack([x[i*tau:i*tau+n] for i in range(m)],1)
    D=np.sqrt(((emb[:,None,:]-emb[None,:,:])**2).sum(-1))
    thr=np.quantile(D, rr); R=(D<thr).astype(int)
    rec_rate=R.mean()
    # determinism: fraction of recurrent points on diagonals length>=2
    diagpts=0; tot=R.sum()
    for k in range(1,n):
        d=np.diag(R,k)
        run=0
        for val in d:
            if val: run+=1
            else:
                if run>=2: diagpts+=run
                run=0
        if run>=2: diagpts+=run
    det=2*diagpts/(tot+1e-9)
    return rec_rate, det
def wavelet_entropy(x, wav='db4', lvl=5):       # wavelet energy entropy (PyWavelets)
    c=pywt.wavedec(x, wav, level=lvl)
    e=np.array([np.sum(ci**2) for ci in c]); p=e/(e.sum()+1e-12)
    return -np.sum(p*np.log(p+1e-12))

W=1024
raw=[]
for gid,f in enumerate(sorted(glob.glob(BASE+"/*/*.txt"))):
    x=load(f); chem=os.path.basename(os.path.dirname(f))
    if len(x)<3*W: continue
    for k in range(len(x)//W):
        s=detrend(np.ascontiguousarray(x[k*W:(k+1)*W],float)); s=(s-s.mean())/(s.std()+1e-12)
        raw.append((s,chem,gid))
y=np.array([r[1] for r in raw]); g=np.array([r[2] for r in raw]); chance=1/len(set(y))

def logo(X):
    X=np.atleast_2d(X.T).T if X.ndim==1 else X
    if X.ndim==1: X=X[:,None]
    return cross_val_score(make_pipeline(StandardScaler(),RandomForestClassifier(200,random_state=0)),
                           X,y,groups=g,cv=LeaveOneGroupOut()).mean()
def build(fns):
    t0=time.time(); cols=[]
    for s,_,_ in raw:
        row=[]
        for fn in fns: r=fn(s); row+= list(r) if isinstance(r,tuple) else [r]
        cols.append(row)
    return np.array(cols), 1000*(time.time()-t0)/len(raw)

print(f"windows={len(raw)} chance={chance:.3f}\n")
print(f"{'method':28s} {'LOGO_acc':>8s} {'cost_ms/win':>11s}")
res={}
for name,fns in {
    "OURS lean (perm3+psd_slope)":[perm3,psd_slope],
    "HEAVY rqa (recrate+determ)":[rqa],
    "HEAVY wavelet_entropy":[wavelet_entropy],
    "HEAVY rqa+wavelet (combo)":[rqa,wavelet_entropy],
}.items():
    X,ms=build(fns); acc=logo(X); res[name]=(acc,ms)
    print(f"{name:28s} {acc:8.3f} {ms:11.3f}")

ours=res["OURS lean (perm3+psd_slope)"]; heavy=res["HEAVY rqa+wavelet (combo)"]
print(f"\nOURS acc={ours[0]:.3f} @ {ours[1]:.2f}ms  vs HEAVY acc={heavy[0]:.3f} @ {heavy[1]:.2f}ms")
print(f"-> accuracy gap = {ours[0]-heavy[0]:+.3f}   speed advantage = {heavy[1]/max(ours[1],1e-6):.0f}x cheaper")
print("Efficiency-moat EARNED iff accuracy gap ~>=0 at large speedup.")
