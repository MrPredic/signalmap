"""T5 on REAL data: apply the validated surrogate detector to CWRU bearing vibration.
Question: does the label-free nonlinearity score (L2) separate inner-race-fault from normal?
That is the material-research payoff: structure/nonlinearity detected WITHOUT labels.
Run: <local-path>/signalmap/.venv-research/bin/python research/t5_cwru_real.py
"""
import numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(2)

def max_acf(x, maxlag=200):
    x=(x-x.mean())/(x.std()+1e-12); n=len(x)
    ac=np.correlate(x,x,'full')[n-1:n-1+maxlag]/n
    return np.max(np.abs(ac[1:]))
def pred_err(x, m=3, tau=5, h=5, theiler=20, step=4):
    x=(x-x.mean())/(x.std()+1e-12); n=len(x)-(m-1)*tau-h
    if n<100: return np.nan
    emb=np.stack([x[i*tau:i*tau+n] for i in range(m)],axis=1); fut=x[(m-1)*tau+h:(m-1)*tau+h+n]; e=[]
    for i in range(0,n,step):
        d=np.sum((emb-emb[i])**2,axis=1); d[max(0,i-theiler):i+theiler+1]=np.inf
        e.append((fut[i]-fut[np.argmin(d)])**2)
    return np.sqrt(np.mean(e))
def iaaft(x, iters=50):
    amp=np.abs(np.fft.rfft(x)); sx=np.sort(x); s=rng.permutation(x)
    for _ in range(iters):
        ph=np.angle(np.fft.rfft(s)); s=np.fft.irfft(amp*np.exp(1j*ph),len(x)); s=sx[np.argsort(np.argsort(s))]
    return s
def detect(x, n_sur=25):
    n1=[max_acf(rng.permutation(x)) for _ in range(n_sur)]
    z1=(max_acf(x)-np.mean(n1))/(np.std(n1)+1e-12)
    r=pred_err(x); null=np.array([pred_err(iaaft(x)) for _ in range(n_sur)])
    z2=(np.nanmean(null)-r)/(np.nanstd(null)+1e-12)
    return z1,z2

df=pd.read_parquet('<local-path>/signalmap/data/cwru_real.parquet')
def decode(b, cap=4000):
    a=np.frombuffer(b,dtype='<i2').astype(np.float64)
    return a[:cap] if len(a)>cap else a
per=25
res={}
for lab in ['normal','ANOMALY_inner_race_fault']:
    sub=df[df.label==lab].sample(n=min(per,(df.label==lab).sum()),random_state=3)
    rows=[]
    for b in sub['samples']:
        x=decode(b)
        if len(x)<500 or x.std()<1e-9: continue
        z1,z2=detect(x); rows.append((z1,z2))
    res[lab]=np.array(rows)
    z1m,z2m=res[lab].mean(0); z1s,z2s=res[lab].std(0)
    print(f"{lab:26s} n={len(rows):2d} L1={z1m:6.1f}±{z1s:4.1f}  L2(nonlin)={z2m:5.2f}±{z2s:4.2f}")

# separation: does L2 (label-free) rank faults above normal?
n=res['normal']; f=res['ANOMALY_inner_race_fault']
y=np.r_[np.zeros(len(n)),np.ones(len(f))]
auc_l2=roc_auc_score(y, np.r_[n[:,1],f[:,1]])
auc_l1=roc_auc_score(y, np.r_[n[:,0],f[:,0]])
print(f"\nAUC fault-vs-normal  L1(structure)={auc_l1:.3f}  L2(nonlinearity)={auc_l2:.3f}")
print("Interpretation: AUC>0.5 => the label-free score carries fault information on REAL data.")
