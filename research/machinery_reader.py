"""Right reader for the machinery class: impulsiveness via kurtosis / envelope kurtosis /
band-pass envelope kurtosis (established bearing diagnostics). Honest test: does an
UNSUPERVISED impulsiveness score separate inner-race-fault from normal on real CWRU?
Also re-confirms the L1-autocorr inversion finding with a proper flipped AUC.
Run: <local-path>/signalmap/.venv-research/bin/python research/machinery_reader.py
"""
import numpy as np, pandas as pd, warnings
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

df = pd.read_parquet('<local-path>/signalmap/data/cwru_real.parquet')
SR = 12000
def decode(b, cap=4000):
    a = np.frombuffer(b, dtype='<i2').astype(np.float64); return a[:cap]

def env_kurt(x, lo=None, hi=None):
    if lo is not None:
        b,a = butter(4, [lo/(SR/2), hi/(SR/2)], btype='band'); x = filtfilt(b,a,x)
    return kurtosis(np.abs(hilbert(x)))

def max_acf(x, maxlag=200):
    x=(x-x.mean())/(x.std()+1e-12); n=len(x)
    ac=np.correlate(x,x,'full')[n-1:n-1+maxlag]/n; return np.max(np.abs(ac[1:]))

feats = {'raw_kurtosis':[], 'env_kurtosis':[], 'bp2-4kHz_env_kurt':[], 'max_acf':[], 'label':[]}
for lab,y in [('normal',0),('ANOMALY_inner_race_fault',1)]:
    for b in df[df.label==lab]['samples']:
        x = decode(b)
        if len(x)<300 or x.std()<1e-9: continue
        feats['raw_kurtosis'].append(kurtosis(x))
        feats['env_kurtosis'].append(env_kurt(x))
        feats['bp2-4kHz_env_kurt'].append(env_kurt(x, 2000, 4000))
        feats['max_acf'].append(max_acf(x))
        feats['label'].append(y)

y = np.array(feats['label'])
print(f"n_normal={int((y==0).sum())}  n_fault={int((y==1).sum())}\n")
print(f"{'feature':20s} {'mean_normal':>12s} {'mean_fault':>12s} {'AUC':>7s} {'AUC_flip':>9s}")
for k in ['raw_kurtosis','env_kurtosis','bp2-4kHz_env_kurt','max_acf']:
    v = np.array(feats[k])
    auc = roc_auc_score(y, v)
    print(f"{k:20s} {v[y==0].mean():12.3f} {v[y==1].mean():12.3f} {auc:7.3f} {max(auc,1-auc):9.3f}")
print("\nAUC_flip = separation power regardless of direction (unsupervised threshold).")
