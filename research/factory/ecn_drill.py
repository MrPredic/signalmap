"""First real drill: Electrochemical Noise (signal that is literally called 'noise').
Standard ECN analysis reduces a whole record to noise-resistance Rn=std(V)/std(I) and
localization index LI. Question: do OUR label-free readers (impulsiveness, entropy, long-range
correlation) reveal structure that Rn/LI miss -- i.e. separate corrosion types better, unsupervised?
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/ecn_drill.py
"""
import numpy as np, glob, os, warnings
import antropy as ant
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")

BASE="<local-path>/signalmap/data/ecn/Electrochemical Noise Data/Corrosion Types"
files={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(BASE+"/*.txt")}

def load(f):
    a=np.loadtxt(f, skiprows=1)
    return a[:,0], a[:,1]           # potential V, current I

W=2048
def features(V,I):
    V=np.ascontiguousarray(V,dtype=float); I=np.ascontiguousarray(I,dtype=float)
    rn = np.std(V)/(np.std(I)+1e-15)
    li = np.std(I)/(np.sqrt(np.mean(I**2))+1e-15)
    std_feats=[rn, li]                                   # STANDARD ECN
    our=[kurtosis(I), skew(I), ant.perm_entropy(I,normalize=True),
         ant.detrended_fluctuation(I), kurtosis(V), ant.perm_entropy(V,normalize=True),
         ant.sample_entropy(I)]                          # OURS (label-free structure)
    return std_feats, our

Xstd,Xour,y=[],[],[]
for cls,(name) in enumerate(files):
    V,I=load(files[name])
    n=len(I)//W
    for k in range(n):
        s=slice(k*W,(k+1)*W)
        sf,of=features(V[s],I[s])
        if np.all(np.isfinite(sf)) and np.all(np.isfinite(of)):
            Xstd.append(sf); Xour.append(of); y.append(cls)
Xstd,Xour,y=np.array(Xstd),np.array(Xour),np.array(y)
print(f"classes={list(files)}  windows={len(y)} (W={W})\n")

def evaluate(X,label):
    Xs=StandardScaler().fit_transform(X)
    km=KMeans(4,n_init=10,random_state=0).fit(Xs)
    ari=adjusted_rand_score(y,km.labels_)
    sil=silhouette_score(Xs,y)
    acc=cross_val_score(RandomForestClassifier(200,random_state=0),Xs,y,cv=5).mean()
    print(f"{label:28s} unsup_ARI={ari:.3f}  silhouette(true)={sil:.3f}  RF_5cv_acc={acc:.3f}")
    return ari,acc

print("Does the feature space recover the 4 corrosion types?  (ARI=1 perfect unsupervised)")
evaluate(Xstd, "STANDARD (Rn, LI)")
evaluate(Xour, "OURS (kurt/entropy/DFA)")
evaluate(np.hstack([Xstd,Xour]), "STANDARD + OURS")
print("\nIf OURS >> STANDARD: our readers see corrosion structure that Rn/LI throw away.")
