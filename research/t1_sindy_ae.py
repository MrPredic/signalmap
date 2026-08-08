"""T1 MOAT: recover an accurate SPARSE latent law from a SINGLE observable.
Joint delay-embed autoencoder + SINDy-in-latent (Champion/Kutz/Brunton 2019 style),
with levers: loss-weight schedule, sequential thresholding, multi-seed keep-best,
refinement phase (freeze mask -> fit unregularised). Honest metric = short-horizon
forecast R2 of the simulated learned latent law, decoded back to the observable.
Run: <local-path>/signalmap/.venv-research/bin/python research/t1_sindy_ae.py
"""
import numpy as np, torch, torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score

def lorenz(t, s): x,y,z=s; return [10*(y-x), x*(28-z)-y, x*y-8/3*z]

dt = 0.01
t = np.arange(0, 60, dt)
sol = solve_ivp(lorenz,(t[0],t[-1]),[1.,1.,1.],t_eval=t,rtol=1e-9,atol=1e-9).y
obs = sol[0]                                    # SINGLE observable (x only)
obs = (obs - obs.mean())/obs.std()

W = 20                                          # delay-embedding window
M = len(obs) - W
Xw = torch.tensor(np.stack([obs[i:i+W] for i in range(M)]), dtype=torch.float32)

D = 3                                           # latent dim (Lorenz true dim)
def theta(z):                                   # poly library deg<=2 on latent
    f=[torch.ones(z.shape[0],1), z]
    for i in range(D):
        for j in range(i,D):
            f.append((z[:,i]*z[:,j]).unsqueeze(1))
    return torch.cat(f,1)
NF = 1 + D + D*(D+1)//2
FEAT = ['1']+[f'z{i}' for i in range(D)]+[f'z{i}z{j}' for i in range(D) for j in range(i,D)]

def mlp(a,b): return nn.Sequential(nn.Linear(a,64),nn.ELU(),nn.Linear(64,64),nn.ELU(),nn.Linear(64,b))

def train_one(seed, epochs=4000):
    torch.manual_seed(seed); np.random.seed(seed)
    enc, dec = mlp(W,D), mlp(D,W)
    Xi = nn.Parameter(0.01*torch.randn(NF,D))
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters())+[Xi], lr=1e-3)
    mask = torch.ones(NF,D)
    for ep in range(epochs):
        z = enc(Xw)
        recon = dec(z)
        dz = (z[2:]-z[:-2])/(2*dt); zc = z[1:-1]
        dz_pred = theta(zc)@(Xi*mask)
        L_rec = ((recon-Xw)**2).mean()
        L_dyn = ((dz-dz_pred)**2).mean()
        wdyn = min(1.0, ep/1500)*0.05          # recon-dominant -> dynamics later
        loss = L_rec + wdyn*L_dyn + 1e-3*(Xi*mask).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep>1500 and ep%500==0:              # sequential thresholding
            with torch.no_grad():
                mask = (Xi.abs() > 0.1).float()
    # refinement: freeze mask, fit unregularised a bit
    for ep in range(800):
        z=enc(Xw); recon=dec(z); dz=(z[2:]-z[:-2])/(2*dt); zc=z[1:-1]
        loss=((recon-Xw)**2).mean()+0.05*((dz-theta(zc)@(Xi*mask))**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z=enc(Xw); rec_r2=r2_score(Xw.numpy().ravel(), dec(z).numpy().ravel())
    return enc,dec,(Xi*mask).detach(), rec_r2, z.detach()

def forecast_r2(enc,dec,Xi,z0, horizon=300):
    # simulate learned latent ODE forward, decode center of window, compare to truth
    Xi=Xi.numpy(); z=z0.numpy().copy()
    def lat(t,zz):
        zz=torch.tensor(zz[None,:],dtype=torch.float32)
        return (theta(zz)@torch.tensor(Xi,dtype=torch.float32)).numpy().ravel()
    ts=np.arange(0,horizon*dt,dt)
    try:
        zs=solve_ivp(lat,(0,ts[-1]),z,t_eval=ts,rtol=1e-6,atol=1e-9,max_step=dt).y.T
    except Exception:
        return float('nan')
    if len(zs)<horizon: return float('nan')
    dec_obs=dec(torch.tensor(zs,dtype=torch.float32)).detach().numpy()[:,W//2]
    true_obs=obs[W//2 : W//2+len(dec_obs)]
    return r2_score(true_obs, dec_obs)

if __name__=='__main__':
    best=None
    for seed in range(4):
        enc,dec,Xi,rec_r2,z=train_one(seed)
        f_r2=forecast_r2(enc,dec,Xi,z[100])
        nnz=int((Xi.abs()>1e-6).sum())
        print(f"seed {seed}: recon_R2={rec_r2:.3f}  nnz(Xi)={nnz:2d}  forecast_R2={f_r2:.3f}")
        if best is None or (f_r2==f_r2 and f_r2>best[0]): best=(f_r2,seed,rec_r2,nnz)
    print(f"\nBEST: seed={best[1]} recon_R2={best[2]:.3f} nnz={best[3]} forecast_R2={best[0]:.3f}")
    print("DONE target: forecast_R2>0.6 sparse+bounded. Report null honestly if not reached.")
