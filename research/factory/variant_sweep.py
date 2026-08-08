"""The 'ping 1000' engine. Auto-generate a large method-variant space, CHEAP-assay every one
across all systems, rank, and spend full multi-start+attractor verification ONLY on the top-K.
Breadth in generation, brutality in selection. Scale to 1000 by widening the grids below.
Writes sweep_ranked.md.
"""
import numpy as np, itertools, time, datetime, signal, os
from harness import get_observable, multistart_forecast, attractor_stats, assay, SYSTEMS

def _alarm(sig, frm): raise TimeoutError("variant timed out")
signal.signal(signal.SIGALRM, _alarm)
from methods import DelaySINDy, HAVOK

# ---- variant space (widen these lists to ping more) ----
def generate():
    V=[]
    for lib,deg in [('poly',2),('poly',3),('poly',4),('poly+fourier',2),('fourier',3)]:
        for tau in [5,8,10,15,20]:
            for d in [3,4,5]:
                for th in [0.02,0.05,0.1,0.2,0.5]:
                    V.append(DelaySINDy(threshold=th,tau=tau,d=d,degree=deg,lib=lib))
    for q in [50,100,150]:
        for r in [7,11,15,21]:
            V.append(HAVOK(q=q,r=r))
    return V

variants=generate()
systems={s:get_observable(s) for s in SYSTEMS}
print(f"pinging {len(variants)} variants x {len(systems)} systems = {len(variants)*len(systems)} assays ...")

# ---- cheap assay pass (checkpointed: append per variant, resume-safe) ----
CKPT="sweep_progress.csv"
done_names={}
if os.path.exists(CKPT):
    for ln in open(CKPT):
        n,m,mx=ln.rstrip("\n").split("\t")
        done_names[n]=(float(m),float(mx))
    print(f"resuming: {len(done_names)} variants already assayed")
t0=time.time(); scored=[]
with open(CKPT,"a") as ck:
    for v in variants:
        if v.name in done_names:
            m,mx=done_names[v.name]; scored.append((m,mx,v)); continue
        a=[]
        for s,(x,dt) in systems.items():
            split=int(0.6*len(x))
            try:
                signal.alarm(30)                  # per-variant timeout (PROCESS.md Step-3 bug)
                v.fit(x[:split],dt); a.append(assay(v,x,split))
            except Exception: a.append(-9.9)
            finally: signal.alarm(0)
        m,mx=float(np.nanmean(a)),float(np.nanmax(a))
        ck.write(f"{v.name}\t{m}\t{mx}\n"); ck.flush()
        scored.append((m,mx,v))
scored.sort(key=lambda z:-z[0])
print(f"assay pass done in {time.time()-t0:.0f}s. top assay means:")
for m,mx,v in scored[:8]: print(f"  {v.name:34s} mean_assay={m:6.2f} best={mx:6.2f}")

# ---- full verification of top-K only ----
K=12; rows=[]
for m,mx,v in scored[:K]:
    for s,(x,dt) in systems.items():
        split=int(0.6*len(x))
        try:
            signal.alarm(120)
            v.fit(x[:split],dt); fc=multistart_forecast(v,x,split); at=attractor_stats(v,x,split)
            rows.append((v.name,s,v.nnz,fc['mean'],fc['min'],at.get('acf_l2'),at.get('bounded')))
        except Exception:
            rows.append((v.name,s,None,None,None,None,None))
        finally: signal.alarm(0)

lines=[f"# Variant Sweep — {datetime.date.today()} — pinged {len(variants)}x{len(systems)}, verified top {K}","",
       "| variant | system | nnz | fc_mean | fc_min | acf_l2 | bounded |","|---|---|---|---|---|---|---|"]
for r in rows:
    n,s,nnz,fm,fn,acf,b=r
    f=lambda z:(f"{z:.3f}" if isinstance(z,float) else "-")
    lines.append(f"| {n} | {s} | {nnz if nnz is not None else '-'} | {f(fm)} | {f(fn)} | {f(acf)} | {b} |")
open("sweep_ranked.md","w").write("\n".join(lines)+"\n")
# headline: best verified multi-start forecast per system
print("\nbest VERIFIED multi-start forecast per system:")
for s in systems:
    cand=[(r[3],r[0]) for r in rows if r[1]==s and isinstance(r[3],float)]
    if cand:
        best=max(cand); print(f"  {s:9s} fc_mean={best[0]:.3f}  <- {best[1]}")
print("-> wrote sweep_ranked.md")
