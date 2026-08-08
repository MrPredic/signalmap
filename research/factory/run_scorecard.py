"""Run the whole bank x all systems -> persistent honest scorecard. Assay-tier first (cheap),
then full multi-start + attractor-statistics only where the assay is not hopeless.
Writes scorecard.md. This is the accumulating moat: every run grows the table.
"""
import numpy as np, time, datetime
from harness import get_observable, multistart_forecast, attractor_stats, assay, SYSTEMS
from methods import REGISTRY

ASSAY_FLOOR = -1.0   # below this, skip expensive verification (still logged)
rows=[]
for sysname in SYSTEMS:
    x, dt = get_observable(sysname)
    split = int(0.6*len(x))
    for meth in REGISTRY:
        t0=time.time()
        try:
            meth.fit(x[:split], dt)
        except Exception as e:
            rows.append((sysname, meth.name, None, "FIT-ERR", None, None, None, None)); continue
        a = assay(meth, x, split)
        if a < ASSAY_FLOOR:
            rows.append((sysname, meth.name, meth.nnz, f"assay={a:.2f} SKIP", None, None, None, time.time()-t0)); continue
        fc = multistart_forecast(meth, x, split)
        at = attractor_stats(meth, x, split)
        rows.append((sysname, meth.name, meth.nnz, f"assay={a:.2f}",
                     fc['mean'], fc['min'], (at['wass'], at['acf_l2'], at['bounded']), time.time()-t0))

# ---- write markdown ----
lines=[f"# Method Factory Scorecard — {datetime.date.today()}", "",
       "Honest multi-start (K=10) forecast + attractor-statistics rescue-metric. "
       "assay = cheap single-start pre-screen. nnz = sparsity (lower=more interpretable).", "",
       "| system | method | nnz | assay | fc_mean | fc_min | attractor(wass,acf,bounded) | sec |",
       "|---|---|---|---|---|---|---|---|"]
for r in rows:
    s,m,nnz,st,fcm,fcn,at,sec = r
    fcm=f"{fcm:.3f}" if isinstance(fcm,float) else "-"
    fcn=f"{fcn:.3f}" if isinstance(fcn,float) else "-"
    at=f"{at[0]:.3f},{at[1]:.3f},{at[2]}" if at else "-"
    nnz="-" if nnz is None else nnz
    sec=f"{sec:.1f}" if isinstance(sec,float) else "-"
    lines.append(f"| {s} | {m} | {nnz} | {st} | {fcm} | {fcn} | {at} | {sec} |")
open("scorecard.md","w").write("\n".join(lines)+"\n")
print("\n".join(lines))
print("\n-> wrote research/factory/scorecard.md")
