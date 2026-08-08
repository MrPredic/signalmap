"""One-glance dashboard: accuracy vs compute-cost per domain, ours vs heavy.
Reads results.json (update numbers there, re-run this). Stars = our methods.
The money picture: top-left corner = accurate AND cheap = the efficiency moat.
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/viz.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
R = json.load(open(os.path.join(HERE, "results.json")))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
fig.suptitle("SignalMap efficiency moat — accuracy vs cost (leakage-free LOGO, identical task)",
             fontsize=12, fontweight="bold")

for ax, key, title in [(axes[0], "ecn", "ECN corrosion (6 chemicals)"),
                       (axes[1], "cwru", "CWRU bearing (6 fault types)")]:
    d = R[key]
    for name, m in d["methods"].items():
        if m["acc"] is None: continue
        x = m["ms"] if m.get("ms") else 0.5
        ours = m.get("ours", False)
        ax.scatter(x, m["acc"], s=210 if ours else 90, marker="*" if ours else "o",
                   color="#d62728" if ours else "#7f7f7f", zorder=3)
        ax.annotate((name + (" " + m["note"] if m.get("note") else "")),
                    (x, m["acc"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.axhline(d["chance"], ls=":", color="k", lw=1)
    ax.text(0.02, d["chance"] + 0.012, f"chance {d['chance']:.2f}", fontsize=7,
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log"); ax.set_ylim(0, 1.0)
    ax.set_xlabel("cost per window (ms, log)"); ax.set_ylabel("accuracy (LOGO)")
    extra = f" · perm-p={d['perm_p_ours']}" if d.get("perm_p_ours") else ""
    ax.set_title(f"{title} — {d['windows']} win / {d['recordings']} rec{extra}", fontsize=10)
    ax.grid(alpha=0.25)

out = os.path.join(HERE, "dashboard.png")
plt.tight_layout()
plt.savefig(out, dpi=140)
print("wrote", out)
