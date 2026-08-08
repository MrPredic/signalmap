"""LIGO-Glitches bank -> standardized gauntlet (Jul 11).

Builds the Blip-vs-Scattered_Light bank from cache/ligo/ (see ligo_loader.py
for selection rules + provenance) and runs the standard leakage-free gauntlet:
LEAN baseline with LOGO over recordings (= glitch events) + group-perm-p,
nested forge with capacity gate, bootstrap CIs, receipts. Deterministic:
all seeds fixed inside gauntlet.py; the bank itself is a frozen npz.

n_perm=200 per the Jul-4 convention (FINAL bank verdict).
Usage: .venv-research/bin/python3 ligo_run.py                       # Blip vs Scattered_Light
       .venv-research/bin/python3 ligo_run.py Koi_Fish Whistle      # replication pair
"""
import sys
import numpy as np
from ligo_loader import load_ligo, CLASSES
from gauntlet import gauntlet


def main():
    classes = tuple(sys.argv[1:3]) if len(sys.argv) >= 3 else CLASSES
    tag = "" if classes == CLASSES else "_" + "_".join(c.lower() for c in classes)
    raw = load_ligo(classes=classes, tag=tag)
    if not raw:
        raise SystemExit("LIGO bank empty — see cache/ligo/provenance.json errors")
    y = [r[1] for r in raw]
    g = [r[2] for r in raw]
    print(f"bank: {len(raw)} windows, {len(set(g))} recordings, "
          f"classes={sorted(set(y))}", flush=True)
    # per-recording composition (sanity, printed not asserted)
    import json, collections
    prov = json.load(open(f"cache/ligo/provenance{tag}.json"))
    comp = collections.Counter((e["ifo"], e["label"]) for e in prov["events"])
    for k in sorted(comp):
        print(f"  {k[0]}/{k[1]}: {comp[k]} events", flush=True)
    res = gauntlet("LIGO-GLITCH" + (tag.upper() if tag else ""), raw, n_perm=200)
    print("\nresult dict:", res, flush=True)


if __name__ == "__main__":
    main()
