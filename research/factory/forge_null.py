"""NULL control (fool's-gold test): nested forge on PURE white noise with
random class labels. There is no structure — an honest pipeline must land at
chance. If it lands meaningfully above, our selection still leaks.
18 recordings, 6 classes x 3, 14 windows each (mirrors the ECN geometry).
Run: <local-path>/signalmap/.venv-research/bin/python research/factory/forge_null.py
"""
import numpy as np
from forge_nested import nested

rng = np.random.default_rng(7)
raw = []
for gid in range(18):
    cls = f"c{gid % 6}"
    for _ in range(14):
        s = rng.standard_normal(1024)
        raw.append(((s - s.mean()) / s.std(), cls, gid))
print("NULL bank: 18 recs, 6 classes, chance=0.167 — nested forge must stay ~chance")
nested("NULL-noise", raw)
