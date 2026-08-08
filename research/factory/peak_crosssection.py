"""Peak-sampling cross-section (RESTART Prio A2) — does the DCASE-valve
discovery (impulse coverage: even 0.238 -> peak 0.875) generalize to the other
impulsive/marginal banks? Source-level rebuilds via the now-shared _readout:
  1. DCASE-pump-anomaly  peak   (old even null 0.425 <= chance)
  2. DCASE-pump-id       peak   (old even marginal, perm-p 0.049)
  3. IMS-fault           peak   (old even null 0.625 n.s.)
  4. IMS-fault           pool4  (honest rebuild of the timescale-screen flag:
                                 proxy pool4 gave 0.755 p=0.005 on cut windows)
Every bank goes through bank_audit first, then the standard gauntlet with
n_perm=200 (final-verdict resolution, Prio A3). Expectation is NOT that peak
always wins: pump physics is continuous (cavitation/flow), so a flat result
there is the negative control that peak-sampling is impulse-physics-specific,
not a universal score pump.
Run: nice -n 19 .venv-research/bin/python peak_crosssection.py
"""
from retro_loaders import load_dcase_pump, load_ims, bank_audit
from gauntlet import gauntlet

BANKS = [
    ("DCASE-pump-anomaly-PEAK", lambda: load_dcase_pump(task="anomaly", sampling="peak")),
    ("DCASE-pump-id-PEAK", lambda: load_dcase_pump(task="id", sampling="peak")),
    ("IMS-fault-PEAK", lambda: load_ims(sampling="peak")),
    ("IMS-fault-POOL4", lambda: load_ims(pool=4)),
]

if __name__ == "__main__":
    for name, fn in BANKS:
        raw = fn()
        if not bank_audit(name, raw):
            print(f"{name}: bank_audit FAIL -> gauntlet skipped", flush=True)
            continue
        gauntlet(name, raw, n_perm=200)
