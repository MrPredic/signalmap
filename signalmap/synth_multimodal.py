"""Synthetic multi-modal data WITH ground truth — to validate the coupling
finder before ever trusting it on real signals.

Scenario (the honest test):
  temp       random-walk drift  ............  a CONFOUND (shared driver)
  heat       controlled bursts  ............  independent excitation
  acoustic   = tanh(heat) + noise  .........  GENUINE coupling heat -> acoustic
  vibration  = a*temp + noise  ..............  driven by temp only
  em         = b*temp + noise  ..............  driven by temp only
  light      = noise  ......................  null channel

vibration and em are strongly correlated with EACH OTHER — but only because both
follow temp. A correct instrument must REJECT (vibration, em) once temp is given,
while KEEPING (heat, acoustic), which is not mediated by temp. That is the line
between discovery and self-deception.
"""
from __future__ import annotations

import numpy as np

# the only genuine cross-coupling among non-confound channels
GROUND_TRUTH_COUPLINGS = {("acoustic", "heat")}
# pairs that are correlated ONLY through the temp confound (must be rejected)
CONFOUNDED_PAIRS = {("em", "vibration")}


def _z(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / s if s > 0 else a - a.mean()


def make_multimodal(n: int = 2000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    temp = _z(np.cumsum(rng.standard_normal(n)) * 0.3)            # confound

    heat = (rng.random(n) < 0.1).astype(float)                   # excitation
    heat = np.convolve(heat, np.exp(-np.arange(20) / 5.0), mode="same")
    heat = _z(heat)

    acoustic = np.tanh(1.5 * heat) + 0.3 * rng.standard_normal(n)  # REAL coupling

    vibration = 0.9 * temp + 0.4 * rng.standard_normal(n)          # via temp
    em = 0.8 * temp + 0.4 * rng.standard_normal(n)                 # via temp

    light = rng.standard_normal(n)                                # null

    return {"temp": temp, "heat": heat, "acoustic": acoustic,
            "vibration": vibration, "em": em, "light": light}
