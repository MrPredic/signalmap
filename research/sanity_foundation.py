"""Foundational sanity gate. If these fail, no downstream 'discovery' is trustworthy.
Run: <local-path>/signalmap/.venv-research/bin/python research/sanity_foundation.py
"""
import numpy as np
from scipy.integrate import solve_ivp

np.random.seed(0)

# ---------- T1 base: full-observation SINDy must recover Lorenz ----------
def lorenz(t, s, sig=10., rho=28., beta=8./3.):
    x, y, z = s
    return [sig*(y-x), x*(rho-z)-y, x*y-beta*z]

t = np.arange(0, 20, 0.002)
sol = solve_ivp(lorenz, (t[0], t[-1]), [1., 1., 1.], t_eval=t, rtol=1e-9, atol=1e-9)
X = sol.y.T  # (N,3)

import pysindy as ps
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1),
    feature_library=ps.PolynomialLibrary(degree=2),
)
model.fit(X, t=0.002)
# analytic nonzero terms: x'=-10x+10y ; y'=28x-y-xz ; z'=-2.667z+xy
coefs = model.coefficients()
names = model.get_feature_names()

def get(eq, term):
    return coefs[eq][names.index(term)] if term in names else 0.0

checks = {
    "x'/x=-10": (get(0, 'x0'), -10.),
    "x'/y=+10": (get(0, 'x1'), 10.),
    "y'/x=+28": (get(1, 'x0'), 28.),
    "y'/y=-1":  (get(1, 'x1'), -1.),
    "y'/xz=-1": (get(1, 'x0 x2'), -1.),
    "z'/z=-2.667": (get(2, 'x2'), -8./3.),
    "z'/xy=+1": (get(2, 'x0 x1'), 1.),
}
sindy_ok = all(abs(v-tgt) < 0.5 for v, tgt in checks.values())
print("=== T1 base: full-obs SINDy on Lorenz ===")
for k, (v, tgt) in checks.items():
    print(f"  {k:14s} got {v:+.3f} (target {tgt:+.3f}) {'OK' if abs(v-tgt)<0.5 else 'FAIL'}")
print(f"  -> SINDy reader {'PASS' if sindy_ok else 'FAIL'}")

# ---------- T5 base: entropy/surrogate must separate noise from signal ----------
import antropy as ant
N = 4000
white = np.random.randn(N)
structured = sol.y[0][:N]              # Lorenz x, deterministic chaos
sine = np.sin(np.linspace(0, 80*np.pi, N)) + 0.1*np.random.randn(N)

def perm_ent(x):  # normalized permutation entropy: ~1 = random, low = structured
    return ant.perm_entropy(x, order=4, normalize=True)

pe = {k: perm_ent(v) for k, v in {"white": white, "lorenz_x": structured, "noisy_sine": sine}.items()}
print("\n=== T5 base: permutation entropy (1=random, low=structured) ===")
for k, v in pe.items():
    print(f"  {k:12s} PE={v:.3f}")
# classifier: signal if PE < 0.9 (structured). white should be >= 0.95, others clearly below.
t5_ok = (pe["white"] > 0.93) and (pe["lorenz_x"] < 0.9) and (pe["noisy_sine"] < 0.9)
print(f"  white classified '{'no-signal' if pe['white']>0.9 else 'SIGNAL(FP!)'}' | "
      f"lorenz '{'signal' if pe['lorenz_x']<0.9 else 'MISS'}' | sine '{'signal' if pe['noisy_sine']<0.9 else 'MISS'}'")
print(f"  -> Entropy reader {'PASS' if t5_ok else 'FAIL'}")

print("\n=== FOUNDATION:", "PASS" if (sindy_ok and t5_ok) else "FAIL", "===")
