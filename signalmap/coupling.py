"""Cross-modal coupling discovery — the honest core.

Finding that two sensor modalities are *correlated* is trivial and usually a
lie: in passive multi-sensor data almost everything correlates through shared
drivers (time, temperature, mains hum, building vibration). The whole value is
separating a genuine cross-modal coupling from a shared-confound correlation.

We do that with three numpy-only tools:
  * mutual_information  — detects *nonlinear* dependence (not just linear corr).
  * partial_correlation — removes the influence of known confounds (residualize
    both channels on the confounds, correlate the residuals). A coupling that
    survives confound-removal is a candidate; one that vanishes was a confound.
  * permutation_pvalue  — significance: shuffle to build a null distribution.

A discovered coupling is a HYPOTHESIS for controlled follow-up, never a proven
new effect. This module is deliberately conservative: it tries to *reject*
couplings, and only what survives is reported.
"""
from __future__ import annotations

import numpy as np


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 12) -> float:
    """Binned mutual information (nats). Catches nonlinear dependence that
    linear correlation misses. >=0; 0 means independent (at this resolution)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c, _, _ = np.histogram2d(x, y, bins=bins)
    total = c.sum()
    if total == 0:
        return 0.0
    pxy = c / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    denom = (px[:, None] * py[None, :])[nz]
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / denom))
    return float(max(mi, 0.0))


def _residualize(x: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Return x with the linear influence of confounds Z removed."""
    n = len(x)
    if Z.size == 0:
        return x - x.mean()
    A = np.column_stack([np.ones(n), Z])
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ beta


def partial_correlation(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Correlation between x and y after removing confounds Z (linear). Near 0
    => the x-y relationship was explained away by the confounds."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    return _safe_corr(_residualize(x, Z), _residualize(y, Z))


def permutation_pvalue(x: np.ndarray, y: np.ndarray, observed: float,
                       n_perm: int = 300, seed: int = 0) -> float:
    """Two-sided permutation p-value for a correlation statistic: shuffle y,
    recompute, count how often |null| >= |observed|."""
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(_safe_corr(x, rng.permutation(y))) >= abs(observed):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def find_couplings(channels: dict[str, np.ndarray], confounds: list[str] | None = None,
                   alpha: float = 0.05, min_effect: float = 0.1,
                   n_perm: int = 300) -> list[dict]:
    """Rank candidate cross-modal couplings, confound-adjusted.

    For every pair of non-confound channels report raw correlation, mutual
    information, and the confound-adjusted partial correlation with a
    permutation p-value. `survives` = the coupling is still significant AND
    non-trivial after removing the confounds — i.e. a real candidate, not a
    shared-driver artifact.
    """
    confounds = confounds or []
    keys = list(channels)
    n = len(channels[keys[0]])
    Z = (np.column_stack([channels[c] for c in confounds])
         if confounds else np.empty((n, 0)))
    names = [k for k in keys if k not in confounds]

    results: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            x, y = np.asarray(channels[a], float), np.asarray(channels[b], float)
            raw = _safe_corr(x, y)
            mi = mutual_information(x, y)
            rx, ry = _residualize(x, Z), _residualize(y, Z)
            adj = _safe_corr(rx, ry)
            p = permutation_pvalue(rx, ry, adj, n_perm=n_perm)
            survives = bool(p < alpha and abs(adj) >= min_effect)
            results.append({
                "a": a, "b": b, "raw_corr": raw, "mi": mi,
                "adj_corr": adj, "p": p, "survives": survives,
            })
    results.sort(key=lambda r: abs(r["adj_corr"]), reverse=True)
    return results
