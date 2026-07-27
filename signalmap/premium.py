"""Premium feature families for `signalmap distill` — expensive, family-specific
featurizers that are NEVER part of the base grammar. Rationale (research receipt,
Jul 2026): RQA is CI-solid better on 3 banks, a tie on 2 and WORSE on 3 — a
family this expensive (O(n^2) per window vs O(n log n) for the base grammar)
must not burn forge budget everywhere. Instead each family runs as an opt-in
distill challenger with a COST RECEIPT, and the champion rule decides: the
family enters the deploy spec only when it beats the base selection with a
paired-CI-solid margin on the caller's own bank.

Self-contained numpy implementation (matches pyunicorn.timeseries.RecurrencePlot
semantics for dim/tau/recurrence_rate thresholding; parity pinned in tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# -------------------------------------------------------------- envelope
def envelope_feature_names(n_bands: int = 5) -> list[str]:
    return [f"env_band{i}" for i in range(n_bands)]


ENVELOPE_FEATURE_NAMES = envelope_feature_names()


def envelope_features(w: np.ndarray, n_bands: int = 5,
                      f_max_frac: float = 0.5) -> np.ndarray:
    """Hilbert-envelope defect-band energy: |analytic signal| -> DC-removed ->
    power spectrum of the envelope -> `n_bands` contiguous RELATIVE energy
    fractions over the low half of the envelope spectrum (bins 1..f_max_frac
    of Nyquist; bin 0 = DC dropped). RELATIVE bands on purpose (not physical
    BPFO/BPFI Hz) — physical defect frequencies need per-bank rpm/geometry,
    which would break the one-fixed-config honesty this family is judged on.
    Deterministic, no randomness. O(n log n) per window (Hilbert + rFFT)."""
    x = np.asarray(w, float)
    n = len(x)
    if n < 2 * n_bands + 4:
        raise ValueError(f"window too short for envelope features (n={n})")
    from scipy.signal import hilbert
    env = np.abs(hilbert(x))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env)) ** 2
    hi = min(len(spec), max(n_bands + 1, int(f_max_frac * len(spec))))
    considered = spec[1:hi]
    if len(considered) < n_bands:
        raise ValueError(
            f"window too short for {n_bands} envelope bands "
            f"(considered={len(considered)})")
    bands = np.array_split(considered, n_bands)
    energies = np.array([b.sum() for b in bands], float)
    return energies / (energies.sum() + 1e-12)


# ------------------------------------------------------------------ RQA
RQA_FEATURE_NAMES = ["rqa_rr", "rqa_det", "rqa_lam", "rqa_entr", "rqa_lmean",
                     "rqa_tt"]


def _embed(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n = len(x) - (dim - 1) * tau
    if n < 2:
        raise ValueError(f"window too short for dim={dim}, tau={tau}")
    return np.stack([x[k * tau:k * tau + n] for k in range(dim)], axis=1)


def _run_lengths(flat: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs in a 1-D boolean array."""
    b = np.concatenate([[False], flat, [False]])
    d = np.diff(b.astype(np.int8))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def _diag_lines(R: np.ndarray) -> np.ndarray:
    """Diagonal line lengths, LOI (main diagonal) excluded — standard RQA."""
    n = R.shape[0]
    out = []
    for k in range(1, n):  # upper triangle; matrix is symmetric
        out.append(_run_lengths(np.diagonal(R, k)))
    lines = np.concatenate(out) if out else np.array([], int)
    return np.concatenate([lines, lines])  # mirror the lower triangle


def _vert_lines(R: np.ndarray) -> np.ndarray:
    """Vertical line lengths over all columns (one run pass on a padded flat)."""
    padded = np.vstack([R, np.zeros((1, R.shape[1]), bool)])
    return _run_lengths(padded.T.ravel())


def _weighted_frac(lines: np.ndarray, lmin: int) -> float:
    tot = lines.sum()
    if tot == 0:
        return 0.0
    return float(lines[lines >= lmin].sum() / tot)


def _mean_len(lines: np.ndarray, lmin: int) -> float:
    sel = lines[lines >= lmin]
    return float(sel.mean()) if len(sel) else 0.0


def _line_entropy(lines: np.ndarray, lmin: int) -> float:
    sel = lines[lines >= lmin]
    if len(sel) == 0:
        return 0.0
    h = np.bincount(sel)[lmin:]
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum())


def rqa_features(w: np.ndarray, dim: int = 3, tau: int = 5,
                 rr: float = 0.1) -> np.ndarray:
    """6 standard RQA measures of one window: RR, DET, LAM, diagonal-line
    entropy, mean diagonal length, trapping time. Supremum-norm recurrence,
    threshold set from the target recurrence rate (distance quantile) —
    the full window is used (no subsampling; that was the fairness lesson)."""
    X = _embed(np.asarray(w, float), dim, tau)
    D = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
    R = D < np.quantile(D, rr)
    diag = _diag_lines(R)
    vert = _vert_lines(R)
    n = R.shape[0]
    return np.array([
        float((R.sum() - np.trace(R)) / (n * n - n)),  # RR off the LOI
        _weighted_frac(diag, 2),                       # DET
        _weighted_frac(vert, 2),                       # LAM
        _line_entropy(diag, 2),                        # diag ENTR
        _mean_len(diag, 2),                            # L_mean
        _mean_len(vert, 2),                            # TT
    ])


# ------------------------------------------------------------- coherence
def coherence_feature_names(n_channels: int, n_bands: int = 2) -> list[str]:
    """Names for the (channel 0, channel j) pair/band grid, row-major in j."""
    return [f"coh_ch{j}_b{k}"
            for j in range(1, n_channels) for k in range(n_bands)]


def coherence_features(w: np.ndarray, nperseg: int = 128,
                       n_bands: int = 2) -> np.ndarray:
    """Magnitude-squared coherence of each pair (channel 0, channel j), averaged
    in `n_bands` equal frequency bands. The defaults are the FIXED product
    config c128b2 — the coherence_fair CONFIRMED-aug config (HYD-cooler,
    logs/coherence_fair.csv, prereg 4bb76422) — never a grid; semantics match
    coherence_fair.coh_feats exactly (parity pinned in tests)."""
    from scipy.signal import coherence
    X = np.asarray(w, float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(
            f"coherence needs a (C>=2, n) multi-channel window, got shape "
            f"{X.shape}")
    a = X[0]
    row: list[float] = []
    for j in range(1, X.shape[0]):
        b = X[j]
        n = min(len(a), len(b))
        if n < 8:
            row += [0.0] * n_bands
            continue
        npe = min(nperseg, n)
        _, Cxy = coherence(a[:n], b[:n], nperseg=npe, noverlap=npe // 2)
        row += [float(np.mean(c)) for c in np.array_split(Cxy, n_bands)]
    return np.array(row, float)


# ------------------------------------------------------------- registry
@dataclass(frozen=True)
class PremiumFamily:
    """An opt-in distill challenger: window -> feature vector, plus the cost
    class the receipt reports. `needs_channels` declares the minimum channel
    count the featurizer requires: 1-channel families receive channel 0 of a
    multi-channel window; multi-channel families receive the full (C, 1024)
    window and distill refuses LOUDLY on banks with fewer channels."""
    name: str
    featurize: Callable[[np.ndarray], np.ndarray]
    feature_names: list[str] = field(default_factory=list)
    cost_note: str = ""
    needs_channels: int = 1


PREMIUM_FAMILIES: dict[str, PremiumFamily] = {
    "rqa": PremiumFamily(
        name="rqa",
        featurize=rqa_features,
        feature_names=RQA_FEATURE_NAMES,
        cost_note="O(n^2) recurrence matrix per window (~200x the base grammar)",
    ),
    "coherence": PremiumFamily(
        name="coherence",
        featurize=coherence_features,
        feature_names=[],  # (C-1)*n_bands, see coherence_feature_names(C)
        cost_note="O(C n log n) Welch cross-spectra per (ch0, chj) pair; fixed "
                  "product config c128b2 (coherence_fair CONFIRMED)",
        needs_channels=2,
    ),
    "envelope": PremiumFamily(
        name="envelope",
        featurize=envelope_features,
        feature_names=ENVELOPE_FEATURE_NAMES,
        cost_note="O(n log n) Hilbert+FFT per window (cheap)",
    ),
}
