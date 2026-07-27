"""Directional causal discovery via Convergent Cross Mapping (CCM / EDM).

This is SignalMap's differentiating core. Correlation — even confound-adjusted,
nonlinear, partial correlation (see ``coupling.py``) — is *symmetric*: it can
tell you two channels are related, never which one drives the other. CCM reads
direction out of the *geometry* of the reconstructed state space:

    If X drives Y, then Y's dynamics carry a signature of X. So the time-delay
    "shadow manifold" of Y can be used to recover X. Strong recovery of X from
    Y's manifold ⇒ X causes Y. The reverse recovery measures Y → X.

This is a fundamentally different learning paradigm from deep nets and from
statistics: model-free, label-free, no gradients — Takens-embedding geometry
(Sugihara et al., *Science* 2012). Pure numpy, tiny compute, interpretable.

A discovered direction is a hypothesis backed by the convergence test
(``ccm_convergence``): real causation makes cross-map skill *rise* with library
length and saturate; a spurious correlation does not converge.
"""
from __future__ import annotations

import numpy as np


def _embed(series: np.ndarray, E: int, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """Time-delay embedding. Returns (vectors [n_pts, E], time-indices)."""
    series = np.asarray(series, float)
    span = (E - 1) * tau
    if span >= len(series):
        raise ValueError("series too short for the requested E and tau")
    idx = np.arange(span, len(series))
    vecs = np.column_stack([series[idx - k * tau] for k in range(E)])
    return vecs, idx


def cross_map(library: np.ndarray, target: np.ndarray, E: int = 3, tau: int = 1,
              lib_size: int | None = None, seed: int = 0) -> float:
    """Cross-map skill: reconstruct ``target`` from the shadow manifold of
    ``library`` and return Pearson rho between estimate and truth.

    Each manifold point is predicted from its E+1 nearest neighbours (excluding
    itself), with Sugihara's exponential distance weighting. High rho means the
    library's state space encodes the target — evidence the target *drives* the
    library.
    """
    vecs, idx = _embed(library, E, tau)
    tgt = np.asarray(target, float)[idx]
    n = len(idx)
    k = E + 1
    if n <= k:
        raise ValueError("not enough embedded points for E+1 neighbours")

    if lib_size is not None and lib_size < n:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(n, lib_size, replace=False))
    else:
        sel = np.arange(n)

    lib_vecs = vecs[sel]
    lib_tgt = tgt[sel]
    # pairwise distances query(all n) x library(sel)
    diff = vecs[:, None, :] - lib_vecs[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    # exclude self-matches: library point j drops out when it is query i itself
    dist[sel[None, :] == np.arange(n)[:, None]] = np.inf

    nn = np.argpartition(dist, k, axis=1)[:, :k]
    nn_d = np.take_along_axis(dist, nn, axis=1)
    # weights: exp(-d / d_min), guard against zero/degenerate distances
    d_min = nn_d.min(axis=1, keepdims=True)
    d_min = np.where(d_min == 0, 1e-12, d_min)
    w = np.exp(-nn_d / d_min)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum == 0, 1e-12, w_sum)
    w /= w_sum
    preds = (w * lib_tgt[nn]).sum(axis=1)

    if preds.std() == 0 or tgt.std() == 0:
        return 0.0
    return float(np.corrcoef(preds, tgt)[0, 1])


def causal_strength(cause: np.ndarray, effect: np.ndarray, E: int = 3,
                    tau: int = 1, lib_size: int | None = None,
                    seed: int = 0) -> float:
    """Evidence in [0, 1] that ``cause`` drives ``effect``.

    Computed by recovering ``cause`` from the shadow manifold of ``effect``: if
    the cause drives the effect, the effect's state space carries the cause.
    Negative cross-map rho (no skill) is clamped to 0.
    """
    rho = cross_map(library=effect, target=cause, E=E, tau=tau,
                    lib_size=lib_size, seed=seed)
    return max(0.0, rho)


def transfer_entropy(source: np.ndarray, target: np.ndarray, bins: int = 8) -> float:
    """Transfer entropy source -> target (nats), history length 1.

    TE = I(target_{t+1} ; source_t | target_t): the extra information the
    source's present gives about the target's future, beyond the target's own
    past. An information-theoretic causal voter that is *independent* of CCM's
    state-space geometry — it works in noisy / non-chaotic regimes where CCM is
    weak (e.g. tightly controlled process data). Binned plug-in estimator,
    numpy only. >= 0 (clamped); small residual values are finite-sample bias.
    """
    x = np.asarray(source, float)
    y = np.asarray(target, float)
    yt1, yt, xt = y[1:], y[:-1], x[:-1]

    def _disc(v):
        edges = np.histogram_bin_edges(v, bins=bins)
        return np.clip(np.digitize(v, edges[1:-1]), 0, bins - 1)

    a, b, c = _disc(yt1), _disc(yt), _disc(xt)  # future, target-past, source-past
    n = len(a)
    # joint counts p(a,b,c)
    p_abc = np.zeros((bins, bins, bins))
    np.add.at(p_abc, (a, b, c), 1.0)
    p_abc /= n
    p_bc = p_abc.sum(axis=0)          # p(target-past, source-past)
    p_ab = p_abc.sum(axis=2)          # p(future, target-past)
    p_b = p_abc.sum(axis=(0, 2))      # p(target-past)

    te = 0.0
    nz = np.argwhere(p_abc > 0)
    for i, j, k in nz:
        num = p_abc[i, j, k] * p_b[j]
        den = p_bc[j, k] * p_ab[i, j]
        if num > 0 and den > 0:
            te += p_abc[i, j, k] * np.log(num / den)
    return float(max(te, 0.0))


def granger_causality(source: np.ndarray, target: np.ndarray, lag: int = 2) -> float:
    """Granger causality source -> target (log variance-ratio, >= 0).

    Does the source's past reduce the prediction error of the target beyond the
    target's own past? Linear AR voter — fast, and strong exactly where CCM/TE
    are weak (near-linear, controlled regimes). Complementary third voter.
    """
    x = np.asarray(source, float)
    y = np.asarray(target, float)
    n = len(y)
    if n <= 2 * lag + 2:
        return 0.0
    yt = y[lag:]

    def _design(series_list):
        cols = [np.ones(len(yt))]
        for s in series_list:
            for l in range(1, lag + 1):
                cols.append(s[lag - l:n - l])
        return np.column_stack(cols)

    def _rss(design):
        beta, *_ = np.linalg.lstsq(design, yt, rcond=None)
        resid = yt - design @ beta
        return float(resid @ resid)

    rss_restricted = _rss(_design([y]))
    rss_full = _rss(_design([y, x]))
    if rss_full <= 0 or rss_restricted <= 0:
        return 0.0
    return float(max(0.0, np.log(rss_restricted / rss_full)))


def _var_edge_scores(channels: dict[str, np.ndarray], lag: int = 1) -> dict[tuple[str, str], float]:
    """Directed coupling = magnitude of the lag-1 VAR coefficient cause->effect,
    fit jointly (so each coefficient is controlled for every other variable's
    past). This is the structural gain; its drift is the leading indicator of a
    controller-masked fault."""
    names = list(channels)
    X = np.column_stack([np.asarray(channels[k], float) for k in names])
    T, m = X.shape
    Y = X[lag:]
    design = np.column_stack([np.ones(T - lag)]
                             + [X[lag - l:T - l] for l in range(1, lag + 1)])
    beta, *_ = np.linalg.lstsq(design, Y, rcond=None)  # (1 + m*lag) x m
    out: dict[tuple[str, str], float] = {}
    for ei, effect in enumerate(names):
        for ci, cause in enumerate(names):
            if ci == ei:
                continue
            out[(cause, effect)] = float(abs(beta[1 + ci, ei]))  # lag-1 block
    return out


def edge_scores(channels: dict[str, np.ndarray], method: str = "ccm",
                E: int = 3, tau: int = 1, lib_size: int | None = None,
                bins: int = 8, lag: int = 2, seed: int = 0) -> dict[tuple[str, str], float]:
    """Directed score for every ordered channel pair under one voter.

    ``method="ccm"`` = cross-map skill (state-space geometry);
    ``method="te"`` = transfer entropy (information flow);
    ``method="granger"`` = linear predictive causality;
    ``method="var"`` = structural lag-1 VAR coupling (for causal-drift / lead-time
    monitoring). They shine in different regimes, so the right one is chosen per
    task rather than blindly fused.
    """
    if method == "var":
        return _var_edge_scores(channels, lag=1)
    names = list(channels)
    out: dict[tuple[str, str], float] = {}
    for cause in names:
        for effect in names:
            if cause == effect:
                continue
            if method == "ccm":
                out[(cause, effect)] = causal_strength(
                    channels[cause], channels[effect], E=E, tau=tau,
                    lib_size=lib_size, seed=seed)
            elif method == "te":
                out[(cause, effect)] = transfer_entropy(
                    channels[cause], channels[effect], bins=bins)
            elif method == "granger":
                out[(cause, effect)] = granger_causality(
                    channels[cause], channels[effect], lag=lag)
            else:
                raise ValueError(f"unknown method: {method!r}")
    return out


def root_cause_scores(baseline: dict[tuple[str, str], float],
                      fault: dict[tuple[str, str], float]) -> dict[str, float]:
    """Per-variable root-cause score = total absolute change of the causal
    edges touching it (as cause or effect), baseline vs fault."""
    variables = {v for edge in baseline for v in edge}
    score = {v: 0.0 for v in variables}
    for edge, base in baseline.items():
        if edge in fault:
            d = abs(base - fault[edge])
            score[edge[0]] += d
            score[edge[1]] += d
    return score


def fuse_rankings(*score_dicts: dict[str, float]) -> list[str]:
    """OR-like rank fusion: order variables by their *best* rank across voters
    (ties broken by total rank). Unlike AND/product consensus, this keeps a
    variable that a single voter strongly flags — essential because CCM and TE
    are complementary (each wins on different fault types). Empirically this is
    more robust across TEP faults than any single method.
    """
    variables = set().union(*[set(d) for d in score_dicts])
    ranks: dict[str, list[int]] = {v: [] for v in variables}
    for d in score_dicts:
        order = sorted(d, key=lambda k: d[k], reverse=True)
        pos = {k: i for i, k in enumerate(order)}
        for v in variables:
            ranks[v].append(pos.get(v, len(order)))
    return sorted(variables, key=lambda v: (min(ranks[v]), sum(ranks[v])))


def source_scores(channels: dict[str, np.ndarray], method: str = "ccm",
                  E: int = 3, tau: int = 1, lib_size: int | None = None,
                  bins: int = 8, lag: int = 2, seed: int = 0) -> dict[str, float]:
    """Net outgoing causal influence per variable = (sum of outgoing edge
    scores) - (sum of incoming). A *driver* scores high; a driven *sink* scores
    negative. This ranks variables by how much they CAUSE the rest — the basis
    for source localization in a flood of co-deviating signals, which magnitude
    and correlation cannot do (they rank by loudness / co-movement, not drive).
    """
    es = edge_scores(channels, method=method, E=E, tau=tau, lib_size=lib_size,
                     bins=bins, lag=lag, seed=seed)
    score = {k: 0.0 for k in channels}
    for (cause, effect), v in es.items():
        score[cause] += v
        score[effect] -= v
    return score


def localize_source(channels: dict[str, np.ndarray],
                    methods: tuple[str, ...] = ("te", "granger"),
                    E: int = 3, tau: int = 1, lib_size: int | None = None,
                    bins: int = 8, lag: int = 2, seed: int = 0) -> list[str]:
    """Rank variables by causal source-ness, fused across voters (OR-like).

    Returns variables most-likely-driver first. Default voters are transfer
    entropy + Granger — both competent at source localization in feed-forward
    systems. CCM is deliberately NOT a default here: it targets chaotic-attractor
    *direction discovery*, and is systematically wrong on linear/noisy feed-
    forward data, where it would *poison* the fusion (a confidently-wrong voter
    cannot be safely fused). Add "ccm" explicitly only for chaotic regimes.
    """
    dicts = [source_scores(channels, method=m, E=E, tau=tau, lib_size=lib_size,
                           bins=bins, lag=lag, seed=seed) for m in methods]
    return fuse_rankings(*dicts)


def causal_consensus(channels: dict[str, np.ndarray], E: int = 3, tau: int = 1,
                     ccm_min: float = 0.3, te_min: float = 0.02,
                     lib_size: int | None = None, bins: int = 8,
                     seed: int = 0) -> list[dict]:
    """Two-voter directed graph: CCM (state-space geometry) + transfer entropy
    (information flow). Each ordered pair gets both scores; ``agree`` is True
    when *both* clear their thresholds. Consensus suppresses the single-method
    failures (CCM under synchrony, TE under sparse data) — the ensemble that is
    the actual moat. Sorted by combined confidence.
    """
    names = list(channels)
    edges: list[dict] = []
    for cause in names:
        for effect in names:
            if cause == effect:
                continue
            ccm = causal_strength(channels[cause], channels[effect],
                                  E=E, tau=tau, lib_size=lib_size, seed=seed)
            te = transfer_entropy(channels[cause], channels[effect], bins=bins)
            agree = ccm >= ccm_min and te >= te_min
            edges.append({"cause": cause, "effect": effect, "ccm": ccm,
                          "te": te, "agree": agree})
    edges.sort(key=lambda e: (e["agree"], e["ccm"] + e["te"]), reverse=True)
    return edges


def causal_graph(channels: dict[str, np.ndarray], E: int = 3, tau: int = 1,
                 min_strength: float = 0.3, require_convergence: bool = True,
                 lib_sizes: tuple[int, int] = (100, 1000), seed: int = 0) -> list[dict]:
    """Directed causal graph over named channels via pairwise CCM.

    For every ordered pair (cause -> effect) measure cross-map skill at a small
    and a large library. An edge is kept when the large-library strength clears
    ``min_strength`` and (if ``require_convergence``) skill did not *fall* with
    more data — the convergence signature that separates causation from a
    spurious correlation. Returned sorted by strength, descending.
    """
    small, large = lib_sizes
    names = list(channels)
    edges: list[dict] = []
    for cause in names:
        for effect in names:
            if cause == effect:
                continue
            s_large = causal_strength(channels[cause], channels[effect],
                                      E=E, tau=tau, lib_size=large, seed=seed)
            s_small = causal_strength(channels[cause], channels[effect],
                                      E=E, tau=tau, lib_size=small, seed=seed)
            converges = s_large >= s_small - 0.05
            if s_large < min_strength:
                continue
            if require_convergence and not converges:
                continue
            edges.append({"cause": cause, "effect": effect,
                          "strength": s_large, "converges": converges})
    edges.sort(key=lambda e: e["strength"], reverse=True)
    return edges


def fit_causal(channels: dict[str, np.ndarray], E: int = 3, tau: int = 1,
               lib_size: int | None = None, seed: int = 0) -> dict[tuple[str, str], float]:
    """Learn the baseline causal-strength of every ordered channel pair from
    healthy data. The returned map is the reference for ``diagnose``."""
    names = list(channels)
    baseline: dict[tuple[str, str], float] = {}
    for cause in names:
        for effect in names:
            if cause == effect:
                continue
            baseline[(cause, effect)] = causal_strength(
                channels[cause], channels[effect],
                E=E, tau=tau, lib_size=lib_size, seed=seed)
    return baseline


def diagnose(baseline: dict[tuple[str, str], float], channels: dict[str, np.ndarray],
             E: int = 3, tau: int = 1, lib_size: int | None = None,
             seed: int = 0) -> list[dict]:
    """Root-cause: rank causal edges by how far their strength dropped vs the
    healthy baseline. The top entry is the link that degraded most — the
    mechanism most likely responsible, not just a global anomaly score."""
    ranking: list[dict] = []
    for (cause, effect), base in baseline.items():
        cur = causal_strength(channels[cause], channels[effect],
                              E=E, tau=tau, lib_size=lib_size, seed=seed)
        ranking.append({"cause": cause, "effect": effect, "baseline": base,
                        "current": cur, "drop": base - cur})
    ranking.sort(key=lambda r: r["drop"], reverse=True)
    return ranking


def ccm_convergence(cause: np.ndarray, effect: np.ndarray, E: int = 3,
                    tau: int = 1, lib_sizes=(50, 150, 400, 1000),
                    seed: int = 0) -> tuple[list[int], list[float]]:
    """Cross-map skill of ``cause`` -> ``effect`` at growing library lengths.

    Rising, saturating skill is the signature of genuine causation (vs. a
    spurious correlation, which stays flat). Returns (lib_sizes, skills).
    """
    sizes = [s for s in lib_sizes]
    skills = [causal_strength(cause, effect, E=E, tau=tau, lib_size=s, seed=seed)
              for s in sizes]
    return sizes, skills
