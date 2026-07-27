"""Ground-truth tests for directional causal discovery (CCM / EDM).

The whole value proposition lives or dies here: on a system where X drives Y
but Y does NOT drive X, linear correlation is *symmetric* and cannot reveal the
direction. Convergent Cross Mapping must recover the asymmetry from the geometry
of the reconstructed state space alone — no labels, no model, numpy only.
"""
from __future__ import annotations

import numpy as np

from signalmap.causal import (causal_strength, ccm_convergence,
                              causal_graph, fit_causal, diagnose,
                              transfer_entropy, causal_consensus)


def coupled_logistic(n: int, beta_xy: float, beta_yx: float,
                     rx: float = 3.8, ry: float = 3.5,
                     x0: float = 0.4, y0: float = 0.2,
                     burn: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Sugihara (2012) two-species logistic system.

    beta_yx = strength with which X drives Y.
    beta_xy = strength with which Y drives X.
    """
    m = n + burn
    x = np.empty(m)
    y = np.empty(m)
    x[0], y[0] = x0, y0
    for t in range(m - 1):
        x[t + 1] = x[t] * (rx - rx * x[t] - beta_xy * y[t])
        y[t + 1] = y[t] * (ry - ry * y[t] - beta_yx * x[t])
    return x[burn:], y[burn:]


def test_ccm_recovers_direction_correlation_cannot():
    # X drives Y (0.10); Y has ZERO effect on X.
    x, y = coupled_logistic(n=1500, beta_xy=0.0, beta_yx=0.10)

    s_x_drives_y = causal_strength(cause=x, effect=y, E=3, tau=1)
    s_y_drives_x = causal_strength(cause=y, effect=x, E=3, tau=1)

    # True causal direction is strong...
    assert s_x_drives_y > 0.7, s_x_drives_y
    # ...the non-causal direction is weak...
    assert s_y_drives_x < 0.4, s_y_drives_x
    # ...and the directional gap is unambiguous.
    assert s_x_drives_y - s_y_drives_x > 0.35

    # Correlation is direction-blind: identical both ways, so it can NEVER
    # produce the asymmetry CCM just produced. This is the unique capability.
    r_xy = abs(np.corrcoef(x, y)[0, 1])
    r_yx = abs(np.corrcoef(y, x)[0, 1])
    assert abs(r_xy - r_yx) < 1e-9


def test_ccm_convergence_increases_with_library():
    # Convergence (skill rising with library length) is the signature that
    # distinguishes real causation from a spurious one-off correlation.
    x, y = coupled_logistic(n=1500, beta_xy=0.0, beta_yx=0.10)
    sizes, skills = ccm_convergence(cause=x, effect=y, E=3, tau=1,
                                    lib_sizes=(50, 150, 400, 1000))
    assert list(sizes) == [50, 150, 400, 1000]
    # Skill at the largest library exceeds skill at the smallest by a clear margin.
    assert skills[-1] - skills[0] > 0.1
    assert skills[-1] > skills[0]


def chain3(n, b_ab, b_bc, ra=3.72, rb=3.81, rc=3.91,
           a0=0.4, b0=0.3, c0=0.2, burn=300):
    """Directed chain A -> B -> C (each link strength configurable).

    r values are kept in the fully chaotic regime on purpose: CCM/EDM assumes a
    chaotic attractor. Periodic windows (e.g. r=3.5) make even *independent*
    series spuriously cross-mappable and would invalidate the test.
    """
    m = n + burn
    a = np.empty(m); b = np.empty(m); c = np.empty(m)
    a[0], b[0], c[0] = a0, b0, c0
    for t in range(m - 1):
        a[t + 1] = a[t] * (ra - ra * a[t])
        b[t + 1] = b[t] * (rb - rb * b[t] - b_ab * a[t])
        c[t + 1] = c[t] * (rc - rc * c[t] - b_bc * b[t])
    return {"A": a[burn:], "B": b[burn:], "C": c[burn:]}


def test_causal_graph_recovers_directed_chain():
    ch = chain3(n=1500, b_ab=0.1, b_bc=0.1)
    edges = causal_graph(ch, E=3, tau=1, min_strength=0.3)
    found = {(e["cause"], e["effect"]) for e in edges}

    # The two true forward links must be discovered...
    assert ("A", "B") in found
    assert ("B", "C") in found
    # ...and the reverse links (C cannot drive B, B cannot drive A) rejected.
    assert ("B", "A") not in found
    assert ("C", "B") not in found
    # Every reported edge passed the convergence gate.
    assert all(e["converges"] for e in edges)


def test_root_cause_identifies_broken_link():
    healthy = chain3(n=1500, b_ab=0.1, b_bc=0.1)
    faulty = chain3(n=1500, b_ab=0.1, b_bc=0.0)  # B -> C link severed

    baseline = fit_causal(healthy, E=3, tau=1)
    ranking = diagnose(baseline, faulty, E=3, tau=1)

    # Most-degraded causal edge is the one that actually broke.
    top = ranking[0]
    assert (top["cause"], top["effect"]) == ("B", "C")
    assert top["drop"] > 0.3
    # The intact A -> B link degraded far less than the severed B -> C link.
    a_b = next(r for r in ranking if (r["cause"], r["effect"]) == ("A", "B"))
    assert top["drop"] > a_b["drop"]


def test_causal_demo_reports_graph_and_root_cause():
    from signalmap.causal_discover import demo
    out = demo(n=1500, seed=0)
    found = {(e["cause"], e["effect"]) for e in out["graph"]}
    assert ("A", "B") in found and ("B", "C") in found
    assert ("C", "B") not in found and ("B", "A") not in found
    top = out["root_cause"][0]
    assert (top["cause"], top["effect"]) == ("B", "C")


def _indep_logistic(n, r, x0, burn=200):
    m = n + burn
    x = np.empty(m); x[0] = x0
    for t in range(m - 1):
        x[t + 1] = x[t] * (r - r * x[t])
    return x[burn:]


def test_transfer_entropy_directional():
    # X drives Y, Y does not drive X (independent voter, must agree with CCM).
    x, y = coupled_logistic(n=3000, beta_xy=0.0, beta_yx=0.10)
    te_xy = transfer_entropy(x, y, bins=8)
    te_yx = transfer_entropy(y, x, bins=8)
    assert te_xy >= 0 and te_yx >= 0
    assert te_xy > te_yx + 0.02
    assert te_yx < 0.05


def test_transfer_entropy_independent_near_zero():
    a = _indep_logistic(3000, 3.9, 0.31)
    b = _indep_logistic(3000, 3.7, 0.77)  # fully chaotic, independent
    assert transfer_entropy(a, b, bins=8) < 0.1
    assert transfer_entropy(b, a, bins=8) < 0.1


def test_causal_consensus_recovers_chain():
    ch = chain3(n=2000, b_ab=0.1, b_bc=0.1)
    edges = causal_consensus(ch, E=3, tau=1)
    agreed = {(e["cause"], e["effect"]) for e in edges if e["agree"]}
    assert ("A", "B") in agreed and ("B", "C") in agreed
    assert ("B", "A") not in agreed and ("C", "B") not in agreed
    # consensus edges carry both votes
    for e in edges:
        assert "ccm" in e and "te" in e


def test_edge_scores_per_method():
    from signalmap.causal import edge_scores
    ch = chain3(n=1500, b_ab=0.1, b_bc=0.1)
    ccm_s = edge_scores(ch, method="ccm", E=3, tau=1)
    te_s = edge_scores(ch, method="te")
    assert ccm_s[("A", "B")] > ccm_s[("B", "A")]
    assert te_s[("A", "B")] > te_s[("B", "A")]


def test_root_cause_scores_aggregates_changed_edges():
    from signalmap.causal import root_cause_scores
    base = {("X", "Y"): 0.8, ("Y", "X"): 0.1, ("X", "Z"): 0.5, ("Z", "X"): 0.2}
    fault = {("X", "Y"): 0.2, ("Y", "X"): 0.1, ("X", "Z"): 0.5, ("Z", "X"): 0.2}
    s = root_cause_scores(base, fault)
    assert s["Z"] == 0.0          # Z's edges unchanged
    assert s["X"] > 0 and s["Y"] > 0


def test_fuse_rankings_is_or_like_not_and():
    from signalmap.causal import fuse_rankings
    # a tops method 1, b tops method 2; OR-fusion must surface BOTH at the top,
    # unlike AND/product which would bury the single-method winners.
    d1 = {"a": 0.9, "b": 0.1, "c": 0.5}
    d2 = {"a": 0.1, "b": 0.9, "c": 0.5}
    fused = fuse_rankings(d1, d2)
    assert set(fused[:2]) == {"a", "b"}
    assert fused[2] == "c"


def test_ensemble_rootcause_recovers_severed_link():
    from signalmap.causal import edge_scores, root_cause_scores, fuse_rankings
    healthy = chain3(n=1500, b_ab=0.1, b_bc=0.1)
    faulty = chain3(n=1500, b_ab=0.1, b_bc=0.0)
    rc = {}
    for m in ("ccm", "te"):
        rc[m] = root_cause_scores(edge_scores(healthy, method=m, E=3, tau=1),
                                  edge_scores(faulty, method=m, E=3, tau=1))
    fused = fuse_rankings(rc["ccm"], rc["te"])
    # the severed B->C link involves B and C; both must rank above the untouched A.
    assert set(fused[:2]) == {"B", "C"}


def _linear_var(n, seed=0, burn=200):
    """Linear system X -> Y: Granger's home turf (where CCM/TE may be weaker)."""
    rng = np.random.default_rng(seed)
    m = n + burn
    x = np.zeros(m); y = np.zeros(m)
    ex = rng.standard_normal(m); ey = rng.standard_normal(m)
    for t in range(1, m):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.5 * y[t - 1] + 0.6 * x[t - 1] + ey[t]   # X drives Y, not vice versa
    return x[burn:], y[burn:]


def test_granger_directional_on_linear_system():
    from signalmap.causal import granger_causality
    x, y = _linear_var(2000)
    g_xy = granger_causality(x, y, lag=2)
    g_yx = granger_causality(y, x, lag=2)
    assert g_xy >= 0 and g_yx >= 0
    assert g_xy > g_yx + 0.05          # X->Y clearly stronger
    assert g_yx < 0.05                 # Y->X negligible


def test_granger_independent_near_zero():
    from signalmap.causal import granger_causality
    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    assert granger_causality(a, b, lag=2) < 0.05
    assert granger_causality(b, a, lag=2) < 0.05


def test_edge_scores_supports_granger():
    from signalmap.causal import edge_scores
    x, y = _linear_var(2000)
    s = edge_scores({"X": x, "Y": y}, method="granger", lag=2)
    assert s[("X", "Y")] > s[("Y", "X")]


def _propagation(n, fault, seed, a=0.4, g=1.6, burn=300):
    """Linear propagation S -> M -> L with downstream amplification (g>1).

    A fault injected at the SOURCE S deviates S the least and the downstream
    sink L the most. Magnitude-based RCA therefore picks the loud symptom L;
    only a directed-causality method can name S as the driver. This is the
    capability correlation/anomaly cannot match even in principle.
    """
    rng = np.random.default_rng(seed)
    m = n + burn
    S = np.zeros(m); M = np.zeros(m); L = np.zeros(m)
    es = rng.standard_normal(m); em = rng.standard_normal(m); el = rng.standard_normal(m)
    for t in range(1, m):
        sh = 3.0 if fault else 0.0
        S[t] = a * S[t - 1] + es[t] + sh
        M[t] = a * M[t - 1] + g * S[t - 1] + em[t]
        L[t] = a * L[t - 1] + g * M[t - 1] + el[t]
    return {"S": S[burn:], "M": M[burn:], "L": L[burn:]}


def test_source_scores_driver_positive_sink_negative():
    from signalmap.causal import source_scores
    f = _propagation(1800, fault=True, seed=0)
    s = source_scores(f, method="granger", lag=2)
    assert s["S"] > s["M"] > s["L"]   # net outgoing influence: driver high, sink negative


def test_causal_finds_source_where_magnitude_finds_symptom():
    # The headline claim, quantified across seeds.
    from signalmap.causal import localize_source
    trials = 6
    mag_found = causal_found = 0
    for seed in range(trials):
        h = _propagation(1800, fault=False, seed=seed)
        f = _propagation(1800, fault=True, seed=seed)
        mag = {k: abs(f[k].mean() - h[k].mean()) / (h[k].std() + 1e-9) for k in f}
        mag_top = max(mag, key=mag.get)
        causal_top = localize_source(f, lib_size=600)[0]
        mag_found += (mag_top == "S")
        causal_found += (causal_top == "S")
    assert mag_found == 0               # magnitude structurally never finds the quiet source
    assert causal_found >= trials - 1   # causal almost always does
