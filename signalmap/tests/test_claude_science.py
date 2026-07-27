"""The "Claude Science factor": acceptance tests for using SignalMap as a
skill/connector inside Anthropic's Claude Science workbench.

Claude Science (launched 2026-06-30) is *not a new model* — it is a coordinating
agent that orchestrates 60+ curated **skills and remote-MCP connectors** over
scientific domains (genomics, single-cell, systems biology, cheminformatics...),
and its headline promise is that **"every result is reproducible and traced to
its code."**  https://www.anthropic.com/news/claude-science-ai-workbench

For SignalMap to become an additional standbein there, its causal-discovery core
must satisfy three platform requirements. Each group below pins one of them:

  GROUP 1  Agent-invocability   — outputs are JSON round-trippable & schema-stable
                                   so the coordinating agent can call the skill
                                   over MCP and parse the result.
  GROUP 2  Auditable/reproducible — same inputs => identical results, and every
                                   run carries a self-describing provenance
                                   manifest (the Claude Science core promise).
  GROUP 3  Scientific-domain fit — the CCM/TE ensemble recovers *directed*
                                   mechanism and rejects shared-confound illusions
                                   on a biology-flavored dynamical system — the
                                   regime CCM was invented for (Sugihara 2012).

GREEN tests prove SignalMap already qualifies. The xfail tests are executable
specs for the two remaining gaps (a JSON adapter + a provenance manifest) that a
thin `signalmap.science` module must close to ship the skill; they flip to pass
the moment that module lands.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from signalmap.causal import (
    causal_consensus,
    causal_strength,
    ccm_convergence,
    edge_scores,
)
from signalmap.coupling import find_couplings


# --- shared ground truth --------------------------------------------------
# Sugihara (2012) two-species logistic system: X drives Y (beta_yx>0), Y does
# NOT drive X. Linear correlation is symmetric here and cannot reveal the arrow;
# CCM must. This is the canonical *biological/ecological* causality benchmark —
# exactly a Claude Science life-sciences use case, not an industrial-bearing one.
def coupled_logistic(n: int, beta_xy: float, beta_yx: float,
                     rx: float = 3.8, ry: float = 3.5,
                     x0: float = 0.4, y0: float = 0.2,
                     burn: int = 200) -> tuple[np.ndarray, np.ndarray]:
    m = n + burn
    x = np.empty(m)
    y = np.empty(m)
    x[0], y[0] = x0, y0
    for t in range(m - 1):
        x[t + 1] = x[t] * (rx - rx * x[t] - beta_xy * y[t])
        y[t + 1] = y[t] * (ry - ry * y[t] - beta_yx * x[t])
    return x[burn:], y[burn:]


# =========================================================================
# GROUP 1 — agent-invocability: the skill's output must survive MCP JSON I/O
# =========================================================================
def test_consensus_result_is_json_round_trippable():
    """The Claude Science agent receives skill output as JSON. causal_consensus
    already emits only native str/float/bool, so it round-trips losslessly."""
    x, y = coupled_logistic(n=800, beta_xy=0.0, beta_yx=0.10)
    edges = causal_consensus({"X": x, "Y": y}, E=3, tau=1)

    encoded = json.dumps(edges)          # must not raise
    restored = json.loads(encoded)
    assert restored == edges             # lossless: no numpy leaking through


def test_consensus_edge_schema_is_stable():
    """A skill is only agent-callable if its output schema is a fixed contract.
    Every consensus edge must carry exactly these keys with these types."""
    x, y = coupled_logistic(n=800, beta_xy=0.0, beta_yx=0.10)
    edges = causal_consensus({"X": x, "Y": y}, E=3, tau=1)

    assert edges, "expected at least one directed edge"
    for e in edges:
        assert set(e) == {"cause", "effect", "ccm", "te", "agree"}
        assert isinstance(e["cause"], str) and isinstance(e["effect"], str)
        assert isinstance(e["ccm"], float) and isinstance(e["te"], float)
        assert isinstance(e["agree"], bool)


def test_edge_scores_needs_json_adapter():
    from signalmap.science import jsonable  # does not exist yet -> xfail

    x, y = coupled_logistic(n=800, beta_xy=0.0, beta_yx=0.10)
    scores = edge_scores({"X": x, "Y": y}, method="ccm")
    # raw dict is NOT serializable (tuple keys) — the adapter must fix that:
    with pytest.raises(TypeError):
        json.dumps(scores)
    encoded = json.dumps(jsonable(scores))
    assert "X->Y" in encoded


# =========================================================================
# GROUP 2 — auditable & reproducible ("every result traced to its code")
# =========================================================================
def test_causal_discovery_is_deterministic():
    """Reproducibility is Claude Science's core guarantee. Same inputs + seed
    must yield byte-identical results across independent runs."""
    x, y = coupled_logistic(n=1000, beta_xy=0.0, beta_yx=0.10)
    a = causal_consensus({"X": x, "Y": y}, E=3, tau=1, seed=0)
    b = causal_consensus({"X": x, "Y": y}, E=3, tau=1, seed=0)
    assert a == b


def test_run_carries_provenance_manifest():
    from signalmap.science import run_manifest  # does not exist yet -> xfail

    x, y = coupled_logistic(n=800, beta_xy=0.0, beta_yx=0.10)
    env = run_manifest(
        result=causal_consensus({"X": x, "Y": y}),
        method="causal_consensus",
        params={"E": 3, "tau": 1},
        seed=0,
        inputs={"X": x, "Y": y},
    )
    for field in ("signalmap_version", "method", "params", "seed",
                  "input_hash", "result"):
        assert field in env
    json.dumps(env)  # the whole audit envelope must be transportable


# =========================================================================
# GROUP 3 — scientific-domain fit: recover the arrow, reject the confound
# =========================================================================
def test_recovers_directional_coupling_on_biological_system():
    """On the Sugihara predator/prey-style system, the ensemble must find the
    real arrow X->Y and NOT the reverse — the discovery correlation cannot make."""
    x, y = coupled_logistic(n=1500, beta_xy=0.0, beta_yx=0.10)

    s_xy = causal_strength(cause=x, effect=y, E=3, tau=1)
    s_yx = causal_strength(cause=y, effect=x, E=3, tau=1)
    assert s_xy > 0.7                      # X genuinely drives Y
    assert s_xy - s_yx > 0.35              # asymmetry is unambiguous

    # linear correlation is symmetric here => it is blind to direction
    r_xy = float(np.corrcoef(x[:-1], y[1:])[0, 1])
    r_yx = float(np.corrcoef(y[:-1], x[1:])[0, 1])
    assert abs(abs(r_xy) - abs(r_yx)) < 0.2  # correlation gives no clean arrow


def test_convergence_signature_separates_causation_from_correlation():
    """The audit-grade evidence: genuine causation makes cross-map skill *rise
    and saturate* with library size. A spurious correlation stays flat."""
    x, y = coupled_logistic(n=1500, beta_xy=0.0, beta_yx=0.10)
    sizes, skills = ccm_convergence(cause=x, effect=y, E=3, tau=1)
    assert list(sizes) == [50, 150, 400, 1000]
    assert skills[-1] > skills[0]          # convergent -> causal
    assert skills[-1] - skills[0] > 0.1


def test_shared_confound_correlation_is_rejected():
    """The failure mode that terrifies observational scientists: two readouts
    look coupled only because a hidden common driver moves both. Confound-adjusted
    discovery must NOT report the spurious A-B edge as a survivor."""
    rng = np.random.default_rng(0)
    n = 2000
    driver = np.sin(np.linspace(0, 40, n)) + 0.1 * rng.standard_normal(n)  # hidden Z
    a = driver + 0.05 * rng.standard_normal(n)   # readout A = f(Z)
    b = driver + 0.05 * rng.standard_normal(n)   # readout B = f(Z), no A<->B link

    results = find_couplings({"A": a, "B": b, "Z": driver}, confounds=["Z"])
    ab = next(r for r in results if {r["a"], r["b"]} == {"A", "B"})

    assert abs(ab["raw_corr"]) > 0.6       # naively they look strongly coupled...
    assert not ab["survives"]              # ...but the confound explains it away
    assert abs(ab["adj_corr"]) < 0.2       # residual coupling collapses near zero
