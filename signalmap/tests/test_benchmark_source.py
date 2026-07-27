"""Rigorous, deterministic benchmark test for the headline capability:
directed-causal source localization vs magnitude across random feed-forward
topologies and dynamical regimes. This is the regression guard for the ">=90%"
claim — it must keep holding for any change to ship.
"""
from __future__ import annotations

from signalmap.causal_benchmark import source_localization_benchmark


def test_fused_source_localization_dominates_magnitude_linear():
    r = source_localization_benchmark(trials=20, regime="linear")
    # Causal names the true root cause in the top-3 ~always; magnitude rarely
    # (the source is the QUIETEST node — amplified downstream symptoms are louder).
    assert r["fused"]["top3"] >= 0.85
    assert r["magnitude"]["top3"] <= 0.5
    assert r["fused"]["top1"] - r["magnitude"]["top1"] >= 0.4


def test_fused_source_localization_dominates_magnitude_nonlinear():
    r = source_localization_benchmark(trials=20, regime="nonlinear")
    assert r["fused"]["top3"] >= 0.85
    assert r["fused"]["top1"] - r["magnitude"]["top1"] >= 0.3
