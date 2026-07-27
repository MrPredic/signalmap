"""Benchmark: directed-causal source localization vs magnitude/anomaly.

The headline capability, quantified honestly. We generate random feed-forward
systems (a fault at the root, amplified downstream so the root is the QUIETEST
node), then measure how often each method names the true root cause. Magnitude/
anomaly structurally cannot win here — it ranks by loudness, so it picks the
amplified downstream symptom. Directed causality ranks by who-drives-whom.

This module is import-light (numpy + signalmap.causal) and deterministic, so it
doubles as a regression test of the core claim.
"""
from __future__ import annotations

import numpy as np

from .causal import localize_source, source_scores


def random_feedforward(n_nodes: int, fault: bool, seed: int, regime: str = "linear",
                       a: float = 0.4, g: float = 1.3, shift: float = 3.0,
                       n: int = 1500, burn: int = 300):
    """Random feed-forward tree: each node has one random earlier parent; the
    fault is a step injected at the root v0 and amplified (g>1) downstream.
    Returns (channels, source_name). ``regime`` is "linear" or "nonlinear".
    """
    rng = np.random.default_rng(seed)
    parents = {i: int(rng.integers(0, i)) for i in range(1, n_nodes)}
    T = n + burn
    X = np.zeros((n_nodes, T))
    noise = rng.standard_normal((n_nodes, T))
    X[:, 0] = rng.uniform(0.2, 0.6, n_nodes)
    for t in range(1, T):
        for i in range(n_nodes):
            v = a * X[i, t - 1] + noise[i, t]
            if i == 0:
                v += shift if fault else 0.0
            else:
                p = X[parents[i], t - 1]
                v += g * p + (0.5 * np.sin(p) if regime == "nonlinear" else 0.0)
            X[i, t] = v
    return {f"v{i}": X[i, burn:] for i in range(n_nodes)}, "v0"


def _hits(order, truth, k):
    return truth in order[:k]


def source_localization_benchmark(trials: int = 20, regime: str = "linear",
                                  node_range: tuple[int, int] = (3, 6),
                                  seed0: int = 2000) -> dict:
    """Run ``trials`` random feed-forward systems; return per-method top-1/top-3
    accuracy at naming the true source. Methods: magnitude (anomaly baseline),
    each causal voter, and the fused causal localizer.
    """
    methods = ["magnitude", "te", "granger", "fused"]
    acc = {m: {"top1": 0, "top3": 0} for m in methods}
    for s in range(trials):
        n_nodes = int(np.random.default_rng(seed0 + s).integers(*node_range))
        healthy, _ = random_feedforward(n_nodes, False, s, regime)
        faulty, truth = random_feedforward(n_nodes, True, s, regime)

        mag = {k: abs(faulty[k].mean() - healthy[k].mean()) / (healthy[k].std() + 1e-9)
               for k in faulty}
        orders = {
            "magnitude": sorted(mag, key=mag.get, reverse=True),
            "te": sorted(s_te := source_scores(faulty, method="te"), key=s_te.get, reverse=True),
            "granger": sorted(s_g := source_scores(faulty, method="granger"), key=s_g.get, reverse=True),
            "fused": localize_source(faulty),
        }
        for m, order in orders.items():
            acc[m]["top1"] += _hits(order, truth, 1)
            acc[m]["top3"] += _hits(order, truth, 3)
    return {m: {"top1": v["top1"] / trials, "top3": v["top3"] / trials}
            for m, v in acc.items()}


def main() -> None:
    for regime in ("linear", "nonlinear"):
        r = source_localization_benchmark(regime=regime, trials=30)
        print(f"\n=== source localization — {regime} regime (30 random trees) ===")
        print(f"  {'method':10s} {'top-1':>7} {'top-3':>7}")
        for m, v in r.items():
            print(f"  {m:10s} {v['top1']:6.0%} {v['top3']:6.0%}")


if __name__ == "__main__":
    main()
