"""Causal discovery demo: learn a *directed* mechanism graph from unlabeled
multivariate signals, then root-cause a fault by the causal edge that broke.

This is the honest, runnable face of SignalMap's pivot from correlation to
causation. It synthesizes a known directed chain A -> B -> C (chaotic regime,
the only regime where CCM applies), recovers the graph with no labels, then
severs the B -> C link and shows that diagnosis points at exactly that edge —
something a correlation/anomaly score cannot localize.

    signalmap causal                 # synthetic directed-chain demo
"""
from __future__ import annotations

import numpy as np

from .causal import causal_graph, diagnose, fit_causal


def _chain(n: int, b_ab: float, b_bc: float, ra: float = 3.72, rb: float = 3.81,
           rc: float = 3.91, seed: int = 0, burn: int = 300) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a0, b0, c0 = rng.uniform(0.2, 0.6, 3)
    m = n + burn
    a = np.empty(m); b = np.empty(m); c = np.empty(m)
    a[0], b[0], c[0] = a0, b0, c0
    for t in range(m - 1):
        a[t + 1] = a[t] * (ra - ra * a[t])
        b[t + 1] = b[t] * (rb - rb * b[t] - b_ab * a[t])
        c[t + 1] = c[t] * (rc - rc * c[t] - b_bc * b[t])
    return {"A": a[burn:], "B": b[burn:], "C": c[burn:]}


def demo(n: int = 1500, seed: int = 0) -> dict:
    """Return {'graph': edges, 'root_cause': ranking} for a healthy chain whose
    B->C link is then severed."""
    healthy = _chain(n, b_ab=0.1, b_bc=0.1, seed=seed)
    faulty = _chain(n, b_ab=0.1, b_bc=0.0, seed=seed)
    graph = causal_graph(healthy, E=3, tau=1, min_strength=0.3)
    baseline = fit_causal(healthy, E=3, tau=1)
    ranking = diagnose(baseline, faulty, E=3, tau=1)
    return {"graph": graph, "root_cause": ranking}


def run(n: int = 1500, seed: int = 0) -> dict:
    out = demo(n=n, seed=seed)
    print("directed causal graph (unlabeled, CCM/EDM)")
    print(f"  ground truth: A->B->C, {n} samples\n")
    for e in out["graph"]:
        print(f"  {e['cause']} -> {e['effect']}   strength={e['strength']:.3f}"
              f"   converges={e['converges']}")
    print("\nfault injected: B->C link severed — root-cause ranking by strength drop")
    for r in out["root_cause"]:
        flag = "  <== ROOT CAUSE" if r is out["root_cause"][0] else ""
        print(f"  {r['cause']} -> {r['effect']}   baseline={r['baseline']:.3f}"
              f"   current={r['current']:.3f}   drop={r['drop']:+.3f}{flag}")
    return out


def run_file(path, columns: list[str] | None = None, E: int = 3, tau: int = 1,
             min_strength: float = 0.3) -> list[dict]:
    """Discover a directed causal graph from a real CSV/NPY recording."""
    from .multichannel import load_channels
    channels = load_channels(path, columns=columns)
    graph = causal_graph(channels, E=E, tau=tau, min_strength=min_strength)
    print(f"directed causal graph (unlabeled, CCM/EDM) — {path}")
    print(f"  {len(channels)} channels, {len(next(iter(channels.values())))} samples\n")
    for e in graph:
        print(f"  {e['cause']} -> {e['effect']}   strength={e['strength']:.3f}"
              f"   converges={e['converges']}")
    if not graph:
        print("  (no causal edges cleared the strength/convergence gate)")
    return graph


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Directed causal discovery + root-cause")
    p.add_argument("--csv", help="real recording (CSV/NPY) instead of the synthetic demo")
    p.add_argument("--column", action="append", help="repeatable; select channels")
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.csv:
        run_file(args.csv, columns=args.column)
    else:
        run(n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
