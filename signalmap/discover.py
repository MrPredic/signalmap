"""Cross-modal discovery run: deploy many modalities on one phenomenon, learn
which channels are genuinely coupled (vs. correlated only through a shared
confound), and report survivors as candidates for controlled follow-up.

This is the rigorous, honest realization of "build many points that measure very
different things, then see what couples" — with confound ablation so it cannot
fool itself. v0 runs on synthetic ground-truth data; real multi-modal capture
plugs in via the same `channels` dict.

    signalmap discover                 # synthetic ground-truth demo
    signalmap discover --confound temp
"""
from __future__ import annotations

import numpy as np

from .coupling import find_couplings
from .synth_multimodal import (CONFOUNDED_PAIRS, GROUND_TRUTH_COUPLINGS,
                               make_multimodal)


def run(confounds: list[str] | None = None, naive: bool = False,
        n: int = 2000, seed: int = 0) -> list[dict]:
    confounds = confounds if confounds is not None else ["temp"]
    channels = make_multimodal(n=n, seed=seed)
    results = find_couplings(channels, confounds=[] if naive else confounds)

    mode = "NAIVE (no confound removal)" if naive else f"confound-adjusted (given: {confounds})"
    print(f"cross-modal coupling scan — {mode}")
    print(f"  {len(channels)} channels, {n} samples\n")
    print(f"  {'pair':22s} {'raw':>7} {'MI':>6} {'adj':>7} {'p':>7}  verdict")
    for r in results:
        pair = f"{r['a']}–{r['b']}"
        verdict = "★ COUPLING" if r["survives"] else ("confound" if abs(r["raw_corr"]) > 0.3 else "—")
        print(f"  {pair:22s} {r['raw_corr']:+.3f} {r['mi']:6.3f} "
              f"{r['adj_corr']:+.3f} {r['p']:7.4f}  {verdict}")

    survivors = {tuple(sorted((r["a"], r["b"]))) for r in results if r["survives"]}
    print(f"\n  survivors (candidates for controlled validation): "
          f"{sorted(survivors) or 'none'}")
    return results


def evaluate(seed: int = 0) -> dict:
    """Self-check against ground truth: does the apparatus keep the real
    coupling AND reject the confounded pair? Returns a verdict dict."""
    channels = make_multimodal(seed=seed)
    res = find_couplings(channels, confounds=["temp"])
    survivors = {tuple(sorted((r["a"], r["b"]))) for r in res if r["survives"]}

    truth = {tuple(sorted(p)) for p in GROUND_TRUTH_COUPLINGS}
    confounded = {tuple(sorted(p)) for p in CONFOUNDED_PAIRS}

    by_pair = {tuple(sorted((r["a"], r["b"]))): r for r in res}
    cp = next(iter(confounded))
    return {
        "true_coupling_found": truth.issubset(survivors),
        "confound_rejected": confounded.isdisjoint(survivors),
        "confound_raw_corr": abs(by_pair[cp]["raw_corr"]),
        "confound_adj_corr": abs(by_pair[cp]["adj_corr"]),
        "survivors": sorted(survivors),
    }


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Cross-modal coupling discovery")
    p.add_argument("--confound", action="append", help="channel(s) to treat as confound")
    p.add_argument("--naive", action="store_true", help="no confound removal (shows the trap)")
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()
    run(confounds=args.confound, naive=args.naive, n=args.n)


if __name__ == "__main__":
    main()
