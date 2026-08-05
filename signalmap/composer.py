"""Bounded compositional measurement search.

Research/product bridge: expressions are data, not executable code. The module
only evaluates a fixed operator vocabulary and is safe to serialize by its
canonical expression string.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np

# Fixed lag view, deliberately distinct from the lag-1 difference (`diff`).
# The 16-sample floor in `_view` keeps this well defined for every scale.
LAG_SAMPLES = 8

GRAMMAR_VERSION = "composer-v2"


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    params: tuple = ()

    def canonical(self) -> str:
        return json.dumps({"op": self.op,
                           "args": [json.loads(a.canonical()) for a in self.args],
                           "params": list(self.params)},
                          sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def cost(self) -> int:
        own = {"view": 1, "stat": 2, "difference": 2, "ratio": 3,
               "coherence": 5}.get(self.op, 1)
        return own + sum(a.cost() for a in self.args)


def _windows(windows: np.ndarray) -> np.ndarray:
    x = np.asarray(windows, dtype=float)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3:
        raise ValueError("windows must have shape (n,time) or (n,channels,time)")
    return x


def _view(expr: Expr, x: np.ndarray) -> np.ndarray:
    rep, channel, scale = expr.params
    if not isinstance(channel, int) or channel < 0 or channel >= x.shape[1]:
        raise ValueError(f"channel {channel} unavailable")
    v = x[:, channel, :]
    n = max(16, int(v.shape[1] * {"short": 0.25, "medium": 0.5,
                                  "full": 1.0}[scale]))
    v = v[:, -n:]
    if rep == "raw":
        return v
    if rep == "diff":
        return np.diff(v, axis=1)
    if rep == "envelope":
        k = min(8, max(2, v.shape[1] // 16))
        kernel = np.ones(k) / k
        return np.stack([np.convolve(np.abs(row), kernel, mode="valid") for row in v])
    if rep == "spectrum":
        return np.abs(np.fft.rfft(v, axis=1))
    if rep == "phase":
        return np.unwrap(np.angle(np.fft.rfft(v, axis=1)), axis=1)
    if rep == "lag":
        return v[:, LAG_SAMPLES:] - v[:, :-LAG_SAMPLES]
    raise ValueError(f"unknown representation {rep!r}")


def _stat(name: str, v: np.ndarray) -> np.ndarray:
    if name == "meanabs":
        return np.mean(np.abs(v), axis=1)
    if name == "std":
        return np.std(v, axis=1)
    if name == "iqr":
        return np.quantile(v, 0.9, axis=1) - np.quantile(v, 0.1, axis=1)
    if name == "slope":
        t = np.arange(v.shape[1], dtype=float)
        tc = t - t.mean()
        return ((v - v.mean(axis=1, keepdims=True)) @ tc) / (tc @ tc + 1e-12)
    if name == "entropy":
        p = np.abs(v) + 1e-12
        p /= p.sum(axis=1, keepdims=True)
        return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])
    raise ValueError(f"unknown statistic {name!r}")


def _eval(expr: Expr, x: np.ndarray) -> np.ndarray:
    if expr.op == "view":
        return _view(expr, x)
    if expr.op == "stat":
        return _stat(expr.params[0], _eval(expr.args[0], x))
    if expr.op in ("difference", "ratio"):
        a, b = (_eval(e, x) for e in expr.args)
        if a.ndim == 2 or b.ndim == 2:
            raise ValueError("relations require scalar statistics")
        if expr.op == "difference":
            return a - b
        return a / (np.abs(b) + 1e-12)
    raise ValueError(f"unknown expression op {expr.op!r}")


def evaluate_expr(expr: Expr, windows: np.ndarray) -> np.ndarray:
    values = np.asarray(_eval(expr, _windows(windows)), dtype=float)
    if values.ndim != 1:
        raise ValueError("expression must reduce each window to one scalar")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _scalar(rep: str, channel: int, scale: str, stat: str) -> Expr:
    return Expr("stat", (Expr("view", params=(rep, channel, scale)),), (stat,))


def enumerate_candidates(channels: int, budget: int, seed: int = 0) -> list[Expr]:
    """Enumerate a deterministic prefix of the fixed v1 recipe vocabulary."""
    if channels < 1 or budget < 0:
        raise ValueError("channels must be positive and budget non-negative")
    reps = ("raw", "diff", "envelope", "spectrum", "phase", "lag")
    scales = ("short", "medium", "full")
    stats = ("meanabs", "std", "iqr", "slope", "entropy")
    unary: list[Expr] = []
    for c in range(channels):
        for rep in reps:
            for scale in scales:
                for stat in stats:
                    unary.append(_scalar(rep, c, scale, stat))
    mixed: list[Expr] = []
    if channels > 1:
        for a in range(channels):
            for b in range(a + 1, channels):
                for rep in reps:
                    for scale in scales:
                        left = _scalar(rep, a, scale, "std")
                        right = _scalar(rep, b, scale, "std")
                        mixed.extend((Expr("difference", (left, right)),
                                      Expr("ratio", (left, right))))
    # Expression ordering is canonical and independent of hash randomization.
    unary.sort(key=lambda e: (e.cost(), e.canonical()))
    mixed.sort(key=lambda e: (e.cost(), e.canonical()))
    mixed_budget = min(len(mixed), max(1, budget // 3)) if mixed else 0
    unary_budget = max(0, budget - mixed_budget)
    out = unary[:unary_budget] + mixed[:mixed_budget]
    out.sort(key=lambda e: (e.cost(), e.canonical()))
    return out[:budget]


def mechanism_null(windows: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """Create a structure-preserving null without changing the sensor shape."""
    x = _windows(windows).copy()
    rng = np.random.default_rng(seed)
    if kind == "phase":
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            for c in range(x.shape[1]):
                spec = np.fft.rfft(x[i, c])
                phase = rng.uniform(-np.pi, np.pi, size=spec.shape)
                phase[[0, -1]] = 0.0
                out[i, c] = np.fft.irfft(np.abs(spec) * np.exp(1j * phase), n=x.shape[2])
        return out
    if kind == "shuffle":
        # Destroy temporal ordering while preserving each window's amplitude
        # distribution. Time reversal cannot serve here: reversing a series
        # negates and reverses its differences, leaving every order-invariant
        # statistic of a difference view exactly unchanged.
        out = x.copy()
        for i in range(x.shape[0]):
            for c in range(x.shape[1]):
                out[i, c] = x[i, c, rng.permutation(x.shape[2])]
        return out
    if kind == "channel":
        if x.shape[1] < 2:
            return x
        out = x.copy()
        for i in range(x.shape[0]):
            out[i] = x[i, rng.permutation(x.shape[1])]
        return out
    raise ValueError("null kind must be phase, shuffle, or channel")


def _reps_used(expr: Expr) -> set:
    reps = {expr.params[0]} if expr.op == "view" else set()
    for a in expr.args:
        reps |= _reps_used(a)
    return reps


def _null_kind_for(expr: Expr) -> str:
    """Pick the mechanism null that actually destroys the expression's structure.

    A phase-randomized null preserves the power spectrum, so it is only a valid
    control for phase features. Temporal features (diff/lag) need a sample-shuffle
    null; cross-channel relations need a channel-shuffle null.

    No null in this vocabulary can perturb a statistic that depends only on a
    view's value distribution (``std(raw)``, any summary of ``spectrum``) — the
    scorer detects that case instead of pretending a control ran.
    """
    if expr.op in ("difference", "ratio"):
        return "channel"
    reps = _reps_used(expr)
    if reps & {"diff", "lag"}:
        return "shuffle"
    return "phase"


def _gid(u):
    return u.item() if isinstance(u, np.generic) else u


def _group_labels(labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return unique groups and their constant label, rejecting mixed groups.

    The permutation null and group holdout both assume each independent unit
    carries one label; a group with more than one label silently invalidates them.
    """
    y = np.asarray(labels)
    g = np.asarray(groups)
    uniq = np.unique(g)
    mapping: dict = {}
    for u in uniq:
        vals = np.unique(y[g == u])
        if len(vals) != 1:
            raise ValueError(f"group {u!r} has multiple labels; each group must carry a single label")
        mapping[u.item() if isinstance(u, np.generic) else u] = vals[0]
    return uniq, mapping


def _split_groups(groups: np.ndarray, labels: np.ndarray, seed: int,
                  replication_frac: float = 0.4, min_side: int = 3):
    """Stratified split of whole groups into discovery and replication sets.

    Selection happens only on the discovery groups; the frozen top recipe is
    confirmed on the untouched replication groups. Both classes are kept on each
    side; if that is impossible the replication set is marked unusable.
    """
    uniq, glabel = _group_labels(labels, groups)
    rng = np.random.default_rng(seed)
    disc: list = []
    repl: list = []
    for cls in sorted({glabel[_gid(u)] for u in uniq}):
        members = np.array([u for u in uniq if glabel[_gid(u)] == cls], dtype=uniq.dtype)
        members = members[rng.permutation(len(members))]
        n = len(members)
        k = int(round(n * (1 - replication_frac)))
        k = min(max(k, 1), n - 1) if n >= 2 else n
        disc.extend(members[:k])
        repl.extend(members[k:])
    disc_arr = np.array(sorted(disc, key=_gid), dtype=uniq.dtype)
    repl_arr = np.array(sorted(repl, key=_gid), dtype=uniq.dtype)
    repl_ok = (len(disc_arr) >= min_side and len(repl_arr) >= min_side
               and len({glabel[_gid(u)] for u in disc_arr}) >= 2
               and len({glabel[_gid(u)] for u in repl_arr}) >= 2)
    return disc_arr, repl_arr, repl_ok


def _group_accuracy(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> tuple[float, dict]:
    """Group-held-out accuracy; model imports stay optional for edge installs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    f = np.asarray(features, dtype=float).reshape(-1, 1)
    y = np.asarray(labels)
    g = np.asarray(groups)
    scores: dict = {}
    for held in np.unique(g):
        train = g != held
        test = ~train
        if len(np.unique(y[train])) < 2 or not np.any(test):
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
        model.fit(f[train], y[train])
        scores[str(held)] = float(np.mean(model.predict(f[test]) == y[test]))
    return (float(np.mean(list(scores.values()))) if scores else 0.0), scores


def _bootstrap_ci(values: list[float], seed: int, n: int = 1000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    draws = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def perm_resolution(group_labels, permutations: int) -> dict:
    """Smallest permutation p this design can report.

    A group-label permutation test draws from a *discrete* space: with the
    group labels held fixed there are only ``n! / prod(count!)`` distinct
    assignments, and every relabelling of a perfect separator (identity, and
    for binary labels the swap) reproduces the observed accuracy. Sampling
    ``permutations`` draws therefore expects ``permutations * symmetries /
    distinct`` hits before a single genuine one, so

        expected_floor = (1 + permutations * symmetries / distinct)
                         / (permutations + 1)

    is the expected smallest reportable p — attained exactly on small splits
    (6 groups, 50 draws -> 0.118). It is an expectation, not a hard bound: a
    lucky sample can fall under it, and the only hard bound is
    ``sampling_floor = 1/(permutations+1)``. Group count moves the expected
    floor; more draws alone cannot push it under ``symmetries / distinct``.
    """
    from collections import Counter
    from math import factorial

    counts = list(Counter(list(group_labels)).values())
    n = sum(counts)
    distinct = factorial(n)
    for c in counts:
        distinct //= factorial(c)
    symmetries = factorial(len(counts))
    sampling_floor = 1.0 / (permutations + 1)
    expected = (1 + permutations * min(symmetries / distinct, 1.0)) / (permutations + 1)
    return {"groups": n, "distinct_assignments": int(distinct),
            "label_symmetries": int(symmetries),
            "sampling_floor": float(sampling_floor),
            "expected_floor": float(expected),
            "permutations": int(permutations)}


def score_candidate(expr: Expr, windows: np.ndarray, labels: np.ndarray,
                    groups: np.ndarray, baseline: Expr | None = None,
                    null_kind: str = "phase", seed: int = 0,
                    permutations: int = 200) -> dict:
    """Score one recipe with group holdout, group-label permutation, and a mechanism null."""
    x = _windows(windows)
    y = np.asarray(labels)
    g = np.asarray(groups)
    if len(x) != len(y) or len(y) != len(g):
        raise ValueError("windows, labels, and groups must have equal length")
    values = evaluate_expr(expr, x)
    baseline_values = evaluate_expr(baseline, x) if baseline else np.zeros(len(x))
    accuracy, per_group = _group_accuracy(values, y, g)
    base_accuracy, base_per_group = _group_accuracy(baseline_values, y, g)
    common = sorted(set(per_group) & set(base_per_group))
    deltas = [per_group[k] - base_per_group[k] for k in common]
    rng = np.random.default_rng(seed)
    unique_groups, group_label = _group_labels(y, g)
    permuted = []
    for _ in range(permutations):
        shuffled = rng.permutation([group_label[_gid(group)] for group in unique_groups])
        yp = np.array([shuffled[np.flatnonzero(unique_groups == group)[0]] for group in g])
        permuted.append(_group_accuracy(values, yp, g)[0])
    null_values = evaluate_expr(expr, mechanism_null(x, null_kind, seed + 1))
    null_accuracy, null_per_group = _group_accuracy(null_values, y, g)
    # A null that leaves the feature values untouched is not a control. Record
    # that instead of reporting a null accuracy that merely restates the result.
    shift = float(np.max(np.abs(null_values - values))) if len(values) else 0.0
    informative = shift > 1e-9 * (float(np.max(np.abs(values))) + 1e-12)
    p = (1 + sum(a >= accuracy for a in permuted)) / (len(permuted) + 1)
    resolution = perm_resolution(
        [group_label[_gid(group)] for group in unique_groups], permutations)
    return {
        "group_perm_p_floor": resolution["expected_floor"],
        "perm_resolution": resolution,
        "expr": expr.canonical(), "digest": expr.digest(), "cost": expr.cost(),
        "accuracy": accuracy, "baseline_accuracy": base_accuracy,
        "delta": float(np.mean(deltas)) if deltas else 0.0,
        "delta_ci95": _bootstrap_ci(deltas, seed),
        "group_perm_p": float(p), "null_accuracy": null_accuracy,
        "null_delta": null_accuracy - base_accuracy, "groups": len(common),
        "per_group": per_group, "null_per_group": null_per_group,
        "null_kind": null_kind, "null_informative": bool(informative),
        "null_reason": ("mechanism null shifted the feature values"
                        if informative else
                        f"{null_kind} null leaves this statistic unchanged; "
                        "no mechanism control was exercised"),
    }


def evidence_gate(record: dict, replicated: bool = False) -> str:
    lo, _ = record["delta_ci95"]
    if record["groups"] < 3 or not np.isfinite(record["accuracy"]):
        return "inconclusive"
    if lo <= 0.0:
        return "null"
    if record["group_perm_p"] >= 0.05:
        # A p that never *could* have cleared the threshold is a resolution
        # limit of the design, not evidence of absence. Say so.
        if record.get("group_perm_p_floor", 0.0) >= 0.05:
            return "inconclusive"
        return "null"
    if not record.get("null_informative", True):
        # The spec makes "the mechanism null removes the effect" a hard
        # condition for `supported`. An unexercised control cannot satisfy it.
        return "candidate"
    if record["null_delta"] > record["delta"] * 0.5:
        return "candidate"
    return "supported" if replicated else "candidate"


def compose(windows: np.ndarray, labels: np.ndarray, groups: np.ndarray,
            budget: int = 32, seed: int = 0, permutations: int = 200) -> dict:
    """Run bounded search and return a portable, auditable research receipt.

    Selection is confined to a discovery split; the single frozen top recipe is
    then re-scored on untouched replication groups. The discovery delta is a
    selection statistic — never confirmatory — and ``supported`` is reachable
    only when the replication split independently confirms the effect.
    """
    started = time.perf_counter()
    x = _windows(windows)
    y = np.asarray(labels)
    g = np.asarray(groups)
    if len(x) != len(y) or len(y) != len(g):
        raise ValueError("windows, labels, and groups must have equal length")
    _, glabels = _group_labels(y, g)  # reject mixed-label groups before scoring
    candidates = enumerate_candidates(x.shape[1], budget, seed)
    baseline = _scalar("raw", 0, "full", "std")
    disc_ids, repl_ids, repl_ok = _split_groups(g, y, seed)
    disc_mask = np.isin(g, disc_ids)

    # Phase A — select candidates on discovery groups only.
    disc_started = time.perf_counter()
    records = []
    for i, e in enumerate(candidates):
        rec = score_candidate(e, x[disc_mask], y[disc_mask], g[disc_mask],
                              baseline=baseline, null_kind=_null_kind_for(e),
                              seed=seed + i, permutations=permutations)
        rec["verdict"] = evidence_gate(rec)  # discovery verdict — not confirmatory
        records.append((e, rec))
    discovery_runtime_s = time.perf_counter() - disc_started
    records.sort(key=lambda er: (-er[1]["delta"], er[1]["cost"], er[1]["digest"]))
    ordered = [rec for _, rec in records]
    best_expr, best = records[0] if records else (None, None)

    # Phase B — confirm the frozen top recipe on untouched replication groups.
    # Early stop: a discovery-null/inconclusive recipe is not worth replicating.
    repl_started = time.perf_counter()
    replication = None
    replicated = False
    if best is not None:
        if repl_ok and best["verdict"] not in {"null", "inconclusive"}:
            repl_mask = np.isin(g, repl_ids)
            replication = score_candidate(
                best_expr, x[repl_mask], y[repl_mask], g[repl_mask],
                baseline=baseline, null_kind=_null_kind_for(best_expr),
                seed=seed + 1_000_000, permutations=permutations)
            replicated = (replication["group_perm_p"] < 0.05
                          and replication["delta_ci95"][0] > 0.0
                          and replication["null_delta"] <= replication["delta"] * 0.5)
        else:
            replication = {"available": False,
                           "reason": ("insufficient replication groups" if not repl_ok
                                      else f"discovery verdict {best['verdict']}; not replicated")}
        best = dict(best)
        best["replication"] = replication
        best["replicated"] = replicated
        best["verdict"] = evidence_gate(best, replicated=replicated)
        ordered[0] = best
    replication_runtime_s = time.perf_counter() - repl_started

    return {
        "grammar_version": GRAMMAR_VERSION,
        "selection_scope": ("discovery groups select; replication groups confirm; "
                            "a candidate delta is a selection statistic, not proof"),
        "seed": seed, "budget": budget, "n_windows": len(x),
        "n_channels": x.shape[1], "n_groups": len(np.unique(g)),
        "discovery_groups": [_gid(u) for u in disc_ids],
        "replication_groups": [_gid(u) for u in repl_ids],
        "resolution": {
            "discovery": perm_resolution(
                [glabels[_gid(u)] for u in disc_ids], permutations),
            "replication": perm_resolution(
                [glabels[_gid(u)] for u in repl_ids], permutations),
            "note": ("expected_floor is the smallest p each split can report; "
                     "a verdict that fails a threshold below its floor is a "
                     "resolution limit, not evidence of absence"),
        },
        "baseline": baseline.canonical(), "candidates": ordered,
        "best": best,
        "discovery_runtime_s": discovery_runtime_s,
        "replication_runtime_s": replication_runtime_s,
        "runtime_s": time.perf_counter() - started,
    }


def write_receipt(receipt: dict, path: str) -> None:
    """Write JSON only; a recipe is never silently promoted to production."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")
