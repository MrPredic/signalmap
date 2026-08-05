"""The prove demo capped at `candidate`, and the binding constraint was
never the effect size.

Measured on 2026-08-05 (sweep over the planted synthetic effect, delta ≈ 0.95–0.99):

    groups  perms  disc p / floor   repl p / floor   verdict
    16      50     0.0392 / 0.0274  0.1176 / 0.1176  candidate
    16      200    0.0249 / 0.0129  0.0896 / 0.1045  candidate
    20      50     0.0196 / 0.0217  0.0588 / 0.0476  candidate
    20      200    0.0050 / 0.0071  0.0398 / 0.0334  supported
    24      200    0.0050 / 0.0056  0.0199 / 0.0129  supported

A group-label permutation test draws from a discrete space: with G groups split
k/(G-k) there are only C(G,k) distinct assignments, and the identity plus the
label swap both reproduce the observed accuracy. So in expectation
perms*2/C draws tie the observed result before a single genuine one — with 6
replication groups that puts the expected smallest p at 0.118, and no effect,
however perfect, clears 0.05 there. The floor is an expectation, not a hard
bound: a lucky sample can land under it (measured: discovery p 0.0050 vs floor
0.0071), and the only hard bound is the sampling floor 1/(perms+1).

So the receipt records that floor, and a p that merely fails to clear a floor
above the threshold is `inconclusive` (cannot resolve), never `null` (no
evidence). Absence of evidence is not evidence of absence.
"""
import numpy as np
import pytest

from signalmap.composer import (
    Expr,
    compose,
    evidence_gate,
    perm_resolution,
    score_candidate,
)


def _planted(n_groups=6, per=6, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * per
    groups = np.repeat(np.arange(n_groups), per)
    labels = np.repeat(np.arange(n_groups) % 2, per)
    x = rng.normal(size=(n, 2, 128))
    x[:, 1, :] *= np.where(labels == 1, 2.0, 0.5)[:, None]
    return x, labels, groups


def _diff_expr():
    return Expr("difference", (
        Expr("stat", (Expr("view", params=("diff", 0, "full")),), ("std",)),
        Expr("stat", (Expr("view", params=("diff", 1, "full")),), ("std",)),
    ))


# --- the floor itself --------------------------------------------------

def test_floor_counts_the_discrete_label_space():
    r = perm_resolution([0, 1] * 3, permutations=50)      # 6 groups, 3/3
    assert r["distinct_assignments"] == 20                 # C(6,3)
    assert r["label_symmetries"] == 2                      # identity + swap
    assert r["expected_floor"] == pytest.approx(6 / 51)    # 0.1176
    assert r["sampling_floor"] == pytest.approx(1 / 51)


def test_more_groups_lower_the_floor_more_than_more_permutations():
    """Which knob binds: 6 groups stay unresolvable however many draws."""
    many_draws = perm_resolution([0, 1] * 3, permutations=2000)
    more_groups = perm_resolution([0, 1] * 5, permutations=50)
    assert many_draws["expected_floor"] > 0.05
    assert more_groups["expected_floor"] < 0.05


def test_six_groups_cannot_clear_the_gate_however_clean_the_effect():
    """A perfect effect on 6 groups still cannot get near 0.05 — the sweep's
    replication split attained the floor exactly (0.118); here chance
    separations push the observed p higher still."""
    x, y, g = _planted(n_groups=6)
    rec = score_candidate(_diff_expr(), x, y, g, null_kind="channel", seed=0,
                          permutations=50)
    assert rec["accuracy"] == 1.0
    assert rec["group_perm_p_floor"] == pytest.approx(6 / 51)
    assert rec["group_perm_p"] >= rec["perm_resolution"]["sampling_floor"]
    assert rec["group_perm_p"] >= 0.05, "6 groups can never clear the gate"
    assert evidence_gate(rec) == "inconclusive"


# --- what the gate is allowed to say -----------------------------------

def _rec(p, floor, lo=0.4):
    return {"groups": 8, "accuracy": 1.0, "delta": 0.9, "delta_ci95": (lo, 1.0),
            "group_perm_p": p, "group_perm_p_floor": floor, "null_delta": 0.0,
            "null_informative": True}


def test_resolution_limited_result_is_inconclusive_not_null():
    assert evidence_gate(_rec(0.118, 0.118)) == "inconclusive"


def test_a_real_miss_is_still_null():
    """Same p, but the design could have resolved it — that is evidence."""
    assert evidence_gate(_rec(0.118, 0.004)) == "null"


def test_negative_ci_is_null_even_when_resolution_limited():
    assert evidence_gate(_rec(0.118, 0.118, lo=-0.1)) == "null"


def test_gate_without_a_recorded_floor_behaves_as_before():
    rec = _rec(0.118, 0.0)
    del rec["group_perm_p_floor"]
    assert evidence_gate(rec) == "null"


# --- receipt ------------------------------------------------------------

def test_every_candidate_and_the_split_carry_their_resolution():
    x, y, g = _planted(n_groups=8, per=4)
    r = compose(x, y, g, budget=4, seed=0, permutations=20)
    assert all("group_perm_p_floor" in c for c in r["candidates"])
    res = r["resolution"]
    assert res["discovery"]["groups"] == len(r["discovery_groups"])
    assert res["replication"]["groups"] == len(r["replication_groups"])
    assert res["discovery"]["expected_floor"] > 0


# --- the demo defaults --------------------------------------------------

def test_prove_demo_defaults_can_actually_reach_supported():
    """The spec criterion (>=1 non-trivial candidate) is unreachable unless
    BOTH splits can resolve p<0.05. Regression guard on the CLI defaults."""
    from signalmap.cli import build_parser, synthetic_prove_groups

    args = build_parser().parse_args(["prove"])
    n_groups = synthetic_prove_groups(args.synthetic)
    per_class = n_groups // 2
    disc = 2 * max(round(per_class * 0.6), 1)
    repl = n_groups - disc
    for name, count in (("discovery", disc), ("replication", repl)):
        floor = perm_resolution([0, 1] * (count // 2),
                                permutations=args.perms)["expected_floor"]
        assert floor < 0.05, (
            f"{name} split of {count} groups cannot resolve p<0.05 "
            f"(floor {floor:.4f}) with --perms {args.perms}")
