"""Review findings against the 2026-07-23 composer design spec.

Two defects, both found by measurement rather than by reading:

1. The spec's representation axis lists "difference ... and fixed lag views" as
   distinct, and requires candidates to be "deduplicated before scoring".
   ``lag`` was implemented as ``v[:,1:] - v[:,:-1]`` — byte-identical to
   ``diff`` — so 36 of 216 enumerated candidates were exact semantic duplicates
   burning search budget, and the fixed-lag view did not exist.

2. The spec makes "the relevant mechanism null removes the effect" a hard gate
   for ``supported``. For 16 of 30 representation/statistic combinations the
   chosen null provably cannot change the feature value at all (time reversal
   leaves every order-invariant statistic of a difference series unchanged;
   phase randomization preserves the power spectrum, hence ``std(raw)`` and
   every statistic of ``spectrum``). The gate was therefore inoperative and
   silently unreachable — it could only ever cap a candidate, never confirm it,
   and the receipt gave no sign that no control had been exercised.
"""
import numpy as np
import pytest

from signalmap.composer import (
    LAG_SAMPLES,
    Expr,
    _null_kind_for,
    _scalar,
    _view,
    _windows,
    compose,
    enumerate_candidates,
    evaluate_expr,
    evidence_gate,
    mechanism_null,
    score_candidate,
)


@pytest.fixture
def x():
    return _windows(np.random.default_rng(0).standard_normal((8, 2, 256)))


# --- defect 1: lag was a duplicate of diff --------------------------------

def test_lag_view_is_a_fixed_lag_difference_not_the_lag_one_diff(x):
    lag = _view(Expr("view", params=("lag", 0, "full")), x)
    diff = _view(Expr("view", params=("diff", 0, "full")), x)
    assert not np.array_equal(lag, diff)
    raw = x[:, 0, :]
    np.testing.assert_allclose(lag, raw[:, LAG_SAMPLES:] - raw[:, :-LAG_SAMPLES])


def test_enumerated_candidates_are_semantically_deduplicated(x):
    """Spec: candidates are 'canonicalized, hashed, costed, and deduplicated
    before scoring' — two candidates must not compute the same thing."""
    seen: dict = {}
    for e in enumerate_candidates(2, 10 ** 6):
        key = tuple(np.round(evaluate_expr(e, x), 12))
        assert key not in seen, f"{e.canonical()} duplicates {seen[key].canonical()}"
        seen[key] = e


# --- defect 2: mechanism nulls that cannot perturb the statistic ----------

TEMPORAL = [(rep, stat) for rep in ("diff", "lag")
            for stat in ("meanabs", "std", "iqr", "slope", "entropy")]


@pytest.mark.parametrize("rep,stat", TEMPORAL)
def test_temporal_null_actually_perturbs_the_statistic(x, rep, stat):
    """Time reversal leaves order-invariant statistics of a difference series
    exactly unchanged; the null routed for temporal views must not."""
    e = _scalar(rep, 0, "full", stat)
    values = evaluate_expr(e, x)
    nulled = evaluate_expr(e, mechanism_null(x, _null_kind_for(e), 1))
    rel = np.max(np.abs(nulled - values)) / (np.max(np.abs(values)) + 1e-12)
    assert rel > 1e-6, f"null for {rep}/{stat} is a no-op (rel change {rel:.1e})"


def test_score_candidate_flags_an_uninformative_null(x):
    """``std(raw)`` is fixed by the power spectrum, so a phase-randomized null
    cannot move it. That must be recorded, not silently absorbed."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    g = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    rec = score_candidate(_scalar("raw", 0, "full", "std"), x, y, g,
                          null_kind="phase", seed=0, permutations=10)
    assert rec["null_informative"] is False
    assert "null" in rec["null_reason"].lower()


def test_score_candidate_marks_a_working_null_informative(x):
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    g = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    rec = score_candidate(_scalar("phase", 0, "full", "slope"), x, y, g,
                          null_kind="phase", seed=0, permutations=10)
    assert rec["null_informative"] is True


def test_evidence_gate_never_reports_supported_on_an_uninformative_null():
    """The spec makes the mechanism null a hard condition for `supported`.
    If no control was actually exercised, the verdict must stay `candidate`."""
    passing = {"delta_ci95": (0.2, 0.4), "groups": 6, "accuracy": 0.9,
               "group_perm_p": 0.001, "delta": 0.3, "null_delta": -0.1,
               "null_informative": True}
    assert evidence_gate(passing, replicated=True) == "supported"
    vacuous = dict(passing, null_informative=False)
    assert evidence_gate(vacuous, replicated=True) == "candidate"


def test_compose_receipt_records_grammar_version_and_null_informativeness():
    """Provenance: the grammar changed, so receipts must not be confusable with
    pre-fix ones, and every candidate carries its null-control status."""
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(12), 4)
    labels = np.repeat(np.arange(12) % 2, 4)
    windows = rng.normal(size=(48, 2, 128))
    windows[:, 1, :] *= np.where(labels == 1, 2.0, 0.5)[:, None]
    receipt = compose(windows, labels, groups, budget=6, seed=0, permutations=20)
    assert receipt["grammar_version"] == "composer-v2"
    assert all("null_informative" in c for c in receipt["candidates"])


# --- defect 3: routed-family filter admitted excluded families ------------

def test_filter_programs_requires_every_family_a_program_uses():
    """`filter_programs` classified a program by the first matching branch, so
    an envelope-transformed spectral program was admitted whenever `spectral`
    was compatible — even on a source where `envelope` had been routed out for
    a missing hard precondition (windows < 128 samples). A program must need
    ALL the families it touches, not just one of them."""
    from signalmap.distill import enumerate_programs
    from signalmap.qualification import filter_programs

    programs = enumerate_programs()
    kept = filter_programs(programs, {"time_domain", "spectral"})
    leaked = [p for p in kept if p.t1 == "env" or p.t2 == "env"]
    assert not leaked, (
        f"{len(leaked)} envelope programs admitted with envelope incompatible, "
        f"e.g. {leaked[0].name if leaked else ''}")
    # the spectral+envelope combination is admitted once BOTH are routed
    both = filter_programs(programs, {"time_domain", "spectral", "envelope"})
    assert any(p.t1 == "env" or p.t2 == "env" for p in both)
    assert len(both) > len(kept)
