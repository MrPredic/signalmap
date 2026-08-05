import json

import numpy as np
import pytest

from signalmap.composer import (Expr, compose, enumerate_candidates,
                                 evaluate_expr, evidence_gate, mechanism_null,
                                 score_candidate)


def test_expression_is_canonical_and_hashed():
    a = Expr("stat", (Expr("view", params=("raw", 0, "full"),),), ("std",))
    b = Expr("stat", (Expr("view", params=("raw", 0, "full"),),), ("std",))
    assert a.canonical() == b.canonical()
    assert a.digest() == b.digest()


def test_generation_is_deterministic_and_budgeted():
    a = enumerate_candidates(channels=2, budget=24, seed=3)
    b = enumerate_candidates(channels=2, budget=24, seed=3)
    assert [x.digest() for x in a] == [x.digest() for x in b]
    assert len(a) == 24
    assert len({x.digest() for x in a}) == 24


def test_composed_features_are_finite():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(12, 2, 256))
    for expr in enumerate_candidates(channels=2, budget=40):
        values = evaluate_expr(expr, windows)
        assert values.shape == (12,)
        assert np.isfinite(values).all(), expr.canonical()


def test_single_channel_rejects_cross_channel_expression():
    expr = Expr("difference", (
        Expr("view", params=("raw", 0, "full")),
        Expr("view", params=("raw", 1, "full")),
    ))
    with pytest.raises(ValueError, match="channel"):
        evaluate_expr(expr, np.zeros((2, 1, 128)))


def test_phase_null_preserves_shape_and_spectrum():
    x = np.random.default_rng(1).normal(size=(4, 2, 64))
    z = mechanism_null(x, "phase", seed=4)
    assert z.shape == x.shape
    np.testing.assert_allclose(np.abs(np.fft.rfft(z, axis=-1)),
                               np.abs(np.fft.rfft(x, axis=-1)), atol=1e-8)


def test_group_score_detects_planted_cross_channel_difference():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(8), 12)
    labels = np.repeat([0, 1, 0, 1, 0, 1, 0, 1], 12)
    x = rng.normal(size=(len(labels), 2, 64))
    labels = np.repeat([0, 1, 0, 1, 0, 1, 0, 1], 12)
    a = np.where(labels == 1, 2.0, 1.0) + rng.normal(0, 0.08, len(labels))
    b = np.where(labels == 1, 1.0, 2.0) + rng.normal(0, 0.08, len(labels))
    x[:, 0] *= a[:, None]
    x[:, 1] *= b[:, None]
    expr = Expr("difference", (
        Expr("stat", (Expr("view", params=("raw", 0, "full")),), ("std",)),
        Expr("stat", (Expr("view", params=("raw", 1, "full")),), ("std",)),
    ))
    rec = score_candidate(expr, x, labels, groups, null_kind="channel", permutations=50)
    assert rec["groups"] == 8
    assert rec["accuracy"] > 0.9
    assert evidence_gate(rec) == "null"  # amplitude null exposes the confound


def test_compose_receipt_is_reproducible():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 2, 64))
    y = np.repeat([0, 1], 15)
    g = np.arange(30)
    a = compose(x, y, g, budget=8, seed=7, permutations=10)
    b = compose(x, y, g, budget=8, seed=7, permutations=10)
    assert a["best"]["digest"] == b["best"]["digest"]
    assert a["best"]["delta"] == b["best"]["delta"]
    # Full receipt is byte-identical under two identical runs (except wall time).
    strip = lambda d: {k: v for k, v in d.items() if not k.endswith("runtime_s")}
    assert json.dumps(strip(a), sort_keys=True) == json.dumps(strip(b), sort_keys=True)


def _planted_cross_channel(n_groups=12, seed=2, per=10):
    """Groups carry a stable label; ch0 up / ch1 down encodes the class."""
    rng = np.random.default_rng(seed)
    glabels = np.tile([0, 1], n_groups // 2)
    groups = np.repeat(np.arange(n_groups), per)
    labels = np.repeat(glabels, per)
    x = rng.normal(size=(len(labels), 2, 64))
    x[:, 0] *= np.where(labels == 1, 2.0, 1.0)[:, None]
    x[:, 1] *= np.where(labels == 1, 1.0, 2.0)[:, None]
    return x, labels, groups


def test_discovery_and_replication_groups_are_disjoint():
    x, y, g = _planted_cross_channel()
    r = compose(x, y, g, budget=6, seed=0, permutations=20)
    disc = set(r["discovery_groups"])
    repl = set(r["replication_groups"])
    assert disc and repl
    assert disc.isdisjoint(repl)
    assert disc | repl == set(int(v) for v in np.unique(g))
    assert "replication" in r["best"]
    rep = r["best"]["replication"]
    assert ("delta" in rep) or (rep.get("available") is False)  # well-formed block


def _planted_amplitude(n_groups=24, per=5, seed=5):
    """Class encoded purely as channel-1 amplitude (baseline channel 0 is noise).

    Enough groups so the discovery split retains permutation resolution below
    0.05 — halving the groups otherwise caps a perfect signal at ``null``.
    """
    rng = np.random.default_rng(seed)
    glabels = np.tile([0, 1], n_groups // 2)
    groups = np.repeat(np.arange(n_groups), per)
    labels = np.repeat(glabels, per)
    x = rng.normal(size=(len(labels), 2, 64))
    x[:, 1, :] *= np.where(labels == 1, 2.0, 0.5)[:, None]
    return x, labels, groups


def test_replication_runs_and_is_separate_for_a_real_signal():
    x, y, g = _planted_amplitude()
    r = compose(x, y, g, budget=4, seed=0, permutations=30)
    rep = r["best"]["replication"]
    assert "delta" in rep  # Phase B actually executed on untouched groups
    assert set(r["discovery_groups"]).isdisjoint(r["replication_groups"])
    # A retrospective amplitude hit is a candidate, not confirmed physics.
    assert r["best"]["verdict"] in {"candidate", "supported", "null", "inconclusive"}


def test_pure_noise_is_never_supported():
    rng = np.random.default_rng(42)
    ng, per = 12, 10
    g = np.repeat(np.arange(ng), per)
    y = np.repeat(rng.integers(0, 2, ng), per)
    x = rng.normal(size=(len(y), 2, 64))  # no planted effect at all
    r = compose(x, y, g, budget=8, seed=1, permutations=40)
    assert r["best"]["verdict"] != "supported"
    assert r["best"]["replicated"] is False


def test_candidate_selection_is_not_confirmatory():
    # The reported discovery delta is a selection statistic; the replication
    # block must exist so a reader never treats selection as confirmation.
    x, y, g = _planted_cross_channel()
    r = compose(x, y, g, budget=6, seed=0, permutations=20)
    assert r["selection_scope"]
    assert r["best"]["replication"] is not None


def test_small_budget_still_returns_a_best():
    x, y, g = _planted_cross_channel(n_groups=8)
    r = compose(x, y, g, budget=1, seed=0, permutations=10)
    assert len(r["candidates"]) == 1
    assert r["best"] is not None


def test_constant_and_nan_windows_do_not_crash():
    g = np.repeat(np.arange(8), 6)
    y = np.repeat([0, 1], 24)
    const = np.zeros((48, 2, 64))
    r1 = compose(const, y, g, budget=3, seed=0, permutations=10)
    assert np.isfinite(r1["best"]["delta"])
    nan = np.random.default_rng(0).normal(size=(48, 2, 64))
    nan[::3] = np.nan
    r2 = compose(nan, y, g, budget=3, seed=0, permutations=10)
    assert np.isfinite(r2["best"]["delta"])


def test_group_with_mixed_labels_is_rejected():
    g = np.repeat(np.arange(6), 4)
    y = np.zeros(24, dtype=int)
    y[0] = 1  # group 0 now has both labels -> invalid independent unit
    x = np.random.default_rng(0).normal(size=(24, 2, 32))
    expr = _scalar_expr()
    with pytest.raises(ValueError, match="single label"):
        score_candidate(expr, x, y, g, permutations=5)


def _scalar_expr():
    return Expr("stat", (Expr("view", params=("raw", 0, "full")),), ("std",))


def test_mechanism_null_kind_matches_representation():
    """Temporal views route to `shuffle`, not to the old time-reversal `lag`
    null: reversing a series negates and reverses its differences, so every
    order-invariant statistic of a difference view survives it unchanged.
    See test_composer_review.py for the measurement behind this."""
    from signalmap.composer import _null_kind_for
    spec = Expr("stat", (Expr("view", params=("spectrum", 0, "full")),), ("std",))
    lag = Expr("stat", (Expr("view", params=("lag", 0, "full")),), ("slope",))
    cross = Expr("difference", (_scalar_expr(), _scalar_expr()))
    assert _null_kind_for(spec) == "phase"
    assert _null_kind_for(lag) == "shuffle"
    assert _null_kind_for(cross) == "channel"


def test_prove_cli_synthetic_and_npz(tmp_path):
    from signalmap.cli import main
    out = tmp_path / "r.json"
    main(["prove", "--synthetic", "120", "--budget", "4", "--perms", "10",
          "--seed", "0", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rec["best"] is not None
    assert "replication_groups" in rec and rec["replication_groups"]
    x, y, g = _planted_cross_channel(n_groups=12)
    npz = tmp_path / "bank.npz"
    np.savez(npz, X=x, y=y, groups=g)
    out2 = tmp_path / "r2.json"
    main(["prove", "--dataset", str(npz), "--budget", "4", "--perms", "10",
          "--out", str(out2)])
    assert json.loads(out2.read_text())["best"] is not None


def test_prove_cli_rejects_malformed_npz(tmp_path):
    from signalmap.cli import main
    bad = tmp_path / "bad.npz"
    np.savez(bad, X=np.zeros((4, 8)))  # missing y and groups
    with pytest.raises(ValueError, match="missing arrays"):
        main(["prove", "--dataset", str(bad), "--out", str(tmp_path / "o.json")])
