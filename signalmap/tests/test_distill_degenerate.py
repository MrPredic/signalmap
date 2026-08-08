"""A feature with no healthy variation must not hijack the score.

`window()` z-normalises every window, so `std(id(id(x)))` is exactly 1.0 for
every window ever produced — and the lean-base spec that `distill` emits
contains it. Its healthy MAD is therefore 0, and the guard in
`DistilledDetector.fit` used to clamp that to 1e-12 and divide by it, turning
last-bit rounding into a term twelve orders of magnitude above every real
feature.

Measured consequence before the fix: on the MAFAULDA bank two programs sat on
the guard floor (`std` and, data-dependently, `peakcv`) and the self-calibrated
threshold came out at 9.67e8, with the alarm firing on 73% of healthy windows
and only 45% of faulty ones.

The fix is NOT to drop such features. A discrete feature that is genuinely
constant on healthy data must still alert on a real shift, and
`test_detector_survives_zero_mad_and_keeps_sensitivity` pins exactly that. So
the guard floor is made relative instead — one part per billion of the
feature's own magnitude — which leaves float accumulation noise around
z ~ 1e-3 while any physically meaningful change stays orders of magnitude
above it. Features that land on the floor are named in `det.degenerate` so a
receipt can carry the fact.
"""
import numpy as np
import pytest

from signalmap.distill import DistilledDetector, FeatureSpec, window

PROGRAMS = ["acf1(id(id(x)))", "crest(id(id(x)))", "meanabs(id(id(x)))",
            "std(id(id(x)))", "zcr(id(id(x)))"]
DEGENERATE = "std(id(id(x)))"


def _windows(n=40, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.extend(window(rng.normal(0.0, scale, 20 * 1024))[:20])
    return out


@pytest.fixture()
def fitted():
    spec = FeatureSpec(programs=list(PROGRAMS), premium=[], window=1024)
    return DistilledDetector.fit(spec, _windows(), envelope=3.0)


def test_constant_feature_is_reported_as_degenerate(fitted):
    assert fitted.degenerate is not None
    flagged = [p for p, d in zip(PROGRAMS, fitted.degenerate) if d]
    assert flagged == [DEGENERATE]


def test_degenerate_feature_cannot_dominate_the_score(fitted):
    """Score an unseen window whose amplitude differs from the fit data.

    Amplitude is removed by z-normalisation, so a well-behaved detector barely
    notices. Before the fix the rounding signature of the new scale entered
    through the degenerate feature and dwarfed everything else.
    """
    unseen = window(np.random.default_rng(7).normal(0.0, 0.4, 20 * 1024))[:20]
    scores = [fitted.score(w) for w in unseen]
    assert max(scores) < 100.0, f"degenerate feature still dominating: {max(scores)}"
    assert all(np.isfinite(scores))


def test_threshold_stays_in_a_sane_range(fitted):
    assert 0.0 < fitted.threshold < 1e3


def test_non_degenerate_features_are_untouched(fitted):
    """The fix must not quietly rescale the features that were fine."""
    healthy = _windows(n=2, seed=99)
    raw = np.array([fitted.spec.featurize(w) for w in healthy])
    z = np.abs(raw - fitted.med) / fitted.mad
    keep = ~np.asarray(fitted.degenerate)
    assert np.allclose([fitted.score(w) for w in healthy], z[:, keep].max(axis=1))


def test_detector_survives_a_save_load_round_trip(fitted, tmp_path):
    path = tmp_path / "det.json"
    fitted.save(str(path))
    back = DistilledDetector.load(str(path))
    assert list(np.asarray(back.degenerate)) == list(np.asarray(fitted.degenerate))
    w = window(np.random.default_rng(3).normal(0.0, 1.0, 20 * 1024))[0]
    assert back.score(w) == pytest.approx(fitted.score(w))


def test_detector_without_degenerate_info_still_scores(fitted):
    """Older saved detectors carry no mask; they must keep working."""
    legacy = DistilledDetector(fitted.spec, fitted.med, fitted.mad, fitted.threshold)
    w = window(np.random.default_rng(5).normal(0.0, 1.0, 20 * 1024))[0]
    assert np.isfinite(legacy.score(w))


# --------------------------------------------------- the capacity gate's reach
def test_the_capacity_gate_is_inert_above_its_crossover():
    """Measured, and it decides how the gate may be described.

    `budget = C * n_recordings` with C=50 exceeds the whole enumerated grammar
    once a bank has more than ~43 recordings, so from there the gate admits
    every candidate and protects nothing. The MIMII banks in study/ carry
    891-968 fit recordings; the gate did nothing on any of them. This pins the
    crossover so the claim in the README cannot drift away from it.
    """
    from signalmap.distill import enumerate_programs, gate

    progs = list(enumerate_programs())
    crossover = len(progs) / 50.0
    assert len(gate(progs, n_recordings=int(crossover) - 10, C=50)) < len(progs)
    assert len(gate(progs, n_recordings=int(crossover) + 10, C=50)) == len(progs)
    assert len(gate(progs, n_recordings=900, C=50)) == len(progs)
