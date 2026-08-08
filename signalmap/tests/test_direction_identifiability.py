"""A detector must refuse a decision whose direction it cannot know.

Fitting on healthy data alone fixes a threshold but never observes an anomaly,
so "far from healthy means faulty" is an assumption, not something learned.
Measured over nine domains under one frozen recipe, that assumption is wrong
about as often as it is right: MIMII valve ranks anomalies as MORE normal
(AUC 0.2786), MIMII slider as less (0.8254), and Paderborn phase current
inverts again (0.3172) — each with a bootstrap CI clear of 0.5.

So the detector carries a direction that starts UNKNOWN. A handful of labelled
anchors can identify it; without them the honest answer to "is this window an
anomaly?" is REFUSED, not a coin flip dressed as a score.
"""
import json

import numpy as np
import pytest

from signalmap.distill import DistilledDetector, FeatureSpec, window

PROGRAMS = ["acf1(id(id(x)))", "crest(id(id(x)))", "meanabs(id(id(x)))",
            "zcr(id(id(x)))"]


def _ar1(phi, rng, n=20 * 1024):
    e = rng.normal(0.0, 1.0, n + 2000)
    x = np.empty(n + 2000)
    x[0] = e[0]
    for i in range(1, n + 2000):
        x[i] = phi * x[i - 1] + e[i]
    return x[2000:]


def _windows(phis, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for phi in phis:
        out.extend(window(_ar1(phi, rng))[:20])
    return out


@pytest.fixture(scope="module")
def fitted():
    """Healthy spans phi in [0.2, 0.8]; the centre is 0.5."""
    rng = np.random.default_rng(0)
    phis = list(rng.uniform(0.2, 0.8, 30))
    spec = FeatureSpec(programs=list(PROGRAMS), premium=[], window=1024)
    return DistilledDetector.fit(spec, _windows(phis, seed=1), envelope=3.0)


def _anchors(fitted, anomaly_phi, n=12, seed=5):
    """Anchors plus the recording id of every window — 20 windows cut from one
    signal are one observation, not twenty."""
    rng = np.random.default_rng(seed)
    normal = _windows(list(rng.uniform(0.2, 0.8, n)), seed=seed)
    anomalous = _windows([anomaly_phi] * n, seed=seed + 1)
    wins = normal + anomalous
    labels = [0] * len(normal) + [1] * len(anomalous)
    groups = [i // 20 for i in range(len(wins))]
    return wins, labels, groups


def test_direction_starts_unknown(fitted):
    assert fitted.direction is None


def test_decide_refuses_while_direction_is_unknown(fitted):
    w = window(np.random.default_rng(2).normal(0, 1, 20 * 1024))[0]
    out = fitted.decide(w)
    assert out.verdict == "REFUSED"
    assert "direction" in out.reason.lower()


def test_anchors_far_outside_healthy_identify_an_aligned_direction(fitted):
    wins, labels, groups = _anchors(fitted, 0.95)
    verdict = fitted.calibrate_direction(wins, labels, groups=groups)
    assert verdict.identified and verdict.sign == 1
    assert verdict.ci_lo > 0.5
    assert fitted.direction == 1


def test_anchors_at_the_healthy_centre_identify_an_inverted_direction(fitted):
    """The valve case: the anomaly is more stereotyped than normal operation."""
    wins, labels, groups = _anchors(fitted, 0.5)
    verdict = fitted.calibrate_direction(wins, labels, groups=groups)
    assert verdict.identified and verdict.sign == -1
    assert verdict.ci_hi < 0.5
    assert fitted.direction == -1


def test_indistinguishable_anchors_leave_the_direction_unidentified(fitted):
    """A true null: both anchor groups are the SAME process, different noise.

    Anything the detector finds here is chance, so the CI must cover 0.5 and
    the direction must stay unknown.
    """
    rng = np.random.default_rng(11)
    phis = list(rng.uniform(0.2, 0.8, 12))
    a = _windows(phis, seed=20)
    b = _windows(phis, seed=21)
    groups = [i // 20 for i in range(len(a) + len(b))]
    verdict = fitted.calibrate_direction(
        a + b, [0] * len(a) + [1] * len(b), groups=groups)
    assert not verdict.identified and verdict.sign is None
    assert verdict.ci_lo <= 0.5 <= verdict.ci_hi
    assert fitted.direction is None
    assert fitted.decide(a[0]).verdict == "REFUSED"


def test_grouping_changes_the_inferential_unit(fitted):
    """Why `groups` exists: the recording is the observation, not the window.

    Both calls see the same 480 windows and report the same point estimate;
    only the resampling unit differs, and `n_anchors` says which was used.
    Which interval ends up wider depends on the case — this pins the unit, not
    a width ordering.
    """
    wins, labels, groups = _anchors(fitted, 0.95)
    windowed = fitted.calibrate_direction(wins, labels)
    grouped = fitted.calibrate_direction(wins, labels, groups=groups)
    assert windowed.n_anchors == 480 and grouped.n_anchors == 24
    # The estimates must differ: one ranks 480 windows, the other 24 recording
    # means. An earlier version reported the window AUC while bootstrapping
    # recordings, which is how `checkup` ended up contradicting itself.
    assert windowed.auc != pytest.approx(grouped.auc)
    assert (windowed.ci_lo, windowed.ci_hi) != (grouped.ci_lo, grouped.ci_hi)


def test_too_few_anchors_is_refused_rather_than_guessed(fitted):
    wins, labels, _ = _anchors(fitted, 0.95, n=1)
    verdict = fitted.calibrate_direction(wins[:3], labels[:3])
    assert not verdict.identified and "anchor" in verdict.reason.lower()


def test_too_few_anchor_recordings_is_refused(fitted):
    """Forty windows from two recordings are still two observations."""
    wins, labels, _ = _anchors(fitted, 0.95, n=1)
    groups = [i // 20 for i in range(len(wins))]
    verdict = fitted.calibrate_direction(wins, labels, groups=groups)
    assert not verdict.identified and "recordings" in verdict.reason.lower()


def test_anchors_of_a_single_class_are_refused(fitted):
    wins, _, _ = _anchors(fitted, 0.95)
    verdict = fitted.calibrate_direction(wins, [0] * len(wins))
    assert not verdict.identified and "both" in verdict.reason.lower()


def test_an_identified_inverted_direction_flips_the_alarm(fitted):
    """With sign -1, LOW scores are the anomalous ones — that is the point."""
    wins, labels, groups = _anchors(fitted, 0.5)
    fitted.calibrate_direction(wins, labels, groups=groups)
    assert fitted.direction == -1
    centre = _windows([0.5], seed=31)          # anomalous under this direction
    spread = _windows([0.2, 0.8], seed=32)     # ordinary healthy
    assert np.mean([fitted.decide(w).verdict == "ALARM" for w in centre]) > \
        np.mean([fitted.decide(w).verdict == "ALARM" for w in spread])


def test_direction_survives_a_save_load_round_trip(fitted, tmp_path):
    wins, labels, groups = _anchors(fitted, 0.95)
    fitted.calibrate_direction(wins, labels, groups=groups)
    p = tmp_path / "det.json"
    fitted.save(str(p))
    back = DistilledDetector.load(str(p))
    assert back.direction == fitted.direction
    w = window(np.random.default_rng(4).normal(0, 1, 20 * 1024))[0]
    assert back.decide(w).verdict == fitted.decide(w).verdict


def test_legacy_alert_is_untouched(fitted):
    """`alert()` is the shipped surface and must keep its exact meaning."""
    w = window(np.random.default_rng(6).normal(0, 1, 20 * 1024))[0]
    assert fitted.alert(w) == (fitted.score(w) >= fitted.threshold)


# --------------------------------------------- the cut must come from anchors
def test_calibrated_detector_actually_alarms_on_anomalies(fitted):
    """Found by an end-to-end test on a fresh clone: identifying the sign was
    not enough. `decide` still used the healthy-envelope threshold (99th
    percentile x3), which the study showed never fires on real data, so a
    calibrated detector answered QUIET to an obvious anomaly.

    The anchors that identify the direction also locate the cut, so use them.
    """
    wins, labels, groups = _anchors(fitted, 0.95)
    v = fitted.calibrate_direction(wins, labels, groups=groups)
    assert v.identified and v.sign == 1

    anomalous = _windows([0.95], seed=41)
    healthy = _windows([0.3, 0.7], seed=42)
    hit = np.mean([fitted.decide(w).verdict == "ALARM" for w in anomalous])
    fp = np.mean([fitted.decide(w).verdict == "ALARM" for w in healthy])
    assert hit > 0.5, f"calibrated detector stayed quiet on anomalies: {hit:.2f}"
    assert fp < 0.5, f"too many false alarms on healthy: {fp:.2f}"


def test_inverted_direction_alarms_on_the_low_side(fitted):
    wins, labels, groups = _anchors(fitted, 0.5)
    v = fitted.calibrate_direction(wins, labels, groups=groups)
    assert v.identified and v.sign == -1
    anomalous = _windows([0.5], seed=43)
    healthy = _windows([0.2, 0.8], seed=44)
    hit = np.mean([fitted.decide(w).verdict == "ALARM" for w in anomalous])
    fp = np.mean([fitted.decide(w).verdict == "ALARM" for w in healthy])
    assert hit > fp, f"inverted cut does not separate: hit={hit:.2f} fp={fp:.2f}"


def test_the_anchor_cut_survives_save_load(fitted, tmp_path):
    wins, labels, groups = _anchors(fitted, 0.95)
    fitted.calibrate_direction(wins, labels, groups=groups)
    p = tmp_path / "d.json"
    fitted.save(str(p))
    back = DistilledDetector.load(str(p))
    assert back.decision_cut == pytest.approx(fitted.decision_cut)
    w = _windows([0.95], seed=45)[0]
    assert back.decide(w).verdict == fitted.decide(w).verdict


# ----------------------------------------------------- hardening on load/save
def test_a_corrupt_direction_is_rejected_not_silently_inverted(tmp_path, fitted):
    """`decide` branches on direction == 1 and treats everything else as -1.

    A det.json carrying direction: 0 or 7 would therefore be read as an
    inverted detector and quietly flip every decision. Only +1, -1 and null
    are meaningful, so anything else must fail loudly at load time.
    """
    p = tmp_path / "d.json"
    fitted.save(str(p))
    d = json.loads(p.read_text())
    d["direction"] = 7
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="direction"):
        DistilledDetector.load(str(p))


def test_a_non_finite_cut_is_rejected_on_load(tmp_path, fitted):
    """The project's own receipt verifier refuses NaN/Infinity literals; a
    detector file must not be laxer than that."""
    p = tmp_path / "d.json"
    fitted.save(str(p))
    d = json.loads(p.read_text())
    d["decision_cut"] = float("nan")
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="decision_cut"):
        DistilledDetector.load(str(p))


def test_anchor_scores_with_nan_do_not_produce_a_nan_cut(fitted):
    """A NaN cut compares False against everything, so the detector would go
    silent for good — and json.dump would write an invalid `NaN` literal."""
    from signalmap.distill import _best_cut
    y = np.array([0, 0, 1, 1])
    s = np.array([1.0, float("nan"), 5.0, 6.0])
    cut = _best_cut(y, s, +1)
    assert cut is None or np.isfinite(cut)


def test_grouped_calibration_judges_recordings_not_windows(fitted):
    """With `groups` the whole question moves to the recording level.

    Found by `signalmap checkup`, which computes its AUC over recording means
    and then asked calibrate_direction the same question: the two disagreed,
    because the bootstrap resampled recordings while the estimate was still
    taken over windows. One command reported a decided direction and no
    decided direction in the same breath.
    """
    wins, labels, groups = _anchors(fitted, 0.95)
    v = fitted.calibrate_direction(wins, labels, groups=groups)

    # the point estimate must be the recording-level AUC
    import collections
    per = collections.defaultdict(list)
    lab = {}
    for w, y, g in zip(wins, labels, groups):
        per[g].append(fitted.score(w)); lab[g] = y
    ids = sorted(per)
    from signalmap.distill import _rank_auc
    expected = _rank_auc(np.array([lab[g] for g in ids]),
                         np.array([float(np.mean(per[g])) for g in ids]))
    assert v.auc == pytest.approx(expected)
    assert v.n_anchors == len(ids)
