"""Tests for premium feature families (signalmap.premium) and their distill
integration: RQA physics properties, pyunicorn parity (skipped if absent),
cost-receipted champion rule (premium enters the deploy spec ONLY on a
paired-CI-solid win), and spec.json roundtrip with premium features."""
import numpy as np
import pytest

from signalmap.premium import (
    ENVELOPE_FEATURE_NAMES,
    PREMIUM_FAMILIES,
    PremiumFamily,
    RQA_FEATURE_NAMES,
    envelope_features,
    rqa_features,
)


# ------------------------------------------------------------- RQA physics
def _sine(n=1024, f=0.02, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * f * t) + noise * rng.standard_normal(n)


def test_rqa_features_shape_and_determinism():
    w = _sine(noise=0.2)
    v1 = rqa_features(w)
    v2 = rqa_features(w)
    assert v1.shape == (len(RQA_FEATURE_NAMES),)
    assert np.all(np.isfinite(v1))
    np.testing.assert_allclose(v1, v2)


def test_rqa_recurrence_rate_matches_target():
    # rr is enforced via the distance quantile -> RR must land near the target
    for w in (_sine(noise=0.3), np.random.default_rng(1).standard_normal(1024)):
        rr = rqa_features(w)[RQA_FEATURE_NAMES.index("rqa_rr")]
        assert abs(rr - 0.1) < 0.03, f"RR {rr:.3f} far from target 0.1"


def test_rqa_determinism_separates_periodic_from_noise():
    det_i = RQA_FEATURE_NAMES.index("rqa_det")
    lam_i = RQA_FEATURE_NAMES.index("rqa_lam")
    det_sine = rqa_features(_sine(noise=0.05))[det_i]
    det_noise = rqa_features(np.random.default_rng(2).standard_normal(1024))[det_i]
    assert det_sine > det_noise + 0.1, (det_sine, det_noise)
    # laminarity in [0, 1]
    for w in (_sine(), np.random.default_rng(3).standard_normal(1024)):
        assert 0.0 <= rqa_features(w)[lam_i] <= 1.0


def test_rqa_pyunicorn_parity():
    pyunicorn = pytest.importorskip("pyunicorn.timeseries")
    w = _sine(noise=0.3, seed=4)
    rp = pyunicorn.RecurrencePlot(np.asarray(w, float), dim=3, tau=5,
                                  recurrence_rate=0.1, silence_level=3)
    ref = np.array([rp.recurrence_rate(), rp.determinism(l_min=2),
                    rp.laminarity(v_min=2), rp.diag_entropy(),
                    rp.average_diaglength(l_min=2), rp.trapping_time(v_min=2)])
    ours = rqa_features(w, dim=3, tau=5, rr=0.1)
    np.testing.assert_allclose(ours, ref, rtol=0.05, atol=0.05)


def test_registry_carries_rqa_with_cost_note():
    fam = PREMIUM_FAMILIES["rqa"]
    assert isinstance(fam, PremiumFamily)
    assert fam.feature_names == RQA_FEATURE_NAMES
    assert "O(n" in fam.cost_note  # the receipt must be able to name the cost class
    v = fam.featurize(_sine(noise=0.2))
    assert v.shape == (len(RQA_FEATURE_NAMES),)


# --------------------------------------------------------- envelope physics
def _am_signal(n=1024, f_def=128 / 1024, f_c=300 / 1024, m=0.8, noise=0.0,
               seed=0):
    """AM ground truth: carrier modulated at f_def (cycles/sample), both an
    integer number of cycles over the window -> no spectral leakage, closed-
    form Hilbert envelope = 1 + m*cos(2*pi*f_def*t) (Bedrosian holds, f_c >>
    f_def)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    env = 1 + m * np.cos(2 * np.pi * f_def * t)
    x = env * np.cos(2 * np.pi * f_c * t)
    return x + noise * rng.standard_normal(n)


def test_envelope_features_shape_and_determinism():
    w = _sine(noise=0.2)
    v1 = envelope_features(w)
    v2 = envelope_features(w)
    assert v1.shape == (len(ENVELOPE_FEATURE_NAMES),)
    assert np.all(np.isfinite(v1))
    assert v1.sum() == pytest.approx(1.0, abs=1e-6)  # relative-fraction features
    np.testing.assert_array_equal(v1, v2)  # byte-identical, no randomness


def test_envelope_am_parity_concentrates_in_defect_band():
    """Closed-form AM physics parity pin: f_def=128/1024 falls squarely inside
    band index 2 of 5 (considered range is bins 1..256 of a 513-bin spectrum,
    band width 51 bins: band2 = bins 103..153) -> that band's energy fraction
    must be both the max AND clearly dominant."""
    x = _am_signal()
    v = envelope_features(x)
    assert v.argmax() == 2
    assert v[2] > 0.5


def test_envelope_pure_carrier_is_null_no_defect_concentration():
    """Null side of the parity pin: an unmodulated carrier (+ small noise
    floor, to stay off the degenerate zero-energy branch) must NOT
    concentrate in any band -> near-uniform fractions."""
    x = _am_signal(m=0.0, noise=0.05, seed=7)
    v = envelope_features(x)
    assert v.max() < 0.35  # comfortably above uniform (0.2) but far from the
                            # >0.5 AM-concentration case


def test_registry_carries_envelope_with_cost_note():
    fam = PREMIUM_FAMILIES["envelope"]
    assert isinstance(fam, PremiumFamily)
    assert fam.feature_names == ENVELOPE_FEATURE_NAMES
    assert fam.needs_channels == 1
    assert "O(n" in fam.cost_note
    v = fam.featurize(_sine(noise=0.2))
    assert v.shape == (len(ENVELOPE_FEATURE_NAMES),)


# ------------------------------------------------- distill integration
from signalmap.distill import FeatureSpec, distill, load_bank


def _make_bank(tmp_path, seed=0):
    """Same separable synthetic bank as test_distill (base grammar solves it)."""
    rng = np.random.default_rng(seed)
    n = 6 * 1024
    t = np.arange(n)
    for cls in ("A", "B"):
        for r in range(3):
            if cls == "A":
                s = np.sin(2 * np.pi * 0.01 * t) + 0.1 * rng.standard_normal(n)
            else:
                s = (np.sin(2 * np.pi * 0.30 * t)
                     + 4.0 * (rng.random(n) < 0.02) * rng.standard_normal(n)
                     + 0.1 * rng.standard_normal(n))
            np.save(tmp_path / f"{cls}_{r}.npy", s.astype(np.float64))
    return load_bank(str(tmp_path), label_by="prefix")


def _noise_bank(tmp_path, seed=0):
    """Label-free noise: the base grammar CANNOT separate these classes."""
    rng = np.random.default_rng(seed)
    for cls in ("A", "B"):
        for r in range(3):
            np.save(tmp_path / f"{cls}_{r}.npy",
                    rng.standard_normal(6 * 1024).astype(np.float64))
    return load_bank(str(tmp_path), label_by="prefix")


@pytest.fixture
def oracle_family():
    """A planted premium family whose feature = mean window level. Combined with
    _oracle_bank (class B offset before z-norm is invisible... so instead we use
    a marker embedded in the first sample) it separates classes perfectly."""
    fam = PremiumFamily(
        name="oracle",
        featurize=lambda w: np.array([float(np.asarray(w)[0])]),
        feature_names=["oracle_first"],
        cost_note="O(1) test-only",
    )
    PREMIUM_FAMILIES["oracle"] = fam
    yield fam
    del PREMIUM_FAMILIES["oracle"]


def _plant_marker(bank):
    """Overwrite the first sample of each window with a class marker so ONLY the
    oracle family (which reads exactly that sample) can see the label."""
    for w, y in zip(bank.windows, bank.y):
        w[0] = 5.0 if y == "B" else -5.0
    return bank


def test_premium_excluded_when_base_already_wins(tmp_path):
    """Champion rule: on a bank the base grammar solves, premium must NOT enter
    the deploy spec (no silent replacement, receipt says EXCLUDED)."""
    bank = _make_bank(tmp_path)
    res = distill(bank, C=50, kmax=3, thr=0.005, n_perm=10, trees=25,
                  cand=15, null_check=False, premium=("rqa",))
    assert res.premium_receipts, "premium run must produce a receipt entry"
    rec = res.premium_receipts[0]
    assert rec["family"] == "rqa"
    assert rec["included"] is False
    assert res.spec.premium == []
    # the receipt is honest about cost either way
    assert rec["cost_ms"] > 0
    assert "premium" in res.report and "rqa" in res.report
    assert "EXCLUDED" in res.report


def test_premium_included_on_ci_solid_win(tmp_path, oracle_family):
    """Champion rule, other direction: when ONLY the premium family carries the
    label, the paired CI is solid and the family enters the deploy spec."""
    bank = _plant_marker(_noise_bank(tmp_path))
    res = distill(bank, C=50, kmax=3, thr=0.005, n_perm=10, trees=25,
                  cand=15, null_check=False, premium=("oracle",))
    rec = res.premium_receipts[0]
    assert rec["included"] is True, rec
    assert rec["ci_lo"] > 0
    assert res.spec.premium == ["oracle"]
    assert "INCLUDED" in res.report
    # deploy surface: featurize now appends the premium features
    v = res.spec.featurize(bank.windows[0])
    assert v.shape == (len(res.spec.programs) + 1,)


def test_run_cli_forwards_premium(tmp_path):
    """CLI surface: run_cli(premium=...) must run the challenger and write the
    premium block into the receipt file."""
    from signalmap.distill import run_cli
    _make_bank(tmp_path / "bank", seed=1) if (tmp_path / "bank").mkdir() is None else None
    # budget 6, not 50: this test pins CLI flag forwarding, not statistical
    # power, and the full budget costs ~70s of quadratic RQA to prove nothing extra.
    res = run_cli(str(tmp_path / "bank"), "prefix", 6, str(tmp_path / "spec.json"),
                  n_perm=10, trees=25, premium=("rqa",))
    assert res.premium_receipts and res.premium_receipts[0]["family"] == "rqa"
    report = open(tmp_path / "spec_report.md").read()
    assert "premium families" in report


def test_spec_premium_roundtrip_and_backward_compat(tmp_path):
    spec = FeatureSpec(programs=["std(id(id(x)))"], classes=["A", "B"],
                       premium=["rqa"])
    p = tmp_path / "spec.json"
    spec.save(str(p))
    back = FeatureSpec.load(str(p))
    assert back.premium == ["rqa"]
    w = np.random.default_rng(0).standard_normal(1024)
    v = back.featurize(w)
    assert v.shape == (1 + len(RQA_FEATURE_NAMES),)
    np.testing.assert_allclose(v, back.featurize(w))  # deterministic
    # old spec.json without the premium key must still load
    import json
    d = json.load(open(p))
    del d["premium"]
    json.dump(d, open(p, "w"))
    old = FeatureSpec.load(str(p))
    assert old.premium == []
    assert old.featurize(w).shape == (1,)


def test_spec_envelope_roundtrip(tmp_path):
    """Envelope analog of the rqa roundtrip: spec.json premium=['envelope']
    survives save/load and featurize appends the 5 relative-fraction features."""
    spec = FeatureSpec(programs=["std(id(id(x)))"], classes=["A", "B"],
                       premium=["envelope"])
    p = tmp_path / "spec_env.json"
    spec.save(str(p))
    back = FeatureSpec.load(str(p))
    assert back.premium == ["envelope"]
    w = np.random.default_rng(0).standard_normal(1024)
    v = back.featurize(w)
    assert v.shape == (1 + len(ENVELOPE_FEATURE_NAMES),)
    np.testing.assert_allclose(v, back.featurize(w))  # deterministic
