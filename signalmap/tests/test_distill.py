"""Tests for `signalmap distill` — capacity gate, spec.json roundtrip (fit/monitor
deploy surface), and an end-to-end run on a small synthetic 2-class bank."""
import numpy as np
import pytest

from signalmap.distill import (
    DistilledDetector,
    FeatureSpec,
    Program,
    budget_for,
    distill,
    enumerate_programs,
    gate,
    load_bank,
    window,
)


# ------------------------------------------------------------ synthetic bank
def _make_bank(tmp_path, seed=0):
    """2 clearly separable classes: A = low-freq tonal, B = high-freq + bursts.
    3 recordings each, ~6 windows/recording."""
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


# ---------------------------------------------------------------- gate rules
def test_budget_formula():
    assert budget_for(24, 2128, C=50) == 1200          # min(2128, 50*24)
    assert budget_for(24, 2128, C=50) / 24 <= 78        # sanity threshold from spec table
    assert budget_for(100, 2128, C=50) == 2128          # grammar-bounded
    assert budget_for(0, 2128, C=50) == 0


def test_enumeration_is_complexity_ordered_and_cheap_first():
    progs = enumerate_programs()
    keys = [p.complexity() for p in progs]
    assert keys == sorted(keys), "grammar must be totally ordered by complexity"
    # the very first programs are depth-0 (id∘id) with the cheapest pools
    assert all(p.t1 == "id" and p.t2 == "id" for p in progs[:5])
    # a depth-2 fft program must sort AFTER a depth-0 cheap program
    cheap = next(i for i, p in enumerate(progs) if p.name == "std(id(id(x)))")
    pricey = next(i for i, p in enumerate(progs)
                  if p.t1 != "id" and p.t2 != "id" and p.pool == "specratio")
    assert cheap < pricey


def test_gate_is_deterministic_and_reproducible():
    progs = enumerate_programs()
    k1 = gate(progs, n_recordings=6, C=50)
    k2 = gate(enumerate_programs(), n_recordings=6, C=50)
    assert [p.name for p in k1] == [p.name for p in k2]      # reproducible
    assert len(k1) == budget_for(6, len(progs), 50)          # budget-cut
    # the cut keeps the CHEAPEST prefix: max complexity kept <= min complexity dropped
    dropped = progs[len(k1):]
    if dropped:
        assert max(p.complexity() for p in k1) <= min(p.complexity() for p in dropped)


def test_gate_respects_capacity_per_recording():
    progs = enumerate_programs()
    for n_rec in (2, 6, 24):
        kept = gate(progs, n_rec, C=50)
        assert len(kept) / n_rec <= 78     # never above the collapse-zone margin


# ------------------------------------------------- spec.json roundtrip (deploy)
def test_featurespec_roundtrip(tmp_path):
    spec = FeatureSpec(programs=["std(id(id(x)))", "crest(diff(id(x)))"],
                       classes=["A", "B"], nested_acc=0.9)
    p = tmp_path / "spec.json"
    spec.save(str(p))
    back = FeatureSpec.load(str(p))
    assert back.programs == spec.programs and back.classes == spec.classes

    w = np.random.default_rng(0).standard_normal(1024)
    v1 = back.featurize(w)
    v2 = np.array([Program("std(id(id(x)))", "id", "id", "std")(w),
                   Program("crest(diff(id(x)))", "diff", "id", "crest")(w)])
    np.testing.assert_allclose(v1, v2)               # matches direct computation
    np.testing.assert_allclose(v1, back.featurize(w))  # deterministic


def test_distilled_detector_fit_monitor_roundtrip(tmp_path):
    """The fit/monitor deploy surface: fit a detector on HEALTHY windows via the
    distilled spec, save/load, then alert on the fault class, stay quiet on
    held-out healthy."""
    bank = _make_bank(tmp_path)
    res = distill(bank, C=50, kmax=3, thr=0.005, n_perm=10, trees=25,
                  cand=15, null_check=False)
    spec_p = tmp_path / "spec.json"
    res.spec.save(str(spec_p))
    spec = FeatureSpec.load(str(spec_p))

    healthy = [w for w, y in zip(bank.windows, bank.y) if y == "A"]
    fault = [w for w, y in zip(bank.windows, bank.y) if y == "B"]
    det = DistilledDetector.fit(spec, healthy[:12])   # threshold auto-calibrated

    d_p = tmp_path / "det.json"
    det.save(str(d_p))
    det = DistilledDetector.load(str(d_p))

    h_rate = np.mean([det.alert(w) for w in healthy[12:]])
    f_rate = np.mean([det.alert(w) for w in fault])
    assert h_rate < 0.2, f"too many false alarms on healthy: {h_rate:.0%}"
    assert f_rate > 0.8, f"missed the fault class: only {f_rate:.0%}"


class _StubSpec(FeatureSpec):
    """Feature = first sample of the window; lets the test control the healthy
    feature distribution exactly (regression pin for the WS2 detector fix)."""
    def featurize(self, w):
        return np.array([float(np.asarray(w)[0])])


def test_detector_survives_zero_mad_and_keeps_sensitivity():
    """Regression: a discrete, near-constant healthy feature (MAD = 0, e.g.
    acflag-style outputs) must neither crash nor deafen the detector — a small
    absolute shift still alerts, identical healthy stays quiet."""
    spec = _StubSpec(programs=[])
    healthy = [np.full(8, 1.0) for _ in range(30)]        # feature exactly constant
    det = DistilledDetector.fit(spec, healthy)
    assert np.all(np.isfinite(det.mad)) and det.threshold > 0
    assert not det.alert(np.full(8, 1.0)), "false alarm on identical healthy"
    assert det.alert(np.full(8, 1.001)), "under-fired on a small absolute shift"


def test_detector_threshold_calibrates_to_healthy_envelope():
    """Regression: with a heavy-tailed healthy feature the robust z routinely
    exceeds a fixed 4-sigma cut; the threshold must come from the healthy
    envelope instead (quiet on held-out healthy, loud beyond the envelope)."""
    rng = np.random.default_rng(7)
    draw = lambda: np.full(8, float(rng.standard_t(df=1.5)))  # heavy tails
    healthy = [draw() for _ in range(200)]
    det = DistilledDetector.fit(spec := _StubSpec(programs=[]), healthy)
    assert det.threshold > 4.0, "threshold did not scale to the healthy envelope"
    h_rate = np.mean([det.alert(draw()) for _ in range(200)])
    assert h_rate < 0.1, f"fixed-cut false alarms are back: {h_rate:.0%}"
    far_out = np.full(8, float(np.max(np.abs(healthy))) * 50)
    assert det.alert(far_out), "missed a fault far outside the healthy envelope"


# --------------------------------------------------------------- end-to-end
def test_distill_end_to_end_separable_classes(tmp_path):
    bank = _make_bank(tmp_path)
    assert bank.n_recordings == 6 and bank.classes == ["A", "B"]
    res = distill(bank, C=50, kmax=3, thr=0.005, n_perm=15, trees=25,
                  cand=15, null_check=True)

    # capacity gate applied
    assert res.budget == budget_for(6, res.grammar_total, 50)
    assert res.budget / bank.n_recordings <= 78

    # separable classes -> nested acc well above chance, null falls back to chance
    assert res.nested_acc > bank.chance + 0.05
    assert res.null_acc <= bank.chance + 0.25
    assert res.p_forged <= 0.2

    # receipt is human-readable and names the selection + gate numbers
    assert "gauntlet receipt" in res.report
    assert "capacity gate" in res.report
    assert res.spec.programs, "spec must carry at least one program"
    assert all(prog in res.report for prog in res.spec.programs)


def test_spec_save_creates_missing_parent_dir(tmp_path):
    """The README writes `--out artifacts/spec.json`; a fresh checkout has no
    artifacts/ dir. save() must create the parent so the gauntlet's result isn't
    thrown away by a FileNotFoundError after minutes of work (L3 fresh-checkout)."""
    spec = FeatureSpec(programs=["mean"], classes=["A", "B"])
    out = tmp_path / "artifacts" / "spec.json"
    assert not out.parent.exists()
    spec.save(str(out))
    assert out.exists()
    assert FeatureSpec.load(str(out)).programs == ["mean"]


def test_distill_needs_at_least_two_recordings(tmp_path):
    """A one-recording bank cannot run the LOGO gauntlet (LeaveOneGroupOut needs
    >=2 groups). distill() must fail closed with a clear message, not leak a raw
    sklearn ValueError (L2 review minor #2)."""
    t = np.arange(6 * 1024)
    x = np.sin(2 * np.pi * 0.02 * t) + 0.1 * np.random.default_rng(0).standard_normal(len(t))
    np.save(tmp_path / "A_0.npy", x.astype(np.float64))
    bank = load_bank(str(tmp_path), label_by="prefix")
    assert bank.n_recordings == 1
    with pytest.raises(SystemExit, match="2 recordings"):
        distill(bank)


def test_load_bank_windows_are_normalised():
    t = np.arange(3 * 1024)
    x = np.sin(2 * np.pi * 0.02 * t) + 0.01 * t + 10.0   # tone + DC + linear trend
    w = window(x)
    assert len(w) == 3
    for s in w:
        assert len(s) == 1024
        assert abs(s.mean()) < 1e-6              # detrended + centered
        assert abs(s.std() - 1.0) < 1e-6         # z-normed (real post-detrend variance)
