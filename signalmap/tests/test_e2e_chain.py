"""End-to-end chain test: distill -> spec.json -> fit/monitor -> alert.

Covers the one composition no other test exercises: a champion-rule INCLUDED
premium family flowing through the WHOLE deploy surface — spec.json save/load,
DistilledDetector fit on healthy, detector save/load, then alert on faults and
silence on held-out healthy. Small params (trees=25, n_perm=10) keep it in
seconds; the full-size verdicts live in research/factory preregs.
"""
import numpy as np
import pytest

from signalmap.distill import DistilledDetector, FeatureSpec, distill, load_bank
from signalmap.premium import PREMIUM_FAMILIES, PremiumFamily


def _noise_bank(tmp_path, seed=0):
    """Label-free noise: the base grammar cannot separate the classes."""
    rng = np.random.default_rng(seed)
    for cls in ("A", "B"):
        for r in range(3):
            np.save(tmp_path / f"{cls}_{r}.npy",
                    rng.standard_normal(6 * 1024).astype(np.float64))
    return load_bank(str(tmp_path), label_by="prefix")


def _separable_bank(tmp_path, seed=0):
    """A = low-freq tonal, B = high-freq + bursts (same as test_distill)."""
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


@pytest.fixture
def oracle_family():
    """Planted family reading the first sample — the only feature that can see
    a marker planted there (same trick as test_premium)."""
    fam = PremiumFamily(
        name="oracle",
        featurize=lambda w: np.array([float(np.asarray(w)[0])]),
        feature_names=["oracle_first"],
        cost_note="O(1) test-only",
    )
    PREMIUM_FAMILIES["oracle"] = fam
    yield fam
    del PREMIUM_FAMILIES["oracle"]


def test_chain_premium_included_to_alert(tmp_path, oracle_family):
    """distill (champion rule INCLUDES premium) -> spec.json -> fit on healthy
    -> detector.json -> monitor alerts on fault, stays quiet on healthy."""
    bank = _noise_bank(tmp_path)
    for w, y in zip(bank.windows, bank.y):
        w[0] = 5.0 if y == "B" else -5.0   # only the oracle family sees this

    res = distill(bank, C=50, kmax=3, thr=0.005, n_perm=10, trees=25,
                  cand=15, null_check=False, premium=("oracle",))
    assert res.spec.premium == ["oracle"], "premium must win CI-solid here"

    spec_p = tmp_path / "spec.json"
    res.spec.save(str(spec_p))
    spec = FeatureSpec.load(str(spec_p))
    assert spec.premium == ["oracle"]

    healthy = [w for w, y in zip(bank.windows, bank.y) if y == "A"]
    fault = [w for w, y in zip(bank.windows, bank.y) if y == "B"]
    det = DistilledDetector.fit(spec, healthy[:12])

    det_p = tmp_path / "det.json"
    det.save(str(det_p))
    det = DistilledDetector.load(str(det_p))
    assert det.spec.premium == ["oracle"], "premium must survive the detector roundtrip"

    f_rate = np.mean([det.alert(w) for w in fault])
    h_rate = np.mean([det.alert(w) for w in healthy[12:]])
    assert f_rate == 1.0, f"marker fault must always alert: {f_rate:.0%}"
    assert h_rate < 0.2, f"too many false alarms on healthy: {h_rate:.0%}"


def test_chain_real_rqa_spec_fit_monitor_alert(tmp_path):
    """The real shipped RQA family through the deploy chain. distill the base
    spec, then include 'rqa' as a champion-rule winner would (the CWRU-case
    output shape), and require the fit/monitor surface to stay coherent:
    feature length, JSON roundtrips, alert on faults, quiet on healthy."""
    bank = _separable_bank(tmp_path)
    res = distill(bank, C=50, kmax=3, thr=0.005, n_perm=10, trees=25,
                  cand=15, null_check=False)
    res.spec.premium = ["rqa"]   # simulate an INCLUDED verdict for the chain

    spec_p = tmp_path / "spec.json"
    res.spec.save(str(spec_p))
    spec = FeatureSpec.load(str(spec_p))
    n_feat = len(spec.programs) + len(PREMIUM_FAMILIES["rqa"].feature_names)
    assert spec.featurize(bank.windows[0]).shape == (n_feat,)

    healthy = [w for w, y in zip(bank.windows, bank.y) if y == "A"]
    fault = [w for w, y in zip(bank.windows, bank.y) if y == "B"]
    det = DistilledDetector.fit(spec, healthy[:8])
    det_p = tmp_path / "det.json"
    det.save(str(det_p))
    det = DistilledDetector.load(str(det_p))
    assert len(det.med) == n_feat

    f_rate = np.mean([det.alert(w) for w in fault[:6]])
    h_rate = np.mean([det.alert(w) for w in healthy[8:12]])
    assert f_rate > 0.8, f"missed the fault class: {f_rate:.0%}"
    assert h_rate < 0.5, f"false alarms on healthy: {h_rate:.0%}"


def _coherence_bank(tmp_path, seed=0):
    """2-channel bank where ONLY cross-channel structure carries the label:
    channel 0 is standard normal noise in BOTH classes (base grammar blind);
    class B's channel 1 is coupled to channel 0, class A's is independent."""
    rng = np.random.default_rng(seed)
    n = 6 * 1024
    for cls in ("A", "B"):
        for r in range(3):
            ch0 = rng.standard_normal(n)
            other = rng.standard_normal(n)
            ch1 = 0.8 * ch0 + 0.2 * other if cls == "B" else other
            np.save(tmp_path / f"{cls}_{r}.npy", np.stack([ch0, ch1]))
    return load_bank(str(tmp_path), label_by="prefix", multichannel=True)


def test_chain_coherence_included_to_alert(tmp_path):
    """SPEC_MULTICHANNEL_DISTILL step 3: real coherence family through the whole
    chain — multichannel bank -> distill --premium coherence -> champion rule
    INCLUDES it -> spec.json/detector roundtrip -> alert on coupled windows,
    quiet on held-out independent ones."""
    bank = _coherence_bank(tmp_path)
    res = distill(bank, C=50, kmax=3, thr=0.005, n_perm=10, trees=25,
                  cand=15, null_check=False, premium=("coherence",))
    rec = res.premium_receipts[0]
    assert rec["family"] == "coherence"
    assert rec["included"] is True, rec
    assert res.spec.premium == ["coherence"]
    assert res.spec.channels == ["ch0", "ch1"]

    spec_p = tmp_path / "spec.json"
    res.spec.save(str(spec_p))
    spec = FeatureSpec.load(str(spec_p))
    assert spec.channels == ["ch0", "ch1"]

    healthy = [w for w, y in zip(bank.windows, bank.y) if y == "A"]
    fault = [w for w, y in zip(bank.windows, bank.y) if y == "B"]
    det = DistilledDetector.fit(spec, healthy[:12])
    det_p = tmp_path / "det.json"
    det.save(str(det_p))
    det = DistilledDetector.load(str(det_p))
    assert det.spec.channels == ["ch0", "ch1"]

    f_rate = np.mean([det.alert(w) for w in fault])
    h_rate = np.mean([det.alert(w) for w in healthy[12:]])
    assert f_rate > 0.8, f"missed the coupled class: {f_rate:.0%}"
    assert h_rate < 0.5, f"false alarms on healthy: {h_rate:.0%}"
