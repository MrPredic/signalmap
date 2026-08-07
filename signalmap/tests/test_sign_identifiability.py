"""Sign-identifiability readout — the machinery that produces the headline.

The study's whole value is that one number (AUC, with a CI over recordings)
is computed identically in every domain. So the parts that could quietly
distort it are pinned here: the hand-written AUC against sklearn's, the
upward bias of max(AUC, 1-AUC) that H2 must be tested against, the alarm-gap
sign, the sign-transfer statistic, and the bank checker's leakage detection.

Offline and deterministic: no banks, no network, no detector fitting.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

FACTORY = Path(__file__).resolve().parents[2] / "study" / "tools"

pytestmark = pytest.mark.skipif(
    not (FACTORY / "sign_identifiability_readout.py").exists(),
    reason="study/tools not present (e.g. packaged install)")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, FACTORY / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


readout = _load("sign_identifiability_readout")
checker = _load("check_domain_bank")


# ------------------------------------------------------------------ the AUC
@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_auc_matches_sklearn(seed):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 200)
    if y.sum() in (0, y.size):
        pytest.skip("degenerate draw")
    s = rng.normal(size=200) + 0.4 * y
    assert readout.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_handles_ties_like_sklearn():
    from sklearn.metrics import roc_auc_score
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])  # heavy ties across classes
    assert readout.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_is_one_for_perfect_and_zero_for_inverted():
    y = np.array([0, 0, 1, 1])
    assert readout.auc(y, np.array([0.0, 0.1, 0.9, 1.0])) == pytest.approx(1.0)
    assert readout.auc(y, np.array([1.0, 0.9, 0.1, 0.0])) == pytest.approx(0.0)


def test_auc_is_nan_when_a_class_is_missing():
    assert np.isnan(readout.auc(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0])))


# ------------------------------------------------------- the null and its bias
def test_shuffle_null_is_centred_on_half():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 60)
    s = rng.normal(size=120) + 0.8 * y  # a real effect; the null must erase it
    null_auc, _ = readout.perm_null(y, s, n=400, seed=0)
    assert null_auc.mean() == pytest.approx(0.5, abs=0.03)
    lo, hi = np.percentile(null_auc, [2.5, 97.5])
    assert lo <= 0.5 <= hi


def test_directionfree_statistic_is_biased_upward_under_the_null():
    """Why H2 is tested against the shuffle quantile and never against 0.5."""
    rng = np.random.default_rng(1)
    y = np.array([0] * 50 + [1] * 50)
    s = rng.normal(size=100)  # pure noise, no effect whatsoever
    null_auc, null_star = readout.perm_null(y, s, n=600, seed=0)
    assert null_star.min() >= 0.5 - 1e-12          # max(a, 1-a) can never dip below
    assert null_star.mean() > null_auc.mean()      # and it sits above the AUC null
    assert np.percentile(null_star, 95) > 0.55     # so 0.5 would be a wrong yardstick


def test_bootstrap_ci_brackets_a_strong_effect_above_half():
    rng = np.random.default_rng(3)
    y = np.array([0] * 80 + [1] * 80)
    s = rng.normal(size=160) + 2.0 * y
    lo, hi, used = readout.boot_ci(y, s, n=400, seed=0)
    assert used > 0 and 0.5 < lo <= hi <= 1.0


def test_bootstrap_ci_brackets_an_inverted_effect_below_half():
    rng = np.random.default_rng(4)
    y = np.array([0] * 80 + [1] * 80)
    s = rng.normal(size=160) - 2.0 * y  # anomalies score LOWER: the valve case
    lo, hi, _ = readout.boot_ci(y, s, n=400, seed=0)
    assert 0.0 <= lo <= hi < 0.5


# --------------------------------------------------------------- alarm gap M3
def test_alarm_gap_is_negative_when_anomalies_alarm_less():
    y = np.array([0] * 20 + [1] * 20)
    s = np.concatenate([np.full(20, 10.0), np.full(20, 0.1)])
    out = readout.alarm_gap(y, s, threshold=1.0)
    assert out["alarm_rate_normal"] == 1.0 and out["alarm_rate_anomaly"] == 0.0
    assert out["gap"] == -1.0 and out["gap_ci_hi"] < 0


# ------------------------------------------------------------- sign transfer M4
def _res(domain, direction, verdict="PASS"):
    return {"domain": domain, "direction": direction, "verdict": verdict}


def test_transfer_hit_rate_is_zero_when_signs_disagree():
    out = readout.transfer_matrix([_res("a", "inverted"), _res("b", "aligned")])
    assert out["n_ordered_pairs"] == 2 and out["sign_transfer_hit_rate"] == 0.0


def test_transfer_hit_rate_is_one_when_signs_agree():
    out = readout.transfer_matrix([_res("a", "aligned"), _res("b", "aligned")])
    assert out["sign_transfer_hit_rate"] == 1.0


def test_transfer_ignores_a_domain_whose_null_control_failed():
    """A CI clear of 0.5 is worthless if the healthy set separates from itself.

    MAFAULDA is the real case: sorting its healthy eval files sorts them by
    rotation speed, so N2 fails and the apparent inversion may be an
    operating-point difference. Such a domain must not support H1.
    """
    out = readout.transfer_matrix([
        _res("clean", "aligned"),
        _res("control_failed", "inverted", verdict="REFUSED"),
    ])
    assert out["n_decided_domains"] == 1
    assert out["inverted"] == [] and out["aligned"] == ["clean"]


def test_transfer_ignores_undetermined_domains():
    out = readout.transfer_matrix(
        [_res("a", "aligned"), _res("b", "undetermined"), _res("c", "aligned")])
    assert out["n_decided_domains"] == 2 and "b" not in out["aligned"]


# ------------------------------------------------------------- bank contract
def _write_bank(root, n_fit=3, n_norm=3, n_anom=3, leak=False, seed=0):
    """Minimal contract-shaped bank; counts are lowered by the caller."""
    rng = np.random.default_rng(seed)
    (root / "fit").mkdir(parents=True)
    (root / "eval").mkdir(parents=True)
    files, first_fit = {}, None
    for i in range(n_fit):
        arr = rng.normal(size=checker.MIN_SAMPLES)
        if i == 0:
            first_fit = arr
        np.save(root / "fit" / f"h{i}.npy", arr)
    for i in range(n_norm):
        arr = first_fit if (leak and i == 0) else rng.normal(size=checker.MIN_SAMPLES)
        np.save(root / "eval" / f"normal_{i}.npy", arr)
    for i in range(n_anom):
        np.save(root / "eval" / f"anomaly_{i}.npy", rng.normal(size=checker.MIN_SAMPLES))
    for path in sorted(root.rglob("*.npy")):
        files[str(path.relative_to(root))] = checker.sha256_file(path)
    (root / "manifest.json").write_text(json.dumps({
        "domain": root.name, "source_url": "https://example.invalid/x",
        "license": "CC-BY-4.0", "modality": "synthetic", "fs_hz": 1000,
        "channel": "0", "anomaly_mapping": "prefix", "files": files}))
    return root


@pytest.fixture()
def small_counts(monkeypatch):
    monkeypatch.setattr(checker, "MIN_FIT", 3)
    monkeypatch.setattr(checker, "MIN_EVAL_PER_CLASS", 3)


def test_contract_accepts_a_clean_bank(tmp_path, small_counts):
    problems, stats = checker.check_bank(_write_bank(tmp_path / "clean"))
    assert problems == []
    assert stats == {"n_fit": 3, "n_eval_normal": 3, "n_eval_anomaly": 3}


def test_contract_rejects_fit_eval_leakage(tmp_path, small_counts):
    problems, _ = checker.check_bank(_write_bank(tmp_path / "leaky", leak=True))
    assert any(p.startswith("LEAKAGE:") for p in problems)


def test_contract_rejects_a_tampered_file(tmp_path, small_counts):
    root = _write_bank(tmp_path / "tampered")
    np.save(root / "eval" / "anomaly_0.npy",
            np.random.default_rng(99).normal(size=checker.MIN_SAMPLES))
    problems, _ = checker.check_bank(root)
    assert any("sha256 mismatch" in p for p in problems)


def test_contract_rejects_short_and_constant_recordings(tmp_path, small_counts):
    root = _write_bank(tmp_path / "short")
    np.save(root / "fit" / "h0.npy", np.zeros(checker.MIN_SAMPLES))
    np.save(root / "fit" / "h1.npy", np.ones(10))
    problems, _ = checker.check_bank(root)
    assert any("constant signal" in p for p in problems)
    assert any("samples <" in p for p in problems)


def test_contract_rejects_unlabelled_eval_file(tmp_path, small_counts):
    root = _write_bank(tmp_path / "stray")
    np.save(root / "eval" / "mystery.npy",
            np.random.default_rng(5).normal(size=checker.MIN_SAMPLES))
    problems, _ = checker.check_bank(root)
    assert any("without normal_/anomaly_ prefix" in p for p in problems)


# ------------------------------------------------- degenerate spec feature
def test_std_feature_is_constant_by_construction_after_windowing():
    """`window()` z-norms every window, so std(x) is exactly 1 — always.

    The lean-base spec that `distill` produces contains this program, so its
    healthy MAD collapses onto the 1e-12 guard floor and last-bit rounding
    becomes a first-order term of max|z|. Pinned here so a future change to
    the windowing or the grammar has to confront it deliberately.
    """
    from signalmap.distill import DistilledDetector, FeatureSpec, window

    rng = np.random.default_rng(0)
    wins = []
    for scale in (0.5, 1.0, 3.0):  # amplitude must not matter after z-norm
        wins.extend(window(rng.normal(0, scale, 20 * 1024))[:20])
    spec = FeatureSpec(programs=list(readout.SPEC_PROGRAMS), premium=[], window=1024)

    values = np.array([spec.featurize(w) for w in wins])
    idx = readout.SPEC_PROGRAMS.index("std(id(id(x)))")
    assert values[:, idx] == pytest.approx(1.0, abs=1e-9)

    det = DistilledDetector.fit(spec, wins, envelope=3.0)
    degenerate = np.asarray(det.degenerate, dtype=bool)
    assert degenerate[idx], "the constant-by-construction program is not flagged"
    assert degenerate.sum() == 1, \
        "exactly one program of the frozen spec is expected to be degenerate here"
    # and it must no longer capture the score: the floor is relative, so what
    # is left of float accumulation noise stays far below the real features.
    z = np.abs(spec.featurize(wins[0]) - det.med) / det.mad
    assert z[idx] < 1.0, f"degenerate feature still loud: z={z[idx]}"


def test_contract_requires_minimum_counts(tmp_path):
    problems, _ = checker.check_bank(_write_bank(tmp_path / "tiny"))
    assert any("< 20 required" in p for p in problems)
    assert any("< 30 required" in p for p in problems)
