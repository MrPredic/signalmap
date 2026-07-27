"""Multi-channel distill (SPEC_MULTICHANNEL_DISTILL.md): 2D windows (C, 1024),
primary-channel convention (base grammar sees only channel 0), needs_channels
refusal, spec.json `channels` field, the coherence premium family (fixed
product config c128b2 = coherence_fair CONFIRMED-aug), and multichannel bank
ingest. The 1-channel path must stay byte-identical (pinned here)."""
import json

import numpy as np
import pytest

from signalmap.distill import Bank, FeatureSpec, W, distill, load_bank, window
from signalmap.premium import PREMIUM_FAMILIES, PremiumFamily


def _mc_signal(n_ch=2, n=3 * W, seed=0):
    return np.random.default_rng(seed).standard_normal((n_ch, n))


# ------------------------------------------------------------- window() 2D
def test_window_1d_byte_identical_pin():
    """Backward-compat pin: the 1-D path must produce exactly detrend+z-norm
    per 1024 slice (spec §2.1) — byte-identical to the status quo."""
    from scipy.signal import detrend
    x = np.random.default_rng(1).standard_normal(2 * W + 100)
    wins = window(x)
    assert len(wins) == 2
    for k, w in enumerate(wins):
        s = detrend(np.ascontiguousarray(x[k * W:(k + 1) * W], float))
        expected = (s - s.mean()) / (s.std() + 1e-12)
        np.testing.assert_array_equal(w, expected)


def test_window_2d_slices_synchronously_per_channel():
    """2-D (C, n) input -> list of (C, 1024) windows, each channel detrended +
    z-normed with EXACT window() semantics (per-channel identity to the 1-D path)."""
    X = _mc_signal(n_ch=3, n=2 * W + 50)
    wins = window(X)
    assert len(wins) == 2
    for w in wins:
        assert w.shape == (3, W)
    for ch in range(3):
        ref = window(X[ch])
        for k in range(2):
            np.testing.assert_array_equal(wins[k][ch], ref[k])


# ----------------------------------------------------- needs_channels refusal
def test_premium_families_declare_needs_channels():
    assert PREMIUM_FAMILIES["rqa"].needs_channels == 1


@pytest.fixture
def pair_family():
    fam = PremiumFamily(
        name="pairtest",
        featurize=lambda w: np.array([float(np.asarray(w)[1].mean())]),
        feature_names=["pair_mean1"],
        cost_note="O(1) test-only",
        needs_channels=2,
    )
    PREMIUM_FAMILIES["pairtest"] = fam
    yield fam
    del PREMIUM_FAMILIES["pairtest"]


def _bank_1ch(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    for cls in ("A", "B"):
        for r in range(3):
            np.save(tmp_path / f"{cls}_{r}.npy",
                    rng.standard_normal(6 * W).astype(np.float64))
    return load_bank(str(tmp_path), label_by="prefix")


def test_distill_refuses_pair_family_on_single_channel_bank(tmp_path, pair_family):
    """SPEC decision 3: a family declaring needs_channels=2 on a 1-channel bank
    must fail LOUDLY (SystemExit naming the channel counts), never silently."""
    bank = _bank_1ch(tmp_path)
    with pytest.raises(SystemExit, match=r"(?s)pairtest.*2.*1"):
        distill(bank, C=50, kmax=3, thr=0.005, n_perm=5, trees=10,
                cand=10, null_check=False, premium=("pairtest",))


# --------------------------------------------------------- spec.json channels
def test_spec_channels_roundtrip_and_backward_compat(tmp_path):
    spec = FeatureSpec(programs=["std(id(id(x)))"], classes=["A", "B"],
                       channels=["ps1", "ts1"])
    p = tmp_path / "spec.json"
    spec.save(str(p))
    back = FeatureSpec.load(str(p))
    assert back.channels == ["ps1", "ts1"]
    # old spec.json without the channels key must still load as 1-channel
    d = json.load(open(p))
    del d["channels"]
    json.dump(d, open(p, "w"))
    old = FeatureSpec.load(str(p))
    assert old.channels == []
    w = np.random.default_rng(0).standard_normal(W)
    assert old.featurize(w).shape == (1,)


def test_spec_featurize_multichannel_base_uses_channel0():
    """Primary-channel convention: base grammar programs see only channel 0."""
    spec1 = FeatureSpec(programs=["std(id(id(x)))", "acf1(diff(id(x)))"])
    spec2 = FeatureSpec(programs=list(spec1.programs), channels=["c0", "c1"])
    w2d = np.stack(window(_mc_signal(2))[0])
    np.testing.assert_array_equal(spec2.featurize(w2d), spec1.featurize(w2d[0]))


def test_spec_featurize_fails_closed_on_channel_mismatch():
    spec = FeatureSpec(programs=["std(id(id(x)))"], channels=["c0", "c1", "c2"])
    w2d = window(_mc_signal(2))[0]          # 2 channels, spec wants 3
    with pytest.raises(ValueError, match=r"3.*2|2.*3"):
        spec.featurize(w2d)
    with pytest.raises(ValueError):
        spec.featurize(np.zeros(W))          # 1-D window into a channel spec


# ------------------------------------------------------- coherence family
from signalmap.premium import coherence_feature_names, coherence_features


def _coupled_window(n_ch=3, n=W, seed=0, coupled=True):
    """ch0 = noise; other channels share ch0 (coupled) or are independent."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    rows = [base]
    for _ in range(n_ch - 1):
        noise = rng.standard_normal(n)
        rows.append(0.8 * base + 0.2 * noise if coupled else noise)
    return np.stack(rows)


def test_coherence_product_config_is_c128b2():
    """The product config is FIXED to the coherence_fair CONFIRMED-aug config
    (HYD-cooler, logs/coherence_fair.csv): nperseg=128, n_bands=2. No grid —
    same honesty rule as the RQA m3-tau5 default."""
    import inspect
    sig = inspect.signature(coherence_features)
    assert sig.parameters["nperseg"].default == 128
    assert sig.parameters["n_bands"].default == 2


def test_coherence_features_shape_names_range():
    w = _coupled_window(n_ch=3)
    v = coherence_features(w)
    names = coherence_feature_names(3)
    assert names == ["coh_ch1_b0", "coh_ch1_b1", "coh_ch2_b0", "coh_ch2_b1"]
    assert v.shape == (len(names),)
    assert np.all(np.isfinite(v)) and np.all(v >= 0.0) and np.all(v <= 1.0)
    np.testing.assert_allclose(v, coherence_features(w))  # deterministic


def test_coherence_separates_coupled_from_independent():
    hi = coherence_features(_coupled_window(coupled=True)).mean()
    lo = coherence_features(_coupled_window(coupled=False, seed=1)).mean()
    assert hi > lo + 0.3, (hi, lo)


def test_coherence_parity_with_coherence_fair():
    """Parity pin against the research implementation whose config was
    CONFIRMED (coherence_fair.coh_feats): pairs (0, j) must match exactly."""
    import os, sys
    factory = os.path.join(os.path.dirname(__file__), "..", "..",
                           "research", "factory")
    sys.path.insert(0, os.path.abspath(factory))
    try:
        from coherence_fair import coh_feats
    except Exception:
        pytest.skip("research/factory coherence_fair not importable here")
    finally:
        sys.path.pop(0)
    X2 = _coupled_window(n_ch=2, seed=2)
    ref2, _ = coh_feats([(X2, "A", 0)], 128, 2)
    np.testing.assert_allclose(coherence_features(X2), ref2[0])
    # C=3: coh_feats orders pairs (0,1),(0,2),(1,2) -> ours == the (0,j) prefix
    X3 = _coupled_window(n_ch=3, seed=3)
    ref3, _ = coh_feats([(X3, "A", 0)], 128, 2)
    np.testing.assert_allclose(coherence_features(X3), ref3[0][:4])


def test_coherence_registry_entry():
    fam = PREMIUM_FAMILIES["coherence"]
    assert fam.needs_channels == 2
    assert fam.featurize is coherence_features
    assert "c128b2" in fam.cost_note or "128" in fam.cost_note
    v = fam.featurize(_coupled_window(n_ch=2))
    assert v.shape == (len(coherence_feature_names(2)),)


# ------------------------------------------------- report verdict clarity
def test_report_explains_fail_plus_included(tmp_path):
    """When the base gate FAILs but a premium family is INCLUDED (signal lives
    only in the premium features), the receipt must say the two verdicts have
    different scopes instead of looking contradictory."""
    rng = np.random.default_rng(20)
    for cls in ("A", "B"):
        for r in range(3):
            n = 6 * W
            ch0 = rng.standard_normal(n)
            other = rng.standard_normal(n)
            ch1 = 0.8 * ch0 + 0.2 * other if cls == "B" else other
            np.save(tmp_path / f"{cls}_{r}.npy", np.stack([ch0, ch1]))
    bank = load_bank(str(tmp_path), label_by="prefix", multichannel=True)
    res = distill(bank, C=50, kmax=3, thr=0.005, n_perm=10, trees=25,
                  cand=15, null_check=False, premium=("coherence",))
    assert res.passed is False and res.spec.premium == ["coherence"]
    assert "verdict above gates the BASE selection" in res.report


# ------------------------------------------------------ multichannel ingest
def test_load_bank_multichannel_npy_both_orientations(tmp_path):
    """2-D .npy: orientation heuristic = the longer axis is time; (C, n) and
    (n, C) recordings load to the same (C, 1024) windows."""
    X = _mc_signal(n_ch=2, n=3 * W, seed=5)
    np.save(tmp_path / "A_0.npy", X)          # (C, n)
    np.save(tmp_path / "A_1.npy", X.T)        # (n, C)
    np.save(tmp_path / "B_0.npy", _mc_signal(n_ch=2, n=3 * W, seed=6))
    np.save(tmp_path / "B_1.npy", _mc_signal(n_ch=2, n=3 * W, seed=7))
    bank = load_bank(str(tmp_path), label_by="prefix", multichannel=True)
    assert bank.channels == ["ch0", "ch1"]
    assert all(w.shape == (2, W) for w in bank.windows)
    a0 = [w for w, g in zip(bank.windows, bank.g) if g == 0]
    a1 = [w for w, g in zip(bank.windows, bank.g) if g == 1]
    for u, v in zip(a0, a1):                  # same signal, both orientations
        np.testing.assert_array_equal(u, v)


def test_quasi_square_npy_warns_on_ambiguous_axis(tmp_path):
    """When a 2-D .npy is near-square the longer-axis heuristic is a coin flip;
    load_bank must warn (channel_axis unset) so a silent inversion is visible.
    A clearly rectangular recording and an explicit channel_axis stay silent
    (L2 review minor #3)."""
    near_sq = np.random.default_rng(0).standard_normal((W + 40, W))   # (1064, 1024)
    np.save(tmp_path / "A_0.npy", near_sq)
    with pytest.warns(UserWarning, match="near-square"):
        load_bank(str(tmp_path), label_by="prefix", multichannel=True)
    # explicit axis silences the ambiguity
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_bank(str(tmp_path), label_by="prefix", multichannel=True,
                  channel_axis=1)


def test_rectangular_npy_does_not_warn(tmp_path):
    rect = _mc_signal(n_ch=2, n=3 * W, seed=1)                        # (2, 3072)
    np.save(tmp_path / "A_0.npy", rect)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_bank(str(tmp_path), label_by="prefix", multichannel=True)


def test_load_bank_multichannel_channel_axis_override(tmp_path):
    X = _mc_signal(n_ch=3, n=2 * W, seed=8)
    np.save(tmp_path / "A_0.npy", X)
    np.save(tmp_path / "B_0.npy", _mc_signal(n_ch=3, n=2 * W, seed=9))
    bank = load_bank(str(tmp_path), label_by="prefix", multichannel=True,
                     channel_axis=0)
    assert len(bank.channels) == 3
    assert all(w.shape == (3, W) for w in bank.windows)


def test_load_bank_multichannel_csv_named_columns(tmp_path):
    """CSV with header -> channels = named columns (multichannel.load_channels
    reuse: NaN rows drop as a unit, synchrony preserved)."""
    rng = np.random.default_rng(10)
    for cls in ("A", "B"):
        for r in range(2):
            n = 2 * W + 3
            a, b = rng.standard_normal(n), rng.standard_normal(n)
            lines = ["ps1,ts1"]
            for i in range(n):
                if i in (5, 99, 2 * W):       # NaN rows must drop as a unit
                    lines.append(f",{b[i]}")
                else:
                    lines.append(f"{a[i]},{b[i]}")
            (tmp_path / f"{cls}_{r}.csv").write_text("\n".join(lines))
    bank = load_bank(str(tmp_path), label_by="prefix", multichannel=True)
    assert bank.channels == ["ps1", "ts1"]
    assert all(w.shape == (2, W) for w in bank.windows)
    assert bank.n_recordings == 4


def test_load_bank_multichannel_rejects_mismatched_channel_counts(tmp_path):
    np.save(tmp_path / "A_0.npy", _mc_signal(n_ch=2, n=2 * W))
    np.save(tmp_path / "B_0.npy", _mc_signal(n_ch=3, n=2 * W))
    with pytest.raises(SystemExit, match=r"channel"):
        load_bank(str(tmp_path), label_by="prefix", multichannel=True)


def test_load_bank_multichannel_rejects_unsupported_formats(tmp_path):
    (tmp_path / "A_0.txt").write_text("1.0 2.0\n3.0 4.0\n")
    np.save(tmp_path / "B_0.npy", _mc_signal(n_ch=2, n=2 * W))
    with pytest.raises(SystemExit, match=r"\.txt|multi-?channel"):
        load_bank(str(tmp_path), label_by="prefix", multichannel=True)


def test_run_cli_forwards_multichannel(tmp_path):
    """CLI surface: run_cli(multichannel=True) loads the 2-D bank and the saved
    spec.json carries the channel names."""
    from signalmap.distill import run_cli
    bankdir = tmp_path / "bank"
    bankdir.mkdir()
    rng = np.random.default_rng(12)
    for cls, f in (("A", 0.01), ("B", 0.3)):
        for r in range(3):
            n = 4 * W
            ch0 = np.sin(2 * np.pi * f * np.arange(n)) + 0.1 * rng.standard_normal(n)
            np.save(bankdir / f"{cls}_{r}.npy", np.stack([ch0, rng.standard_normal(n)]))
    res = run_cli(str(bankdir), "prefix", 50, str(tmp_path / "spec.json"),
                  n_perm=5, trees=10, multichannel=True)
    assert res.spec.channels == ["ch0", "ch1"]
    assert json.load(open(tmp_path / "spec.json"))["channels"] == ["ch0", "ch1"]


def test_cli_parser_accepts_multichannel_flags():
    from signalmap.cli import build_parser
    args = build_parser().parse_args(
        ["distill", "--bank", "bank", "--multichannel", "--channel-axis", "0",
         "--premium", "coherence"])
    assert args.multichannel is True
    assert args.channel_axis == 0


def test_distill_multichannel_writes_channels_into_spec(tmp_path):
    """distill on a multi-channel bank: base grammar sees only channel 0, the
    spec carries the channel names, featurize round-trips a 2-D window."""
    rng = np.random.default_rng(11)
    for cls in ("A", "B"):
        for r in range(3):
            n = 4 * W
            ch0 = (np.sin(2 * np.pi * (0.01 if cls == "A" else 0.3)
                          * np.arange(n)) + 0.1 * rng.standard_normal(n))
            np.save(tmp_path / f"{cls}_{r}.npy", np.stack([ch0,
                    rng.standard_normal(n)]))
    bank = load_bank(str(tmp_path), label_by="prefix", multichannel=True)
    res = distill(bank, C=50, kmax=3, thr=0.005, n_perm=5, trees=10,
                  cand=10, null_check=False)
    assert res.spec.channels == ["ch0", "ch1"]
    v = res.spec.featurize(bank.windows[0])
    assert v.shape == (len(res.spec.programs),)
    assert res.nested_acc > 0.8              # class signal lives in channel 0
