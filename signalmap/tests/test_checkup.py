"""`signalmap checkup` — the answer to "does this work on my recordings?".

An outside reader given only the README could not work out what to type with
their own files, and the study had already shown that the outcome swings from
clean separation to outright inversion depending on the data. checkup exists so
that question gets a verdict in one command instead of a guess.

These tests pin the three answers it must be able to give — separates,
inverted, refuses — and the refusals that come before any number is computed.
"""
import numpy as np
import pytest

from signalmap.checkup import checkup, render


def _ar1(phi, rng, n=8 * 1024):
    e = rng.normal(0, 1, n + 1000)
    x = np.empty(n + 1000)
    x[0] = e[0]
    for i in range(1, n + 1000):
        x[i] = phi * x[i - 1] + e[i]
    return x[1000:]


def _bank(tmp_path, healthy_phis, fault_phis, name="bank"):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, phi in enumerate(healthy_phis):
        np.save(root / f"normal_{i:03d}.npy", _ar1(phi, rng))
    for i, phi in enumerate(fault_phis):
        np.save(root / f"anomaly_{i:03d}.npy", _ar1(phi, rng))
    return str(root)


def test_separates_when_the_fault_is_far_outside_healthy(tmp_path):
    rng = np.random.default_rng(1)
    b = _bank(tmp_path, list(rng.uniform(0.2, 0.8, 16)), [0.97] * 8)
    res = checkup(b)
    assert res["verdict"] == "SEPARATES"
    assert res["direction"] == "aligned"
    assert res["ci_lo"] > 0.5


def test_reports_inversion_when_the_fault_sits_at_the_healthy_centre(tmp_path):
    """The valve case: an anomaly more stereotyped than normal operation.

    Needs both a wide healthy spread and enough recordings before the
    interval clears 0.5. Measured on the way here: at 16 recordings the
    estimate is already 0.22 but the upper bound sits at 0.508, and checkup
    refuses — the correct answer at that size, and the reason this test is
    built at 60/30 rather than tuned until it passes.
    """
    rng = np.random.default_rng(2)
    b = _bank(tmp_path, list(rng.uniform(0.05, 0.95, 60)), [0.5] * 30)
    res = checkup(b)
    assert res["verdict"] == "SEPARATES"
    assert res["direction"] == "inverted"
    assert res["ci_hi"] < 0.5
    assert "LOWER" in render(res)


def test_refuses_when_the_two_classes_are_the_same_process(tmp_path):
    """A real null: both classes carry the SAME parameters, only the noise
    differs. Reusing a slice of the healthy draw is not a null — the two
    halves then have genuinely different means and checkup rightly separates
    them, which is how this test was wrong the first time."""
    rng = np.random.default_rng(3)
    phis = list(rng.uniform(0.2, 0.8, 20))
    b = _bank(tmp_path, phis, phis)
    res = checkup(b)
    assert res["verdict"] == "REFUSED"
    assert res["direction"] == "undetermined"
    assert "does not tell your two classes apart" in render(res)


def test_refuses_before_computing_anything_without_a_second_class(tmp_path):
    rng = np.random.default_rng(4)
    b = _bank(tmp_path, list(rng.uniform(0.2, 0.8, 12)), [])
    res = checkup(b)
    assert res["verdict"] == "REFUSED"
    assert "one class" in res["reason"]
    assert "auc" not in res
    assert "could not run" in render(res)


def test_refuses_when_too_few_healthy_recordings_to_hold_any_out(tmp_path):
    rng = np.random.default_rng(5)
    b = _bank(tmp_path, list(rng.uniform(0.2, 0.8, 3)), [0.97] * 4)
    res = checkup(b)
    assert res["verdict"] == "REFUSED"
    assert "hold any of them out" in res["reason"]


def test_the_detector_is_never_scored_on_what_calibrated_it(tmp_path):
    rng = np.random.default_rng(6)
    b = _bank(tmp_path, list(rng.uniform(0.2, 0.8, 16)), [0.97] * 8)
    res = checkup(b)
    assert res["n_fit_recordings"] + res["n_eval_recordings"] == res["n_recordings"]
    assert res["n_eval_faulty"] == 8


def test_the_alarm_is_judged_on_recordings_not_windows(tmp_path):
    """The cut is calibrated on recording means, so the rates must count
    recordings — mixing the two units is what made an earlier version report
    a decided direction and no decided direction in the same output."""
    rng = np.random.default_rng(7)
    b = _bank(tmp_path, list(rng.uniform(0.2, 0.8, 16)), [0.97] * 8)
    res = checkup(b)
    a = res["alarm"]
    assert a is not None
    n_faulty = res["n_eval_faulty"]
    assert a["hit_rate"] * n_faulty == pytest.approx(round(a["hit_rate"] * n_faulty))


def test_render_never_claims_an_alarm_and_no_direction_at_once(tmp_path):
    rng = np.random.default_rng(8)
    for faults in ([0.97] * 8, [0.5] * 8):
        res = checkup(_bank(tmp_path, list(rng.uniform(0.2, 0.8, 16)), faults,
                            name=f"b{faults[0]}"))
        text = render(res)
        assert not ("ALARM READY" in text and "without a decided direction" in text)


# ------------------------------------------------------------ WAV recordings
def _write_wav(path, x, rate=16000, width=2, n_ch=1):
    import wave
    scale = {1: 127, 2: 32767, 4: 2 ** 31 - 1}[width]
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    y = np.clip(x / (np.max(np.abs(x)) or 1.0), -1, 1) * scale
    y = (y + 128 if width == 1 else y).astype(dt)
    if n_ch > 1:
        y = np.repeat(y[:, None], n_ch, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(n_ch); w.setsampwidth(width); w.setframerate(rate)
        w.writeframes(y.tobytes())


def test_checkup_reads_wav_banks(tmp_path):
    """Machine acoustics arrives as WAV — MIMII and DCASE ship nothing else —
    so a bank loader that could not read it turned away exactly the users the
    command is for."""
    root = tmp_path / "wav"
    root.mkdir()
    rng = np.random.default_rng(0)
    for i, phi in enumerate(rng.uniform(0.05, 0.95, 20)):
        _write_wav(root / f"normal_{i:03d}.wav", _ar1(phi, rng))
    for i in range(10):
        _write_wav(root / f"anomaly_{i:03d}.wav", _ar1(0.5, rng))
    res = checkup(str(root))
    assert res["n_recordings"] == 30
    assert res["verdict"] in ("SEPARATES", "REFUSED")
    assert "auc" in res


@pytest.mark.parametrize("width", [1, 2, 4])
def test_all_pcm_widths_load(tmp_path, width):
    from signalmap.distill import _read_recording
    rng = np.random.default_rng(1)
    p = tmp_path / f"w{width}.wav"
    _write_wav(p, _ar1(0.5, rng, n=4096), width=width)
    x = _read_recording(str(p))
    assert x.size == 4096 and np.isfinite(x).all() and x.std() > 0


def test_stereo_picks_the_requested_channel(tmp_path):
    from signalmap.distill import _read_recording
    rng = np.random.default_rng(2)
    p = tmp_path / "stereo.wav"
    _write_wav(p, _ar1(0.5, rng, n=4096), n_ch=2)
    assert _read_recording(str(p), column=0).size == 4096
    with pytest.raises(SystemExit, match="channel 5"):
        _read_recording(str(p), column=5)


def test_drivers_name_what_carried_the_verdict(tmp_path):
    """SEPARATES without naming what separated is a number to trust rather
    than a finding to check — and the per-feature directions are where the
    inversion becomes visible."""
    rng = np.random.default_rng(9)
    b = _bank(tmp_path, list(rng.uniform(0.05, 0.95, 40)), [0.97] * 20)
    res = checkup(b)
    assert res["verdict"] == "SEPARATES"
    d = res["drivers"]
    assert d and len(d) <= 5
    assert d == sorted(d, key=lambda x: -x["separation"])
    assert all({"program", "auc", "separation", "degenerate_on_healthy"} <= set(x)
               for x in d)
    assert d[0]["program"] in res["spec_programs"]
    assert d[0]["program"] in render(res)


def test_a_flat_feature_is_marked_rather_than_presented_as_a_driver(tmp_path):
    """std(x) is exactly 1.0 after windowing, so it can rank raw values and
    still contribute nothing to the score. The report must not let a reader
    mistake the first for the second."""
    rng = np.random.default_rng(10)
    b = _bank(tmp_path, list(rng.uniform(0.05, 0.95, 40)), [0.97] * 20)
    res = checkup(b)
    # render() shows the top three; only those can carry a visible mark.
    shown = res["drivers"][:3]
    flat_shown = [d for d in shown if d["degenerate_on_healthy"]]
    text = render(res)
    assert ("mute in the score" in text) == bool(flat_shown)
    assert all("degenerate_on_healthy" in d for d in res["drivers"])
