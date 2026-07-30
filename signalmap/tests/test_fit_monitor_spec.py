"""CLI gap closure (NEXT_SESSION_PLAN PRIO 1.2): `signalmap fit`/`monitor`
with a spec.json backend. Before this, DistilledDetector was library-only —
the E2E chain existed in tests but not on the product surface.

Product surface:
    signalmap fit --spec spec.json --bank healthy_dir/ --out det.json
    signalmap monitor --detector det.json --bank stream_dir/
The detector threshold is calibrated from the healthy envelope at fit time
(DistilledDetector.fit semantics); monitor scores every 1024-window of every
recording and reports per-recording alert rates."""
import json

import numpy as np
import pytest

from signalmap.distill import DistilledDetector, FeatureSpec, W


def _write_recordings(d, kind, n_rec=2, seed=0):
    """Healthy = low-freq tonal; faulty = high-freq + bursts (separable via
    the std(diff) family of base programs)."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 4 * W
    t = np.arange(n)
    for r in range(n_rec):
        if kind == "healthy":
            s = np.sin(2 * np.pi * 0.01 * t) + 0.1 * rng.standard_normal(n)
        else:
            s = (np.sin(2 * np.pi * 0.30 * t)
                 + 4.0 * (rng.random(n) < 0.02) * rng.standard_normal(n)
                 + 0.1 * rng.standard_normal(n))
        np.save(d / f"{kind}_{r}.npy", s.astype(np.float64))


@pytest.fixture
def spec_path(tmp_path):
    spec = FeatureSpec(programs=["acf1(id(id(x)))", "specratio(id(id(x)))"],
                       classes=["healthy", "faulty"])
    p = tmp_path / "spec.json"
    spec.save(str(p))
    return str(p)


def test_fit_spec_backend_writes_calibrated_detector(tmp_path, spec_path):
    from signalmap.monitor import fit_spec_backend
    _write_recordings(tmp_path / "healthy", "healthy")
    det = fit_spec_backend(spec_path, str(tmp_path / "healthy"),
                           str(tmp_path / "det.json"))
    assert isinstance(det, DistilledDetector)
    back = DistilledDetector.load(str(tmp_path / "det.json"))
    assert back.threshold == det.threshold > 0
    # quiet on healthy windows, alert on faulty ones
    _write_recordings(tmp_path / "faulty", "faulty", seed=1)
    from signalmap.distill import load_bank
    h = load_bank(str(tmp_path / "healthy"), label_by="stem")
    f = load_bank(str(tmp_path / "faulty"), label_by="stem")
    assert np.mean([back.alert(w) for w in f.windows]) > 0.8
    assert np.mean([back.alert(w) for w in h.windows]) < 0.3


def test_monitor_spec_backend_reports_alert_rates(tmp_path, spec_path):
    from signalmap.monitor import fit_spec_backend, monitor_spec_backend
    _write_recordings(tmp_path / "healthy", "healthy")
    _write_recordings(tmp_path / "faulty", "faulty", seed=1)
    fit_spec_backend(spec_path, str(tmp_path / "healthy"),
                     str(tmp_path / "det.json"))
    bad = monitor_spec_backend(str(tmp_path / "det.json"), str(tmp_path / "faulty"))
    ok = monitor_spec_backend(str(tmp_path / "det.json"), str(tmp_path / "healthy"))
    assert bad["n"] > 0 and bad["rate"] > 0.8
    assert ok["rate"] < 0.3
    assert "per_recording" in bad and len(bad["per_recording"]) == 2


def test_cli_fit_monitor_accept_spec_backend_flags(tmp_path, spec_path):
    from signalmap.cli import build_parser
    p = build_parser()
    a = p.parse_args(["fit", "--spec", spec_path, "--bank", "h/", "--out", "d.json"])
    assert a.spec == spec_path and a.bank == "h/"
    m = p.parse_args(["monitor", "--detector", "d.json", "--bank", "s/"])
    assert m.detector == "d.json" and m.bank == "s/"


def test_cli_fit_spec_requires_bank(tmp_path, spec_path):
    """--spec without --bank must fail loudly, not fall through to parquet."""
    from signalmap.cli import build_parser, cmd_fit
    args = build_parser().parse_args(["fit", "--spec", spec_path,
                                      "--out", str(tmp_path / "d.json")])
    with pytest.raises(SystemExit, match="bank"):
        cmd_fit(args)


def test_cli_fit_monitor_spec_end_to_end(tmp_path, spec_path, capsys):
    """The two commands as the user runs them (via cmd_ handlers)."""
    from signalmap.cli import build_parser, cmd_fit, cmd_monitor
    _write_recordings(tmp_path / "healthy", "healthy")
    _write_recordings(tmp_path / "faulty", "faulty", seed=1)
    p = build_parser()
    out = str(tmp_path / "det.json")
    cmd_fit(p.parse_args(["fit", "--spec", spec_path,
                          "--bank", str(tmp_path / "healthy"), "--out", out]))
    assert json.load(open(out))["threshold"] > 0
    cmd_monitor(p.parse_args(["monitor", "--detector", out,
                              "--bank", str(tmp_path / "faulty")]))
    assert "alert" in capsys.readouterr().out.lower()
