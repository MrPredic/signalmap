"""The synthetic demo path — the first thing a new user runs, so it must decode."""
import numpy as np

from signalmap import simulate
from signalmap.dsp import raw_to_features
from signalmap.frame import decode


def test_simulated_frame_round_trips_through_the_wire_decoder():
    samples = simulate.ordinary(seq=3)
    frame = decode(simulate.make_raw_frame(node_id=9, seq=3, samples=samples))
    assert (frame.node_id, frame.seq, frame.sr_hz) == (9, 3, simulate.SR)
    np.testing.assert_array_equal(frame.payload, samples)


def test_injected_anomaly_carries_more_raw_energy_than_ordinary_traffic():
    # The demo's claim is that the planted frames stand out on raw energy; if
    # that stops holding, the demo silently stops demonstrating anything.
    np.random.seed(0)
    ordinary = [raw_to_features(simulate.ordinary(s).astype(np.float32),
                                simulate.SR).energy_rms for s in range(20)]
    anomalous = [raw_to_features(simulate.anomaly(s).astype(np.float32),
                                 simulate.SR).energy_rms for s in range(5)]
    assert min(anomalous) > max(ordinary)


def test_demo_main_runs_end_to_end_and_reports_the_planted_frames(capsys):
    simulate.main()
    out = capsys.readouterr().out
    assert "ANOMALY" in out
    assert out.count("ANOMALY") >= 1
