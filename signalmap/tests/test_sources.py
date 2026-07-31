"""Source plugins. The mic and MQTT sources need hardware/a broker and are
covered by the smoke run in CI instead; this pins the two offline sources.
"""
import numpy as np

from signalmap.core import available
from signalmap.sources import SimulatorSource


def test_all_builtin_sources_are_discoverable_by_name():
    assert {"sim", "replay", "mic", "mqtt"} <= set(available("source"))


def test_sim_source_yields_the_requested_frame_count():
    frames = list(SimulatorSource(count=25).frames())
    assert len(frames) == 25
    assert [f.seq for f in frames] == list(range(25))


def test_sim_source_is_reproducible_for_a_seed():
    a = list(SimulatorSource(count=12, seed=3).frames())
    b = list(SimulatorSource(count=12, seed=3).frames())
    for fa, fb in zip(a, b):
        np.testing.assert_array_equal(fa.payload, fb.payload)
        assert fa.sensor_class == fb.sensor_class


def test_sim_source_spans_several_sensor_classes():
    classes = {f.sensor_class for f in SimulatorSource(count=200).frames()}
    assert len(classes) > 1, "the universal claim needs more than one modality"


def test_anomaly_rate_one_makes_every_frame_an_injected_anomaly():
    # The anomaly branch carries its own sample rate and generator; at rate 1.0
    # it must be the only branch taken.
    quiet = [f for f in SimulatorSource(count=40, anomaly_rate=0.0, seed=1).frames()]
    loud = [f for f in SimulatorSource(count=40, anomaly_rate=1.0, seed=1).frames()]

    def rms(fr):
        return float(np.sqrt(np.mean(fr.payload.astype(np.float64) ** 2)))

    assert np.mean([rms(f) for f in loud]) > np.mean([rms(f) for f in quiet])
    assert all(f.sr_hz == 16000 for f in loud)


def test_frames_are_raw_not_spectra_so_the_dsp_stage_still_runs():
    assert all(not f.is_spectrum for f in SimulatorSource(count=10).frames())
