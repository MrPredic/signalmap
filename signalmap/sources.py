"""Built-in Sources — anything that produces raw Frames.

Add your own by subclassing nothing — just satisfy the Source protocol and
@register("source", "yourname"). Salvaged hardware, USB capture, public dataset
replay: all plug in here.
"""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .core import register
from .frame import Frame, decode


def _mk(samples: np.ndarray, node_id: int, seq: int, sr_hz: int,
        sensor_class: int) -> Frame:
    s = np.asarray(samples, dtype=np.int16)
    return Frame(False, node_id, seq, int(time.time() * 1e6), sr_hz, len(s), s,
                 sensor_class)


@register("source", "sim")
class SimulatorSource:
    """Synthetic multi-domain signals (8 sensor classes) with injected
    novel-effect anomalies. Zero hardware — validates the whole pipeline."""

    def __init__(self, count: int = 200, anomaly_rate: float = 0.03, seed: int = 7):
        self.count = count
        self.anomaly_rate = anomaly_rate
        self.rng = np.random.default_rng(seed)

    def frames(self) -> Iterator[Frame]:
        from .synth import SensorClass, _SR, anomaly, generate
        domains = [c for c in SensorClass if c != SensorClass.UNKNOWN]
        for seq in range(self.count):
            if self.rng.random() < self.anomaly_rate:
                cls = self.rng.choice(domains)
                yield _mk(anomaly(self.rng), 0, seq, 16000, int(cls))
            else:
                cls = self.rng.choice(domains)
                yield _mk(generate(cls, self.rng), 0, seq, _SR[cls], int(cls))


@register("source", "replay")
class ReplaySource:
    """Replay a recorded/synthetic Parquet dataset (label, sr_hz, samples,
    optional sensor_class). Lets you re-run real captures through the pipeline."""

    def __init__(self, path: str):
        self.path = path

    def frames(self) -> Iterator[Frame]:
        import pyarrow.parquet as pq
        t = pq.read_table(self.path)
        cols = t.column_names
        srs = t.column("sr_hz").to_pylist()
        blobs = t.column("samples").to_pylist()
        classes = t.column("sensor_class").to_pylist() if "sensor_class" in cols else [0] * len(srs)
        for i, (b, sr, c) in enumerate(zip(blobs, srs, classes)):
            s = np.frombuffer(b, dtype="<i2")
            yield _mk(s, 0, i, sr, int(c))


@register("source", "mic")
class MicSource:
    """Laptop microphone = a free 44.1 kHz ADC node. Raw, no filtering."""

    def __init__(self, sr: int = 44100, count: int = 200, frame_n: int = 512):
        self.sr, self.count, self.frame_n = sr, count, frame_n

    def frames(self) -> Iterator[Frame]:
        import sounddevice as sd
        seq = 0
        with sd.InputStream(samplerate=self.sr, channels=1, dtype="float32") as st:
            while seq < self.count:
                block, _ = st.read(self.frame_n)
                s = np.clip(block[:, 0] * 2047.0, -2048, 2047).astype(np.int16)
                yield _mk(s, 1001, seq, self.sr, 0)
                seq += 1


@register("source", "mqtt")
class MqttSource:
    """Subscribe to live nodes (ESP32 / salvaged sensors) over MQTT."""

    def __init__(self, broker: str = "localhost", topic: str = "signals/+/raw",
                 count: int = 1000):
        self.broker, self.topic, self.count = broker, topic, count

    def frames(self) -> Iterator[Frame]:
        import queue
        import paho.mqtt.client as mqtt
        q: queue.Queue = queue.Queue()
        cli = mqtt.Client()
        cli.on_message = lambda c, u, msg: q.put(bytes(msg.payload))
        cli.connect(self.broker, 1883, 60)
        cli.subscribe(self.topic)
        cli.loop_start()
        try:
            for _ in range(self.count):
                try:
                    yield decode(q.get(timeout=30))
                except (ValueError, queue.Empty):
                    continue
        finally:
            cli.loop_stop()
