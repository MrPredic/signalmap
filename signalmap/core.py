"""SignalMap plugin core — the open, everything-pluggable foundation.

A pipeline is four swappable stages:

    Source  -> raw frames (bytes on the wire / Frame objects)
    Transform -> Frame to a numeric feature vector
    Model   -> feature vector to (embedding, anomaly score)
    Sink    -> do something with the result (store, print, emit)

Anything that satisfies these tiny Protocols plugs in: a microphone, a salvaged
piezo on an ESP32, a webcam, an SDR, a public dataset replay, a new embedding
model, a new database. Register a class with @register(...) and it becomes
available to the CLI by name. That is the whole extensibility story.

Nothing here knows about a specific sensor, material, or standard — by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from .frame import Frame


@dataclass
class Result:
    """One processed frame flowing out of a pipeline."""
    frame: Frame
    feature: np.ndarray
    embedding: np.ndarray
    score: float
    meta: dict


@runtime_checkable
class Source(Protocol):
    """Yields raw Frames from any origin (mic, MQTT, file, simulator, ...)."""
    def frames(self) -> Iterator[Frame]: ...


@runtime_checkable
class Transform(Protocol):
    """Frame -> feature vector. Must NOT inject domain/label knowledge."""
    def __call__(self, frame: Frame) -> np.ndarray: ...


@runtime_checkable
class Model(Protocol):
    """Feature -> (embedding, anomaly score)."""
    def process(self, feature: np.ndarray) -> tuple[np.ndarray, float]: ...


@runtime_checkable
class Sink(Protocol):
    """Consume a Result (store / print / forward). May be a no-op."""
    def emit(self, result: Result) -> None: ...
    def close(self) -> None: ...


# --- plugin registry -------------------------------------------------------
_REGISTRY: dict[str, dict[str, type]] = {
    "source": {}, "transform": {}, "model": {}, "sink": {}
}


def register(kind: str, name: str):
    """Class decorator: register a plugin under a kind+name for CLI lookup."""
    if kind not in _REGISTRY:
        raise ValueError(f"unknown plugin kind {kind!r}")

    def deco(cls):
        _REGISTRY[kind][name] = cls
        return cls
    return deco


def get(kind: str, name: str):
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        avail = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "(none)"
        raise SystemExit(f"no {kind} named {name!r}. available: {avail}")


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


# --- pipeline --------------------------------------------------------------
class Pipeline:
    def __init__(self, source: Source, transform: Transform, model: Model,
                 sinks: list[Sink]) -> None:
        self.source = source
        self.transform = transform
        self.model = model
        self.sinks = sinks

    def run(self, limit: int | None = None) -> int:
        n = 0
        try:
            for frame in self.source.frames():
                if frame.is_spectrum:
                    continue
                feat = self.transform(frame)
                emb, score = self.model.process(feat)
                res = Result(frame, feat, emb, score,
                             meta={"node_id": frame.node_id, "seq": frame.seq,
                                   "sensor_class": frame.sensor_class})
                for s in self.sinks:
                    s.emit(res)
                n += 1
                if limit and n >= limit:
                    break
        finally:
            for s in self.sinks:
                s.close()
        return n
