# Architecture

SignalMap is a thin, open core with everything interesting living in plugins.

## The pipeline
```
Source ──▶ Transform ──▶ Model ──▶ Sink(s)
```
- **Source** (`signalmap/sources.py`) — yields raw `Frame`s from any origin.
  Built-ins: `sim`, `replay`, `mic`, `mqtt`.
- **Transform** (`signalmap/transforms.py`) — `Frame -> feature vector`.
  Built-in: `fft` (full-band rfft magnitude, fixed bins, energy kept aside).
- **Model** (`signalmap/models.py`) — `feature -> (embedding, anomaly score)`.
  Built-in: `autoencoder` (Conv-AE, reconstruction error = score).
- **Sink** (`signalmap/sinks.py`) — consume `Result`. Built-ins: `stdout`,
  `parquet`, `questdb`, `qdrant`.

All four are tiny `Protocol`s defined in `signalmap/core.py`. A class becomes
available to the CLI by name via `@register(kind, name)`.

## The wire format
`schema/frame.md` — a versioned 28-byte binary header + raw payload. Identical
on the ESP32 firmware (`firmware/`) and every host source, so a laptop mic, a
salvaged piezo, and a fleet of MCUs are all just "nodes".

## The bias contract (non-negotiable)
The platform's value depends on NOT injecting prior knowledge:
1. Edge sends raw int16 — no filtering/AGC/DC-removal/scaling.
2. Transforms must not crop bands or discard information silently; any lossy
   step is optional and documented in `schema/frame.md`.
3. Raw amplitude/energy is preserved as signal, never normalized away.
4. `sensor_class` and any label are **metadata only** — never model input.
5. Sequence gaps are reported, never interpolated.

A plugin that violates this is a bug, not a feature.

## Data flow at scale
```
nodes ──MQTT──▶ ingest ──▶ QuestDB (raw time series)
                   └────▶ Transform ─▶ Model ─▶ Qdrant (embeddings + scores)
                                                   └─▶ FastAPI /map /anomalies
```
QuestDB/Qdrant are optional: the simulation proof and HTML maps run fully
embedded with no servers.

## Honesty boundary
The machinery (ingest → embed → anomaly) is verified. "Discovery of new
effects/materials" is a research goal validated by physical experiment, not by
the anomaly score alone. An outlier is a hypothesis, never a conclusion.

## Method modules and evidence

The scientific method layer is intentionally open-ended. A method module
declares a `MethodCapability` (family, hard technical requirements, cost, and
evidence status) and can be loaded into a `MethodCatalog` without changing the
core pipeline.

- `confirmed` methods have receipts that passed the required evidence gates.
- `candidate` methods are plausible but incomplete.
- `speculative` methods are allowed for exploratory work and remain visibly
  separate from confirmed results.

Routing only hard-excludes technically impossible methods, such as a
two-channel method on a one-channel source. Noise diagnostics may change
priority or budget, but do not erase a method's opportunity to discover
something unexpected. Method-specific evidence lives in extensible JSON; the
core requires stable identity, status, and replayable inputs.
