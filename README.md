<h1 align="center">SignalMap</h1>
<p align="center"><b>Map the unknown in any signal.</b></p>
<p align="center">An open, sensor-agnostic platform that streams raw signals from <i>any</i> sensor —
including salvaged e-waste hardware — embeds them with unsupervised learning, and
lets you explore the latent landscape for patterns nobody labeled yet.</p>
<p align="center">
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
<img alt="Status" src="https://img.shields.io/badge/status-alpha-orange">
</p>

---

## The vision

Every material, every mechanism, every environment radiates signals — vibration,
light, heat, fields, triboelectric charge. Almost all of it is thrown away
because no standard tells us what to measure. **SignalMap refuses to throw it
away.** It ingests raw, unfiltered signal streams from the cheapest possible
sensors, maps them into a shared latent space with no DIN/ISO labels, and surfaces
the **outliers** — the signatures that don't fit anything seen before.

The north star: discover unexpected physical effects (vibration energy
harvesting, raindrop triboelectricity, thermal anomalies) and ultimately **new
material properties and new norms** — in unconventional material mixtures nobody
thought to characterize. One open platform, any sensor, even sensors pulled from
the trash.

## What works today (verified) vs. the research frontier

We keep this line bright on purpose — bold mission, honest maturity.

| Capability | Status |
|---|---|
| Sensor-agnostic ingest (raw frames, any source) | ✅ working |
| Pluggable pipeline: Source → Transform → Model → Sink | ✅ working |
| Unsupervised embedding (Conv-AE) + anomaly scoring | ✅ working |
| Cross-domain proof in **simulation**: one model separates 8 sensor domains it was never told about (76% NN purity) | ✅ working, synthetic |
| **Unsupervised fault detection on REAL sensor data** — trained on healthy data only, scores held-out faults. **CWRU bearing dataset: ROC-AUC ≈ 1.00** (recon-error and raw-energy each AUC 1.00) | ✅ **validated on real data** |
| Discovery of *genuinely new / unknown* effects, material properties or norms | 🔬 **research goal — not yet demonstrated. An anomaly is a hypothesis, never a discovery.** |

> The **machinery is validated**: on real bearing-vibration data it separates
> faults from healthy with AUC ≈ 1.0, fully unsupervised — reproduce with
> `signalmap benchmark`. What is **not** yet shown is discovery of *new* physics:
> detecting a known fault type ≠ discovering an unknown one. Treat any anomaly as
> a hypothesis to be physically validated, never as fact.

> ⚠️ CWRU is a deliberately clean, well-separated benchmark — AUC 1.0 there proves
> the pipeline works end-to-end on real signals, not that hard real-world cases
> are solved. Harder datasets are on the roadmap.

## Design principle: bias-free by construction
Every "normalization" is an assumption, and assumptions hide the unexpected:
- **Raw int16 ADC** from the edge — no filtering, AGC, DC-removal, scaling.
- **Full spectrum** — no frequency cropping.
- **Raw amplitude is signal, not nuisance** — kept as a separate energy scalar.
- **Sensor class is metadata only** — never fed to the model.
- **Gaps are data** — sample loss is reported, never interpolated.

## Quick start (no hardware)
```bash
pip install -e .[all]
signalmap plugins                                   # see everything pluggable
signalmap universal                                 # cross-domain proof + HTML map
signalmap benchmark                                 # ROC-AUC on a synthetic PdM set
signalmap train --synthetic 2000 --epochs 30
signalmap run --source sim --weights artifacts/model.pt --sink stdout --limit 20
pytest -q
```

## Validate on real data (the litmus test)
```bash
# any recording -> frames -> unsupervised benchmark. Example with CWRU bearings:
signalmap ingest-file healthy.wav --label normal        --out data/real.parquet
signalmap ingest-file faulty.wav  --label ANOMALY_fault --out data/real.parquet
signalmap benchmark --dataset data/real.parquet --anomaly-label ANOMALY
```

## The pluggable core
A pipeline is four swappable stages. Implement a tiny Protocol, `@register(...)`,
and it shows up in the CLI by name:
```
Source ──▶ Transform ──▶ Model ──▶ Sink
(mic,      (fft, ...)    (auto-     (stdout, parquet,
 replay,                  encoder)   questdb, qdrant)
 mqtt, sim)
```
```bash
signalmap run --source replay --dataset data/x.parquet \
  --transform fft --model autoencoder --weights artifacts/model.pt \
  --sink parquet --sink qdrant
```
Add a webcam, an SDR, a salvaged piezo on an ESP32, a new embedding model, a new
database — without touching the core. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
and [CONTRIBUTING.md](CONTRIBUTING.md).

## Recycle the trash (0-€ sensors)
SignalMap is built to run on salvaged hardware. A laptop mic is a free 44.1 kHz
ADC; a DVD pickup head is a published 24 V/g accelerometer; any LED is a
photodiode; a speaker is a microphone; an old phone is a full sensor suite.
The transducer comes from e-waste, an MCU or host (Rust) is the bridge, and it
all becomes the same frames. Catalog + safety notes in the project docs.

## Architecture
```
Edge (Rust no_std, ESP32-S3) ── MQTT ──▶ Ingest ──▶ Transform(FFT) ──▶ Conv-AE
  raw ADC / salvaged transducer            │                              │
  custom binary frame v1                    ▼                              ▼
  (schema/frame.md)                     QuestDB (raw TS)        embedding + score
                                                                     │
                                                              Qdrant (vectors)
                                                                     │
                                                          FastAPI /map /anomalies
```

## Roadmap
- [x] Pluggable Source/Transform/Model/Sink core + CLI
- [x] Cross-domain unsupervised proof (simulation)
- [x] Real-recording ingestion (WAV/CSV/NPY) + ROC-AUC benchmark
- [x] Validated on **real** public sensor data (CWRU bearing, AUC ≈ 1.0)
- [ ] Harder real datasets (MIMII, IMS, MAFAULDA) + leaderboard
- [ ] HDBSCAN auto-clustering + cluster naming
- [ ] Host Rust capture adapters (audio/camera/SDR)
- [ ] Live latent-novelty (Qdrant kNN) in the streaming path
- [ ] Energy-harvesting measurement rig (quantify, not just classify)

## License
Apache-2.0. Contributions welcome — keep the bias-free principle intact.
