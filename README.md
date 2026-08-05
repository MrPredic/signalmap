<h1 align="center">SignalMap</h1>
<p align="center"><b>Map the unknown in any signal.</b></p>
<p align="center">An open, sensor-agnostic platform that streams raw signals from <i>any</i> sensor —
including salvaged e-waste hardware — embeds them with unsupervised learning, and
lets you explore the latent landscape for patterns nobody labeled yet.</p>
<p align="center"><i>Its distinctive piece: <b>distill</b> — a feature-selection
step with a capacity gate and a statistical receipt that <b>refuses</b> premium
feature families when they don't generalize, instead of silently adding them.</i></p>
<p align="center">
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
<img alt="Status" src="https://img.shields.io/badge/status-0.5.0-orange">
</p>

---

## Installation
```bash
python3 -m pip install 'signalmap[all]'   # quote it so zsh doesn't glob the brackets
signalmap plugins                         # confirm it installed: lists everything pluggable
```
Or from the clone, editable, so `signalmap` tracks your checkout:
```bash
git clone https://github.com/MrPredic/signalmap.git
cd signalmap
python3 -m pip install -e '.[all]'
```
Extras: `[all]` pulls in everything below; `[distill]` alone is enough for
`distill`/`fit`/`monitor` on `.npy`/`.csv` banks (scipy + scikit-learn); parquet
I/O additionally needs `pyarrow` (bundled in `[all]`).

## Table of contents
- [The vision](#the-vision)
- [What works today vs. the research frontier](#what-works-today-verified-vs-the-research-frontier)
- [Real use case in two commands](#real-use-case-in-two-commands)
- [Distill: per-domain features with a receipt](#distill-per-domain-features-with-a-receipt)
- [Signed verdicts, verifiable without us](#signed-verdicts-verifiable-without-us)
- [Design principle: bias-free by construction](#design-principle-bias-free-by-construction)
- [Quick start (no hardware)](#quick-start-no-hardware)
- [Validate on real data (the litmus test)](#validate-on-real-data-the-litmus-test)
- [The pluggable core](#the-pluggable-core)
- [Recycle the trash (0-€ sensors)](#recycle-the-trash-0--sensors)
- [Architecture](#architecture)
- [Cross-modal discovery (experimental, honest)](#cross-modal-discovery-experimental-honest)
- [Compositional search (experimental)](#compositional-search-experimental)
- [Roadmap](#roadmap)
- [License](#license)

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

## Real use case in two commands
**No fault labels needed** — you only supply (or mark) healthy data; anomalies
are never labeled. Fit a detector on healthy operation, then monitor for
deviations. The same two commands work for any recorded signal — vibration,
acoustics, current (see [Installation](#installation) first):
```bash
signalmap fit     --dataset healthy.parquet --healthy-label normal --out detector.pt
signalmap monitor --source replay --dataset live.parquet --detector detector.pt
```

Model artifacts (`.pt`) are a trust boundary: load only files you trust and
keep them as tensor state dictionaries. SignalMap explicitly loads weights
with PyTorch's `weights_only=True`; arbitrary pickle objects are rejected.
**On real CWRU bearing data:** fit on 945 healthy frames → monitor 1183 frames →
**238/238 faults caught (100%), 2/945 false alarms (0.2%)**, fully unsupervised.
The *same* two commands flag injected faults in the synthetic set at 0 false
positives. That is the USP: zero-config unsupervised condition monitoring over
arbitrary recorded/raw signals, with an extensible adapter model for new sources
(including salvaged hardware — see the roadmap for capture-adapter status).

Reproduce the CWRU result yourself in one script, plus acoustic/electrical
recipes: see [examples/](examples/).

## Distill: per-domain features with a receipt
`signalmap distill` searches a compositional feature grammar for the handful of
programs that separate *your* recordings — and refuses to fool itself. The
search budget is cut by a **capacity gate** (`budget = C × n_recordings`,
C=50): below it the distilled set generalises, above it selection noise wins —
a coupling neither a fixed feature list (catch22) nor an ungated 7700-feature
sweep (hctsa) has. Every run emits a human-readable **gauntlet receipt**:
leakage-free nested LOGO accuracy, a group-permutation p-value, a label-shuffle
null, and cost per window. The output `spec.json` plugs straight into the
`fit`/`monitor` deploy surface; the detector calibrates its alert threshold
from the healthy envelope at fit time, so it self-scales to features of any
magnitude.
```bash
# quote the extras so zsh doesn't glob the brackets; parquet banks also need pyarrow
python3 -m pip install 'signalmap[distill]'   # distill/fit/monitor on .npy/.csv banks (scipy + scikit-learn)
signalmap distill --bank recordings/ --label-by prefix --out artifacts/spec.json
```

**Premium families** (`--premium rqa,coherence`): specialised featurizers —
full-window recurrence quantification (RQA, O(n²) per window) and cross-channel
magnitude-squared coherence — run as opt-in *challengers* against the distilled
base selection, never inside the base grammar. The receipt prints a paired
bootstrap CI over the LOGO folds plus the real cost ratio (ms/window), and the
**champion rule** admits a family into `spec.json` only on a CI-solid win.

We preregistered these verdicts on public datasets before reading them out, and
both outcomes happened. The receipt *admitted* RQA on CWRU bearings (accuracy
0.914 → 0.980, paired +0.066, 95% CI [+0.033, +0.106], at ~585× the base cost)
and coherence on the UCI hydraulic rig (0.750 → 0.889, +0.139, CI [+0.042,
+0.236], ~81×). It *refused* RQA on CALCE batteries and on the same hydraulic
rig, and refused coherence on the UCI gas sensor array — in each case because
the CI over recordings did not clear zero. A tool that can say "this expensive
family does not pay for itself on your data" is the point; a receipt that only
ever says yes proves nothing. Note the two verdicts have different scopes: the
PASS/FAIL gate judges the base selection, the champion rule judges each premium
family — a bank can be base-FAIL with a family INCLUDED when the signal lives
only in the premium features (the receipt says so explicitly).

**Multi-channel banks** (`--multichannel`): recordings can be 2-D — a CSV with
a header of channel names, or a 2-D `.npy` (the longer axis is taken as time;
override with `--channel-axis`). Windows become synchronous `(C, 1024)` slices.
The base grammar always sees channel 0 only, so single-channel behaviour is
bit-for-bit unchanged; families that need cross-channel structure declare it
and are refused loudly on single-channel banks instead of silently degrading.

**Deploying a spec** is two commands, no labels needed:
```bash
signalmap fit --spec artifacts/spec.json --bank healthy_recordings/ --out det.json
signalmap monitor --detector det.json --bank incoming_recordings/
```
`fit` calibrates the alert threshold from the healthy envelope; `monitor`
scores every window and reports per-recording alert rates.

## Signed verdicts, verifiable without us
Every verdict-producing command — `distill`, `fit`, `monitor` — writes a signed
JSON receipt next to its output: the claim, the verdict
(`INCLUDED`/`EXCLUDED`/`PASS`/`REFUSED`), the evidence behind it, the sha256 of
every input, and an Ed25519 signature. The signing key lives in
`~/.signalmap/signing_key` (mode 0600) and never leaves the machine; only the
public key travels in the receipt.

The verifier is deliberately separate from the tool. `tools/verify_receipt.py`
imports **nothing** from signalmap — stdlib plus `cryptography` — so checking a
receipt never means running our code:
```bash
pip install signalmap
signalmap corpus                              # rebuild + list the shipped verdict corpus
python3 tools/verify_receipt.py research/factory/receipts/cwru_rqa.receipt.json
# PASS: … — verdict INCLUDED, integrity only
```
```text
EXCLUDED CALCE       rqa       research/factory/receipts/calce_rqa.receipt.json
EXCLUDED CWRU        envelope  research/factory/receipts/cwru_envelope.receipt.json
INCLUDED CWRU        rqa       research/factory/receipts/cwru_rqa.receipt.json
EXCLUDED GAS-id      coherence research/factory/receipts/gasid_coherence.receipt.json
INCLUDED HYD-cooler  coherence research/factory/receipts/hydcooler_coherence.receipt.json
EXCLUDED HYD-cooler  rqa       research/factory/receipts/hydcooler_rqa.receipt.json
EXCLUDED IMS         envelope  research/factory/receipts/ims_envelope.receipt.json
EXCLUDED MFPT        envelope  research/factory/receipts/mfpt_envelope.receipt.json
8 verdicts across 6 banks · 2 included with a cost receipt · 6 honest exclusions · 0 silent adoptions · offline verifiable
```
Those eight are the preregistered premium verdicts described above, shipped in
the repo as signed receipts. They are labelled `archive_signature`: the
statistics were decided in the preregistered runs, and the signature attests to
the origin and integrity of the transcription, **not** to a re-execution. Each
one pins the sha256 of the report it transcribes, and the test suite fails if a
shipped receipt drifts from its source.

What a receipt does not do: it does not make a verdict true. It makes a verdict
**attributable and tamper-evident** — flip one byte and verification fails.

## Design principle: bias-free by construction
Every "normalization" is an assumption, and assumptions hide the unexpected:
- **Raw int16 ADC** from the edge — no filtering, AGC, DC-removal, scaling.
  (`ingest-file` maps WAV/CSV that weren't captured as int16 into int16 range via
  a single global gain + DC-center — spectrum-preserving, no per-window scaling.)
- **Full spectrum** — no frequency cropping.
- **Raw amplitude is signal, not nuisance** — kept as a separate energy scalar.
- **Sensor class is metadata only** — never fed to the model.
- **Gaps are data** — sample loss is reported, never interpolated.

## Quick start (no hardware)
After [installing](#installation):
```bash
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

## Cross-modal discovery (experimental, honest)
Deploy many *different* sensors on one phenomenon and ask: which modalities are
**genuinely coupled**, versus merely correlated through a shared driver (time,
temperature, mains)? Naive correlation lies — almost everything correlates. We
remove known confounds (partial correlation) and keep only couplings that
*survive*, with a permutation p-value.
```bash
signalmap discover --naive            # the trap: flags confounds as "couplings"
signalmap discover --confound temp    # honest: confounded pairs collapse, real coupling survives
```
On the built-in ground-truth set, a temp-driven `vibration–em` pair (raw corr
0.82) is correctly **rejected** after conditioning on temperature, while a real
`heat→acoustic` coupling survives. A survivor is a **hypothesis for controlled
validation**, never a proven new effect — the instrument generates candidates,
nature certifies them.

## Compositional search (experimental)
`prove` searches a fixed, auditable vocabulary of sensor views, statistics and
cross-channel relations. It uses group-held-out scoring, group-label
permutations and a structure-preserving mechanism null. The output is a JSON
receipt; `candidate` is a lead, not a discovery claim.

```bash
signalmap prove --synthetic 96 --budget 24 --perms 50 \
  --out artifacts/composer_receipt.json
```

For real data, pass an `.npz` containing `X` (`N×channels×time`), `y`, and
`groups`. Only a receipt with independent replication should be promoted to a
production recipe.

## Roadmap
- [x] Pluggable Source/Transform/Model/Sink core + CLI
- [x] Cross-modal coupling discovery with confound ablation (`discover`)
- [x] Deployable detector: `fit` on healthy → `monitor` for anomalies (any recorded signal)
- [x] Cross-domain unsupervised proof (simulation)
- [x] Real-recording ingestion (WAV/CSV/NPY) + ROC-AUC benchmark
- [x] Validated on **real** public sensor data (CWRU bearing, AUC ≈ 1.0)
- [x] Signed, offline-verifiable verdict receipts + shipped verdict corpus
- [ ] Harder real datasets (MIMII, IMS, MAFAULDA) + leaderboard
- [ ] HDBSCAN auto-clustering + cluster naming
- [ ] Host Rust capture adapters (audio/camera/SDR)
- [ ] Live latent-novelty (Qdrant kNN) in the streaming path
- [ ] Energy-harvesting measurement rig (quantify, not just classify)

## License
Apache-2.0. Contributions welcome — keep the bias-free principle intact.
