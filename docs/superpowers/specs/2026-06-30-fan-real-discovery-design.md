# SignalMap — First Real Cross-Modal Discovery Proof (Fan / CPU-Load)

**Date:** 2026-06-30
**Status:** Design approved, ready for implementation plan
**Scope:** Single, self-contained experiment. 0 € hardware. Local repo `/Users/macbook/signalmap`, not pushed.

## 1. Goal & Honest Claim

Produce the "visible proof" the project DD repeatedly names as the only real investment asset:
the discovery engine, run on **self-captured real multi-sensor data**, finds a **physical
cross-modal coupling**, separates it **confound-controlled** from a spurious one, and the
surviving coupling **tracks a controlled intervention** while the spurious one collapses.

**Claim boundary (non-negotiable):** what is demonstrated is an *unsupervised-discovered,
intervention-validated, real latent quantity / coupling* — **not** "new physics" and **not**
"new material law". This is the honest rung toward the vision (manufacture differentiated
knowledge), captured on real data instead of self-generated synthetics. Anything stronger does
not survive due diligence.

**Why this experiment is more than a smoke test (moat framing, revised 2026-06-30):** the moat
is **not** any single off-the-shelf algorithm (CCM is 2012, public — no moat). The moat is the
**novel synthesis of discovery methods that make currently-unobservable quantities observable
from cheap, noisy, heterogeneous multimodal sensors**, packaged as a usable engine, plus the
proprietary data corpus and validated track record that being early lets us accumulate. Academia
holds the pieces (SINDy, SciNet, Koopman, conservation-law discovery, deep-CCA) but all siloed,
single-modality, clean-data, researcher-only. Nobody fuses them for noisy cheap-sensor streams +
active intervention as OSS. Classical methods (correlation/anomaly/CCM) are accounting over
*known* variables; the vision needs methods that recover the variables, laws, invariants and
categories nature uses — which are in no label or norm. This experiment is the first testbed for
one such method beyond classical, not a correlation demo.

## 2. Phenomenon

Laptop's own fan under a scripted CPU-load sweep. Chosen because it is fully scriptable, 0 €,
runs headless, and is reproducible inside the repo (unlike an external bench setup).

Three real, independent channels captured synchronously:

| Channel  | Transducer                          | Feature(s) @ ~10 Hz                       |
|----------|-------------------------------------|-------------------------------------------|
| acoustic | laptop mic (ffmpeg/avfoundation)    | RMS energy + spectral centroid            |
| optical  | webcam (ffmpeg/avfoundation)        | mean frame brightness (photodiode proxy)  |
| load/therm | CPU load % + CPU temp (powermetrics/SMC) | load %, temp °C                     |

`load` is deliberately included as a **confound driver**: CPU load drives fan acoustics *and*
temperature/brightness simultaneously, so a naive correlation will flag confounded pairs as
"coupled". The apparatus must reject those after conditioning.

## 3. Architecture & Data Flow

```
[capture] synchronous, timestamped
  ├─ acoustic   : mic  -> RMS energy + spectral centroid @ ~10 Hz
  ├─ optical    : webcam -> mean brightness/frame @ ~10 Hz
  └─ load/therm : CPU load % + CPU temp @ ~10 Hz
        v  multichannel.py (dropna-sync onto common time grid)
[prefilter] coupling.py: mutual information + partial correlation (confound ablation)
            + permutation p-value  -> reject confounded pairs
        v  surviving edge hypotheses
[latent]   SciNet-style bottleneck autoencoder + deep-CCA across channels
           -> recover a minimal shared latent (no label) ; predicted: latent ~ fan RPM
        v  latent series + reconstruction skill
[intervene] load_sweep.py: scripted load stages (idle -> burst -> idle ...) = ground truth
        v
[validate] does the latent track the manipulation (and decode to a physical proxy)
           while confounded edges collapse?
```

**Method order (decided 2026-06-30):** (1) latent-variable recovery (SciNet bottleneck +
deep-CCA) is the first method-beyond-classical built here — most directly falsifiable (latent
should align to fan RPM), lowest risk on real noise. (2) SINDy / Koopman (law/mode layer) next,
own follow-up spec. (3) conservation-quantity discovery + active-intervention loop later. The
classical `coupling.py` stays as the confound prefilter in front of the latent step.

## 4. Components (new, small, focused)

All under `experiments/fan_real/`. The discovery core (`signalmap/coupling.py`,
`signalmap/multichannel.py`, `signalmap/discover.py`) is reused **unchanged** as the confound
prefilter — capture, intervention and the latent-recovery method are new.

- `capture.py` — 3-channel synchronous capture, 0 €. Each channel degrades gracefully if its
  device is unavailable (mic-only / webcam-only still produce a partial run, logged honestly).
  Pure-stdlib + ffmpeg subprocess; numpy for feature reduction.
- `load_sweep.py` — reproducible CPU-load driver in pure stdlib (worker processes, timed
  idle/burst stages). Emits a ground-truth stage log aligned to the capture clock.
- `latent.py` — SciNet-style bottleneck autoencoder + deep-CCA across the synced channels
  (torch, small, laptop-only). Recovers a minimal shared latent without labels; exposes the
  latent series + reconstruction skill for validation against the load ground truth.
- `run_discovery.py` — orchestrates capture -> multichannel sync -> coupling prefilter ->
  latent recovery -> report. Writes raw vs. conditioned edge table (p-values) and the
  latent-vs-ground-truth alignment.
- `README.md` — **pre-registration** (predictions written before the run) + one-line repro command.

## 5. Validation Protocol (pre-registered, falsifiable)

1. **Predict before running** (in `README.md`, committed before any capture):
   - acoustic <-> load expected to survive conditioning (fan tracks load — real, mechanical).
   - acoustic/optical <-> temp suspected confound (common cause = load); must collapse when
     conditioned on load.
   - latent prediction: the recovered shared latent should align (high correlation) to a
     physical proxy of fan RPM / load stage, *without* ever being given it.
   No post-hoc reinterpretation (no HARKing).
2. **Run A:** load sweep on -> coupling prefilter + latent recovery. Record raw/conditioned
   edges + p-values, and latent-vs-ground-truth alignment.
3. **Run B (control interventions):**
   - cover webcam -> any optical edge must die; latent must lose the optical contribution.
   - hold load constant -> load-driven edges must collapse; latent variance must drop.
4. **Success** = predicted structure recovered, interventions move edges as predicted, AND the
   unsupervised latent tracks the controlled manipulation.

## 6. Honesty Clause (non-negotiable)

- Result is reported **as observed**, including a null or messy result. Pre-registration blocks
  outcome-shopping. Ref: feedback "fix don't ask / honesty > confidence theater".
- If webcam micro-vibration is too weak to capture at ~30 fps (plausible), that is itself an
  honest finding: the optical channel then carries brightness/flicker rather than vibration, and
  the report states this openly.
- No claim of new physics. Claim ceiling: "unsupervised-discovered, intervention-validated real
  coupling on 0 € self-captured sensors."

## 7. Out of Scope

- Causal-RCA (CCM/TE/Granger) stays as an optional layer, untouched by this experiment.
- No external hardware, no ESP32, no PyPI/GitHub push as part of this work.
- No materials-specific claims; fan/CPU is a methodology demonstrator, not a materials study.

## 8. Implementation Pre-Check (step 0 of the plan)

Before building capture: verify on this machine that ffmpeg + avfoundation can enumerate the mic
and webcam, and that a CPU-temp source (powermetrics or SMC) is readable without paid tooling.
Pick the working sources; if a channel is unreadable, the design's graceful-degradation path
applies and is documented.
