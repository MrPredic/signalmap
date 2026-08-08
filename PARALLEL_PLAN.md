# SignalMap — Parallel Work Plan (2026-06-30)

How to use: open ONE terminal per track. In each, paste the track's "Kickoff" line. Tracks are
independent (different data, different methods, no shared files except this doc + memory). Each
writes results to its own scratchpad/branch and appends a 1-line status below. No subagents
(token rule) — these are human-opened terminals.

Shared facts: python = `/usr/bin/python3` (numpy 1.26.4 / scipy 1.13.1 / torch 2.8.0). Repo =
`<local-path>/signalmap`, NOT pushed. Honesty > confidence-theater; report nulls as-is.
Background reading: memory `signalmap_project.md` top entry; this session's probes are in the
session scratchpad (probe1..6).

---

## T1 — Robust joint coordinate+law discovery (THE MOAT, hardest)
**Goal:** accurate SPARSE closed-form law from PARTIAL observation (single observable) on
Lorenz + Rössler. Today's joint SINDy-AE (Champion/Lusch/Kutz/Brunton 2019) got sparse+stable
but NOT accurate (short-horizon forecast R² negative). Make it accurate.
**Levers to try:** weak-form / integral SINDy (avoid noisy derivative), multi-seed + keep best,
latent smoothness regularisation, refinement phase (freeze sparsity mask then fit unregularised),
loss-weight schedule (recon-dominant → dynamics-dominant), attractor-statistics success metric
(not exact trajectory). Baseline code pattern in session probe5/probe6.
**Done =** ≥1 system where simulated learned latent law is sparse + bounded + short-horizon
R²>0.6 over multiple starts, reproducibly.
**Kickoff:** "Track T1 from PARALLEL_PLAN.md: make joint SINDy-AE recover an accurate sparse law
from a single observable. Start from probe6 pattern, add weak-form SINDy + multi-seed."
**Status:** 2026-07-01 method battery M1-M8 (research/m*.py), MULTI-START verified.
SPLIT RESULT: (a) prediction from 1 observable SOLVED -> HAVOK/Hankel-DMD mean forecast_R2=0.68
over 10 starts (never negative), but compact-LINEAR not sparse-nonlinear. (b) sparse interpretable
law OPEN -> every sparse attempt fails Lorenz multi-start (M4 0.68 single -> -1.36 multi; M1 0.65 ->
-0.29), gplearn bloats, AE gives dense non-predictive law. Identifiability barrier real, unbeaten.
RULE: multi-start mean only, never single-start (it produced 2 false 'breakthroughs' today).
NEXT bets: differentiable law-sparsity regularizer as coord PRESSURE; attractor-statistics loss;
multi-delay consistency; weak-form SINDy inside AE latent.

## T2 — Latent-recovery on REAL multimodal data (the DNA-thesis test)
**Goal:** the working half (SciNet latent recovered hidden state R²=0.92 on Lorenz) — does it
extract structure from REAL synchronous multi-sensor data? Pull PAMAP2 (or Opportunity / UCI-HAR):
accelerometer+gyro+magnetometer+HR on human activity. Train SciNet-style latent on a SUBSET of
sensors, test if latent reconstructs the HELD-OUT sensors + separates activities unsupervised.
**Done =** quantified: held-out-sensor reconstruction R² + unsupervised activity separation on
real data, honestly reported (incl. if it fails).
**Kickoff:** "Track T2 from PARALLEL_PLAN.md: download PAMAP2, run SciNet latent recovery on real
multimodal sensors, measure held-out-sensor R² + unsupervised structure."
**Status:** (unstarted)

## T3 — Method breadth on known systems (complementary readers)
**Goal:** add + benchmark Koopman/DMD (Schmid 2010) and conservation-law discovery (Liu & Tegmark
2021) as additional "readers", tested on `dysts` (130+ chaotic systems with known ODEs, `pip
install dysts`). Which method wins on which system class → builds the honest method-portfolio.
**Done =** a results table: method × system-class → recovery score, on independent ground-truth.
**Kickoff:** "Track T3 from PARALLEL_PLAN.md: pip install dysts; implement Koopman/DMD +
conservation-law discovery; benchmark vs CCM/SINDy across dysts systems."
**Status:** (unstarted)

## T5 — Label-free signal-vs-noise-vs-determinism (goal #1; material-research aim)
**Goal:** decide WITHOUT labels whether a segment is unstructured noise, linearly-correlated
noise, or nonlinear-deterministic signal. Surrogate-data test, threshold-free (z vs null).
**Done =** 2026-07-01 DONE for baseline (research/sanity_t5.py): L1 shuffled+maxACF separates
white (z=-0.5, FP-control OK) from any structure (z>120); L2 IAAFT+nonlinear-prediction-error
flags Lorenz as nonlinear (z=3.4) while pink-1/f + periodic sine correctly stay linear-correlated.
Sweep DONE (research/t5_dysts_sweep.py): 18/18 nonlinear dysts systems detected (L2 z 5.7..43),
0/4 control false-positives (white/pink-1f/AR1/sine all z<3). Thin single-run margin was a
data-length artifact (dysts resample gives Lorenz z=25.7). REMAINING: apply to a REAL material/
vibration series (CWRU bearing data already in repo) — that is the only non-synthetic gap.
**Kickoff:** "Track T5 from PARALLEL_PLAN.md: apply validated surrogate detector to real CWRU
bearing vibration — does fault vibration read as nonlinear-deterministic vs healthy/noise?"
**Status:** synthetic benchmark DONE (18/18, 0 FP). Real-data application = next.

## T4 — Consolidation into repo (sequential / lowest risk)
**Goal:** fold validated methods into `signalmap/` cleanly: causal.py stays as OPTIONAL layer,
discovery-engine (coupling/discover/latent) becomes the documented core; add tests; honest README
that separates "verified" from "research goal". No push until user says so.
**Done =** repo reflects the reset, tests green (`/usr/bin/python3 -m pytest`), README honest.
**Kickoff:** "Track T4 from PARALLEL_PLAN.md: consolidate validated discovery methods into the
repo, causal as optional layer, tests green, honest README. Do not push."
**Status:** (unstarted)

---
Coordination: when a track produces a real result, append a 1-line status above and add a note to
memory `signalmap_project.md`. T4 should run AFTER T1–T3 produce something worth integrating.
