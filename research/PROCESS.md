# Discovery Process v2 — consolidated (2026-07-01)

Purpose: find structure that standard methods MISS, in data where it is NOT already published.
Keep what worked, fold in every failure lesson, gate hard, prior-art FIRST.

## STEP 0 — PRIOR-ART GATE (mandatory, before any drilling)  ← the missing step that cost us ECN
Before investing, answer in writing:
- Is the DISCREPANCY (method X reveals structure standard Y misses, on this data-TYPE) already published?
  Run targeted searches: "<domain> <our-method-family> <claim> prior work".
- Is the DATA direction genuinely under-analyzed, or a worked mine?
Score gap-likelihood LOW/MED/HIGH. Only HIGH proceeds as a novelty bet. LOW = product use only.
LESSON: ECN entropy-vs-amplitude is heavily published — we replicated, not discovered. Never again
claim discovery without this gate first (= the DIN "competitor-scan before build" lesson).

## STEP 1 — SANITY GATE (before trusting any method)
Ground-truth first: method must recover a KNOWN answer within tolerance, else the reader is broken.
(full-obs SINDy -> Lorenz exact; white noise -> "no signal"; DMD -> linear eigenvalues.)

## STEP 2 — HONEST EVALUATION (what actually saved us)
- MULTI-START mean, never single-start (single-start faked 2 "breakthroughs": M4, M1).
- CAUSAL harness, no look-ahead (caught HAVOK's leaked initial condition).
- Leakage-free CV: hold out whole RECORDINGS (LeaveOneGroupOut), not windows.
- CONFOUND kill: detrend + z-score; if effect survives it is not a DC/scale/drift artifact.
- SIGNIFICANCE: label-permutation p-value (group-preserving). Report CIs, not point estimates.
- RESCUE-METRIC: pointwise forecast can fail while attractor-statistics pass — right metric,
  never lower thresholds. Rescue "dirty gold" by better metric only.
- Scaler/preproc INSIDE the CV fold (a scaler-outside-CV leak inflated us ~0.06).

## STEP 3 — BREADTH ENGINE
Ping many method variants (variant_sweep), cheap assay first, verify top-K only.
BUG TO FIX: per-variant timeout (a divergent-ODE variant hung the sweep 37 min).

## STEP 4 — RESIDUAL-RISK LEDGER (state what you canNOT exclude)
e.g. ECN: batch/session confound not excludable without cross-session recordings. Always name it.

## VALIDATED ASSETS (keep)
- Signal-vs-noise-vs-determinism surrogate detector (18/18 synthetic, 0 FP).
- Method factory (harness/methods/scorecard) + retrodiscovery bank.
- Honest gates above. Pipeline PROVEN to find a real (if known) effect cleanly = no fool's gold.

## OPEN FRONTIER (where novelty likely still lives)
- Accurate SPARSE law from a SINGLE observable (T1 moat) — confirmed HARD, unsolved here.
- Cross-modal latent recovery on REAL data (reconstruct an UNMEASURED variable) — untested on real.
Both are candidate MOONSHOTS pending Step-0 gate.
