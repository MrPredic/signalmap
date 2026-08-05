# signalmap distill — gauntlet receipt

- verdict: PASS
- classes (6): B007, B021, IR007, IR021, OR007, OR021
- recordings: 24 · windows: 2849 · chance: 0.167

## capacity gate
- grammar total: 2128
- budget = min(2128, 50*24) = 1200 (50.0 programs/recording)

## accuracy receipts (LOGO over recordings)
- nested LOGO (honesty anchor): 0.874
- deploy selection (biased UB): 0.914
- lean baseline (permEnt+psdSlope): 0.872 (+0.042 vs lean)
- group-permutation p: 0.005
- NULL self-test (labels shuffled): 0.082 (want ~chance 0.167)
- cost: 0.116 ms/window (5 programs)

## premium families (cost-receipted champion rule)
The PASS/FAIL verdict above gates the BASE selection only (nested LOGO on grammar programs). Premium families are judged separately by the champion rule below — a bank can honestly be base-FAIL with a premium family INCLUDED when the signal lives only in the premium features.
- envelope: base 0.914 -> augmented 0.926 (paired +0.012, 95% CI [-0.031, +0.054]) — **EXCLUDED** (not CI-solid over base)
  cost: 0.08 ms/window vs base 0.116 ms/window (~1x); O(n log n) Hilbert+FFT per window (cheap)

## deploy spec (selected programs)
- acf1(abs(clip(x)))
- meanabs(diff(abs(x)))
- meanabs(sq(sq(x)))
- meanabs(abs(abs(x)))
- crest(tanh(diff(x)))

Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend.