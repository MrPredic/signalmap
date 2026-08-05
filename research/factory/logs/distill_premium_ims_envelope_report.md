# signalmap distill — gauntlet receipt

- verdict: PASS
- classes (2): defect, healthy
- recordings: 12 · windows: 192 · chance: 0.500

## capacity gate
- grammar total: 2128
- budget = min(2128, 50*12) = 600 (50.0 programs/recording)

## accuracy receipts (LOGO over recordings)
- nested LOGO (honesty anchor): 0.688
- deploy selection (biased UB): 0.797
- lean baseline (permEnt+psdSlope): 0.604 (+0.193 vs lean)
- group-permutation p: 0.005
- NULL self-test (labels shuffled): 0.474 (want ~chance 0.500)
- cost: 0.165 ms/window (5 programs)

## premium families (cost-receipted champion rule)
The PASS/FAIL verdict above gates the BASE selection only (nested LOGO on grammar programs). Premium families are judged separately by the champion rule below — a bank can honestly be base-FAIL with a premium family INCLUDED when the signal lives only in the premium features.
- envelope: base 0.797 -> augmented 0.766 (paired -0.031, 95% CI [-0.104, +0.026]) — **EXCLUDED** (not CI-solid over base)
  cost: 0.19 ms/window vs base 0.165 ms/window (~1x); O(n log n) Hilbert+FFT per window (cheap)

## deploy spec (selected programs)
- acf1(diff(abs(x)))
- acf1(diff2(rollstd(x)))
- crest(diff2(tanh(x)))
- acf1(diff2(sq(x)))
- lcross(diff(sq(x)))

Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend.