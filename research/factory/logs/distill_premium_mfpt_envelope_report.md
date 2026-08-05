# signalmap distill — gauntlet receipt

- verdict: PASS
- classes (3): baseline, inner, outer
- recordings: 20 · windows: 3718 · chance: 0.333

## capacity gate
- grammar total: 2128
- budget = min(2128, 50*20) = 1000 (50.0 programs/recording)

## accuracy receipts (LOGO over recordings)
- nested LOGO (honesty anchor): 0.812
- deploy selection (biased UB): 0.830
- lean baseline (permEnt+psdSlope): 0.780 (+0.050 vs lean)
- group-permutation p: 0.005
- NULL self-test (labels shuffled): 0.387 (want ~chance 0.333)
- cost: 0.388 ms/window (5 programs)

## premium families (cost-receipted champion rule)
The PASS/FAIL verdict above gates the BASE selection only (nested LOGO on grammar programs). Premium families are judged separately by the champion rule below — a bank can honestly be base-FAIL with a premium family INCLUDED when the signal lives only in the premium features.
- envelope: base 0.830 -> augmented 0.842 (paired +0.012, 95% CI [-0.009, +0.035]) — **EXCLUDED** (not CI-solid over base)
  cost: 0.15 ms/window vs base 0.388 ms/window (~0x); O(n log n) Hilbert+FFT per window (cheap)

## deploy spec (selected programs)
- acf1(cumsum(sq(x)))
- acf1(cumsum(abs(x)))
- hent(tanh(id(x)))
- peakcv(cumsum(abs(x)))
- crest(tanh(rollstd(x)))

Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend.