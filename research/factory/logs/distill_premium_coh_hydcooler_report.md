# signalmap distill — gauntlet receipt

- verdict: PASS
- classes (3): cooler100, cooler20, cooler3
- recordings: 18 · windows: 72 · chance: 0.333

## capacity gate
- grammar total: 2128
- budget = min(2128, 50*18) = 900 (50.0 programs/recording)

## accuracy receipts (LOGO over recordings)
- nested LOGO (honesty anchor): 0.694
- deploy selection (biased UB): 0.750
- lean baseline (permEnt+psdSlope): 0.625 (+0.125 vs lean)
- group-permutation p: 0.005
- NULL self-test (labels shuffled): 0.361 (want ~chance 0.333)
- cost: 0.183 ms/window (5 programs)

## premium families (cost-receipted champion rule)
- coherence: base 0.750 -> augmented 0.889 (paired +0.139, 95% CI [+0.042, +0.236]) — **INCLUDED**
  cost: 14.84 ms/window vs base 0.183 ms/window (~81x); O(C n log n) Welch cross-spectra per (ch0, chj) pair; fixed product config c128b2 (coherence_fair CONFIRMED)

## deploy spec (selected programs)
- acf1(sq(diff(x)))
- acf1(env(abs(x)))
- acf1(abs(env(x)))
- acf1(diff(diff2(x)))
- acf1(rollstd(tanh(x)))

Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend.