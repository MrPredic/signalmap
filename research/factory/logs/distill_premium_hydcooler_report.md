# signalmap distill — gauntlet receipt

- verdict: PASS
- classes (3): cooler100, cooler20, cooler3
- recordings: 18 · windows: 90 · chance: 0.333

## capacity gate
- grammar total: 2128
- budget = min(2128, 50*18) = 900 (50.0 programs/recording)

## accuracy receipts (LOGO over recordings)
- nested LOGO (honesty anchor): 0.733
- deploy selection (biased UB): 0.800
- lean baseline (permEnt+psdSlope): 0.533 (+0.267 vs lean)
- group-permutation p: 0.005
- NULL self-test (labels shuffled): 0.311 (want ~chance 0.333)
- cost: 0.079 ms/window (3 programs)

## premium families (cost-receipted champion rule)
- rqa: base 0.800 -> augmented 0.833 (paired +0.033, 95% CI [-0.022, +0.100]) — **EXCLUDED** (not CI-solid over base)
  cost: 59.87 ms/window vs base 0.079 ms/window (~755x); O(n^2) recurrence matrix per window (~200x the base grammar)

## deploy spec (selected programs)
- acf1(diff(diff2(x)))
- acf1(sq(diff(x)))
- zcr(sign(diff2(x)))

Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend.