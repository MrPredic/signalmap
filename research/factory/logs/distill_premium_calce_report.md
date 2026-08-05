# signalmap distill — gauntlet receipt

- verdict: PASS
- classes (3): early, late, mid
- recordings: 18 · windows: 306 · chance: 0.333

## capacity gate
- grammar total: 2128
- budget = min(2128, 50*18) = 900 (50.0 programs/recording)

## accuracy receipts (LOGO over recordings)
- nested LOGO (honesty anchor): 0.892
- deploy selection (biased UB): 0.935
- lean baseline (permEnt+psdSlope): 0.585 (+0.351 vs lean)
- group-permutation p: 0.005
- NULL self-test (labels shuffled): 0.366 (want ~chance 0.333)
- cost: 0.319 ms/window (5 programs)

## premium families (cost-receipted champion rule)
- rqa: base 0.935 -> augmented 0.905 (paired -0.030, 95% CI [-0.072, +0.006]) — **EXCLUDED** (not CI-solid over base)
  cost: 58.05 ms/window vs base 0.319 ms/window (~182x); O(n^2) recurrence matrix per window (~200x the base grammar)

## deploy spec (selected programs)
- acf1(clip(diff2(x)))
- std(rollstd(id(x)))
- meanabs(cumsum(diff(x)))
- iqr90(rollstd(id(x)))
- runmean(abs(env(x)))

Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend.