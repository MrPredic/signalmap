# signalmap distill — gauntlet receipt

- verdict: PASS
- classes (2): quiet, storm
- recordings: 16 · windows: 128 · chance: 0.500

## capacity gate
- grammar total: 2128
- budget = min(2128, 50*16) = 800 (50.0 programs/recording)

## accuracy receipts (LOGO over recordings)
- nested LOGO (honesty anchor): 0.883
- deploy selection (biased UB): 0.969
- lean baseline (permEnt+psdSlope): 0.922 (+0.047 vs lean)
- group-permutation p: 0.016
- NULL self-test (labels shuffled): 0.523 (want ~chance 0.500)
- cost: 0.193 ms/window (4 programs)

## deploy spec (selected programs)
- peakcv(diff(rank(x)))
- zcr(abs(diff(x)))
- acf1(abs(diff2(x)))
- meanabs(diff2(diff(x)))

Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend.