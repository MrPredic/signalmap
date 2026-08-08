# signalmap distill — gauntlet receipt

- verdict: PASS
- classes (6): FeCl3, H2SO4, KCl, KOH, NaCl, NaOH
- recordings: 14 · windows: 196 · chance: 0.167

## capacity gate
- grammar total: 2128
- budget = min(2128, 50*14) = 700 (50.0 programs/recording)

## accuracy receipts (LOGO over recordings)
- nested LOGO (honesty anchor): 0.653
- deploy selection (biased UB): 0.643
- lean baseline (permEnt+psdSlope): 0.607 (+0.036 vs lean)
- group-permutation p: 0.016
- NULL self-test (labels shuffled): 0.066 (want ~chance 0.167)
- cost: 0.174 ms/window (5 programs)

## deploy spec (selected programs)
- specratio(tanh(id(x)))
- specratio(diff2(id(x)))
- zcr(diff(tanh(x)))
- acf1(diff2(env(x)))
- zcr(diff(sq(x)))

Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend.