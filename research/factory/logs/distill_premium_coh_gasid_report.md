# signalmap distill — gauntlet receipt

- verdict: PASS
- classes (4): CO, ethanol, ethylene, methane
- recordings: 20 · windows: 159 · chance: 0.250

## capacity gate
- grammar total: 2128
- budget = min(2128, 50*20) = 1000 (50.0 programs/recording)

## accuracy receipts (LOGO over recordings)
- nested LOGO (honesty anchor): 0.333
- deploy selection (biased UB): 0.429
- lean baseline (permEnt+psdSlope): 0.237 (+0.191 vs lean)
- group-permutation p: 0.005
- NULL self-test (labels shuffled): 0.252 (want ~chance 0.250)
- cost: 0.272 ms/window (2 programs)

## premium families (cost-receipted champion rule)
- coherence: base 0.429 -> augmented 0.435 (paired +0.006, 95% CI [-0.100, +0.113]) — **EXCLUDED** (not CI-solid over base)
  cost: 19.48 ms/window vs base 0.272 ms/window (~72x); O(C n log n) Welch cross-spectra per (ch0, chj) pair; fixed product config c128b2 (coherence_fair CONFIRMED)

## deploy spec (selected programs)
- peakcv(clip(diff(x)))
- crest(sq(rollstd(x)))

Deploy: `signalmap fit`/`monitor` load spec.json as the feature backend.