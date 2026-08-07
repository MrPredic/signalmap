# Sign identifiability of label-free anomaly scores

Prereg: `PREREG_SIGN_IDENTIFIABILITY.md` (frozen before any bank was built).
One frozen spec for every domain, K=20 windows of W=1024, envelope=3.0, seed=0, 2000 bootstrap / 2000 permutations.

| domain | n | AUC | 95% CI | direction | verdict | perm p | N1 | N2 | AUC* vs null q95 |
|---|---|---|---|---|---|---|---|---|---|
| synth_neg | 80 | 0.1525 | [0.0763, 0.2412] | inverted | PASS | 0.0000 | ok | ok | 0.848 vs 0.626 (+0.222) |
| mimii_valve_id00 | 219 | 0.2786 | [0.2110, 0.3507] | inverted | PASS | 0.0000 | ok | ok | 0.721 vs 0.580 (+0.141) |
| mafaulda | 362 | 0.3124 | [0.2370, 0.3819] | inverted | REFUSED | 0.0005 | ok | FAIL | 0.688 vs 0.607 (+0.080) |
| paderborn_kat_n15 | 120 | 0.3172 | [0.2235, 0.4194] | inverted | PASS | 0.0005 | ok | ok | 0.683 vs 0.607 (+0.076) |
| paderborn_kat | 480 | 0.4976 | [0.4434, 0.5513] | undetermined | REFUSED | 0.9210 | ok | FAIL | 0.502 vs 0.553 (-0.051) |
| mimii_pump_id00 | 243 | 0.5203 | [0.4438, 0.5967] | undetermined | REFUSED | 0.5730 | ok | ok | 0.520 vs 0.573 (-0.053) |
| mimii_fan_id00 | 507 | 0.5609 | [0.4952, 0.6236] | undetermined | REFUSED | 0.0485 | ok | ok | 0.561 vs 0.560 (+0.000, marginal) |
| mimii_slider_id00 | 456 | 0.8254 | [0.7703, 0.8781] | aligned | PASS | 0.0000 | ok | ok | 0.825 vs 0.562 (+0.263) |
| synth_pos | 80 | 1.0000 | [1.0000, 1.0000] | aligned | PASS | 0.0000 | ok | ok | 1.000 vs 0.627 (+0.373) |

**H1 (sign not identifiable): CONFIRMED** — inverted: ['mimii_valve_id00', 'paderborn_kat_n15', 'synth_neg']; aligned: ['mimii_slider_id00', 'synth_pos']. Counted only where the verdict is PASS: a CI clear of 0.5 does not count if a null control failed. Excluded on that ground: ['mafaulda'].
**H2 (magnitude carries information):** 6/9 domains clear their own shuffle-null 95th percentile by more than 0.02; 1 sit on it (margin < 0.02) and are counted as undecided, not as support.
**H3 (shipped alarm): NOT VALIDLY MEASURED** — see prereg AMENDMENT 3. The threshold is the 99th percentile of *window* scores, but this run compared it against the *recording* mean, so the alarm is structurally silent (0.000 on both classes in every domain) regardless of any anomaly. No alarm claim is made from this run, for or against H3.
**M4 sign transfer:** hit rate 0.400 over 20 ordered pairs of CI-fest domains (0.5 = the sign of one domain says nothing about another).

Reproduce: `nice -n 19 .venv/bin/python research/factory/sign_identifiability_readout.py --refresh`
Signed receipts: `research/factory/logs/sign_receipts/` — verify offline with `python tools/verify_receipt.py <receipt>`, which imports nothing from signalmap.
