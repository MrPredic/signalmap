# Does the valve AUC inversion depend on a feature that is constant by construction?

Bank: `data/mimii/valve_id00_bank` (frozen by PREREG_DCASE_VALVE_EXTERNAL.md).
Fit windows: 17820 | eval clips: 204 (104 anomaly, 100 normal). Seed 0, K=20.

| spec | AUC | 95% CI | direction | features on the 1e-12 guard floor |
|---|---|---|---|---|
| shipped_9 | 0.2642 | [0.1929, 0.3360] | inverted | ['std(id(id(x)))'] |
| without_degenerate_8 | 0.2642 | [0.1929, 0.3360] | inverted | none |

**Finding: the inversion SURVIVES removal of the degenerate feature.**

Reference: the 2026-07-21 readout reported AUC 0.2642, CI [0.1929, 0.3360] on this bank with the shipped 9-program spec.

Reproduce: `nice -n 19 .venv/bin/python research/factory/degenerate_feature_probe.py` (191.3s)
