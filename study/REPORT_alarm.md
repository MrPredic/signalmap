# M3 — does the shipped alarm fire? (POST-HOC, not preregistered)

Prereg AMENDMENT 3: the preregistered M3 was invalid (window-calibrated threshold compared against a recording mean) and its zeros had been seen before the error was found. This rerun uses `score(w) >= threshold` per window, exactly as `DistilledDetector.alert()` does, and is reported as post-hoc. It is not evidence for or against H3.

| domain | threshold | window a/n | recording a/n | recording gap [95% CI] |
|---|---|---|---|---|
| mafaulda | 4945755.85 | 0.448 / 0.729 | 0.673 / 1.000 | -0.327 [-0.381, -0.276] |
| paderborn_kat | 19.86 | 0.000 / 0.000 | 0.000 / 0.006 | -0.006 [-0.019, +0.000] |
| mimii_fan_id00 | 35.09 | 0.000 / 0.000 | 0.000 / 0.000 | +0.000 [+0.000, +0.000] |
| mimii_pump_id00 | 26.88 | 0.000 / 0.000 | 0.000 / 0.000 | +0.000 [+0.000, +0.000] |
| mimii_slider_id00 | 29.86 | 0.000 / 0.000 | 0.000 / 0.000 | +0.000 [+0.000, +0.000] |
| mimii_valve_id00 | 92.01 | 0.000 / 0.000 | 0.000 / 0.000 | +0.000 [+0.000, +0.000] |
| paderborn_kat_n15 | 28.39 | 0.000 / 0.000 | 0.000 / 0.000 | +0.000 [+0.000, +0.000] |
| synth_neg | 10.18 | 0.000 / 0.000 | 0.000 / 0.000 | +0.000 [+0.000, +0.000] |
| synth_pos | 10.18 | 0.124 / 0.000 | 0.950 / 0.000 | +0.950 [+0.875, +1.000] |

**Never fires at all** (no window alarms in either class): ['mimii_fan_id00', 'mimii_pump_id00', 'mimii_slider_id00', 'mimii_valve_id00', 'paderborn_kat_n15', 'synth_neg'].
**Alarm gap not above zero** (CI upper bound <= 0): ['mafaulda', 'mimii_fan_id00', 'mimii_pump_id00', 'mimii_slider_id00', 'mimii_valve_id00', 'paderborn_kat', 'paderborn_kat_n15', 'synth_neg'].

Reproduce: `nice -n 19 .venv/bin/python research/factory/m3_alarm_posthoc.py`
