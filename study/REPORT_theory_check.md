# Theory check — AUC(r) = (2/pi) arctan(r) against the measurement

`r_hat` is a moment ratio (RMS of anomaly scores over RMS of normal scores); the measured AUC is a rank statistic. Two independent routes to the same quantity, with different failure modes.

| domain | AUC measured | r_hat (recording) | AUC predicted | error | direction agrees |
|---|---|---|---|---|---|
| synth_neg | 0.1525 | 0.819 | 0.4368 | +0.2843 | yes |
| mimii_valve_id00 | 0.2786 | 0.781 | 0.4221 | +0.1435 | yes |
| mimii_pump_id00 | 0.5203 | 0.856 | 0.4506 | -0.0697 | NO |
| mimii_fan_id00 | 0.5609 | 0.866 | 0.4545 | -0.1064 | NO |
| mimii_slider_id00 | 0.8254 | 1.587 | 0.6420 | -0.1834 | yes |
| synth_pos | 1.0000 | 5.132 | 0.8775 | -0.1225 | yes |

**Direction agrees in 4/4 domains whose direction the measurement actually decided** (CI clear of 0.5), and in 4/6 overall. The two disagreements are exactly the domains the readout calls `undetermined`, where the measured sign is noise and there is nothing to agree with.

Magnitudes are compressed toward 0.5 (errors -0.183 to +0.284). The formula assumes one centred feature; the product takes a max over nine correlated ones and then averages 20 windows per recording, so the magnitude is expected to be off and is reported as measured, not corrected.

Reproduce: `nice -n 19 .venv/bin/python research/factory/theory_auc_check.py`
