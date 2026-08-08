# PREREG — Kilauea Episode 52 (eingefroren 17. Jul 2026, VOR Ep-52-Onset)

Ledger-Receipt `EP52-PREREG` wird VOR jedem Ep-52-Readout appended. Ep 52 ist zum
Freeze-Zeitpunkt NICHT gestartet (HVO 16. Jul 19:55Z: paused, Re-Inflation, „another
episode likely", Forecast offen). Selektionsquelle = die Ep-51-Retro-Diagnostik
(`ep52_retro.py`, Ledger EP52-RETRO 78ed7e45), explizit exploratorisch, n=1 Episode.

## Was die Retro gezeigt hat (Begründung der Auswahl)
- **Robust nur in der Overflow-Phase:** `psd_slope` an UWE und RIMD fällt steil und
  monoton, je näher am Onset (T-12h −0.59/−0.97, T-6h −0.83/−1.73, T-1h −1.05/−1.61),
  beide Stationen gleiche Richtung. Mechanistisch plausibel: Lava an der Oberfläche
  (dokumentierter Overflow ab Ep-51 00:51Z) rötet das seismische Spektrum.
- **Kein sauberer >24h-Vorläufer:** in der echten Vorhersage-Zone (T-24/36/48h) flippt
  jedes Feature (perm_ent, psd_slope, ar1, rqa_det, speccent) die Richtung über die
  Offsets → Rausch-Niveau. Der Top-Score der Retro (T-24h ar1, 291) war ein
  Nenner-Artefakt (zwei Kontrolltage zufällig fast gleich), rohe Deltas bescheiden.
- Ergebnis passt zum gesamten Vulkan-Strang: Triangulation entwertete den −12h-Fund,
  Horizon-Scan fand nur Artefakte. Ep 51 bestätigt das eher, als es zu widerlegen.

## PRIMARY (falsifizierbar, prospektiv, gerichtet — 1-seitig)
**Während der HVO-dokumentierten precursory-Overflow-Phase von Ep 52 ist `psd_slope`
(Welch, nperseg=256, aus `volcano_precursor._fetch`-Fenstern) an UWE UND RIMD NIEDRIGER
als an tiefen Pause-Kontrollen zur gleichen Uhrzeit.**
- Offsets relativ zum HVO-Overflow-Start (nicht zum Fountain-Onset), gemessen bei
  −1h, −6h, −12h in die Overflow-Phase hinein, wie bei Ep 51.
- Kontrollen: gleiche Uhrzeit an ≥2 tiefen Pause-Tagen ≥48h vor jedem Overflow.
- **PASS-Kriterium (vorab fix): auf BEIDEN Stationen bei ≥2 der 3 Offsets
  psd_slope(pre) < psd_slope(ctrl_mean).** Sonst FAIL. Kein p-Wert (n=1 pro Episode);
  der Test ist die Richtungs-Replikation, aus Ep-51-n=1 wird Ep-52-n=2.
- **Ehrlich: das ist ein LAUFEND-Detektor, kein Mehrtage-Forecaster.** Genau das ist der
  Befund — kein magisches Frühwarnfenster behaupten.

## SECONDARY (exploratorisch, Vorhersage-Zone, niedrige Konfidenz)
`rqa_det` bei T−24h (vor Onset) an beiden Stationen NIEDRIGER als Kontrolle (Ep-51:
−0.035/−0.143, beide). Registriert, damit später nicht nachgefischt wird; ein Miss hier
ist erwartbar und kein Widerspruch zum Primary.

## Apply (nach HVO-Onset, mit dokumentierten UTC-Zeiten, KEIN Nach-Tuning)
Onset + Overflow-Start aus dem HVO-Update in `data/volcano/ep51_watch/` ablesen, dann
`ep52_retro.py` mit den Ep-52-Zeiten laufen lassen (ONSET/OVERFLOW_START ersetzen) und
gegen die Kriterien oben verdikten. Die 3 bestehenden apply51-Protokolle
(ep51_prereg/ep51_prereg2/volcano_precursor apply51) laufen als Kontinuitäts-Readout
unverändert mit — ihre Preregs gelten weiter, kein Re-Freeze.
