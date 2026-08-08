# FRESH-DATA-SCAN (Jul 3, 2026) — Ziel: Ergebnisse auf UNBERÜHRTEN Daten (Weg zu „echter Entdeckung")

Prinzip: frozen families aus RESULTS.md werden VOR dem ersten Load des frischen
Materials fixiert (GAS-R2-Muster = Prä-Registrierungs-Ersatz), Ledger-Eintrag
vor dem Lauf. Alle Kandidaten unten sind web-/API-verifiziert (Jul 3).

## Kandidaten-Ranking

### 1. ★★ VOLCANO-EPISODIC — Kilauea 2024–2026 (IRIS HV, live)
- **Frisch:** Bank endet 2023. Seit 23.12.2024 episodische Fountaining-Eruption:
  **50 präzise HVO-datierte Episoden bis 27.6.2026** (Episode 49: 14.6., 7,5 h;
  Episode 50: 27.6.). **Episode 51 prognostiziert 7.–14. Jul 2026.**
- **Verfügbarkeit geprüft:** HV.UWE HHZ 100 Hz durchgehend 2025-06 ✓;
  HV.RIMD 2026-01 ✓ (mit kleinen Gaps). Loader existiert (harvest3_loaders.py),
  nur ERUPTIVE/QUIET-Tageslisten erweitern + HVO-Episodenliste holen.
- **Tests:** (a) frozen lean-Readout out-of-time: eruptiv vs. Pause 2025/26,
  gleiche Stationen, **n≈50 Episoden statt 16 Tage** → kleine-n-Problem gelöst;
  (b) **NEUE FRAGE (Discovery-Kandidat): seismischer Textur-PRÄKURSOR** —
  Fenster X h vor Episoden-Start vs. Mitte-Pause. HVO forecastet über Tilt
  (8–10 µrad Inflations-Schwelle); Textur-basierter Präkursor ist offen.
- **★ Prospektive Quittung möglich:** Familie + Kriterium JETZT einfrieren,
  Ledger-Hash VOR Episode 51 → Vorhersage eines noch nicht eingetretenen
  Events = härteste Quittung, die das System erzeugen kann.
- Caveats: site-lokal (bekannt, Claim entsprechend); Episoden-Cluster zeitlich
  dicht → Perioden-Spreizung über 18 Monate designen.

### 2. GEOMAG out-of-time (USGS geomag API, live) — BILLIGSTER Frische-Beweis
- **13 frische Sturmtage Kp≥7 Nov 2024–Jul 2026** (GFZ verifiziert: 2025-01-01
  Kp8.0, 2025-04-16, 2025-06-01..03, 2025-09-30, 2025-11-12 Kp8.667,
  2025-11-13, 2026-01-19 Kp8.667, 2026-01-20/21, 2026-03-21/22).
  Bank endete Okt 2024 → alle unberührt.
- Ruhetage Kp≤0.67 nur 3 frische (2024-12-26, 2025-01-26, 2026-05-06) →
  Schwelle auf ≤1.0 lockern, Änderung dokumentieren.
- frozen lean family, BOU+FRD day-matched, reiner Holdout (keine Neu-Selektion),
  n_perm=200. Aufwand: nur STORM_DAYS/QUIET_DAYS in harvest2_loaders erweitern.

### 3. SEIS backward-fresh (USGS/IRIS) — Billig-Beifang
- Bank nutzt 2024-01–2026-06. Forward zu dünn (seit 2026-06: 9 shallow/0 deep).
  **Rückwärts 2021–2023: 150 shallow / 19 deep M6.3+ (verifiziert)** = unberührte
  Periode. frozen lean-Duo, ANMO+KONO, cross-LOGO wie Härtung A1.
- Aufwand: starttime-Parameter + Cache-Name.

### 4. DCASE 2026 Task 2 (Zenodo 19336329 dev / 20437238 eval) — Thesen-Doppeltest
- Fabrikfrisch (Challenge 2026), noise-aware, **ZWEI-KANAL (Mikro nah+fern)**,
  Valve wieder dabei; 7 Maschinentypen dev / 5 eval.
- Fit: These #1-Erweiterung (Impuls-Sampling auf frischen Valves) × These #4
  (Kanal-Kombinator: chdiff/logratio(near,far) = Rausch-Trennung — exakt die
  Frage der Challenge, prä-registrierbar).
- Aufwand: Download (GB), Loader nach load_dcase_valve-Muster.

### 5. MOX-Drift 12 Monate (Zenodo 10.5281/zenodo.15681119; Sci Data Okt 2025)
- **62 SnO₂-Nanowire-Sensoren**, 700 Messungen / 39 Sessions / 12 Monate,
  3 Analyte (Diacetyl, 2-Phenylethanol, Ethanol), CSV + Feature-Notebook.
- Fit: These #4 auf 62 Kanälen; **NEUE FRAGE: drift-invariante Kanal-Muster-
  Signatur** (Autoren dokumentieren Drift, testen Invarianz nicht) —
  Monats-LOGO als Zeit-Achse.
- Caveat: 1 Gerät (kein LODO) → nur Zeit-/Session-Splits; Session-Gruppen n=39.

### 6. Batterie Feb-2026 (22 NMC-Zellen, Kalender+Zyklus, EIS multi-SOC; PMC12741404)
- CALCE-frozen-family auf frische Chemie/Protokoll; Zell-LOCO n=22 > CALCE.
- Aufwand mittel (EIS-Format ≠ Zyklus-Kurven, prüfen).

## Nicht weiterverfolgt
- GRIDFREQ-fresh: Claim bereits auf Standort/Perioden-Komponente heruntergestuft
  → Out-of-time-Test misst v. a. Drift, nicht Signal.
- Bearings: Familie via CWRU/MFPT/IMS gesättigt, kein Frische-Bedarf.

## Empfohlene Reihenfolge
**Session n+1: „Fresh-Holdout-Triple"** = GEOMAG-fresh (#2) + SEIS-backward (#3)
[beide fast gratis, beantworten die Kernfrage „funktioniert unser System auf
unberührten Daten?"] + VOLCANO-episodic out-of-time (#1a).
**Session n+2:** VOLCANO-Präkursor (#1b) — mit prospektiver Episode-51-Quittung,
falls Zeitfenster (7.–14. Jul) noch offen. Danach #5 MOX-Drift (Discovery) / #4 DCASE.

Regeln unverändert: bank_audit + gauntlet je Bank, finale Verdikte n_perm=200,
frozen families + Ledger-Eintrag VOR erstem Load, nice -n 19, checkpointen.
