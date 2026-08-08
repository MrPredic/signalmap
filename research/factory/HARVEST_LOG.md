# HARVEST LOG — Terminal B (Jul 2, 2026)

Daten-Harvest nicht-personenbezogener Domänen (Industrie/Physik). Jede Bank:
`bank_audit()` (Gruppen-Integrität + Recording-Label-Shuffle) ZUERST, dann
`gauntlet()` (lean+perm-p, nested Forge mit Kapazitäts-Gate C=50×n_rec,
Bootstrap-CIs, gepaart forge−lean). Alle Zahlen leakage-frei LOGO.
Ergebnisse NUR hier (RESULTS.md = Terminal A). Fold-Accs in logs/gauntlet_folds.csv.

## Neue Banken

### 1. DCASE2020 pump (MIMII, Akustik — andere Physik als valve: Kavitation/Fluss statt Impuls)
- Quelle: zenodo.org/record/3678171 `dev_data_pump.zip` (1.03 GB) → `data/dcase_pump/`
- Loader: `load_dcase_pump(task='id'|'anomaly')` — Fix des valve-Coverage-Caveats:
  Fenster GLEICHMÄSSIG über den ganzen 10s-Clip verteilt (nicht nur erste 0.32s).
- Tasks: id = 4-Kl. Maschinen-ID (24 Recs=Clips), anomaly = normal-vs-anomaly id_00 (16 Recs).

### 2. Seismik IU.ANMO (IRIS/USGS Web-Services — komplett neue Physik)
- Quelle: live gefetcht via `irisws/timeseries` (ASCII) + USGS fdsnws-event; gecacht
  `data/seismic/bank_depth.npz`. Kein obspy nötig. Reproduzierbar: `load_seismic_depth(refetch=True)`.
- Task: Herdtiefe shallow (<70 km) vs deep (>300 km), M6.3+ 2024–2026, 8+8 Events.
- Design gegen Confounds: ALLE Events an DERSELBEN Station (IU.ANMO.00.BHZ, 40 sps)
  → Stations-Identität kann nicht diskriminieren. Recording = Event (16, 1 Label/Event).
  Fenster = 8×1024 (25.6 s) am Rolling-RMS-Peak des 30-min-Fensters ab Origin+60s.
- Fetch-Bug notiert: `&nodata=404` → HTTP 400 bei irisws/timeseries (Param weggelassen).

### 3. IMS/NASA Bearing run-to-failure (Rexnord ZA-2115, 20 kHz Snapshots alle 10 min)
- Quelle: phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip → `data/ims/` (zip→7z→3 rar;
  Archiv-Quirk: 3rd_test.rar entpackt als `4th_test/txt` = dokumentierter Test 3).
- Loader: `load_ims(task='fault'|'stage')`; Ground-Truth-Defekte: t1-b3 inner, t1-b4
  roller, t2-b1 outer, t3-b3 outer.
- `fault`: defekt-vs-gesund am Lebensende (letztes Terzil), Recording = Test×Bearing
  (12 Recs, 4 defect/8 healthy). Zeit/Rig-Confound NEUTRALISIERT: beide Klassen teilen
  identische Timestamps im selben Test.
- `stage`: early/mid/late-Terzile, Recording = Test×Bearing×Stage (36 Recs).
  CAVEAT (wie HYD): Bearings eines Tests zeit-synchron → Held-out-Recording teilt
  Timestamps mit Trainings-Bearings (Rig-State-Adjazenz). Nur mit Caveat lesen.

## Ergebnisse

(Stand: konsolidiert durch Terminal A nach Unterbrechung von Terminal B;
Jobs liefen weiter, Logs: harvestB_ims.log, harvestB_pump_seis.log)

| Bank | Chance | LEAN [CI] (perm-p) | FORGE nested [CI] | gepaart | Verdikt |
|---|---|---|---|---|---|
| IMS-fault (12 Recs) | 0.500 | 0.609 [0.417,0.781] (0.164) | 0.625 [0.432,0.797] | +0.016 n.s. | ehrliches Null (bank_audit 2/2 PASS; Zeit-Confound designseitig neutralisiert) |
| PUMP-id (24 Recs) | 0.250 | 0.292 [0.175,0.417] (0.049) | 0.358 [0.233,0.500] | +0.067 n.s. | marginal: lean knapp signifikant, Forge-CI berührt Chance |
| PUMP-anomaly (16 Recs) | 0.500 | 0.362 [0.238,0.500] (0.820) | 0.425 [0.287,0.575] | +0.062 n.s. | ehrliches Null (beide ≤ Chance) |
| IMS-stage (36 Recs) | 0.333 | 0.257 (0.820) | 0.365 [0.316,0.415] | +0.108 [+0.045,+0.170] | Null vs Chance (CI-Untergrenze < 0.333); forge>lean gepaart CI-fest, aber ohne Chance-Klärung wertlos; CAVEAT Rig-Adjazenz |
| **SEIS-depth (16 Events)** | 0.500 | **0.688 [—] (0.033)** | **0.734 [0.586,0.867]** | +0.047 n.s. | **★ Signal in 4. Physik (Seismologie): CI über Chance, lean perm-p signifikant** |

## VALVE-Retest mit Coverage-Fix (Jul 2 Nacht, load_dcase_valve full_coverage=True)
Fenster jetzt gleichmäßig über den ganzen 10s-Clip (wie pump). Logs: valve_fullcov.log.
| Task | Chance | LEAN (perm-p) | FORGE nested | gepaart | Verdikt |
|---|---|---|---|---|---|
| VALVE-id-fullcov | 0.250 | 0.325 (0.033) | **0.167 [0.092,0.250]** | **−0.158 [−0.258,−0.058]** | lean-Signal bestätigt; **Forge ANTI-lernt (erste Bank, CI-fest unter lean)** — Textur-Bank, cumsum-Pool ohne Guard; Guard-Test s.u. |
| VALVE-anomaly-fullcov | 0.500 | 0.238 (0.98) | 0.400 [0.288,0.500] | +0.163 n.s. | Null bestätigt auch mit voller Coverage; Fold-Selektionen voller cumsum-Programme (Attraktor-Symptom) |

Coverage-Fix-Effekt auf lean: id 0.342→0.325 (≈gleich), anomaly 0.463→0.238 —
die alte 0.32s-Zahl war eher Zufallsrauschen; MIMII-valve-anomaly gibt bei
unserem Fenster-Design schlicht nichts her.

**Guard-Test #2 auf VALVE-id (valve_id_guard.log): Guard heilt NICHT** — forge mit
texture_guard 0.200 (ohne 0.167), beide CI-fest unter lean 0.325. Diagnose über
Fold-Selektionen: **komplett instabil** (jeder Fold andere Features) vs stabile
Gewinner-Banken (CALCE 6/6, SEIS 15/16 identisch). → Der VALVE-id-Absturz ist
SELEKTIONSRAUSCHEN bei schwachem Signal, kein cumsum-Attraktor. Zwei Lehren für
distill: (1) **Champion = paired-CI-Sieger; Forge ersetzt lean NIE stillschweigend**
(Forge kann CI-fest schlechter sein — hier bewiesen); (2) **Selektions-Stabilität
über Folds = billige Vertrauens-Diagnose** (instabil ⇒ Forge-Zahl nicht trauen) —
empirische Bestätigung des geplanten Stabilitäts-Screenings.

**★ SEIS-depth Detail:** bank_audit 2/2 PASS (Shuffle 0.688→0.469). Stabile Familie:
`hent(clip(diff(x)))` Top-1 in 15/16 Folds — Amplitudenverteilungs-Entropie der
Ableitung trennt shallow/deep-Wavetrains. Forge=lean-Parität (konsistent mit dem
Muster: lean deckt „impulsive/spektrale" Physiken ab). Stations-Confound
designseitig eliminiert (alle Events an IU.ANMO). n=16 Events → Replikation mit
2. Station (z.B. IU.KONO) wäre der nächste Härtungsschritt.

Bemerkenswert IMS-fault: Fold-Bimodalität (0.938er neben 0.000/0.062) — einzelne
Bearings dominieren; mit nur 4 Defekt-Recordings zu wenig Gruppen für Stabilität.
PUMP-id: `specratio(tanh(diff2(x)))` in fast allen Folds Top-1 (stabile Familie),
aber Niveau schwach — Akustik-ID über Clips hinweg bleibt hart (konsistent mit
DCASE-valve-id 0.342).
