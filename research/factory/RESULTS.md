# Validation Bank — 2026-07-02 (alle Zahlen selbst gemessen, leakage-frei LOGO)

## READOUT-SCREEN — 4-Familien-Auslese-Matrix über ALLE 17 Banken (Jul 4, readout_screen.py)
Ein Screening-Werkzeug für die 4 Auslese-Familien (Methoden-Thesen #1–#4) als
billige Fenster-Proxies auf den fertig geschnittenen Banken: sampling (peak- vs
center-Subfenster), Zeitskala (pool 1/4/16), Norm (Level-Quittungen log-std/
log-ptp), window-mode (integ = cumsum-Proxy für phase). Flags = gepaarte
Fold-CIs (strong CI-lo>0, WEAK Δ≥0.10 über Chance). Checkpoint logs/readout_screen.csv.

**Kalibrierung: alle 17 base-Zellen replizieren die bekannten lean-LOGO-Werte exakt**
(ECN 0.592, CWRU 0.872, MFPT 0.780, GEOMAG 0.938, HYD-valve 0.192 …).
**Recall gegen bekannte Ground-Truth: IMS→TIMESCALE WEAK ✓ (pool4 +0.146, das
bekannte Artefakt), DCASE-valve→TIMESCALE strong ✓ (Bank korrekt gezeigt, Achse
proxybedingt Zeitskala statt Sampling: even geschnittene Fenster ENTHALTEN die
Impulse nicht mehr — Subfenster-Proxy kann Sampling nur begrenzt sehen), GAS
korrekt KEIN Flag (Kanal-Achse ist kein Proxy dieses Screens — Scope ehrlich).**

| Bank | Flags (Screen-Zeiger, KEINE Claims) |
|---|---|
| CWRU | SAMPLING +0.065 CI-fest; NORM/LEVEL +0.122 CI-fest (crest: lean 0.872→0.993!) |
| MFPT | SAMPLING +0.162 CI-fest (peak-Sub 0.686 vs even 0.524) |
| CALCE-soh | TIMESCALE monoton; NORM/LEVEL +0.203 CI-fest; WINDOW/PHASE +0.170 CI-fest |
| DCASE-valve-anomaly | TIMESCALE monoton +0.325 CI-fest (bekannt geknackt via peak) |
| IMS-fault | TIMESCALE WEAK pool4 +0.146 (source-level bestätigt, s.u.) |
| SEIS-depth | NORM/LEVEL +0.117 CI-fest; TIMESCALE WEAK pool4 +0.102 |
| GRIDFREQ-area | SAMPLING +0.224 CI-fest |
| VOLCANO | TIMESCALE WEAK pool16 +0.133 |
| ECN, GEOMAG, HYD×3, DCASE-valve-id, DCASE-pump×2, GAS | kein Flag (Null/Claim auf allen 4 Proxy-Achsen robust) |

Produkt-Argument steht: **Standard-Auslese testet 1 Zelle von 12; 8 von 17
Banken zeigen mindestens einen Auslese-Zeiger.** Level ist bei 16/17 Banken
schon beim Laden zerstört (z-Norm) — Level-Flags messen nur noch Crest; echtes
Level braucht Loader-Rebuild. Nächste Kandidaten aus der Matrix: CWRU+crest
(0.993 — RQA-fair 0.961 wäre eingeholt, bei ~0 Kosten!), MFPT peak-Sampling,
SEIS crest.

## PEAK-SAMPLING-QUERSCHNITT (Jul 4, peak_crosssection.py, n_perm=200 final)
Generalisiert die DCASE-valve-Entdeckung? **Nein — und genau das ist der Befund:
peak-Sampling ist impuls-physik-spezifisch, kein universeller Score-Pump.**
| Bank (source-level Rebuild) | Ergebnis | Verdikt |
|---|---|---|
| DCASE-pump-anomaly PEAK | bank_audit FAIL (Label-Shuffle kollabiert nicht) | korrekt ausgegattert |
| DCASE-pump-id PEAK | lean 0.233, perm-p 0.39; forge UNSTABLE | NULL (Pump = kontinuierliche Physik) |
| IMS-fault PEAK | lean 0.479, perm-p 0.62 | NULL (peak schadet sogar) |
| **IMS-fault POOL4** | **lean 0.766 CI[0.573,0.922], perm-p 0.005 (200 Perms)**, forge 0.641 UNSTABLE → Champion=lean | **★ Zeitskalen-Befund jetzt SOURCE-LEVEL bestätigt** (vorher nur Proxy auf geschnittenen Fenstern) |

Caveat IMS-POOL4: CLF-Robustheit 0.651 (LogReg) vs 0.766 (RF) → RF-abhängig,
Quittung dokumentiert. Loader-Infrastruktur: `_readout()` in retro_loaders.py
= geteilter source-level Auslese-Helper (valve-Refactor fenster-exakt gegen
HEAD verifiziert; pump/IMS jetzt sampling=/pool=-parametrisiert).

## ★ GAS-MC R2-HOLDOUT — Prä-Registrierungs-Ersatz BESTANDEN (Jul 4, gas_r2_holdout.py)
Kanal-Kombinator-Familie auf ALL-R1 eingefroren (top-5 F-Stat, **vor** erstem
R2-Load; R2 = 2. Konzentrations-Repetition, von keiner Such-Session je berührt);
auf R2 KEINE Neu-Selektion, nur Fold-interne Classifier-Fits.
| Quittung | Acc | CI (5 Geräte-Folds) | Verdikt |
|---|---|---|---|
| A R2-LODO frozen-family | **0.709** | [0.610,0.787] | **CI-fest > Chance 0.25** |
| B train-R1→test-R2 (cross-rep+cross-device, härtester Schnitt) | **0.660** | [0.563,0.756] | **CI-fest** |

Frisches Holdout schlägt sogar den R1-Suchwert (0.621) — kein Selektions-
Overfit; Schwach-Gerät B5 (R1: 0.290) jetzt 0.516/0.548. Frozen family:
logratio(5,7)/logratio(0,7)/chdiff(2,5)/logratio(1,5)/chdiff(1,5) — 5/5
Kanal-Kombinatoren, 0/5 Einzel-Kanal: die These #4 trägt die ganze Familie.
Caveat: n=5 Geräte-Folds; Ledger-Einträge geloggt.

## HARVEST 2 — Replizierbarkeits-Test, 3 neue Sensor-Modalitäten (Jul 2 Nacht)
Details/Caveats: HARVEST2_LOG.md. Neue Quittungen (Stabilität, Champion mit
Chance-Gate) liefen in allen Runs automatisch.
| Bank (Sensor) | Chance | LEAN (perm-p) | FORGE [CI] | Champion | Verdikt |
|---|---|---|---|---|---|
| **GEOMAG-storm** (Fluxgate BOU) | 0.500 | **0.938 (0.016)** | 0.875 | lean | ★ Physik #5 |
| **GRIDFREQ-area** (PMU, 6 Netze) | 0.167 | 0.536 (0.016) | **0.693 [0.578,0.802]**, gepaart +0.156 CI-fest | **forge** | ★ Physik #6, Forge-Win #5 |
| GAS-id (MOX e-nose, 5 Varianten) | 0.250 | ≤0.257 | best 0.346 [0.252,—] marginal | NULL/marginal | ehrliches Null; Grenze: Multi-Kanal-Muster fehlt; **Zeitskala=Suchdimension belegt** (0.158@10s→0.346@256s) |

→ Methodik repliziert in fremden Umgebungen: 2/3 Modalitäten sofort Signal
(beide perm-p 0.016), 1/3 fünffach austestiertes Null mit Ursachen-Hypothese.
~1 h pro Modalität inkl. Akquise (OSF/USGS/GFZ/UCI). Score gesamt: 15 Banken,
9 Physiken, 6 mit CI-festem Signal, Forge-Wins 5 (alle Nicht-Vibration).

## Effizienz-Moat: LEAN (perm3+psd_slope) vs HEAVY (RQA+Wavelet) — identischer Task
**⚠️ SUPERSEDED (Jul 3): Die HEAVY-Spalte war gehandicapt (sub=400, 2 Features,
Eigen-Impl). Faires Rematch → Abschnitt „RQA-FAIR-REMATCH" unten. Accuracy-
Dominanz-Claim ist TOT; der Kosten-Claim hält.**
| Domäne | Rec/Win | Chance | LEAN | HEAVY kombo | Δacc | Kosten-Vorteil | perm-p (lean) |
|---|---|---|---|---|---|---|---|
| ECN Chemikalien (6-Kl.) | 14/196 | 0.167 | 0.592 | 0.434 | +0.158 | ~37× | 0.016 |
| CWRU Fault-Typen (6-Kl.) | 24/2849 | 0.167 | 0.872 | 0.543 | +0.328 | 33× | 0.010 (100 Perms) |
| MFPT (3-Kl., anderes Rig) | 20/3718 | 0.333 | 0.779 | 0.684 | +0.095 | 31× | 0.016 |

**→ 3/3 Domänen, 2 Physiken, 2 Rigs.** Vorbehalte (gelten überall): RQA nicht
experten-getunt, RQA sieht sub=400 von 1024, Determinism-Loop pure-Python
(Kostenfaktor teils Impl; O(n²) vs O(n log n) inhärent), hohe Fold-Varianz bei
wenigen Recordings. MFPT: sr-Confound eliminiert (Baseline decimiert 2×).

## Feature-Forge (Grammatik v1: 1092 Programme) — biased UB vs NESTED (Selektion pro Fold)
| Domäne | biased UB | **NESTED (ehrlich)** | LEAN | Δ nested | Kosten | perm-p |
|---|---|---|---|---|---|---|
| ECN | 0.704 | **0.709** | 0.592 | **+0.117** | 0.24ms | 0.016 |
| CWRU | 0.911 | **0.902** | 0.872 | **+0.030** | 0.08ms | 0.016 |
| MFPT | 0.787 | **0.775** | 0.780 | ±0 (Parität) | 0.54ms | 0.016 |

**→ Forge > Hand auf 2/3, Parität 1/3, immer billiger.** Stabile Familien:
ECN=Spektralstruktur der Vorzeichenwechsel (`specratio(sign(diff(x)))`),
CWRU=Energie/Crest (Roh-Energie wiederentdeckt!), MFPT=`hent(tanh(x))`.
perm-p=0.016 ist das Minimum bei 60 Perms (0 Treffer).

## Sonstiges heute
- fit/monitor Produkt-Pfad: 238/238 Faults, 2/945 Warns auf echten CWRU ✓
- causal CLI: A→B→C korrekt, severed Link = Root-Cause Top-1 ✓
- science.py: jsonable + run_manifest (Audit-Envelope) — 8/8 Tests grün
- ECN "Corrosion Types" Subset: nur 1 Recording/Klasse → LOGO-ehrlich NICHT evaluierbar (notiert, nicht benutzt)
- Free-API-Keys (OpenRouter/Groq/Gemini): tot/stale → Task #0 Refresh ist Blocker
- variant_sweep: pausiert nach 132 min CPU (SIGSTOP, per `pkill -CONT -f variant_sweep` fortsetzbar)

## NEUE RETRO-BANKEN (Jul 2 nachm. — Daten-Harvest, 3 neue Physiken, gauntlet.py)
Alle leakage-frei LOGO, Kapazitäts-Gate C=50×n_rec, CIs = Bootstrap über Recordings.
| Bank (Physik) | Chance | LEAN (perm-p) | FORGE nested [CI] | gepaart forge−lean | Verdikt |
|---|---|---|---|---|---|
| CALCE-soh (Batterie) | 0.333 | 0.579 (0.016) | **0.903 [0.853,0.949]** | **+0.324 [+0.155,+0.511]** | ★ CI-fester Win #2 |
| HYD-cooler (Hydraulik) | 0.333 | 0.533 (0.016) | 0.722 [0.622,0.822] | **+0.189 [+0.067,+0.322]** | CI-fester Win #3 |
| HYD-valve Textur | 0.250 | 0.192 (0.62) | 0.242 | +0.050 n.s. | ehrliches Null |
| **HYD-valve PHASE** | 0.250 | 0.458 (0.049) | 0.625 [0.417,0.833] | +0.167 n.s. | **Methoden-Effekt s.u.** |
| HYD-accumulator | 0.250 | 0.275 (0.131) | 0.300 | +0.025 n.s. | ehrliches Null |
| DCASE-valve-id (Akustik) | 0.250 | 0.342 (0.049) | 0.342 | ±0.000 | Parität (Coverage-Caveat) |
| DCASE-valve-anomaly | 0.500 | 0.463 (0.54) | 0.463 | ±0.000 | Null (Coverage-Caveat) |

**★ METHODEN-ENTDECKUNG (hyd_valve_phase.log):** gleicher Task, gleiche Zyklen —
phasen-alignierte + skalen-erhaltende Fenster vs Textur-Fenster: **gepaart +0.383,
CI [+0.208, +0.558]** (15/24 Folds). Fenster-Modus ist eine eigene Suchdimension;
cumsum-Programme (Textur-Gift) werden bei Alignment zu Informationsträgern
(`speccent(cumsum(diff(x)))` Top-1). → distill: window-mode als Grammatik-Zweig,
Guard kontext-abhängig.

**Forge-Scoreboard über 7 Banken/9 Tasks:** CI-feste Wins = ECN +0.117, CALCE
+0.324, HYD-cooler +0.189 (alles NICHT-Vibration!); Parität auf Vibration
(CWRU/MFPT) + DCASE-id; ehrliche Nulls wo kein Signal (HYD-valve-Textur/-accum,
DCASE-anomaly). Muster: **Forge gewinnt genau dort, wo lean's Vibrations-Features
die Domänen-Familie verfehlen.**

**Kapazitäts-KURVE (capacity_curve.csv, Random-v2-Subsets auf ECN, 3 Seeds):**
125→0.46, 250→0.51, 500→0.53, 700→0.48, 1092→0.48, 1500→0.42, 2128→0.47 — ALLE
unter lean 0.592 und weit unter kuratiertem v1 0.709. **Revision: Zusammensetzung
(Attraktor-Freiheit) dominiert Suchraum-Größe** (Größen-Effekt ~0.1, Gift ~0.2).
Guard = Pflicht #1, Gate = #2.

Caveats neue Banken: HYD same-rig-Adjazenz (Zyklen eines Experiments); DCASE nur
erste 0.32s des Clips (Impulse verpasst) → energie-gepeaktes Sampling als Fix;
PHASE n=24 Einzel-Fenster (binäre Folds, valve90 systematisch verfehlt).

**CALCE-Härtung Leave-one-CELL-out (calce_loco.log): BESTANDEN** — Forge 0.846
CI [0.749,0.936], lean 0.773, Zelle komplett held-out → 0.903 war kein
Zell-Leakage. Fußnote: bei Label-Variation INNERHALB einer Gruppe sind
group_perm_p + ANOVA-Ranking (1-Label-pro-Gruppe-Annahme) nicht anwendbar
(perm-p=1.000 = Artefakt, nicht Null); Accuracies valide. Gauntlet-Variante
für mixed-label groups = TODO.

## FRONTIER-SPRÜNGE (Jul 2 abends — fachfremde High-Potential-Felder)
Gleiche Methodik, komplett neue Domänen. LOGO über die ehrliche Einheit.
| Bank | Chance | LEAN [CI] (perm-p) | FORGE | Einheit | Verdikt |
|---|---|---|---|---|---|
| **ECG-AFib-vs-NSR** (Medizin) | 0.500 | **0.875 [0.727,0.984]** (0.016) | **0.953 [0.875,1.000]** | Patient (LOPO) | ★ starkes klin. Signal, 1 Feature `speccent(abs(diff))`; forge +0.078 knapp n.s. |
| HAR-activity (Wearable) | 0.167 | 0.347 [0.318,0.379] | forge zu langsam | Subjekt (LOSO) | lean 2× Chance; mixed-label |

**★ AUDIT 6/6 BESTANDEN (audit.py) — Pipeline ist kein Rauch:**
1. Rauschen+Zufallslabels → 0.075 (< Chance): kein Leak.
2. Gepflanztes 3-Kl.-Signal → 1.000: findet echtes Signal.
3. Label-Shuffle ECN: 0.592 → 0.158: Shuffle zerstört Signal.
4. Gruppen-Integrität: 0 Folds mit Train/Test-Recording-Überlappung.
5. Scaler kein Leak-Vektor (RF skalen-invariant, dokumentiert).
6. **Selektions-Bias sichtbar: In-Sample-Score auf Rauschen = 1.000 (!) vs
   ehrlich-LOGO 0.075** → Bias real UND nested/perm-p entfernen ihn. Das ist
   der Investoren-Beleg, warum „Quittungen" nötig sind.

**Methodische Lehre (Restart-relevant): mixed-label-Gauntlet (gauntlet_mixed.py)
ist zu langsam** (per-Fold-Greedy × LOGO). HAR-Forge nach Kill unvollständig,
nur lean gültig. Fix: prescreen→feste Selektion statt per-Fold-Greedy.

## Offen / nächste intensive Validierung
1. Grammatik v2 (2128 Programme: +ordtrans/lcross/acflag/specflat/cumsum/clip) nested auf ECN+CWRU — läuft
2. RQA fair machen: volle 1024 Samples + pyunicorn-Referenz-Impl
3. Fold-Varianz: CIs per Bootstrap über Recordings
4. NULL-Kontrolle der Forge: auf reinem Rauschen darf nested nichts finden (Fool's-Gold-Test)
5. Domäne 4 (andere Physik, z.B. akustisch/Batterie) — Datenbeschaffung nötig

## Grammatik v1 vs v2 (nested) — Kapazitäts-Befund (Jul 2, abends)
| Bank | v1 (1092 Progr.) | v2 (2128 Progr.) | lean |
|---|---|---|---|
| ECN (14 Rec.) | 0.709 | **0.469 (Kollaps)** | 0.592 |
| CWRU (24 Rec.) | 0.902 | 0.870 (24/24, rec 23=1.000 nachgerechnet Jul 2) | 0.872 |

## Bootstrap-CIs + gepaarte Tests (Jul 2, Reboot-Session; bootstrap_cis.py/paired_lean.py)
**Kapazitäts-Regel gepaart (v1−v2, gleiche Folds):**
- ECN: **+0.240, CI [+0.102, +0.393], v2 gewinnt 0/14 Folds** → CI-fest
- CWRU: +0.034, CI [+0.001, +0.084], v2 gewinnt 3/23 → hält, knapp

**Forge vs LEAN gepaart (ehrliche Revision der Δ-nested-Spalte oben):**
- ECN: **+0.117, CI [+0.036, +0.214]** → Gewinn CI-FEST ✓
- CWRU: +0.031, CI [−0.069, +0.144] → NICHT CI-fest (Punktgewinn, statistisch Parität)
- MFPT: −0.005, CI [−0.125, +0.118] → Parität bestätigt

→ Korrigierte Story: **Forge schlägt lean CI-fest auf 1/3 Banken (ECN), Parität 2/3,
immer billiger.** ECN = die Domäne, wo lean die Familie NICht abdeckt; CWRU/MFPT:
lean reicht. Genau das soll distill pro Domäne ehrlich reporten.

**→ EIGENE DESIGN-REGEL: Suchraum-Budget ∝ #Recordings.** Größere Grammatik
verliert überall; bei kleiner Bank katastrophal (Selektionsrauschen). `distill`
braucht ein Kapazitäts-Gate. Roh-Logs: research/factory/logs/.

## NULL-Kontrolle (Fool's-Gold-Test) — BESTANDEN (Jul 2, Reboot-Session)
Nested Forge auf weißem Rauschen (18 Recs, 6 Klassen, ECN-Geometrie, Chance 0.167):
**forged nested = 0.123, lean = 0.087** — beide ≤ Chance, kein Fold > 0.357.
→ Pipeline leakt nicht; die Bank-Zahlen oben sind echt. Log: logs/forge_null_1422.log.
Nebenbefund: Rausch-Attraktor `lcross(sign(diff2(x)))` in ~90% der Null-Folds
gewählt (generalisiert korrekt nicht) → Blocklist-Kandidaten-Katalog, s. ANALYSIS.

## KONSOLIDIERUNG Terminal A (Audit) + Terminal B (Harvest) — Jul 2, spät
**Terminal A (fbed07d, Details in AUDIT_REPORT.md):**
- audit.py = CI-Gate: 6→12 Checks, 12/12 PASS in 114s (F-Cache in logs/cache/,
  Exit-Code=#Fails). Neu u.a. window-provenance, seed-stability, determinism,
  nested-vs-biased auf echtem ECN (biased 0.398 ≥ nested 0.362).
- gauntlet_mixed v2: per-Fold-Prescreen→feste Top-k. Fixte v1-PRESCREEN-LEAK
  (ANOVA auf allen Fenstern inkl. Test) + Speed (SYNTH-20G 28s, CALCE-LOCO 12s).
- **REVISION CALCE-LOCO: Forge 0.910 CI[0.849,0.957], gepaart forge−lean +0.137
  CI[+0.025,+0.231] = jetzt CI-FEST** (ersetzt v1-Greedy 0.846 / +0.073 n.s. oben) —
  feste Top-5 ist auf mixed-label-Banken nicht nur schneller, sondern besser
  (Greedy mit innerem 5-Gruppen-LOGO war rauschig). HAR-Forge damit wieder machbar.
- Befund B1: v2-Grammatik enthält Identitäts-Klone (diff∘cumsum≈id): nur 13/30
  unique im ECN-Top-30-Ranking → Dedup/Canonicalisierung für distill nötig.

**Terminal B (Session unterbrochen, Jobs liefen weiter; Details in HARVEST_LOG.md):**
3 neue nicht-personenbezogene Banken (DCASE-pump/Akustik, IU.ANMO-Seismik-Herdtiefe,
IMS-Bearing-run-to-failure) mit `bank_audit()`-Pre-Gate.
**★ SEIS-depth (4. Physik, Seismologie): lean 0.688 perm-p 0.033, forge 0.734
CI[0.586,0.867] über Chance 0.5** — Familie `hent(clip(diff(x)))` in 15/16 Folds;
Stations-Confound designseitig eliminiert (alle Events an IU.ANMO); n=16 Events,
Replikation an 2. Station = nächster Härtungsschritt. Ehrliche Nulls: IMS-fault
0.625 n.s. (Zeit-Confound neutralisiert), PUMP-anomaly 0.425 ≤ Chance; PUMP-id
0.358 marginal (lean-perm-p 0.049); IMS-stage lean=Null, Forge läuft noch.
bank_audit (Bank-Ebene) + audit.py (Pipeline-Ebene) ergänzen sich → zusammen =
vollständiges CI-Gate pro neuer Bank. Details: HARVEST_LOG.md.

## GRAMMATIK v2.1 — Dedup + Stationaritäts-Guard VALIDIERT (Jul 2, Nacht)
Audit-Befund B1 an der Wurzel gefixt (`feature_forge.programs(dedup, texture_guard)`):
1. **Dedup** (probe-basiert, datenunabhängig, dual: |r|>0.999 ODER ≥95% bit-equal
   für diskrete Features): 2128 → 1150 Programme; ECN-Top-30-Ranking 13 → **24/30
   effektiv unique**. Audit v2.1: 12/12 PASS.
2. **★ Stationaritäts-Guard = belegte INTERVENTION (nicht mehr nur Diagnose):**
   ECN nested v2-voll 0.469 (Kollaps) → dedup 0.485 → **dedup+guard (cumsum raus,
   897 Progr.) 0.689 ≈ kuratiertes v1 0.709**, lean 0.592. Der Kollaps IST der
   cumsum-Attraktor; Dedup allein heilt nicht (Attraktoren sind keine Klone).
   Guard nur für texture-mode; bei phase-aligned Fenstern bleibt cumsum
   Informationsträger (HYD-PHASE). → distill: `texture_guard=True` als Default
   für Textur-Banken, Kapazitäts-Gate bleibt Pflicht #2.
3. CALCE-LOCO auf v2.1 (gauntlet_mixed v2): 0.858, gepaart +0.086 CI[+0.010,+0.164]
   — CI-fest über drei Grammatik/Selektions-Varianten (0.846 greedy / 0.910
   v2-top5 / 0.858 v2.1-top5) = robuster Befund.
4. **Grenze des Guards (VALVE-id-fullcov): Forge kann CI-fest UNTER lean fallen
   (0.167/0.200 mit Guard vs lean 0.325, gepaart −0.158/−0.125)** — Ursache dort
   nicht Attraktoren, sondern Selektionsrauschen (Fold-Selektion komplett instabil
   vs CALCE 6/6 / SEIS 15/16 stabil). → distill-Regeln: Champion = paired-CI-Sieger
   (Forge ersetzt lean nie stillschweigend); Selektions-Stabilität = Pflicht-Quittung.
   IMS-stage final: 0.365 [0.316,0.415] = Null vs Chance (Rig-Caveat). VALVE-anomaly
   auch mit Coverage-Fix Null. Details HARVEST_LOG.md.

## Kollaps-Diagnose v2-ECN (aus Fold-Logs, Jul 2)
v1 wählt `specratio(sign(diff(x)))` als Top-1 in 13/14 Folds (stabile Familie).
v2: `specflat(cumsum(·))` verdrängt den Gewinner; cumsum-Top-1-Folds kollabieren
(0.143–0.214). Mechanismus: cumsum ⇒ random-walk (1/f²) ⇒ hohe In-Sample-F auf
14 Recording-Means, null Generalisierung. → distill braucht zusätzlich
Stationaritäts-Guard + Stabilitäts-Screening (DISTILL_DESIGN.md).

## HÄRTUNG — Cross-Station/Standort-Replikation (Jul 3, RESTART Prio A)
Design: cross-LOGO = Train auf Quell-Station OHNE Event/Tag r, Test NUR auf
Ziel-Stations-Fenster von r → killt Station- UND Event/Tag-Leakage gleichzeitig.
Skripte: harden_transfer.py (SEIS/GEOMAG), gridfreq_disentangle.py.

**A1 SEIS-depth ✅ BESTANDEN (2. Station IU.KONO, exakt dieselben 16 Events,
beide 40 sps → kein sr-Confound):**
- KONO within-LOGO (unabhängige Replikation): 0.688, perm-p 0.049 — identischer
  Mittelwert wie ANMO 0.688.
- Cross-Station beide Richtungen CI-fest: ANMO→KONO 0.680 [0.531,0.812],
  KONO→ANMO 0.711 [0.547,0.859] (Chance 0.5). Stations-Confound tot.
- Nebenbefund: Forge-Familie `hent(clip(diff))` allein transferiert schwach
  (0.570 n.s.) — das lean-Duo trägt den Cross-Station-Transfer.

**A3 GEOMAG-storm ✅ BESTANDEN (2. Observatorium FRD, gleiche Tage, day-matched;
2023-12-09 fehlt bei FRD → 15 Tage):**
- FRD within-LOGO: 0.892, perm-p 0.016 (BOU war 0.938) = unabhängige Replikation.
- Cross: BOU→FRD 0.783 [0.658,0.883], FRD→BOU 0.817 [0.708,0.917] — CI-fest.
- Caveat bleibt: Stürme sind global, gleiche Tage an beiden Stationen = inhärent
  (cross-LOGO hält den Tag aus dem Training, mehr geht designseitig nicht).

**A2 GRIDFREQ-area ⚠️ TEIL-BESTANDEN (4 Zweit-Standorte von OSF by5hu:
PT01→FR01/CE, GB02→GB01, US_TX02→US_TX01, ZA02→ZA01; andere Site, anderes
Gerät, andere Kampagnen-Periode — PT01 2018, GB02 2019, ZA02 2025):**
- Train Original-6-Netz-Bank → Test Zweit-Standorte: lean 0.625 [0.445,0.797],
  lean+Champion 0.633 (Chance 0.167) → Netz-Signal ist real und CI-fest.
- Rückrichtung 0.406 [0.266,0.547] — über Chance, aber deutlich schwächer.
- FR01→PT01 (CE) = 1.000 perfekt; rückwärts PT01(2018)→FR01(2019) 0.156 →
  Perioden/Geräte-Drift asymmetrisch. GB 0.31, ZA 0.44-0.47.
- **Ehrliches Verdikt: Claim heruntergestuft von „Netz-Fingerprint" auf
  „Netz-Fingerprint mit Standort/Perioden-Komponente"** (within-campaign 0.693
  vs cross-location 0.63/0.41). Für RESULTS-Scorecard: Signal hält, Confound
  quantifiziert.

## RQA-FAIR-REMATCH (Jul 3, Prio A4 — Pflicht vor Public) — rqa_fair.py
Fair = pyunicorn 0.9.0 (Referenz-Impl), VOLLE 1024 Samples, 6 Standard-Maße
(RR, DET, LAM, diag-ENTR, L_mean, TT), (dim,τ)-Grid {3,5}×{5,10}, bestes Config
gewinnt → Selektionsbias PRO RQA (konservativ für uns). Gleiches Protokoll
(LOGO, RF150, gepaarte Fold-CIs). Checkpoints: logs/rqa_fair.csv.

| Bank | lean | best fair RQA | gepaart lean−RQA | Kosten |
|---|---|---|---|---|
| ECN | 0.592 @ 0.4ms | 0.520 (m3τ5) @ 71ms | +0.071 CI[−0.046,+0.189] = **TIE** | lean 192× billiger |
| CWRU | 0.872 @ 0.4ms | **0.961 (m3τ10)** @ 93ms | −0.089 CI[−0.193,−0.009] = **RQA CI-fest besser** | lean 259× billiger |

**Ehrliches Verdikt:**
1. Der alte „lean schlägt heavy 3/3"-Accuracy-Claim war ein HANDICAP-ARTEFAKT
   (sub=400 kostete RQA auf CWRU 0.42 Accuracy!). Er wird zurückgezogen.
2. Was HÄLT: der KOSTEN-Moat. Lean liefert 0.872/0.592 bei ~200–260× weniger
   Rechenzeit; auf ECN ist fair-RQA nicht mal besser. Edge/TinyML-Story intakt:
   O(n log n) vs O(n²) ist inhärent, kein Impl-Artefakt.
3. Auf Vibration (CWRU) schlägt Experten-Nichtlinearik (RQA voll) auch den
   Forge (nested 0.902 < 0.961). Plattform-Story korrekt formuliert:
   **Kosten-Accuracy-Frontier mit ehrlichen Quittungen**, nicht Accuracy-Dominanz.
4. MFPT (0.779 vs 0.684) wurde nicht neu getestet → alter MFPT-Moat-Wert gilt
   als SUSPEKT (gleiche Handicaps), nicht zitieren bis Fair-Rerun.

## WERKZEUG-KOMPLETTIERUNG (Jul 3, Prio A5)
- gauntlet_mixed v2.1: Stabilitäts- + Champion-Quittung (chance-gated) jetzt
  identisch zu gauntlet.py; Selbsttest PASS (synth20 29s: STABLE 1.00,
  champion=forge; CALCE-LOCO 10s: 0.858, +0.086 CI[+0.010,+0.164], STABLE).
- audit.py cwru mit Grammatik v2.1: **12/12 PASS** (erstes Mal auf cwru mit
  v2.1; Dedup wirkt: 29/30 effektiv unique im Top-30; label-shuffle 0.872→0.060;
  nested-vs-biased-Gap +0.038; Determinismus exakt).

## SESSION Jul 3 (nachm.) — Methoden-Offensive auf die Nulls + Prozess-Ausbau
Neue Quittungen (alle Läufe): CLF-Robustheit (LogReg auf Champion-Features,
Modell-Abhängigkeit sichtbar) + Hash-Chain-Ledger (receipt_ledger.py, jeder
Gauntlet-Lauf manipulationssicher verkettet; tip-Hash extern ankerbar).

**RQA-FAIR KOMPLETT (MFPT nachgezogen):** best fair RQA 0.822 vs lean 0.780,
gepaart −0.042 CI[−0.137,+0.047] = TIE, lean 264× billiger. Finale ehrliche
Moat-Tabelle: ECN TIE / CWRU RQA CI-fest besser / MFPT TIE — lean ist nie
CI-fest genauer, aber immer ~200–260× billiger. Kosten-Frontier-Story steht.

**★★ GAS-NULL GEKNACKT — Multi-Kanal-Grammatik (gas_multichannel.py, These #4):**
4. Grammatik-Slot Kanal-Kombinierer (logratio/chdiff/mean8/std8) × Grammatik
v2.1 × {shape,level}, Kapazitäts-Gate 1000/151800, 4 Hz. Nach 5× Einzel-Kanal-
Null jetzt:
- GAS-id LOGO: **0.573 CI[0.531,0.623]** (Chance 0.250), gepaart vs single-ch
  lean +0.273 CI[+0.169,+0.386], STABLE 0.95, Champion=mc-forge.
- **GAS-id-LODO (Erfolgskriterium RESTART B6): 0.621 CI[0.449,0.738] CI-fest
  über Chance** — Gas-Signatur transferiert auf physisch anderes Sensor-Exemplar.
  Gewinner-Familien = Kanal-Muster (chdiff(1,5)-crest, logratio(5,7)-hent) =
  Hypothese „Multi-Kanal-Muster fehlte" direkt bestätigt.
- Caveats: Gerät B5 transferiert schwach (0.290); n=5 Gruppen (dünnes CI);
  Kombinierer-Slot noch nicht in feature_forge-Kern integriert (Experiment-Skript).

**★ TIMESCALE-SCREEN (These #3 als Standard-Werkzeug, timescale_screen.py) —
IMS-fault-Null REVIDIERT:** Screen über 5 Null-Banken; IMS-fault zeigte
pool4-Spike → Nachtest: **lean 0.755, perm-p 0.0050 (200 Perms)** — das alte
Null (0.625 n.s.) war ein Zeitskalen-Artefakt der Standard-Fensterung.
DCASE-valve-anomaly-Flag (0.238→0.562 monoton) bestätigte sich im 200-Perm-Test
NICHT (p=0.10) → ehrlich offen; echter Test braucht Source-Level-Rebuild.
HYD-accum/DCASE-pump: flat = Null ist kein Fenster-Artefakt auf dieser Achse.

**Lehre (Kern der „bessere Standards"-These):** 2 von 7 ehrlichen Nulls waren
keine Physik-Grenzen, sondern AUSLESE-Grenzen (fehlender Kanal-Slot; falsche
Zeitskala). Die Standard-Verfahren (Einzel-Kanal, feste Fensterung) verstecken
Signal — unsere Suchdimensionen finden es und liefern die Quittung mit.

**★ PHYSIK #10 — VULKAN-TREMOR (Kilauea, harvest3_loaders.py, IRIS HV):**
Eruptions-Zustand aus 10 s Einzel-Stations-Seismik-Textur (HVO-dokumentierte
Labels, beide Klassen über 2018+2023 gespreizt — GRIDFREQ-Lektion designseitig
drin; identische Fenster-Regel beide Klassen; bank_audit PASS).
- UWE: lean **0.767 CI[0.575,0.933], perm-p 0.033**, CLF-robust (LogReg 0.767
  HOLDS), Champion=lean (Forge tie, STABLE 0.73).
- RIMD (2. Station, unabhängige Replikation): **0.773, perm-p 0.033** ✓.
- ABER Cross-Station-Transfer VERSAGT (UWE→RIMD 0.433, RIMD→UWE 0.608 n.s.):
  Nahfeld-Tremor trägt Site-Effekte — im Kontrast zu SEIS (teleseismisch,
  transferiert frei). **Stützt These „Transfer folgt der Physik" von der
  anderen Seite.** Ehrlicher Claim: stations-lokal kalibrierbares
  Eruptions-Readout, kein universeller Tremor-Fingerprint.
- Caveat: 2023-Pausen-Tage teils als eruptiv gelesen (Unrest zwischen
  Episoden?); 15/16 Tage UWE (1 Skip).

**Score nach Jul 3: 17 Banken (+GAS-MC, +VOLC), 10 Physiken, 9 CI-feste
Signale (+GAS-MC-LODO, +IMS-fault@pool4, +VOLC), 5 ehrliche Nulls (2 der 7
alten Nulls als Auslese-Artefakte revidiert), 4 Methoden-Thesen (Window-Mode,
Norm-Modus, Zeitskala, KANAL-KOMBINATOR).**

**★★ DCASE-valve-anomaly-Null GEKNACKT (Jul 3 abends) — Auslese-Artefakt #3:**
Source-Level-Varianten (load_dcase_valve, identische Regel je Klasse, 200 Perms):
| Variante | lean | perm-p |
|---|---|---|
| even/pool1 (Baseline) | 0.238 | 0.99 |
| **peak/pool1 (energie-gepeakt)** | **0.875** | **0.0050** |
| even/pool16 | 0.438 | 0.59 |
| peak/pool16 | 0.475 | 0.49 |
Mechanismus: Ventil-Anomalie lebt in SPARSAMEN IMPULSEN; gleichmäßige Fenster
treffen Stille (0.238 = Anti-Signal). Pooling hilft NICHT → der
timescale_screen-Flag von heute Mittag zeigte auf die richtige Bank aus dem
FALSCHEN Grund (Pooling verbesserte indirekt die Impuls-Abdeckung). Lehre:
Screen = Hinweisgeber, Auslese-FAMILIEN-Test (sampling-Mode) findet die Ursache.
Sampling-Mode = Erweiterung von Methoden-These #1 (Window-Mode).
**Nulls-Bilanz: 3 von 7 „ehrlichen Nulls" waren Auslese-Grenzen, nicht Physik
(GAS→Kanal-Slot, IMS→Zeitskala, DCASE-valve→Impuls-Sampling).**

**HYD-Multi-Kanal (hyd_multichannel.py, Kern-API-Dogfood): beide Targets
bleiben NULL** (accumulator MC 0.188 [0.094,0.292], valve-Textur MC 0.271
[0.219,0.323], Chance 0.25 — Chance-Gate griff korrekt, STABLE-Selektion
täuscht nicht). Kanal-Muster knackt HYD NICHT → diese Nulls sind nach 3
Auslese-Familien (Textur, Zeitskala via Screen, Kanal) robust; valve bleibt
nur phasen-aligniert lesbar (bekannt). Negativ-Beleg: der Kombinator-Slot
erfindet kein Signal — Quittungssystem hält auch auf der Gegenseite.

## FRESH-HOLDOUT-TRIPLE (Jul 3, frozen models + Ledger VOR jedem Load)
Prä-Registrierungs-Muster durchgängig: frozen family (lean-Duo, per Station/Obs
auf ALTER Bank gefittet) + Kriterium + Ledger-Hash vor dem ersten frischen Sample.
| Bank (frisch, unberührt) | Ergebnis | Verdikt |
|---|---|---|
| SEIS-backward 2021-23 (16 shallow/16 deep) | ANMO 0.938 CI[0.844,1.000]; KONO 0.844 CI[0.719,0.969] | ★ PASS beidseitig |
| GEOMAG-fresh Nov24-Jul26 (13 Kp>=7 + 13 Kp<=1.0-Tage, Lockerung dok.) | BOU day-acc 0.808 CI[0.654,0.962] (storm 13/13, quiet 0.615); FRD 0.885 CI[0.731,1.000] | ★ PASS beidseitig |
| VOLCANO out-of-time (50 HVO-Episoden Dez24-Jun26, 94 Seg/Station) | UWE 0.755 CI[0.681,0.824], ABER invertiert: quiet 0.978 / eruptiv 0.542; Paare 22/44 p=0.0003. RIMD 0.553 CI[0.457,0.652] FAIL | TEIL: UWE PASS, RIMD FAIL |

**Kernbefund: System generalisiert auf unberührte Daten in 2 von 3 Physiken voll.**
VOLCANO ehrlich: frozen 2018-23-Modell (Lavasee/LERZ-Tremor) liest das NEUE
episodische Fountaining-Regime schwach (eruptiv 0.542) — Regime-Shift, deckt
sich mit bekannter Site-/Stil-Lokalität. Quiet-Erkennung 0.978 (UWE) zeigt:
Pausen-Textur ist stabil über 8 Jahre. RIMD zusätzlich durch Gaps (4 Fetch-Fails).
**Konsequenz Ep-51: Original-Prereg (17eb13d) bleibt eingefroren und wird
ausgewertet wie registriert; ZUSÄTZLICH legitime PREREG-2 möglich = Modell auf
frischer Episoden-Bank (Caches data/volcano/fresh_oot) trainieren + einfrieren,
solange Ep 51 nicht eingetreten ist (Stand Jul 2: Prognose 8.-15. Jul).**
Skripte: ep51_prereg.py (freeze/apply), volcano_fresh.py, geomag_fresh.py,
seis_fresh.py. Ledger-Tips: prereg 8642426b/06ab2661/3b24f800/23816fab;
Resultate 0e36a2ab (SEIS), 6e49b7d8 (GEOMAG), de943693 (VOLCANO).

## EP-51-PREREG-2 (Jul 4, EINGEFROREN VOR Episode — Ledger b4677fdf)
USGS-Check Jul 4: Ep 51 NICHT eingetreten (HVO Jul 3: Pause seit 27. Jun,
Prognose 9.–15. Jul, Inflation seit 2. Jul wieder) → zweite Prä-Registrierung
legitim. `ep51_prereg2.py`: gleiche lean-Familie + Modellklasse wie Prereg-1
(verbatim, keine Selektion), aber trainiert auf der FRISCHEN Episoden-Bank
(fresh_oot-Caches, 94 Segmente/752 Fenster pro Station, Labels aus
volcano_fresh.plan(), Fetch-Fails ausgeschlossen — kein Netz-Zugriff).
**LOGO-Sanity (Episode=Gruppe, VOR Freeze, Ergebnis-unabhängig registriert):
UWE 0.851 (eruptiv 39/48=0.813, quiet 41/46=0.891), RIMD 0.830 (eruptiv
37/48=0.771, quiet 0.891)** — registrierte Erwartung bestätigt: episoden-
trainiert liest eruptiv deutlich >0.542 (frozen-2018-23-Wert). Apply-Protokoll
identisch Prereg-1 (K<=6 stündl. Segmente, quiet -3d/-4d, Majority über 8
Fenster, Kriterium >=ceil(0.75K)/Klasse/Station, Unrest-Fallback-Caveat) +
Zusatz: Apply-quiet-Segmente mit Trainings-Fenster-Überlappung werden gedroppt
und berichtet. Prereg-1 (17eb13d) UNBERÜHRT — beide werden ausgewertet wie
registriert; Prereg-1 testet das 8-Jahre-alte Modell, Prereg-2 das Regime-
Modell. Spec+Modelle+Bank-Hashes in frozen/ep51p2_*, Ledger-Tip b4677fdf.

## VOLCANO-PRÄKURSOR (Jul 4, volcano_precursor.py, Prereg 8996bb7d VOR Load)
Frage: Textur kurz vor Episodenstart vs. Mitte-Pause (HVO forecastet nur über
Tilt, 7-Tage-Fenster = deren Schwäche). Bank: 50 Episoden, PRE −2h (primär)/
−6h/−12h (exploratorisch registriert) vs. Mid-Pause same-clock, 544 Segmente
gecacht (1,1% Fetch-Fails). **AMENDMENT (317bce69, geledgert VOR Verdikt):
Audit-Checks 6/7 deckten auf, dass group_perm_p ein Label pro Gruppe annimmt —
paired Bank verletzt das strukturell → exakter binomialer Paired-Sign-Test
ersetzt den perm-Test; gauntlet-perm-p ohne Verdikt-Gewicht. Checks 6/7 =
expected-FAIL auf paired Banken (Werkzeug-Lücke, nicht Leak: 4/5/8/9/10 PASS).**
| Offset | UWE | RIMD | Verdikt |
|---|---|---|---|
| −2h PRIMARY | 0.567 CI[0.465,0.670] p=0.146 | 0.522 CI[0.418,0.629] p=0.087 | **NULL beidseitig** |
| −6h expl. | 0.551 p=0.226 | 0.539 p=0.560 | nichts |
| −12h expl. | **0.656 CI[0.556,0.756] p=0.0080** | **0.656 CI[0.567,0.744] p=0.0033** | ★ Cross-Station-Hit |
Forge knackt −2h auch nicht (nested 0.512/0.481 ≈ Chance) → der −2h-Null ist
Physik, kein Feature-Problem. **★ −12h: beide unabhängigen Stationen
replizieren denselben Effekt** — Textur ~12h vor Start ≠ Mitte-Pause.
Interpretation offen (Präkursor ODER Späte-Pause-Gradient) — beides wäre
Timing-Information gegen das 7-Tage-Fenster. Ledger affbdc1d.
**★ PREREG-3 (89d7773a, EINGEFROREN VOR Ep 51): prospektiver −12h-Test am
ungesehenen Ereignis.** Selektions-Ehrlichkeit registriert (−12h NACH
exploratorischem Blick gewählt → prospektive Bestätigung nötig; n=1 = Demo-
Quittung, keine Statistik). Protokoll: PRE=[start−12.5h,start−12h] vs. MID der
aktuellen Pause same-clock; primär ORDERING pre-prob(PRE)>pre-prob(MID),
strikt beide Majorities; UWE primär, RIMD Replikation.
`volcano_precursor.py apply51 <start-UTC>` nach HVO-Dokumentation.

## ZEIT-FAKTOR-PROGRAMM (Jul 4/5, Prio 2b a+b — beide preregistriert VOR Load/Readout)
### (a) PAUSE-PHASE-ORDINAL (pause_phase.py, Prereg 887a5723, Verdikt 2e8bd867)
Disambiguiert den −12h-Hit (affbdc1d): late-spezifisch oder Pausen-Gradient?
Neuer 3. Phasenpunkt EARLY = prev_end+24h auf der −12h-Clock (84 frische
IRIS-Segmente, 2 Fails = Datenlücke 3. Jun beide Stationen → 41/42 Pausen).
Scorer = registrierte Präkursor-Maschinerie verbatim (LOGO-RF pre-vs-mid,
EARLY nie im Training); primär = exakter ZWEISEITIGER Sign-Test EARLY vs MID.
| Station | early>mid | sign_p(2s) | late>mid in-fold | monotone | pre-prob early/mid/pre |
|---|---|---|---|---|---|
| UWE PRIMARY | 22/41 | **0.755** | **30/41** | 10/41 | 0.457/0.457/0.542 |
| RIMD Replikation | 26/41 | 0.117 | **29/41** | 9/41 | 0.473/0.421/0.556 |
**Registrierte Map → LATE-PAUSE-SPEZIFISCH:** EARLY≈MID beidseitig, late>mid
persistiert in denselben Folds. Gradient-Lesart hätte early<mid verlangt —
RIMD-Trend zeigt wenn überhaupt early>mid. **Die Präkursor-seitige Lesart des
−12h-Effekts überlebt** → erhöht den Wert von PREREG-3 (prospektiver −12h-Test
an Ep 51). Gauntlet auf early-vs-mid (NEW-bank-Regel, paired-Caveats 317bce69):
UWE lean 0.569 (RF-only-Flag, LogReg 0.555), forge 0.483≈Chance; RIMD forge
nested 0.520, paired forge−lean −0.049, Champion lean → early vs mid ist auch
für Forge kaum trennbar, konsistent mit dem Sign-Test-Null.
### (b) PRECURSOR-SAMPLING — Impuls-Zweittest (precursor_sampling.py, Prereg 3c979096, Verdikt 71bdf123)
Methoden-These #1 auf den Präkursor-Banken (peak_sub verbatim aus
readout_screen._sub; Readout-Familie #2, −2h-lean-NULL bleibt wie registriert):
| Bank | acc | CI | pairs | sign_p | Verdikt |
|---|---|---|---|---|---|
| UWE −2h PRIMARY | 0.533 | [0.433,0.630] | 21/44 | 0.674 | **NULL — Impuls-Sampling rettet −2h nicht** |
| RIMD −2h Repl. | 0.644 | [0.554,0.733] | 29/44 | 0.024 | isoliert Kriterium erfüllt, ohne Primär KEIN Befund (Multiplizität registriert) |
| UWE/RIMD −12h expl. | 0.578/0.622 | — | 26/45, 30/45 | 0.186/0.018 | **schwächer als lean (0.656)** |
Befund: der −12h-Effekt ist NICHT impuls-getragen — er lebt in der
Vollfenster-Textur (peak_sub verliert Signal). −2h bleibt Physik-Null auch
unter der zweiten Readout-Familie.
### Prozess-Quittung
pause_phase.py `run` lief 3h17 (Audit+Gauntlet inline, ~10-min-Regel verletzt);
stdout durch `tail -30` gekappt → UWE-Konsole weg, aber ALLE Verdikte/Gauntlets
im Ledger (PAUSEPHASE-UWE/RIMD + PAUSE-PHASE-ORDINAL). Fix künftig: Gauntlet
separat mit eigenem Log starten, nie durch tail pipen.

## MOX-DRIFT — Physik #11: Chemical-Sensing Drift-Invarianz (Jul 5)
Unabhängige-Physik-Generalisierung (Prio 2b(c)). Zenodo 15681119: 62 SnO₂-
Sensoren, 700 Expositionen über 39 Sessions/12 Monate, 3 Analyte. **Frischer,
vor Freeze nie auf Label-Trennbarkeit eingesehener Holdout** (data/mox, SHA-frozen).
Prereg VOR Readout: Ledger 483bf8c7 (spec 13723ba5), Commit cb04013.
Frage: überlebt eine drift-invariante Kanal-Muster-Signatur den 12-Monats-Drift?
Split = **leave-one-DAY-out (LODO, 39 Zeit-Gruppen)** — gehaltener Tag = ungesehener
Zeitpunkt → Chance dort schlagen = Drift-Invarianz. Reuse der Multichannel-
Grammatik (logratio/chdiff, aus gas_multichannel).

| Readout | Chance | MC-FORGE LODO [CI] | lean (single-ch) [CI] | gepaart mc−lean [CI] | Stabilität | Champion |
|---|---|---|---|---|---|---|
| **MOX-DRIFT-id** | 0.333 | **0.591 [0.554,0.629]** | 0.510 [0.482,0.539] | **+0.081 [+0.030,+0.134]** | top1 1.00 | **mc-forge** |
| Shuffle-NULL (Label ↻ within-day) | 0.333 | 0.349 [0.321,0.377] | 0.338 [0.305,0.370] | +0.011 [−0.036,+0.059] | — | NULL |

**Verdikt: DISCOVERY (PASS)** — alle 4 frozen Kriterien erfüllt: CI-lo 0.554 ≫ 0.333;
Champion=mc-forge (Multichannel schlägt Single-Channel CI-fest); Stabilität 1.00;
Shuffle-NULL ≈ Chance (kein Pipeline-Leck). **Schließt die alte GAS-id-Grenze**
(„Multi-Kanal-Muster fehlt", NULL/marginal, RESULTS §GAS): dieselbe logratio/chdiff-
Grammatik, gebaut gegen genau diese Lücke, trägt auf einer NEUEN 62-Kanal-e-Nose
quer durch 12 Monate Drift = Reuse-These bestätigt.
**Ehrliche Caveats:** (a) 0.591 ist real aber moderat, gepaarter Win über lean nur
+0.081; (b) 1 Gerät → nur Zeit-/Session-Drift getestet, KEIN Device-Transfer (LODO
über Geräte nicht möglich, dataset-inhärent); (c) Konzentration in Analyt-Label
gepoolt (Identität soll konz-robust sein). Ledger-Result-Tip ecc6f37f.

## BATTERY-TRANSFER — Physik #12: Device-Transfer auf frischen Batterie-EIS (Jul 5)
Schließt den offenen MOX-#11-Gap (1 Gerät → nur Zeit-Drift). Datensatz = OSF j2sn4
(DOI 10.17605/OSF.IO/J2SN4, Data-in-Brief 2025/26): 24 NMC/Graphit-Zellen, EIS @5 SOC
× 6 Aging-Tage. **Neue Input-Modalität: Impedanz-Spektren** (|Z| vs Frequenz, nicht
Zyklus-Kurven). Prereg VOR Readout: Ledger 3faebe49 (spec 170dcad4), Commit edcd9ba.
Frage: trägt eine Aging-Signatur (fresh d∈{0,10,20} vs aged {40,70,90}) über den
Zell-zu-Zell-Impedanz-Unterschied? Split = **leave-one-CELL-out (24 Zellen) = echter
Device-Transfer** (gehaltene Zelle nie gesehen). 935 Spektren, 470/465 balanciert.

| Readout | Chance | FORGE LOCO [CI] | lean [CI] | gepaart forge−lean [CI] | Stab. | perm-p | Champion |
|---|---|---|---|---|---|---|---|
| **BATTERY-TRANSFER-aging** | 0.500 | **0.718 [0.676,0.763]** | 0.568 [0.532,0.605] | **+0.150 [+0.108,+0.193]** | 0.96 | **0.0050** | **forge** |
| Shuffle-NULL (Label ↻ within-cell) | 0.500 | 0.508 [0.465,0.550] | 0.497 [0.465,0.526] | +0.011 [−0.035,+0.059] | 0.79 | — | NULL |

**Verdikt: DEVICE-TRANSFER PASS** — alle 3 frozen Kriterien: forge CI-lo 0.676 > 0.5;
perm-p 0.0050 < 0.05 (0/200 within-cell-Permutationen ≥ obs); Shuffle-NULL ≈ chance
(kein Leck). Der Forge liest eine Aging-Signatur aus rohem EIS-Spektrum, die auf
**komplett ungesehene Zellen** überträgt und lean CI-fest schlägt (+0.150). Gemeinsam
mit #11 ist die Generalisierungs-Matrix jetzt in BEIDEN Achsen belegt: Zeit-Drift (MOX)
UND Device-Transfer (Batterie).
**Ehrliche Caveats:** (a) fresh-vs-aged ist 2-Klassen-grob (chance 0.5), 0.718 real aber
moderat; (b) Transfer über ZELLEN gleicher Chemie/Protokoll, KEIN Chemie-Transfer;
(c) SOH-tertile (sekundär exploratorisch, FCC.xlsx) noch nicht ausgewertet — offen;
(d) S2/S18 (Paper-exkludiert) mitgeführt → 24 statt 22 Zellen, unkritisch für LOCO.
Ledger-Result-Receipt: BATTERY-TRANSFER + BATTERY-TRANSFER-PERMP.
Durabilität: launchd `com.signalmap.batterytransfer` (RunAtLoad+30min), idempotent,
DONE-Marker gesetzt → reboot-fest, kein Re-Run-Spam.

### #12 SEKUNDÄR — SOH-tertile aus EIS (Device-Transfer, feineres Ziel)
Prereg VOR Readout: Ledger 1905f403 (spec a06c8dad), Commit b7ebd63. Label = SOH-tertile
(SOH=C_discharge(cell,day)/C(cell,0) aus FCC, Duplikat-Tage gemittelt; globale 33/67-Quantile),
gleiche EIS-Bank, LOCO über Zellen, chance 0.333. 859 Spektren (nan-SOH-Tag gedroppt),
Tertil-Cuts SOH 0.968/0.990 (enges Degradations-Band).

| Readout | Chance | FORGE LOCO [CI] | lean [CI] | gepaart [CI] | Stab. | perm-p | Champion |
|---|---|---|---|---|---|---|---|
| **BATTERY-SOH-tertile** | 0.333 | **0.559 [0.521,0.597]** | 0.387 [0.347,0.428] | **+0.172 [+0.128,+0.212]** | 0.96 | **0.0050** | **forge** |
| Shuffle-NULL | 0.333 | 0.342 [0.316,0.366] | 0.341 [0.305,0.375] | +0.001 [−0.044,+0.047] | 0.96 | — | NULL |

**Verdikt: SOH-tertile PASS** — forge CI-lo 0.521 > 0.333; perm-p 0.0050 (0/200); Shuffle-NULL
≈ chance. Der Forge liest **feine SOH-Stufen** (Tertile im engen 0.968–0.990-Band) aus rohem
EIS und überträgt auf ungesehene Zellen, schlägt lean CI-fest (+0.172). Stärkere/feinere
Bestätigung des Device-Transfers als das grobe aging-2-Klassen-Primär; kohärent (SOH-Tertil
trackt Degradation). **Damit #12 sekundär GESCHLOSSEN.**
_Ledger-Hygiene: erster BATTERY-SOH-Eintrag war ein nan-Bug-Lauf (globale statt nan-robuste
Quantile) — vom korrigierten zweiten Lauf abgelöst; append-only Kette, daher offen dokumentiert
statt gelöscht. Fix = nanquantile + nan-SOH-Spektren droppen (Loader-Bug, keine Re-Selektion)._

## HÄRTUNG #11+#12 — Konfirmations-Pass (Jul 5, harden_1112.py, Ledger HARDEN-1112 + -WINDOWPROV)
Ziel: maximale Sicherheit dass Daten & Resultate bestätigt sind. Alle Checks label-agnostisch
oder Modell-Swap (können Confidence nur SENKEN — ehrliche Richtung, keine Re-Selektion).

| Check | MOX #11 | BATTERY #12 | Bedeutung |
|---|---|---|---|
| Provenienz SHA-256 | Dataset.zip 6962b58e… (141MB) | csv.zip 6ffda60a…, FCC.xlsx ecc93edd… | Rohdaten tamper-evident eingefroren |
| Harness neg-noise | \multicolumn — 0.067 (chance 0.167) | | Pipeline erfindet kein Signal aus Rauschen |
| Harness pos-planted | \multicolumn — 1.000 recovered | | Pipeline verliert kein echtes Signal |
| Linear-Swap (LogReg) | **0.603 CI[0.567,0.639]** > 0.333 | **0.591 CI[0.549,0.640]** > 0.5 | Signal Modell-unabhängig (überlebt RF→linear) |
| Noise-Placebo (ganze Pipeline) | 0.338 ≈ chance | Champion NULL (0.529 CI-lo 0.490<0.5) | Feature-Build+Selektion+LODO auf Rauschen → chance |
| exact-dup über Gruppen | **0** | **0** | keine Datei/Messung über Folds dupliziert |

**window-provenance |r|>0.999 (15322 MOX / 4663 BATTERY near-identical über Gruppen): BENIGNER
Fehlalarm, aufgelöst.** exact-dup=0; die near-Twins sind label-BALANCIERT (same-label MOX 0.405 /
BATTERY 0.599 → ~Hälfte gegenteiliges Label) → Nähe im Rohraum ist **label-uninformativ → kann das
Label über den Gruppen-Split nicht leaken**; near-Twins verdünnen, inflatieren nicht. Der |r|>0.999-
Schwellwert ist für verrauschte Zeitreihen kalibriert und false-positivt auf glatter, reproduzierbarer
e-Nose/EIS-Morphologie. Der Forge erreicht 0.59/0.72 TROTZ near-identischer Rohfenster → er extrahiert
echte subtile Struktur, die die Roh-Korrelation wegmittelt. Korroboriert durch exact-dup=0 +
Shuffle-NULL≈chance + Noise-Placebo≈chance + Linear-Swap.

**Gesamt-Konfidenz #11+#12: hoch bestätigt.** Vorregistriert-vor-Readout, LODO device/time-transfer,
chance-gated paired-CI, within-group perm-p 0.005, Shuffle-NULL≈chance, Stabilität 0.96, Modell-
unabhängig, Rohdaten SHA-frozen, kein exact-Leak, Harness-Kontrollen sauber. Verbleibende ehrliche
Grenzen unverändert (2-Klassen-grob bei aging; Zell- nicht Chemie-Transfer; enges SOH-Band).

## ZEIT-FAKTOR — IMS-RUL: CSD-Theorie-Anker auf Lager-Run-to-Failure (Jul 8, ims_csd.py)

**Fortsetzung des Zeit-Faktor-Programms nach dem ehrlichen Vulkan-Downgrade** (User-vorentschieden:
„falls Vulkan-Timing scheitert → Zeit bei anderen Familien: IMS-RUL, CALCE, GEOMAG-Onset"). Getestet
wird derselbe **Theorie-Anker Critical Slowing Down** (Scheffer/Dakos Early-Warning, `csd.py`) — diesmal
auf der Domäne, wo CSD **lehrbuchmäßig ERSCHEINEN müsste**: mechanische Degradation bis zum Ausfall.
Gerichtet, lern-frei, selektions-frei → Fehler-Modus orthogonal zu jedem gelernten Klassifikator.
Reuse: `csd.py` ar1/logvar + Freeze-Muster; Delta = Run-to-Failure-Trajektorie + Rolling-Kendall-τ +
Fourier-Phasen-Surrogat-Null. Beide Indikatoren **prereg-VOR-Readout** (Ledger IMS-CSD-RMS-PREREG befc598b,
IMS-CSD-KURT-PREREG 396bf8d7).

**Design:** IMS/NASA (Rexnord ZA-2115, 20 kHz 1-s-Snapshots ~alle 10 min bis Ausfall). 3 Rigs
(1st_test 2156/8ch, 2nd_test 984/4ch, 4th_test=dok. Test 3 6324/4ch). Ground-Truth-Ausfälle (IMS_FAILED):
1st-b2 (inner race), 1st-b3 (roller), 2nd-b0 (outer race), 4th-b2 (outer). Healthy Lager derselben Rigs
= **Spezifitäts-Kontrolle** (gleiche Laufzeit, gleiche Betriebs-Drift, NICHT ausgefallen). Slow-Variable
HI[t] = per-Snapshot-Indikator über den chronologischen Lauf; Gauss-detrend (σ=0.05·N); Rolling-Fenster
0.5·N; pro Fenster ar1 + logvar; Kendall-τ vs. Zeit; Signifikanz per Lager via Phasen-Surrogat (N=500,
1-seitig). PASS(Lager) = τ_ar1>0 & p<0.05 & τ_var>0 & p<0.05.

**Ergebnis — CSD-NULL bei BEIDEN Indikatoren, aber die 2 orthogonalen Auslesen erzählen konsistent:**

| Indikator | failed CSD+ | healthy CSD+ | ar1_rise failed | var_rise failed | Fisher p (f>h) | Verdikt |
|---|---|---|---|---|---|---|
| RMS (breitband, rig-global) | 0/4 | **2/8** | 0/4 | 2/4 | 1.0 | CSD-NULL |
| Kurtosis (impulsivität, failure-spezifisch) | **1/4** | **0/8** | 1/4 | 3/4 | 0.333 | CSD-NULL |

- **RMS:** kein failed Lager passt; 2 HEALTHY Lager (2nd-b1/b3) passen sogar → Spezifität invertiert.
  Ursache = **rig-weite Kopplung**: im Run-to-Failure vibriert der ganze Prüfstand mit → breitband-RMS-CSD
  ist eine Prüfstands-Eigenschaft, nicht failure-spezifisch. Die Kontrolle hat einen falschen CSD-Claim
  strukturell verhindert.
- **Kurtosis (failure-spezifischer Indikator):** Spezifität dreht in die RICHTIGE Richtung — **0/8 healthy**
  Falsch-Positive verschwinden, das einzige PASS ist ein failed Lager (2nd-b0). → Die RMS-Null war
  „Indikator zu rig-global", nicht „CSD abwesend".
- **ABER:** selbst mit Kurtosis erreicht nur der **Varianz**-Anteil breit Signifikanz (var_rise 3/4 =
  Impulsivitäts-Fluktuationen wachsen, ≈ trivial „es degradiert"); der eigentliche **CSD-AR1-Fingerabdruck
  bleibt der Flaschenhals (1/4)**, und n_failed=4 deckelt die Fisher-Power (1 vs 0 kann p<0.05 nicht
  erreichen). → Verdikt korrekt CSD-NULL nach dem gefrorenen Kriterium.

**Cross-Family-Befund:** Der gerichtete CSD-Pfad feuert weder auf dem Vulkan-Präkursor (`csd.py` NULL/NULL)
noch auf IMS-Run-to-Failure als bestätigter Frühwarner unter prereg-freiem, leck-freiem, spezifitäts-
kontrolliertem Test. Das ist ein **ehrlicher, disziplinierter Negativ-Beleg**: unser Theorie-Anker ist
stringent und produziert selten Falsch-Positive — die Spezifitäts-Kontrolle deckte auf RMS sogar den
rig-weiten Confound auf, der einen naiven Claim ausgelöst hätte. Determinismus: RMS-Readout exakt
reproduziert (0/4, 2/8). Ledger-Kette intakt (verify()=True). Ehrliche Grenzen: n_failed=4 (nur 3 Rigs
extrahiert, 4th≙Test3); a-priori gepinnte σ/Fenster nicht getunt; broadband-RMS/kurtosis sind 2 von vielen
möglichen Health-Indikatoren; pre-registrierbarer Nachfolger = defekt-Band-Envelope + Restriktion auf
Vor-Ausfall-Fenster (NEUE Prereg, nicht diese tunen).

## VULKAN-PRÄKURSOR — WEG 4+5 + HORIZON-SCAN (Jul 10, coherence.py/rqa_precursor.py/horizon_scan.py)

**User-Auftrag:** weitere, unabhängige Methodiken auf die Kilauea-Präkursor-Frage anwenden und den
Zeitrahmen (Vorlaufzeit vor Episodenbeginn) systematisch verkürzen/erweitern testen — "schaffen wir 24h
oder 12h, oder ist ein anderer Horizont relevanter". Zwei NEUE orthogonale Pfade + eine Horizont-Grid-
Erweiterung, alle prereg-VOR-Readout, alle auf bereits gecachte Daten wo möglich (0 neue IRIS-Calls für
Weg4/5), gezielte neue Fetches nur für die neuen Zeit-Offsets.

**Weg 5 — RQA (`rqa_precursor.py`, Ledger RQA-PRECURSOR-PREREG f183e141):** phasenraum-Rekurrenzstruktur
(pyunicorn, dim=3/tau=5/rr=0.1 fix, KEIN Grid-Search), Deskriptoren DET/LAM/diag-ENTR — mathematisch
unabhängig von Weg1 (gelernte Grenze), Weg2 (lineare AR1/Varianz), Weg3 (Fourier-Form). Lief auf dem
bereits gecachten Precursor-Bank (0 neue Fetches) bei 2h/6h/12h. **Ergebnis: NULL bei allen 3 Offsets,
beiden Stationen, Holm-Bonferroni-korrigiert über 9 Zellen/Station** (bestes rohes p=0.02 RIMD-6h-entr,
übersteht Korrektur nicht).

**Weg 4 — Cross-Stations-Kohärenz (`coherence.py`, Ledger COHERENCE-PREREG bd6d4dff):** erster RÄUMLICHER
Pfad (alle anderen sind Einzelstation) — magnitude-squared Kohärenz zwischen UWE- und RIMD-Hüllkurve
(zeitgleich, unverfälscht durch stations-individuelles Peak-Picking, reuse aus dem CSD-Envelope-Cache,
0 neue Fetches). Fehler-Modus strukturell orthogonal: eine Kopplungs-Verschiebung kann nicht durch
Rauschen einer einzelnen Station erzeugt werden. **Ergebnis: NULL** (PRE-vs-MID 21/38, sign_p=0.63;
EARLY-vs-MID identisch — keine Kopplungsänderung in keiner Pause-Phase).

**Horizon-Scan (`horizon_scan.py`, Ledger HORIZON-SCAN-PREREG b58d8bd5, Fetch 364 neue IRIS-Segmente,
27 erwartete Daten-Lücken FETCH-FAIL ehrlich geloggt):** zwei NEUE Offsets — **1h** (kürzer/direkter
actionable als der bisher kürzeste getestete Punkt) und **24h** (länger, testet ob ein früherer,
stabilerer Vorlauf existiert) — mit den 3 bereits gebauten fensterbasierten Methoden (Weg1 Textur, Weg3
Spektral, Weg5 RQA), da diese sich eine Fetch-Repräsentation teilen. 28 Zellen (2 Offsets × 2 Stationen ×
7 Deskriptoren) gemeinsam Holm-Bonferroni-korrigiert (strengste sinnvolle Korrektur für einen reinen
Scan).

| Offset | Station | Textur (Weg1) | Spektral (Weg3, 3 Deskr.) | RQA (Weg5, 3 Deskr.) |
|---|---|---|---|---|
| 1h  | UWE  | 27/43 p=.126 | centroid+spec_ent p≈.03 (roh) | alle n.s. |
| 1h  | RIMD | **33/43 p=.0006** ★ | alle n.s. | alle n.s. |
| 24h | UWE  | 23/42 p=.644 | spec_ent p=.008 (roh) | alle n.s. |
| 24h | RIMD | 26/42 p=.164 | alle n.s. | alle n.s. |

**Holm-korrigiert (28 Zellen gemeinsam) überlebt genau EINE Zelle: RIMD-1h-Textur (p=.0006).**

**Ehrliche Einordnung (registriert, nicht als Entdeckung verkauft):** dieser einzelne Treffer zeigt exakt
das Fehler-Muster, das den ursprünglichen −12h-Textur-Treffer (0.656) bereits einmal zu Fall gebracht hat
([[TRIANGULATION.md]], Jul 7 Downgrade): (a) **keine Stations-Replikation** — UWE (primär) zeigt am selben
Offset nichts (p=.126); (b) **keine Methoden-Konvergenz** — Weg3/Weg5 an derselben Station/demselben Offset
zeigen nichts (RIMD-1h spectral/RQA alle n.s.). Genau dieses Muster (Weg1 allein blinkt, orthogonale Pfade
schweigen) war beim −12h-Fund der Vorbote des Downgrades. Verdikt hier: **NICHT triangulations-fähig,
voraussichtlich derselbe Lernfenster-Textur-Artefakt-Mechanismus** — kein neuer Prereg für eine
prospektive Bestätigung angesetzt, da die Diagnose bereits am eigenen Datensatz eindeutig ist (im
Gegensatz zum −12h-Fall, wo erst die volle 4-Wege-Triangulation das Urteil brachte).

**Gesamtbild Zeitrahmen-Verkürzung (Stand Jul 10): über den gesamten getesteten Bereich 1h–24h vor
Episodenbeginn, 5 unabhängige Methoden (Textur/CSD/Spektral/Kohärenz/RQA), 2 Stationen — KEIN
Präkursor-Signal, das cross-station UND cross-method übersteht.** Der einzige vormals interessante Punkt
(−12h Textur, 0.656) wurde bereits Jul 7 ehrlich abgewertet. Die USGS-Mehrtage-Vorhersage (aktuell
rollend, Stand Jul 8: 11.–15. Jul) bleibt der einzige derzeit belastbare Zeitrahmen; unser
Methoden-Portfolio hat KEINE kürzere, verlässliche Alternative gefunden — ein disziplinierter,
mehrfach-korroborierter Negativ-Befund, kein Fehlschlag des Prozesses.

**Nächste ehrliche Schritte, falls fortgesetzt:** (a) 3h-Offset nachziehen (Lücke zwischen 2h/6h aktuell
ungetestet, würde die Kurve verdichten); (b) Weg2 CSD + Weg4 Kohärenz auf 1h/24h nachziehen (aktuell
bewusst ausgelassen, siehe Prereg "out_of_scope_registered", würde neue Envelope-Fetches brauchen); (c)
falls User-Priorität wechselt: Zeit-Faktor auf CALCE/GEOMAG-Onset (wie schon nach dem IMS-Ergebnis
vorgeschlagen) statt weiter im Vulkan-Fenster zu graben, da 5 Methoden × 6 Offsets bereits eine breite,
konsistente Nullfläche zeigen.

## ZEIT-FAKTOR — CALCE-Batterie + GEOMAG-Onset: CSD-NULL × 2 (Jul 11/12, calce_csd.py / geomag_onset_csd.py)

WS4-Abschluss, volle Verdikte + Receipts in `TIMEFACTOR_CALCE_GEOMAG.md`. Kurzfassung: derselbe
unveränderte CSD-Anker (ims_csd._csd_bearing, prereg-VOR-Readout: CALCE-CSD-CDCG/VDCG-PREREG
321da4ed/181dc8e9, GEOMAG-ONSET-CSD-PREREG 929e1460) auf die beiden vorentschiedenen Familien:
**CALCE** (In-Test-Entlade-Traces, aged n=29 vs fresh n=72): cdcg 0/29 & vdcg 0/29 CSD+, 0 FP auf
fresh, Fisher p=1.0 → CSD-NULL beidseitig. **GEOMAG** (12–24h vor Sturm-Onset, BOU+FRD, ~19/20 je
Klasse): BOU 1/19 vs quiet 2/19 (p=0.885, leicht invertiert), FRD 1/19 vs 0/20 (p=0.487) → CSD-NULL
beide Stationen. Alle 3 Readouts am 12. Jul byte-identisch re-run = Determinismus belegt (Ledger
dbed33cc/3bcc467c/c5b3db83). **Cross-Family-Bilanz: CSD 0/4 Familien (Vulkan, IMS-RUL, CALCE,
GEOMAG-Onset)** — stringenter Theorie-Anker, praktisch keine Falsch-Positiven, keine Frühwarnung in
unseren Familien/Fenstern; dieselbe Maschine zertifiziert Discoveries UND Nulls.

## DISTILL PREMIUM-FAMILIEN — RQA-Champion-Regel-Praxis-Case (Jul 13/14, distill_premium_case.py)

**Entscheid (PRIO 1, per Empfehlung):** RQA = distill-Premium-Familie mit Kosten-Quittung,
NICHT Forge-Slot; Kohärenz = 2. Premium-Familie (Follow-up, braucht Multi-Channel-Bank).
Bau: `signalmap/premium.py` (numpy-RQA dim=3/tau=5/rr=0.1, 6 Maße, pyunicorn-Parity rtol 0.05
in Suite gepinnt) + `distill(premium=…)`/CLI `--premium` mit Champion-Regel (Premium in
Deploy-Spec NUR bei paired-CI-festem Sieg über die base-Selektion). Suite 66/66.

**Prereg VOR Readout** (PREREG_DISTILL_PREMIUM_CALCE.md, Ledger DISTILL-PREMIUM-PREREG
5f0a92b9, Commit 2391836): PRIMARY = INCLUDED auf beiden RQA-Gewinner-Banken; EXCLUDED
explizit als gültiges Ergebnis registriert. Fixe Parameter (C=50, kmax=5, n_perm=200, seed=0),
kein Grid, kein Nach-Tuning.

**Ergebnis: PRIMARY beidseitig WIDERLEGT — Champion-Regel verdiktet 2× EXCLUDED (ehrlich):**
- CALCE (18 Rec/306 Win): distill PASS (nested 0.892, perm-p 0.005, NULL 0.366≈chance,
  +0.351 vs lean). Premium: base 0.935 → aug 0.905, paired **−0.030 CI [−0.072, +0.006]**
  → EXCLUDED. Kosten-Quittung: 58.05 vs 0.319 ms/window (~182×).
- HYD-cooler (18 Rec/90 Win): distill PASS (nested 0.733, perm-p 0.005, NULL 0.311).
  Premium: base 0.800 → aug 0.833, paired **+0.033 CI [−0.022, +0.100]** → EXCLUDED.
  Kosten: 59.87 vs 0.079 ms/window (~755×).

**Interpretation (Frontier-Präzisierung):** Die RQA-CI-Wins vom 12. Jul galten **vs lean
(2 Features)**. Gegen eine volle distill-base-Selektion (CALCE 0.935 > RQA-allein 0.860)
verschwindet der Vorsprung — RQA-Premium zahlt sich dort NICHT aus, und die Quittung
verweigert die ~10²× teurere Familie automatisch. Das IST das Produkt-Verhalten
(„receipt refuses premium where it doesn't pay"), kein Fehlschlag der Familie an sich.
Offener Kandidat für INCLUDED = CWRU (RQA 0.961 > forge 0.902 fair) — NEUE Prereg nötig,
nicht post-hoc nachgeschoben. Ledger: DISTILL-PREMIUM-CALCE + -HYDCOOLER, tip 43624253.
Determinismus: Seeds fix (RF 0, Bootstrap 0, NULL 1); Repro-Befehl im Prereg-File
(voller Re-Run ~6.6 h, nicht erneut ausgeführt).

**14. Jul — CWRU: ERSTER INCLUDED-CASE (prereg'd VOR Readout,
PREREG_DISTILL_PREMIUM_CWRU.md, Freeze-Ledger cb7653a3, Commit 768a2d0):**
- CWRU (24 Rec/2849 Win, 6 Klassen): distill PASS (nested 0.874, perm-p 0.005,
  NULL 0.082, chance 0.167, +0.042 vs lean). Premium: base 0.914 → aug **0.980**,
  paired **+0.066 CI [+0.033, +0.106]** → **INCLUDED**, `spec.premium=["rqa"]`.
  Kosten-Quittung: 66.03 vs 0.113 ms/window (**~585×** — der Preis steht im Receipt).
- Ehrlichkeits-Klausel hielt: Produkt-Default m3τ5 (NICHT die 0.961er-τ10-Config),
  kein Grid, kein Nach-Tuning; Lauf 2317 s. Ledger DISTILL-PREMIUM-CWRU, chain OK
  (tip 0bf18785). Damit ist die Premium-Story komplett: **die Quittung verweigert
  (CALCE/HYD) UND lässt zu (CWRU) — beides CI-gated, beides signiert.**

## MULTI-CHANNEL DISTILL + KOHÄRENZ-PREMIUM — 2. INCLUDED (16./17. Jul, SPEC→Impl→Prereg→Lauf)

**Bau (TDD nach SPEC_MULTICHANNEL_DISTILL.md, Suite 76→102, 1-Kanal-Pfad byte-identisch
gepinnt):** `window()`/`load_bank`/spec.json multichannel (2D C×1024, detrend+z-norm PRO
Kanal, CSV-Header via multichannel.load_channels, 2D-.npy mit Orientierungs-Heuristik +
`--channel-axis`), Primary-Kanal-Konvention (base-Grammatik sieht NUR ch0 → rein additiv),
`PremiumFamily.needs_channels` (coherence=2, verweigert LAUT auf 1-Kanal-Bank),
`coherence_features` **fixe Produkt-Config c128b2** (coherence_fair CONFIRMED-aug,
Parity-Pin gegen coh_feats exakt). CLI-Gap geschlossen: `fit --spec spec.json --bank dir/`
+ `monitor --detector det.json --bank dir/` (DistilledDetector jetzt Produkt-Oberfläche,
Threshold kalibriert sich aus dem Healthy-Envelope). CLI-E2E live verifiziert
(distill→INCLUDED→fit→monitor: 100% Alerts auf gekoppelt, 0% auf healthy).

**Prereg VOR Readout** (PREREG_DISTILL_PREMIUM_COHERENCE.md, Ledger
DISTILL-PREMIUM-COH-PREREG acad1724, Commit 7bceaeb): EINE fixe Config (c128b2, GAS-
CONFIRMED c256b8 bewusst NICHT übernommen), Kanal-Reihenfolge deklariert (HYD ch0=PS1,
GAS natürlich), Fenster-Normierung als Abweichung zu coherence_fair deklariert.

**Ergebnis — die Quittung verweigert UND lässt zu, jetzt auch bei Kohärenz:**
- **HYD-cooler (18 Rec/72 Win, 6ch): 2. INCLUDED.** distill PASS (nested 0.694, perm-p
  0.005, NULL 0.361≈chance 0.333). Premium: base 0.750 → aug **0.889**, paired **+0.139
  CI [+0.042, +0.236]** → INCLUDED, `spec.premium=["coherence"]`, Kosten ~81× (991 s).
- **GAS-id (20 Rec/159 Win, 8ch): EXCLUDED** (wie im Prereg als möglich deklariert —
  die fixe c128b2-Config kostet GAS die c256b8-Punkte): base 0.429 → aug 0.435, paired
  +0.006 CI [−0.100, +0.113], ~72× Kosten (1546 s). distill selbst PASS (perm-p 0.005,
  NULL 0.252≈chance 0.250) — base+Prädikate tragen GAS auf 0.429 > lean 0.237.
  Ledger DISTILL-PREMIUM-COH-HYDCOOLER/-GASID, Reports logs/distill_premium_coh_*.

**Produkt-Befund (offen, notiert):** `passed`-Gate hängt am base-nested-LOGO. Auf einer
Bank, wo NUR die Premium-Familie das Signal trägt (synthetischer CLI-E2E-Fall), ist das
Verdikt FAIL + Premium-INCLUDED gleichzeitig — ehrlich, aber Report sollte das erklären.

## EP-51 PROSPEKTIV — Onset 15. Jul 18:30Z (HVO), 3 registrierte applies (17. Jul)

Ep 51: Fountaining 2026-07-15T18:30Z bis 2026-07-16T02:46Z (8.3 h, HVO-dokumentiert;
Präkursor-Overflow ab 15. Jul 00:51Z). Watcher-Flag 15. Jul 10:49Z. Alle 3 Preregs
prospektiv angewendet, KEIN Nach-Tuning:
- **ep51_prereg (Weg-1-Klassifikator): FAIL** — UWE eruptive 1/6, RIMD 0/6 (quiet 6/6
  beide; 3 Fetches segmentiert=FAIL gezählt, konservativ). Ledger c60d8a95.
- **ep51_prereg2: FAIL** — RIMD eruptive 4/6 aber quiet nur 4/6. Ledger 8edde088.
- **volcano_precursor apply51 (−12h-Präkursor):** 1. Readout = **FETCH-FAIL pre**
  (beide Stationen, registriert a3c98d35/2b4f2432): IRIS ascii1 trägt pro Segment einen
  TIMESERIES-Header, Parser (frozen) brach. **Amended fetch** (nur Header-Filter in
  `_fetch`, Analyse unberührt, im Code + hier deklariert; prereg1/2 NICHT re-run — die
  hatten evaluierbare Readouts): **ORDERING=PASS auf BEIDEN Stationen** (UWE pre-prob
  PRE 0.562 > MID 0.469; RIMD 0.712 > 0.682), **strict=FAIL**. Ledger 4524a4b6.

**Ehrliche Bilanz:** die beiden Zustands-Klassifikatoren scheitern prospektiv (erwartbar
— Prereg-3-Linie war „retrospektiv-nicht-gestützt"); der −12h-Präkursor besteht auf dem
ungesehenen Ereignis das ORDERING-Kriterium auf beiden Stationen, nicht das strikte.
n=1-Demo-Quittung, kein Beweis — aber der erste prospektive Datenpunkt der Kette, und
er ist append-only im Ledger.
