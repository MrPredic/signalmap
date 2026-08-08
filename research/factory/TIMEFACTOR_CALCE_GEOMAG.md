# ZEIT-FAKTOR — CALCE-Batterie + GEOMAG-Onset: CSD-NULL × 2 (Readout 11./12. Jul 2026)

**WS4-Abschluss des Zeit-Faktor-Programms.** Derselbe Theorie-Anker **Critical Slowing Down**
(Scheffer/Dakos Early-Warning: steigende lag-1-Autokorrelation UND Varianz vor einem Kipp-Punkt),
dieselbe unveränderte Statistik (`ims_csd._csd_bearing`: Gauss-detrend σ=0.05·N, Rolling ar1/logvar
Fenster 0.5·N, Kendall-τ, Fourier-Phasen-Surrogat N=500, α=0.05, seed 20260708), auf die beiden
User-vorentschiedenen Fallback-Familien nach den Vulkan- und IMS-Nulls. Alle Designs
**prereg-VOR-Readout gefroren** (Ledger + frozen/*.json), gerichtet, lern-frei, selektions-frei.

## 1. CALCE-Batterie (NMC Kalender-Aging, OSF j2sn4 — dieselbe Bank wie BATTERY-TRANSFER/-SOH)

**Design (`calce_csd.py`, Preregs CALCE-CSD-CDCG-PREREG `321da4ed` + CALCE-CSD-VDCG-PREREG
`181dc8e9`, beide 11. Jul 10:44 VOR erstem Load):** Einheit = Checkpoint (Zelle, Tag);
Slow-Variable = chronologische In-Test-Spur des Entlade-Segments eines Checkpoints
(`cdcg` = Kapazitäts-Trace, `vdcg` = Spannungs-Trace, orthogonaler Marker; ~3300–4000 Samples,
verifiziert kontiguierlich). Life-Tercile-Split reused unverändert `battery_pipeline.FRESH_DAYS`
{0,10,20} (n=72) vs `AGED_DAYS` {40,70,90} (n=29; Survivorship-Dropout auf 3 Zellen ab Tag 70
dokumentiert). PRIMARY gerichtet: CSD+-Rate aged > fresh, Fisher exact greater. Spezifität:
fresh darf nicht auf aged-Niveau feuern.

| Indikator | CSD+ aged | CSD+ fresh | ar1_rise aged | var_rise aged | Fisher p (a>f) | Verdikt |
|---|---|---|---|---|---|---|
| cdcg (Kapazitäts-Trace) | **0/29** | 0/72 | 0/29 | 0/29 | 1.0 | **CSD-NULL** |
| vdcg (Spannungs-Trace) | **0/29** | 0/72 | 0/29 | 0/29 | 1.0 | **CSD-NULL** |

- **Totale Stille auf beiden Seiten** (0/101 Checkpoints gesamt, 0 Falsch-Positive auf fresh) —
  anders als IMS, wo der rig-globale RMS-Indikator healthy-FPs produzierte. Der Anker ist stringent,
  aber die In-Test-Fluktuationsdynamik eines kontrollierten Entlade-Tests trägt schlicht keine
  CSD-Signatur des Zell-Alters: der Test ist ein geregelter Prozess (vgl. TEP-Lektion:
  Regelung maskiert Dynamik-Marker).
- **Ehrliche Grenze:** Das echte Run-to-Failure-Analog wäre die ÜBER-Checkpoint-Trajektorie pro
  Zelle — mit nur 4–6 Checkpoints/Zelle für Rolling-Fenster strukturell zu kurz (deshalb a priori
  die In-Test-Spur gewählt). CSD „innerhalb des Tests" ≠ CSD „über das Leben"; letzteres bleibt
  mit diesem Datensatz untestbar, nicht widerlegt.

## 2. GEOMAG-Onset (Sturm-Beginn-Timing, USGS 1Hz X, BOU + FRD)

**Design (`geomag_onset_csd.py`, Prereg GEOMAG-ONSET-CSD-PREREG `929e1460`, 11. Jul 18:40):**
Onset = Kp≥5 nach ≥24h sauberer Kp<5-Baseline (G1+), quiet-Referenz = 30h in einen ≥48h-Kp<3-Run,
Zeitraum 2015–2026 fix, N≈20 je Klasse je Station. Fenster = **[T−24h, T−12h)** vor Referenz
(12h, roh 1Hz X). PRIMARY gerichtet: CSD+-Rate onset > quiet, Fisher exact greater.
**Prereg-Integrität:** Ein früherer Freeze (`7e0c8ac5`, 16:55) wurde VOR jedem Readout durch
`929e1460` (18:40, neue Event-Liste, anderes events_sha256) ersetzt; erster Readout 19:15.
Beide Freezes stehen append-only im Ledger; der Readout ist per `events_sha256`-Assert an den
gültigen Freeze gebunden — kein Post-hoc-Drehen möglich.

| Station | CSD+ onset | CSD+ quiet | ar1_rise onset | var_rise onset | Fisher p (o>q) | Verdikt |
|---|---|---|---|---|---|---|
| BOU | 1/19 | 2/19 | 6/19 | 4/19 | 0.8851 | **CSD-NULL** |
| FRD | 1/19 | 0/20 | 3/19 | 3/19 | 0.4872 | **CSD-NULL** |

- BOU sogar leicht invertiert (quiet feuert öfter als onset) → kein Trend, der mit mehr n
  signifikant würde. Einzel-Anstiege (ar1_rise 6/19 BOU) bleiben unter Chance-Niveau-Erwartung
  eines α=0.05-Doppelkriteriums nicht interpretierbar.
- Ehrliche Grenze: genau EIN Vorlauf-Fenster (12–24h) getestet, a priori gepinnt; andere Horizonte
  wären NEUE Preregs (nicht diese tunen). Claim-Scope war eng registriert:
  „ground X-component fluctuation dynamics, 12-24h pre-onset, BOU+FRD only".

## 3. Receipts + Determinismus

- Results (Ledger, append-only): CALCE-CSD-CDCG-RESULT `0c25336e` (11. Jul) + Re-Run `dbed33cc`
  (12. Jul), CALCE-CSD-VDCG-RESULT `8589a5a6` + `3bcc467c`, GEOMAG-ONSET-CSD-RESULT `6285bc3d`
  + `c5b3db83`. **Alle 3 Re-Run-Payloads byte-identisch zum Erstlauf = Determinismus empirisch
  belegt.** Ledger-Kette `verify()=True`, tip `c5b3db83…`.
- Result-JSONs: `logs/calce_csd_cdcg_result.json`, `logs/calce_csd_vdcg_result.json`,
  `logs/geomag_onset_csd_result.json`; Konsole `logs/timefactor_readouts.log`.

## 4. Cross-Family-CSD-Bilanz (Programm-Verdikt)

| Familie | Indikator(en) | Verdikt |
|---|---|---|
| Vulkan-Präkursor (Kilauea UWE/RIMD) | Hüllkurven-AR1/Var (`csd.py`) | NULL/NULL |
| IMS-RUL (Lager Run-to-Failure) | RMS + Kurtosis (`ims_csd.py`) | NULL (RMS-Spezifität invertiert = rig-Confound aufgedeckt) |
| CALCE-Batterie (Kalender-Aging) | cdcg + vdcg | NULL + NULL (0 FP) |
| GEOMAG-Onset (Sturm-Beginn) | 1Hz X, 12–24h Vorlauf | NULL (BOU+FRD) |

**CSD als universeller Frühwarner: 0/4 Familien** unter prereg-disziplinierten, leck-freien,
spezifitäts-kontrollierten Tests mit identischer, nie getunter Statistik. Das ist der Wert:
ein gerichteter Theorie-Anker, der praktisch nie falsch-positiv feuert (einzige FPs waren der
diagnostizierte IMS-Rig-Confound) — und in unseren Familien/Fenstern schlicht keine Frühwarnung
liefert. Ehrlicher, disziplinierter Negativ-Beleg; die Trust-Harness-These trägt: dieselbe
Maschine, die Discoveries zertifiziert (#11/#12), zertifiziert auch Nulls.

**Pre-registrierbare Nachfolger (NICHT diese Preregs tunen):** IMS defekt-Band-Envelope +
Vor-Ausfall-Fenster-Restriktion; GEOMAG andere Vorlauf-Horizonte; CALCE über-Checkpoint-CSD
erst mit dichterem Aging-Datensatz.
