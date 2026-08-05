# PREREG — distill Premium-Familie ENVELOPE (Hilbert-Hüllkurve → Defektband-
Energie) auf CWRU + MFPT + IMS (19. Jul 2026)

**FROZEN BEFORE READOUT.** Ledger-Receipts `DISTILL-PREMIUM-ENV-<BANK>` werden
NACH diesem Commit angehängt, danach wird an diesem Dokument, an
`envelope_features` und an den distill-Parametern nichts mehr geändert.
EXCLUDED ist ein gültiges, berichtenswertes Ergebnis — insbesondere für diese
Familie, siehe Kontext.

## Kontext / Warum envelope
3. Premium-Familie neben RQA (teuer, O(n²)) und coherence (Multi-Channel).
Envelope ist PdM-Standard (Hüllkurven-Ordnungsanalyse, Lagerdefekt-Physik) und
BILLIG (O(n log n) Hilbert+FFT — kein Grund, sie NUR dort einzusetzen, wo es
garantiert zahlt). Sie dreht die bisherige Kosten-Story um: die Champion-Regel
entscheidet unabhängig vom Preis, auch eine billige Familie kann verweigert
werden (das Gate ist der Punkt, nicht die Familie). Erwartung ist deshalb
bewusst OFFEN — kein vorab-erwarteter Gewinn wie bei CWRU-RQA.

## Hypothese (PRIMARY, ein Verdikt pro Bank)
> Auf Bank X schlägt distill-augmented (volle base-Selektion + FIXE 5-Band
> envelope-Familie, Produkt-Default) die base-Selektion paired-CI-fest
> (95%-Bootstrap-CI der per-Fold-Differenz über LOGO-Recordings, lo > 0) →
> Champion-Regel-Verdikt **INCLUDED**. Drei unabhängige Verdikte, kein
> gemeinsamer Threshold zwischen den Bänken.

## Ehrlichkeits-Klauseln (alle VOR dem Lauf registriert)
1. **RELATIVE Bänder, nicht physisch.** `envelope_features` teilt das untere
   Hälfte-Spektrum der Hüllkurve (Bins 1..⌊0.5·N⌋, DC-Bin 0 verworfen) in
   n_bands=5 gleich breite Bänder und gibt relative Energieanteile zurück —
   NICHT physische BPFO/BPFI/BSF/FTF-Frequenzen. Physische Bänder bräuchten
   pro Bank Drehzahl+Lagergeometrie (CWRU/MFPT/IMS je unterschiedlich
   dokumentiert/fehlend) — das wäre eine versteckte Bank-spezifische Suche und
   bricht die Ein-Config-Ehrlichkeit. Die relative Bandaufteilung ist die
   PRODUKT-DEFAULT-Familie, uniform über alle Bänke, kein Grid.
2. Fixe Parameter: n_bands=5, f_max_frac=0.5 (`envelope_features` Defaults,
   `signalmap/premium.py`). Kein Config-Wechsel nach Sicht irgendeines
   Readouts, keine Suche über n_bands/f_max_frac.
3. EXCLUDED ist für diese Familie ERWARTBAR und ein valides Produkt-Ergebnis
   auf jeder der drei Bänke einzeln — die billige Familie muss sich am
   gleichen Gate beweisen wie RQA/coherence, kein Rabatt für niedrige Kosten.
   Kein Nach-Tuning, kein Re-Run mit anderen Parametern nach einem EXCLUDED.
4. Drei Bänke = drei unabhängige PRIMARY-Verdikte (kein Multiple-Comparisons-
   Downgrade nötig, jede Bank ist ihre eigene Hypothese, wie CWRU-RQA vs
   CALCE/HYD-RQA vorher separat verdiktet wurden).

## Fixe Parameter (distill, identisch zu CWRU-RQA/coherence-Prereg)
C=50, kmax=5, thr=0.005, n_perm=200, trees=100, cand=60, seed=0,
null_check=True, premium=("envelope",).

## Bänke (Loader-Calls fix, `research/factory/distill_premium_case.py::BANKS`)
1. **CWRU** — `feature_forge.load_cwru()`: 24 Recordings, 2849 Fenster (W=1024,
   detrend+z-norm), 6 Klassen (B007/B021/IR007/IR021/OR007/OR021), chance
   0.167. Dieselbe Bank wie die CWRU-RQA-Prereg (INCLUDED-Präzedenz).
2. **MFPT** — `mfpt_run.load_mfpt()`: 20 Recordings, 3718 Fenster (W=1024,
   sr auf 48828 Hz harmonisiert), 3 Klassen (baseline/inner/outer), chance
   0.333.
3. **IMS** — `retro_loaders.load_ims()` (task="fault" Default): 12 Recordings
   (Test×Bearing), 192 Fenster (W=1024, 20 kHz Snapshots letztes Lebens-
   Tertial), 2 Klassen (defect/healthy, zeitgleich koexistent → Rig-State-
   Confound konstruktionsbedingt neutralisiert), chance 0.5.

## Sekundäre Quittungen (müssen im Receipt stehen, keine Pass-Pflicht für PRIMARY)
- distill-Gates: nested LOGO > chance+0.05, group-perm p ≤ 0.05, NULL ≈ chance.
- Kosten-Quittung: envelope ms/window vs base ms/window (Erwartung: nahe an
  base, O(n log n) beidseitig — DAS ist der Punkt der billigen Familie).

## Triangulation (Familien-Korrektheit, proportional zur Neuheit)
1. Geschlossene AM-Physik-Parität (Test-Suite,
   `test_envelope_am_parity_concentrates_in_defect_band` +
   `test_envelope_pure_carrier_is_null_no_defect_concentration`): Hüllkurve
   eines amplitudenmodulierten Signals mit bekannter Defektfrequenz
   konzentriert Energie exakt im vorhergesagten Band (>0.5, argmax korrekt);
   unmodulierter Träger (+Rauschboden) bleibt nahe uniform (max < 0.35) —
   Positiv- UND Null-Seite gepinnt, kein Referenz-Package nötig (Hilbert-
   Transformation ist scipy-Standard, keine externe Lager-RQA-Bibliothek wie
   bei pyunicorn verfügbar/nötig).
2. distill-Premium-Receipt selbst (augmented vs base, paired CI über LOGO).
3. Null-Kontrollen: eingebauter Label-Shuffle-NULL + group-perm-p (wie
   RQA/coherence).
4. Drei strukturell verschiedene Bänke (Lager-Vibration CWRU/MFPT gegen
   Rig-Betriebs-Snapshot IMS, unterschiedliche Klassenzahl/chance) — Konvergenz
   oder Divergenz über Bänke ist selbst ein Befund, kein einzelner Datenpunkt.

## Determinismus
Ein Befehl pro Bank reproduziert alles (Seeds fix: RF random_state=0,
Bootstrap seed=0, NULL seed=1):
```
cd research/factory && ../../.venv-research/bin/python3 distill_premium_case.py --family=envelope cwru
cd research/factory && ../../.venv-research/bin/python3 distill_premium_case.py --family=envelope mfpt
cd research/factory && ../../.venv-research/bin/python3 distill_premium_case.py --family=envelope ims
```
Output je Bank: `logs/distill_premium_<bank>_envelope_report.md` +
`logs/distill_premium_<bank>_envelope_spec.json` + Ledger-Eintrag
`DISTILL-PREMIUM-<BANK>-ENVELOPE`.
