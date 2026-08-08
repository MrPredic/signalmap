# PREREG — External-User-Simulation DCASE2020-Task2 Valve id_00 (Acoustic
Anomaly Detection, öffentlicher Benchmark) (20. Jul 2026)

**FROZEN BEFORE READOUT.** Ledger-Receipt `DCASE-VALVE-EXTERNAL` wird NACH
diesem Commit angehängt; danach wird an diesem Dokument, an
`dcase_valve_adapter.py`, an der Bank-Aufteilung (welche Clips wohin) und an
den distill/fit-Parametern nichts mehr geändert. EXCLUDED (Premium) und ein
schwacher AUC sind beide gültige, berichtenswerte Ergebnisse — valve ist laut
öffentlicher DCASE2020-Baseline die HÄRTESTE der sechs Maschinenklassen
(Baseline-AUC ≈ 0.5–0.6, Challenge-SOTA ≈ 0.7–0.9 je nach ID/Methode). Dies ist
eine externe Nutzer-Simulation von SignalMap auf einem Datensatz, den das
Projekt nie gesehen hat — kein Tuning auf dieses Ergebnis hin.

## Kontext / Warum
Phase 1 hat den öffentlichen DCASE2020 Task2 valve/id_00-Datensatz bereits
geladen und verifiziert (`data/mimii/valve_id00/{train,test}/`, 891 train
NORMAL, 100 test normal + 119 test anomaly, mono/16kHz/160000 Frames/16-bit
PCM). Phase 2 spielt den Standpunkt eines externen Nutzers, der SignalMap zum
ersten Mal auf seinen eigenen (hier: fremden, öffentlichen) Rohdaten
einsetzt — exakt der dokumentierte Zwei-Befehl-Workflow aus `README.md`
(`distill` → `fit --spec … --bank …` → `monitor --detector … --bank …`).
`load_bank()` akzeptiert `.npy/.csv/.txt/.mat`, nicht `.wav` — ein
Adapter-Skript konvertiert deterministisch.

## Hypothese (PRIMARY, ein Verdikt)
> `signalmap fit` auf gesunden (NORMAL) valve-Aufnahmen, dann `signalmap
> monitor` auf einer disjunkten Held-out-Menge aus normalen+anomalen
> Aufnahmen trennt Anomalie von Normal: **AUC > 0.5 CI-fest**
> (95%-Bootstrap-CI über Test-Clips, lo > 0.5 verlangt für ein klares
> "trennt").

## Hypothese (SECONDARY, ein Verdikt, unabhängig von PRIMARY)
> Premium-Familie **envelope** (Hilbert-Hüllkurve, billig, needs_channels=1 —
> `rqa` wäre O(n²) und auf dieser RAM/Zeit-Budget-Session vermieden) wird von
> der Champion-Regel (paired-Bootstrap-CI über LOGO-Recordings, lo > 0)
> **INCLUDED** in `spec.json`. EXCLUDED ist ein gültiges Ergebnis — envelope
> bekommt keinen Rabatt fürs Billigsein (Präzedenz: PREREG_DISTILL_PREMIUM_
> ENVELOPE.md).

## Ehrlichkeits-Klauseln (alle VOR dem Lauf registriert)
1. **Mono-Einschränkung deklariert.** DCASE-wavs sind mono → `coherence`
   (braucht 2 Kanäle) ist auf dieser Bank strukturell nicht testbar und wird
   nicht versucht. Kein Ersatz-Multichannel-Trick.
2. **Kein Leaderboard-SOTA-Anspruch.** Dies testet SignalManeuvers Rezept
   (Receipt-Verhalten: Gate, Kosten, Ehrlichkeit) auf echten externen Daten,
   nicht ob es DCASE-Challenge-Modelle schlägt. Valve ist die schwerste Klasse
   im öffentlichen Benchmark; ein AUC nahe 0.5–0.6 ist ein plausibles,
   ehrliches Ergebnis und KEIN Bug.
3. **Distill braucht zwingend 2 Klassen — train allein (891 Clips, alle
   NORMAL) reicht dafür nicht.** Deshalb wird die distill/Premium-Bank aus
   EINER FESTEN Teilmenge von train-normal + einer FESTEN Teilmenge von
   test-anomaly gebaut (siehe Bank unten). Diese exakten Dateien sind ab
   diesem Commit eingefroren.
4. **Kapazitäts-Deviation deklariert (WICHTIG, VOR Sicht jeder Zahl):**
   distills LOGO-Gauntlet hält je Recording einen vollen Leave-one-out-Fold
   (+ `n_perm` × Fold für die Gruppen-Permutation). Bisherige Bänke
   (CWRU/MFPT/IMS) hatten 12–24 Recordings. Würde man — wie die grobe
   Formulierung "K Fenster pro Clip, K klein" suggerieren könnte — alle ~900
   Clips einzeln als Recording einspeisen, explodiert die LOGO-Kosten
   (Größenordnung 900 Folds × 100 Permutationen) weit über jedes
   RAM/Zeit-Budget dieser Session UND liefert wegen train-only-NORMAL keine
   2. Klasse. Die distill/Premium-Bank hält deshalb die Recording-ZAHL auf
   Präzedenz-Skala (**30 Recordings**, wie CWRU/MFPT/IMS) und hebt stattdessen
   K PRO CLIP an (K=120 statt K=3–4), um das Fenster-Zielband selbst zu
   treffen. `fit`/`monitor` (PRIMARY) haben keine LOGO-Kosten (kein
   Klassifikator, nur Healthy-Envelope-Kalibrierung + Distanz-Score) und
   laufen deshalb über ALLE 891+204 Clips mit einem kleineren K=20 —
   RAM-sicher (~180 MB) und mit realistischer Clip-Zahl.
5. **Keine Überlappung Distill-Anomalie ↔ Eval-Anomalie.** Die 15 Anomalie-
   Clips, die distill für die "anomaly"-Klasse sieht, werden aus der
   `monitor`-Eval-Bank ENTFERNT (104 der 119 bleiben für AUC). Die
   "normal"-Klasse für distill kommt ausschließlich aus train (15 von 891);
   train-Clips fließen zusätzlich VOLLSTÄNDIG (alle 891, inkl. dieser 15) in
   `fit` — das ist keine Leckage, weil `fit` keine Klassen-/Anomalie-Info
   verwendet (nur Healthy-Envelope), exakt der README-Workflow (distill auf
   einer kleinen gelabelten Referenz, fit auf reichlich gesunden Daten,
   Überlappung im Healthy-Teil ist harmlos). test-normal (alle 100) wird von
   distill nie berührt.
6. Aggregation je Clip = **Mittelwert** der `DistilledDetector.score()`-Werte
   (max-|z| über Spec-Features) über die K=20 Fenster des Clips — fix
   deklariert, kein Max/Median-Vergleich nach Sicht der Zahlen.
7. `n_perm` von distill-Default 200 auf **100** gesenkt (Task-Erlaubnis, Zeit-
   Budget), deklariert, keine weitere Absenkung nach Sicht der Laufzeit.
8. Bootstrap-CI (AUC) = 2000 Resamples ÜBER CLIPS (nicht Fenster), seed=0,
   perzentil-basiert (2.5/97.5%).

## Bank
**Distill/Premium-Bank** (`data/mimii/valve_id00_bank/distill/`, 30
Recordings, label_by=prefix → Klassen {anomaly, normal}, chance=0.5):
- normal (15): erste 15 `normal_id_00_*.wav` aus train, sortiert aufsteigend
  nach Dateiname.
- anomaly (15): erste 15 `anomaly_id_00_*.wav` aus test, sortiert aufsteigend
  nach Dateiname.
- K=120 nicht-überlappende 1024-Fenster/Clip (aus den ersten 120×1024=122880
  von 160000 Samples) → 30×120 = **3600 Fenster**.

**Fit-Bank** (`data/mimii/valve_id00_bank/train/`, PRIMARY): ALLE 891
train-normal-Clips, K=20 Fenster/Clip → 17820 Fenster.

**Monitor/Eval-Bank** (`data/mimii/valve_id00_bank/test/`, PRIMARY, disjunkt
von der Distill-Anomaly-Menge): 100 test-normal (alle) + 104 test-anomaly
(119 − 15 für distill = 104, die 15 mit dem niedrigsten Dateinamen
ausgeschlossen) = 204 Recordings, K=20 Fenster/Clip → 4080 Fenster.

## Fixe Parameter (keine Suche)
- Fensterung: W=1024, detrend+z-norm (Produkt-Default, `distill.py::window`).
- distill: C=50, kmax=5, thr=0.005, **n_perm=100** (siehe Klausel 7),
  trees=100, cand=60, seed=0, null_check=True, premium=("envelope",).
- DistilledDetector.fit: envelope=3.0 (Produkt-Default, `signalmap/distill.py`
  `DistilledDetector.fit`), threshold selbstkalibriert (99. Healthy-Perzentil
  × envelope).
- AUC: `sklearn.metrics.roc_auc_score`, Label anomaly=1/normal=0 aus
  Dateiname-Präfix. TPR@FPR=0.1 aus `sklearn.metrics.roc_curve` (linear
  interpoliert am nächsten FPR-Gitterpunkt ≤/≥ 0.1).

## Sekundäre Quittungen (kein Pass-Zwang für PRIMARY)
- distill-Gates auf der 30-Recording-Bank: nested LOGO > chance+0.05,
  group-perm p ≤ 0.05, NULL ≈ chance (Standard-Ausgabe von `distill()`,
  informativ — das Premium-Verdikt hängt NICHT davon ab, siehe README:
  "PASS/FAIL gate judges the base selection, the champion rule judges each
  premium family").
- Kosten-Quittung: envelope ms/window vs base ms/window.
- TPR@FPR=0.1 als Zusatzzahl neben AUC.

## Determinismus
```
cd <local-path>/signalmap && source .venv-research/bin/activate
python research/factory/dcase_valve_adapter.py
nice -n 19 python research/factory/dcase_valve_readout.py
```
Seeds fix (distill seed=0, group-perm seed=0, Bootstrap seed=0). Output:
`research/factory/logs/dcase_valve_distill_envelope_report.md` +
`_spec.json`, `research/factory/logs/dcase_valve_readout_report.md`
(AUC/CI/TPR), Ledger-Einträge `DCASE-VALVE-DISTILL-ENVELOPE` +
`DCASE-VALVE-EXTERNAL`.

## AMENDMENT 2026-07-21 (decoupling + bound, declared BEFORE this rerun)

**Root-cause finding, not a design change.** A prior execution of
`dcase_valve_readout.py` was killed/reported as a runaway. Forensics on the
actual crash log (`research/factory/logs/dcase_secondary.log`) show it did
**not** run away over ~900 LOGO folds — the committed bank composition above
(§Bank) already bounds the distill/premium LOGO to **30 recordings** (15
normal + 15 anomaly, chance=0.5), same precedent scale as CWRU/MFPT/IMS. The
actual failure was an immediate crash inside `_logo_mean`'s
`RandomForestClassifier(n_jobs=-1)` (`signalmap/distill.py:409`):
`ValueError: cannot find context for 'threading'`, caused by an environment
variable `JOBLIB_START_METHOD=threading` present in whatever shell launched
that run (`threading` is a joblib *backend* name, not a valid
`multiprocessing.get_context()` *method* — only `fork`/`spawn`/`forkserver`
are valid contexts for that call). This env var is **not** part of this
project's declared config and is absent from the current shell
(`env | grep -i joblib` → empty, verified 2026-07-21).

**Decisions, declared before rerun (no bank/parameter changes — bank
composition and distill/fit parameters remain exactly as frozen above):**
1. Rerun in a shell with `JOBLIB_START_METHOD` unset/verified absent. No
   other environment change.
2. The distill/premium bank stays the originally-frozen **30 recordings**
   (§Bank) — this was already the bounded, declared subsample and satisfies
   the intent of "decouple + bound": the premium champion-rule verdict is
   computed on this fixed 30-recording bank and is reported as a **bounded
   estimate on that subsample**, not a claim about all ~900 recordings.
3. PRIMARY AUC is unchanged: evaluated on **all 204 held-out test clips**
   (100 normal + 104 anomaly, §Bank "Monitor/Eval-Bank"), via `fit` on all
   891 train recordings — neither step performs LOGO, so neither is
   affected by the LOGO-fold cost that caused the original kill report.
4. Every heavy launch for this rerun is wrapped in
   `nice -n 19 timeout 1800 python ...` (secondary/distill) and
   `nice -n 19 timeout 600 python ...` (primary/fit+monitor). If either hits
   its wall-clock cap, this is reported as a timeout, not silently retried.

## AMENDMENT 2 — 2026-07-21 (secondary timed out → gate-only spec for PRIMARY, declared BEFORE building spec)

**Observed:** the secondary/distill launch (30-recording bank, but K=120 →
**3600 windows**, n_perm=100, RandomForest LOGO) hit its `timeout 1800`
(30-min) wall-cap without producing `spec.json` (crash log ends mid-run with a
multiprocessing semaphore-leak-at-shutdown = SIGTERM/SIGKILL by `timeout`, no
verdict line). At 3600 windows the LOGO champion-rule + permutation test is
infeasible within this session's safe wall-clock bound. This is itself a valid,
reported external-user finding.

**Decision, declared before touching any PRIMARY number:**
1. **SECONDARY premium verdict = NOT COMPUTED** (distill infeasible within the
   30-min bound at 3600 windows). Reported honestly as "not obtained", no
   INCLUDED/EXCLUDED claim, no post-hoc config shrink to force it.
2. **PRIMARY AUC is still evaluated** (that is the main hypothesis). Because
   `run_primary` needs a `FeatureSpec` and the distill one does not exist, the
   spec is built DETERMINISTICALLY from the base grammar:
   `programs = [p.name for p in gate(enumerate_programs(), n_recordings=30,
   C=50)]`, `premium=[]` — i.e. the LEAN BASE feature set (the deterministic
   budget-cut that distill itself starts from), with NO statistical
   champion-rule selection and NO premium family. This is a weaker spec than a
   full distill would produce; the AUC is therefore a conservative lower-ish
   estimate of what SignalMap's full pipeline could reach, and is labelled as
   the "lean-base spec" AUC. Declared here BEFORE the spec is built or any AUC
   is seen. Everything else (fit on all 891 train, AUC on all 204 held-out test
   clips, seeds, aggregation, bootstrap) is unchanged from §Bank / §Fixe
   Parameter.

## AMENDMENT 2b — 2026-07-21 (spec-size correction, declared before any AUC seen)

Building the spec per Amendment 2 revealed that `gate(enumerate_programs(),
n_recordings=30, C=50)` returns **all 1500** enumerated programs (the capacity
budget is not binding at this scale). A 1500-program featurization over
~130k windows is itself infeasible (the very runaway class we are guarding
against). Corrected, deterministically and BEFORE any AUC is computed: the
lean-base spec = the **5 cheapest** gate-survivors, i.e.
`[p.name for p in gate(enumerate_programs(), 30, 50)][:5]` — `kmax=5` is
distill's product default (§Fixe Parameter), and `gate` orders cheapest-first,
so these are the 5 lowest-cost base programs (`acf1(id(id(x)))`,
`crest(id(id(x)))`, `lcross(id(id(x)))`, `meanabs(id(id(x)))`,
`peakcv(id(id(x)))`). No AUC has been seen at the time of this correction. This
is the fixed spec for the PRIMARY readout.
