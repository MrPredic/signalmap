# NEXT SESSION PLAN (aktualisiert 17. Jul 2026 — Multi-Channel + Ep-51-Session)

## ✅ ERLEDIGT 16./17. Jul (Commit 7bceaeb + Folge-Commit; Suite 76→102; Details RESULTS.md)
1. **Multi-Channel-Distill KOMPLETT nach SPEC (TDD):** 2D-Fenster/-Ingest (CSV-Header,
   2D-.npy, `--channel-axis`), ch0-Primary-Konvention, `needs_channels`-Verweigerung,
   1-Kanal-Pfad byte-identisch gepinnt; **Kohärenz-Premium-Familie c128b2 fix**
   (Parity-Pin gegen coherence_fair exakt).
2. **Prereg (acad1724 VOR Readout) + Lauf: HYD-cooler = 2. INCLUDED** (base 0.750 →
   aug 0.889, paired +0.139 CI [+0.042,+0.236], ~81×); **GAS-id = EXCLUDED** (+0.006
   CI [−0.100,+0.113]) wie deklariert möglich. Premium-Story gilt jetzt für BEIDE
   Familien: Quittung verweigert UND lässt zu.
3. **CLI-Gap ZU:** `fit --spec spec.json --bank` + `monitor --detector det.json --bank`
   (DistilledDetector = Produkt-Oberfläche, kalibrierter Threshold); CLI-E2E live PASS.
4. **🌋 EP-51 GEFEUERT (Onset 15. Jul 18:30Z, 8.3 h) — 3 applies prospektiv:** prereg1
   FAIL, prereg2 FAIL (beide ehrlich, erwartbar), **apply51 ORDERING=PASS beide
   Stationen / strict=FAIL** (nach deklariertem amended-fetch; 1. Readout FETCH-FAIL
   registriert). Ledger c60d8a95/8edde088/4524a4b6. n=1-Demo-Quittung, ehrlich gelabelt.

## PRIO 1 (nächste Session)
1. **Ep-52-Watch weiterlaufen lassen** (HVO: Re-Inflation, nächste Episode wahrscheinlich,
   Forecast offen) — `session_status.py` 1. Befehl; bei Feuer: dieselben 3 applies
   (Preregs gelten unverändert weiter, KEIN Re-Freeze nötig). ORDERING-PASS auf Ep 52
   wiederholen = aus n=1 wird n=2, das wäre ein echtes Upgrade.
2. **Premium-Report-Klarheit:** FAIL+INCLUDED-Fall (base-Gate vs Premium-Sieg, Befund
   RESULTS.md) im Receipt-Text erklären; kleiner Render-Fix + Test.
3. **Adoption/Marketing mit Premium-Story** (2 Familien, verweigert UND lässt zu,
   CI-gated, signierte Receipts) — outward-facing, auf User-Trigger.
Sanity-Einstieg: `session_status.py`, dann `pytest -q` = **102**.

---
# [Historie] PLAN vom 14. Jul abends (nach INCLUDED-Session)

## ✅ ERLEDIGT 14. Jul 2. Session (Commits 768a2d0→0ef95a5; Ledger tip 0bf18785, chain OK; Suite 76)
1. **CWRU-Premium-Verdikt = ERSTER INCLUDED (prereg'd VOR Readout, Freeze cb7653a3):**
   base 0.914 → aug **0.980**, paired **+0.066 CI[+0.033,+0.106]**, `spec.premium=["rqa"]`,
   Kosten-Quittung ~585× (66.03 vs 0.113 ms/win); Gates PASS (nested 0.874, perm-p 0.005,
   NULL 0.082). Ehrlichkeits-Klausel hielt (Produkt-Default m3τ5, kein Grid). Lauf 2317 s.
   **★ Premium-Story komplett: Quittung verweigert (CALCE/HYD) UND lässt zu (CWRU).**
2. **E2E-Kette** `signalmap/tests/test_e2e_chain.py`: Premium-INCLUDED durch
   spec.json → DistilledDetector → det.json → alert; reale RQA im Detector-Pfad.
3. **Ingest-Härtung** (Befund 13. Jul GESCHLOSSEN): `_read_text_signal` — BOM/CRLF/
   Header tolerant, ragged/non-finite gezählt, fail-closed >5% fehlende Spalte;
   echtes ECN-File lädt direkt (14654 Samples). .npy-Workaround obsolet.
4. **Multi-Channel als SPEC:** `SPEC_MULTICHANNEL_DISTILL.md` (Design fix, DoD fix).

## PRIO 1 (nächste Session)
1. **Multi-Channel-Impl nach SPEC_MULTICHANNEL_DISTILL.md** (eigene Session, TDD-
   Reihenfolge im SPEC) → Kohärenz-Premium-Familie; danach Prereg HYD-cooler/GAS-id
   (erwartetes 2. INCLUDED auf HYD-cooler, Referenz aug 0.944).
2. **CLI-Gap schließen:** `signalmap fit`/`monitor` kann kein spec.json-Backend —
   DistilledDetector ist library-only (Befund aus E2E-Bau). Kleines Wiring + Test.
3. **Adoption/Marketing mit Premium-Story** (verweigert UND lässt zu, CI-gated,
   signierte Receipts) — outward-facing, auf User-Trigger.
Sanity-Einstieg: `session_status.py` 1. Befehl (**Ep-51-Onset-Forecast 14.–16. Jul!**),
dann `pytest -q` = 76.

---
# [Historie] PLAN vom 14. Jul früh (nach Premium-Familien-Session)

## ✅ ERLEDIGT 13./14. Jul (Commit 2391836 + Verdikt-Commit; Ledger tip 43624253, chain OK)
1. **PRIO 1 ENTSCHIEDEN + GEBAUT:** RQA = **distill-Premium-Familie mit Kosten-Quittung**
   (nicht Forge-Slot). `signalmap/premium.py` (numpy-RQA, pyunicorn-Parity gepinnt) +
   `distill(premium=…)`/CLI `--premium` mit **Champion-Regel** (Premium in Deploy-Spec nur
   bei paired-CI-festem Sieg). Suite **66/66**. Kohärenz = 2. Premium-Familie VERTAGT
   (braucht Multi-Channel-Bank-Ingest in distill — eigener Bau).
2. **Praxis-Case preregistriert (VOR Readout, Ledger 5f0a92b9) + ehrlich WIDERLEGT:**
   Champion-Regel verdiktet **EXCLUDED auf BEIDEN Gewinner-Banken** — CALCE base 0.935 →
   aug 0.905 (−0.030 CI[−0.072,+0.006], ~182× Kosten), HYD-cooler 0.800 → 0.833
   (+0.033 CI[−0.022,+0.100], ~755×). **Frontier-Präzisierung: RQA-Wins galten vs LEAN;
   gegen volle distill-base-Selektion verschwindet der Vorsprung.** Die Verweigerung ist
   das Produkt-Verhalten. Details RESULTS.md §DISTILL PREMIUM-FAMILIEN.

## PRIO 1 (neu) — offene Hebel
1. **CWRU-Premium-Prereg:** einzige Bank wo RQA fair ÜBER forge liegt (0.961 vs 0.902)
   → NEUE Prereg, dann `distill --premium rqa` auf CWRU; erwarteter erster INCLUDED-Case.
2. **Kohärenz-Premium-Familie:** braucht Multi-Channel-Fenster in distill-Bank
   (Ingest-Design!); Beleg liegt vor (HYD-cooler aug 0.944 / GAS-id 0.516, 13. Jul).
3. **Distill-Ingest-Härtung** (BOM/Header/ragged .txt, Produkt-Befund 13. Jul).

## ✅ ERLEDIGT 13. Jul (Commits e246dfb, 605ea22; Ledger 72fa75d6, chain OK)
1. **PRIO 1 Kohärenz-Flags BEIDE CONFIRMED CI-fest** (`coherence_fair.py`, Prereg 4bb76422
   VOR Readout, exakt Screen-Loader): HYD-cooler aug 0.944 (+0.319 CI [+0.194,+0.444],
   alone 0.972 — Zustand steht fast komplett im Cross-Sensor-Muster), GAS-id aug 0.516
   (+0.216 CI [+0.106,+0.335] — das alte Doppel-Null, mit FIXER Familie ohne Forge-Suche).
   Jackknife-Stabilität 1.00, chance-gates klar. Details COHERENCE_RQA_SCREEN.md §13. Jul.
2. **Distill-Sanities BEIDE PASS:** ECN nested 0.653 vs lean 0.607 (perm-p 0.016, NULL
   0.066≈chance) + GEOMAG binär perm-p 0.016, NULL 0.523≈chance — ehrlich: GEOMAG nested
   0.883 < lean 0.922 → Receipt zeigt korrekt „distill lohnt hier nicht über lean"
   (lean auf GEOMAG war schon 0.938). logs/{ecn,geomag}_spec*.{json,md} committed.
3. **Produkt-Befund Distill-Ingest:** BOM+Header+ragged .txt (ECN) bricht `genfromtxt` →
   Banken als .npy exportiert (`data/distill_banks/{ecn,geomag}/`, 14/16 Recordings).
   Ingest-Härtung = Kandidat für die Produkt-Liste.

## [✅ ENTSCHIEDEN + GEBAUT 13./14. Jul, s.o.] PRIO 1 (alt) — RQA-als-Familie-Entscheid
Empfehlung wurde übernommen: eigene distill-Premium-Familie mit Kosten-Quittung, NICHT
Forge-Slot. Praxis-Case auf CALCE/HYD-cooler gelaufen → beidseitig EXCLUDED (ehrlich, s.o.).

# ARCHIV: Stand 12. Jul (PRIO 0 Ep-51 unten gilt WEITER)

> **⚠️ REGEL: keine Subagents** (2× Session-Limit gerissen am 11. Jul). Alles inline,
> schwere Rechnung als lokale Background-Prozesse (nie nohup+& im Harness).

## 🌋 PRIO 0 — Ep 51: Fenster verschoben auf **13.–15. Jul** (HVO-Update 12. Jul 18:12Z)
Deflation hat das Fenster zurückgeschoben („eruption is paused … likely between July 13 and
July 15"). `started_heuristic=False`, Watcher gesund (blackout_streak=0).
**1. Befehl: `python3 research/factory/session_status.py`.** Wenn gekippt → SOFORT die 3 applies
mit HVO-Onset-Zeit (UTC!), ehrlich (Prereg-3 = „retrospektiv-nicht-gestützt", prospektives NULL
erwartbar und wertvoll):
```
.venv-research/bin/python3 research/factory/ep51_prereg.py apply --start <UTC> --end <UTC>
.venv-research/bin/python3 research/factory/ep51_prereg2.py apply --start <UTC> --end <UTC>
.venv-research/bin/python3 research/factory/volcano_precursor.py apply51 <UTC-start>
```
Danach committen. NICHT nachträglich an Preregs drehen.

## ✅ ERLEDIGT 12. Jul (Commits e33f227, fb195ca, d018946 + LIGO-Folge)
1. **WS4 Zeit-Faktor ABGESCHLOSSEN:** CALCE (cdcg+vdcg 0/29, 0 FP) + GEOMAG-Onset (BOU/FRD)
   alle **CSD-NULL**; Determinismus (3 Readouts byte-identisch re-run); Cross-Family-Bilanz
   **CSD 0/4 Familien** → `TIMEFACTOR_CALCE_GEOMAG.md` + RESULTS.md-Sektion.
2. **WS2 Distill-MVP FERTIG:** Fix war appliziert, mit 2 Regression-Tests gepinnt (MAD=0 +
   Envelope-vs-fixes-4σ), Suite 57/57, CWRU-Sanity PASS (Gate 50 Prog/Rec, nested 0.864,
   perm-p 0.032, NULL≈chance), README-Absatz.
3. **PRIO-3-Fährten 1+2 VERIFIZIERT (fair RQA, source-rebuilt, Ledger 851c8338):**
   CALCE-soh RQA 0.860 vs lean 0.579 (+0.281 CI-fest), HYD-cooler 0.689 vs 0.533
   (+0.156 CI-fest); Frontier-Update in COHERENCE_RQA_SCREEN.md. MFPT-Screen-Flag =
   cheap-config-Artefakt (A4-fair-TIE gilt); CWRU war schon fair verifiziert.
4. **PRIO 4 LIGO-Replikation BESTANDEN (Ledger ced71461):** Koi_Fish vs Whistle, 31 Recs:
   forge 0.798 CI[0.698,0.887], gepaart +0.169 CI-fest → Champion forge (1. Forge-Win auf
   LIGO), **CLF-ROBUST HOLDS (LogReg 0.766)**, perm-p 0.015, STABLE 0.84. **Domäne #11 =
   robuster Zeiger** (Blip-Paar behält RF-only-Flagge). Details LIGO_RESULTS.md.

## PRIO 1 — offene Fährten aus dem Coh/RQA-Screen
1. **HYD-cooler × Kohärenz +0.153** (Doppel-Flag-Bank, 2. Familie noch UN-verifiziert):
   Kohärenz-Familie braucht eigenen fairen Rebuild-Pfad (Screen-Features = cheap proxy).
2. **GAS-id × Kohärenz +0.147** (8-Kanal): dito.
   Muster: gauntlet-artige Quittungen (paired CI + Stabilität + chance-gate), Ledger-Beleg.

## PRIO 2 — RQA-als-Familie konsolidieren
RQA ist jetzt CI-fest besser auf 3 Banken (CWRU/CALCE-soh/HYD-cooler), TIE 2, schadet 3.
Kandidat: RQA-Slot in die Forge-Grammatik (wie Kanal-Kombinator These #4) MIT Kapazitäts-Gate,
oder als eigene distill-Familie mit Kosten-Quittung (~200× teurer, Wert nur wo CI-fest).
Entscheidung + ggf. Bau erst nach User-Rücksprache zur Prioritätenlage.

## PRIO 3 — Distill-MVP Ausbau (nach CWRU-Sanity)
Weitere Bank-Sanities (ECN klein/schnell; GEOMAG binary), dann distill auf eine der
RQA-Gewinner-Banken als Praxis-Case. Public-Push weiterhin NUR nach User-Freigabe.

## Disziplin / Gotchas (unverändert)
- venv `../../.venv-research/bin/python3`; cwd research/factory (heredocs brauchen `cd`).
- prereg-VOR-Readout, LOGO leak-frei, chance-gated CI + perm-p + shuffle-null, Ledger-Beleg.
- 2 launchd-Services NICHT killen (ep51watch, batterytransfer=DONE, geomagwatch).
- Nur-SignalMap-Scope [[feedback-signalmap-scope-only]]. Gauntlet NIE inline/durch tail pipen.
- zsh splittet unquoted Vars NICHT (`for cfg in "3 5"` → als 1 Arg übergeben; explizit loopen).
# ARCHIV: NEXT SESSION PLAN (aktualisiert 8. Jul 2026)

> **✅ ERLEDIGT 8. Jul 2026 — ZEIT-FAKTOR auf IMS-RUL (Commits 5508eaa/82adfc8, `ims_csd.py`).**
> CSD-Theorie-Anker (Scheffer/Dakos) auf Lager-Run-to-Failure, 2 Indikatoren prereg-VOR-Readout
> (Ledger befc598b RMS + 396bf8d7 KURT). **CSD-NULL bei beiden**, aber die Spezifitäts-Kontrolle
> deckte auf: RMS-CSD ist rig-global (2/8 HEALTHY passen, 0/4 failed → invertiert); mit dem
> failure-spezifischen Kurtosis-Indikator drehen die Healthy-FPs auf 0/8, aber nur 1/4 failed passt
> (AR1-Fingerabdruck = Flaschenhals, n_failed=4 deckelt Power). Verdikt korrekt CSD-NULL. Cross-Family:
> CSD feuert weder auf Vulkan noch auf IMS als bestätigter Frühwarner → disziplinierter Negativ-Beleg,
> stringenter Theorie-Anker. Details: RESULTS.md §ZEIT-FAKTOR — IMS-RUL. Determinismus repro, Ledger OK.
>
> **⚡ NÄCHSTE Session (Reihenfolge):**
> 1. `session_status.py` (1. Befehl). Ep 51 Fenster jetzt **10.–14. Jul** — falls gefeuert: 3 Ep51-applies
>    (ep51_prereg/ep51_prereg2/apply51) ehrlich als „retro-divergent" ausführen+committen.
> 2. Sonst Zeit-Faktor weiter auf **CALCE-Batterie** (life-terciles, echter Degradations-Prozess, CSD-Anker
>    direkt reuse via ims_csd-Trajektorie-Muster) ODER **GEOMAG-Onset** (Sturm-Beginn-Timing).
>    Pre-registrierbarer IMS-Nachfolger falls gewünscht: defekt-Band-Envelope-Indikator + Restriktion auf
>    Vor-Ausfall-Fenster (NEUE Prereg, ims_csd nicht tunen).
> 3. ODER WS3 Phase 2/3 (MOX-Discovery-Readout als Plattform-Fähigkeit generalisieren) — nicht zeitkritisch.
>
> ---
> **Historie (7. Jul 2026, Commits 421408d…d8d0250):** Triangulation am Vulkan-Präkursor. Weg 2 (CSD,
> `csd.py`) NULL/NULL, Weg 3 (Spektral, `spectral.py`) NULL/NULL, Konsilienz
> (`consilience.py`) DIVERGENCE (rho≈0). VERDIKT = ehrlicher DOWNGRADE des
> −12h-Präkursors (Weg-1-Hit überlebt nicht → Textur-Artefakt). Alle prereg-VOR-
> Readout gefroren. Details: `TRIANGULATION.md`. Weg 4 unnötig (2 Theorie-Wege
> flach + Konsilienz auf Zufall). PREREG-3 Ep-51 bleibt frozen, jetzt korrekt
> „retrospektiv-nicht-gestützt" gelabelt. **NÄCHSTE Session:** wenn Ep 51 feuert
> (Fenster 9.–14. Jul) → 3 Ep51-applies ehrlich ausführen+committen; sonst
> Zeit-Faktor auf andere Familien (IMS-RUL/CALCE/GEOMAG-Onset) oder WS3 Ph 2/3.
> Das Untenstehende ist der erledigte Original-Plan (Referenz).


**Strategischer Kontext (aus Strategie-Session Jul 5/6):** Fokus weg von „mehr Domänen
messen" hin zu **un-fälschbarer prospektiver Vorhersage** + **Multi-Methoden-Konsilienz**.
Gegen-Position zu Milliarden-AI-for-Science: nicht Skala, sondern Disziplin + übersehenes
Signal + Realitäts-Verifikation. Trust-Harness IST das Produkt.

**⏰ ZEITKRITISCH:** Ep 51 prognostiziert **9.–14. Jul** (heute 6. Jul). Alle Triangulations-
Methoden müssen **preregistriert + eingefroren SEIN, BEVOR** Ep 51 eruptiert — sonst ist der
prospektive Test wertlos. **Priorität = Methoden bauen + freezen VOR dem 9. Jul.**
1. Befehl jeder Session: `python research/factory/session_status.py` (Ep-51-Watcher-Status).

---

## PRINZIP (stehende Regel, [[feedback-multi-method-triangulation]])
Jede Vorhersage-Behauptung braucht **3-4 Wege mit ORTHOGONALEN Fehler-Modi** (nicht Anzahl,
sondern Unabhängigkeit). Konvergenz-unter-echt vs. Streuung-unter-Shuffle = der echte Test.
Divergenz = ehrlicher Downgrade (gültiges Ergebnis). Mind. 1 Weg theorie-getrieben + gerichtet.

---

## DIE 4 WEGE (auf −12h-Präkursor, LOGO über HVO-Episoden, leak-frei)

Gemeinsame Bank: die 544 gecachten Präkursor-Segmente + 84 EARLY-Segmente (pause_phase),
2 Stationen (UWE/RIMD). T−Δ-Fenster darf NUR Prä-Onset-Daten sehen (Leak-Gefahr!).

1. **Weg 1 — Textur-Klassifikator [EXISTIERT, `volcano_precursor.py`].** lean-Duo (perm_entropy
   + psd_slope) LOGO, −12h vs. tiefe-Pause-Kontrolle. Failure-Mode: overfittet Fenster-Statistik.
   Bereits exploratorischer −12h-Hit (UWE/RIMD 0.656, p .008/.003); late-pause-spezifisch belegt.

2. **Weg 2 — Critical Slowing Down [THEORIE-ANKER, gerichtet, NEU].** Early-Warning-Signals
   (Scheffer/Dakos): nahe einem Kipp-Punkt MUSS die lag-1-Autokorrelation (AR1) + Varianz
   STEIGEN. Gleitfenster über die Pause → teste, ob AR1 & Var im letzten 12h-Fenster höher als
   in tiefer Pause (gerichteter, falsifizierbarer 1-seitiger Test, NICHT Klassifikation).
   Failure-Mode: rausch-empfindlich, tritt nicht immer auf. **Unabhängig auf allen 3 Achsen von
   Weg 1** (dynamische Systeme vs. lernbasiert). Reuse: detrend/window aus volcano_precursor.
   Prereg: AR1-Anstieg + Var-Anstieg als PRIMARY, je 1-seitig; Kontrolle = EARLY-Pause (sollte
   NICHT erhöht sein → sonst Gradient statt Präkursor).

3. **Weg 3 — Spektral-Wanderung [NEU].** 1/f-Exponent (spektrale Steigung) und/oder Wavelet-
   Band-Energie-Verschiebung −12h vs. Kontrolle. Failure-Mode: Amplituden-Konfund (→ amplitude-
   normalisieren, als Kontrolle mitführen). Repräsentation = spektral (orthogonal zu 1+2).

4. **Weg 4 (optional, wenn Zeit) — Cross-Station-Informationsfluss.** Transfer-Entropy /
   prädiktive Information UWE↔RIMD steigt vor Onset? Failure-Mode: schätz-instabil, n klein.

---

## KONSILIENZ-LAYER (das eigentlich Neue — nicht nur 4 Zahlen)
- **Per-Episode-Agreement:** flaggen die Wege DIESELBEN Episoden bei −12h? (nicht nur gleiche
  Mittel-Accuracy) → Korrelation der per-Episode-Scores.
- **Konvergenz-unter-echt vs. Streuung-unter-Shuffle:** Episode-Labels permutieren; die Wege
  MÜSSEN bei Shuffle auseinanderlaufen. Übereinstimmung auch bei Shuffle = gemeinsamer Artefakt.
- **Entscheidung:** unabhängige Wege stimmen bei echt überein + streuen bei Shuffle → KONSILIENZ-
  Upgrade (erster echter Beweiskette-Punkt). Divergenz → diagnostizieren + ehrlicher Downgrade.

## PROSPEKTIVER HOOK (un-fälschbar)
Wenn Ep 51 im Fenster feuert: `session_status.py` → alle 4 Wege + die 3 bestehenden Ep51-
Preregs prospektiv auf das ungesehene Pre-Onset-Fenster anwenden. **Hält die Konsilienz auf
einem echten, unbekannten Ereignis?** Das ist der Diamant. n=1 ist Demo-Quittung, ehrlich so labeln.

---

## REIHENFOLGE (nächste Session)
1. `session_status.py` (Ep-51-Status; falls schon eruptiert → SOFORT bestehende 3 applies + so
   viele Wege wie fertig, ehrlich dokumentieren was pre/post-Freeze war).
2. Weg 2 (CSD) bauen + prereggen + freezen (Ledger, commit VOR Readout). Theorie-Anker zuerst.
3. Weg 3 (Spektral) bauen + prereggen + freezen.
4. Konsilienz-Layer (per-episode agreement + shuffle-Streuung) als eigenes Skript.
5. Weg 4 nur wenn Zeit vor 9. Jul.
6. Alle Verdikte in Ledger + TRIANGULATION.md; Gauntlet NIE inline/tail.

## Disziplin / Gotchas (unverändert)
- venv `../../.venv-research/bin/python`; cwd research/factory (heredocs brauchen `cd`!).
- prereg-VOR-Readout, LOGO leak-frei, chance-gated CI + perm-p + shuffle-null, Ledger-Beleg.
- 2 launchd-Services NICHT killen (ep51watch, batterytransfer=DONE). RESEARCH_OVERVIEW.html nicht anfassen.
- Nur-SignalMap-Scope [[feedback-signalmap-scope-only]]. Reuse volcano_precursor/pause_phase-Primitive.
- Status offen: WS1 DONE (WS1_AUDIT.md), WS3 Phase 1 DONE (WS3_DISCOVERY.md, MOX-Fingerprint
  CONFIRMED). WS3 Phase 2/3 (Discovery-Readout generalisieren) = NACH Triangulation, nicht zeitkritisch.
