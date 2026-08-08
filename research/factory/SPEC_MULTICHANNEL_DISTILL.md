# SPEC — Multi-Channel-Ingest für distill (Kohärenz-Premium-Familie)

**Status: SPEC ONLY (14. Jul 2026). Impl in eigener Folge-Session.** Beleg für den
Wert liegt vor (coherence_fair, Prereg 4bb76422, 13. Jul): HYD-cooler aug 0.944
(+0.319 CI-fest), GAS-id aug 0.516 (+0.216 CI-fest) — beides Multi-Channel-Signale,
die die heutige 1-Kanal-Bank strukturell nicht tragen kann.

## Ziel
`distill --premium coherence` auf einer Multi-Channel-Bank, mit unveränderter
Champion-Regel (paired CI über Recordings) und unveränderter Kosten-Quittung.

## Design-Entscheidungen (fix)
1. **Fenster-Form:** Multi-Channel-Window = 2D-Array `(C, 1024)`, synchron
   geschnitten, detrend + z-norm PRO Kanal (exakt `window()`-Semantik je Kanal).
   1D-Fenster bleiben byte-identisch zum Status quo (Backward-Compat-Pin im Test).
2. **Primary-Kanal-Konvention:** Grammar-Programme (base) sehen IMMER nur Kanal 0
   (`w[0]` bei 2D). Damit bleibt die gesamte base-Selektion, Gauntlet, Budget-
   Formel und jede bestehende Bank unberührt — Multi-Channel ist REIN additiv.
3. **PremiumFamily bekommt ein Feld `needs_channels: int = 1`.**
   - `rqa` bleibt 1 → erhält `w[0]` bei 2D-Fenstern (keine Änderung).
   - `coherence` deklariert 2 → erhält das volle 2D-Fenster; bei 1-Kanal-Bank
     verweigert distill die Familie LAUT (SystemExit mit Kanalzahl), nicht still.
4. **Kohärenz-Familie (fixe Produkt-Config, kein Grid):** magnitude-squared
   coherence je Paar (Kanal 0, Kanal j), band-gemittelt in n_bands gleichbreite
   Bänder; nperseg/n_bands = die in coherence_fair CONFIRMED-Config (aus
   logs/coherence_fair.csv übernehmen, NICHT neu suchen — Lektion: fixe Familie
   = ehrlicher Produkt-Claim). Feature-Namen `coh_ch{j}_b{k}`.
5. **Ingest:** `load_bank` akzeptiert zusätzlich
   - CSV mit Header → Kanäle = benannte Spalten (Reuse `signalmap.multichannel.
     load_channels`, inkl. NaN-Zeilen-Drop bei erhaltener Synchronität),
   - 2D-.npy `(C, n)` bzw. `(n, C)` mit expliziter Orientierungs-Heuristik
     (längere Achse = Zeit) + Override-Flag.
   Text-Ingest-Härtung (BOM/Header/ragged, 37fd107) gilt pro Spalte weiter.
6. **spec.json:** neues optionales Feld `channels: list[str]` (Namen, Reihenfolge
   = Fenster-Zeilen). Fehlt es → 1-Kanal-Spec (alte Files laden unverändert;
   Roundtrip-Pin wie `test_spec_premium_roundtrip_and_backward_compat`).
7. **DistilledDetector:** unverändert — `spec.featurize` kapselt die Kanal-Logik;
   fit/monitor/alert sehen weiter nur Feature-Vektoren. `fit` prüft, dass die
   Fenster-Kanalzahl zur Spec passt (fail-closed, klare Meldung).

## NICHT in Scope
- Kein Multi-Channel in der base-Grammatik (kein neuer Suchraum, Budget-Formel
  unangetastet). Kein Kanal-Alignment über Dateigrenzen. Kein Resampling.

## Impl-Reihenfolge (TDD, Folge-Session)
1. Rot: `needs_channels`-Verweigerung + 2D-`window()`-Slicing + spec.json-Feld.
2. Kohärenz-Familie + pyscipy-Parity-Pin gegen `coherence_fair.coh_feats`.
3. E2E: HYD-cooler-Mini-Bank (echte Kanäle) → distill --premium coherence →
   Champion-Verdikt + Kette bis alert (Erweiterung test_e2e_chain).
4. Prereg + voller Lauf HYD-cooler/GAS-id (erwartet: INCLUDED auf HYD-cooler,
   Referenz aug 0.944 vs alone 0.972 — Ehrlichkeits-Klausel wie CWRU-Prereg).

## Akzeptanz (DoD der Folge-Session)
Suite grün inkl. neuer Pins; 1-Kanal-Pfad byte-identisch; `distill --premium
coherence` auf 1-Kanal-Bank verweigert laut; Mini-E2E in Sekunden.
