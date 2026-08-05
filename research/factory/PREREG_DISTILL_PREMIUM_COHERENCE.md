# PREREG — distill Premium-Familie COHERENCE auf HYD-cooler + GAS-id (16. Jul 2026)

**FROZEN VOR READOUT.** Ledger-Receipt `DISTILL-PREMIUM-COH-PREREG` wird VOR dem
ersten Lauf appended; danach wird an diesem Dokument und an den Parametern nichts
mehr geändert. EXCLUDED ist ein gültiges, verkaufbares Ergebnis (die Quittung,
die verweigern kann — CALCE/HYD-RQA-Präzedenz).

## Was getestet wird
`distill(premium=("coherence",))` — die NEUE Multi-Channel-Premium-Familie
(SPEC_MULTICHANNEL_DISTILL.md, Impl 16. Jul, Suite 102 grün inkl. Parity-Pin
gegen `coherence_fair.coh_feats`) — auf den beiden Bänken, deren Kohärenz-Flags
am 13. Jul CI-fest CONFIRMED wurden (Prereg 4bb76422, Ledger 72fa75d6).

## Fixe Parameter (Produkt-Default, KEIN Grid — Ehrlichkeits-Klausel wie CWRU-RQA)
- Kohärenz-Familie: **c128b2** (nperseg=128, n_bands=2, Paare (ch0, chj)) —
  die coherence_fair CONFIRMED-**aug**-Config auf HYD-cooler (0.944). Sie ist
  der Package-Default von `coherence_features` und wird NICHT pro Bank gesucht.
  (GAS-id-CONFIRMED war c256b8 — bewusst NICHT übernommen: EINE Produkt-Config,
  ein ehrlicher Claim. Erwartbar kostet das GAS Punkte.)
- distill: C=50, kmax=5, thr=0.005, n_perm=200, trees=100, cand=60, seed=0,
  null_check=True — identisch zum CWRU-RQA-INCLUDED-Lauf.
- Fenster: (C, 1024), detrend + z-norm PRO Kanal (exakte `window()`-Semantik).
  ⚠️ Deklarierte Abweichung zu coherence_fair: dort liefen die Kohärenz-Features
  auf ROHEN Fenstern. MSC ist skalen-/offset-invariant (Welch detrendet pro
  Segment), der lineare Detrend ändert nur marginal Tieffrequentes — die
  Referenzwerte sind Kontext, kein Pin.

## Bänke (Loader-Calls fix, identisch zur 13.-Jul-Verifikation)
1. **HYD-cooler:** `hyd_multichannel.load_hyd_mc("cooler")` → (6, 1024)-Fenster,
   Kanäle ps1..ps6 in natürlicher Reihenfolge (**ch0 = PS1** = die alte
   lean-Referenz). 3 Klassen, 18 Recordings (LOGO-Einheit = Cycle).
2. **GAS-id:** `gas_multichannel.load_gas_mc()` → (8, 1024)-Fenster @ 4 Hz,
   Kanäle mox1..mox8 in **natürlicher Loader-Reihenfolge (ch0 = erste
   MOX-Spalte)** — bewusst NICHT auf den historisch besten Kanal umsortiert
   (das wäre versteckte Suche). Die alte fair-lean-Referenz nutzte ch1;
   deklarierte Nicht-Vergleichbarkeit.

## Primary (Champion-Regel, unverändert aus distill)
Pro Bank: paired LOGO-Differenz (aug − base) über Recordings, 10k-Bootstrap-CI.
**INCLUDED ⇔ CI-lo > 0.** base = volle distill-base-Selektion (nicht lean!) —
die Frontier-Präzisierung vom 14. Jul gilt: gegen base ist die Messlatte höher
als gegen lean.

## Erwartung (vorab, ehrlich)
- HYD-cooler: coherence_fair aug 0.944 / alone 0.972 vs distill-base 0.800
  (RQA-Lauf 14. Jul) → **erwartet: 2. INCLUDED-Case.**
- GAS-id: fair-aug mit c256b8 war 0.516, mit der fixen c128b2-Config nur 0.429
  (CSV-Zeile c128b2_aug) bei lean 0.300; distill-base unbekannt → offen,
  EXCLUDED gut möglich und als Produkt-Verhalten willkommen.

## Sekundär-Receipts
distill-Gates (nested LOGO, group-perm-p, NULL-Selbsttest) wie immer im Report;
Kosten-Quittung ms/window vs base im premium-Receipt. Alles ins Ledger
(`DISTILL-PREMIUM-COH-<BANK>`), Reports nach logs/.
