# `signalmap distill` — Design (2026-07-02, Reboot-Session)

Ziel: automatische Feature-Destillation pro Domäne mit **Quittung** (Gauntlet-Report),
deploybar in `fit`/`monitor`. USP-Kern = **Kapazitäts-Gate**: Suchraum-Budget ∝ #Recordings.
Das hat weder catch22 (statische 22) noch hctsa (7700+, kein Gate, kein nested).

## 1. Kapazitäts-Gate — Kalibrierung aus der Validierungs-Bank

| Bank | #Rec | Progr./Rec (v1=1092) | Ergebnis | Progr./Rec (v2=2128) | Ergebnis |
|---|---|---|---|---|---|
| ECN | 14 | 78 | 0.709 ✓ (+0.117 vs lean) | 152 | **0.469 Kollaps** |
| MFPT | 20 | 55 | 0.775 ✓ (Parität) | — | — |
| CWRU | 24 | 45 | 0.902 ✓ (+0.030) | 89 | 0.864 degradiert |

Befund: ≤78/Rec überall gesund; 89/Rec degradiert mild; 152/Rec katastrophal.

**Gate-Regel (konservativ, Faktor-2-Marge zur Kollaps-Zone):**

```
budget = min(len(grammar), C * n_recordings)   # C = 50 (Default)
```

- C=50 hält alle drei validierten Banken im gesunden Bereich und bleibt Faktor ~3
  unter der Kollaps-Zone.
- Enumeration deterministisch nach Komplexität (Transform-Tiefe, dann Pool-Kosten),
  damit der Budget-Cut reproduzierbar ist und billige Programme bevorzugt.
- Mechanismus dahinter: die ANOVA-Rangfolge arbeitet auf Recording-Means
  (n = #Recordings Punkte). max-über-N F-Statistiken wächst ~log N; bei kleinem n
  gewinnen Rauschfeatures die Selektion → Kollaps ist Selektionsrauschen, kein
  Overfitting des Klassifikators.
- Sekundäres Gate (optional, Stufe 2): pro Fold Label-Permutations-Null der
  Kandidaten-F-Werte; nur Kandidaten über dem 95%-Quantil des Perm-Max zulassen.
  Adaptiv statt fixem C — braucht eigene Validierung, nicht Teil von v0.

## 2. Pipeline (CLI)

```
signalmap distill --bank <dir|manifest> --label-by <rule> [--budget-c 50] [--out spec.json]
```

1. **Ingest**: Recordings + Labels (bestehender ingest-file-Pfad), Fenster 1024.
2. **Gate**: Grammatik enumerieren, auf `C * n_rec` kappen (deterministische Ordnung).
3. **Nested LOGO** (exakt forge_nested-Logik): pro Held-out-Recording Ranking +
   Greedy-Selektion NUR auf Rest; kmax=5, Schwelle +0.005.
4. **Quittungen** (Gauntlet-Report):
   - nested acc + Chance-Level + lean-Baseline-Vergleich
   - group-perm-p (Labels über Recordings permutiert, ≥60 Perms)
   - NULL-Selbsttest: gleiche Pipeline auf label-geshuffelter Bank → muss ~Chance
   - Kapazität: budget benutzt / Grammatik gesamt / C
   - Kosten: ms/Fenster der finalen Feature-Menge vs catch22-Referenz
5. **Output**: `spec.json` = finale Programm-Strings (Mehrheits-Selektion über Folds
   oder Selektion auf voller Bank NACH bestandener Quittung) + Report-MD.
   `fit`/`monitor` können spec.json als Feature-Backend laden.

## 3. Fold-Aggregation für das Deploy-Spec

Nested liefert pro Fold eine (leicht andere) Selektion — fürs Deployment eine Menge:
- Stabilität = Anteil Folds, die Programm-Familie wählen (ECN: specratio(sign(diff)),
  CWRU: Energie/Crest, MFPT: hent(tanh)) → Familien sind stabil, Instanzen variieren.
- Deploy-Wahl: Selektion auf der VOLLEN Bank, aber nur berichtet mit der nested-Zahl
  als Ehrlichkeits-Anker (biased UB ≈ nested in allen 3 Banken → Verfahren traubar).

## 4. Offene Prios danach (Reihenfolge aus Session-Plan)

1. RQA fair: volle 1024 Samples + pyunicorn-Referenz (Moat-Vorbehalt schließen)
2. Bootstrap-CIs über Recordings (Fold-Varianz beziffern)
3. Domäne 4 (akustisch/Batterie — Datenbeschaffung)
4. E-Waste-Messeigenschaften-Shot (billig-Sensor ≈ teuer-Sensor bei lean features)

## Status-Anker (heute)

- NULL-Kontrolle: ✓ BESTANDEN — forged 0.123 / lean 0.087 vs Chance 0.167
- CWRU-v2 Fold rec 23: ✓ acc=1.000 → v2 komplett 0.870 (24/24)
- Transfer-Befund (family_transfer.py): distill pro PHYSIK-Domäne nötig (ECN-Insel),
  Vibration transferiert cross-rig → optionaler „universal vibration"-Preset.

## Ergänzung aus Kollaps-Diagnose (Pflicht für v0)

Neben dem Kapazitäts-Gate (§1):
- **Stationaritäts-Guard:** integrierende Transforms (cumsum) erzeugen
  1/f²-Attraktoren mit hoher In-Sample-F ohne Generalisierung — nur mit
  Diff/Detrend-Abschluss zulassen oder im Ranking straffen.
- **Stabilitäts-Screening:** Programm-Familie muss über Inner-Folds mehrheitlich
  gewählt werden (v1-ECN-Gewinner: Top-1 in 13/14 Folds = Vorbild).

## Revision nach Kapazitäts-Kurve + Daten-Harvest (Jul 2 nachm.)

1. **Guard vor Gate:** Random-v2-Subsets jeder Größe (125–2128) bleiben auf ECN
   bei 0.42–0.53 — unter lean, weit unter kuratiertem v1 (0.709). Zusammensetzung
   dominiert Größe. Priorität im Design: kuratierte, attraktor-freie
   Basis-Grammatik; Gate begrenzt nur zusätzlich.
2. **Window-Mode = eigene Suchdimension:** HYD-valve gepaart PHASE−Textur =
   **+0.383 CI [+0.208,+0.558]**. distill sucht über {Textur-Fenster,
   phasen-aligniert+skalen-erhaltend} — für zyklische Prozesse Pflicht.
3. **Guard ist kontext-abhängig:** cumsum ist Gift im Textur-Modus, aber
   Informationsträger im Alignment-Modus (`speccent(cumsum(diff))` rekonstruiert
   Zyklusform). Guard-Regeln pro Window-Mode, nicht global.
4. Neue stabile Familien-Bibliothek: Batterie=`specflat(rollstd(diff))`
   (Volatilitäts-Textur der dV-Kurve), Hydraulik-Zyklus=`speccent(cumsum(diff))`.
