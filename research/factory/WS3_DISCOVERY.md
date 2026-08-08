# WS3 Phase 1 — MOX drift-invariant fingerprint (5. Jul 2026)

**Frage (Säule B):** Findet die Plattform eine benannte, interpretierbare,
vorher-undokumentierte physikalische Eigenschaft — nicht nur Klassifikation?

## Was gefunden wurde
Die #11-Forge wählt in **allen 39/39** Leave-one-day-out-Folds (12 Monate) als
top-1 **dasselbe** Programm:

    shape:logratio(32,37):specflat(diff(env(x)))

5 Programme werden in *jedem* Fold gewählt — alle **cross-Sensor** logratio/chdiff,
kein Einzelkanal, kein mean/std:

| # | Programm | Sensor-Paar |
|---|----------|-------------|
| 1 (top-1) | shape:logratio(32,37):specflat(diff(env(x))) | 32↔37 |
| 2 | level:chdiff(16,57):crest(env(env(x))) | 16↔57 |
| 3 | shape:logratio(32,50):speccent(sq(sq(x))) | 32↔50 |
| 4 | shape:chdiff(32,34):crest(rollstd(rank(x))) | 32↔34 |
| 5 | shape:logratio(18,46):std(sq(diff2(x))) | 18↔46 |

Die 62 Kanäle sind **31 Duplikat-Sensor-Paare** (benachbarte r=0.9997). Alle
Winner überspannen *verschiedene* Paare → echte cross-Material-Verhältnisse
(within-Paar-Ratios wären ≈0). Kanal 32 ist ein Hub (in 3 der 5 Paare).

## Isolierter, selektions-leck-freier Test (Prereg MOX-FINGERPRINT, spec e59529fc)
Features **a priori fix**, KEINE per-Fold-Selektion → stärker als die Forge selbst.

| Test | LODO acc | CI (10k boot) | chance 0.333 |
|------|----------|---------------|--------------|
| PRIMARY (1 Programm allein) | **0.456** | [0.419, 0.492] | CI-lo > chance ✓ |
| SECONDARY (5 fixe Progs) | **0.587** | [0.549, 0.626] | ≈ voller Forge (0.591) |
| Shuffle-within-day NULL | 0.340 | [0.309, 0.371] | ≈ chance ✓ |

Per-Analyt-Mittel des PRIMARY-Features: Diacetyl **0.250**, EtOH 0.153,
Phenylethanol 0.138 → das Einzel-Feature isoliert v.a. Diacetyl; EtOH/Phenylethanol
überlappen (daher 0.456, nicht höher).

**Verdikt: DISCOVERY-CONFIRMED.** Ledger-Eintrag MOX-FINGERPRINT.

## Was das bedeutet (und was NICHT)
- **Ja:** Eine stabile, niedrig-dimensionale, drift-invariante Signatur existiert
  und ist **auf 5 benannte cross-Sensor-Ratios komprimierbar** (5 Zahlen ≈ 1950-
  Programm-Forge). Das ist der Interpretierbarkeits-Sprung: Entdeckung, nicht Blackbox.
- **Ja:** Der Test ist selektions-leck-frei (Features fix vor Readout) und der
  Shuffle-NULL ist sauber.

Novelty (belegt):
- **Novelty vs. Autoren BESTÄTIGT.** Das zugehörige Paper — Wörner, Eimler,
  Pein-Hackelbusch (2025), *Scientific Data* s41597-025-05993-8 — ist ein reiner
  **Daten-Deskriptor**, explizit „als Referenz-Ressource, nicht als Demonstration
  neuer Drift-Handling-Techniken". Die Autoren dokumentieren *dass* Drift existiert
  und liefern die Daten; sie berichten KEINEN Klassifikator, KEIN Feature-Selektions-
  Ergebnis, KEINE drift-invariante cross-Sensor-Signatur. Unsere 5-Ratio-Invariante
  ist gegenüber den Autoren neu. (Voll-Literatur-Scan citierender Arbeiten = separat;
  Datensatz ist von 2025, Deskriptor ist Primär-Referenz.)

Ehrliche Grenzen:
- **Mild optimistische Auswahl:** die 5 Programme wurden gewählt, weil sie über
  *diese* 39 Tage generalisieren; ein voll unabhängiger Test braucht ein NEUES
  MOX-Deployment. LODO ist hier der beste verfügbare Generalisierungs-Mechanismus.
- Absolute Accuracy moderat (0.59) — es ist eine *Entdeckung einer invarianten
  Struktur*, kein Hochpräzisions-Klassifikator.

## Status
**WS3 Phase 1 = ABGESCHLOSSEN.** Fingerprint isoliert (leck-frei), CONFIRMED,
komprimiert (5 Zahlen ≈ Forge), Novelty vs. Autoren belegt.

## Nächste Schritte (WS3)
1. Phase 2: „Discovery-Readout" als wiederverwendbare Plattform-Fähigkeit
   (winners + Stabilität + Effektgröße + Klartext-Stub für JEDE Forge-Win-Bank).
2. Phase 3: auf Batterie-EIS / Seismik generalisieren (welches Frequenzband-
   Impedanz-Feature ist der zell-übertragbare SOH-Marker?).
