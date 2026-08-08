# TRIANGULATION — Kilauea −12h Präkursor (retrospektiv, HVO-Episoden)

**Frage:** Ist der explorative −12h-Textur-Hit aus `volcano_precursor.py`
(Weg 1, UWE/RIMD segment-majority 0.656, sign-p .008/.003, ledger affbdc1d,
late-pause-spezifisch belegt via `pause_phase.py`) ein **echter** Präkursor
oder ein methoden-spezifisches Artefakt?

**Test (stehende Regel [[feedback-multi-method-triangulation]]):** 3–4 Wege mit
ORTHOGONALEN Fehler-Modi. Konvergenz-unter-echt vs. Streuung-unter-Shuffle.
Divergenz = ehrlicher Downgrade (gültiges Ergebnis, kein Fehlschlag).

Alle Wege prereg-VOR-Readout in `LEDGER.jsonl` gefroren + im Git committed,
BEVOR die jeweiligen Ergebnisse berechnet wurden.

## Die Wege (alle auf PRE(−12h) vs. MID(mittige Pause), gleiche Pause/Uhrzeit)

| Weg | Repräsentation / Fehler-Modus | Prereg-Ledger | Verdikt (UWE / RIMD) |
|-----|-------------------------------|---------------|----------------------|
| 1 — Textur (existiert) | learned RF auf perm_entropy+psd_slope (znorm-Fenster); FM: overfit Fenster-Statistik | affbdc1d | explorativ HIT 0.656 (p .008/.003) |
| 2 — CSD (Theorie-Anker, gerichtet) | AR1+Varianz der langsamen RMS-**Hüllkurve** (roh); FM: rausch-empfindlich | 7dec0460 | **NULL / NULL** (AR1 21/41 p=.50, VAR 20/41 p=.62) |
| 3 — Spektral-Wanderung | centroid/spec_ent/hf_ratio Form (znorm, amplituden-invariant); FM: Amplituden-Konfund (neutralisiert) | d850f965 | **NULL / NULL** (kein Deskriptor bewegt, p .13–1.0) |
| Konsilienz | Spearman pro-Episode-Score zwischen Weg-Paaren; Perm-Null zerstört Episoden-Zuordnung (n=10000) | 4232029f | **DIVERGENCE** |

Weg 4 (Cross-Station-Info-Fluss) nicht mehr nötig: 2 unabhängige Theorie-Wege
bereits flach + Konsilienz auf Zufall → weiterer Weg ändert das Verdikt nicht.

## Konsilienz-Detail (Headline = maximal orthogonales Paar weg1↔weg2)

| Paar | UWE rho / perm_p | RIMD rho / perm_p |
|------|------------------|-------------------|
| weg1_texture ↔ weg2_ar1 (INDEP) | 0.077 / 0.61 | 0.004 / 0.98 |
| weg1_texture ↔ weg2_var (INDEP) | −0.002 / 0.99 | −0.006 / 0.97 |
| weg1 ↔ weg3_* (geteilte Repräs., keine Korroboration) | rho .04–.15, p .32–.79 | rho .00–.15, p .32–.98 |

Ledger: CSD 62117117 · SPECTRAL c71c8e83 · CONSILIENCE 8ab14a04.

## VERDIKT — ehrlicher DOWNGRADE des −12h-Präkursors

Der explorative Weg-1-Hit **überlebt die Triangulation nicht**:
- Zwei **theorie-getriebene, gerichtete** Methoden (CSD, Spektral) finden bei
  −12h **kein** Signal gegenüber der mittigen Pause — auf beiden Stationen.
- Das pro-Episode-Agreement zwischen unabhängigen Wegen ist **nicht von Zufall
  unterscheidbar** (rho ≈ 0, alle perm_p ≫ 0.05); die Wege flaggen NICHT
  dieselben Episoden.

Parsimonischste ehrliche Lesart: der −12h-Hit ist Weg-1-spezifisch (learned
Fenster-Textur = registrierter Fehler-Modus „overfit Fenster-Statistik"), kein
robuster physikalischer Präkursor. Das ist ein **gültiges falsifizierendes
Ergebnis** — genau die Währung der Multi-Methoden-Disziplin, kein Fehlschlag.

## Konsequenz für den prospektiven Ep-51-Test (PREREG-3, frozen)

PREREG-3 (`volcano_precursor.freeze51`, −12h prospektiv auf Ep 51) bleibt
eingefroren und wird bei Eruption ausgeführt — aber jetzt korrekt gelabelt:
er testet eine Hypothese, die die **retrospektive Triangulation NICHT stützt**.
Ein prospektiver n=1-Treffer wäre danach schwach (eine Methode, retrospektiv
schon divergent); ein Fehlschlag ist erwartet. Der ehrliche Beweiskette-Punkt
ist HIER: die Triangulation hat einen explorativen Hit diszipliniert entwertet,
bevor er zu einer Behauptung wurde.

**Nächster echter Präkursor-Kandidat braucht Konvergenz ZUERST retrospektiv,
DANN prospektiv** — nicht die Reihenfolge dieser Runde.
