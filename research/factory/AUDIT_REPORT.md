# AUDIT-HÄRTUNG — Terminal A, 2026-07-02

Scope: `audit.py` (6 → 12 Checks, CI-Gate), `gauntlet_mixed.py` (Speed- + Honesty-Fix).
Alle Läufe `nice -n 19`, jeder Check <2 min, F-Matrizen gecacht (`logs/cache/`, Checkpoint —
Erstlauf +~2 min Cache-Aufbau, danach Gesamt-Audit **114s**). Logs: `logs/audit_v2.log`,
`logs/gauntlet_mixed_v2.log`.

## 1. audit.py v2 — 12/12 PASS, Exit-Code = #Fails (CI-Gate)

Architektur: `synthetic_controls()` (grammatik-weite Kontrollen) + `audit_bank(name, raw)`
(bank-generisch — **jede neue Bank läuft durch dieselben 9 Checks**, CLI: `audit.py ecn cwru …`).

| # | Check | Failure-Mode | ECN-Ergebnis | Zeit |
|---|---|---|---|---|
| 1 | neg-control-noise | Leak in Ranking/CV | 0.075 (Chance 0.167) | 2s* |
| 2 | pos-control-planted | Pipeline blind für echtes Signal | 1.000 recovered | 2s* |
| 3 | selection-bias-visible | nested/perm-p nur Dekoration | in-sample 1.000 vs ehrlich 0.075 | 0s |
| 4 | group-integrity | Recording in Train+Test | 0 Leaks | 0s |
| 5 | window-provenance **NEU** | identisches/überlappendes Fenster in 2 Gruppen (Datei-Dup, Recording über 2 gids gesplittet) | 0 exakt + 0 near (\|r\|>0.999) cross-group | 0s |
| 6 | class-balance **NEU** | „chance"-Vergleich irreführend bei Majority-Klasse | majority 0.214 < 1.5×chance | 0s |
| 7 | label-shuffle | Signal ist Artefakt | 0.592 → 0.296 | 6s |
| 8 | seed-stability **NEU** | Ergebnis hängt am RF-Seed | Seeds 0/1/2: 0.592/0.571/0.571, spread 0.020 | 9s |
| 9 | determinism **NEU** | unseeded Randomness | Features + LOGO 2× bit-identisch | 6s |
| 10 | feature-degeneracy **NEU** | Konstanten/Duplikate im Ranking | 0 Konstanten; **13/30 unique → Befund B1** | 0s |
| 11 | nested-vs-biased **NEU** | Selektions-Bias auf ECHTEN Daten unsichtbar | biased 0.398 ≥ nested 0.362 (Gap +0.036), nested > Chance+0.05 | 87s |
| 12 | scaler-note | Scaling als Leak-Vektor | 0.592 = 0.592 (RF skalen-invariant) | 3s |

*mit F-Cache; Erstaufbau 40s/Bank (Rauschen/planted) bzw. 29s (ECN), checkpointed.

## 2. Befunde (echt, nicht Check-Kosmetik)

### 🔴 B1: v2-Grammatik enthält algebraische Identitäts-Duplikate
Top-30-ANOVA-Ranking auf ECN: nur **13/30 effektiv unique Cluster** (\|r\|>0.999).
Täter: `diff∘cumsum ≈ id` / `cumsum∘diff ≈ id` ⇒ `acflag(diff(cumsum(x)))` ≡
`acflag(id(id(x)))` etc. — **57% des Greedy-Kandidaten-Budgets verbrannt.**
Kein Accuracy-Leak (Greedy-Schwelle +0.005 skippt Klone), aber Selektions-Diversität
halbiert → verschärft den dokumentierten v2-Kollaps auf kleinen Banken (RESULTS.md).
**Empfehlung (Terminal B / distill): Identitäts-Paare in `programs()` canonicalisieren
oder Dedup (\|r\|>0.999 auf Trainings-F) vor der Selektion.** Check-10-Kalibrierung:
Konstanten in Top-30 = harter Fail (0 toleriert); unique <8 = Fail (Selektion kaputt);
8–29 = PASS mit Warnhinweis im Detail-String.

### 🟡 B2: Reduzierter nested-vs-biased-Check bestätigt v2-Kollaps nebenbei
Check 11 (top=8, kmax=3, trees=40, 87s): biased 0.398 ≥ nested 0.362 — Gap sichtbar,
nested > Chance. Absolutwerte weit unter Voll-Protokoll v1 (0.709), konsistent mit
v2-Grammatik-Kollaps + reduzierten Params. Der Check prüft die **Relation**
(biased ≥ nested > Chance), nicht das Niveau.

### 🔴 B3: gauntlet_mixed v1 hatte einen Prescreen-LEAK (zusätzlich zum Speed-Problem)
v1 berechnete den ANOVA-F-Prescreen auf **allen Fenstern inkl. Test** (alte Zeilen
54–60); nur der Greedy war per-Fold. Mild (Prescreen grob, 250 Kandidaten), aber ein
echter Selektions-Leak. v2 rankt pro Fold **nur auf Trainings-Fenstern** → Leak weg.

## 3. gauntlet_mixed.py v2 — prescreen → feste Top-k ✅ Ziel <5 min übertroffen

- v1: per-Fold-Greedy, jeder Kandidat via innerem LOGO → O(folds × cand × folds × RF);
  HAR (>20 Gruppen) musste gekillt werden.
- v2: pro Fold ANOVA-F auf Train-Fenstern → **feste Top-k (k=5), 1 RF-Fit pro Fold**.
  Selektion sieht held-out-Gruppe nie → weiterhin nested. group_perm_p entfernt
  (bei mixed-label-Gruppen nicht anwendbar, RESULTS.md-Fußnote bleibt gültig).
- **Speed-Regression-Test im `__main__`** (CI-fähig: assert <300s + Signal-Recovery):
  synthetische 20-Gruppen-Bank (die Skala, die v1 killte) + CALCE-LOCO (echt).

| Lauf | v1 | v2 | Accuracy |
|---|---|---|---|
| SYNTH-20G (20 Gruppen, 1000 Progr.) | unbrauchbar (HAR-Skala gekillt) | **28s** | 1.000 (geplantetes Signal, lean 0.346) |
| CALCE-LOCO (6 Zellen, 300 Progr.) | ~Minuten (Greedy) | **12s** | **0.910 CI[0.849,0.957] vs v1 0.846** |

**Nebenbefund B4: feste Top-5 schlägt den Greedy auf CALCE-LOCO** — 0.910 vs 0.846,
gepaart forge−lean **+0.137 CI[+0.025,+0.231]** (v1: +0.073 CI[−0.057,+0.184] n.s.)
→ mit leak-freiem Prescreen wird der CALCE-LOCO-Gewinn erstmals **CI-fest**. Selektion
über Folds komplett stabil (identische 5 Features in 6/6 Folds). Greedy mit innerem
LOGO war auf mixed-label-Banken nicht nur langsam, sondern (bei 6 Gruppen = 5er-inner-LOGO)
auch rauschig.

## 3b. NACHTRAG: B1 GEFIXT + Guard validiert (gleiche Nacht)
`feature_forge.programs(dedup=True, texture_guard=False)`:
- **Dedup implementiert** (synthetische Probes → datenunabhängig, kein Leak;
  dual-Kriterium |r|>0.999 ODER ≥95% bit-equal wegen diskreter argmax-Features):
  2128 → 1150 Programme, ECN-Top-30 13 → **24/30 unique**. Audit v2.1 12/12 PASS,
  Caches auto-invalidiert (Key enthält Programmzahl).
- **Stationaritäts-Guard als Intervention belegt:** ECN nested 0.469 → 0.485
  (nur dedup) → **0.689 (dedup+guard)** ≈ v1 0.709. Dedup allein heilt den
  Kollaps NICHT — Attraktoren sind keine Klone; erst der Guard entfernt sie.
- gauntlet_mixed-Regression auf v2.1: PASS (40s/15s); CALCE-LOCO 0.858,
  gepaart +0.086 CI[+0.010,+0.164] — Win bleibt CI-fest.

## 4. Verdikt
- **audit.py = einsatzbereites CI-Gate**: 12/12 PASS, 114s (gecacht), Exit-Code ≠ 0
  blockt; neue Bank = `audit_bank(name, loader())` bzw. Loader in `BANKS` eintragen.
- **gauntlet_mixed v2 = 10–60× schneller, leak-frei, und auf CALCE-LOCO besser** —
  HAR-Forge (offen seit gestern) ist damit wieder machbar (~20 Gruppen ≈ 30s–5 min
  je nach Fenstern).
- Für Terminal B / distill: B1-Dedup einbauen; Identitäts-Kompositionen beim
  Grammatik-Ausbau vermeiden (`diff∘cumsum`, `cumsum∘diff`, doppelte Monotone wie
  `rank∘tanh` sind Klon-Fabriken).
