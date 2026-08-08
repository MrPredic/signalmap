# WS1 — Bug / Daten-Integritäts-Audit (5. Jul 2026)

Auditiert: die 8 selbst-geflaggten Verdächtigen aus `NEXT_SESSION_PLAN.md`.
Methode: systematic-debugging (reproduzieren → Root-Cause → fix → verifizieren).
Verdikt vorweg: **kein #11/#12-Verdikt muss korrigiert werden** — die Labels/Features
waren korrekt; zwei fragile Stellen wurden robust gemacht, damit sie es *bleiben*.

| # | Verdächtiger | Befund | Aktion |
|---|--------------|--------|--------|
| 1 | `battery._soh_map` Duplikat-Tage-Mittelung | **BUG (real, aber neutralisiert).** Die zwei „Duplikat"-Blöcke sind NICHT zwei Aging-Bedingungen, sondern **Raw-Kapazität (Ah)** + **bereits normalisierte SOH (≤1)**. `groupby("day").mean()` mittelte Ah (~5.4) mit SOH (~1.0). Ergebnis ist nur *versehentlich* korrekt: Block2 = raw/raw₀, der gemeinsame Faktor `(raw₀+1)/2` kürzt sich bei Division durch day-0 exakt weg. Bricht bei jedem (cell,day) mit nicht-übereinstimmender NaN-Maske. Gemessen: **0/144 Masken-Mismatch → Labels waren korrekt.** | **Fix:** nur Raw-Block lesen + selbst normalisieren (Ground Truth, keine Mittelung), fail-loud bei Struktur. Neue Labels == alte (0/120 diff). Verdikt unverändert. |
| 2 | `_read_eis` stille None-Drops | **0/935 Drops**, 0 ohne cell/day. Build-Meldung „935/935" korrekt, kein LODO-Bias. | **Härtung:** Build zählt Drops jetzt laut (`n_drop`, REVIEW-Flag bei >0). |
| 3 | `run_prog` `except: return 0.0` maskiert Fehler | Gemessen über echte Banks: **battery 0% exc / 0% non-finite**, **mox 0% / 0%**. Die 6–8 % Nullen sind *legitime* 0-Feature-Werte. Feature-Matrizen nicht degeneriert. | Keine Aktion (0 % Trefferquote; kein Over-Engineering im Hot-Path). |
| 4 | `_feature_matrix` Cache-Key = `len(raw)` | Cache-File ist bereits **per-Tag** (`F_aging`=935 vs `F_soh`=859, verschiedene Längen) → keine Kollision zwischen den Banks. Residual: gleicher Tag + gleiche Länge + anderer X-Inhalt → stale F. | **Fix:** Content-Hash (`_bank_hash`) in Cache-Key. Recompute == alter Cache (max-diff 0.0). |
| 5 | Label/Group-Korrektheit (`_cell_day`, MOX `startswith`, `_day_ord`) | battery: **0/935** Cell≠Pfad-Mismatch, 24 Zellen (Regex ankert auf `_convert` → kein SOC-Token-Leak). MOX: **700/700** gelabelt, 0 geskippt, alle 39 Tage × 3 Analyte, g monoton chronologisch. | Keine Aktion (sauber). |
| 6 | nan/inf-Maskierung | Alle Sites: `to_numeric(coerce)` **gefolgt von explizitem finite-Mask** (sound); `_fstat` nan→0 = korrekt-per-Design (zero-variance-Feature nicht selektiert); Feature-`nan_to_num` = No-op (0 % non-finite verifiziert); `nanquantile(sohs)` = sohs upstream finite-gefiltert. **Keine versteckt einen Bug.** | Keine Aktion. |
| 7 | LODO-Leak (prescreen/scaler/lean per-fold) | battery `_loco` + mox `lodo`: prescreen `_fstat(F[tr])` **train-only**, `StandardScaler` **im Pipeline** (fit auf tr), lean_baseline **per-Fenster** (perm_entropy+psd_slope, kein cross-window fit), Feature-Matrix per-Fenster deterministisch, `picks` label-unabhängig (seed 0). **Kein Leak.** | Keine Aktion. |
| 8 | Determinismus | Alle Randomness geseedet: RF `random_state=0`, alle `default_rng(0/seed)`, alle `permutation/choice`. Empirisch: Fold-Scores 2× **identisch**, mean acc **0.7185** = reproduziert das #12-Battery-Verdikt (0.718) exakt. | Keine Aktion. |

## Änderungen (behavior-preserving, verifiziert)
- `_soh_map` — Raw-Block-only + fail-loud (kein Mittel von Ah mit SOH).
- `build()` — lauter Drop-Zähler (IGEL-Soundness: kein stiller Verlust).
- `_bank_hash` + `_feature_matrix` — Content-Hash-Cache-Key gegen stale/kollidierende F.

Alte `F_aging.npz`/`F_soh.npz` haben kein `bank_hash`-Feld → werden beim nächsten
Readout invalidiert und neu berechnet (verifiziert identisch, Verdikt unverändert).
47/47 Package-Tests grün.
