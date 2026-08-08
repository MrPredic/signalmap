# LIGO-Glitches (Domäne #11, GWOSC H1+L1) — Gauntlet-Ergebnis (11. Jul 2026)

**Bank:** Gravity-Spy-Glitch-Klassen **Blip vs Scattered_Light**, 4s-Strain-Segmente
um GPS-Zeitstempel, H1+L1 je 8 Events/Klasse → **32 Recordings (=Events), 256 Fenster,
chance=0.500**. Cache: `cache/ligo/ligo_bank.npz` + `provenance.json` (URLs/SHA256).
Loader: `ligo_loader.py`. Run: `ligo_run.py` (deterministisch, n_perm=200).

## Zahlen (log: `logs/ligo_gauntlet.log`)

| Metrik | Wert |
|---|---|
| LEAN (LOGO über Events) | **0.934** CI [0.898, 0.965] |
| group-perm-p (200 Perms) | **0.005** |
| FORGE nested (Gate 897 Programme = C=50×32) | 0.934 CI [0.887, 0.973] |
| PAIRED forge−lean | +0.000 CI [−0.035, +0.035] → tie→lean |
| STABILITY | top1-freq 0.97, mean-jaccard 0.70 (STABLE) |
| CLF-ROBUSTNESS (LogReg auf Champion-Features) | **0.570 → FLAG: model-dependence (RF-only)** |

Stabile Selektion über Folds: `specflat(sign(diff2(x)))` + `zcr(diff2(rank(x)))` +
`iqr90(diff(rollstd(x)))`.

## Ehrliches Verdikt

**HIT mit Vorbehalt.** Physik-Familie #11 trennt Glitch-Klassen weit über Chance
(0.934, p=0.005, cross-detector H1+L1 ab Tag 1, leak-frei LOGO). ABER die
CLF-Robustness-Flagge steht: LogReg auf denselben Features fällt auf 0.570 —
die Trennung ist aktuell RF-spezifisch (nichtlinear), nicht klassifikator-invariant.
Kein Upgrade zu „robuster Zeiger" ohne (a) Robustness-Fix oder (b) Replikation
auf zweitem Klassen-Paar. n ist klein (8 Events/Zelle) — bewusst kleiner
Erstkontakt, Skalierung nur wenn gerechtfertigt.

**Nächster ehrlicher Schritt (nicht heute):** 2. Klassen-Paar (z.B. Koi_Fish vs
Whistle) als Replikation + LogReg-Robustness erneut prüfen.

## Replikation Klassen-Paar 2: Koi_Fish vs Whistle (12. Jul, Ledger ced71461)

**Bank:** gleiche Selektionsregel (dichteste 8 Events je (Detektor, Klasse), conf≥0.9, O1),
H1/Whistle nur 7 Events verfügbar → **31 Recordings, 248 Fenster**, chance 0.500.
Loader jetzt parametrisiert (`load_ligo(classes=, tag=)`, Original-Bank byte-identisch);
Run: `ligo_run.py Koi_Fish Whistle`; Log `logs/ligo_koi_whistle.log`; eigene Provenance
`cache/ligo/provenance_koi_fish_whistle.json`.

| Metrik | Wert |
|---|---|
| LEAN (LOGO über Events) | 0.629 CI [0.544, 0.706], group-perm-p **0.015** (200) |
| FORGE nested (Gate 897 = C=50×31, alle) | **0.798** CI [0.698, 0.887] |
| PAIRED forge−lean | **+0.169 CI [+0.060, +0.274]**, forge>lean 23/31 → **CHAMPION = forge** |
| STABILITY | top1-freq 0.84, mean-jaccard 0.52 (STABLE) |
| CLF-ROBUSTNESS (LogReg auf Champion-Features) | **0.766 → HOLDS** |

Stabile Familie: `specratio(rollstd(rank(x)))` + `ordtrans(rollstd(·))`-Varianten +
`specratio(sign(diff2(x)))` — Rolling-Volatilitäts-Morphologie (orthogonal zur lean-Textur,
passend zu Koi-Fish/Whistle-Unterschied in Burst- vs. Gleitton-Struktur).

## Verdikt-Update (ersetzt „HIT mit Vorbehalt" als Domänen-Urteil)

**Domäne #11 REPLIZIERT auf 2. Klassen-Paar — und diesmal klassifikator-invariant** (LogReg
0.766 vs RF 0.798; die Blip-Paar-Flagge betraf also das PAAR, nicht die Domäne). Zudem
erster **Forge-Win auf LIGO** (+0.169 CI-fest) — konsistent mit dem Muster „Forge gewinnt
auf Nicht-Vibrations-Domänen". Ehrliche Grenzen: n klein (31 Events, H1/Whistle 7/8),
Fold-Streuung hoch (einzelne 0.0-Folds bei 8 Fenstern/Event), O1-only. **Status: Domäne #11
= robuster Zeiger** (2 Paare, beide p≤0.015, eines CLF-robust + forge-CI-fest); Blip-Paar
behält seine RF-only-Flagge.
