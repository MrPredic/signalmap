# PREREG — distill Premium-Familie (RQA) Praxis-Case (13. Jul 2026)

**FROZEN BEFORE READOUT.** Nach dem Einfrieren (Ledger-Eintrag + Commit) wird an
Parametern, Bank-Definitionen und Kriterien NICHTS mehr geändert.

## Kontext / Entscheid
RQA ist ab heute **distill-Premium-Familie mit Kosten-Quittung** (NICHT Forge-Slot)
— Entscheid gemäß Empfehlung NEXT_SESSION_PLAN.md PRIO 1: RQA hilft 3 / schadet 3
Banken familienspezifisch; O(n²)-Precompute würde Forge-Budget überall fressen.
Implementiert in `signalmap/premium.py` + `signalmap/distill.py` (Champion-Regel:
Premium kommt NUR bei paired-CI-festem Sieg in die Deploy-Spec; Kosten-Quittung
ms/window im Receipt). pyunicorn-Parity in Test-Suite gepinnt (66/66 grün).

## Hypothese (PRIMARY, je Bank ein eigener Verdikt — BEIDE werden berichtet)
Auf den beiden RQA-Gewinner-Banken des fairen Rebuilds (Ledger 851c8338, 12. Jul:
CALCE +0.281 CI-fest, HYD-cooler +0.156 CI-fest, jeweils RQA vs lean) gilt:

> distill-augmented (base-Selektion + FIXE 6-Feature-RQA-Familie) schlägt die
> base-Selektion paired-CI-fest (95%-Bootstrap-CI der per-Fold-Differenz, lo > 0)
> → Champion-Regel-Verdikt **INCLUDED**.

**Ehrlichkeits-Klausel:** Die Referenzwerte belegen RQA vs LEAN. Die base-distill-
Selektion liegt auf CALCE bei ≈0.90 (Forge-Niveau) > RQA-allein 0.860 → INCLUDED
ist NICHT garantiert. **EXCLUDED ist ein gültiges, berichtenswertes Produkt-Ergebnis**
(„Receipt verweigert Premium wo es nicht zahlt") und wird ohne Nach-Tuning so
committet. Kein Grid, kein Re-Run mit anderen Parametern.

## Banken (exakt wie Screen/rqa_fair, Jul 12)
1. **CALCE** — `retro_loaders.load_calce(per_stage=2)`: 6 Zellen × 3 Life-Terciles
   (early/mid/late), Recording = Zelle×Stage-Block, LOGO-Einheit = Recording (18).
   Caveat (registriert): absolute Accuracy optimistisch vs leave-one-CELL-out;
   die gepaarte Δ-Aussage teilt dieselben Folds → fair für den Verdikt.
2. **HYD-cooler** — `retro_loaders.load_hydraulic(target="cooler")` (PS1,
   per_class=6): 18 Recordings, 3 Klassen.

## Fixe Parameter (keine Suche)
- distill: C=50, kmax=5, thr=0.005, cand=60, trees=100, seed=0, **n_perm=200**
  (Verdikt-Konvention), null_check=True.
- RQA-Familie: dim=3, tau=5, rr=0.1, Supremum-Norm, volle 1024-Fenster,
  6 Maße (RR/DET/LAM/ENTR/L_mean/TT) — die PRODUKT-DEFAULT-Config, kein Grid
  (Lektion coherence_fair: fixe Familie = der ehrliche Produkt-Claim).

## Sekundäre Quittungen (müssen im Receipt stehen, keine Pass-Pflicht für PRIMARY)
- distill-Gates: nested LOGO > chance+0.05, group-perm p ≤ 0.05, NULL ≈ chance.
- Kosten-Quittung: RQA ms/window vs base ms/window (erwartete Größenordnung ~10²×).

## Triangulation (proportional — Produkt-Receipt-Demo, kein neues Physik-Novum)
1. distill-Premium-Receipt selbst (numpy-RQA, augmented vs base, paired CI).
2. Unabhängige Referenz existiert: pyunicorn-RQA-allein vs lean (Ledger 851c8338,
   andere Implementierung + andere Pipeline).
3. Null-Kontrollen: eingebauter Label-Shuffle-NULL + group-perm-p.
4. Implementierungs-Parity: numpy-RQA == pyunicorn (rtol 0.05, Test-Suite).

## Determinismus
Ein Befehl reproduziert alles (Seeds fix: RF random_state=0, Bootstrap seed=0,
NULL seed=1):
```
cd research/factory && ../../.venv-research/bin/python3 distill_premium_case.py
```
Output: `logs/distill_premium_{calce,hydcooler}_report.md` + spec.json je Bank +
Ledger-Einträge DISTILL-PREMIUM-{CALCE,HYDCOOLER}.
