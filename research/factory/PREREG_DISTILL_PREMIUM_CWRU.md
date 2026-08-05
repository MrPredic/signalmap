# PREREG — distill Premium-Familie (RQA) Praxis-Case CWRU (14. Jul 2026)

**FROZEN BEFORE READOUT.** Nach dem Einfrieren (Ledger-Eintrag + Commit) wird an
Parametern, Bank-Definition und Kriterien NICHTS mehr geändert.

## Kontext / Warum CWRU
CALCE + HYD-cooler verdikteten beide EXCLUDED (Ledger DISTILL-PREMIUM-CALCE/-HYDCOOLER,
tip 43624253): die RQA-CI-Wins vom 12. Jul galten vs LEAN, nicht vs volle
distill-base-Selektion. CWRU ist die EINZIGE Bank, wo RQA fair ÜBER Forge-Niveau
liegt (rqa_fair Jul 3: RQA m3τ10 0.961 CI-fest vs lean 0.872, paired −0.089
CI[−0.193,−0.009]; forge nested 0.902 — RESULTS.md §rqa_fair, logs/rqa_fair.csv)
→ erwarteter erster INCLUDED-Case. Diese Prereg ist NEU und wird VOR dem Lauf
gefroren, nicht post-hoc nachgeschoben (Auflage aus RESULTS.md §DISTILL PREMIUM).

## Hypothese (PRIMARY, ein Verdikt)
> Auf CWRU schlägt distill-augmented (base-Selektion + FIXE 6-Feature-RQA-Familie,
> Produkt-Default) die base-Selektion paired-CI-fest (95%-Bootstrap-CI der
> per-Fold-Differenz, lo > 0) → Champion-Regel-Verdikt **INCLUDED**.

**Ehrlichkeits-Klauseln (beide registriert VOR dem Lauf):**
1. Die Referenz 0.961 war Config m3**τ10**; die shipped Produkt-Default-Familie ist
   m3**τ5** (rqa_features Defaults). m3τ5 auf CWRU = mean 0.9445 (logs/rqa_fair.csv)
   — immer noch > forge 0.902, aber der Abstand ist kleiner. Es wird KEIN
   Config-Wechsel auf τ10 vorgenommen, kein Grid: die fixe Default-Familie IST der
   Produkt-Claim (Lektion coherence_fair).
2. Die base-distill-Selektion kann Forge-Niveau (~0.90) erreichen oder übertreffen;
   INCLUDED ist NICHT garantiert. **EXCLUDED ist ein gültiges, berichtenswertes
   Produkt-Ergebnis** („Receipt verweigert Premium wo es nicht zahlt") und wird ohne
   Nach-Tuning, ohne Re-Run mit anderen Parametern so committet.

## Bank (exakt wie rqa_fair/forge, Jul 3–12)
**CWRU** — `feature_forge.load_cwru()`: data/cwru_mats (24 .mat, DE_time), W=1024,
detrend+z-norm pro Fenster → 2849 Fenster, 24 Recordings (= LOGO-Einheit),
6 Klassen (B007/B021/IR007/IR021/OR007/OR021).

## Fixe Parameter (keine Suche; identisch zur CALCE/HYD-Prereg)
- distill: C=50, kmax=5, thr=0.005, cand=60, trees=100, seed=0, **n_perm=200**
  (Verdikt-Konvention), null_check=True.
- RQA-Familie: dim=3, tau=5, rr=0.1, Supremum-Norm, volle 1024-Fenster,
  6 Maße (RR/DET/LAM/ENTR/L_mean/TT) — PRODUKT-DEFAULT, kein Grid.

## Sekundäre Quittungen (müssen im Receipt stehen, keine Pass-Pflicht für PRIMARY)
- distill-Gates: nested LOGO > chance+0.05, group-perm p ≤ 0.05, NULL ≈ chance.
- Kosten-Quittung: RQA ms/window vs base ms/window (Erwartung ~10²×, rqa_fair: 93
  vs 0.36 ms/win ≈ 259×).

## Triangulation (proportional — Produkt-Receipt-Demo, kein neues Physik-Novum)
1. distill-Premium-Receipt selbst (numpy-RQA, augmented vs base, paired CI).
2. Unabhängige Referenz existiert: pyunicorn-RQA vs lean auf DERSELBEN Bank,
   andere Implementierung + andere Pipeline (rqa_fair Jul 3, logs/rqa_fair.csv,
   RESULTS.md §rqa_fair).
3. Null-Kontrollen: eingebauter Label-Shuffle-NULL + group-perm-p.
4. Implementierungs-Parity: numpy-RQA == pyunicorn (rtol 0.05, Test-Suite 66/66).

## Determinismus
Ein Befehl reproduziert alles (Seeds fix: RF random_state=0, Bootstrap seed=0,
NULL seed=1):
```
cd research/factory && ../../.venv-research/bin/python3 distill_premium_case.py cwru
```
Output: `logs/distill_premium_cwru_report.md` + `logs/distill_premium_cwru_spec.json`
+ Ledger-Eintrag DISTILL-PREMIUM-CWRU.
