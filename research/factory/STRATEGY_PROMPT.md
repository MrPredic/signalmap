# STRATEGIE-SESSION-PROMPT (einmalig, vor der nächsten Arbeits-Session)

SignalMap Research — STRATEGISCHER ÜBERBLICK statt sofortigem Abarbeiten.
Lies memory signalmap_efficiency_moat + .remember/remember.md +
research/factory/RESULTS.md (Jul 3–4) + RESTART_PROMPT.md. Branch research,
upstream=private, NICHTS public. Erst DENKEN/bewerten, dann (nach meinem Go)
arbeiten.

## Was gebaut wurde (Stand Jul 4, ehrlich)
- **Validierungskern (~60-70% fertig, das eigentliche Asset):** gauntlet v2.1
  (nested Selektion, group-perm-p, Stabilitäts-/Champion-/CLF-Robustheits-
  Quittung, Chance-Gate), bank_audit-Pre-Gate, audit.py als 12-Check-CI-Gate,
  SHA256-Hash-Ledger, readout_screen (4 Auslese-Familien × 17 Banken),
  capacity gate + Stationaritäts-Guard, alles checkpointed.
- **Evidenz-Score:** 17 Banken, 10 Physiken, 9 CI-feste Signale, 4 Methoden-
  Thesen — JEDE These inzwischen gehärtet (GAS-Kanal-Kombinator per R2-Holdout
  0.709 CI-fest; IMS-Zeitskala source-level 0.766 p=.005/200; DCASE-Impuls-
  Sampling 0.875 p=.005; HYD-Phase-Fenster +0.383). 3/7 alte Nulls waren
  Auslese-Artefakte; HYD-Nulls bleiben nach 3 Familien ehrlich Null.
- **Kern-Narrativ belegt:** Standard-Auslese (1 Kanal, feste Fenster, even
  Sampling, z-Norm) versteckt Signal — „Standard testet 1 Zelle von 12";
  8/17 Banken zeigen Auslese-Zeiger; Screen-Recall an bekannter Ground-Truth
  validiert.

## Realistisch wichtige Dinge, die passiert sind (nicht schönreden)
1. **RQA-Accuracy-Dominanz ZURÜCKGEZOGEN** (fair getestet schlägt RQA auf CWRU
   sogar Forge); was hält, ist der Kosten-Moat ~200-260× → Story ist
   „Kosten-Accuracy-Frontier mit Quittungen", nicht „wir sind besser".
2. **GRIDFREQ heruntergestuft** (Standort/Perioden-Komponente); Vulkan-Transfer
   versagt cross-station (Site-Effekte). Ehrliche Grenzen sind dokumentiert —
   das ist Feature, nicht Bug, aber es begrenzt die Claims.
3. **Alle Signale kommen aus öffentlichen Benchmark-Datensätzen** — der Moat
   ist der PROZESS (Quittungen, Auslese-Suchdimensionen), nicht die Daten.
   Entdeckungen bisher = Methoden-Artefakte aufgedeckt, keine neue Physik.
4. **Kleine n überall** (5 GAS-Geräte, 12-24 Recordings/Bank, 16 SEIS-Events);
   mehrere Ergebnisse RF-abhängig (CLF-Quittung flaggt). CIs entsprechend breit.
5. **Zeitfenster:** SensiML geht 2026 open-source, catch22 = Prior-Art für
   generisches Lean, NanoEdge = engster Produkt-Konkurrent. Validation-first
   ist die Differenzierung — aber Produkt (distill) + Distribution stehen bei
   ~10% bzw. 0%, per User-Entscheid bewusst verschoben.
6. Free-API-Keys alle tot; alles privat (signalmap-research); Public-Repo
   signalmap ist eingefroren auf Plattform-Stand Juni.

## Strategische Optionen für die nächsten Sessions (bewerten, dann eine wählen)
A. **Matrix-Ernte** (RESTART Prio A): CWRU+crest (0.872→0.993 ≈ RQA eingeholt
   bei ~0 Kosten → Frontier-Tabelle kippt zu unseren Gunsten), MFPT peak,
   SEIS/VOLCANO Rebuilds. Kurzfristig, stärkt die Kern-Story direkt.
B. **Breite: Domäne #11 LIGO-Glitches** (GWOSC ok, H1+L1 = 2. Quelle ab Tag 1).
   Stärkt „domänen-agnostisch", bringt aber keinen Produkt-Fortschritt.
C. **Konsolidierung Richtung Werkzeug:** readout_screen + gauntlet + Ledger zu
   EINEM reproduzierbaren `receipts`-Workflow bündeln (Vorstufe distill, ohne
   Public-Push) — macht das Asset übergabefähig/anschlussfähig.
D. **Entscheidungen, die nur der User treffen kann:** Wann Public? distill-MVP
   starten? Ledger-tip als Solana-Memo ankern (B6)?

Aufgabe der Session: (1) diesen Stand in 5-10 Zeilen eigenständig bewerten
(wo ist das größte Risiko, wo der größte Hebel?), (2) EINE Empfehlung mit
Begründung geben, (3) auf mein Go die gewählte Option abarbeiten. Regeln wie
immer: nice -n 19, ~10-min-Prozesse, checkpointen, keine Subagents, bank_audit+
gauntlet für jede Bank, finale Verdikte n_perm=200, git pull --rebase vor
commit, NICHTS public.
