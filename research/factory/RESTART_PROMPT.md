# RESTART-PROMPT (nächste Session) — Ep-51-Apply-Bereitschaft + Präkursor-Verdikt

SignalMap Research. ERSTER BEFEHL (ersetzt Web-Checks + Explorations-Calls):
  .venv-research/bin/python research/factory/session_status.py
Der zeigt: git/Ledger-Stand, **lokalen USGS-Watcher-Status (Ep 51 gestartet?)**,
Cache-Zähler, laufende Prozesse. Dann RESULTS.md (§EP-51-PREREG-2, §PRECURSOR)
+ memory signalmap_efficiency_moat. Branch research, upstream=private. NICHTS public.
REGELN: nice -n 19; Prozesse ~10 min; checkpointen; keine Subagents; Token-Rahmen
→ eine Sache nach der anderen; bank_audit + gauntlet je NEUE Bank; finale Verdikte
n_perm=200; git pull --rebase vor commit; Bash-Background NIE mit nohup+&;
frozen family + Kriterium + Ledger VOR jedem frischen Load.

## STAND (Jul 4)
- ★ **PREREG-2 EINGEFROREN VOR Episode (Commit 90f1f84, Ledger b4677fdf):**
  episoden-trainiertes lean-Duo (fresh_oot-Bank, 94 Seg/Station), LOGO-Sanity
  UWE 0.851 / RIMD 0.830 (eruptiv 0.813/0.771 >> 0.542 frozen-2018-23).
  Prereg-1 (17eb13d, Ledger 8642426b) UNBERÜHRT. Beide werden ausgewertet wie
  registriert: Prereg-1 = 8-Jahre-Modell, Prereg-2 = Regime-Modell.
- ★ **Zero-Token-USGS-Watcher AKTIV** (launchd com.signalmap.ep51watch,
  stündlich): CAP-Feed + HTML-Snapshots → logs/ep51_watch.jsonl +
  data/volcano/ep51_watch/. KEINE WebFetch-Checks mehr nötig.
- Präkursor-Bank PREREG geschrieben (Ledger 8996bb7d VOR Load), Fetch
  gestartet (data/volcano/precursor, 544 Segmente) — Status im session_status.

## PRIO 1 — Wenn Watcher „started" zeigt (Fenster 9.–15. Jul): DREI Preregs
HVO-Zeiten (UTC!) aus Snapshot in data/volcano/ep51_watch/ ablesen, dann:
  ep51_prereg.py  apply --start <UTC> --end <UTC>     # 8-Jahre-Modell
  ep51_prereg2.py apply --start <UTC> --end <UTC>     # Regime-Modell
  volcano_precursor.py apply51 <start-UTC>            # ★ -12h-Präkursor-Demo
Ergebnis egal wie es ausgeht committen (Ehrlichkeit = das Produkt). HST+10h=UTC.

## PRIO 2 — ERLEDIGT Jul 4: Präkursor-Verdikt (RESULTS.md §VOLCANO-PRÄKURSOR)
-2h NULL beidseitig (auch Forge), ABER ★ -12h Cross-Station-Hit (UWE 0.656
p=.008 / RIMD 0.656 p=.003, exploratorisch) → PREREG-3 (89d7773a) eingefroren.
Amendment 317bce69: paired Banken brauchen Sign-Test statt group_perm_p;
Audit-Checks 6/7 expected-FAIL auf paired Designs (als Check-13-Idee offen).

## PRIO 2b — Zeit-Faktor-Programm: (a)+(b) ERLEDIGT Jul 4/5 (RESULTS.md §ZEIT-FAKTOR)
(a) PAUSE-PHASE-ORDINAL (Prereg 887a5723, Verdikt 2e8bd867): EARLY≈MID
    beidseitig + late>mid persistiert → −12h-Effekt ist LATE-PAUSE-SPEZIFISCH,
    Gradient-Lesart entkräftet → PREREG-3 (prospektiver −12h-Test) aufgewertet.
(b) PRECURSOR-SAMPLING (Prereg 3c979096, Verdikt 71bdf123): peak_sub rettet
    −2h nicht (UWE NULL 0.533); −12h unter peak_sub SCHWÄCHER als lean →
    Effekt lebt in der Vollfenster-Textur, nicht in Impulsen.
(c) OFFEN, nach Ep-51-Apply: Zeit-Faktor auf andere Familien (IMS run-to-
    failure RUL, CALCE Batterie-Degradation, GEOMAG Sturm-Onset).
PROZESS: Gauntlet nie inline/durch tail pipen — separater Lauf + eigenes Log
(pause_phase-run lief 3h17; Verdikte trotzdem vollständig im Ledger).

## PRIO 3 — danach: MOX-Drift 62ch (zenodo.15681119, These #4) oder
DCASE-2026 Task2 2-Kanal (These #1×#4). NICHT: Public, distill, LIGO,
GRIDFREQ-fresh, personenbezogen. B6 Solana-Anker: User-Entscheid, später.

## Token-Sparen (User-Auftrag, fortlaufend)
Automatisiert (0 Token, kein Verlust): USGS-Polling (Watcher), Session-Start-
Exploration (session_status.py), Fetches checkpointed im Hintergrund.
Prinzip: Tokens nur für (1) neue Automatisierung, (2) Validierung/Design;
alles Wiederholbare in Skripte gießen.
