# SignalMap — Session Handover (Stand 2026-07-30)

**Zweck:** intensive Übergabe, was als Nächstes an OSS-SignalMap verbessert werden kann.
Diese Session hat geshippt + gehärtet, nicht geforscht. Lies das hier VOR jeder neuen
Forschungsschleife — der Engpass ist Adoption/Reife, nicht neue Methodik.

## 0. Woran diese Session gearbeitet hat (damit nichts doppelt gemacht wird)
- **v0.4.0 live**: `origin/main` = `2428481`, Tag `v0.4.0`, CI grün (`19cfdc6` bestätigt
  success; `2428481` lief bei Sessionende noch — **ERSTER CHECK nächste Session:**
  `gh run list --repo MrPredic/signalmap --branch main --limit 1`).
- Kuratierter Cut aus dem 88845a5-Härtungsstand + Envelope-Familie + Pickle-RCE-Fix.
  Composer/prove/qualify-Schicht bewusst NICHT im Cut (unreviewed, siehe §3).
- **Bugs gefunden+gefixt+gepusht:** Pickle-RCE (`torch.load weights_only=True`, 3 Stellen),
  4 Ruff-Findings (raise-from, 2× unused import, unused loop var).
- **README-Bug gefixt:** erster gezeigter Befehl war `pip install signalmap[all]` (PyPI) →
  404, Paket ist NICHT auf PyPI. Jetzt `## Installation` (git clone + editable install,
  in echtem Fresh-Checkout verifiziert) + Table of Contents (Anchors gg. GitHub-Slugify
  geprüft, inkl. Euro-Zeichen-Sonderfall).
- **Leak-Fund in Git-Historie** (nicht von dieser Session verursacht, schon vorher public):
  Commits `0fe9b79`/`54e152e` (2. Jul) enthalten `<local-path>/signalmap` in einer
  Spec-Markdown. Kein Secret, aber ein lokaler Pfad. Fix wäre History-Rewrite + Force-Push
  = destruktiv, NICHT ohne explizites User-OK ausführen.
- **Invent-Gate-Pivot beschlossen** (Kill-Scan: SensiML Piccolo ist seit Jun 2024 AGPLv3-OSS,
  „AutoML"-Framing damit tot; MCP-TAD = akademischer Nachbar ohne Signatur/Refusal):
  → `PLAN_V0.5_RECEIPTS_MCP.md` (im Repo, privat committed) ist der volle Plan für
  signierte Verdicts + Standalone-Verifier + MCP-Server. **Das ist der wichtigste
  Einzel-Pointer dieser Übergabe — lies die Datei komplett, sie wird hier nicht dupliziert.**

## 1. Sofort-Checks zu Sessionbeginn (Reihenfolge)
1. `gh run list --repo MrPredic/signalmap --branch main --limit 1` — CI von `2428481` grün?
2. `git -C <local-path>/signalmap fetch origin -q && git -C <local-path>/signalmap log origin/main --oneline -3`
3. `git -C <local-path>/signalmap worktree list` — sollte NUR die Hauptzeile zeigen
   (Session hat aufgeräumt, aber verifizieren).
4. `source .venv-research/bin/activate && python -m pytest -q` — Sanity, letzter Stand war 149.

## 2. Offene Lücken direkt am Produkt (klein, schnell, hoher Nutzerwert)

### 2a. PyPI-Publish — der größte Adoptions-Blocker
Aktuell installiert NUR `git clone` + editable install. `pip install signalmap` schlägt
fehl. Braucht **User-Token** (PyPI-Account `MrPredic`, Trusted-Publisher-Setup via GitHub
Actions ist die sauberere Variante als ein statischer Token — beides braucht User-Aktion,
kann nicht autonom von Claude erledigt werden). DoD: `pip install signalmap[all]` in
frischem venv funktioniert, dann README wieder auf den kurzen Pfad umstellen (aktuell
korrekt als "not on PyPI yet" formuliert — muss NACH Publish aktualisiert werden, sonst
lügt das README in die andere Richtung).

### 2b. Test-Coverage-Lücken (kein `coverage`-Tool je gelaufen — `coverage` ist installiert,
noch nie benutzt)
Module ohne erkennbare Testreferenz (grep-basiert, verifizieren nicht blind vertrauen):
`api.py`, `ingestor.py`, `simulate_universal.py`, `sinks.py`, `synth_multimodal.py`,
`visualize.py`. DoD: `coverage run -m pytest -q && coverage report -m` — echte Zahl statt
Vermutung, dann gezielt Tests für die niedrigsten Module ergänzen (TDD: erst roten Test,
der eine reale Lücke zeigt, z.B. `visualize.py` HTML-Output-Struktur oder `sinks.py`
QuestDB/Qdrant-Fehlerpfade).

### 2c. CONTRIBUTING.md Dev-Setup-Redundanz
Hat noch die alte `pip install -e '.[all]'`-Zeile parallel zur neuen README-Installation-
Sektion — nicht falsch, aber zwei Quellen der Wahrheit. Klein: auf `[Installation](README.md#installation)`
verweisen.

### 2d. Roadmap-Punkte aus README (unverändert seit Wochen, ehrlich offen):
- Härtere Realdatensätze (MIMII, IMS, MAFAULDA) + Leaderboard — **NUR nach Prereg**,
  nicht als freie Exploration (Doktrin: Erst-Korpus-Disziplin gilt auch hier).
- HDBSCAN Auto-Clustering + Cluster-Naming (aktuell nur NN-Purity-Heuristik).
- Host-Rust-Capture-Adapter (Audio/Kamera/SDR) — angekündigt, 0 % gebaut.
- Live-Latenz-Novelty via Qdrant kNN im Streaming-Pfad statt Energy-Proxy.

## 3. Composer/Qualify/Methods-Schicht — der größte unreviewte Batzen
Liegt NUR auf `research` (privat gesichert, Commit `501e274`+Folgecommits), **NICHT auf
origin/main**. ~780 LOC + 221 LOC Tests (`composer.py`, `qualification.py`, `methods.py`,
`test_composer.py`, `test_qualification.py`, `test_methods.py`). CLI hat bereits
`prove`/`qualify`-Subcommands — aber NUR im research-Branch, origin/main verifiziert
sauber OHNE diese Referenzen (kein toter Code public).
**Vor jedem Public-Ship dieser Schicht (= v0.5-Plan R4):**
- fresh-eyes-Review gegen die Specs `docs/superpowers/specs/2026-07-21-source-qualification-design.md`
  und `2026-07-23-method-composer-design.md` (liegen NUR lokal/privat, nie in docs/ARCHITECTURE.md
  öffentlich verlinkt — beim Schreiben eines Public-Docs daraus NICHT den lokalen Pfad
  reinkopieren, siehe §0 Leak-Lektion).
- `prove` im README ist aktuell als "experimental" gelabelt — das Framing muss beim
  Review neu bewertet werden: passt es noch, oder wird es Teil des Receipt-Kerns (v0.5 R1)?

## 4. v0.5 „Signed Verdicts for Agents" — der strategische Haupt-Pfad
Siehe `PLAN_V0.5_RECEIPTS_MCP.md` für R1–R5 im Detail. Kurzfassung der Reihenfolge:
**R1 Ed25519-Receipts** (indelible-Muster reusen, NICHT neu bauen) → **R2 Standalone-
Verifier** (importiert nichts aus signalmap, HEDG3-`verify_shipped.py`-Muster) →
**R3 stdio-MCP-Server** (härten wie mcp-shield/HEDG3 Lektionen: bounded readline,
Timeouts, kein Netz) → **R4 Composer-Review→public** (siehe §3) → **R5 Verdict-Korpus +
Announce**. Explizite Kill-Liste im Plan: keine GNN/Graph-Achse, kein FastAPI-Viewer vor
R1–R3, keine neue Feature-Familie ohne Prereg.

## 5. Ökosystem-Radar (aus Kill-Scan 27.Jul, für die nächste Konkurrenz-Prüfung)
- **SensiML Piccolo** (AGPLv3, seit Jun 2024 OSS) — Edge-AutoML-Konkurrent, hat KEINE
  Receipts/Refusal/Signatur. Beim v0.5-Ship gegenprüfen ob sich das geändert hat.
- **MCP-TAD** (Research Square, 2026) — akademisches MCP-Anomalie-Routing-Paper, kein
  geshippt Tool. Nächster Nachbar für R3 — prüfen ob inzwischen Code/Repo dazu existiert.
- **FastAPI 0.138** — Single-Deployment Frontend+Backend, billig für einen späteren
  Receipt-Viewer (NICHT vor R1–R3 bauen).
- GNN/Graph-Anomalie-Papers (GDN/MST-GAT/TopoGDN) — akademisch gesättigt, kein OSS-
  Platzhirsch. Bewusst gekillt als Achse; nur als spätere prereg'd Premium-Familie über
  bestehende Kohärenz-Familie denkbar, NICHT als neue Infrastruktur.

## 6. Was NICHT tun (wiederholte Fehler vermeiden)
- Keine neue Forschungsachse ohne Prereg (Doktrin gilt projektübergreifend).
- Kein History-Rewrite auf origin/main ohne explizites User-OK (§0-Leak).
- Kein Public-Cut der Composer-Schicht ohne fresh-eyes-Review (§3).
- Kein Announce/Marketing vor PyPI-Publish (§2a) — sonst zeigt der erste Eindruck einen
  kaputten Install-Befehl (das war exakt der Bug, der diese Session gefixt wurde).
- CI-Push-Workflow dieser Session als Muster reusen: Worktree von origin/main +
  `git checkout <commit> -- <whitelist>` + Ship-Gate (volle Suite) + Leak-Grep VOR Push,
  nicht direkt auf research→main durchreichen (research enthält private/unreviewte Dateien).

## 7. Sanity-Kommandos (Kopier-fertig)
```bash
cd <local-path>/signalmap
git fetch origin -q && git log origin/main --oneline -3
gh run list --repo MrPredic/signalmap --branch main --limit 1
source .venv-research/bin/activate && python -m pytest -q          # Baseline: 149
coverage run -m pytest -q && coverage report -m                    # noch nie gelaufen — TU DAS zuerst
git worktree list                                                  # muss sauber sein
```
