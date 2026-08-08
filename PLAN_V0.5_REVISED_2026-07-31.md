# PLAN v0.5.0 — revidiert nach Kill-Scan (2026-07-31)

**Ersetzt:** `PLAN_V0.5_RECEIPTS_MCP.md` (2026-07-27) — Kern überlebt, **R3 gestrichen**.
**Vorbedingung erledigt:** v0.4.0 live auf PyPI (`pip install signalmap`), Trusted
Publishing via GitHub Actions OIDC, CI grün (lint + 3 Python-Versionen), 172 Tests, 89 % Coverage.

---

## 0. Was der Kill-Scan geändert hat (verifiziert, nicht übernommen)

Der 0.5-Plan vom 27.07. pitchte den leeren Quadranten
*(signierte Verdicts) × (Verweigerung) × (Agent-Konsum via MCP)*. Zwei Befunde
wurden **eigenständig nachgeprüft**, nicht vom Subagenten geglaubt:

| Behauptung | Prüfung | Ergebnis |
|---|---|---|
| PyOD liefert MCP-Server | PyPI-JSON `pyod` 3.6.2 | **bestätigt**: `pip install pyod[mcp]`, `pyod mcp serve` steht wörtlich in der Beschreibung |
| `predictive-maintenance-mcp` besetzt gleiche Domäne | GitHub-API | **bestätigt**: 61 Stars, zuletzt gepusht 2026-07-13, aktiv |
| PyOD deckt unsere Fähigkeit ab | Textanalyse der PyPI-Beschreibung (55 149 Zeichen) | **widerlegt**: „Each row is one timestep; columns are channels/features"; `tabular` 11×, `time series` 13×, aber `raw signal` 0×, `spectrogram` 0×, `sensor` 0×, `permutation` 0× |

**Schlussfolgerung:** MCP ist ein **Kanal**, den der Kategorienführer besetzt — keine
Alleinstellung. Unsere Fähigkeit (Feature-Programm-Suche aus Rohsignal mit Gate,
Permutations-p, Null-Kontrolle, Kostenquittung, Verweigerung) fasst PyOD nicht an.
→ Positionierung wechselt von *„Verdicts für Agenten"* zu *„der Teil, den niemand macht"*.

**Prior Art, die ab jetzt im README zugegeben wird** (nicht als eigene Neuheit führen):
OpenSSF Model Signing / sigstore model-transparency (Signatur + unabhängiger Verifier),
in-toto/SLSA (Attestation), Conformal-Prediction-Reject-Option (arXiv 2506.21802),
Preregistration for Predictive Modeling (arXiv 2311.18807), PyOD (MCP + Rejection-API),
predictive-maintenance-mcp (Abstention in gleicher Vertikale).

---

## 1. Positionierung v0.5 (eine Zeile, in Käufer-Vokabular)

> „Du hast rohe Sensoraufzeichnungen und keine Labels. SignalMap entscheidet, **welche
> Merkmale sich für diese Domäne überhaupt zu berechnen lohnen** — mit nested-LOGO-Genauigkeit,
> Permutations-p, Label-Shuffle-Null und Kosten pro Fenster — und **verweigert schriftlich**,
> wenn keins die Baseline schlägt. Die Quittung ist signiert und offline prüfbar."

**Nie wieder pitchen:** „MCP-Server für Anomalieerkennung" (PyOD), „AutoML für Edge"
(SensiML Piccolo, AGPLv3 seit Jun 2024), „wir signieren Modelle" (OpenSSF OMS).

---

## 2. Milestones — Reihenfolge = Abhängigkeit, jedes DoD = EIN Befehl, EIN PASS

### V1 — Receipt-Format v1 + Ed25519 *(aus R1, unverändert)*
Jede Quittung aus `distill`/`fit`/`monitor` wird versioniertes JSON: claim, verdict
(`INCLUDED`/`EXCLUDED`/`PASS`/`REFUSED`), Evidenz (nested-LOGO, perm-p, NULL, Kosten),
Input-Hashes, tool+version, Zeit, `countersignatures: []`.
Key in `~/.signalmap/signing_key` (0600, nie im Repo, nur pubkey im Receipt).
Reuse aus `indelible` — nicht neu erfinden.
**DoD:** `signalmap fit … && signalmap monitor …` erzeugt signiertes Receipt; 1 Byte
manipuliert → Verify FAIL, als Red-Team-Test im Suite-Gate.

### V2 — Standalone-Verifier *(aus R2 — der OSS-Wedge, jetzt der Hauptträger)*
`tools/verify_receipt.py` importiert **nichts** aus `signalmap` (HEDG3-`verify_shipped`-Muster),
nur stdlib + cryptography. Prüft Signatur, Schema, interne Konsistenz
(`REFUSED` ⇒ kein Deploy-Spec-Eintrag).
**DoD:** frisches venv **ohne** signalmap-Install verifiziert echtes Receipt → PASS,
manipuliertes → FAIL. Im CI.

### V3 — Composer/Qualify reviewen → public *(aus R4)*
`prove`/`qualify`/`composer` (Commit `501e274`, ~780 LOC + 221 LOC Tests) ist ungereviewt.
fresh-eyes-Review gegen Specs 2026-07-21/23, dann in den 0.5-Cut.
**Kein Public-Ship vor Review** — das ist eine stehende Regel, keine Vorsichtsformel.

### V4 — Verdict-Korpus = die Traction-Zeile *(aus R5, hochgestuft zur Headline)*
Bestehende Prereg-Verdicts als signierte Receipts nachziehen (rqa 1 IN/2 EX ·
coherence 1 IN/1 EX · envelope 0 IN/3 EX). Ehrlich labeln, ob re-run oder Archiv-Signatur.
**Traction-Zeile statt Feature-Liste:** „N Verdicts über M Banken, X ehrliche
Verweigerungen, 0 stille Übernahmen, offline verifizierbar."
Das ist der Teil, den ein Konkurrent nicht in einem Wochenende nachbaut.

### V5 — Produkt-Hygiene *(neu, aus der Session vom 30./31.07.)*
- **Suite-Boden senken.** Erledigt: `-n auto` → 5:05 → 1:47. Offen: die zwei
  CLI-Forwarding-Tests (~64–70 s) über eine session-scoped Bank-Fixture, Ziel < 45 s gesamt.
- **Restliche Coverage-Lücken:** `store.py` 68 %, `monitor.py` 75 %, `ingest.py`,
  `synth.py`, `ingestor.py`/`simulate_universal.py` (0 %, Demo-/Daemon-Pfade).
- **`qdrant_client.search`** ist in neueren Clients zugunsten `query_points` deprecated;
  aktuell schluckt ein `except` das still und meldet „keine Novelty". Prüfen und
  entweder migrieren oder die Degradation laut machen.

---

## 3. Explizit NICHT bauen (Kill-Liste, erweitert)

- **MCP-Server als Differenzierung — GESTRICHEN** (PyOD 3.6.2 `pyod mcp serve`,
  46 M Downloads). Später höchstens als *Distribution*, nie als Pitch, und erst wenn
  V1/V2/V4 stehen.
- GNN-/Graph-Achse (akademisch gesättigt, GPU, gegen den Lean-Moat).
- FastAPI-Receipt-Viewer vor V1–V2.
- Neue Feature-Familien ohne Prereg; keine neue Forschung in dieser Produktphase.
- Kein zweiter Anlauf auf „AutoML"-Framing.

---

## 4. Risiken, ehrlich

- **Adoption bleibt das echte Risiko** (0 Nutzer). PyPI ist jetzt gelöst, das war der
  größte mechanische Blocker — der verbleibende ist Vertrieb, nicht Code. V1–V4 klein halten.
- **Die überlebende Neuheit ist schmal:** „Signatur des *Verdikts* statt des *Artefakts*"
  ist ein Feature, kein Produkt. Trägt nur zusammen mit dem Korpus (V4).
- Signatur beweist **Herkunft und Integrität, nicht Korrektheit** der Statistik.
  Wording: „signed, reproducible, refusal-honest" — nie „proven".
- Lizenz-Hygiene: Piccolo (AGPLv3) nicht ansehen/kopieren.

---

## 5. Sanity-Kommandos

```bash
# Suite (parallel, wie CI)
python -m pytest -q -n auto            # 172 passed, 2 skipped, ~1:47

# Lint, exakt wie CI
ruff check signalmap/                  # ruff==0.16.0, Regeln explizit in pyproject

# Release: Tag + GitHub Release -> Actions publiziert via OIDC
#   Notfall/dry run:  gh workflow run release.yml --ref main
#   Echt-Upload:      gh workflow run release.yml --ref main -f publish=true
```

**Release-Hinweis für 0.5.0:** Tag `v0.4.0` zeigt auf `938d079` und ist **älter** als
`release.yml`. Ab 0.5.0 gilt: erst Version in `pyproject.toml` bumpen, committen, **dann**
taggen — der Workflow prüft, dass Tag und `pyproject`-Version übereinstimmen.
