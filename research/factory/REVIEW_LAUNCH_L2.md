# L2 Launch-Review — Ground-Truth-Pass (18. Jul 2026)

Ein lokaler Sonnet-Subagent hat den Launch-Kandidaten (signalmap/ v0.4.0) gegen
selbstgebaute Ground-Truth-Fälle getestet (nicht Code gelesen — die Regel, die
im HEDG3-Review jeden echten Bug fand). Scope: distill.py, premium.py,
monitor.py, cli.py, multichannel.py.

## Kernergebnis: Champion-Regel (der USP) = sauberer Freibrief
Härtest getestet, kein Bug:
- Geplantes Signal, das NUR Kohärenz sieht (ch0 klassengleich, ch1 gekoppelt) →
  base nested 0.484 (blind), aug 1.0, CI-lo +0.281 → korrekt INCLUDED.
- Base löst schon (Frequenzunterschied in ch0) → delta 0.0, CI [0,0] → korrekt
  EXCLUDED (Grenzfall lo=0 nicht >0, kein Off-by-one).
- Reines Rauschen, 6 Seeds → 0/6 falsche Inklusionen.
- `_paired_ci` mit analytisch bekannten Diffs → keine Vorzeichen-Inversion.
Ebenfalls verifiziert korrekt: Kapazitäts-Gate (deterministisch, cheapest-first,
monoton), NULL-Selbsttest (1.0→0.359≈chance), Multichannel-Identität
(ch0 bit-identisch 1D/2D), needs_channels-Verweigerung, featurize fail-closed,
coherence in [0,1], fit/monitor (0% healthy / 100% fault, kein Leak),
Determinismus (spec.json exakt reproduzierbar).

## Findings
### 1. CRITICAL — GEFIXT (Commit siehe unten)
`_read_text_signal`: kaputte Zeilen VOR dem ersten geparsten Wert wurden
bedingungslos als „Header" behandelt → die fail-closed-5%-Garantie hing an der
Zeilen-Reihenfolge statt am echten Bad-Anteil (Repro: 60 leere + 40 gute Zeilen,
Spalte 60% leer, lud still 40 Werte ohne Warnung). Fix: Header = keine Zahl in
IRGENDEINER Spalte UND vor Datenbeginn; eine Zeile mit Zahl anderswo aber leerer
Zielspalte zählt als skipped. Test `test_leading_missing_rows_still_fail_closed`,
alle 9 Ingest-Tests grün.

### 2. MINOR — GEFIXT (18. Jul, TDD)
Bank mit n_recordings=1 → roher sklearn-`ValueError` (LeaveOneGroupOut braucht
≥2 Gruppen). Laut, kein falsches Verdikt, aber unschön für einen OSS-Erstkontakt.
Fix: früher Guard in `distill()` (NICHT in load_bank — fit_spec_backend darf 1
Recording laden) → `SystemExit` mit klarer „>=2 recordings"-Botschaft. Test
`test_distill_needs_at_least_two_recordings`.

### 3. MINOR — GEFIXT (18. Jul, TDD)
2D-.npy-Achsen-Heuristik (`argmin(shape)`) invertiert still, wenn Kanäle >
Samples. Fix: `_read_mc_recording` warnt (UserWarning) bei quasi-quadratischem
Array (lo/hi > 0.5) mit Hinweis auf `--channel-axis`; klar rechteckige
Recordings und ein explizites channel_axis bleiben still. Tests
`test_quasi_square_npy_warns_on_ambiguous_axis`, `test_rectangular_npy_does_not_warn`.

Scratch-Probes des Reviewers lagen unter scratchpad/sm/probe_*.py.

## L3 — Fresh-Checkout-Smoke (18. Jul)
Sdist gebaut (kein research/.remember/superpowers/Privatpfad-Leak, 49 Files),
frisches venv, `pip install signalmap[distill]` aus sdist, README-`distill`-
Kommando wörtlich. Zwei Funde, beide gefixt:

### 4. CRITICAL (UX) — GEFIXT (TDD)
README-Quickstart `--out artifacts/spec.json` crashte am Ende mit
`FileNotFoundError`, weil ein frischer Checkout kein artifacts/ hat — die
komplette Gauntlet-Arbeit ging verloren. Fix: `_ensure_parent()` in beiden
`save()`-Methoden (FeatureSpec + DistilledDetector) legt das Zielverzeichnis an.
Test `test_spec_save_creates_missing_parent_dir`. Nach Fix: Fresh-venv-Smoke
grün, spec.json + spec_report.md erzeugt (nested_acc 1.0).

### 5. MINOR — GEFIXT (README)
distill-Quickstart hatte keine Install-Zeile → frischer `pip install signalmap`
zieht torch (core dep) aber NICHT scipy/sklearn → ImportError bei `distill`.
Fix: `pip install signalmap[distill]` vor dem distill-Kommando ergänzt.

### 6. MINOR — NOTIERT (nicht gefixt)
`signalmap --version` ist kein echtes Flag (argparse druckt usage). Kein
README-Kommando, niedrige Prio; optional ein `--version` action ergänzen.

Suite 104 → 108 grün. Externes Doku-Cross-Review (ChatGPT/Gemini, nur
public-bound Files) = offener User-Schritt vor L4.

## L3 — Voller README-Verbatim-Replay (18. Jul, frisches venv aus sdist[all])
ALLE 10 Quickstart-Kommandos wörtlich im frischen Checkout: plugins, benchmark
(AUC 1.0), universal, train (--synthetic 2000 --epochs 30), run, discover
(--naive/--confound temp, Verdikt matcht README: vibration–em→confound,
heat–acoustic überlebt), distill, fit, monitor (Fault 100%/Healthy 0% alerting).

### 7. CRITICAL (Launch-Blocker) — GEFIXT (TDD)
data/ und artifacts/ sind gitignored → frischer `git clone` hat sie nicht →
benchmark/universal/train crashten mit FileNotFoundError bzw. torch
„Parent directory artifacts does not exist". Gleicher Footgun wie #4, aber in
den benchmark/universal/train/ingest/sink/map-Schreibpfaden. Fix: geteilter
Helper `signalmap/_io.py::ensure_parent`, aufgerufen an ALLEN Schreibstellen
(synth ×2, train, simulate_universal ×2, sinks parquet, ingest, detector,
visualize). Test `test_fresh_checkout_io.py` (3 Tests inkl. build_pdm_benchmark
in fehlendes Verzeichnis). Nach Fix: alle 10 Kommandos exit=0.

### 8. CRITICAL (Launch-Blocker) — GEFIXT
CI (`.github/workflows/ci.yml`) installierte nur `pytest`, nicht scipy/sklearn
→ `pytest -q` wäre auf der distill/premium-Suite rot (reproduziert:
ModuleNotFoundError scipy). Fix: `pip install -e .[dev]` ([dev]=pytest+scipy+
scikit-learn).

Suite 108 → 111 grün. Doku-Self-Review sauber (Links, Zahlen matchen Preregs,
Honesty-Guards intakt, .[all]-Quickstart kohärent). OFFEN (User-Schritte):
externes Doku-Cross-Review (ChatGPT/Gemini, public-bound), L4-Push, PyPI-Token.

## L3 — Externes Doku-Cross-Review (19. Jul, ChatGPT/Gemini durch User)
Paket = README+CHANGELOG+ARCHITECTURE+CONTRIBUTING+examples/README (public-bound).
Alle Blocker + High-Prio gefixt (doc-only, kein Runtime-Code → 111 Tests unberührt):

BLOCKER
- #B1 Refusal-Zahl: CHANGELOG sagte „four refusals", Ground Truth (5 Premium-
  Report-Files) = 2 INCLUDED (CWRU-RQA, HYD-coherence) / 3 EXCLUDED (CALCE-RQA,
  HYD-RQA, GAS-coherence). README war korrekt (3). Fix: CHANGELOG → „three";
  STRATEGY-Doku „vier"→„drei".
- #B2 zsh-Glob: `pip install -e .[all]` / `signalmap[distill]` failen in zsh
  (`no matches found`, verifiziert). Fix: alle public-Docs auf `python3 -m pip
  install '...'` mit Quotes (README ×3, CONTRIBUTING, examples).
- #B3 „No labels" irreführend (direkt danach --healthy-label): umformuliert zu
  „No fault labels needed — you supply/mark healthy data".
- #B4 fit/monitor-Install unvollständig: Parquet-Beispiele brauchen pyarrow →
  Install-Zeile `'signalmap[all]'` am Parquet-fit-Block + Klarstellung
  [distill]=nur .npy/.csv-Banken.

HIGH-PRIO
- „any sensor" > Stand: Rust-Capture-Adapter laut Roadmap offen (nur sim/replay/
  mic/mqtt). Umformuliert zu „arbitrary recorded/raw signals + extensible adapter
  model"; Roadmap „(any sensor)"→„(any recorded signal)".
- „raw/no scaling" vs ingest: `_to_int16` macht DC-center + globalen Gain
  (Docstring: „Documented bias"). Konvertierung im Design-Principle dokumentiert
  (spektrum-erhaltend, kein per-window-scaling).
- Status inkonsistent (alpha vs launch candidate): Badge → `0.4.0-rc`.
- CWRU-Repro nicht self-contained: examples-Install auf `'signalmap[all]'`;
  fetch_cwru.py Mirror auf Commit-SHA gepinnt (19f5bd67…, beide .mat @ HTTP 200
  verifiziert) statt moving `master`; Original-Quelle (case.edu) genannt.

NICE-TO-HAVE
- Differentiator-Einzeiler direkt nach Tagline (distill=gate+receipt+refuse).
- MIMII: voller https-Link + Konvertierungshinweis.

OFFEN: L4-Push (User-OK), PyPI-Token. Test-Suite 111 (doc-only Änderungen).
