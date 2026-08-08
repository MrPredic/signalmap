# NEXT SESSION — Forschung (Stand 19. Jul 2026)

Launch (L4 Push / PyPI / Announce) wartet auf User-OK und ist NICHT Teil dieser
Session. Diese Session = reine Forschung: neue Familie, Härtung, opportunistisch
Ep-52. Regeln: nur SignalMap · Prereg VOR Readout · gg. GROUND TRUTH testen ·
max 1 Sonnet-Subagent seriell · jede neue Behauptung 3–4 unabhängige Methoden,
ehrlicher Downgrade bei Divergenz · kein Public-Push ohne User-OK.

Sanity zuerst:
```bash
cd <local-path>/signalmap && source .venv-research/bin/activate
python research/factory/session_status.py      # Ep-52-Watch!
python -m pytest -q                             # erwartet 111
```

## PRIO 1 — 3. Premium-Familie: Envelope / Defektband-Spektrum
Warum: PdM-Standard (Hüllkurven-Ordnungsanalyse, Lager-Physik), billig
(O(n log n)), fehlt uns. Dreht die Premium-Story um: bisher „teuer nur wo es
zahlt" → dann „das Gate entscheidet, unabhängig vom Preis". **Erwartung EHRLICH
OFFEN** — auf CWRU kann die base-Grammatik sie verweigern, auch das = gutes
Ergebnis (die Quittung, die verweigert, gilt dann für eine BILLIGE Familie).

Bauen (TDD, REUSE `premium.py::PremiumFamily`-Interface wie rqa/coherence):
- Hilbert-Hüllkurve → FFT → Energie in Defektbändern (BPFO/BPFI/BSF/FTF, aus
  Drehzahl + Lagergeometrie; wo unbekannt: Bänder als relative Anteile am
  Hüllkurven-Spektrum). Self-contained numpy/scipy, KEINE neue Heavy-Dep.
- `needs_channels`-Semantik beachten (1-Kanal reicht, anders als coherence).
- Parität gegen eine Referenz-Impl pinnen (wie pyunicorn-Parity bei RQA), damit
  die Featurizer-Korrektheit nicht bei Null verifiziert wird.

Verdikt (Champion-Regel, paired-CI über LOGO-Folds, wie rqa/coherence):
- **Prereg VOR Readout** auf CWRU + MFPT + IMS (neue Prereg-Datei, Ledger-Freeze,
  Commit VOR Run = externer Timestamp). EXCLUDED ist ein gültiges Produkt-Ergebnis.
- Kosten-Quittung (ms/window) mitführen — der Punkt ist gerade, dass eine BILLIGE
  Familie trotzdem am Gate scheitern kann.
Erwartetes Ergebnis: 3. Familien-Datenpunkt (INCLUDED oder EXCLUDED), beides
verkaufbar. 1–2 Sessions inkl. Prereg.

## PRIO 2 — Ed25519-Signierung der Receipts (Härtung, Attestation-Vorstufe)
Muster aus indelible wiederverwenden (Ed25519, offline-verifizierbar). Receipt
+ spec.json signieren, `verify`-Pfad im Package. Voraussetzung für den späteren
Attestation-Service, aber schon ohne Service ein Doku-/Vertrauens-Argument.
Eine Session. REUSE indelible-Signatur-Code, nur Anbindung an FeatureSpec/Receipt
selbst bauen.

## PRIO 3 (opportunistisch) — Ep-52
Watcher läuft (ADVISORY/YELLOW, Re-Inflation gemeldet). BEI FEUER: dieselben 3
applies wie Ep-51, Preregs gelten unverändert:
- `PREREG_EP52.md` (psd_slope-Overflow-Precursor, BEIDE Stationen) anwenden
- apply51-Kontinuität (ORDERING auf n=2 = echtes Upgrade aus dem n=1-Demo)
Kosten ~null, höchster Wert pro Aufwand. KEIN Nach-Tuning, prereg-treu.

## NICHT tun
- GAS-id mit c256b8 nachverhandeln (bricht Ein-Config-Ehrlichkeit).
- Weitere Benchmark-Banken sammeln (Skala ist nicht unsere Position).
- Kohärenz auf alle Paare ausbauen (Beleg reicht).
- Launch-Artefakte anfassen ohne User-OK.

## Offene Kleinigkeiten (falls Zeit)
- Report-Klarheit: `passed`-Gate hängt am base-nested → FAIL+INCLUDED möglich;
  Receipt sollte das explizit erklären.
- `signalmap --version` ist kein echtes Flag (argparse druckt usage) — optionale
  `--version`-action.
