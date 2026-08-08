# PLAN v0.5 — „Signed Verdicts for Agents" (Receipts + MCP)

**Datum:** 2026-07-27 · **Status:** beschlossen nach Invent-Gate (PIVOT Positionierung, Kern bleibt)
**Vorbedingung:** v0.4.0 auf origin/main + Tag (erledigt in dieser Session, siehe git log)

## These / Moat
Piccolo AI (SensiML, AGPLv3, seit Jun 2024) besitzt „Edge-AutoML" → dieses Framing ist tot.
Leerer Quadrant: **(signierte, offline-verifizierbare Verdicts) × (Verweigerungs-Disziplin) ×
(Agent-Konsum via MCP)**. MCP-TAD (Research Square 2026) = nächster Nachbar: akademisch, routet
Detektoren, aber KEINE Signatur, KEIN Refusal, kein shipped Tool.
Moat = Verdict-KORPUS (prereg'd INCLUDED/EXCLUDED über Banken) + Refusal-Disziplin + signierte
Provenienz — Feature ist nachbaubar, der Korpus nicht. Doktrin: Erst Korpus, dann Kunden
(Standard-Traction-Zeile: „N Verdicts über M Banken, X ehrliche Verweigerungen, 0 stille
Übernahmen, offline verifizierbar").

## Milestones (Reihenfolge = Abhängigkeit; jedes DoD = EIN Befehl, EIN PASS)

### R1 — Receipt-Format v1 + Ed25519-Signatur
- Reuse: indelible (Ed25519, canonical JSON, offline-verify) — NICHT neu erfinden.
- Jede Quittung aus `distill`/`prove`/`fit`/`monitor` wird ein versioniertes JSON-Receipt:
  claim, verdict (INCLUDED/EXCLUDED/PASS/REFUSED), Evidenz (nested-LOGO, perm-p, NULL,
  Kosten), Input-Hashes, tool+version, Zeit; Feld `countersignatures: []` (Attestation-
  Service-Andockpunkt, jetzt leer).
- Key: `~/.signalmap/signing_key` (0600, NIE im Repo/Receipt; nur pubkey im Receipt).
- DoD: `signalmap fit … && signalmap monitor …` erzeugt signiertes Receipt; Tampering
  (1 Byte) → Verify-FAIL (Red-Team-Test im Suite-Gate).

### R2 — Standalone-Verifier (der OSS-Wedge)
- `tools/verify_receipt.py`: importiert NICHTS aus signalmap (HEDG3-`verify_shipped`-Muster),
  nur stdlib + pynacl/cryptography. Prüft Signatur + Schema + interne Konsistenz
  (z.B. verdict REFUSED ⇒ kein deploy-spec-Eintrag).
- DoD: frisches venv ohne signalmap-Install verifiziert ein echtes Receipt → PASS;
  manipuliertes → FAIL. Test im CI.

### R3 — MCP-Server (Agent-Oberfläche)
- stdio-only, lokal (kein Netz). Tools: `qualify`, `fit`, `monitor`, `prove` — Antwort =
  Verdict + Receipt-Pfad + Kurzbegründung; Verweigerung ist eine ERFOLGS-Antwort mit
  verdict=REFUSED, nie Exception.
- Härtung ab Tag 1 (Lektionen HEDG3/mcp-shield): bounded `readline` (Memory-DoS),
  Input-Schema-Validierung, keine Pfad-Traversal über Tool-Args, Timeouts pro Tool-Call.
- DoD: Claude-Code-Session ruft via MCP `monitor` auf CWRU-Bank → signiertes Verdict;
  `verify_receipt.py` (R2) bestätigt es ohne signalmap-Import.

### R4 — Composer/Qualify-Schicht reviewen → public
- prove/qualify/composer (committed 501e274, ~780 LOC + 221 LOC Tests, 149-Suite grün)
  ist das Substrat für R3, aber UNREVIEWED: fresh-eyes-Review gg. Specs 2026-07-21/23,
  danach in den 0.5-Cut. Kein Public-Ship vor Review.

### R5 — Korpus + Announce 0.5
- Verdict-Korpus konsolidieren: bestehende Prereg-Verdicts (rqa: 1 IN/2 EX · coherence:
  1 IN/1 EX · envelope: 0 IN/3 EX) als signierte Receipts nachziehen (re-run oder
  Archiv-Signatur, ehrlich labeln welches von beiden).
- README-Dreh auf „receipt-gated signal verdicts, agent-ready"; PyPI (User-Token);
  Announce mit Korpus-Zeile statt Feature-Liste.

## Explizit NICHT bauen (Kill-Liste)
- GNN/Graph-Achse (akademisch gesättigt: GDN/MST-GAT/TopoGDN; GPU; gegen Lean-Moat).
  Cross-Channel-Graph nur später als prereg'd Premium-Familie über existierende Kohärenz.
- FastAPI-Receipt-Viewer: nice-to-have via 0.138 Single-Deployment, NACH R1–R3, nur bei Bedarf.
- Neue Feature-Familien ohne Prereg; keine neue Forschung in dieser Produktphase.

## Risiken / Ehrlichkeit
- Adoption bleibt das echte Risiko (0 Nutzer) — R1–R3 sind klein halten (~2–3 Sessions),
  danach Vertrieb/Distribution, nicht mehr Code.
- Signatur beweist HERKUNFT+INTEGRITÄT, nicht Korrektheit der Statistik — Wording im
  Receipt/README entsprechend („signed, reproducible, refusal-honest", nie „proven").
- AGPLv3-Nachbar Piccolo: nicht kopieren/anschauen für Code (Lizenz-Hygiene).

## Sanity-Kommandos
- Suite: `source .venv-research/bin/activate && python -m pytest -q` (Stand heute: 149)
- Public-Cut-Muster: Worktree von origin/main + `git checkout <commit> -- <whitelist>`
  (diese Session, funktioniert; Whitelist = origin/main-Top-Level + CHANGELOG.md + docs/ARCHITECTURE.md)
