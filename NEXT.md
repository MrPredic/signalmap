# NEXT — signalmap — 2026-08-05
STATE: **0.5.2 ist auf PyPI** (`pip install signalmap`), origin/main `51d81b4`, research `8a8214c` = private/research, CI grün. Der komplette README-Fluss ist aus einem frischen venv gegen die veröffentlichten Wheels durchgespielt: `corpus --out` → curl-Verifier → PASS → Tamper → FAIL.
VERIFY: .venv/bin/python -m pytest -q -n auto → 280 passed, 2 skipped, ~70 s (2026-08-05); .venv/bin/ruff check signalmap/ tools/ → All checks passed
DREI DEFEKTE, DIE NUR DAS AUSPROBIEREN GEFUNDEN HAT (Lehre: nach jedem Release den README-Fluss aus einem leeren venv fahren, nicht die Metadaten lesen):
- 0.5.0 `corpus` → FileNotFoundError auf pip-Install (Reports gibt es nur im Clone) → 0.5.1 shippt die 8 Receipts als Package-Data (`signalmap/verdicts/`), Rebuild im Clone spiegelt dorthin, Test bricht bei Drift.
- 0.5.1 `distill`/`fit`/`monitor` → ModuleNotFoundError: `cryptography` war ein Extra, obwohl JEDER Verdict signiert wird → 0.5.2 macht es zur Kern-Dependency + Test.
- CI-Gate `verify-receipt-standalone` lief im Checkout-Root und hat nie verifiziert (cwd-Import) → läuft jetzt außerhalb.
PUBLIC/PRIVATE-GRENZE (8. Aug 2026 aufgehoben): der komplette Arbeitsstand ist jetzt oeffentlich — Code, research/factory, Plaene, Handovers, Ergebnisse. Lokal bleiben nur `.remember/` (Assistenz-Sitzungsstand), `research/factory/logs/cache/` und `*.err`-Dumps sowie die ~11 GB Rohdaten unter `data/` (oeffentlich beziehbar via `research/factory/fetch_remaining_domains.sh`, sha256 in `study/manifests/`). Vor jedem Push pruefen: `grep -rnEI 'BEGIN .*PRIVATE KEY|@gmail|/Users/|Co-Authored-By|Claude-Session' --exclude-dir=.git --exclude-dir=.venv --exclude-dir=data .` muss leer sein.
TODO:
1. Adoption messen statt bauen (Kill-Scan 31.7.: Adoption ist das Risiko, nicht die Fähigkeit). Konkret: PyPI-Downloads 0.5.x über 2 Wochen, GitHub-Traffic, und EIN gezielter Ort, an dem die Receipt-Story hingehört (Issue/Diskussion in einem Sensor-ML-Repo, kein Broadcast).
2. Wenn Adoption sich rührt: `signalmap verify` als CLI-Unterbefehl (Bequemlichkeit) — die Trust-Story bleibt der standalone `tools/verify_receipt.py`, der nichts von uns importiert.
3. Härtere Datensätze (MIMII/MAFAULDA) sind der einzige offene Fähigkeits-Punkt in der Roadmap — erst nach 1.
DO-NOT: MCP-Achse tot (Kill-Scan 31.7.). Mechanism-Null-Vokabular NICHT um „bessere" Nulls erweitern ohne Messung (12/216: `null_informative=false` ist ehrlich, kein Bug). `.gitignore` auf research NICHT mit dem `research/*`-Filterblock von main angleichen — der gehört nur auf den öffentlichen Branch.
