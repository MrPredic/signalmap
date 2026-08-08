# Coherence + RQA Screen

Two new screening families over the existing bank set, same paired-LOGO-CI contract as readout_screen.py: lean vs lean+RQA (all banks) and lean vs lean+coherence-summary (genuinely multi-channel banks only: GAS-id 8ch, HYD-cooler/valve/accumulator 6ch). A flag does not change any claim -- it points to a source-level rebuild candidate for bank_audit + gauntlet.

RQA config: dim=3, tau=5, recurrence_rate=0.1 (rqa_fair.py's cheapest grid point, fixed not searched -- screen is a cheap proxy, not a maximized method thesis). Point cap MAX_PTS=2000/window (no-op today: every bank uses W=1024).

| Bank | Family | n_win | n_rec | chance | lean | lean+aug | diff | CI | Flag |
|---|---|---|---|---|---|---|---|---|---|
| ECN | rqa | 196 | 14 | 0.167 | 0.592 | 0.653 | +0.061 | [-0.015,+0.138] | - |
| CWRU | rqa | 2849 | 24 | 0.167 | 0.872 | 0.997 | +0.125 | [+0.052,+0.225] | strong |
| MFPT | rqa | 3718 | 20 | 0.333 | 0.780 | 0.899 | +0.119 | [+0.041,+0.191] | strong |
| CALCE-soh | rqa | 306 | 18 | 0.333 | 0.579 | 0.859 | +0.280 | [+0.092,+0.486] | strong |
| HYD-cooler | rqa | 90 | 18 | 0.333 | 0.533 | 0.733 | +0.200 | [+0.078,+0.322] | strong |
| HYD-valve | rqa | 120 | 24 | 0.250 | 0.192 | 0.167 | -0.025 | [-0.117,+0.067] | - |
| HYD-accumulator | rqa | 120 | 24 | 0.250 | 0.275 | 0.183 | -0.092 | [-0.175,-0.008] | - |
| DCASE-valve-id | rqa | 120 | 24 | 0.250 | 0.325 | 0.308 | -0.017 | [-0.117,+0.083] | - |
| DCASE-valve-anomaly | rqa | 80 | 16 | 0.500 | 0.238 | 0.250 | +0.012 | [-0.125,+0.150] | - |
| DCASE-pump-id | rqa | 120 | 24 | 0.250 | 0.292 | 0.383 | +0.092 | [-0.042,+0.217] | - |
| DCASE-pump-anomaly | rqa | 80 | 16 | 0.500 | 0.362 | 0.488 | +0.125 | [-0.025,+0.275] | - |
| IMS-fault | rqa | 192 | 12 | 0.500 | 0.609 | 0.677 | +0.068 | [-0.010,+0.146] | - |
| SEIS-depth | rqa | 128 | 16 | 0.500 | 0.688 | 0.773 | +0.086 | [-0.039,+0.219] | - |
| GEOMAG-storm | rqa | 128 | 16 | 0.500 | 0.938 | 0.898 | -0.039 | [-0.094,+0.008] | - |
| GRIDFREQ-area | rqa | 192 | 24 | 0.167 | 0.536 | 0.625 | +0.089 | [-0.031,+0.203] | - |
| GAS-id | rqa | 240 | 20 | 0.250 | 0.121 | 0.071 | -0.050 | [-0.092,-0.008] | - |
| VOLCANO-eruption | rqa | 120 | 15 | 0.500 | 0.767 | 0.617 | -0.150 | [-0.342,-0.008] | - |
| HYD-cooler | coherence | 72 | 18 | 0.333 | 0.625 | 0.778 | +0.153 | [+0.069,+0.222] | strong |
| HYD-valve | coherence | 96 | 24 | 0.250 | 0.198 | 0.219 | +0.021 | [-0.042,+0.083] | - |
| HYD-accumulator | coherence | 96 | 24 | 0.250 | 0.219 | 0.177 | -0.042 | [-0.125,+0.042] | - |
| GAS-id | coherence | 159 | 20 | 0.250 | 0.300 | 0.447 | +0.147 | [+0.047,+0.256] | strong |

21 bank x family cells screened, 6 flagged.

## Flag-Verifikation (12. Jul) — fair RQA (rqa_fair.py, Ledger 851c8338)

Beide Top-Fährten source-rebuilt (exakt dieselben Loader-Calls wie der Screen) und mit der FAIREN
RQA-Familie re-getestet (pyunicorn, volle 1024, 6 Maße, 4-Config-Grid best-wins — Selektion
begünstigt registriert RQA):

| Bank | lean | RQA best (cfg) | gepaart RQA−lean | Verdikt | Kosten |
|---|---|---|---|---|---|
| CALCE-soh | 0.579 | **0.860** (m5t10) | **+0.281 CI [+0.072,+0.502]** | Flag CONFIRMED, CI-fest | lean 180× billiger |
| HYD-cooler | 0.533 | **0.689** (m3t10) | **+0.156 CI [+0.011,+0.300]** | Flag CONFIRMED, CI-fest | lean 191× billiger |

Beide klar über Chance 0.333. Damit erweitert sich die ehrliche Frontier-Tabelle: RQA CI-fest
besser auf CWRU (0.961, Jul 3) + CALCE-soh + HYD-cooler; TIE auf ECN/MFPT; RQA SCHADET auf
VOLCANO/HYD-accumulator/GAS-id (Screen) → familienspezifisch, kein „RQA überall". Kosten-Moat
(~180–260×, O(n²) inhärent) hält überall. Ehrliche Grenzen: Grid-best-wins = mild optimistisch
für RQA (bewusst, konservativ für den Kosten-Claim); kein perm-p in diesem Pfad (identisch zur
A4-Methodik der bestehenden Frontier-Zeilen); HYD-cooler-Kohärenz-Flag (+0.153) noch UN-verifiziert
(eigene Familie, nächster Schritt); CWRU/MFPT-RQA-Zeilen waren schon fair getestet (A4), Screen
stimmt mit ihnen überein (CWRU strong, MFPT-Screen-Flag vs A4-TIE = cheap-config-Artefakt,
fair-Wert gilt). GAS-id × Kohärenz (+0.147) = offene Fährte.

## Flag-Verifikation (13. Jul) — faire Kohärenz (coherence_fair.py, Prereg 4bb76422, Ledger 72fa75d6)

Die beiden offenen Kohärenz-Flags source-rebuilt (exakt dieselben Loader-Calls wie der Screen) und
mit einer FAIREN Kohärenz-Familie re-getestet. Der Screen-Proxy hatte drei Handicaps: Paar-Struktur
auf 3 Summary-Stats kollabiert, Frequenz-Struktur auf 1 Breitband-Mittel kollabiert, nur nperseg=64.
Fair = per-Paar × per-Band-MSC, Grid (nperseg, n_bands) ∈ {(64,1),(128,2),(256,4),(256,8)} best-wins
(Selektion begünstigt Kohärenz). Entscheidungsregel VOR Readout eingefroren (Ledger COH-FAIR-PREREG):
PRIMARY = aug (lean+coh) vs lean — der geflaggte Vergleich; SECONDARY = alone vs lean (Frontier-Semantik).

| Bank | lean | aug best (cfg) | gepaart aug−lean | alone best (cfg) | Stabilität | Verdikt | Kosten |
|---|---|---|---|---|---|---|---|
| HYD-cooler (6ch) | 0.625 | **0.944** (c128b2) | **+0.319 CI [+0.194,+0.444]** | **0.972** (c256b8) | 1.00 | Flag CONFIRMED, CI-fest | lean ~56× billiger |
| GAS-id (8ch) | 0.300 | **0.516** (c256b8) | **+0.216 CI [+0.106,+0.335]** | 0.485 (c256b8) | 1.00 | Flag CONFIRMED, CI-fest | lean ~71× billiger |

Chance-Gates klar (HYD acc-CI-lo 0.889 > 0.333; GAS 0.460 > 0.250), Config-Jackknife-Stabilität
beide 1.00. Der Screen hat beide Flags massiv UNTERschätzt (HYD 0.778→0.944/0.972, GAS 0.447→0.516):
die Information liegt in Paar- UND Band-Struktur, genau das, was der Proxy wegkollabierte.
Bemerkenswert: GAS-id — das alte Doppel-Null — kommt mit einer FIXEN 224-Feature-Kohärenz-Familie
(ohne Forge-Suche) CI-fest über lean; HYD-cooler alone 0.972 heißt, der Zustand steht fast
vollständig im Cross-Sensor-Muster. Ehrliche Grenzen: Grid-best-wins mild optimistisch für Kohärenz;
kein perm-p (identisch A4/rqa_fair-Methodik); Same-Rig-Adjacency-Caveat der HYD-Bank gilt weiter;
GAS n=159 Fenster. Hinweis: LEDGER enthält COH-FAIR doppelt (Report 2× gelaufen, identisches
Ergebnis, append-only — 9b91ce22 + 72fa75d6).
