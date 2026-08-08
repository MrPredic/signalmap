# PREREG — Richtungs-Identifizierbarkeit label-freier Anomalie-Scores (6. Aug 2026)

**FROZEN BEFORE READOUT.** Ab diesem Commit wird an Hypothesen, Domänen-Liste,
Spec, Aggregation, Null-Kontrollen und Schwellen nichts mehr geändert. Jede
spätere Abweichung wird als datierter AMENDMENT-Block angehängt, mit der
ausdrücklichen Angabe, ob zum Zeitpunkt der Abweichung bereits eine Zahl der
PRIMARY-Auswertung gesehen wurde.

## Auslöser (eine bereits gemessene Zahl, keine Vermutung)
`research/factory/logs/dcase_valve_readout_fallback_report.md` (21. Jul 2026,
Prereg `PREREG_DCASE_VALVE_EXTERNAL.md`) berichtet für MIMII/DCASE2020-Task2
valve id_00:

    AUC = 0.2642, 95% CI [0.1929, 0.3360], n=204 held-out Clips

Das CI liegt **vollständig unter 0.5**. Das ist nicht „keine Trennung" (das
wäre CI über 0.5), sondern **Inversion**: der Detektor rankt Anomalien
systematisch als *normaler* als Normal. Gespiegelt (1 − AUC) wären es 0.736.
Der Betrag trägt Information, das Vorzeichen zeigt in die falsche Richtung.

Der Score wird ausschließlich auf **gesunden** Daten kalibriert
(`DistilledDetector.fit(spec, healthy_windows, envelope=3.0)`; Score =
max |z| über die Spec-Features, z gegen Healthy-Mittelwert/Streuung). Ein
solcher Fit sieht per Konstruktion **nie** eine Anomalie. Damit ist die
Annahme „Anomalien liegen WEITER vom Healthy-Zentrum" eine unbelegte
Zusatzannahme, keine aus den Daten gelernte Eigenschaft.

## H1 — PRIMARY (Identifizierbarkeit)
> Das **Vorzeichen** der Abweichung von AUC = 0.5 ist bei healthy-only-
> Kalibrierung domänenabhängig und nicht aus den Trainingsdaten bestimmbar.

Entscheidungsregel, vor jedem Lauf fixiert: H1 gilt als **bestätigt**, wenn es
unter den ausgewerteten Domänen **mindestens eine mit CI-Obergrenze < 0.5**
UND **mindestens eine mit CI-Untergrenze > 0.5** gibt (95%-Bootstrap,
2000 Resamples über Recordings, seed=0). Eine einzige invertierte Domäne
(valve, bereits vorliegend) reicht **nicht** — es braucht beide Vorzeichen
CI-fest im selben Lauf mit identischem Rezept.

H1 gilt als **widerlegt**, wenn kein CI vollständig unter 0.5 liegt (dann war
valve ein Einzelbefund oder ein Bug, und dieser Bug wird gesucht und
berichtet).

## H2 — SECONDARY (Betrag trägt trotzdem Information)
> Richtungsfrei gemessen trennt der Score: AUC\* = max(AUC, 1 − AUC) liegt
> CI-fest über dem Shuffle-Null-Wert derselben Statistik.

Wichtig und vorab deklariert: **max(·) ist unter der Nullhypothese nach oben
verzerrt** (E[AUC\*] > 0.5 auch bei reinem Rauschen). Deshalb wird AUC\* NICHT
gegen 0.5 getestet, sondern gegen das **empirische 95%-Perzentil der
Shuffle-Verteilung derselben Domäne** (N1 unten). Nur `AUC*_beobachtet >
q95(AUC*_shuffle)` zählt.

## H3 — PRODUKT-KONSEQUENZ (die eigentlich verkaufte Größe)
> Das Produkt liefert keinen AUC, sondern einen **Alarm** an einer selbst
> kalibrierten Schwelle. Wo das Vorzeichen invertiert ist, ist die Alarmrate
> auf Anomalien **niedriger** als auf Normal — der ausgelieferte Detektor ist
> dort nicht nur nutzlos, sondern anti-korreliert.

Gemessen als `alarm_rate(anomaly) − alarm_rate(normal)` an der vom Produkt
selbst gesetzten Schwelle (99. Healthy-Perzentil × envelope=3.0), mit
Bootstrap-CI. Vorab deklarierte Konsequenz, falls H1 und H3 bestätigt sind:
Der ehrliche Verdict für eine healthy-only kalibrierte Domäne **ohne
gelabelten Anker** ist **REFUSED**, nicht ein AUC. Diese Konsequenz wird als
Verdict-Receipt ausgeliefert, unabhängig davon, wie die Zahl ausfällt.

## Methoden-Triangulation (4 Wege, orthogonale Fehlermodi, ≥1 theoriegetrieben)
- **M1 rangbasiert** — AUC + Perzentil-Bootstrap-CI über Recordings. Fehlermodus:
  Rangstatistik, schwellenunabhängig.
- **M2 theoriegetrieben** — formale Aussage + **konstruktiver Gegenbeweis**: eine
  synthetische Domäne, in der die Anomalie *geringere* Dispersion hat als
  Healthy (z. B. blockiertes Ventil → weniger Varianz). Für sie ist Inversion
  analytisch vorhergesagt, BEVOR sie gemessen wird. Hängt an keiner realen
  Datenquelle.
- **M3 schwellenbasiert** — Alarmraten-Differenz (H3). Fehlermodus orthogonal zu
  M1: eine Domäne kann rangmäßig trennen und an der Schwelle trotzdem
  versagen (und umgekehrt).
- **M4 Transfer** — Vorzeichen aus Domäne A auf Domäne B angewandt: sagt das in A
  gemessene Vorzeichen das Vorzeichen in B voraus? Trefferquote über alle
  geordneten Paare. Direkter Test der Nicht-Übertragbarkeit.

Divergenz der Wege → **ehrlicher Downgrade** im Report, kein Rosinenpicken.

## Null-Kontrollen (Pflicht, alle Domänen)
- **N1 Label-Shuffle** (2000 Permutationen, seed=0): AUC-CI muss 0.5 überdecken.
  Liefert zugleich die Null-Verteilung für AUC\* (H2).
- **N2 Healthy-vs-Healthy**: die Normal-Recordings der Eval-Menge werden nach
  sortiertem Dateinamen in zwei Hälften geteilt, die zweite Hälfte wird als
  „Anomalie" etikettiert. Erwartung: CI überdeckt 0.5. Fängt Reihenfolge-,
  Datei-Präfix- und Aufnahme-Drift-Artefakte, die N1 nicht fängt.
- **N3 Leakage-Check**: kein Eval-Recording darf in der Fit-Menge vorkommen.
  Geprüft über sha256 der Fenster-Arrays, nicht über Dateinamen. Ein Treffer
  invalidiert die Domäne (Bericht: EXCLUDED wegen Leckage).

## Domänen (Liste hier eingefroren; Nachträge nur per AMENDMENT)
Jede Domäne liefert (a) NUR-gesunde Fit-Fenster, (b) eine davon **disjunkte**
Eval-Menge aus gelabelten Recordings (binär), (c) ein Manifest mit sha256.

Lokal vorhanden (kein Download):
1. `mimii_valve_id00` — Referenz-Domäne des Auslösers (Bank existiert).
2. `cwru` — Wälzlager, Vibration (`data/cwru_real.parquet`).
3. `ecn` — elektrochemisches Rauschen, Korrosion (`data/distill_banks/ecn`).
4. `geomag` — geomagnetisch, quiet = healthy vs storm = anomaly.
5. `battery_eis` — Impedanz-Spektroskopie / Alterung.
6. `volcano` — Kilauea, seismisch, Precursor als Anomalie.
7. `synth_pos` — synthetische Positivkontrolle: Anomalie = **höhere** Dispersion.
   Analytisch vorhergesagt AUC > 0.5. Kontrolliert, dass die Pipeline
   überhaupt in die *richtige* Richtung zeigen kann.
8. `synth_neg` — konstruktiver Gegenbeweis (M2): Anomalie = **niedrigere**
   Dispersion, sonst identisch. Analytisch vorhergesagt AUC < 0.5.

Download (je mit hartem Größen-Deckel, siehe Abbruchregeln):
9. `mimii_fan_id00`, 10. `mimii_pump_id00`, 11. `mimii_slider_id00` — gleiche
    Quelle wie valve, andere Maschinenphysik.
12. `paderborn_kat` — Wälzlager, Motorstrom + Vibration, andere Modalität.
13. `mafaulda` — Rotor-Prüfstand, Unwucht/Fluchtungsfehler.

**Domänen, die nicht materialisieren** (Download tot, Lizenz, Deckel
gerissen, Formatbruch), werden als **not-obtained** berichtet. Es wird
**keine Ersatz-Domäne nachgeschoben**, um eine Zahl zu retten.

## Fixe Parameter (keine Suche, keine Domänen-Anpassung)
- **Spec, identisch für ALLE Domänen** — der deterministische Lean-Base-Satz
  `[p.name for p in gate(enumerate_programs(), n_recordings=30, C=50)][:9]`:
  `acf1(id(id(x)))`, `crest(id(id(x)))`, `lcross(id(id(x)))`,
  `meanabs(id(id(x)))`, `peakcv(id(id(x)))`, `runcv(id(id(x)))`,
  `runmean(id(id(x)))`, `std(id(id(x)))`, `zcr(id(id(x)))`;
  `premium=[]`. sha256(repr) = `0c98e6c7a7d72be8…`. Präzedenz: identisch mit
  dem Spec des valve-Fallback-Reports, damit die Auslöser-Zahl 1:1
  vergleichbar bleibt. **Kein per-Domäne-distill** — sonst wäre die
  Spec-Auswahl ein Confounder für das Vorzeichen.
- Fensterung W=1024, detrend + z-Norm (Produkt-Default `distill.py::window`).
- `DistilledDetector.fit(..., envelope=3.0)`, Schwelle selbstkalibriert.
- Score je Fenster = `det.score(w)` (max |z| über Spec-Features).
- Aggregation je Recording = **Mittelwert** über dessen Fenster (fix, kein
  Max/Median-Vergleich nach Sicht der Zahlen — Präzedenz Klausel 6 valve).
- K = 20 Fenster je Recording, nicht überlappend, vom Anfang des Signals.
- Bootstrap: 2000 Resamples **über Recordings**, seed=0, Perzentil 2.5/97.5.
- Permutation N1: 2000, seed=0.
- Label-Konvention: anomaly = 1, normal = 0, aus dem Domänen-Manifest.

## Abbruch- und Kosten-Regeln
- Download-Deckel je Domäne: **4 GB**; Gesamt-Deckel `data/`: **+40 GB**.
  Gerissen → Domäne = not-obtained, kein Teil-Subsample „nach Gefühl".
- Jeder schwere Lauf: `nice -n 19 timeout 1800`. Wanduhr gerissen = Timeout
  berichten, nicht still neu starten.
- Nie einen pausierten Job killen.

## Determinismus
Ein Befehl reproduziert alles:

    .venv/bin/python research/factory/sign_identifiability_readout.py

Seeds fix (0). Ausgabe: `research/factory/logs/sign_identifiability_report.md`
+ ein signiertes Receipt je Domäne + ein Gesamt-Receipt.

## Veröffentlichung (Prozess-Beweis, nicht Marketing)
Jede Domänen-Auswertung wird als `signalmap.receipt/1` **signiert** und
veröffentlicht, inklusive der Domänen, deren ehrlicher Verdict **REFUSED**
lautet. Prüfbar ohne signalmap-Installation:

    python tools/verify_receipt.py <receipt.json> --pubkey <pinned>

Ein Fremder muss das Ergebnis offline nachprüfen können, ohne uns zu glauben
und ohne unseren Code zu importieren. Ein Receipt, das nur mit unserem Paket
prüfbar wäre, zählt für diesen Zweck nicht.

## AMENDMENT 1 — 2026-08-06 (Speicherform der Bänke, deklariert VOR jeder Zahl)

Der Bank-Vertrag sagt, längere Signale schneide das Readout ab, nicht der
Bank-Bauer. Die MIMII-Bänke speichern stattdessen **exakt K·W = 20480
Samples** je Recording. Begründung, vor dem Lauf festgehalten: das Readout
liest ohnehin ausschließlich `arr[:20480]`, die gespeicherten Werte sind also
byte-identisch zu dem, was es aus einem vollen Clip gelesen hätte — **keine
Zahl dieser Studie kann sich dadurch ändern**. Ein voller 10-s-Clip als
float64 kostet dagegen das 8-Fache des Quellarchivs. Präzedenz: die
valve-Bank vom 20. Jul (`dcase_valve_adapter.py`) speichert bereits genau
diesen Präfix.

Ebenfalls deklariert: die valve-Eval-Menge dieser Studie umfasst **alle 219
test-Clips** (100 normal + 119 anomaly), nicht die 204 des Juli-Laufs. Dort
waren 15 Anomalie-Clips für den distill-Schritt reserviert; diese Studie
distilliert nicht, also entfällt der Grund für den Ausschluss. Die
Ankerzahl 0.2642 auf genau jenen 204 Clips ist unabhängig davon in
`logs/degenerate_feature_probe.md` reproduziert.

## AMENDMENT 2 — 2026-08-06 (Domänen-Verluste, deklariert VOR jeder Zahl)

Die Beschaffung von `paderborn_kat`, `mafaulda`, den lokalen Domänen
(`cwru`, `ecn`, `geomag`, `battery_eis`, `volcano`) und den synthetischen
Kontrollen (`synth_pos`, `synth_neg`) brach ab, als das Session-Limit riss.
Nach Prereg-Regel gelten sie als **not-obtained**; es wird keine
Ersatz-Domäne nachgeschoben. Ausgewertet werden die vier MIMII-Domänen.

Konsequenz, ausdrücklich vorab: mit vier Domänen aus **einer** Quelle ist H1
nur eingeschränkt prüfbar. Fällt H1 hier positiv aus, ist die Aussage
„Vorzeichen wechselt zwischen Maschinentypen derselben Aufnahmekampagne" —
stärker als ein Einzelbefund, schwächer als „über unabhängige Datensätze
hinweg". Das wird so und nicht stärker berichtet.

## AMENDMENT 3 — 2026-08-06 (M3 ungültig gemessen — Zahl war BEREITS GESEHEN)

**Offenlegung: dieser Block wird geschrieben, NACHDEM die M3-Zahlen sichtbar
waren.** Sie lauteten in allen vier Domänen `alarm_rate(anomaly) =
alarm_rate(normal) = 0.000`, Lücke `+0.000 [+0.000, +0.000]`.

Der Grund ist ein Aggregations-Bruch in der Readout-Implementierung, nicht in
den Daten: die Schwelle des Produkts ist das 99. Perzentil der **Fenster**-
Scores × envelope, verglichen wurde sie aber gegen den **Recording-Mittelwert**
über 20 Fenster. Ein Mittelwert erreicht ein 99.-Perzentil praktisch nie, also
schweigt der Alarm strukturell — unabhängig von jeder Anomalie.

**M3 gilt damit als NICHT GÜLTIG GEMESSEN.** Es wird KEINE Alarm-Aussage aus
diesem Lauf berichtet, weder für noch gegen H3. Die korrekte Messung
(Alarmrate je Fenster, der Einheit, in der `DistilledDetector.alert()`
tatsächlich arbeitet) steht aus und wird als eigener Lauf nachgeholt. Weil die
ungültigen Zahlen bereits gesehen wurden, wird die korrigierte M3-Messung im
Report ausdrücklich als **post-hoc** gekennzeichnet und zählt nicht als
preregistriert.

H1 ist davon unberührt: es ist rein rangbasiert und verwendet keine Schwelle.

## AMENDMENT 4 — 2026-08-06 (Konstruktion der synthetischen Kontrollen, VOR jeder Zahl)

Die Prereg beschrieb `synth_pos`/`synth_neg` über die **Dispersion** der
Anomalie („höhere" bzw. „niedrigere"). Ein Smoke-Test heute zeigte, warum das
so nicht baubar ist: `window()` z-normiert jedes Fenster, Amplituden- und
Varianzunterschiede des Rohsignals sind danach **weg**. Eine Kontrolle über
die Roh-Dispersion hätte nichts gemessen.

Präzisierte Konstruktion, deklariert bevor irgendeine Zahl dieser beiden
Domänen existiert: der Score ist `max|z|`, also **Abstand vom
Healthy-Zentrum**. Eine Verschiebung in *irgendeine* Richtung hebt |z| und
träfe AUC > 0.5. Unter 0.5 kommt man nur, wenn die Anomalie **näher am
Healthy-Zentrum liegt als ein typisches Healthy-Fenster** — eine Kontraktion
im Feature-Raum, keine Verschiebung.

Umgesetzt über den AR(1)-Koeffizienten φ (Autokorrelation = Form, übersteht
die z-Normierung):
- Healthy und Eval-Normal: φ ~ Uniform(0.20, 0.80), Zentrum 0.50
- `synth_pos`: φ = 0.95 → weit außerhalb der Healthy-Streuung
- `synth_neg`: φ = 0.50 → exakt im Healthy-Zentrum

**Registrierte Vorhersagen:** `synth_pos` AUC deutlich **über** 0.5 (aligned),
`synth_neg` AUC **unter** 0.5 (inverted). `synth_neg` ist der konstruktive
Gegenbeweis zu M2: dort ist die Inversion per Bauart wahr, was zeigt, dass die
Richtung eine Eigenschaft der **unbeobachteten** Alternative ist.

Beide Domänen unterscheiden sich in **nichts** außer der φ-Verteilung der
Anomalie; wären sie sonst verschieden, wäre die Kontrolle wertlos.

## AMENDMENT 5 — 2026-08-06 (fünf lokale Domänen not-obtained, unabhängig nachgeprüft)

Die lokalen Domänen der eingefrorenen Liste sind **nicht** baubar. Jede Zahl
unten selbst nachgemessen, nicht übernommen:

| Domäne | Befund | Verstoß |
|---|---|---|
| `geomag` | 16 Dateien (8 quiet, 8 storm), je **8192** Samples | Anzahl (<80) und Länge (<20480) |
| `ecn` | 14 Dateien, 14503–14966 Samples; Dateinamen kodieren **Elektrolyt**, nicht Zustand | Länge; zusätzlich **kein** quellenseitiges healthy/fault-Label |
| `cwru` | 1183 Zeilen à 512 Samples aus **zwei** `.mat`-Quellen (`normal`, `ANOMALY_inner_race_fault`), keine Recording-ID | faktisch 1 gesunde + 1 defekte Aufnahme |
| `battery_eis` | 1870 CSVs mit je **20 Zeilen** (Frequency, ReZ, ImZ) | Impedanz-**Spektren**, keine Zeitreihen; kein binäres Label |
| `volcano` | X-Matrizen (120×1024, 128×1024), zeilenweise \|mean\| ≈ 1e-16, std **exakt 1.0000** | bereits gefenstert und z-normiert → **Rohsignal existiert auf Platte nicht mehr** |

Alle fünf werden als **not-obtained** berichtet. Keine Ersatzdomäne, kein
Auffüllen durch Zerschneiden einzelner Aufnahmen.

**Lehre, die über diese Studie hinausgeht:** die lokalen „Bänke" dieses Repos
sind durchweg **nach** der Vorverarbeitung gespeichert (gefenstert,
z-normiert, längenbeschnitten). Damit sind sie für jede spätere Frage, die
das Rohsignal braucht, unbrauchbar — genau der Fall, der hier eingetreten
ist. Wer eine Bank ablegt, sollte das Rohsignal daneben behalten.

Damit bleiben von 13 Domänen: 4 MIMII + 2 synthetische ausgewertet,
5 lokale not-obtained, `paderborn_kat` und `mafaulda` offen.

## AMENDMENT 6 — 2026-08-06 (Paderborn: Kanalregel und Aufteilung, VOR jeder Zahl)

Die Paderborn-`.mat`-Dateien führen sieben Kanäle in dieser Quellenreihenfolge:
`force`, `phase_current_1`, `phase_current_2`, `speed`, `temp_2_bearing_module`,
`torque`, `vibration_1`. Die Vertragsregel „Kanal 0" trifft `force` mit
**16 001** Samples und verfehlt damit die Mindestlänge von 20 480.

**Ersatzregel, deterministisch und vor jeder Zahl festgelegt:** der **erste
Kanal in der Quellenreihenfolge, der die Mindestlänge erfüllt**. Das ist
`phase_current_1` (Index 1, 258 669 Samples). Für **jede** Aufnahme dieser
Bank derselbe Kanal. Keine Auswahl nach Güte, kein Kanalvergleich.

**Aufteilung, ebenfalls vorab:** getrennt nach **Lager-Code**, nicht nach Lauf —
das ist strenger als der Vertrag verlangt und schließt aus, dass dasselbe
physische Lager in fit und eval auftaucht:
- `fit/` = K001, K002, K003 (unbeschädigt)
- `eval/normal_` = K004, K005 (unbeschädigt, andere Lager)
- `eval/anomaly_` = KA01, KA03 (Außenring), KI01, KI03 (Innenring)

Alle Läufe aller vier Betriebspunkte je Code werden verwendet, in sortierter
Dateireihenfolge; es wird kein Betriebspunkt bevorzugt. Die Schadenscodes
folgen der Quelle, nicht eigener Einschätzung.

## AMENDMENT 7 — 2026-08-07 (MAFAULDA: Kanal und eine bewusste Vertragsunterschreitung, VOR jeder Zahl)

**Zum Zeitpunkt dieser Deklaration existiert keine einzige Zahl der Domäne
`mafaulda`.** Die Bank ist noch nicht gebaut.

**Kanal.** Die Quelle dokumentiert acht Spalten: (1) Tachometer, (2–4)
Underhang-Beschleunigungsaufnehmer (axial, radial, tangential), (5–7)
Overhang, (8) Mikrofon. Spalte 1 ist ein **Drehzahl-Referenzsignal**, keine
Zustandsmessung der Maschine. Verwendet wird deshalb die **erste
Zustands-Messspalte in Quellenreihenfolge**: Spalte 2, Underhang axial. Für
jede Aufnahme dieselbe Spalte, kein Kanalvergleich, keine Auswahl nach Güte.

**Unterschreitung, offen ausgewiesen.** Die Quelle enthält exakt **49**
gesunde Sequenzen (je 5 s bei 50 kHz). Der Vertrag verlangt ≥20 fit **und**
≥30 eval-normal, zusammen 50 — einer zu wenig. Aufteilung daher:
`fit` = die ersten **20** nach sortiertem Dateinamen, `eval/normal_` = die
restlichen **29**. Das ist **eine** Aufnahme unter der Vertragsgrenze und wird
als solche berichtet.

Die Alternative wäre gewesen, eine 5-s-Sequenz in Pseudo-Aufnahmen zu
zerschneiden — das verbietet der Vertrag ausdrücklich, und es wäre schlechter:
das Bootstrap-CI läuft über Aufnahmen, geteilte Aufnahmen täuschten Präzision
vor. Eine ehrlich ausgewiesene 29 ist besser als eine erfundene 30.

**Anomalie.** Alle Unwucht-Sequenzen aller Schweregrade (6 g bis 35 g), in
sortierter Reihenfolge, ohne Auswahl. Die Quelle labelt sie als fehlerhaft;
es wird nicht nachgeprüft, ob ein Schweregrad „stark genug" ist.

## AMENDMENT 8 — 2026-08-07 (Zusatzdomäne `paderborn_kat_n15`, VOR jeder Zahl)

**Beobachtet, mit Zahl:** `paderborn_kat` (AUC 0.4976) und `mafaulda`
(AUC 0.3124) sind **beide an N2 gescheitert**. Ursache nachgewiesen: sortiert
man die gesunden Eval-Dateien, sortiert man bei MAFAULDA nach **Drehzahl**
(32,97 → 61,44 Hz) und bei Paderborn nach **Betriebspunkt** (N09 → N15). Die
gesunde Menge ist in sich trennbar, bevor überhaupt ein Schaden im Spiel ist —
ihre AUC kann den Betriebspunkt statt des Schadens messen. Beide Domänen
bleiben deshalb REFUSED und stützen H1 nicht.

**Neue Domäne, deklariert bevor sie gebaut ist:** `paderborn_kat_n15` — exakt
dieselbe Bank wie `paderborn_kat`, aber beschränkt auf den **einen** nominalen
Betriebspunkt der Quelle, `N15_M07_F10` (1500 min⁻¹, 0,7 Nm, 1000 N). Damit
ist die gesunde Menge homogen und N2 hat eine faire Chance. Alles andere —
Kanal `phase_current_1`, Aufteilung nach Lager-Code (fit K001–K003, eval
normal K004–K005, anomal KA01/KA03/KI01/KI03) — bleibt unverändert.
Erwartete Größen: 60 fit, 40 eval-normal, 80 eval-anomal.

`paderborn_kat` wird **nicht** ersetzt: beide Domänen werden nebeneinander
berichtet. Fällt N2 auch nach der Homogenisierung durch, ist das ein Befund
über den Datensatz und keine Einladung, weiter zu filtern — es wird **kein**
dritter Zuschnitt versucht.

## AMENDMENT 9 — 2026-08-07 (Produkt-Fix mitten in der Studie, Zahlen BEREITS GESEHEN)

**Offenlegung: alle Zahlen der Tabelle lagen vor, als dieser Fix entstand.**

MAFAULDAs post-hoc-M3-Lauf zeigte eine selbstkalibrierte Schwelle von
**9,67·10⁸**. Ursache: auf dieser Bank lagen **zwei** Spec-Programme auf der
absoluten Schutzgrenze von `1e-12` — `std(id(id(x)))` (nach der z-Normierung
per Konstruktion exakt 1.0) und, datenabhängig, `peakcv(id(id(x)))`.
Float-Akkumulationsrauschen von ~1e-12 durch 1e-12 geteilt ergibt einen Term,
der `max|z|` vollständig beherrscht.

**Der Fix ist nicht, solche Features zu streichen.** Ein diskretes Feature,
das auf Healthy echt konstant ist, muss auf eine reale Verschiebung weiterhin
anschlagen — der bestehende Test
`test_detector_survives_zero_mad_and_keeps_sensitivity` schreibt genau das
fest, und ein erster Versuch, degenerierte Features aus dem Score zu nehmen,
brach ihn zu Recht. Stattdessen wird die Schutzgrenze **relativ**:
`max(1e-12, 1e-9·|median|)`. Akkumulationsrauschen landet damit bei z ≈ 1e-3
und bleibt still; jede physikalisch bedeutsame Änderung liegt Größenordnungen
darüber und alarmiert weiterhin. Betroffene Features werden zusätzlich in
`det.degenerate` benannt, damit ein Receipt die Tatsache tragen kann.

**Konsequenz für diese Studie:** das Rezept hat sich geändert, nachdem seine
Ergebnisse sichtbar waren. Die Studie wird deshalb vollständig neu gerechnet
und **beide** Zahlensätze werden berichtet — die Vorher-Werte liegen
unverändert in `logs/sign_cache_before_guardfix/`. Ändert der Fix eine
Richtung oder ein Verdict, wird das ausdrücklich benannt und nicht geglättet.

## AMENDMENT 10 — 2026-08-07 (Korrektur an Amendment 9: Ursachenzuschreibung war falsch)

Amendment 9 schrieb MAFAULDAs Schwelle von 9,67·10⁸ dem
Float-Akkumulationsrauschen zweier Features auf der Schutzgrenze zu. Nach
Messung der Feature-Verteilungen auf der MAFAULDA-Fit-Menge (400 Fenster)
stimmt das nur zur Hälfte:

| Programm | Anteil exakt auf Median | \|Abweichung\| p99 (relativ) | Deutung |
|---|---|---|---|
| `std(id(id(x)))` | 0,0 % | 2,95·10⁻¹³ | Rundungsrauschen — wie angenommen |
| `peakcv(id(id(x)))` | **41,8 %** | **1,65·10⁻³** | **echt diskret** — Annahme falsch |

`peakcv` springt auf dieser Bank um ~3,1·10⁻⁴ absolut. Das ist **kein**
numerisches Artefakt, sondern genau das quantisierte Feature, dessen
Empfindlichkeit `test_detector_survives_zero_mad_and_keeps_sensitivity`
absichtlich schützt. Die extreme Schwelle ist dort korrektes
Selbstkalibrieren, kein Bug — entsprechend bleibt sie auch nach dem Fix bei
4,95·10⁶.

**Was vom Fix bleibt:** die relative Schutzgrenze ist für den
`std`-Fall richtig — ein Programm, das nach der z-Normierung für *jede*
Eingabe exakt 1.0 ist, darf sein Rundungsrauschen nicht durch 1e-12 teilen.
Auf allen neun Bänken hat dieses Feature das Maximum allerdings **nie**
erreicht: der Vergleich vorher/nachher zeigt **bit-identische AUCs in allen
neun Domänen und null Richtungs- oder Verdict-Änderungen**. Der Fix verhindert
also einen **latenten** Fehler, er behebt keinen beobachteten. So und nicht
stärker wird er berichtet.

Die Studienergebnisse sind von alledem unberührt.
