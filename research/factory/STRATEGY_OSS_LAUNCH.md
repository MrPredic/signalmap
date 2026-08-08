# SignalMap — Strategie und Launch-Plan

Stand 17. Juli 2026. Dieses Dokument ist intern (research-Branch, nicht auf main
pushen). Es hält die Richtungsentscheidung fest, den Weg bis zum fertigen
Produkt inklusive Reviews, und die Launch-Roadmap.

## 1. Entscheidung

SignalMap bleibt vollständig open source, inklusive der Premium-Familien.
Umsatz kommt später aus einer Verankerungs-Schicht, nicht aus dem Code.

Begründung in Kürze: Der Feature-Code (RQA, Kohärenz) ist in einem Nachmittag
nachbaubar und taugt nicht als bezahltes Asset. Was nicht nachbaubar ist, sind
die Verdikts-Disziplin (Prereg vor Readout, append-only Ledger, Champion-Regel
die verweigern kann) und mit der Zeit die Marke dahinter. Wir haben inzwischen
sieben preregistrierte Verdikte auf öffentlichen Datensätzen: RQA zugelassen auf
CWRU (+0.066, CI [+0.033, +0.106]), Kohärenz zugelassen auf dem UCI-Hydraulik-
Rig (+0.139, CI [+0.042, +0.236]), drei Verweigerungen (CALCE-RQA, HYD-RQA, GAS-coherence),
dazu ein erster prospektiver Datenpunkt am Kilauea (Ep 51, ORDERING-Kriterium
auf beiden Stationen bestanden, striktes Kriterium nicht). Die Verweigerungen
sind das Verkaufsargument, nicht die Treffer.

Geschäftsmodell danach (nicht jetzt): lokale Receipts bleiben für immer gratis;
bezahlt wird die Gegensignatur plus Eintrag in ein öffentliches, append-only
Verifikations-Log ("attested receipt"), abgerechnet als Prepaid-Pakete.
Preisanker nicht bei 1 Cent — Receipts sind selten und hochwertig, realistisch
sind 1–5 € pro attestiertem Receipt oder 2–5 €/Asset/Monat, Pakete z. B.
100 Proofs für 199 €. Dieser Schritt startet erst, wenn es echte Nutzer gibt;
eine Signatur ohne Reputation ist ein wertloser Stempel. Bekanntes Risiko:
Sigstore/Rekor ankert Hashes gratis, reine Zeitstempel sind Commodity. Unsere
Attestation muss deshalb Prüfung bedeuten (Format, Kette, Gates, Stichproben-
Reproduktion), nicht nur Zeitstempel.

## 2. Was "fertig" für den Launch heißt

- Suite grün (aktuell 103 Tests), 1-Kanal-Pfad byte-identisch gepinnt.
- README/CHANGELOG beschreiben nur, was verifiziert ist; jedes Kommando im
  README wurde wörtlich in einem frischen Checkout ausgeführt.
- Keine privaten Pfade, Mails oder Forschungs-Interna im public Tree.
- pip install aus dem sdist funktioniert, `signalmap distill` läuft mit dem
  `[distill]`-Extra (scipy/scikit-learn fehlten bis 17. Jul als Dependency —
  genau die Art Fehler, die ein externer Reviewer sofort findet).
- Zwei unabhängige Reviews sind durch (siehe §3), Findings gefixt oder bewusst
  und dokumentiert vertagt.

## 3. Härtung und Reviews (vor dem Push)

Reihenfolge ist bewusst seriell, ein Review-Kanal nach dem anderen, damit
Findings nicht doppelt gefixt werden.

Session L1 — erledigt am 17. Jul: Report-Klarheit (base-Gate vs. Champion-Regel
im Receipt erklärt, mit Test), Dependency-Fix, README/CHANGELOG/Version 0.4.0,
Security-Scan der public-bound Files (ein Fund: docs/superpowers/ liegt schon
auf main und enthält einen lokalen Pfad — fliegt beim kuratierten Push raus,
kein Secret, kein History-Rewrite).

Session L2 — Code-Review, ca. ein halber Tag:
ein lokaler fresh-eyes-Subagent (Sonnet, ein einzelner, seriell) reviewt den
kompletten Diff public-main → Launch-Kandidat, mit dem Auftrag, gegen Ground
Truth zu testen statt Code zu lesen (die Regel stammt aus dem HEDG3-Review:
jeder echte CRIT dort kam aus Replay, keiner aus Code-Lesen). Konkret: er baut
sich kleine Banken mit bekannter Wahrheit und prüft, ob Gate, Champion-Regel
und fail-closed-Pfade sich falsch überzeugen lassen. Findings werden TDD-gefixt.

Session L3 — externes Cross-Review plus Frische-Probe, ca. ein halber Tag:
README, CHANGELOG und die distill/premium-Doku gehen durch ein externes Modell
(ChatGPT oder Gemini, freie Tier reicht; ausschließlich public-bound Dateien,
nichts aus research/). Auftrag: Ton, faktische Konsistenz, überverkaufte
Claims, tote Links, und ob ein PdM-Ingenieur ohne Kontext den Quickstart
nachvollziehen kann. Parallel dazu die Frische-Probe: neues venv, pip install
aus dem gebauten sdist, README-Kommandos wörtlich, CI-Konfiguration auf main
gegen die neue Testzahl geprüft.

Session L4 — kuratierter Push und Release, ca. ein halber Tag:
Launch-Branch von origin/main, Package-Verzeichnisse und Docs aus dem
research-Stand rüberkopieren (explizit NICHT: research/, PARALLEL_PLAN.md,
docs/superpowers/, .remember/), Commit, Tag v0.4.0, Push auf main erst nach
kurzem Gegencheck. PyPI-Upload braucht ein Token, das nur du anlegen kannst —
das ist ein User-Schritt. GitHub-Release mit dem CHANGELOG-Text.

Session L5 — Announce, wenige Stunden, getrennt vom Push:
Show HN plus ein Post in r/PredictiveMaintenance oder r/MachineLearning, Hook
ist die Verweigerungs-Story ("a feature-selection receipt that refuses"), mit
den preregistrierten Zahlen und dem Hinweis, dass CWRU/HYD/GAS öffentlich
nachrechenbar sind. Keine Discovery-Claims, der Kilauea-Punkt wird als n=1
erwähnt oder weggelassen, je nach Ep-52-Stand.

Summe bis Launch: 3 Arbeitssessions ab jetzt (L2–L4), plus L5. Kalendarisch
eine Woche, wenn nichts Größeres im Review auftaucht; zwei, wenn doch.

## 4. Nach dem Launch: Mess-Gate und Attestation-Entscheid

Vier Wochen nach Announce wird gemessen, nicht früher entschieden: Stars sind
Deko, zählen tun Issues von fremden Leuten, Clones/Installs, und ob irgendwer
distill auf eigenen Daten laufen lässt und das Receipt postet. Gate für den
Attestation-Service: erster externer Pilot ODER belastbares Nutzer-Feedback
(mehrere unabhängige echte Nutzer), was zuerst eintritt. Dann erst die 2–3
Sessions für Countersign-API, öffentliches Log, verify-URL und Stripe-Prepaid.
Die GitHub-Action (signalmap-verify, CI-Gate wie bei indelible geplant) ist die
Recurring-Variante derselben Infrastruktur und kommt direkt danach.

Wenn das Gate nach acht Wochen nicht erreicht ist: kein Service bauen, sondern
Pilot-Akquise über die Systemhaus-Schiene priorisieren oder das Projekt bewusst
im Reputations-Modus weiterlaufen lassen. Ein Service ohne Nutzer wäre nur
Infrastruktur-Pflege.

## 5. Fachliche Roadmap (parallel, unabhängig vom Launch)

Erstens Ep-52: Watcher läuft, HVO meldet Re-Inflation. Bei Feuer dieselben drei
applies, Preregs gelten unverändert. Ein zweiter ORDERING-Pass würde aus dem
n=1-Demo-Punkt den Anfang einer Serie machen — Kosten praktisch null, höchster
Wert pro Aufwand im ganzen Projekt.

Zweitens eine dritte Premium-Familie: Envelope-/Defektband-Spektrum
(Hüllkurven-Ordnungsanalyse, Lager-Physik). Sie ist der Standard im PdM-Feld,
uns fehlt sie, und sie ist billig (O(n log n)). Das dreht die Premium-Story
elegant um: bisher zeigt die Quittung "teuer nur wo es zahlt", dann zeigt sie
"das Gate entscheidet, unabhängig vom Preis". Erwartung ehrlich offen — auf
CWRU könnte die base-Grammatik sie verweigern, auch das wäre ein gutes
Ergebnis. Eine bis zwei Sessions inklusive Prereg auf CWRU/MFPT/IMS.

Drittens, niedriger priorisiert: Ed25519-Signierung der Receipts im Package
(Muster aus indelible wiederverwenden, eine Session). Sie ist Voraussetzung
für die Attestation, aber auch ohne Service ein Doku-Argument.

Nicht tun: GAS-id mit der c256b8-Config nachverhandeln (bricht die
Ein-Config-Ehrlichkeit), weitere Benchmark-Banken sammeln (Skala ist nicht
unsere Position), Kohärenz auf alle Paare ausbauen (Beleg reicht).

## 6. Risiken, ehrlich

- Alle Zahlen stammen von öffentlichen Lab-Datensätzen. Ohne einen echten
  Nicht-Benchmark-Fall bleibt die Paid-Story dünn. Der E-Waste-Sensor-Katalog
  oder MIMII/DCASE sind die billigsten Wege zu so einem Fall.
- catch22/tsfresh haben Community-Vorsprung; unser Unterschied (Gate + Receipt
  + Verweigerung) muss in den ersten 30 Sekunden der README stehen, sonst
  werden wir als "noch eine Feature-Library" einsortiert.
- Ein-Personen-Projekt: Announce erzeugt Support-Last. Deshalb Announce als
  eigene Session, nicht am Push-Tag.
- Der Kilauea-Strang darf im Launch-Material nicht überverkauft werden. n=1
  und ein bestandenes ORDERING-Kriterium sind ein Anfang, keine Vorhersage.

## 7. Zusammenfassung der Zeiten

Bis zum launch-fertigen Produkt: L2 + L3 + L4 = drei Sessions ab 17. Jul,
Announce (L5) separat. Attestation-Entscheid: vier bis acht Wochen nach
Announce, gebunden an das Nutzer-Gate, nicht an ein Datum. Fachlich laufen
Ep-52 (Kosten null) und die Envelope-Familie (1–2 Sessions) parallel.
