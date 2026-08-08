# DOMAIN BANK CONTRACT — eingefroren 6. Aug 2026

Verbindlich für jede Domäne aus `PREREG_SIGN_IDENTIFIABILITY.md`. Wer eine
Bank baut, hält sich exakt hieran. Abweichung = Domäne wird als
**not-obtained** berichtet, nicht als angepasste Bank.

## Verzeichnis

    data/signdomains/<domain>/
      fit/<name>.npy              # NUR gesunde Recordings
      eval/normal_<name>.npy      # gelabelte Eval-Recordings
      eval/anomaly_<name>.npy
      manifest.json

`<domain>` ist exakt einer der Namen aus der Prereg-Domänenliste.

## Datei-Format
- Genau **ein Recording pro `.npy`**, eindimensional, `float64`, C-contiguous.
- **Rohsignal**, unverändert. Keine Normierung, keine Filterung, kein
  Detrend, kein Resampling, kein Ausreißer-Trimmen. Fensterung, Detrend und
  z-Normierung macht ausschließlich das Readout-Skript.
- Mindestens **20480 Samples** (= K=20 × W=1024). Längere Signale werden vom
  Readout vorne abgeschnitten, nicht vom Bank-Bauer.
- Kein NaN, kein Inf, nicht konstant (`std > 0`).
- Mehrkanalige Quellen: **Kanal 0** nach dokumentierter Kanal-Reihenfolge der
  Quelle, im Manifest benannt. Kein Kanal-Mischen, keine Kanal-Auswahl nach
  Güte.

## Label
Ausschließlich über das Dateipräfix in `eval/`: `normal_` bzw. `anomaly_`.
Die Zuordnung folgt dem **Label der Quelle**, nicht einer eigenen
Einschätzung. Ist die Quelle mehrstufig (z. B. Schweregrade), zählt jeder
von der Quelle als fehlerhaft markierte Zustand als `anomaly_`; das Mapping
kommt wörtlich ins Manifest.

## Disjunktheit (harte Regel)
Ein Recording, das in `fit/` liegt, darf in `eval/` **nicht** vorkommen —
weder dieselbe Datei noch ein überlappender Ausschnitt derselben Aufnahme.
Wo die Quelle nur einen gesunden Block hat: nach Aufnahme-ID trennen, nie
denselben Zeitabschnitt zweimal verwenden. Das Readout prüft das über
sha256 der Fenster und wirft die Domäne bei Treffer raus.

## Mindestgrößen
- `fit/` ≥ **20** Recordings
- `eval/normal_` ≥ **30**, `eval/anomaly_` ≥ **30**

Nicht erreichbar → Domäne = **not-obtained** melden. Nicht durch Zerschneiden
eines Signals in Pseudo-Recordings auffüllen: künstlich aufgeblähte
Recording-Zahlen zerstören das Bootstrap-CI, das über Recordings läuft.

## manifest.json

```json
{
  "domain": "paderborn_kat",
  "source_url": "https://…",
  "license": "…",
  "modality": "motor current | vibration | acoustic | …",
  "fs_hz": 64000,
  "channel": "0 = phase current, per source docs",
  "anomaly_mapping": "KA01–KA09 (outer race) + KI01–KI08 (inner race) => anomaly",
  "n_fit": 24, "n_eval_normal": 40, "n_eval_anomaly": 40,
  "bytes_downloaded": 3221225472,
  "files": {"fit/K001_1.npy": "<sha256>", "…": "…"},
  "notes": "alles, was ein Prüfer wissen muss, um dieselbe Bank zu bauen"
}
```

## Verboten (wichtigster Abschnitt)
Wer eine Bank baut, **berechnet keine Trenn-Metrik**: kein AUC, kein ROC,
keine Accuracy, kein t-Test, kein Mittelwertvergleich normal-vs-anomaly,
kein Plot der beiden Klassen gegeneinander. Auch nicht „nur zum
Plausibilisieren". Grund: sobald die Bank gegen ein Ergebnis geprüft wurde,
ist die Bank-Auswahl Teil des Ergebnisses, und die Vorzeichen-Messung wäre
wertlos. Erlaubte Prüfungen sind ausschließlich **klassenblind**: Länge,
dtype, NaN, `std > 0`, Datei-Anzahl, sha256, Disjunktheit.

## Selbstprüfung
Jede Bank wird mit dem gemeinsamen Prüfer abgenommen, der genau diese
klassenblinden Zusagen testet:

    .venv/bin/python research/factory/check_domain_bank.py data/signdomains/<domain>

Grün = geliefert. Rot = reparieren oder not-obtained melden.
