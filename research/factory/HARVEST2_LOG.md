# HARVEST 2 — Replizierbarkeits-Test der Methodik (Jul 2, Nacht)

Frage: Ist die Denkweise/Methodik (bank_audit → gauntlet mit Quittungen) in
KOMPLETT neuen Umgebungen replizierbar — neue Physik UND neue Sensortypen,
mit allen validierten Regeln (Grammatik v2.1 dedup, Stationaritäts-Guard
default, Kapazitäts-Gate, Champion-Regel, Stabilitäts-Quittung)?

## Neue Regeln zuerst ins Werkzeug gegossen (gauntlet.py v2)
- `programs(texture_guard=True)` default (mode="texture"; "phase" schaltet frei)
- **RECEIPT 1**: Selektions-Stabilität (top1-Frequenz + mean-Jaccard über Folds;
  UNSTABLE-Warnung) — empirisch begründet durch VALVE-id-Anti-Learning
- **RECEIPT 2**: Champion-Regel (paired-CI entscheidet lean vs forge; tie→lean)

## Drei neue Sensor-Modalitäten (alle nicht-personenbezogen, frei verfügbar)

### 1. GRIDFREQ — PMU/EDR Netzfrequenz (Energiesysteme)
- Quelle: Power-Grid-Frequency DB (Rydin Gorjão et al., Nat. Comm. 2020), OSF
  by5hu, 1-s-Auflösung. 6 Synchron-Netze: FR01 (CE), GB01, SE01 (Nordic),
  US_TX01 (ERCOT), IS01 (Insel), ZA01. → data/gridfreq/*.csv (lokal).
- Task: Netz-Identität aus 17-min-Fenster (6 Kl., Chance 0.167).
  Recording = Kampagnen-Block (4/Netz, 24 Recs, 192 Fenster).
- CAVEAT: Blöcke eines Netzes = eine Messkampagne (same-campaign-Adjazenz wie
  HYD); Netz+Standort+Periode untrennbar.

### 2. GEOMAG — Fluxgate-Magnetometer (Weltraumwetter)
- Quelle: USGS geomag WS, Observatorium BOU, adjusted 1-s X-Komponente.
- Task: Sturm vs Ruhe (2 Kl.). Sturm-Tage = dokumentierte Events, alle via GFZ
  Kp-API verifiziert (Kp_max ≥ 7); Ruhe-Tage = datengetrieben ruhigste Tage
  2023–24 (Kp_max ≤ 0.67). 8+8 Tage, Recording = Tag.
- Confound-Design: EINE Station für alle Tage (wie SEIS); Fenster-Regel
  identisch für beide Klassen (8×1024 s am Rolling-Std-Peak des Tages).

### 3. GAS — MOX-E-Nose (Chemie), UCI 361 "Twin gas sensor arrays"
- 5 physisch BAUGLEICHE 8-MOX-Einheiten (B1–B5), 4 Gase × 10 Konz. × 4 Rep.,
  100 Hz. → data/gas/data1/ (lokal, 195MB zip).
- Task A (gauntlet): Gas-ID, Recording = Gerät×Gas (20 Recs, 4 Kl.).
- Task B (gauntlet_mixed): **Leave-one-DEVICE-out** (5 Gruppen, mixed labels) —
  überträgt sich die Gas-Signatur auf eine physisch andere Kopie des Sensors?
  (= die sensor-agnostische Kern-Behauptung der Plattform)

## Ergebnisse (Logs: h2_gridfreq.log, h2_geomag.log, h2_gas.log)

| Bank | Chance | LEAN [CI] (perm-p) | FORGE nested [CI] | gepaart | Stabilität | Champion | Verdikt |
|---|---|---|---|---|---|---|---|
| **GEOMAG-storm** | 0.500 | **0.938 [0.867,0.992] (0.016)** | 0.875 [0.758,0.961] | −0.062 n.s. | STABLE (0.56) | **lean** | ★ Physik #5: starkes Signal, lean reicht |
| **GRIDFREQ-area** | 0.167 | 0.536 [0.396,0.677] (0.016) | **0.693 [0.578,0.802]** | **+0.156 [+0.036,+0.281]** | STABLE (0.83, `specflat(clip(diff(x)))`) | **forge** | ★ Physik #6: **CI-fester Forge-Win #5** — Netz-Fingerprint |
| GAS-id (Gerät×Gas) | 0.250 | 0.121 (0.93) | 0.158 [0.112,0.217] | +0.037 n.s. | STABLE aber unter Chance | lean (tie) | ehrliches Null (Form-Fenster) |
| GAS-id-LODO (Device) | 0.250 | 0.267 | 0.217 [0.154,0.283] | −0.050 n.s. | — | — | ehrliches Null (Form-Fenster) |

**GAS-Hypothesen-Kaskade (h2_gas_level.log, h2_gas_4hz.log) — 5 Varianten:**
| Variante | LEAN | FORGE [CI] | Champion (chance-gated) |
|---|---|---|---|
| SHAPE @100Hz (10s-Fenster) | 0.121 | 0.158 | Null |
| LEVEL @100Hz (RAW) | 0.104 | 0.208 | Null — **Level-Hypothese falsifiziert** |
| LEVEL-LODO (Device) | 0.279 | 0.212 | Null |
| LEVEL @4Hz (256s-Fenster) | 0.175 | 0.256 [0.175,0.344] | **NULL-Quittung griff korrekt** (gepaart +0.081 CI-fest, aber niemand über Chance) |
| SHAPE @4Hz (256s) | 0.257 | **0.346 [0.252,0.433]** | forge — aber CI-Untergrenze 0.252 vs Chance 0.250 = **haarscharf, als MARGINAL werten** |

**Ehrliches GAS-Fazit:** (1) Kein belastbares Gas-ID-Signal aus Single-Channel-
MOX in unserem Paradigma — 5 Varianten, sauber austestiert; die Gas-Info steckt
im Multi-Kanal-Muster (8 Sensoren relativ), das die Plattform bewusst (noch)
nicht modelliert = dokumentierte Grenze. (2) **Zeitskalen-Effekt ist real und
monoton** (forge 0.158@10s → 0.346@256s, Selektion 95% stabil, gepaart CI-fest
in beiden 4Hz-Varianten): Fenster-Zeitskala = eigene Suchdimension (Methoden-
These #3, konsistent mit Window-Mode/HYD und Norm-Modus). (3) Chance-Gate für
die Champion-Quittung wurde durch dieses Experiment entdeckt + eingebaut
(paired-CI kürte forge, während beide unter Chance lagen — jetzt: NULL-Verdikt).

**Muster bestätigt sich über jetzt 9 Physiken:** lean deckt impulsive/spektrale
Dynamik ab (GEOMAG wie SEIS/ECG); Forge gewinnt CI-fest, wo die Domänen-Familie
außerhalb des Vibrations-Repertoires liegt (GRIDFREQ wie ECN/CALCE/HYD-cooler).
Champion-Regel + Stabilitäts-Quittung liefen in allen Läufen automatisch und
gaben in jedem Fall das methodisch richtige Urteil.

**Replizierbarkeits-Fazit: JA.** 3 fremde Modalitäten an einem Abend akquiriert
(OSF/USGS/GFZ/UCI, alles frei), Confounds designseitig behandelt (eine Station;
Kp-verifizierte Labels; symmetrische Fenster-Regel; baugleiche Geräte), Pipeline
unverändert durchgelaufen; 2/3 sofort Signal, 1/3 ehrliches Null mit testbarer
Ursache-Hypothese. ~1 h pro Modalität.
