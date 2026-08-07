# M2 — Warum die Richtung nicht identifizierbar ist (Theorie-Pfad)

Gehört zu `PREREG_SIGN_IDENTIFIABILITY.md`, Methode M2. Der konstruktive
Gegenbeweis dazu ist als Domäne `synth_neg` gebaut und gemessen; dieses
Dokument liefert die Aussage, aus der er folgt.

## Aufbau

Ein Fenster ist $w \in \mathbb{R}^{1024}$ (nach Detrend und z-Normierung).
Die Spec bildet es über $d = 9$ Programme ab:
$\varphi(w) = (\varphi_1(w), \dots, \varphi_d(w))$.

`DistilledDetector.fit` sieht **ausschließlich** Healthy-Fenster,
$w \sim P_0$, und schätzt daraus komponentenweise Median $m_j$ und MAD $s_j$.
Der Score ist

$$S(w) \;=\; \max_{j \le d} \frac{|\varphi_j(w) - m_j|}{s_j}.$$

Entscheidend: $S$ ist **eine Funktion von $P_0$ allein**. Die Anomalie-
Verteilung $P_1$ geht in die Kalibrierung nirgends ein — sie wird nie
beobachtet.

## Satz (Nicht-Identifizierbarkeit)

Sei $P_0$ fest und so, dass $S(X_0)$ für $X_0 \sim P_0$ atomlos ist. Die
Trenngüte

$$\mathrm{AUC}(P_0, P_1) \;=\; \Pr[S(X_1) > S(X_0)] + \tfrac12 \Pr[S(X_1) = S(X_0)],
\qquad X_1 \sim P_1,$$

nimmt bei **festem $P_0$** — und damit bei festem, vollständig kalibriertem
Detektor — jeden Wert in $[0,1]$ an.

**Beweis.** Zwei Wahlen von $P_1$ genügen.

*(i) AUC = 0.* Wähle $w^\*$ mit $\varphi(w^\*) = m$, also $S(w^\*) = 0$, und
setze $P_1 = \delta_{w^\*}$. Da $S \ge 0$ und $S(X_0)$ atomlos ist, gilt
$S(X_0) > 0$ fast sicher, also $\Pr[S(X_1) > S(X_0)] = 0$ und die Gleichheits-
Menge ist eine Nullmenge.

*(ii) AUC = 1.* Wähle $w^{\*\*}$ mit $S(w^{\*\*}) > \operatorname{ess\,sup} S(X_0)$
und setze $P_1 = \delta_{w^{\*\*}}$.

Zwischenwerte folgen aus Mischungen $\alpha \delta_{w^\*} + (1-\alpha)
\delta_{w^{\*\*}}$, $\alpha \in [0,1]$. $\square$

**Korollar.** Auch $\operatorname{sign}(\mathrm{AUC} - \tfrac12)$ ist aus $P_0$
nicht bestimmbar. Die Aussage „Anomalien liegen weiter vom Healthy-Zentrum"
ist damit kein aus Daten gelernter Befund, sondern eine **Zusatzannahme über
die unbeobachtete Alternative**.

Das ist keine Schwäche der Implementierung: **jeder** Detektor, der nur
Healthy sieht und über einen Abstand zum Healthy-Zentrum entscheidet, erbt
diese Lücke. Sie ist eine Eigenschaft des Aufgabenzuschnitts.

## Was die Richtung festlegt: Streuung, nicht Verschiebung

$S$ misst **Abstand**, nicht Vorzeichen. Eine Verschiebung der Anomalie in
irgendeine Richtung hebt $|z|$ und ergäbe AUC > 0.5. Unter 0.5 kommt man nur,
wenn die Anomalie im Feature-Raum **enger am Zentrum konzentriert** ist als
ein typisches Healthy-Fenster — eine Kontraktion.

Quantitativ im sauberen Fall $d = 1$, beide Verteilungen zentriert, Skalen
$\sigma_0$ (healthy) und $\sigma_1$ (anomal), Streuungsverhältnis
$r = \sigma_1/\sigma_0$: mit $Z_0, Z_1$ unabhängig standardnormal ist
$Z_1/Z_0$ Cauchy-verteilt, also $\Pr[|Z_1|/|Z_0| \le t] = \tfrac{2}{\pi}\arctan t$,
und damit

$$\boxed{\;\mathrm{AUC}(r) \;=\; \Pr\!\big[r|Z_1| > |Z_0|\big] \;=\; \frac{2}{\pi}\arctan r \;}$$

Kontrollpunkte: $r = 1 \Rightarrow 0.5$; $r \to 0 \Rightarrow 0$;
$r \to \infty \Rightarrow 1$. Die Umkehrung $r = \tan(\tfrac{\pi}{2}\mathrm{AUC})$
macht daraus eine **prüfbare Vorhersage**: aus einer gemessenen AUC folgt ein
Streuungsverhältnis, das man unabhängig aus den Feature-Matrizen ausrechnen
kann. Genau das tut `theory_auc_check.py`.

## Grenzen — welche Schritte exakt sind und welche nicht

- **Exakt:** der Satz und sein Korollar. Sie brauchen nur Atomlosigkeit von
  $S(X_0)$.
- **Exakt:** $\mathrm{AUC}(r) = \tfrac{2}{\pi}\arctan r$ unter seinen eigenen
  Annahmen ($d=1$, beide Verteilungen zentriert und normal, gemeinsames
  Zentrum, MAD-Skalierung ersetzt durch die wahre Skala).
- **Nicht exakt bei uns:** $d = 9$, und $S$ ist das **Maximum** über neun
  korrelierte Komponenten. Das Maximum verschiebt beide Verteilungen nach
  oben und komprimiert das Verhältnis; die Formel ist deshalb eine
  Näherung, keine Identität. Erwartung: die Richtung (über/unter 0.5) trifft,
  der Betrag weicht ab.
- **Nicht exakt bei uns:** die Aggregation je Recording ist ein Mittelwert
  über 20 Fenster, was beide Verteilungen zusätzlich konzentriert.
- **Offen:** reale Anomalien sind selten exakt zentriert. Verschiebung und
  Kontraktion treten gemeinsam auf und wirken gegenläufig.

Divergenz zwischen Formel und Messung wird nach Prereg als **ehrlicher
Downgrade** berichtet, nicht weggerechnet.

## Was die Richtung identifizieren würde

1. **Gelabelte Anker** — schon wenige bestätigte Anomalien legen das
   Vorzeichen fest. Das ist der billigste Ausweg und der ehrlichste Pitch:
   nicht „label-frei", sondern „label-arm, mit ausgewiesenem Anker".
2. **Ein einseitiges physikalisches Modell** — z. B. „Lagerschaden erzeugt
   zusätzliche Impulsivität", das die Richtung a priori festlegt. Dann ist
   die Annahme benannt und prüfbar statt implizit.
3. **Ein richtungsfreier Score** — Auswertung über $\max(\mathrm{AUC},
   1-\mathrm{AUC})$ gegen die Shuffle-Null (H2 der Prereg). Der kauft
   Entdeckung, aber **keine** Entscheidung: man weiß, dass etwas anders ist,
   nicht in welche Richtung.

Ohne eines dieser drei ist der ehrliche Verdict **REFUSED** — nicht „AUC
0.26".
