"""Ep-51 Retro-Diagnostik -> Hypothese fuer Ep 52 (17. Jul 2026).

ZWECK, EHRLICH DEKLARIERT: Das hier ist EXPLORATIV. Ep 51 ist gelaufen
(Onset 2026-07-15T18:30Z, HVO; Praekursor-Overflow ab 00:51Z). Wir schauen
rueckwaerts, welche Methode x Offset die echten Pre-Onset-Fenster von tiefen
Pause-Kontrollen getrennt HAETTE. n=1 Episode, ~50 Zellen -> Selektions-Risiko
maximal. Deshalb: kein p-Wert-Theater, nur Effekt-Richtung + Konsistenz ueber
beide Stationen; das Top-Ergebnis wird als EP52-Prereg EINGEFROREN (VOR Ep 52)
und dort bestaetigt oder beerdigt. Selektionsprozess steht im Ledger.

Offsets: 1h/3h/6h/12h liegen IN der dokumentierten Overflow-Phase (ab 00:51Z),
24h/36h/48h davor = echte Vorhersage-Zone. Getrennt ausgewiesen.
Kontrollen: gleiche Uhrzeit am 11. + 12. Jul (tiefe Pause, vor jedem Overflow).
Fenster/Fetch: exakt volcano_precursor._fetch (gehaertet, znorm) -> Varianz-CSD
ist hier NICHT testbar (znorm), ar1 schon. Cross-Station-Kohaerenz: nicht
alignt in diesem Cache-Pfad, vertagt.

Run: cd research/factory && ../../.venv-research/bin/python3 ep52_retro.py
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from receipt_ledger import log_receipt  # noqa: E402
from volcano_precursor import _fetch  # noqa: E402

from signalmap.premium import rqa_features  # noqa: E402

ONSET = datetime.fromisoformat("2026-07-15T18:30")
OFFSETS_H = [1, 3, 6, 12, 24, 36, 48]
OVERFLOW_START = datetime.fromisoformat("2026-07-15T00:51")
CTRL_DAYS = ["2026-07-11", "2026-07-12"]
STATIONS = ["UWE", "RIMD"]


def psd_slope(x):
    from scipy.signal import welch
    f, P = welch(x, nperseg=min(256, len(x)))
    f, P = f[1:], P[1:]
    return float(np.polyfit(np.log(f), np.log(P + 1e-20), 1)[0])


def perm_ent(x):
    a, b, c = x[:-2], x[1:-1], x[2:]
    pat = (a < b).astype(int) * 2 + (b < c).astype(int)
    h = np.bincount(pat, minlength=4) / max(len(pat), 1)
    h = h[h > 0]
    return float(-(h * np.log(h)).sum() / np.log(4))


def ar1(x):
    v = x - x.mean()
    d = (v * v).sum()
    return float((v[1:] * v[:-1]).sum() / d) if d > 0 else 0.0


def speccent(x):
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.arange(len(P))
    return float((f * P).sum() / (P.sum() + 1e-12) / len(P))


def rqa_det(x):
    return float(rqa_features(x)[1])


FEATS = {"perm_ent": perm_ent, "psd_slope": psd_slope, "ar1": ar1,
         "speccent": speccent, "rqa_det": rqa_det}


def seg_means(sta, t0):
    X = _fetch(sta, t0.strftime("%Y-%m-%dT%H:%M"))
    if X is None:
        return None
    return {k: float(np.mean([fn(w) for w in X])) for k, fn in FEATS.items()}


def main():
    rows = []
    for off in OFFSETS_H:
        pre_t = ONSET - timedelta(hours=off)
        zone = "overflow" if pre_t >= OVERFLOW_START else "pre-overflow"
        for sta in STATIONS:
            pre = seg_means(sta, pre_t)
            ctrls = []
            for day in CTRL_DAYS:
                c = seg_means(sta, datetime.fromisoformat(
                    f"{day}T{pre_t.strftime('%H:%M')}"))
                if c:
                    ctrls.append(c)
            if pre is None or len(ctrls) < 2:
                print(f"T-{off}h {sta}: unvollstaendig (fetch)", flush=True)
                continue
            for k in FEATS:
                cvals = [c[k] for c in ctrls]
                cm = float(np.mean(cvals))
                spread = max(abs(cvals[0] - cvals[1]) / 2, 1e-9)
                rows.append({"offset_h": off, "zone": zone, "station": sta,
                             "feature": k, "pre": pre[k], "ctrl_mean": cm,
                             "delta": pre[k] - cm,
                             "delta_over_spread": (pre[k] - cm) / spread})
        print(f"T-{off}h done", flush=True)

    # Rangliste: |delta/spread| gemittelt ueber Stationen, nur Zellen wo BEIDE
    # Stationen dieselbe Richtung zeigen (Konsistenz > Magnitude).
    print("\n== Zellen mit gleicher Richtung auf BEIDEN Stationen ==", flush=True)
    ranked = []
    for off in OFFSETS_H:
        for k in FEATS:
            cell = [r for r in rows if r["offset_h"] == off and r["feature"] == k]
            if len(cell) != 2:
                continue
            if np.sign(cell[0]["delta"]) == np.sign(cell[1]["delta"]) != 0:
                score = float(np.mean([abs(r["delta_over_spread"]) for r in cell]))
                ranked.append((score, off, cell[0]["zone"], k,
                               float(np.sign(cell[0]["delta"])),
                               [round(r["delta"], 4) for r in cell]))
    ranked.sort(reverse=True)
    for score, off, zone, k, sgn, deltas in ranked:
        print(f"  T-{off}h [{zone}] {k}: dir={'+' if sgn > 0 else '-'} "
              f"score={score:.1f} deltas={deltas}", flush=True)

    tip = log_receipt("EP52-RETRO", {
        "purpose": "EXPLORATORY hypothesis generation on the single Ep-51 "
                   "episode; feeds a frozen EP52 prereg; selection process "
                   "declared (rank = |delta/ctrl-day-spread|, both-station "
                   "direction agreement required)",
        "onset_utc": "2026-07-15T18:30", "overflow_start": "2026-07-15T00:51",
        "offsets_h": OFFSETS_H, "ctrl_days": CTRL_DAYS,
        "not_testable_here": ["variance-CSD (znorm cache)",
                              "cross-station coherence (unaligned windows)"],
        "rows": rows,
        "ranked_consistent": [
            {"score": s, "offset_h": o, "zone": z, "feature": f, "dir": d,
             "deltas": dl} for s, o, z, f, d, dl in ranked]})
    print(f"\nledger tip = {tip}", flush=True)


if __name__ == "__main__":
    main()
