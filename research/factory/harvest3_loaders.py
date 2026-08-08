"""Harvest 3 (Jul 3) — Physics #10: VOLCANIC TREMOR (Kilauea, HV network).

Task: is the volcano ERUPTING right now, read from 10 s of single-station
seismic texture? Barely explored as a continuous-state readout (literature
classifies discrete events — VT/LP quakes — not eruptive state from ambient
texture); data free (IRIS), non-personal.

Design (lessons from SEIS/GEOMAG/GRIDFREQ baked in from day 1):
  - SAME station for all days (station confound designed out); SECOND station
    (RIMD, ~2 km away) fetched with identical rules -> cross-station hardening
    is part of the bank, not an afterthought.
  - Labels = documented USGS-HVO eruption timeline (2018 LERZ; Halema'uma'u
    Dec 2020-May 2021, Sep 2021-...; Jan/Jun 2023) vs confirmed pause windows
    (Aug 2018-Dec 2020, May 2023, Aug 2023).
  - BOTH classes span 2018 and 2023 -> period/instrument confound mitigated
    (GRIDFREQ lesson). Caveat: quiet has no 2021, eruptive no 2019/20 —
    inherent to reality, documented.
  - Identical window rule for both classes: 30 min at 10:00 UTC (00:00 HST,
    anthropogenic-noise minimum), 8x1024 samples @100 sps at the segment's
    rolling-RMS peak (SEIS rule). Recording = day. Cached npz.
"""
import os, urllib.request
import numpy as np
from scipy.signal import detrend

W = 1024
VOLC = "<local-path>/signalmap/data/volcano"

ERUPTIVE_DAYS = ["2018-06-15", "2018-07-15",              # LERZ fissure 8
                 "2021-01-15", "2021-04-01",              # lava lake 12/20-5/21
                 "2021-10-15", "2021-12-01",              # eruption since 9/21
                 "2023-01-20", "2023-06-15"]              # 2023 episodes
QUIET_DAYS = ["2018-10-15", "2019-03-15", "2019-08-15",   # documented pause
              "2020-02-15", "2020-07-15", "2020-11-15",   # (8/18 - 12/20)
              "2023-05-15", "2023-08-15"]                 # inter-episode pauses


def _znorm(seg):
    s = detrend(np.ascontiguousarray(seg, float))
    return (s - s.mean()) / (s.std() + 1e-12)


def load_volcano(win_per_day=8, refetch=False, sta="UWE"):
    """sta: HV station code. UWE = Uwekahuna vault (summit, primary bank);
    RIMD = second station for cross-station transfer. Both 100 sps HHZ,
    verified available 2018-2023. gid = index into ERUPTIVE+QUIET day list
    (fixed order) -> gids are day-matched across stations by construction."""
    os.makedirs(VOLC, exist_ok=True)
    cache = f"{VOLC}/kilauea_{sta.lower()}.npz"
    if os.path.exists(cache) and not refetch:
        z = np.load(cache, allow_pickle=True)
        return [(z["X"][i], str(z["y"][i]), int(z["g"][i])) for i in range(len(z["y"]))]
    X, y, g, days_log = [], [], [], []
    day_list = [("eruptive", d) for d in ERUPTIVE_DAYS] + [("quiet", d) for d in QUIET_DAYS]
    for gid, (lab, day) in enumerate(day_list):
        u = (f"https://service.iris.edu/irisws/timeseries/1/query?net=HV&sta={sta}"
             f"&loc=--&cha=HHZ&starttime={day}T10:00:00&endtime={day}T10:30:00&format=ascii1")
        try:
            d = urllib.request.urlopen(u, timeout=180).read().decode()
            x = np.array(d.strip().split("\n")[1:], float)
        except Exception as e:
            print(f"  skip {day}: {e}", flush=True); continue
        if len(x) < (win_per_day + 2) * W:
            print(f"  skip {day}: only {len(x)} samples", flush=True); continue
        r = np.array([x[i:i + W].std() for i in range(0, len(x) - W, 512)])
        c = int(np.argmax(r)) * 512 + W // 2
        a = int(np.clip(c - win_per_day * W // 2, 0, len(x) - win_per_day * W))
        for k in range(win_per_day):
            X.append(_znorm(x[a + k * W:a + (k + 1) * W])); y.append(lab); g.append(gid)
        days_log.append((gid, lab, day))
        print(f"  {lab} {sta} {day}: ok", flush=True)
    np.savez_compressed(cache, X=np.array(X), y=np.array(y), g=np.array(g),
                        days=np.array(days_log, dtype=object))
    return [(X[i], y[i], g[i]) for i in range(len(y))]


if __name__ == "__main__":
    for sta in ("UWE", "RIMD"):
        raw = load_volcano(sta=sta)
        print(f"{sta}: {len(raw)} windows, {len(set(r[2] for r in raw))} days", flush=True)
