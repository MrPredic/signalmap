"""Harvest 2 (Jul 2 night) — three COMPLETELY new sensor modalities, chosen to
test whether the methodology replicates outside everything used so far:
  PMU/EDR grid-frequency (power systems), fluxgate magnetometer (space weather),
  MOX chemoresistive e-nose (chemistry). All non-personal. Same contract as all
  loaders: list of (z-normed 1024-sample window, label, recording-gid).

GRIDFREQ — Power-Grid-Frequency DB (Rydin Gorjão et al., Nat.Comm. 2020, OSF
by5hu): 1-second frequency deviations, one CSV per location. Task: which
synchronous area does a 17-min window come from? 6 grids of very different size/
inertia: FR01 (Continental Europe), GB01, SE01 (Nordic), US_TX01 (ERCOT),
IS01 (Iceland island grid), ZA01 (South Africa). Recording = contiguous block
of the location's campaign (4 blocks/grid, evenly spread). CAVEAT: blocks of a
grid come from ONE measurement campaign (same-campaign adjacency, like HYD);
labels are grid+location+recorder-period inseparably (device type is the same
EDR family per paper, but full device-identity control would need 2 recorders
per grid).

GEOMAG — USGS geomagnetism web service, observatory BOU (Boulder), adjusted
1-second X component. Task: geomagnetic storm vs quiet day. Storm days chosen
by documented events (all verified Kp_max >= 7 via GFZ API); quiet days = the
data-driven QUIETEST days 2023-24 (Kp_max <= 0.67). Same station for every day
-> station confound designed out (like SEIS); window rule identical for both
classes (8x1024 s at the day's rolling-std peak) -> no procedural confound.
Recording = day. Cached to npz.

GAS — UCI 361 'Twin gas sensor arrays' (Fonollosa et al.): 5 physically
IDENTICAL 8-MOX detection units (B1..B5), 4 gases x 10 concentrations x 4
repetitions, 100 Hz, 640 recordings of 530 s. Signal = one fixed MOX channel.
Two tasks:
  load_gas(unit_groups=False): recording = unit x gas (20 recs, label = gas).
  load_gas(unit_groups=True):  group = UNIT (5 groups, mixed labels) for
    leave-one-DEVICE-out via gauntlet_mixed -> does a gas signature transfer
    to a physically different copy of the sensor? (the sensor-agnostic claim)
"""
import glob, json, os, urllib.request
import numpy as np
from scipy.signal import detrend

W = 1024
BASE = "<local-path>/signalmap/data"

def _znorm(seg):
    s = detrend(np.ascontiguousarray(seg, float))
    return (s - s.mean()) / (s.std() + 1e-12)

# ---------------------------------------------------------------- grid frequency
GRIDS = ("FR01", "GB01", "SE01", "US_TX01", "IS01", "ZA01")

def load_gridfreq(blocks_per_grid=4, win_per_block=8, grids=GRIDS):
    """grids: CSV basenames. Second locations of the same synchronous area
    (PT01=CE like FR01; GB02, US_TX02, ZA02) untangle grid vs location/period:
    they are different recorders, sites and campaign periods within the same
    physical grid (PT01 2018, GB02 2019, ZA02 2025)."""
    raw, gid = [], 0
    for grid in grids:
        v = []
        with open(f"{BASE}/gridfreq/{grid}.csv") as f:
            next(f)
            for ln in f:
                p = ln.split(";")
                try: v.append(float(p[1]))
                except (ValueError, IndexError): v.append(np.nan)
        x = np.array(v)
        seg = len(x) // blocks_per_grid
        for b in range(blocks_per_grid):
            blk = x[b * seg:(b + 1) * seg]
            ok = ~np.isnan(blk)
            # first stretch with win_per_block clean windows
            run, start = 0, None
            for i, o in enumerate(ok):
                run = run + 1 if o else 0
                if run >= win_per_block * W:
                    start = i - run + 1; break
            if start is None: continue
            for k in range(win_per_block):
                raw.append((_znorm(blk[start + k * W:start + (k + 1) * W]), grid, gid))
            gid += 1
    return raw

# ---------------------------------------------------------------- geomagnetic
STORM_DAYS = ["2024-05-10", "2024-05-11", "2024-10-10", "2024-10-11",
              "2023-03-23", "2023-03-24", "2023-04-23", "2024-08-12"]  # Kp>=7 verified
QUIET_DAYS = ["2024-01-07", "2024-02-19", "2024-03-17", "2024-12-26",
              "2023-05-18", "2023-10-17", "2023-10-23", "2023-12-09"]  # Kp<=0.67 (GFZ)

def load_geomag(win_per_day=8, refetch=False, obs="BOU"):
    """obs: USGS observatory code. BOU=Boulder (original bank); FRD=Fredericksburg
    (VA) as second observatory, SAME days -> cross-station transfer test. gid
    numbering identical across obs (day order fixed), so gids are day-matched."""
    os.makedirs(f"{BASE}/geomag", exist_ok=True)
    cache = f"{BASE}/geomag/{obs.lower()}_storm_quiet.npz"
    if os.path.exists(cache) and not refetch:
        z = np.load(cache, allow_pickle=True)
        return [(z["X"][i], str(z["y"][i]), int(z["g"][i])) for i in range(len(z["y"]))]
    X, y, g, days_log = [], [], [], []  # days_log[gid] = day -> day-matched pairing across obs
    gid = 0
    for lab, days in (("storm", STORM_DAYS), ("quiet", QUIET_DAYS)):
        for day in days:
            u = (f"https://geomag.usgs.gov/ws/data/?id={obs}&starttime={day}T00:00:00Z"
                 f"&endtime={day}T23:59:59Z&elements=X&sampling_period=1&format=json")
            try:
                d = json.load(urllib.request.urlopen(u, timeout=180))
                x = np.array([v if v is not None else np.nan
                              for v in d["values"][0]["values"]], float)
            except Exception as e:
                print(f"  skip {day}: {e}", flush=True); continue
            x = x[~np.isnan(x)]
            if len(x) < (win_per_day + 2) * W:
                print(f"  skip {day}: only {len(x)} samples", flush=True); continue
            # identical rule for both classes: block at the day's rolling-std peak
            r = np.array([x[i:i + W].std() for i in range(0, len(x) - W, 512)])
            c = int(np.argmax(r)) * 512 + W // 2
            a = int(np.clip(c - win_per_day * W // 2, 0, len(x) - win_per_day * W))
            for k in range(win_per_day):
                X.append(_znorm(x[a + k * W:a + (k + 1) * W])); y.append(lab); g.append(gid)
            days_log.append(day)
            gid += 1
            print(f"  {lab} {obs} {day}: ok", flush=True)
    np.savez_compressed(cache, X=np.array(X), y=np.array(y), g=np.array(g),
                        days=np.array(days_log))
    return [(X[i], y[i], g[i]) for i in range(len(y))]

# ---------------------------------------------------------------- e-nose (twin MOX)
GAS_LAB = {"GCO": "CO", "GEa": "ethanol", "GEy": "ethylene", "GMe": "methane"}

def load_gas(unit_groups=False, channel=1, files_per_rec=4, win_per_file=3,
             znorm=True, pool=1):
    """pool>1 = TIMESCALE variant: mean-pool the 100 Hz signal by `pool` (e.g.
    25 -> 4 Hz), so one 1024-sample window spans 256 s of the ~100 s gas
    transient instead of 10 s — tests whether the double-null is a timescale
    mismatch between fixed-W windowing and slow chemistry."""
    """znorm=False = LEVEL-PRESERVING variant (method experiment, analogous to
    the HYD phase-aligned finding): MOX gas identity lives in response LEVEL/
    ramp/amplitude, which per-window z-norm deletes — and a 10s window of a
    ~100s transient is locally linear, so even detrend would kill the ramp.
    Windows stay RAW; fold-internal StandardScaler handles scale honestly."""
    raw = []
    rec_ids = {}
    for unit in ("B1", "B2", "B3", "B4", "B5"):
        for gcode, gas in GAS_LAB.items():
            # spread over concentrations, repetition R1 only (R2-4 unused -> fast)
            files = sorted(glob.glob(f"{BASE}/gas/data1/{unit}_{gcode}_F*_R1.txt"))
            picks = [files[i] for i in
                     np.linspace(0, len(files) - 1, min(files_per_rec, len(files))).astype(int)]
            key = unit if unit_groups else f"{unit}-{gas}"
            gid = rec_ids.setdefault(key, len(rec_ids))
            for fp in picks:
                x = np.loadtxt(fp, usecols=channel)
                if pool > 1:
                    n = (len(x) // pool) * pool
                    x = x[:n].reshape(-1, pool).mean(1)
                nw = min(win_per_file, max(len(x) // W, 1))
                offs = np.linspace(0, len(x) - W, nw).astype(int)
                for o in offs:
                    w = (_znorm(x[o:o + W]) if znorm
                         else np.ascontiguousarray(x[o:o + W], float))
                    raw.append((w, gas, gid))
    return raw
