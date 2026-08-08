"""Loaders for the new retro banks (Jul 2 data harvest).
Same contract as feature_forge loaders: list of (normalized window, label, gid).

HYDRAULIC (UCI 447, ZeMA rig): 2205 cycles x 60s, PS1 pressure @100Hz (6000
samples/cycle). Targets: valve condition (4 cls) or cooler (3 cls). Recording =
cycle; per class we pick cycles evenly spread across the experiment. CAVEAT
(documented): same-class cycles come from one continuous rig experiment ->
same-condition adjacency; weaker independence than ECN/CWRU recordings.

CALCE CS2 (LiCoO2 cells, Univ. of Maryland): per-cycle voltage curves from 6
cells. Task: SOH stage (early/mid/late life tercile) from the discharge-voltage
sequence. Recording = cell x life-stage block.
"""
import glob, os
import numpy as np
from scipy.signal import detrend

W = 1024
HYD = "<local-path>/signalmap/data/hydraulic"

def _push(x, cls, gid, raw):
    x = np.asarray(x, float)
    for k in range(len(x) // W):
        s = detrend(np.ascontiguousarray(x[k * W:(k + 1) * W]))
        raw.append(((s - s.mean()) / (s.std() + 1e-12), cls, gid))

def _readout(x, n_win, sampling="even", pool=1):
    """SOURCE-LEVEL readout applied to one clip/channel (Jul 3/4, method theses
    #1+#3; identical rule for every class -> no procedural confound). Returns
    (signal, window offsets). pool=k: mean-pool the WHOLE signal by k first,
    THEN cut W-sample windows (window spans k x more time). sampling='peak':
    windows at the n_win highest non-overlapping rolling-RMS positions (sparse
    impulses; 'even' can straddle silence — cf. DCASE-valve 0.238 -> 0.875)."""
    x = np.asarray(x, float)
    if pool > 1:
        n = (len(x) // pool) * pool
        x = x[:n].reshape(-1, pool).mean(1)
    if sampling == "peak":
        step = 256
        r = np.array([x[i:i + W].std() for i in range(0, len(x) - W, step)])
        offs, blocked = [], np.zeros(len(r), bool)
        for k in np.argsort(-r):
            if not blocked[k]:
                offs.append(k * step)
                lo = max(0, k - W // step); blocked[lo:k + W // step + 1] = True
            if len(offs) == n_win: break
    else:
        offs = np.linspace(0, len(x) - W, n_win).astype(int)
    return x, offs

def load_hydraulic(target="valve", sensor="PS1", per_class=6, phase_aligned=False):
    """phase_aligned=True: NEW-METHOD variant — one window per cycle, locked to
    the cycle start (valve switching transient), detrended but NOT z-normalized
    per window (level/scale preserved; fold-internal StandardScaler handles it).
    Texture gauntlet is phase-agnostic + scale-blind; this tests whether the
    valve information lives in the aligned transient shape instead."""
    prof = np.loadtxt(f"{HYD}/profile.txt")
    col = {"cooler": 0, "valve": 1, "pump": 2, "accumulator": 3}[target]
    stable = prof[:, 4] == 0
    labels = prof[:, col]
    sig = np.loadtxt(f"{HYD}/{sensor}.txt")
    raw, gid = [], 0
    for cls in np.unique(labels):
        cand = np.where((labels == cls) & stable)[0]
        picks = cand[np.linspace(0, len(cand) - 1, per_class).astype(int)]
        for c in picks:
            if phase_aligned:
                s = detrend(np.ascontiguousarray(sig[c][:W], float))
                raw.append((s, f"{target}{cls:g}", gid))
            else:
                _push(sig[c], f"{target}{cls:g}", gid, raw)
            gid += 1
    return raw

DCASE = "<local-path>/signalmap/data/dcase_valve"

def load_dcase_valve(task="id", per_group=6, win_per_clip=5, full_coverage=True,
                     sampling="even", pool=1):
    """DCASE2020 task2 valve (MIMII-derived, 16kHz 10s clips).
    task='id': 4-class machine-ID from normal TRAIN clips (recording = clip).
    task='anomaly': normal-vs-anomaly on TEST clips of id_00.
    full_coverage=True (FIX of the documented coverage caveat, same scheme as
    load_dcase_pump): windows spread EVENLY across the whole 10s clip instead of
    only the first 0.32s — valve impulses are sparse, early-only sampling missed
    them. full_coverage=False reproduces the original (caveated) numbers.
    SOURCE-LEVEL readout variants (Jul 3, timescale-screen follow-up; identical
    rule for every class -> no procedural confound):
      sampling='peak': windows at the win_per_clip highest non-overlapping
        rolling-RMS positions (valve impulses are sparse; 'even' can straddle
        silence). pool=k: mean-pool the WHOLE clip by k first (16 kHz -> 1 kHz
        at k=16), THEN cut W-sample windows — window spans k x more time; the
        honest version of what timescale_screen only proxies."""
    from scipy.io import wavfile
    raw, gid = [], 0
    if task == "id":
        groups = {f"id_{i:02d}": sorted(glob.glob(
            f"{DCASE}/valve/train/normal_id_{i:02d}_*.wav"))
            for i in (0, 2, 4, 6)}
    else:
        groups = {lab: sorted(glob.glob(
            f"{DCASE}/valve/test/{lab}_id_00_*.wav"))
            for lab in ("normal", "anomaly")}
        per_group = 8
    for lab, files in groups.items():
        picks = [files[i] for i in
                 np.linspace(0, len(files) - 1, min(per_group, len(files))).astype(int)]
        for f in picks:
            _, x = wavfile.read(f)
            if full_coverage:
                x, offs = _readout(x, win_per_clip, sampling, pool)
                for o in offs:
                    s = detrend(np.ascontiguousarray(x[o:o + W]))
                    raw.append(((s - s.mean()) / (s.std() + 1e-12), lab, gid))
            else:
                _push(x[:win_per_clip * W].astype(float), lab, gid, raw)
            gid += 1
    return raw

# ---------------------------------------------------------------- Jul 2 harvest B
def bank_audit(name, raw):
    """Per-bank pre-gauntlet audit (audit.py checks the PIPELINE; this checks the
    BANK): (1) group integrity: one label per recording (gauntlet assumes it),
    (2) recording-level label shuffle must collapse the lean score toward chance
    (rules out group-structure artifacts specific to this bank)."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    from feature_forge import lean_baseline
    y = np.array([r[1] for r in raw]); g = np.array([r[2] for r in raw])
    recs = np.unique(g)
    mixed = [r for r in recs if len(set(y[g == r])) > 1]
    ok1 = not mixed
    print(f"[{'PASS' if ok1 else 'FAIL'}] {name} group-integrity: "
          f"{len(mixed)} mixed-label recordings (must be 0)", flush=True)
    L = lean_baseline(raw)
    clf = lambda: make_pipeline(StandardScaler(),
                                RandomForestClassifier(150, random_state=0, n_jobs=-1))
    acc = cross_val_score(clf(), L, y, groups=g, cv=LeaveOneGroupOut()).mean()
    rng = np.random.default_rng(0)
    rc = {r: y[g == r][0] for r in recs}
    perm = rng.permutation([rc[r] for r in recs]); ymap = dict(zip(recs, perm))
    acc_s = cross_val_score(clf(), L, np.array([ymap[r] for r in g]),
                            groups=g, cv=LeaveOneGroupOut()).mean()
    chance = 1 / len(set(y))
    ok2 = acc_s < max(chance + 0.15, acc - 0.10)
    print(f"[{'PASS' if ok2 else 'FAIL'}] {name} label-shuffle: real {acc:.3f} -> "
          f"shuffled {acc_s:.3f} (chance {chance:.3f})", flush=True)
    return ok1 and ok2

DCASE_PUMP = "<local-path>/signalmap/data/dcase_pump"

def load_dcase_pump(task="id", per_group=6, win_per_clip=5, sampling="even", pool=1):
    """DCASE2020 task2 pump (MIMII-derived, 16kHz 10s clips). Different physics
    than valve: continuous cavitation/flow noise instead of impulsive switching.
    task='id': 4-class machine-ID from normal TRAIN clips (recording = clip).
    task='anomaly': normal-vs-anomaly on TEST clips of id_00.
    FIX vs load_dcase_valve coverage caveat: windows are spread EVENLY across
    the full 10s clip (not just the first 0.32s).
    sampling='peak'/pool=k: source-level readout variants, see _readout."""
    from scipy.io import wavfile
    raw, gid = [], 0
    if task == "id":
        groups = {f"id_{i:02d}": sorted(glob.glob(
            f"{DCASE_PUMP}/pump/train/normal_id_{i:02d}_*.wav"))
            for i in (0, 2, 4, 6)}
    else:
        groups = {lab: sorted(glob.glob(
            f"{DCASE_PUMP}/pump/test/{lab}_id_00_*.wav"))
            for lab in ("normal", "anomaly")}
        per_group = 8
    for lab, files in groups.items():
        picks = [files[i] for i in
                 np.linspace(0, len(files) - 1, min(per_group, len(files))).astype(int)]
        for f in picks:
            _, x = wavfile.read(f)
            x, offs = _readout(x, win_per_clip, sampling, pool)
            for o in offs:
                s = detrend(np.ascontiguousarray(x[o:o + W]))
                raw.append(((s - s.mean()) / (s.std() + 1e-12), lab, gid))
            gid += 1
    return raw

SEIS = "<local-path>/signalmap/data/seismic"

def load_seismic_depth(per_class=8, win_per_event=8, refetch=False, sta="ANMO"):
    """Teleseismic depth classification: shallow (<70 km) vs deep (>300 km)
    M6.3+ earthquakes 2024-2026, ALL recorded at the SAME station (IU.<sta>.00.BHZ,
    40 sps, via IRIS timeseries + USGS event web services) -> no station confound
    by construction. Recording = event (one label per event, clean LOGO).
    Signal: 30 min from origin+60s; the win_per_event*1024-sample block with the
    highest rolling RMS (= the wavetrain) is cut into z-normed windows.
    Physics: deep events lack surface waves + differ spectrally. Cached as npz.
    sta: IU station code (ANMO=New Mexico original bank; KONO=Norway, MAJO=Japan
    both verified 40 sps -> cross-station transfer without sr confound). USGS
    event query is deterministic (orderby=magnitude), so all stations iterate
    the same candidate events; meta stores event time for overlap checks."""
    import json, urllib.request, urllib.parse, datetime
    os.makedirs(SEIS, exist_ok=True)
    cache = f"{SEIS}/bank_depth.npz" if sta == "ANMO" else f"{SEIS}/bank_depth_{sta}.npz"
    if os.path.exists(cache) and not refetch:
        z = np.load(cache, allow_pickle=True)
        return [(z["X"][i], str(z["y"][i]), int(z["g"][i])) for i in range(len(z["y"]))]

    def usgs(extra):
        u = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
             "&starttime=2024-01-01&endtime=2026-06-01&minmagnitude=6.3"
             "&orderby=magnitude&limit=16&" + extra)
        return json.load(urllib.request.urlopen(u, timeout=60))["features"]

    def fetch_station(t_ms):
        t0 = datetime.datetime.utcfromtimestamp(t_ms / 1000) + datetime.timedelta(seconds=60)
        s = t0.strftime("%Y-%m-%dT%H:%M:%S")
        e = (t0 + datetime.timedelta(seconds=1800)).strftime("%Y-%m-%dT%H:%M:%S")
        u = (f"https://service.iris.edu/irisws/timeseries/1/query?net=IU&sta={sta}"
             f"&loc=00&cha=BHZ&starttime={s}&endtime={e}&format=ascii1")
        try:
            d = urllib.request.urlopen(u, timeout=120).read().decode()
            return np.array(d.strip().split("\n")[1:], float)
        except Exception:
            return None

    X, y, g, meta = [], [], [], []
    gid = 0
    for lab, extra in (("shallow", "maxdepth=70"), ("deep", "mindepth=300")):
        got = 0
        for ev in usgs(extra):
            if got >= per_class:
                break
            x = fetch_station(ev["properties"]["time"])
            if x is None or len(x) < win_per_event * W + 2048:
                continue
            # rolling-RMS peak -> wavetrain block
            r = np.array([x[i:i + W].std() for i in range(0, len(x) - W, 512)])
            c = int(np.argmax(r)) * 512 + W // 2
            a = int(np.clip(c - win_per_event * W // 2, 0, len(x) - win_per_event * W))
            for k in range(win_per_event):
                s = detrend(np.ascontiguousarray(x[a + k * W:a + (k + 1) * W]))
                X.append((s - s.mean()) / (s.std() + 1e-12)); y.append(lab); g.append(gid)
            meta.append((lab, ev["properties"]["place"], ev["geometry"]["coordinates"][2],
                         ev["properties"]["time"]))
            gid += 1; got += 1
        print(f"seismic {sta} {lab}: {got} events", flush=True)
    np.savez_compressed(cache, X=np.array(X), y=np.array(y), g=np.array(g),
                        meta=np.array(meta, dtype=object))
    return [(X[i], y[i], g[i]) for i in range(len(y))]

IMS = "<local-path>/signalmap/data/ims/4. Bearings/IMS"
# ground truth (IMS readme): test1 bearing3=inner-race, bearing4=roller defect;
# test2 bearing1=outer race; test3 bearing3=outer race (0-indexed below)
IMS_FAILED = {("1st_test", 2), ("1st_test", 3), ("2nd_test", 0), ("4th_test", 2)}
# NOTE: the archive's 3rd_test.rar extracts to a dir named 4th_test/txt — that
# IS the documented "test 3" (bearing3 outer race); dir name kept as extracted.

def _ims_files(test):
    d = f"{IMS}/{test}"
    if os.path.isdir(f"{d}/txt"): d += "/txt"  # 3rd_test rar nests a txt/ dir
    return sorted(glob.glob(f"{d}/2*"))

def load_ims(task="fault", per_rec=4, win_per_file=4, sampling="even", pool=1):
    """IMS/NASA run-to-failure (Rexnord ZA-2115 bearings, 20kHz snapshots of 1s
    every 10min; test1: 8ch/2 per bearing, test2+3: 4ch/1 per bearing).
    task='fault': defective-vs-healthy bearing from the LAST tercile of each
      run (end of life). Recording = test x bearing (12 recs). Both classes
      coexist at IDENTICAL timestamps within a test -> rig-state/time confound
      neutralized by construction.
    task='stage': life-stage terciles (early/mid/late) of the rig experiment.
      Recording = test x bearing x stage (36 recs). CAVEAT (documented): all
      bearings of a test are time-synchronized -> held-out recording shares
      timestamps with training bearings; same-rig adjacency like HYD.
    sampling='peak'/pool=k: source-level readout variants (see _readout);
    pool=4 is the honest rebuild of the timescale-screen pool4 flag."""
    raw = []
    for test in ("1st_test", "2nd_test", "4th_test"):
        files = _ims_files(test)
        if not files: continue
        n_b = 4; col = (lambda b: 2 * b) if test == "1st_test" else (lambda b: b)
        blocks = ({"late": files[2 * len(files) // 3:]} if task == "fault" else
                  {"early": files[:len(files) // 3],
                   "mid": files[len(files) // 3:2 * len(files) // 3],
                   "late": files[2 * len(files) // 3:]})
        for stage, fs in blocks.items():
            picks = [fs[i] for i in np.linspace(0, len(fs) - 1, per_rec).astype(int)]
            sigs = [np.loadtxt(f) for f in picks]
            for b in range(n_b):
                lab = (("defect" if (test, b) in IMS_FAILED else "healthy")
                       if task == "fault" else stage)
                gid = f"{test[:3]}-b{b}" + ("" if task == "fault" else f"-{stage}")
                for x in sigs:
                    ch, offs = _readout(x[:, col(b)], win_per_file, sampling, pool)
                    for o in offs:
                        s = detrend(np.ascontiguousarray(ch[o:o + W]))
                        raw.append(((s - s.mean()) / (s.std() + 1e-12), lab, gid))
    return raw

def load_calce(per_stage=2):
    """6 cells x 3 SOH stages (life terciles). Recording = cell x stage block;
    signal = concatenated discharge-voltage curves resampled per cycle."""
    import pandas as pd
    raw, gid = [], 0
    def fdate(p):  # CS2_33_1_10_11.xlsx -> (2011, 1, 10); month_day_year in name
        parts = os.path.basename(p)[:-5].split("_")[2:]
        m, d, yy = (int(x) for x in parts[:3])
        return (2000 + yy, m, d)
    cells = sorted(glob.glob("<local-path>/signalmap/data/calce/CS2_3*/"))
    for cell in cells:
        files = sorted(glob.glob(cell + "*.xlsx"), key=fdate)
        if not files: continue
        n = len(files)
        stages = {"early": files[: max(n // 3, 1)],
                  "mid": files[n // 3: 2 * n // 3] or files[:1],
                  "late": files[2 * n // 3:] or files[-1:]}
        for stage, fs in stages.items():
            v = []
            for f in fs[:per_stage]:
                try:
                    xl = pd.ExcelFile(f)
                    sh = [s for s in xl.sheet_names if "Channel" in s]
                    df = xl.parse(sh[0] if sh else xl.sheet_names[-1])
                    vc = [c for c in df.columns if "Voltage" in str(c)]
                    if vc: v.append(df[vc[0]].to_numpy(float))
                except Exception:
                    continue
            if v:
                _push(np.concatenate(v), stage, gid, raw)
                gid += 1
    return raw
