"""LIGO-Glitches validation bank (Jul 11) — Physics #new: transient-noise MORPHOLOGY.

Task: classify a short (~4s) H1/L1 strain segment as glitch class Blip vs
Scattered_Light, read from raw strain texture around a documented GPS time.
Labels = Gravity Spy ML classifications (public, Zenodo record 5649212,
DOI 10.5281/zenodo.5649212), restricted to O1 (2015-09..2016-01, single run
-> no cross-run instrument-config confound) and ml_confidence >= 0.9 (clean
labels). Strain = official GWOSC open data (https://gwosc.org), 4096 Hz,
GDS-CALIB_STRAIN-equivalent public release channel.

Recording = one glitch event (fixed GPS center, one detector). Multiple
windows (W=1024 samples = 0.25s @ 4096 Hz) are cut from the ~4s segment
around the event center -> recording-level LOGO in gauntlet.py is leakage-
free (a whole event, not a window, is held out).

Design notes (lessons from harvest3 baked in):
  - BOTH detectors (H1, L1) contribute both classes -> detector is not a
    label confound (label = glitch class only; detector kept in provenance/
    log for reference, never used as y).
  - Event selection: within each (detector, class) group, pick the N
    sorted-by-time events with the SMALLEST GPS time-span (sliding window
    of size N over sorted event_time) -> events cluster into few distinct
    4096s GWOSC archive files, minimizing raw downloads while keeping the
    selection a simple, pre-registered rule (not cherry-picked by hand).
  - Raw 4096s hdf5 files are cached under cache/ligo/raw/ and reused across
    events that happen to share a file; every unique raw file + every
    metadata CSV is SHA256'd into provenance.json for reproducibility.
  - No synthetic data anywhere: if GWOSC/Zenodo is unreachable or the event
    segment falls outside the fetched file's range, that event is SKIPPED
    and logged -- never faked.
"""
import os, csv, json, hashlib, time, urllib.request
import numpy as np
from scipy.signal import detrend

W = 1024                     # samples per window (0.25 s @ 4096 Hz)
FS = 4096                    # native GWOSC sample rate used
SEG_HALF = 2.0                # seconds each side of event center -> ~4s segment
WIN_PER_EVENT = 8
TARGET_N = 8                  # events per (detector, class) group
CONF_THRESH = 0.9
DATASET = "O1"
DETECTORS = ("H1", "L1")
CLASSES = ("Blip", "Scattered_Light")

BASE = "<local-path>/signalmap/research/factory/cache/ligo"
RAW = f"{BASE}/raw"
CACHE = f"{BASE}/ligo_bank.npz"
PROV = f"{BASE}/provenance.json"

ZENODO_RECORD = "5649212"
CSV_URL = ("https://zenodo.org/api/records/{rec}/files/{ifo}_{dset}.csv/content")
LINKS_URL = "https://gwosc.org/archive/links/{dset}/{ifo}/{s}/{e}/json/"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dest, timeout=180):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "signalmap-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def _fetch_metadata_csv(ifo, prov, errors, classes=CLASSES):
    """Download (or reuse) the Gravity Spy ML-classification CSV for one
    detector/O1, return list of (event_time, ml_label, ml_confidence)."""
    dest = f"{RAW}/{ifo}_{DATASET}_meta.csv"
    url = CSV_URL.format(rec=ZENODO_RECORD, ifo=ifo, dset=DATASET)
    if not os.path.exists(dest):
        try:
            print(f"  downloading metadata {ifo}_{DATASET}.csv ...", flush=True)
            _download(url, dest, timeout=300)
        except Exception as e:
            errors.append(f"metadata CSV {ifo}_{DATASET}: {url} -> {e}")
            return []
    sha = _sha256_file(dest)
    prov["files"][os.path.basename(dest)] = {
        "url": url, "sha256": sha, "bytes": os.path.getsize(dest), "role": "gravityspy_metadata_csv"}
    rows = []
    with open(dest, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            lab = row.get("ml_label")
            if lab not in classes:
                continue
            try:
                conf = float(row["ml_confidence"])
                t = float(row["event_time"])
                gid = row.get("gravityspy_id", "")
            except Exception:
                continue
            if conf >= CONF_THRESH:
                rows.append((t, lab, conf, gid))
    return rows


def _pick_dense_window(times, n):
    """Smallest GPS time-span containing exactly n sorted events. Returns
    the index range [i, i+n) that minimizes times[i+n-1]-times[i]."""
    times = sorted(times)
    if len(times) < n:
        return None
    best_i, best_span = 0, float("inf")
    for i in range(len(times) - n + 1):
        span = times[i + n - 1] - times[i]
        if span < best_span:
            best_i, best_span = i, span
    return best_i, best_span


def _locate_url(ifo, gps_center):
    s, e = int(gps_center - SEG_HALF) - 1, int(gps_center + SEG_HALF) + 1
    url = LINKS_URL.format(dset=DATASET, ifo=ifo, s=s, e=e)
    req = urllib.request.Request(url, headers={"User-Agent": "signalmap-research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        j = json.loads(r.read().decode())
    for entry in j.get("strain", []):
        if entry.get("format") == "hdf5" and entry.get("sampling_rate") == FS:
            return entry["url"], entry["GPSstart"], entry["duration"]
    return None, None, None


def _znorm(seg):
    s = detrend(np.ascontiguousarray(seg, float))
    return (s - s.mean()) / (s.std() + 1e-12)


def _extract_windows(h5, gps_center, x_start, dt):
    import h5py
    ds = h5["strain"]["Strain"]
    n_seg = int(round(2 * SEG_HALF * FS))
    c_idx = int(round((gps_center - x_start) / dt))
    a = c_idx - n_seg // 2
    b = a + n_seg
    if a < 0 or b > ds.shape[0]:
        return None
    seg = ds[a:b]
    starts = np.linspace(0, n_seg - W, WIN_PER_EVENT).astype(int)
    return [_znorm(seg[s0:s0 + W]) for s0 in starts]


def load_ligo(refetch=False, classes=CLASSES, tag=""):
    """Returns list of (window[float64,W], label, recording_gid) triples,
    same convention as harvest3_loaders.load_volcano. Caches to CACHE npz;
    provenance (URLs, SHA256, event log) written to PROV every build.
    classes/tag: replication pairs get their own bank cache + provenance
    (tag e.g. "_koi_whistle"); defaults reproduce the original Blip-vs-
    Scattered_Light bank byte-identically."""
    cache = CACHE.replace(".npz", f"{tag}.npz") if tag else CACHE
    provp = PROV.replace(".json", f"{tag}.json") if tag else PROV
    if os.path.exists(cache) and not refetch:
        z = np.load(cache, allow_pickle=True)
        return [(z["X"][i], str(z["y"][i]), int(z["g"][i])) for i in range(len(z["y"]))]

    os.makedirs(RAW, exist_ok=True)
    prov = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": DATASET, "classes": list(classes), "detectors": list(DETECTORS),
            "conf_thresh": CONF_THRESH, "target_n_per_group": TARGET_N,
            "win_per_event": WIN_PER_EVENT, "W": W, "fs": FS,
            "zenodo_record": ZENODO_RECORD, "files": {}, "events": [], "errors": []}
    errors = prov["errors"]

    X, y, g = [], [], []
    gid = 0
    for ifo in DETECTORS:
        rows = _fetch_metadata_csv(ifo, prov, errors, classes)
        if not rows:
            continue
        for lab in classes:
            times = [t for (t, l, c, gsid) in rows if l == lab]
            meta_by_t = {t: (c, gsid) for (t, l, c, gsid) in rows if l == lab}
            picked = _pick_dense_window(times, TARGET_N)
            if picked is None:
                errors.append(f"{ifo}/{lab}: only {len(times)} events >= conf {CONF_THRESH}, need {TARGET_N}")
                continue
            i0, span = picked
            sel_times = sorted(times)[i0:i0 + TARGET_N]
            print(f"  {ifo}/{lab}: {len(times)} candidates, picked {TARGET_N} "
                  f"spanning {span:.0f}s", flush=True)
            for t in sel_times:
                conf, gsid = meta_by_t[t]
                try:
                    url, x_start, dur = _locate_url(ifo, t)
                except Exception as e:
                    errors.append(f"{ifo}/{lab} gps={t}: links API -> {e}")
                    continue
                if url is None:
                    errors.append(f"{ifo}/{lab} gps={t}: no 4096Hz hdf5 entry in links response")
                    continue
                dest = f"{RAW}/{os.path.basename(url)}"
                if not os.path.exists(dest):
                    try:
                        print(f"    downloading {os.path.basename(url)} ...", flush=True)
                        _download(url, dest)
                    except Exception as e:
                        errors.append(f"{ifo}/{lab} gps={t}: download {url} -> {e}")
                        continue
                sha = _sha256_file(dest)
                prov["files"].setdefault(os.path.basename(dest), {
                    "url": url, "sha256": sha, "bytes": os.path.getsize(dest),
                    "role": "gwosc_strain_hdf5"})
                import h5py
                try:
                    with h5py.File(dest, "r") as h5:
                        dt = float(h5["strain"]["Strain"].attrs["Xspacing"])
                        wins = _extract_windows(h5, t, x_start, dt)
                except Exception as e:
                    errors.append(f"{ifo}/{lab} gps={t}: hdf5 read -> {e}")
                    continue
                if wins is None:
                    errors.append(f"{ifo}/{lab} gps={t}: segment out of file range")
                    continue
                for w in wins:
                    X.append(w); y.append(lab); g.append(gid)
                prov["events"].append({"gid": gid, "ifo": ifo, "label": lab,
                                        "gps": t, "ml_confidence": conf,
                                        "gravityspy_id": gsid, "file": os.path.basename(dest)})
                gid += 1

    if X:
        np.savez_compressed(cache, X=np.array(X), y=np.array(y), g=np.array(g))
    with open(provp, "w") as f:
        json.dump(prov, f, indent=2)
    if errors:
        print(f"  [{len(errors)} errors logged in {PROV}]", flush=True)
    return [(X[i], y[i], g[i]) for i in range(len(y))]


if __name__ == "__main__":
    raw = load_ligo(refetch=True)
    print(f"LIGO: {len(raw)} windows, {len(set(r[2] for r in raw))} recordings, "
          f"classes={sorted(set(r[1] for r in raw))}", flush=True)
