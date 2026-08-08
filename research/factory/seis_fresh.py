"""SEIS backward-fresh holdout (FRESH_DATA_SCAN #3). Bank = 2024-01..2026-06.

Backward period 2021-2023 (150 shallow / 19 deep M6.3+ verified) = untouched.
Frozen per-station lean models (perm3+psd_slope, StandardScaler+RF(150,rs=0))
trained on the EXISTING cached banks (ANMO, KONO) — no re-selection, no re-fit.
Protocol identical to load_seismic_depth: USGS orderby=magnitude M6.3+,
shallow<70km vs deep>300km, IU.<sta>.00.BHZ 40 sps, 30 min from origin+60s,
8x1024 block at rolling-RMS peak, znorm; event = recording.
Criterion (pre-registered): per station, event-majority accuracy over fresh
events, cluster-bootstrap CI over events, PASS = CI-lo > 0.5.

Usage: prereg -> run   (.venv-research/bin/python seis_fresh.py X)
"""
import datetime, hashlib, json, os, sys, urllib.request

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_forge import lean_baseline
from retro_loaders import W, load_seismic_depth
from scipy.signal import detrend
from receipt_ledger import log_receipt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "seis_fresh_prereg.json")
CACHE = "<local-path>/signalmap/data/seismic/fresh_backward"
STATIONS = ("ANMO", "KONO")
PER_CLASS = 16
WPE = 8


def _znorm(seg):
    s = detrend(np.ascontiguousarray(seg, float))
    return (s - s.mean()) / (s.std() + 1e-12)


def prereg():
    models = {}
    for sta in STATIONS:
        raw = load_seismic_depth(sta=sta)  # existing cache, 2024-26 bank
        L = lean_baseline(raw)
        y = np.array([r[1] for r in raw])
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(L, y)
        path = os.path.join(FROZEN, f"seis_model_{sta.lower()}.joblib")
        joblib.dump(clf, path)
        models[sta] = {"n_train_windows": len(raw),
                       "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
        print(f"{sta}: frozen on {len(raw)} bank windows", flush=True)
    spec = {"prereg": "SEIS-BACKWARD fresh holdout 2021-2023 (bank=2024-26)",
            "frozen_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "family": "lean duo perm3+psd_slope, StandardScaler+RF(150,rs=0), per station",
            "event_query": "USGS fdsnws 2021-01-01..2024-01-01 M6.3+ orderby=magnitude, "
                           f"shallow maxdepth=70 / deep mindepth=300, per_class={PER_CLASS}",
            "models": models,
            "criterion": "per station: event-majority acc, cluster-bootstrap CI over events, PASS=CI-lo>0.5"}
    json.dump(spec, open(PREREG, "w"), indent=1)
    tip = log_receipt("SEIS-BACKWARD-PREREG", spec)
    print(f"prereg written. ledger tip = {tip}", flush=True)


def _events(extra):
    u = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
         "&starttime=2021-01-01&endtime=2024-01-01&minmagnitude=6.3"
         f"&orderby=magnitude&limit={PER_CLASS}&" + extra)
    return json.load(urllib.request.urlopen(u, timeout=60))["features"]


def _fetch(sta, t_ms):
    path = os.path.join(CACHE, f"{sta.lower()}_{t_ms}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["X"] if z["X"].size else None
    t0 = datetime.datetime.utcfromtimestamp(t_ms / 1000) + datetime.timedelta(seconds=60)
    s = t0.strftime("%Y-%m-%dT%H:%M:%S")
    e = (t0 + datetime.timedelta(seconds=1800)).strftime("%Y-%m-%dT%H:%M:%S")
    u = (f"https://service.iris.edu/irisws/timeseries/1/query?net=IU&sta={sta}"
         f"&loc=00&cha=BHZ&starttime={s}&endtime={e}&format=ascii1")
    try:
        d = urllib.request.urlopen(u, timeout=120).read().decode()
        x = np.array(d.strip().split("\n")[1:], float)
        assert len(x) >= WPE * W + 2048, f"only {len(x)}"
    except Exception as ex:
        print(f"  {sta} {t_ms}: FETCH-FAIL {ex}", flush=True)
        np.savez_compressed(path, X=np.array([]))
        return None
    r = np.array([x[i:i + W].std() for i in range(0, len(x) - W, 512)])
    c = int(np.argmax(r)) * 512 + W // 2
    a = int(np.clip(c - WPE * W // 2, 0, len(x) - WPE * W))
    X = np.array([_znorm(x[a + k * W:a + (k + 1) * W]) for k in range(WPE)])
    np.savez_compressed(path, X=X)
    return X


def run():
    assert os.path.exists(PREREG), "run prereg first"
    os.makedirs(CACHE, exist_ok=True)
    evs = [(lab, ev["properties"]["time"], ev["properties"]["mag"])
           for lab, extra in (("shallow", "maxdepth=70"), ("deep", "mindepth=300"))
           for ev in _events(extra)]
    print(f"{sum(l=='shallow' for l,_,_ in evs)} shallow / "
          f"{sum(l=='deep' for l,_,_ in evs)} deep events", flush=True)
    results = {}
    for sta in STATIONS:
        clf = joblib.load(os.path.join(FROZEN, f"seis_model_{sta.lower()}.joblib"))
        rows = []
        for lab, t_ms, mag in evs:
            X = _fetch(sta, t_ms)
            if X is None:
                continue
            pred = clf.predict(lean_baseline([(x, "?", 0) for x in X]))
            maj = "shallow" if (pred == "shallow").sum() > WPE / 2 else "deep"
            rows.append((lab, maj))
            print(f"  {sta} {lab} M{mag}: majority={maj}", flush=True)
        correct = np.array([l == m for l, m in rows], float)
        acc = float(correct.mean())
        rng = np.random.default_rng(0)
        boots = [rng.choice(correct, len(correct)).mean() for _ in range(10000)]
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        sh = float(np.mean([m == "shallow" for l, m in rows if l == "shallow"]))
        de = float(np.mean([m == "deep" for l, m in rows if l == "deep"]))
        verdict = "PASS" if lo > 0.5 else "FAIL"
        results[sta] = {"n_events": len(rows), "acc": acc, "ci": (float(lo), float(hi)),
                        "shallow_acc": sh, "deep_acc": de, "verdict": verdict}
        print(f"{sta}: n={len(rows)} acc={acc:.3f} CI[{lo:.3f},{hi:.3f}] "
              f"shallow {sh:.3f} deep {de:.3f} -> {verdict}", flush=True)
    tip = log_receipt("SEIS-BACKWARD", {"results": results})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "run": run}[sys.argv[1]]()
