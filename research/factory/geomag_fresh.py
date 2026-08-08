"""GEOMAG out-of-time fresh holdout (FRESH_DATA_SCAN #2). Bank ended Oct 2024.

Frozen per-observatory lean models (perm3+psd_slope, StandardScaler+RF(150,rs=0))
trained on the EXISTING cached bank (BOU/FRD storm-quiet npz) — no re-selection.
Fresh storm days = 13 Kp>=7 days Nov 2024 - Jul 2026 (GFZ-verified, scan Jul 3).
Fresh quiet days: threshold relaxed to Kp_max <= 1.0 (documented change: only 3
fresh days reach the old <=0.67 bar), picked from the GFZ Kp index BEFORE any
waveform is loaded (label acquisition, not signal data), excluding bank days.
Criterion (pre-registered): per obs, day-majority accuracy over fresh days,
cluster-bootstrap CI over days, PASS = CI-lo > 0.5. Secondary: window-level acc.
Windowing identical to load_geomag (8x1024 @1 sps at day's rolling-std peak, znorm).

Usage: pickdays -> prereg -> run   (.venv-research/bin/python geomag_fresh.py X)
"""
import hashlib, json, os, sys, urllib.request
from datetime import datetime

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_forge import lean_baseline
from harvest2_loaders import W, _znorm, load_geomag, STORM_DAYS, QUIET_DAYS
from receipt_ledger import log_receipt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "geomag_fresh_prereg.json")
CACHE = "<local-path>/signalmap/data/geomag/fresh"
OBS = ("BOU", "FRD")
WPD = 8

FRESH_STORM = ["2025-01-01", "2025-04-16", "2025-06-01", "2025-06-02",
               "2025-06-03", "2025-09-30", "2025-11-12", "2025-11-13",
               "2026-01-19", "2026-01-20", "2026-01-21", "2026-03-21",
               "2026-03-22"]  # Kp>=7, GFZ-verified (FRESH_DATA_SCAN Jul 3)


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def pickdays():
    u = ("https://kp.gfz.de/app/json/?start=2024-11-01T00:00:00Z"
         "&end=2026-07-01T23:59:59Z&index=Kp")
    d = json.load(urllib.request.urlopen(u, timeout=180))
    days = {}
    for t, v in zip(d["datetime"], d["Kp"]):
        if v is not None:
            days.setdefault(t[:10], []).append(v)
    quiet = sorted(day for day, vs in days.items()
                   if len(vs) == 8 and max(vs) <= 1.0
                   and day not in QUIET_DAYS and day not in FRESH_STORM)
    # spread over the period: pick every k-th to get ~13
    k = max(1, len(quiet) // 13)
    pick = quiet[::k][:13]
    json.dump({"all_quiet_le1": quiet, "picked": pick},
              open(os.path.join(FROZEN, "geomag_fresh_quietdays.json"), "w"), indent=1)
    print(f"{len(quiet)} quiet days Kp<=1.0; picked {len(pick)}: {pick}", flush=True)


def prereg():
    q = json.load(open(os.path.join(FROZEN, "geomag_fresh_quietdays.json")))["picked"]
    models = {}
    for obs in OBS:
        raw = load_geomag(obs=obs)  # existing cache, old bank
        L = lean_baseline(raw)
        y = np.array([r[1] for r in raw])
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(L, y)
        path = os.path.join(FROZEN, f"geomag_model_{obs.lower()}.joblib")
        joblib.dump(clf, path)
        models[obs] = {"n_train_windows": len(raw),
                       "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
        print(f"{obs}: frozen on {len(raw)} bank windows", flush=True)
    spec = {"prereg": "GEOMAG-FRESH out-of-time holdout (bank ended Oct 2024)",
            "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "family": "lean duo perm3+psd_slope, StandardScaler+RF(150,rs=0), per obs",
            "storm_days": FRESH_STORM, "quiet_days": q,
            "quiet_threshold": "Kp_max<=1.0 (relaxed from 0.67, documented)",
            "models": models,
            "criterion": "per obs: day-majority acc, cluster-bootstrap CI over days, PASS=CI-lo>0.5",
            "plan_sha256": _sha_text(json.dumps([FRESH_STORM, q]))}
    json.dump(spec, open(PREREG, "w"), indent=1)
    tip = log_receipt("GEOMAG-FRESH-PREREG", spec)
    print(f"prereg written. ledger tip = {tip}", flush=True)


def _fetch(obs, day):
    path = os.path.join(CACHE, f"{obs.lower()}_{day}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["X"] if z["X"].size else None
    u = (f"https://geomag.usgs.gov/ws/data/?id={obs}&starttime={day}T00:00:00Z"
         f"&endtime={day}T23:59:59Z&elements=X&sampling_period=1&format=json")
    try:
        d = json.load(urllib.request.urlopen(u, timeout=180))
        x = np.array([v if v is not None else np.nan
                      for v in d["values"][0]["values"]], float)
        x = x[~np.isnan(x)]
        assert len(x) >= (WPD + 2) * W, f"only {len(x)}"
    except Exception as e:
        print(f"  {obs} {day}: FETCH-FAIL {e}", flush=True)
        np.savez_compressed(path, X=np.array([]))
        return None
    r = np.array([x[i:i + W].std() for i in range(0, len(x) - W, 512)])
    c = int(np.argmax(r)) * 512 + W // 2
    a = int(np.clip(c - WPD * W // 2, 0, len(x) - WPD * W))
    X = np.array([_znorm(x[a + k * W:a + (k + 1) * W]) for k in range(WPD)])
    np.savez_compressed(path, X=X)
    return X


def run():
    assert os.path.exists(PREREG), "run prereg first"
    spec = json.load(open(PREREG))
    assert _sha_text(json.dumps([spec["storm_days"], spec["quiet_days"]])) == spec["plan_sha256"]
    os.makedirs(CACHE, exist_ok=True)
    results = {}
    for obs in OBS:
        clf = joblib.load(os.path.join(FROZEN, f"geomag_model_{obs.lower()}.joblib"))
        rows, win_ok, win_n = [], 0, 0
        for lab, days in (("storm", spec["storm_days"]), ("quiet", spec["quiet_days"])):
            for day in days:
                X = _fetch(obs, day)
                if X is None:
                    continue
                pred = clf.predict(lean_baseline([(x, "?", 0) for x in X]))
                maj = "storm" if (pred == "storm").sum() > WPD / 2 else "quiet"
                rows.append((day, lab, maj))
                win_ok += int((pred == lab).sum()); win_n += len(pred)
        correct = np.array([l == m for _, l, m in rows], float)
        acc = float(correct.mean())
        rng = np.random.default_rng(0)
        boots = [rng.choice(correct, len(correct)).mean() for _ in range(10000)]
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        s_acc = float(np.mean([m == "storm" for _, l, m in rows if l == "storm"]))
        q_acc = float(np.mean([m == "quiet" for _, l, m in rows if l == "quiet"]))
        verdict = "PASS" if lo > 0.5 else "FAIL"
        results[obs] = {"n_days": len(rows), "day_acc": acc, "ci": (float(lo), float(hi)),
                        "storm_acc": s_acc, "quiet_acc": q_acc,
                        "window_acc": win_ok / max(win_n, 1), "verdict": verdict}
        print(f"{obs}: n={len(rows)} day-acc={acc:.3f} CI[{lo:.3f},{hi:.3f}] "
              f"storm {s_acc:.3f} quiet {q_acc:.3f} win {win_ok/max(win_n,1):.3f} -> {verdict}",
              flush=True)
    tip = log_receipt("GEOMAG-FRESH", {"plan_sha256": spec["plan_sha256"], "results": results})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"pickdays": pickdays, "prereg": prereg, "run": run}[sys.argv[1]]()
