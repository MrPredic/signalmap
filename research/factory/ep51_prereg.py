"""PRE-REGISTRATION: prospective Episode-51 receipt (Kilauea, frozen readout).

Context (frozen Jul 3, 2026): USGS-HVO update Jul 2, 9:33 HST — Episode 50
ended Jun 27 (7 h fountaining); Episode 51 FORECAST Jul 8-15, 2026 (summit
deflation raised timing uncertainty). Episode 51 has NOT occurred yet.

This script freezes, BEFORE the episode exists:
  - the family: lean duo (perm_entropy order=3 normalized, psd_slope) — the
    RESULTS.md VOLCANO champion (UWE 0.767 perm-p 0.033, RIMD 0.773, CLF-robust)
  - per-station models (StandardScaler + RF(150, random_state=0)) fitted on the
    FULL existing 2018-2023 bank caches (no fresh data touched at freeze time)
  - the exact fetch + windowing rule for the future episode (identical to
    harvest3_loaders.load_volcano: IRIS HV <sta> HHZ 100 sps, 30-min segments,
    8x1024 windows at the rolling-RMS peak, z-norm)
  - the success criterion (below), so no post-hoc selection is possible
  - SHA-256 of spec, models and bank caches -> receipt_ledger hash chain.

PRE-REGISTERED TEST PROTOCOL (apply mode, run only AFTER HVO documents Ep 51):
  Eruptive segments: 30-min segments starting episode_start + 60 min, then every
    60 min while segment end <= episode_end, max K=6.
  Quiet segments: same clock times (UTC) as each eruptive segment, on pause
    reference days episode_start_date - 3d (odd idx: - 4d); all must lie >= 48 h
    after Episode-50 end (Jun 27) and >= 24 h before Ep-51 start.
  Per-segment verdict = majority vote of the frozen model over its 8 windows.
  PRIMARY CRITERION (per station): >= ceil(0.75*K) eruptive segments majority
    "eruptive" AND >= ceil(0.75*K) quiet segments majority "quiet".
    Both stations PASS -> full prospective receipt; one -> partial.
  PRE-REGISTERED CAVEAT: 2024-26 inter-episode pauses carry documented
    inflation/unrest; if eruptive detection passes but quiet control fails, the
    honest verdict is "reads unrest regime, not episode on/off" — reported as
    such, no re-selection, no criterion change.

Usage:
  .venv-research/bin/python ep51_prereg.py freeze
  .venv-research/bin/python ep51_prereg.py apply --start 2026-07-XXTHH:MM \
      --end 2026-07-XXTHH:MM        # HVO-documented episode bounds, UTC
"""
import argparse, hashlib, json, os, sys, urllib.request
from datetime import datetime, timedelta

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_forge import lean_baseline
from harvest3_loaders import W, _znorm, load_volcano
from receipt_ledger import log_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
SPEC = os.path.join(FROZEN, "ep51_spec.json")
VOLC = "<local-path>/signalmap/data/volcano"
STATIONS = ("UWE", "RIMD")
K_MAX = 6
WIN_PER_SEG = 8
EP50_END_UTC = "2026-06-28T03:10"  # Jun 27 17:10 HST


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_segment(sta, t0_iso):
    t0 = datetime.fromisoformat(t0_iso)
    t1 = t0 + timedelta(minutes=30)
    u = (f"https://service.iris.edu/irisws/timeseries/1/query?net=HV&sta={sta}"
         f"&loc=--&cha=HHZ&starttime={t0.strftime('%Y-%m-%dT%H:%M:%S')}"
         f"&endtime={t1.strftime('%Y-%m-%dT%H:%M:%S')}&format=ascii1")
    d = urllib.request.urlopen(u, timeout=180).read().decode()
    x = np.array(d.strip().split("\n")[1:], float)
    if len(x) < (WIN_PER_SEG + 2) * W:
        raise ValueError(f"only {len(x)} samples")
    # identical windowing to load_volcano: 8x1024 at rolling-RMS peak
    r = np.array([x[i:i + W].std() for i in range(0, len(x) - W, 512)])
    c = int(np.argmax(r)) * 512 + W // 2
    a = int(np.clip(c - WIN_PER_SEG * W // 2, 0, len(x) - WIN_PER_SEG * W))
    return [_znorm(x[a + k * W:a + (k + 1) * W]) for k in range(WIN_PER_SEG)]


def freeze():
    os.makedirs(FROZEN, exist_ok=True)
    models = {}
    for sta in STATIONS:
        raw = load_volcano(sta=sta)  # cached npz, 2018-2023 bank — nothing fresh
        L = lean_baseline(raw)
        y = np.array([r[1] for r in raw])
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(L, y)
        path = os.path.join(FROZEN, f"ep51_model_{sta.lower()}.joblib")
        joblib.dump(clf, path)
        acc = float((clf.predict(L) == y).mean())  # in-sample sanity only
        models[sta] = {"path": os.path.relpath(path, HERE), "n_windows": len(raw),
                       "n_days": len(set(r[2] for r in raw)),
                       "insample_sanity": acc, "sha256": _sha(path)}
        print(f"{sta}: {len(raw)} windows, in-sample sanity {acc:.3f}", flush=True)
    spec = {
        "prereg": "VOLCANO-EP51 prospective receipt",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "context": "HVO Jul 2 2026: Ep50 ended Jun 27; Ep51 forecast Jul 8-15; not yet occurred",
        "family": "lean duo: perm_entropy(order=3,normalize=True) + psd_slope(welch nperseg<=256, loglog fit)",
        "model": "StandardScaler + RandomForestClassifier(150, random_state=0), per station",
        "stations": list(STATIONS),
        "fetch_rule": "IRIS HV <sta> -- HHZ ascii1, 30-min segments, 8x1024 @100sps at rolling-RMS peak, znorm (== load_volcano)",
        "eruptive_segments": f"start=ep_start+60min, hourly while seg_end<=ep_end, max K={K_MAX}",
        "quiet_segments": "same UTC clock times, ep_start_date -3d (odd idx -4d); >=48h after Ep50 end, >=24h before Ep51 start",
        "segment_verdict": "majority vote over 8 windows",
        "criterion": "per station: >=ceil(0.75*K) eruptive segs majority-eruptive AND >=ceil(0.75*K) quiet segs majority-quiet; both stations=full, one=partial",
        "preregistered_caveat": "if eruptive passes but quiet control fails -> verdict 'reads unrest regime, not episode on/off'; no re-selection",
        "models": models,
        "bank_caches_sha256": {s: _sha(f"{VOLC}/kilauea_{s.lower()}.npz") for s in STATIONS},
    }
    with open(SPEC, "w") as f:
        json.dump(spec, f, indent=1, sort_keys=True)
    tip = log_receipt("VOLCANO-EP51-PREREG", {**spec, "spec_sha256": _sha(SPEC)})
    print(f"frozen. spec={SPEC}\nledger tip = {tip}", flush=True)


def apply_(start, end):
    with open(SPEC) as f:
        spec = json.load(f)
    t_start = datetime.fromisoformat(start)
    t_end = datetime.fromisoformat(end)
    assert t_start > datetime.fromisoformat(EP50_END_UTC) + timedelta(hours=48), \
        "start implausible (inside Ep50 window?)"
    # eruptive segment start times
    segs = []
    t = t_start + timedelta(minutes=60)
    while t + timedelta(minutes=30) <= t_end and len(segs) < K_MAX:
        segs.append(t)
        t += timedelta(minutes=60)
    K = len(segs)
    assert K >= 2, f"episode too short for the protocol (K={K})"
    need = int(np.ceil(0.75 * K))
    quiet = [datetime.combine((t_start - timedelta(days=3 + (i % 2))).date(), s.time())
             for i, s in enumerate(segs)]
    for q in quiet:
        assert q > datetime.fromisoformat(EP50_END_UTC) + timedelta(hours=48)
        assert q + timedelta(minutes=30) < t_start - timedelta(hours=24)
    print(f"K={K} segments/class, criterion >= {need}/{K} per class per station", flush=True)
    results = {}
    for sta in STATIONS:
        clf = joblib.load(os.path.join(HERE, spec["models"][sta]["path"]))
        assert _sha(os.path.join(HERE, spec["models"][sta]["path"])) == \
            spec["models"][sta]["sha256"], "model file tampered"
        hits = {"eruptive": 0, "quiet": 0}
        detail = []
        for lab, times in (("eruptive", segs), ("quiet", quiet)):
            for t0 in times:
                try:
                    wins = _fetch_segment(sta, t0.isoformat())
                except Exception as e:
                    detail.append((lab, t0.isoformat(), f"FETCH-FAIL {e}"))
                    print(f"  {sta} {lab} {t0}: FETCH-FAIL {e}", flush=True)
                    continue
                pred = clf.predict(lean_baseline([(w, "?", 0) for w in wins]))
                maj = "eruptive" if (pred == "eruptive").sum() > WIN_PER_SEG / 2 else "quiet"
                hits[lab] += int(maj == lab)
                detail.append((lab, t0.isoformat(), maj,
                               f"{(pred == 'eruptive').sum()}/{WIN_PER_SEG} eruptive-votes"))
                print(f"  {sta} {lab} {t0}: majority={maj}", flush=True)
        ok_e, ok_q = hits["eruptive"] >= need, hits["quiet"] >= need
        results[sta] = {"K": K, "need": need, "eruptive_correct": hits["eruptive"],
                        "quiet_correct": hits["quiet"],
                        "pass": bool(ok_e and ok_q),
                        "unrest_fallback": bool(ok_e and not ok_q), "detail": detail}
        print(f"{sta}: eruptive {hits['eruptive']}/{K}, quiet {hits['quiet']}/{K} "
              f"-> {'PASS' if ok_e and ok_q else 'unrest-fallback' if ok_e else 'FAIL'}",
              flush=True)
    n_pass = sum(r["pass"] for r in results.values())
    verdict = {2: "FULL prospective receipt", 1: "PARTIAL (one station)"}.get(n_pass, "FAIL")
    if n_pass == 0 and all(r["unrest_fallback"] for r in results.values()):
        verdict = "unrest-regime readout (pre-registered caveat)"
    print(f"VERDICT: {verdict}", flush=True)
    tip = log_receipt("VOLCANO-EP51-APPLY", {"episode": {"start": start, "end": end},
                                             "results": results, "verdict": verdict,
                                             "spec_sha256": _sha(SPEC)})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "apply"])
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    if a.mode == "freeze":
        freeze()
    else:
        assert a.start and a.end, "apply needs --start/--end (UTC, HVO-documented)"
        apply_(a.start, a.end)
