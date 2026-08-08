"""PRE-REGISTRATION 2: prospective Episode-51 receipt, EPISODE-TRAINED model.

Context (frozen Jul 4, 2026): USGS-HVO update Jul 3, 8:30 HST — Kilauea paused
since Ep-50 end (Jun 27); Episode 51 FORECAST Jul 9-15, NOT yet occurred
(inflation resumed Jul 2). Prereg-1 (commit 17eb13d, ledger 8642426b) stays
FROZEN and will be evaluated exactly as registered — this file never touches
its models or spec.

Why a second prereg is legitimate: the fresh-OOT run (ledger de943693) showed
the 2018-23-trained frozen model reads the NEW episodic-fountaining regime
weakly (UWE eruptive 0.542) while pause texture is stable (quiet 0.978).
As long as Ep 51 has not occurred, a model trained on the 2024-26 EPISODE bank
(cached fresh_oot segments, labels from volcano_fresh.plan()) is still fully
prospective w.r.t. Ep 51. Registered EXPECTATION: episode-trained should read
eruptive clearly better than 0.542; if it does not, that is reported as-is.

No selection happens here: family (lean duo), model class (StandardScaler +
RF(150, random_state=0)) and the apply protocol are copied verbatim from
Prereg-1. LOGO over episodes (episode = group) is run BEFORE the freeze as a
sanity receipt only — its result is recorded in the spec, whatever it is.

PRE-REGISTERED TEST PROTOCOL (apply mode == Prereg-1 protocol, identical
segment rule, majority vote, criterion >= ceil(0.75*K) per class per station;
unrest-fallback caveat carried over) with ONE addition: every apply quiet
segment must be disjoint from every training-bank segment window (30-min,
timestamp check) — else it is dropped and reported (no silent reuse of
training minutes as test minutes).

Usage:
  .venv-research/bin/python ep51_prereg2.py logo     # sanity, cached only
  .venv-research/bin/python ep51_prereg2.py freeze   # logo + fit + ledger
  .venv-research/bin/python ep51_prereg2.py apply --start 2026-07-XXTHH:MM \
      --end 2026-07-XXTHH:MM        # HVO-documented episode bounds, UTC
"""
import argparse, hashlib, json, os, sys
from datetime import datetime, timedelta

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ep51_prereg import EP50_END_UTC, K_MAX, WIN_PER_SEG, _fetch_segment, _sha
from feature_forge import lean_baseline
from receipt_ledger import log_receipt
from volcano_fresh import CACHE, STATIONS, plan

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
SPEC = os.path.join(FROZEN, "ep51p2_spec.json")


def _load_bank(sta):
    """Fresh 2024-26 episode bank from EXISTING caches only (no network).

    Returns raw list of (window, label, episode) tuples — lean_baseline format.
    """
    raw, seg_times = [], []
    for item in plan():
        for lab, t in (("eruptive", item["eruptive"]), ("quiet", item["quiet"])):
            if t is None:
                continue
            path = os.path.join(CACHE, f"{sta}_{t.replace(':', '')}.npz")
            assert os.path.exists(path), f"cache missing (would fetch fresh): {path}"
            X = np.load(path)["X"]
            if X.size == 0:  # documented fetch-fail from the OOT run
                continue
            raw += [(x, lab, item["ep"]) for x in X]
            seg_times.append(t)
    return raw, seg_times


def _fit(L, y):
    clf = make_pipeline(StandardScaler(),
                        RandomForestClassifier(150, random_state=0, n_jobs=-1))
    clf.fit(L, y)
    return clf


def logo(report=True):
    """Leave-one-episode-out sanity on the cached fresh bank (episode=group)."""
    out = {}
    for sta in STATIONS:
        raw, _ = _load_bank(sta)
        L = lean_baseline(raw)
        y = np.array([r[1] for r in raw])
        g = np.array([r[2] for r in raw])
        seg_hits = {"eruptive": [0, 0], "quiet": [0, 0]}
        for ep in np.unique(g):
            tr, te = g != ep, g == ep
            clf = _fit(L[tr], y[tr])
            pred = clf.predict(L[te])
            for lab in ("eruptive", "quiet"):
                m = y[te] == lab
                if not m.any():
                    continue
                # windows arrive in WIN_PER_SEG blocks per segment
                pw, n = pred[m], int(m.sum()) // WIN_PER_SEG
                for k in range(n):
                    blk = pw[k * WIN_PER_SEG:(k + 1) * WIN_PER_SEG]
                    maj = "eruptive" if (blk == "eruptive").sum() > WIN_PER_SEG / 2 else "quiet"
                    seg_hits[lab][0] += int(maj == lab)
                    seg_hits[lab][1] += 1
        e, q = seg_hits["eruptive"], seg_hits["quiet"]
        out[sta] = {"logo_eruptive": f"{e[0]}/{e[1]}", "logo_quiet": f"{q[0]}/{q[1]}",
                    "logo_eruptive_acc": round(e[0] / e[1], 3),
                    "logo_quiet_acc": round(q[0] / q[1], 3),
                    "logo_acc": round((e[0] + q[0]) / (e[1] + q[1]), 3)}
        if report:
            print(f"{sta} LOGO(episode): eruptive {e[0]}/{e[1]} quiet {q[0]}/{q[1]} "
                  f"-> acc {out[sta]['logo_acc']}", flush=True)
    return out


def freeze():
    os.makedirs(FROZEN, exist_ok=True)
    sanity = logo()
    models, train_times = {}, {}
    for sta in STATIONS:
        raw, seg_times = _load_bank(sta)
        L = lean_baseline(raw)
        y = np.array([r[1] for r in raw])
        clf = _fit(L, y)
        path = os.path.join(FROZEN, f"ep51p2_model_{sta.lower()}.joblib")
        joblib.dump(clf, path)
        acc = float((clf.predict(L) == y).mean())  # in-sample sanity only
        models[sta] = {"path": os.path.relpath(path, HERE),
                       "n_windows": len(raw), "n_segments": len(seg_times),
                       "insample_sanity": acc, "sha256": _sha(path), **sanity[sta]}
        train_times[sta] = seg_times
        print(f"{sta}: {len(raw)} windows / {len(seg_times)} segments, "
              f"in-sample {acc:.3f}", flush=True)
    p = plan()
    spec = {
        "prereg": "VOLCANO-EP51 prospective receipt #2 (episode-trained)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "context": "HVO Jul 3 2026: paused since Ep50 end Jun 27; Ep51 forecast Jul 9-15; inflation resumed Jul 2; Ep51 NOT occurred",
        "relation_to_prereg1": "Prereg-1 (17eb13d) untouched, evaluated as registered; this is an additional prospective receipt, motivated by fresh-OOT regime-shift finding (ledger de943693)",
        "expectation": "episode-trained model should read eruptive clearly > 0.542 (frozen-2018-23 value); reported honestly either way",
        "training_bank": "fresh_oot caches, 50 episodes Dec 2024 - Jun 2026, labels from volcano_fresh.plan(), fetch-fails excluded",
        "family": "lean duo: perm_entropy(order=3,normalize=True) + psd_slope (== Prereg-1)",
        "model": "StandardScaler + RandomForestClassifier(150, random_state=0), per station (== Prereg-1)",
        "stations": list(STATIONS),
        "apply_protocol": "identical to Prereg-1 (ep51_prereg.py): eruptive segs hourly from start+60min (max K=6), quiet same clock times -3d/-4d, majority over 8 windows, criterion >=ceil(0.75*K) per class per station; unrest-fallback caveat",
        "apply_addition": "apply quiet segments overlapping any training segment window are dropped and reported",
        "plan_sha256": hashlib.sha256(json.dumps(p).encode()).hexdigest(),
        "models": models,
        "train_segment_times": train_times,
        "bank_caches_sha256": {os.path.basename(f): _sha(os.path.join(CACHE, f))
                               for f in sorted(os.listdir(CACHE)) if f.endswith(".npz")},
    }
    with open(SPEC, "w") as f:
        json.dump(spec, f, indent=1, sort_keys=True)
    tip = log_receipt("VOLCANO-EP51-PREREG2", {**spec, "spec_sha256": _sha(SPEC)})
    print(f"frozen. spec={SPEC}\nledger tip = {tip}", flush=True)


def apply_(start, end):
    with open(SPEC) as f:
        spec = json.load(f)
    t_start, t_end = datetime.fromisoformat(start), datetime.fromisoformat(end)
    assert t_start > datetime.fromisoformat(EP50_END_UTC) + timedelta(hours=48), \
        "start implausible (inside Ep50 window?)"
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
        mpath = os.path.join(HERE, spec["models"][sta]["path"])
        assert _sha(mpath) == spec["models"][sta]["sha256"], "model file tampered"
        clf = joblib.load(mpath)
        train_iv = [(datetime.fromisoformat(x), datetime.fromisoformat(x) + timedelta(minutes=30))
                    for x in spec["train_segment_times"][sta]]
        hits, detail = {"eruptive": 0, "quiet": 0}, []
        n_q = {"eruptive": K, "quiet": K}
        for lab, times in (("eruptive", segs), ("quiet", quiet)):
            for t0 in times:
                t1 = t0 + timedelta(minutes=30)
                if lab == "quiet" and any(t0 < b and a < t1 for a, b in train_iv):
                    detail.append((lab, t0.isoformat(), "DROPPED train-overlap"))
                    n_q[lab] -= 1
                    print(f"  {sta} {lab} {t0}: DROPPED (train-overlap)", flush=True)
                    continue
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
                        "quiet_correct": hits["quiet"], "n_quiet_kept": n_q["quiet"],
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
    tip = log_receipt("VOLCANO-EP51-PREREG2-APPLY",
                      {"episode": {"start": start, "end": end},
                       "results": results, "verdict": verdict,
                       "spec_sha256": _sha(SPEC)})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["logo", "freeze", "apply"])
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    if a.mode == "logo":
        logo()
    elif a.mode == "freeze":
        freeze()
    else:
        assert a.start and a.end, "apply needs --start/--end (UTC, HVO-documented)"
        apply_(a.start, a.end)
