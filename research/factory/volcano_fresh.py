"""VOLCANO out-of-time fresh holdout (FRESH_DATA_SCAN #1a).

Bank ends 2023; since Dec 23, 2024 Kilauea erupts in episodic fountaining.
50 HVO-documented episodes (USGS eruption-information table, fetched Jul 3,
2026) = untouched out-of-time data. The FROZEN per-station lean models from
ep51_prereg.py (hash-anchored in the ledger, commit 17eb13d) are applied as-is:
no re-selection, no re-fit.

PRE-REGISTERED PROTOCOL (ledger entry VOLCANO-FRESH-OOT-PREREG written by
`prereg` BEFORE the first fresh sample is loaded):
  - Eruptive segment per episode: 30 min starting episode_start + 60 min
    (ep51 rule; shortest episode = 4 h, so segment always inside episode).
  - Quiet segment per pause: midpoint(episode_end, next_episode_start) date,
    at the SAME UTC clock time as the paired eruptive segment (diurnal
    control); required >= 24 h from both episode boundaries, else pair has
    no quiet segment. After episode 50 the pause partner is the midpoint of
    (ep50_end, 2026-07-02) — HVO confirms no episode 51 through Jul 2 update.
  - Windowing identical to load_volcano / ep51: 8x1024 @ 100 sps at the
    rolling-RMS peak of the segment, znorm. Segment verdict = majority vote.
  - PRIMARY criterion (per station): segment-majority accuracy over all
    fetched segments, cluster-bootstrap CI over episodes; PASS = CI-lo > 0.5.
  - SECONDARY: exact binomial sign test over complete pairs (eruptive
    majority-eruptive AND quiet majority-quiet = concordant).
  - PRE-REGISTERED CAVEAT (carried from ep51 spec): 2024-26 inter-episode
    pauses carry documented unrest; eruptive-PASS + quiet-FAIL -> honest
    verdict "reads unrest regime, not episode on/off", no re-selection.

Usage:
  .venv-research/bin/python volcano_fresh.py prereg
  nice -n 19 .venv-research/bin/python volcano_fresh.py run   # checkpointed
"""
import hashlib, json, os, sys, urllib.request
from datetime import datetime, timedelta

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_forge import lean_baseline
from harvest3_loaders import W, _znorm
from receipt_ledger import log_receipt

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "volcano_oot_prereg.json")
CACHE = "<local-path>/signalmap/data/volcano/fresh_oot"
STATIONS = ("UWE", "RIMD")
WIN_PER_SEG = 8
LAST_QUIET_CONFIRMED = "2026-07-02T00:00"  # HVO Jul-2 update: still paused

# USGS HVO episode table (HST), fetched 2026-07-03. (num, start_hst, end_hst)
EP_HST = [
    (1, "2024-12-23 02:20", "2024-12-23 16:00"),
    (2, "2024-12-24 08:00", "2024-12-25 11:00"),
    (3, "2024-12-26 08:00", "2025-01-03 20:30"),
    (4, "2025-01-15 09:00", "2025-01-18 10:10"),
    (5, "2025-01-22 14:30", "2025-01-23 04:30"),
    (6, "2025-01-24 23:28", "2025-01-25 12:36"),
    (7, "2025-01-27 18:41", "2025-01-28 10:41"),
    (8, "2025-02-03 21:52", "2025-02-04 19:23"),
    (9, "2025-02-11 10:16", "2025-02-12 08:43"),
    (10, "2025-02-19 20:22", "2025-02-20 09:18"),
    (11, "2025-02-25 18:26", "2025-02-26 07:06"),
    (12, "2025-03-04 07:30", "2025-03-05 10:37"),
    (13, "2025-03-11 02:36", "2025-03-11 15:13"),
    (14, "2025-03-19 09:26", "2025-03-20 13:49"),
    (15, "2025-03-25 12:04", "2025-03-26 19:10"),
    (16, "2025-03-31 22:57", "2025-04-02 12:04"),
    (17, "2025-04-07 22:15", "2025-04-09 09:45"),
    (18, "2025-04-22 03:30", "2025-04-22 13:28"),
    (19, "2025-05-01 21:28", "2025-05-02 05:20"),
    (20, "2025-05-06 17:28", "2025-05-06 21:28"),
    (21, "2025-05-11 12:45", "2025-05-11 20:36"),
    (22, "2025-05-16 05:13", "2025-05-16 15:29"),
    (23, "2025-05-25 16:15", "2025-05-25 22:25"),
    (24, "2025-06-04 20:55", "2025-06-05 04:28"),
    (25, "2025-06-11 11:57", "2025-06-11 20:08"),
    (26, "2025-06-20 01:40", "2025-06-20 10:25"),
    (27, "2025-06-29 09:05", "2025-06-29 19:54"),
    (28, "2025-07-09 04:10", "2025-07-09 13:20"),
    (29, "2025-07-20 05:15", "2025-07-20 18:35"),
    (30, "2025-08-06 01:20", "2025-08-06 12:55"),
    (31, "2025-08-22 14:04", "2025-08-23 02:52"),
    (32, "2025-09-02 06:35", "2025-09-02 20:01"),
    (33, "2025-09-19 03:11", "2025-09-19 12:08"),
    (34, "2025-10-01 00:53", "2025-10-01 07:03"),
    (35, "2025-10-17 20:05", "2025-10-18 03:32"),
    (36, "2025-11-09 11:15", "2025-11-09 16:16"),
    (37, "2025-11-25 14:30", "2025-11-25 23:39"),
    (38, "2025-12-06 08:45", "2025-12-06 20:52"),
    (39, "2025-12-23 20:10", "2025-12-24 02:13"),
    (40, "2026-01-12 08:22", "2026-01-12 18:04"),
    (41, "2026-01-24 11:10", "2026-01-24 19:29"),
    (42, "2026-02-15 13:50", "2026-02-15 23:38"),
    (43, "2026-03-10 09:17", "2026-03-10 18:21"),
    (44, "2026-04-09 11:10", "2026-04-09 19:41"),
    (45, "2026-04-23 01:34", "2026-04-23 10:01"),
    (46, "2026-05-05 08:17", "2026-05-05 17:22"),
    (47, "2026-05-14 15:27", "2026-05-15 00:27"),
    (48, "2026-06-01 04:40", "2026-06-01 13:37"),
    (49, "2026-06-14 09:36", "2026-06-14 17:05"),
    (50, "2026-06-27 10:10", "2026-06-27 17:10"),
]


def _utc(hst):
    return datetime.strptime(hst, "%Y-%m-%d %H:%M") + timedelta(hours=10)


def plan():
    """Deterministic segment plan from the episode table. No data touched."""
    eps = [(n, _utc(a), _utc(b)) for n, a, b in EP_HST]
    out = []
    for i, (n, t0, t1) in enumerate(eps):
        er = t0 + timedelta(minutes=60)
        assert er + timedelta(minutes=30) <= t1, f"ep {n} too short"
        nxt = eps[i + 1][1] if i + 1 < len(eps) else \
            datetime.fromisoformat(LAST_QUIET_CONFIRMED)
        mid = t1 + (nxt - t1) / 2
        q = datetime.combine(mid.date(), er.time())
        ok = (q - t1 >= timedelta(hours=24)) and (nxt - (q + timedelta(minutes=30)) >= timedelta(hours=24))
        out.append({"ep": n, "eruptive": er.isoformat(), "quiet": q.isoformat() if ok else None})
    return out


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prereg():
    os.makedirs(FROZEN, exist_ok=True)
    with open(os.path.join(FROZEN, "ep51_spec.json")) as f:
        ep51 = json.load(f)
    p = plan()
    spec = {
        "prereg": "VOLCANO-FRESH-OOT (50 episodes Dec 2024 - Jun 2026, out-of-time holdout)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_models": {s: ep51["models"][s]["sha256"] for s in STATIONS},
        "episode_table_sha256": _sha_text(json.dumps(EP_HST)),
        "plan_sha256": _sha_text(json.dumps(p)),
        "n_eruptive": len(p), "n_quiet": sum(x["quiet"] is not None for x in p),
        "criterion": "per station: segment-majority acc, cluster-bootstrap CI over episodes, PASS=CI-lo>0.5; secondary exact-binomial sign test on complete pairs",
        "caveat": "eruptive-PASS + quiet-FAIL -> 'reads unrest regime, not episode on/off'",
    }
    with open(PREREG, "w") as f:
        json.dump({"spec": spec, "plan": p}, f, indent=1)
    tip = log_receipt("VOLCANO-FRESH-OOT-PREREG", spec)
    print(f"prereg written ({spec['n_eruptive']} eruptive / {spec['n_quiet']} quiet "
          f"segments per station). ledger tip = {tip}", flush=True)


def _fetch(sta, t0_iso):
    key = f"{sta}_{t0_iso.replace(':', '')}.npz"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        z = np.load(path)
        return z["X"] if z["X"].size else None
    t0 = datetime.fromisoformat(t0_iso)
    t1 = t0 + timedelta(minutes=30)
    u = (f"https://service.iris.edu/irisws/timeseries/1/query?net=HV&sta={sta}"
         f"&loc=--&cha=HHZ&starttime={t0.strftime('%Y-%m-%dT%H:%M:%S')}"
         f"&endtime={t1.strftime('%Y-%m-%dT%H:%M:%S')}&format=ascii1")
    try:
        d = urllib.request.urlopen(u, timeout=180).read().decode()
        x = np.array(d.strip().split("\n")[1:], float)
        assert len(x) >= (WIN_PER_SEG + 2) * W, f"only {len(x)} samples"
    except Exception as e:
        print(f"  {sta} {t0_iso}: FETCH-FAIL {e}", flush=True)
        np.savez_compressed(path, X=np.array([]))  # checkpoint the failure too
        return None
    r = np.array([x[i:i + W].std() for i in range(0, len(x) - W, 512)])
    c = int(np.argmax(r)) * 512 + W // 2
    a = int(np.clip(c - WIN_PER_SEG * W // 2, 0, len(x) - WIN_PER_SEG * W))
    X = np.array([_znorm(x[a + k * W:a + (k + 1) * W]) for k in range(WIN_PER_SEG)])
    np.savez_compressed(path, X=X)
    return X


def run():
    assert os.path.exists(PREREG), "run prereg first (ledger before load!)"
    with open(PREREG) as f:
        pre = json.load(f)
    p = pre["plan"]
    assert _sha_text(json.dumps(p)) == pre["spec"]["plan_sha256"], "plan tampered"
    os.makedirs(CACHE, exist_ok=True)
    results = {}
    for sta in STATIONS:
        clf = joblib.load(os.path.join(FROZEN, f"ep51_model_{sta.lower()}.joblib"))
        rows = []  # (ep, label, majority)
        for item in p:
            for lab, t in (("eruptive", item["eruptive"]), ("quiet", item["quiet"])):
                if t is None:
                    continue
                X = _fetch(sta, t)
                if X is None:
                    continue
                pred = clf.predict(lean_baseline([(x, "?", 0) for x in X]))
                maj = "eruptive" if (pred == "eruptive").sum() > WIN_PER_SEG / 2 else "quiet"
                rows.append((item["ep"], lab, maj))
        correct = np.array([lab == maj for _, lab, maj in rows], float)
        eps = np.array([e for e, _, _ in rows])
        acc = float(correct.mean())
        rng = np.random.default_rng(0)
        uq = np.unique(eps)
        boots = [np.concatenate([correct[eps == e] for e in rng.choice(uq, len(uq))]).mean()
                 for _ in range(10000)]
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        er = [(e, m) for e, l, m in rows if l == "eruptive"]
        qu = dict((e, m) for e, l, m in rows if l == "quiet")
        pairs = [(m == "eruptive", qu[e] == "quiet") for e, m in er if e in qu]
        conc = sum(a and b for a, b in pairs)
        from scipy.stats import binomtest
        p_sign = binomtest(conc, len(pairs), 0.25, alternative="greater").pvalue if pairs else 1.0
        e_acc = float(np.mean([m == "eruptive" for _, m in er])) if er else 0.0
        q_acc = float(np.mean([m == "quiet" for m in qu.values()])) if qu else 0.0
        verdict = ("PASS" if lo > 0.5 else
                   "unrest-fallback" if e_acc >= 0.75 and q_acc < 0.5 else "FAIL")
        results[sta] = {"n_segments": len(rows), "acc": acc,
                        "ci": (float(lo), float(hi)), "eruptive_acc": e_acc,
                        "quiet_acc": q_acc, "n_pairs": len(pairs),
                        "concordant": conc, "sign_p_vs_chance": float(p_sign),
                        "verdict": verdict}
        print(f"{sta}: n={len(rows)} acc={acc:.3f} CI[{lo:.3f},{hi:.3f}] "
              f"eruptive {e_acc:.3f} / quiet {q_acc:.3f} "
              f"pairs {conc}/{len(pairs)} p={p_sign:.4f} -> {verdict}", flush=True)
    tip = log_receipt("VOLCANO-FRESH-OOT", {"prereg_plan_sha256": pre["spec"]["plan_sha256"],
                                            "results": results})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "run": run, "plan": lambda: print(json.dumps(plan(), indent=1))}[sys.argv[1]]()
