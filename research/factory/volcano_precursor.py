"""VOLCANO PRECURSOR bank (FRESH_DATA_SCAN #1b, discovery candidate).

Question: does seismic TEXTURE shortly before an episode start differ from
mid-pause texture of the SAME pause? HVO forecasts episodes via tilt only;
a texture precursor at HHZ stations would be a genuine discovery candidate.

PRE-REGISTERED DESIGN (ledger VOLCANO-PRECURSOR-PREREG written by `prereg`
BEFORE any fresh sample is loaded; plan is deterministic from the episode
table already committed in volcano_fresh.py):
  - Episodes 2..50 (ep 1 excluded: preceding pause undefined / pre-eruption
    regime, claim is INTER-EPISODE precursor).
  - PRE segment (per episode, per offset): 30 min ending exactly offset
    BEFORE episode start. Offsets: 2h = PRIMARY; 6h, 12h = exploratory.
  - MID segment (control, same pause = prev_end -> start): midpoint date of
    the pause at the SAME UTC clock time as the PRE segment (diurnal control).
  - Validity (else pair dropped for that offset, recorded in the plan):
    PRE window starts >= prev_end + 12 h (no episode-tail contamination);
    MID window >= 24 h from both pause boundaries; MID and PRE windows
    >= 6 h apart.
  - Windowing identical to load_volcano: 8x1024 @ 100 sps at rolling-RMS
    peak, znorm. Family: lean duo (fixed a priori, no selection).
  - PRIMARY criterion: UWE @ -2h, LOGO (episode = group) segment-majority
    accuracy, cluster-bootstrap CI over episodes, PASS = CI-lo > 0.5 AND
    group-perm p (n_perm=200) < 0.05. RIMD @ -2h = replication (same test).
    Full bank_audit + gauntlet on the two -2h banks (NEW-bank rule).
    -6h / -12h: lean LOGO + CI + perm reported as EXPLORATORY (registered as
    such; no table verdict from them without a later dedicated prereg).
  - Claim, if any, stays SITE-LOCAL (Kilauea summit stations), texture-level.

Usage:
  .venv-research/bin/python volcano_precursor.py prereg
  nice -n 19 .venv-research/bin/python volcano_precursor.py fetch  # checkpointed
  nice -n 19 .venv-research/bin/python volcano_precursor.py run
"""
import hashlib, json, os, sys, urllib.request
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_forge import lean_baseline
from harvest3_loaders import W, _znorm
from receipt_ledger import log_receipt
from volcano_fresh import EP_HST, STATIONS, WIN_PER_SEG, _utc

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "volcano_precursor_prereg.json")
CACHE = "<local-path>/signalmap/data/volcano/precursor"
OFFSETS_H = (2, 6, 12)  # 2 = PRIMARY, 6/12 exploratory


def plan():
    """Deterministic segment plan from the committed episode table. No data."""
    eps = [(n, _utc(a), _utc(b)) for n, a, b in EP_HST]
    out = []
    for i in range(1, len(eps)):
        n, start, _ = eps[i]
        prev_end = eps[i - 1][2]
        for off in OFFSETS_H:
            pre0 = start - timedelta(hours=off, minutes=30)  # window start
            if pre0 < prev_end + timedelta(hours=12):
                out.append({"ep": n, "offset_h": off, "pre": None, "mid": None,
                            "drop": "pre too close to previous episode"})
                continue
            mid_dt = prev_end + (start - prev_end) / 2
            mid0 = datetime.combine(mid_dt.date(), pre0.time())
            ok_mid = (mid0 - prev_end >= timedelta(hours=24) and
                      start - (mid0 + timedelta(minutes=30)) >= timedelta(hours=24) and
                      abs((mid0 - pre0).total_seconds()) >= 6 * 3600)
            out.append({"ep": n, "offset_h": off, "pre": pre0.isoformat(),
                        "mid": mid0.isoformat() if ok_mid else None,
                        "drop": None if ok_mid else "mid fails validity"})
    return out


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prereg():
    os.makedirs(FROZEN, exist_ok=True)
    p = plan()
    n_pairs = {off: sum(1 for x in p if x["offset_h"] == off and x["pre"] and x["mid"])
               for off in OFFSETS_H}
    spec = {
        "prereg": "VOLCANO-PRECURSOR (texture before episode start vs mid-pause, same pause)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design": "PRE 30min ending offset before start vs MID same-clock-time midpoint of same pause; eps 2..50; validity pre>=prev_end+12h, mid>=24h from boundaries, |mid-pre|>=6h",
        "offsets_h": {"primary": 2, "exploratory": [6, 12]},
        "family": "lean duo perm_entropy(3,norm)+psd_slope, fixed a priori",
        "criterion": "UWE@-2h PRIMARY: LOGO(episode) segment-majority acc, cluster-bootstrap CI over episodes, PASS=CI-lo>0.5 AND group-perm p(n_perm=200)<0.05; RIMD@-2h replication; -6h/-12h exploratory only",
        "new_bank_rule": "bank_audit + gauntlet on both -2h banks before any verdict",
        "claim_scope": "site-local (Kilauea summit), texture-level",
        "episode_table_sha256": _sha_text(json.dumps(EP_HST)),
        "plan_sha256": _sha_text(json.dumps(p)),
        "n_pairs_per_offset": n_pairs,
    }
    with open(PREREG, "w") as f:
        json.dump({"spec": spec, "plan": p}, f, indent=1)
    tip = log_receipt("VOLCANO-PRECURSOR-PREREG", spec)
    print(f"prereg written, pairs/offset={n_pairs}. ledger tip = {tip}", flush=True)


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
        # ascii1 streams with gaps carry a TIMESERIES header PER SEGMENT, not
        # only on line 1 — filter all headers (fetch hardening 17. Jul, after
        # the registered "FETCH-FAIL pre" apply51 readout; analysis untouched)
        x = np.array([ln for ln in d.strip().split("\n")
                      if ln and not ln.startswith("TIMESERIES")], float)
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


def _pre_check():
    assert os.path.exists(PREREG), "run prereg first (ledger before load!)"
    with open(PREREG) as f:
        pre = json.load(f)
    assert _sha_text(json.dumps(pre["plan"])) == pre["spec"]["plan_sha256"], "plan tampered"
    return pre["plan"]


def fetch():
    p = _pre_check()
    os.makedirs(CACHE, exist_ok=True)
    todo = [(sta, t) for sta in STATIONS for x in p if x["pre"] and x["mid"]
            for t in (x["pre"], x["mid"])]
    done = 0
    for i, (sta, t) in enumerate(todo):
        _fetch(sta, t)
        done += 1
        if done % 25 == 0:
            print(f"[checkpoint] {done}/{len(todo)} segments", flush=True)
    print(f"fetch complete: {len(todo)} segments (incl. cached/failed)", flush=True)


def _bank(sta, off, p):
    raw = []
    for x in p:
        if x["offset_h"] != off or not x["pre"] or not x["mid"]:
            continue
        for lab, t in (("pre", x["pre"]), ("mid", x["mid"])):
            X = _fetch(sta, t)  # cache-only in practice after fetch mode
            if X is None:
                continue
            raw += [(w, lab, x["ep"]) for w in X]
    return raw


def _verdict(raw, final):
    """LOGO(episode) segment-majority acc + exact PAIRED sign test.

    AMENDMENT (ledger VOLCANO-PRECURSOR-AMENDMENT, before any verdict was
    seen): the registered group-perm test (group_perm_p) assumes one label
    per group; this bank is PAIRED (pre+mid inside the same episode), which
    audit checks 6/7 flagged structurally. Correct exchangeability unit =
    within-episode pre/mid swap -> the exact test is the binomial sign test
    over episode-wise orderings: P(pre-segment scored more pre-like than its
    mid-segment) = 0.5 under H0. Deterministic and exact; replaces perm_p.
    """
    from scipy.stats import binomtest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    L = lean_baseline(raw)
    y = np.array([r[1] for r in raw])
    g = np.array([r[2] for r in raw])
    per_ep, orderings = [], []
    for ep in np.unique(g):
        tr, te = g != ep, g == ep
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(L[tr], y[tr])
        pred, yt = clf.predict(L[te]), y[te]
        pre_col = list(clf.classes_).index("pre")
        proba = clf.predict_proba(L[te])[:, pre_col]
        hits, seg_score = [], {}
        for lab in ("pre", "mid"):
            m = yt == lab
            n = int(m.sum()) // WIN_PER_SEG
            for k in range(n):
                blk = pred[m][k * WIN_PER_SEG:(k + 1) * WIN_PER_SEG]
                maj = "pre" if (blk == "pre").sum() > WIN_PER_SEG / 2 else "mid"
                hits.append(float(maj == lab))
                seg_score[lab] = float(proba[m][k * WIN_PER_SEG:(k + 1) * WIN_PER_SEG].mean())
        per_ep.append(hits)
        if "pre" in seg_score and "mid" in seg_score and seg_score["pre"] != seg_score["mid"]:
            orderings.append(seg_score["pre"] > seg_score["mid"])
    correct = np.concatenate([np.array(h) for h in per_ep])
    acc = float(correct.mean())
    rng = np.random.default_rng(0)
    boots = [np.concatenate([per_ep[i] for i in rng.integers(0, len(per_ep), len(per_ep))]).mean()
             for _ in range(10000)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    conc = int(sum(orderings))
    p_sign = float(binomtest(conc, len(orderings), 0.5, alternative="greater").pvalue) \
        if orderings else 1.0
    return {"n_segments": int(len(correct)), "n_episodes": len(per_ep),
            "seg_acc": acc, "ci": [lo, hi],
            "pairs_concordant": f"{conc}/{len(orderings)}", "sign_p": p_sign}


def amend():
    """Ledger the perm-test amendment BEFORE any verdict is computed/seen."""
    tip = log_receipt("VOLCANO-PRECURSOR-AMENDMENT", {
        "reason": "audit checks 6/7 flagged structurally: bank is PAIRED (pre+mid in same episode-group), generic recording-label machinery (group_perm_p, label-shuffle, class-balance) assumes one label per group -> registered perm-test invalid for this design",
        "seen_before_amendment": "only window-level lean LOGO 0.538 (UWE-2h, from audit output); no segment verdict, no CI, no sign test computed yet",
        "replacement": "exact binomial sign test over episode-wise paired orderings (P(pre scored more pre-like than its mid)=0.5 under H0), deterministic and exact; criterion becomes CI-lo>0.5 AND sign_p<0.05 (UWE@-2h primary, RIMD@-2h replication)",
        "audit_note": "checks 6 (class-balance) and 7 (label-shuffle) are expected-FAIL on paired banks by construction; leak-relevant checks 4/5/8/9/10 PASS",
        "gauntlet_note": "gauntlet runs n_perm=60 for its forge/CI parts; its group-perm p is flagged invalid-for-paired and carries no verdict weight",
    })
    print(f"amendment ledgered. tip = {tip}", flush=True)


def run():
    p = _pre_check()
    from audit import audit_bank
    from gauntlet import gauntlet
    results = {}
    for sta in STATIONS:
        for off in OFFSETS_H:
            name = f"PRECURSOR-{sta}-{off}h"
            raw = _bank(sta, off, p)
            if not raw:
                results[name] = "EMPTY"
                continue
            primary = off == 2
            if primary:
                print(f"== {name}: bank_audit + gauntlet (NEW-bank rule; "
                      f"checks 6/7 expected-FAIL paired, gauntlet perm-p no weight)",
                      flush=True)
                audit_bank(name, raw)
                gauntlet(name, raw, n_perm=60)
            r = _verdict(raw, final=primary)
            r["role"] = "PRIMARY" if (primary and sta == "UWE") else \
                        "replication" if primary else "exploratory"
            r["verdict"] = ("PASS" if r["ci"][0] > 0.5 and r["sign_p"] < 0.05
                            else "NULL") if primary else "exploratory"
            results[name] = r
            print(f"{name} [{r['role']}]: n={r['n_segments']} acc={r['seg_acc']:.3f} "
                  f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] pairs={r['pairs_concordant']} "
                  f"sign_p={r['sign_p']:.4f} -> {r['verdict']}", flush=True)
    tip = log_receipt("VOLCANO-PRECURSOR", {"plan_sha256": _sha_text(json.dumps(p)),
                                            "results": results})
    print(f"ledger tip = {tip}", flush=True)


SPEC51 = os.path.join(FROZEN, "precursor12h_ep51_spec.json")
EP50_END_UTC = "2026-06-28T03:10"


def freeze51():
    """PREREG-3: prospective -12h precursor receipt on Episode 51 (n=1 demo).

    HONESTY, registered up front: the -12h offset was SELECTED after the
    exploratory look at the 50-episode bank (UWE 0.656 p=.008 / RIMD 0.656
    p=.003, ledger affbdc1d) — that is why this needs a PROSPECTIVE test on
    an episode that does not exist yet. n=1 episode: a demonstrative
    prospective receipt, not statistics. Frozen BEFORE Ep 51 (HVO forecast
    Jul 9-15; watcher confirms not started at freeze time).

    PROTOCOL (apply51, run only after HVO documents Ep 51 start, UTC):
      PRE = 30-min window [start-12.5h, start-12h] (identical rule to the
      bank). MID control = midpoint(Ep50_end, Ep51_start) date at the SAME
      UTC clock time as PRE; validity mid>=24h from both pause boundaries,
      |mid-pre|>=6h, else verdict 'not evaluable' (registered).
      Per station, two registered readouts:
        primary  ORDERING: model pre-probability(PRE) > pre-probability(MID)
        strict   BOTH-MAJORITY: PRE majority 'pre' AND MID majority 'mid'
      UWE = primary station, RIMD = replication. Report all, no re-selection.
    """
    p = _pre_check()
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models = {}
    for sta in STATIONS:
        raw = _bank(sta, 12, p)  # cached only
        L = lean_baseline(raw)
        y = np.array([r[1] for r in raw])
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(L, y)
        path = os.path.join(FROZEN, f"precursor12h_model_{sta.lower()}.joblib")
        joblib.dump(clf, path)
        models[sta] = {"path": os.path.relpath(path, HERE), "n_windows": len(raw),
                       "sha256": hashlib.sha256(open(path, 'rb').read()).hexdigest()}
        print(f"{sta}: frozen on {len(raw)} windows -> {path}", flush=True)
    spec = {
        "prereg": "VOLCANO-PRECURSOR-12H-EP51 (PREREG-3, prospective, n=1 demo receipt)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_honesty": "-12h chosen AFTER exploratory look at 50-episode bank (UWE 0.656 p=.008 / RIMD 0.656 p=.003, both-station replication, ledger affbdc1d); prospective Ep-51 test is the required confirmation",
        "context": "Ep 51 not occurred at freeze (watcher log + HVO forecast Jul 9-15)",
        "protocol": "PRE=[start-12.5h,start-12h]; MID=midpoint(Ep50_end,start) date at same clock; validity mid>=24h from boundaries, |mid-pre|>=6h; primary=ORDERING pre-prob(PRE)>pre-prob(MID), strict=both majorities correct; UWE primary, RIMD replication",
        "models": models,
        "plan_sha256": _sha_text(json.dumps(p)),
    }
    with open(SPEC51, "w") as f:
        json.dump(spec, f, indent=1, sort_keys=True)
    tip = log_receipt("VOLCANO-PRECURSOR-12H-EP51-PREREG",
                      {**spec, "spec_sha256": _sha_text(open(SPEC51).read())})
    print(f"frozen. spec={SPEC51}\nledger tip = {tip}", flush=True)


def apply51(start):
    import joblib
    with open(SPEC51) as f:
        spec = json.load(f)
    t_start = datetime.fromisoformat(start)
    ep50_end = datetime.fromisoformat(EP50_END_UTC)
    assert t_start > ep50_end + timedelta(hours=48), "start implausible"
    pre0 = t_start - timedelta(hours=12, minutes=30)
    mid_dt = ep50_end + (t_start - ep50_end) / 2
    mid0 = datetime.combine(mid_dt.date(), pre0.time())
    ok = (mid0 - ep50_end >= timedelta(hours=24) and
          t_start - (mid0 + timedelta(minutes=30)) >= timedelta(hours=24) and
          abs((mid0 - pre0).total_seconds()) >= 6 * 3600)
    if not ok:
        print("VERDICT: not evaluable (registered validity rules)", flush=True)
        log_receipt("VOLCANO-PRECURSOR-12H-EP51", {"start": start, "verdict": "not evaluable"})
        return
    results = {}
    for sta in STATIONS:
        mp = os.path.join(HERE, spec["models"][sta]["path"])
        assert hashlib.sha256(open(mp, 'rb').read()).hexdigest() == \
            spec["models"][sta]["sha256"], "model tampered"
        clf = joblib.load(mp)
        segs = {}
        for lab, t0 in (("pre", pre0), ("mid", mid0)):
            X = _fetch(sta, t0.isoformat())
            if X is None:
                results[sta] = {"verdict": f"FETCH-FAIL {lab}"}
                break
            F = lean_baseline([(w, "?", 0) for w in X])
            pre_col = list(clf.classes_).index("pre")
            proba = clf.predict_proba(F)[:, pre_col]
            pred = clf.predict(F)
            segs[lab] = {"mean_pre_prob": float(proba.mean()),
                         "majority": "pre" if (pred == "pre").sum() > len(pred) / 2 else "mid"}
        if sta in results:
            continue
        ordering = segs["pre"]["mean_pre_prob"] > segs["mid"]["mean_pre_prob"]
        strict = segs["pre"]["majority"] == "pre" and segs["mid"]["majority"] == "mid"
        results[sta] = {"segments": segs, "ordering_concordant": bool(ordering),
                        "strict_both_correct": bool(strict)}
        print(f"{sta}: ordering={'PASS' if ordering else 'FAIL'} "
              f"strict={'PASS' if strict else 'FAIL'} "
              f"(pre-prob PRE {segs['pre']['mean_pre_prob']:.3f} vs MID "
              f"{segs['mid']['mean_pre_prob']:.3f})", flush=True)
    tip = log_receipt("VOLCANO-PRECURSOR-12H-EP51",
                      {"start": start, "results": results,
                       "spec_sha256": _sha_text(open(SPEC51).read())})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "apply51":
        assert len(sys.argv) > 2, "apply51 needs <start UTC ISO>"
        apply51(sys.argv[2])
    else:
        {"prereg": prereg, "fetch": fetch, "amend": amend, "run": run,
         "freeze51": freeze51,
         "plan": lambda: print(json.dumps(plan(), indent=1))}[sys.argv[1]]()
