"""HORIZON SCAN — extending the lead-time grid beyond 2h/6h/12h (time-factor
program, user-directed: map how far before onset (shorter AND longer) any
precursor signal survives, using the METHODS already built/validated rather
than inventing new ones -- this is a PARAMETER (offset) extension, not a new
representation).

Existing coverage before this script:
  Weg1 texture (RF)  : 2h(primary) / 6h / 12h  -- exploratory, -12h hit 0.656
                        did NOT survive triangulation (Weg2/Weg3 NULL,
                        consilience DIVERGENCE, see TRIANGULATION.md)
  Weg2 CSD (AR1/var)  : 12h only (envelope needs raw fetch)             NULL
  Weg3 spectral       : 12h only (reused znormed cache)                 NULL
  Weg5 RQA            : 2h / 6h / 12h (reused znormed cache), Holm-corr NULL
  Weg4 coherence      : 12h only (envelope needs raw fetch)             NULL

Two NEW offsets added here: 1h (shorter -- tests whether a tighter, more
actionable lead time carries signal even though the -12h texture hit did
not survive) and 24h (longer -- tests whether backing off from the
exploratory -12h clock finds a stabler, earlier-warning signal). Both need
FRESH IRIS fetches (not covered by any existing cache); segment/validity
rules copied verbatim from volcano_precursor.plan() at the new offset
values, so the design is deterministic from the already-committed EP_HST
table -- no new selection freedom.

At each new offset, apply the THREE methods that need no raw envelope
(texture-RF, spectral, RQA) on the newly fetched znormed windows -- one
fetch serves all three, matching the reuse-over-rebuild convention. CSD and
coherence (which need raw envelope) are explicitly OUT OF SCOPE here
(registered omission, not silently dropped) to keep the new-fetch volume
bounded; they can be added in a follow-up if 1h or 24h shows anything.

PRE-REGISTERED DESIGN (ledger HORIZON-SCAN-PREREG, BEFORE any fetch):
  - Offsets: 1h, 24h (NEW). Segment/validity rules IDENTICAL to
    volcano_precursor.plan() (30-min PRE ending `offset` before onset;
    diurnal-matched MID; validity pre>=prev_end+12h, mid>=24h from both
    boundaries, |mid-pre|>=6h). Episodes 2..50, same table.
  - Methods (all pre-existing, unmodified params): Weg1 texture-RF (lean
    duo, LOGO(episode), exact paired sign test, same as volcano_precursor
    -12h clock family test -- TWO-SIDED here since no prior direction is
    established at these new offsets); Weg3 spectral (3 descriptors,
    TWO-SIDED paired); Weg5 RQA (3 descriptors, TWO-SIDED paired, dim=3
    tau=5 rr=0.1, no grid search).
  - MULTIPLICITY (registered, this IS the lead-time scan the user asked
    for): cells = 2 offsets x 2 stations x 7 descriptors (1 texture +
    3 spectral + 3 RQA) = 28 cells. Holm-Bonferroni across ALL 28 NEW cells
    jointly (not just within-method) -- this is the strictest reasonable
    correction for a pure fishing scan across offset x method. A cell that
    survives THIS correction is the only kind of hit worth a follow-up
    prereg; anything else is reported descriptively only.
  - Combined curve: this script's output is ALSO reported alongside the
    already-ledgered 2h/6h/12h verdicts (Weg1/3/5) as one descriptive
    lead-time table (no re-testing of those, just tabulation) so the full
    1h-24h picture is visible in one place.
  - Claim scope: site-local (Kilauea summit), same as all Kilauea paths.

Usage:
  .venv-research/bin/python horizon_scan.py prereg
  nice -n 19 .venv-research/bin/python horizon_scan.py fetch   # checkpointed, NEW IRIS calls
  .venv-research/bin/python horizon_scan.py run                # cache-only after fetch
"""
import hashlib, json, os, sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_forge import lean_baseline
from receipt_ledger import log_receipt
from volcano_fresh import EP_HST, STATIONS, WIN_PER_SEG, _utc
from volcano_precursor import _fetch
import spectral
from rqa_precursor import _rqa, DESCS as RQA_DESCS

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "horizon_scan_prereg.json")
CACHE = "<local-path>/signalmap/data/volcano/precursor"  # shared with volcano_precursor
NEW_OFFSETS_H = (1, 24)


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def plan():
    """Same rules as volcano_precursor.plan(), parameterized to NEW_OFFSETS_H."""
    eps = [(n, _utc(a), _utc(b)) for n, a, b in EP_HST]
    out = []
    for i in range(1, len(eps)):
        n, start, _ = eps[i]
        prev_end = eps[i - 1][2]
        for off in NEW_OFFSETS_H:
            pre0 = start - timedelta(hours=off, minutes=30)
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


def prereg():
    os.makedirs(FROZEN, exist_ok=True)
    p = plan()
    n_pairs = {off: sum(1 for x in p if x["offset_h"] == off and x["pre"] and x["mid"])
               for off in NEW_OFFSETS_H}
    spec = {
        "prereg": "HORIZON-SCAN (lead-time grid extension, 1h+24h, reusing Weg1/Weg3/Weg5 methods verbatim)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "new_offsets_h": list(NEW_OFFSETS_H),
        "design": "identical rules to volcano_precursor.plan(), parameterized to new offsets; episodes 2..50",
        "methods_reused_unmodified": ["weg1_texture_RF(lean_duo,LOGO,sign_test)",
                                       "weg3_spectral(centroid,spec_ent,hf_ratio)",
                                       "weg5_rqa(det,lam,entr;dim=3,tau=5,rr=0.1)"],
        "out_of_scope_registered": "weg2_csd and weg4_coherence need raw envelope fetch, not covered here to bound new-fetch volume",
        "sidedness": "ALL TWO-SIDED (no established a-priori direction at these new offsets)",
        "multiplicity": "2 offsets x 2 stations x 7 descriptors (1 texture-acc-vs-chance-sign + 3 spectral + 3 rqa) = 28 cells; Holm-Bonferroni across ALL 28 jointly (strictest reasonable correction for a fishing scan)",
        "combined_reporting": "output tabulated alongside already-ledgered 2h/6h/12h verdicts as one descriptive 1h-24h curve; no re-testing of those",
        "claim_scope": "site-local (Kilauea summit)",
        "episode_table_sha256": _sha_text(json.dumps(EP_HST)),
        "plan_sha256": _sha_text(json.dumps(p)),
        "n_pairs_per_offset": n_pairs,
    }
    with open(PREREG, "w") as f:
        json.dump({"spec": spec, "plan": p}, f, indent=1)
    tip = log_receipt("HORIZON-SCAN-PREREG", spec)
    print(f"prereg written, pairs/offset={n_pairs}. ledger tip = {tip}", flush=True)


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
    for sta, t in todo:
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
            X = _fetch(sta, t)  # cache-only after fetch()
            if X is None:
                continue
            raw += [(w, lab, x["ep"]) for w in X]
    return raw


def _paired(a, b):
    from scipy.stats import binomtest, wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b) & (a != b)
    a, b = a[m], b[m]
    n = len(a)
    if n == 0:
        return {"n": 0, "pre_gt_mid": "0/0", "sign_p": 1.0, "wilcoxon_p": 1.0}
    up = int((a > b).sum())
    sp = float(binomtest(up, n, 0.5, alternative="two-sided").pvalue)
    try:
        wp = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except ValueError:
        wp = 1.0
    return {"n": n, "pre_gt_mid": f"{up}/{n}", "sign_p": sp, "wilcoxon_p": wp}


def _weg1_scores(raw):
    """Per-episode LOGO RF pre-probability(PRE) - pre-probability(MID)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    L = lean_baseline(raw)
    y = np.array([r[1] for r in raw])
    g = np.array([r[2] for r in raw])
    out = {}
    for ep in np.unique(g):
        tr, te = g != ep, g == ep
        if not (set(y[te]) >= {"pre", "mid"}):
            continue
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(150, random_state=0, n_jobs=-1))
        clf.fit(L[tr], y[tr])
        pre_col = list(clf.classes_).index("pre")
        proba = clf.predict_proba(L[te])[:, pre_col]
        yt = y[te]
        out[int(ep)] = float(proba[yt == "pre"].mean() - proba[yt == "mid"].mean())
    return out


def _holm(pvals_named, alpha=0.05):
    items = sorted(pvals_named.items(), key=lambda kv: kv[1])
    m = len(items)
    sig, out = True, {}
    for i, (k, pv) in enumerate(items):
        thresh = alpha / (m - i)
        passed = sig and (pv < thresh)
        sig = sig and (pv < thresh)
        out[k] = {"p": pv, "holm_thresh": thresh, "PASS": bool(passed)}
    return out


def run():
    p = _pre_check()
    all_p = {}
    detail = {}
    for sta in STATIONS:
        for off in NEW_OFFSETS_H:
            raw = _bank(sta, off, p)
            if not raw:
                print(f"{sta}-{off}h: EMPTY (fetch incomplete?)", flush=True)
                continue
            # Weg1 texture: sign test on per-episode score sign
            w1 = _weg1_scores(raw)
            vals = np.array(list(w1.values()))
            up = int((vals > 0).sum())
            n = len(vals)
            from scipy.stats import binomtest
            w1_p = float(binomtest(up, n, 0.5, alternative="two-sided").pvalue) if n else 1.0
            all_p[f"{sta}:{off}h:texture"] = w1_p
            detail[f"{sta}:{off}h:texture"] = {"n": n, "pre_gt_mid": f"{up}/{n}", "p": w1_p}
            print(f"{sta}-{off}h texture: {up}/{n} p={w1_p:.4f}", flush=True)

            # Weg3 spectral + Weg5 RQA on the same windows
            by = {}
            for w, lab, ep in raw:
                by.setdefault(int(ep), {"pre": [], "mid": []})[lab].append(w)
            spec_cols = {d: {"pre": [], "mid": []} for d in spectral.DESCS}
            rqa_cols = {d: {"pre": [], "mid": []} for d in RQA_DESCS}
            for ep, segs in by.items():
                if not segs["pre"] or not segs["mid"]:
                    continue
                sp_pre = np.mean([spectral._descriptors(w) for w in segs["pre"]], axis=0)
                sp_mid = np.mean([spectral._descriptors(w) for w in segs["mid"]], axis=0)
                rq_pre = np.mean([_rqa(w) for w in segs["pre"]], axis=0)
                rq_mid = np.mean([_rqa(w) for w in segs["mid"]], axis=0)
                for j, d in enumerate(spectral.DESCS):
                    spec_cols[d]["pre"].append(float(sp_pre[j]))
                    spec_cols[d]["mid"].append(float(sp_mid[j]))
                for j, d in enumerate(RQA_DESCS):
                    rqa_cols[d]["pre"].append(float(rq_pre[j]))
                    rqa_cols[d]["mid"].append(float(rq_mid[j]))
            for d in spectral.DESCS:
                t = _paired(spec_cols[d]["pre"], spec_cols[d]["mid"])
                all_p[f"{sta}:{off}h:spectral_{d}"] = max(t["sign_p"], t["wilcoxon_p"])
                detail[f"{sta}:{off}h:spectral_{d}"] = t
                print(f"{sta}-{off}h spectral.{d}: {t['pre_gt_mid']} "
                      f"sign_p={t['sign_p']:.4f} wil_p={t['wilcoxon_p']:.4f}", flush=True)
            for d in RQA_DESCS:
                t = _paired(rqa_cols[d]["pre"], rqa_cols[d]["mid"])
                all_p[f"{sta}:{off}h:rqa_{d}"] = max(t["sign_p"], t["wilcoxon_p"])
                detail[f"{sta}:{off}h:rqa_{d}"] = t
                print(f"{sta}-{off}h rqa.{d}: {t['pre_gt_mid']} "
                      f"sign_p={t['sign_p']:.4f} wil_p={t['wilcoxon_p']:.4f}", flush=True)

    holm = _holm(all_p)
    survivors = [k for k, h in holm.items() if h["PASS"]]
    print(f"\n=> Holm-corrected (28-cell joint) survivors: {survivors or 'NONE'}", flush=True)
    tip = log_receipt("HORIZON-SCAN", {"plan_sha256": _sha_text(json.dumps(p)),
                                       "detail": detail, "holm": holm,
                                       "survivors": survivors})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "fetch": fetch, "run": run,
     "plan": lambda: print(json.dumps(plan(), indent=1))}[sys.argv[1]]()
