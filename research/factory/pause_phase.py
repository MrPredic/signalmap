"""PAUSE-PHASE-ORDINAL (time-factor program, Prio 2b a).

The -12h exploratory hit (ledger affbdc1d: UWE 0.656 sign_p=.008 / RIMD 0.656
sign_p=.003, both stations) left the interpretation OPEN: texture ~12h before
an episode start differs from mid-pause — is that (i) a late-pause-SPECIFIC
change (precursor-side reading) or (ii) a monotone drift across the whole
pause (pause-phase gradient)? This test adds the missing ordinal phase point:
EARLY pause, same clock family as the cached -12h PRE/MID pair.

PRE-REGISTERED DESIGN (ledger PAUSE-PHASE-ORDINAL-PREREG written by `prereg`
BEFORE the first EARLY sample is fetched; PRE/MID segments are the already
cached, already ledgered precursor bank — reused, disclosed as such):
  - Pauses: the -12h rows of the frozen VOLCANO-PRECURSOR plan (episodes
    2..50) with valid pre+mid. Clock = the -12h PRE clock of that pause
    (diurnal control identical to the paired bank).
  - EARLY (NEW fetch, 30 min): first datetime >= prev_episode_end + 24 h
    whose UTC clock equals the PRE clock (deterministic). Validity:
    MID - EARLY >= 6 h, else pause dropped for this test (recorded).
  - Windowing/caching identical to the precursor bank (shared cache + rule).
  - Scorer: the registered precursor machinery verbatim — LOGO(episode)
    RandomForest(150, seed 0) on lean-duo features, trained on pre(-12h) vs
    mid of the OTHER pauses only; held-out pause's EARLY/MID/PRE segments are
    scored (mean pre-probability). EARLY is never trained on.
  - PRIMARY (UWE): exact binomial sign test, TWO-SIDED, on episode-wise
    ordering pre-prob(EARLY) vs pre-prob(MID). RIMD = replication.
  - Registered interpretation map (fixed before any EARLY data is seen):
      EARLY~MID (p>=0.05) AND in-fold LATE>MID concordance persists
        -> late-pause-SPECIFIC change (precursor-side reading survives);
      EARLY<MID significant (same direction as the MID<LATE hit)
        -> monotone pause-phase gradient reading;
      EARLY>MID significant -> non-monotone / artifact flag (episode-tail or
        diurnal residue), honest report, no positive claim.
  - Secondary, descriptive only: strict monotone triples EARLY<MID<LATE,
    mean pre-prob per phase point, in-fold LATE-vs-MID concordance.
  - MULTIPLICITY, registered: this is the SECOND test family run on the
    precursor-bank segments (after the registered PRE/MID paired tests).
    The -12h clock family was chosen BECAUSE the exploratory hit lives
    there; purpose is interpretation disambiguation, NOT a new discovery
    claim. Any claim stays site-local (Kilauea summit), texture-level.
  - NEW-bank rule: bank_audit + gauntlet on the (early vs mid) paired bank;
    checks 6/7 expected-FAIL on paired designs (amendment 317bce69),
    gauntlet perm-p carries no verdict weight; verdicts come from the exact
    sign test only.

Usage:
  .venv-research/bin/python research/factory/pause_phase.py prereg
  nice -n 19 .venv-research/bin/python research/factory/pause_phase.py fetch
  nice -n 19 .venv-research/bin/python research/factory/pause_phase.py run
"""
import hashlib, json, os, sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_forge import lean_baseline
from receipt_ledger import log_receipt
from volcano_fresh import EP_HST, STATIONS, WIN_PER_SEG, _utc
from volcano_precursor import _fetch, _pre_check as _prec_pre_check

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
PREREG = os.path.join(FROZEN, "pause_phase_prereg.json")


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def plan():
    """Deterministic EARLY plan from the frozen precursor plan (-12h rows)."""
    prec = _prec_pre_check()  # verifies precursor plan hash before reuse
    ends = {n: _utc(b) for n, _, b in EP_HST}
    out = []
    for x in prec:
        if x["offset_h"] != 12 or not x["pre"] or not x["mid"]:
            continue
        prev_end = ends[x["ep"] - 1]
        pre0 = datetime.fromisoformat(x["pre"])
        mid0 = datetime.fromisoformat(x["mid"])
        e = datetime.combine((prev_end + timedelta(hours=24)).date(), pre0.time())
        if e < prev_end + timedelta(hours=24):
            e += timedelta(days=1)
        ok = mid0 - e >= timedelta(hours=6)
        out.append({"ep": x["ep"], "early": e.isoformat() if ok else None,
                    "mid": x["mid"], "pre": x["pre"],
                    "drop": None if ok else "early fails validity (mid-early<6h)"})
    return out


def prereg():
    os.makedirs(FROZEN, exist_ok=True)
    p = plan()
    n_valid = sum(1 for x in p if x["early"])
    spec = {
        "prereg": "PAUSE-PHASE-ORDINAL (EARLY vs MID vs PRE(-12h), same pause, same clock)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": "is the -12h!=MID texture difference (ledger affbdc1d) a late-pause-specific change or a monotone pause-phase gradient?",
        "design": "EARLY = first datetime >= prev_end+24h at the -12h PRE clock, 30min, shared fetch rule/cache; validity mid-early>=6h; scorer = LOGO(episode) RF(150,seed0) lean-duo trained on pre/mid of other pauses, held-out EARLY/MID/PRE scored by mean pre-prob",
        "primary": "UWE: exact binomial sign test, two-sided, on episode-wise ordering pre-prob(EARLY) vs pre-prob(MID); RIMD replication",
        "interpretation_map": {
            "early~mid AND late>mid persists": "late-pause-specific (precursor-side reading survives)",
            "early<mid significant": "monotone pause-phase gradient reading",
            "early>mid significant": "non-monotone/artifact flag, no positive claim",
        },
        "multiplicity": "second registered test family on the precursor-bank segments; -12h clock family chosen because the exploratory hit lives there; disambiguation, not a new discovery claim",
        "new_bank_rule": "bank_audit+gauntlet on (early vs mid) paired bank; checks 6/7 expected-FAIL paired (amendment 317bce69); gauntlet perm-p no verdict weight",
        "episode_table_sha256": _sha_text(json.dumps(EP_HST)),
        "plan_sha256": _sha_text(json.dumps(p)),
        "n_pauses_valid": n_valid,
    }
    with open(PREREG, "w") as f:
        json.dump({"spec": spec, "plan": p}, f, indent=1)
    tip = log_receipt("PAUSE-PHASE-ORDINAL-PREREG", spec)
    print(f"prereg written, valid pauses = {n_valid}/{len(p)}. ledger tip = {tip}",
          flush=True)


def _pre_check():
    assert os.path.exists(PREREG), "run prereg first (ledger before load!)"
    with open(PREREG) as f:
        pre = json.load(f)
    assert _sha_text(json.dumps(pre["plan"])) == pre["spec"]["plan_sha256"], "plan tampered"
    return pre["plan"]


def fetch():
    p = _pre_check()
    todo = [(sta, x["early"]) for sta in STATIONS for x in p if x["early"]]
    for i, (sta, t) in enumerate(todo):
        _fetch(sta, t)
        if (i + 1) % 25 == 0:
            print(f"[checkpoint] {i + 1}/{len(todo)} EARLY segments", flush=True)
    print(f"fetch complete: {len(todo)} EARLY segments (incl. cached/failed)", flush=True)


def run():
    from scipy.stats import binomtest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from audit import audit_bank
    from gauntlet import gauntlet
    p = _pre_check()
    rows = [x for x in p if x["early"]]
    results = {}
    for sta in STATIONS:
        # cached segments per episode: early/mid/pre (fetch-fails -> None)
        segs = {}
        for x in rows:
            s = {k: _fetch(sta, x[k]) for k in ("early", "mid", "pre")}
            if all(v is not None for v in s.values()):
                segs[x["ep"]] = s
        # NEW-bank rule on the (early vs mid) paired bank
        eb = [(w, lab, ep) for ep, s in segs.items()
              for lab in ("early", "mid") for w in s[lab]]
        name = f"PAUSEPHASE-{sta}"
        print(f"== {name}: bank_audit + gauntlet on early-vs-mid bank "
              f"(checks 6/7 expected-FAIL paired, gauntlet perm-p no weight)", flush=True)
        audit_bank(name, eb)
        gauntlet(name, eb, n_perm=60)
        # LOGO scoring: train pre/mid on other pauses, score held-out phases
        eps = sorted(segs)
        ord_em, ord_lm, mono, means = [], [], [], {"early": [], "mid": [], "pre": []}
        for ep in eps:
            raw = [(w, lab, e2) for e2 in eps if e2 != ep
                   for lab in ("pre", "mid") for w in segs[e2][lab]]
            L = lean_baseline(raw)
            y = np.array([r[1] for r in raw])
            clf = make_pipeline(StandardScaler(),
                                RandomForestClassifier(150, random_state=0, n_jobs=-1))
            clf.fit(L, y)
            pre_col = list(clf.classes_).index("pre")
            prob = {}
            for lab in ("early", "mid", "pre"):
                F = lean_baseline([(w, "?", 0) for w in segs[ep][lab]])
                prob[lab] = float(clf.predict_proba(F)[:, pre_col].mean())
                means[lab].append(prob[lab])
            if prob["early"] != prob["mid"]:
                ord_em.append(prob["early"] > prob["mid"])
            if prob["pre"] != prob["mid"]:
                ord_lm.append(prob["pre"] > prob["mid"])
            mono.append(prob["early"] < prob["mid"] < prob["pre"])
        k = int(sum(ord_em))
        p_em = float(binomtest(k, len(ord_em), 0.5, alternative="two-sided").pvalue) \
            if ord_em else 1.0
        r = {"role": "PRIMARY" if sta == "UWE" else "replication",
             "n_pauses": len(eps),
             "early_gt_mid": f"{k}/{len(ord_em)}", "sign_p_two_sided": p_em,
             "late_gt_mid_infold": f"{int(sum(ord_lm))}/{len(ord_lm)}",
             "monotone_triples": f"{int(sum(mono))}/{len(eps)}",
             "mean_pre_prob": {lab: round(float(np.mean(v)), 3)
                               for lab, v in means.items()}}
        results[sta] = r
        print(f"{name} [{r['role']}]: n={r['n_pauses']} early>mid {r['early_gt_mid']} "
              f"sign_p(2s)={p_em:.4f} | late>mid in-fold {r['late_gt_mid_infold']} | "
              f"monotone {r['monotone_triples']} | means {r['mean_pre_prob']}", flush=True)
    tip = log_receipt("PAUSE-PHASE-ORDINAL",
                      {"plan_sha256": _sha_text(json.dumps(p)), "results": results})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "fetch": fetch, "run": run,
     "plan": lambda: print(json.dumps(plan(), indent=1))}[sys.argv[1]]()
