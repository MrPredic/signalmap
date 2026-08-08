"""PRECURSOR IMPULSE-SAMPLING second family test (time-factor program, Prio 2b b).

Methods thesis #1 (window/sampling mode): impulse-borne physics is missed by
even windows (DCASE-valve 0.238->0.875 with sampling='peak'). The registered
precursor PRIMARY (-2h, lean duo on full 1024-windows) came out NULL both
stations (ledger affbdc1d). Question here: does an IMPULSE-SAMPLING readout
(peak-energy subwindow, the registered proxy `_sub(w,'peak')` from
readout_screen.py, verbatim) read a -2h difference that the full-window lean
readout cannot see?

PRE-REGISTERED (ledger PRECURSOR-SAMPLING-PREREG written by `prereg` BEFORE
any variant feature is computed; NO new data — cached, audited precursor
segments only):
  - Banks: the frozen VOLCANO-PRECURSOR paired banks, all offsets.
    PRIMARY: UWE@-2h with peak_sub readout; RIMD@-2h replication;
    -6h/-12h exploratory descriptive only.
  - Readout: lean duo on peak-energy subwindow (len//4=256 samples at max
    rolling std, step L//4 — readout_screen._sub verbatim, same rule for
    every class). Scorer/criterion identical to the amended precursor
    verdict: LOGO(episode) RF(150,seed0) segment-majority acc,
    cluster-bootstrap CI over episodes, exact paired sign test (one-sided);
    PASS = CI-lo>0.5 AND sign_p<0.05.
  - MULTIPLICITY, registered: readout family #2 on these banks (the lean
    full-window family was #1 and its -2h NULL STANDS as registered);
    together with PAUSE-PHASE-ORDINAL this is the 3rd registered analysis
    on the precursor segments. A peak_sub-only hit is EVIDENCE FOR THE
    METHOD THESIS and needs its own prospective confirmation before any
    precursor claim — it does NOT overturn the registered -2h lean NULL.
  - Bank identity: segments/labels/groups byte-identical to the audited
    banks (bank_audit + gauntlet ran at -2h under the NEW-bank rule);
    readout variant only -> no re-audit, gauntlet not run (registered).

Usage:
  .venv-research/bin/python research/factory/precursor_sampling.py prereg
  nice -n 19 .venv-research/bin/python research/factory/precursor_sampling.py run
"""
import hashlib, json, os, sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readout_screen import _sub
from receipt_ledger import log_receipt
from volcano_precursor import OFFSETS_H, _bank, _pre_check, _verdict
from volcano_fresh import STATIONS

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "frozen", "precursor_sampling_prereg.json")


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def prereg():
    p = _pre_check()  # verifies the frozen precursor plan hash
    spec = {
        "prereg": "PRECURSOR-SAMPLING (impulse/peak_sub readout, methods thesis #1, second readout family)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": "does an impulse-sampling readout read a -2h PRE/MID difference that the full-window lean readout (registered NULL, ledger affbdc1d) cannot see?",
        "readout": "lean duo on readout_screen._sub(w,'peak') subwindow (len//4, max rolling std, step L//4), verbatim; scorer/criterion identical to amended precursor verdict (LOGO RF(150,seed0), segment-majority, cluster-bootstrap CI, exact paired sign test one-sided); PASS=CI-lo>0.5 AND sign_p<0.05",
        "primary": "UWE@-2h peak_sub; RIMD@-2h replication; -6h/-12h exploratory descriptive",
        "multiplicity": "readout family #2 on these banks; 3rd registered analysis on the precursor segments overall; -2h lean NULL stands as registered; a peak_sub-only hit = method-thesis evidence, needs prospective confirmation, no precursor claim",
        "bank_identity": "cached audited precursor banks, readout variant only, no re-audit, no gauntlet",
        "plan_sha256_ref": "frozen volcano_precursor plan (verified at prereg and run)",
    }
    with open(PREREG, "w") as f:
        json.dump(spec, f, indent=1)
    tip = log_receipt("PRECURSOR-SAMPLING-PREREG", spec)
    print(f"prereg written. ledger tip = {tip}", flush=True)


def run():
    assert os.path.exists(PREREG), "run prereg first (ledger before readout!)"
    p = _pre_check()
    results = {}
    for sta in STATIONS:
        for off in OFFSETS_H:
            raw = _bank(sta, off, p)  # cache-only
            raw_sub = [(_sub(np.asarray(w, float), "peak"), lab, ep)
                       for w, lab, ep in raw]
            primary = off == 2
            r = _verdict(raw_sub, final=primary)
            r["role"] = "PRIMARY" if (primary and sta == "UWE") else \
                        "replication" if primary else "exploratory"
            r["verdict"] = ("PASS" if r["ci"][0] > 0.5 and r["sign_p"] < 0.05
                            else "NULL") if primary else "exploratory"
            name = f"PRECSAMP-{sta}-{off}h"
            results[name] = r
            print(f"{name} [{r['role']}]: n={r['n_segments']} acc={r['seg_acc']:.3f} "
                  f"CI[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] pairs={r['pairs_concordant']} "
                  f"sign_p={r['sign_p']:.4f} -> {r['verdict']}", flush=True)
    tip = log_receipt("PRECURSOR-SAMPLING", {"results": results})
    print(f"ledger tip = {tip}", flush=True)


if __name__ == "__main__":
    {"prereg": prereg, "run": run}[sys.argv[1]]()
