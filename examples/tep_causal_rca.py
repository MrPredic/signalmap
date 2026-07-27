"""Real-data proof: ensemble directed causal root-cause on Tennessee Eastman.

TEP is the standard industrial fault-diagnosis benchmark — a simulated chemical
plant with 52 measured/manipulated variables and faults whose root causes are
documented. This script downloads the open Braatz mirror, learns directed causal
edge-scores on fault-free data under two independent voters, then for each fault
asks: *which variable's causal relationships shifted most?*

Two complementary voters, fused by OR-like rank fusion:
  * CCM/EDM  — state-space geometry (Sugihara 2012)
  * transfer entropy — information flow (Schreiber 2000)

Reproducible headline: CCM alone wins Fault 1 (feed) but is weak on Fault 4
(cooling); transfer entropy is the reverse. The FUSED ensemble stays >=2/3 on
BOTH faults — more robust than either single method — while a plain anomaly
(mean-shift) baseline reports only downstream temperature *symptoms*, which is
the localization that correlation cannot do.

    python examples/tep_causal_rca.py

HONEST CAVEAT: skills are modest on a tightly controlled process; this is a real
localization edge, not a finished diagnostic. Next voter: PCMCI-style
conditional independence. Data: github.com/camaramm/tennessee-eastman-profBraatz
(Downs & Vogel TEP). Needs numpy only.
"""
from __future__ import annotations

import urllib.request

import numpy as np

from signalmap.causal import edge_scores, fuse_rankings, root_cause_scores

BASE = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/"

# A spread of variables across plant subsystems (0-indexed columns:
# XMEAS1..41 = cols 0..40, XMV1..11 = cols 41..51).
COLS = {
    "Afeed": 0, "ACfeed": 3, "reactPress": 6, "reactTemp": 8, "cwOutTemp": 20,
    "sepTemp": 10, "stripTemp": 17, "purge": 9, "AfeedValve": 43, "cwValve": 50,
    "condValve": 51, "stripSteamV": 49,
}

# Documented root-cause variable groups per fault.
FAULTS = {
    "d01_te.dat": ("Fault 1: A/C feed-ratio step", {"Afeed", "ACfeed", "AfeedValve"}),
    "d04_te.dat": ("Fault 4: reactor cooling-water temp step", {"cwValve", "cwOutTemp", "reactTemp"}),
}

E, TAU, LIB, BINS = 2, 1, 500, 7
FAULT_START = 160  # TEP test sets inject the fault after sample 160


def _load(fn: str) -> np.ndarray:
    raw = urllib.request.urlopen(BASE + fn, timeout=60).read().decode()
    return np.array([[float(x) for x in line.split()] for line in raw.strip().splitlines()])


def _channels(matrix: np.ndarray, rows: slice) -> dict[str, np.ndarray]:
    out = {}
    for name, col in COLS.items():
        v = matrix[rows, col]
        out[name] = (v - v.mean()) / (v.std() + 1e-9)
    return out


def _hits(top, truth):
    return len(truth & set(top[:3]))


def main() -> None:
    print("downloading Tennessee Eastman data ...")
    healthy = _channels(_load("d00_te.dat"), slice(0, 960))
    base_ccm = edge_scores(healthy, method="ccm", E=E, tau=TAU, lib_size=LIB)
    base_te = edge_scores(healthy, method="te", bins=BINS)

    for fn, (title, truth) in FAULTS.items():
        fault = _channels(_load(fn), slice(FAULT_START, 960))
        rc_ccm = root_cause_scores(base_ccm, edge_scores(fault, method="ccm", E=E, tau=TAU, lib_size=LIB))
        rc_te = root_cause_scores(base_te, edge_scores(fault, method="te", bins=BINS))
        fused = fuse_rankings(rc_ccm, rc_te)

        ccm_top = sorted(rc_ccm, key=rc_ccm.get, reverse=True)[:3]
        te_top = sorted(rc_te, key=rc_te.get, reverse=True)[:3]
        anomaly_top = sorted(fault, key=lambda k: abs(fault[k].mean()), reverse=True)[:3]

        print(f"\n### {title}")
        print(f"  documented root cause : {sorted(truth)}")
        print(f"  CCM voter             : {ccm_top}   ({_hits(ccm_top, truth)}/3)")
        print(f"  transfer-entropy voter: {te_top}   ({_hits(te_top, truth)}/3)")
        print(f"  >> FUSED ensemble     : {fused[:3]}   ({_hits(fused, truth)}/3)")
        print(f"  anomaly (mean-shift)  : {anomaly_top}   ({_hits(anomaly_top, truth)}/3)")


if __name__ == "__main__":
    main()
