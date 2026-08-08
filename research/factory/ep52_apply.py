"""Apply the frozen EP52 readout after HVO documents the episode times.

No hypothesis selection happens here. The caller must provide the authoritative
HVO onset, overflow start, and deep-pause control dates explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ep52_retro import _fetch  # noqa: E402
from receipt_ledger import log_receipt  # noqa: E402

STATIONS = ("UWE", "RIMD")
OFFSETS_H = (1, 6, 12)


def planned_times(onset: datetime, overflow_start: datetime):
    if overflow_start >= onset:
        raise ValueError("overflow must start before documented onset")
    return [{
        "offset_h": off,
        "pre": (onset.replace(microsecond=0) - timedelta(hours=off)).isoformat(),
    } for off in OFFSETS_H]


def psd_slope(x):
    from scipy.signal import welch
    f, p = welch(x, nperseg=min(256, len(x)))
    return float(np.polyfit(np.log(f[1:]), np.log(p[1:] + 1e-20), 1)[0])


def rqa_det(x):
    from signalmap.premium import rqa_features
    return float(rqa_features(x)[1])


def _mean_feature(sta, t0, fn):
    x = _fetch(sta, t0)
    if x is None:
        return None
    return float(np.mean([fn(w) for w in x]))


def apply(onset, overflow_start, control_days):
    plan = planned_times(onset, overflow_start)
    results = {sta: {"primary": [], "secondary_rqa_24h": None} for sta in STATIONS}
    for sta in STATIONS:
        for item in plan:
            pre = item["pre"]
            controls = []
            clock = pre[11:16]
            for day in control_days:
                controls.append(_mean_feature(sta, f"{day}T{clock}", psd_slope))
            pv = _mean_feature(sta, pre, psd_slope)
            valid = pv is not None and all(v is not None for v in controls)
            row = {"offset_h": item["offset_h"], "pre": pv,
                   "controls": controls, "valid": valid,
                   "lower_than_control": bool(valid and pv < float(np.mean(controls)))}
            results[sta]["primary"].append(row)
        # Registered secondary: fixed T-24h RQA, no selection or pass gate.
        pre24 = (onset - timedelta(hours=24)).isoformat()
        results[sta]["secondary_rqa_24h"] = _mean_feature(sta, pre24, rqa_det)
    pass_by_station = {
        sta: sum(r["lower_than_control"] for r in res["primary"] if r["valid"]) >= 2
        for sta, res in results.items()
    }
    verdict = "PASS" if all(pass_by_station.values()) else "FAIL"
    receipt = {"episode": "EP52", "onset": onset.isoformat(),
               "overflow_start": overflow_start.isoformat(),
               "control_days": list(control_days), "plan": plan,
               "results": results, "pass_by_station": pass_by_station,
               "verdict": verdict,
               "prereg": "research/factory/PREREG_EP52.md"}
    log_receipt("EP52-APPLY", receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onset", required=True, help="HVO-documented UTC ISO time")
    ap.add_argument("--overflow-start", required=True, help="HVO-documented UTC ISO time")
    ap.add_argument("--control-day", action="append", required=True,
                    dest="controls", help="deep pause date, repeat at least twice")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    onset = datetime.fromisoformat(args.onset)
    overflow = datetime.fromisoformat(args.overflow_start)
    plan = planned_times(onset, overflow)
    if args.dry_run:
        print(json.dumps({"episode": "EP52", "plan": plan,
                          "controls": args.controls}, indent=2))
        return
    if len(args.controls) < 2:
        ap.error("at least two explicit control days are required")
    print(json.dumps(apply(onset, overflow, args.controls), indent=2))


if __name__ == "__main__":
    main()
