"""ZEIT-FAKTOR / CRITICAL SLOWING DOWN on CALCE-labelled NMC battery calendar-aging
capacity/voltage traces (OSF j2sn4, the same 22-24-cell bank as battery_pipeline.py's
BATTERY-TRANSFER / BATTERY-SOH physics; that codebase-internal shorthand calls the
SOH-tertile result "CALCE-soh" -- this is the same family, new angle: TIME not device).

Context: ims_csd.py (bearing run-to-failure) tested the Scheffer/Dakos CSD signature
where it should be textbook-present -- verdict was CSD-NULL on both preregistered
indicators (RESULTS.md section ZEIT-FAKTOR -- IMS-RUL). This is the next
pre-decided fallback family on the SAME theory anchor: does fluctuation-dynamics
CSD appear as a NMC cell approaches end-of-life during calendar+cycling aging?

DATA: data/battery_eis/NCM_Calendar_and_Cycling_Dataset/Capacity_data/
  CD_t25_{0,10,20,40}d.xlsx + CDS_t25_{70,90}d.xlsx -- one file per aging-CHECKPOINT
  (day), one sheet per cell (S1..S24, 24 cells day0/10/20, 23 day40; severe
  survivorship dropout to 3 cells S5/S6/S7 by day70/90 -- documented, not hidden).
  Each sheet is a "Folded Graph" charge/discharge test at that checkpoint: columns
  Cchg/Vchg0/Cdcg/Vdcg0, ~28800 rows. Cdcg/Vdcg0 are a valid CONTIGUOUS prefix block
  (~3300-4000 chronological in-test samples; verified, not interleaved with the
  charge segment) -- a genuine chronological trace, reused as the CSD slow-variable
  exactly like an IMS snapshot RMS trajectory, just with in-test sample index as the
  clock instead of run-to-failure index.

UNIT OF ANALYSIS = one CHECKPOINT (cell, day). Life-tercile framing reuses the
codebase's OWN existing split (battery_pipeline.FRESH_DAYS={0,10,20} vs
AGED_DAYS={40,70,90}, unchanged, imported not redefined): FRESH checkpoints =
early-life ("first tercile"), AGED = late-life ("last tercile"). n=72 fresh
checkpoints, n=29 aged checkpoints (counts fixed by the manifest, frozen below).

PRE-REGISTERED DESIGN (ledger CALCE-CSD-PREREG, frozen BEFORE any Cdcg/Vdcg0 loaded):
  - Two indicators (rms/kurt analog): 'cdcg' = discharge-capacity trace (the literal
    "Kapazitätstrajektorie"), 'vdcg' = discharge-voltage trace (orthogonal marker).
  - Per-checkpoint CSD statistic = ims_csd._csd_bearing(hi) UNCHANGED (same Gaussian
    detrend sigma=0.05*N, same rolling ar1/logvar window=0.5*N step=0.005*N, same
    Kendall tau, same N_surr=500 Fourier phase-randomized surrogate, same seed
    20260708, same alpha=0.05) -- ims_csd.py itself is not imported for side effects,
    only its pure statistic function + PARAMS, and is not edited.
  - PRIMARY, directed: CSD+ rate (tau_ar1>0 & p_ar1<alpha & tau_var>0 & p_var<alpha)
    among AGED checkpoints > FRESH checkpoints -- Fisher exact, alternative='greater'.
    Mirrors ims_csd's failed-vs-healthy design one-for-one (aged<->failed,
    fresh<->healthy).
  - SPECIFICITY control: FRESH (early-life) checkpoints must NOT fire at the aged
    rate -- this IS the life-tercile "first tercile should not show the signature"
    check the task asked for, operationalized via the same Fisher test.
  - CONFOUND guard: ar1-rise reported separately from var-rise; CONFIRMED needs BOTH
    (identical to ims_csd).
  - Claim scope: NMC calendar-aging fluctuation dynamics within one charge/discharge
    test per checkpoint; n_aged=29 (vs IMS n_failed=4) gives materially more power
    than IMS but the checkpoint-level pseudo-replication within cell/day is noted
    (fresh/aged are file-level population blocks, not the exact same 4 discrete
    bearings-per-rig structure as IMS).

USAGE (order enforces prereg-before-readout):
  python calce_csd.py prereg cdcg   # (or vdcg) freeze design + ledger
  python calce_csd.py fetch         # cache per-checkpoint Cdcg/Vdcg0 traces
  python calce_csd.py run cdcg      # (or vdcg) CSD taus + surrogates + verdict
"""
import hashlib, json, os, re, sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from ims_csd import _csd_bearing, PARAMS as CSD_PARAMS
from battery_pipeline import FRESH_DAYS, AGED_DAYS
from receipt_ledger import log_receipt

ROOT = "<local-path>/signalmap"
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
CACHE = os.path.join(HERE, "cache", "calce_csd")
DATADIR = f"{ROOT}/data/battery_eis/NCM_Calendar_and_Cycling_Dataset/Capacity_data"

DAY_FILES = {0: "CD_t25_0d.xlsx", 10: "CD_t25_10d.xlsx", 20: "CD_t25_20d.xlsx",
             40: "CD_t25_40d.xlsx", 70: "CDS_t25_70d.xlsx", 90: "CDS_t25_90d.xlsx"}
INDICATORS = ("cdcg", "vdcg")
COL_NAME = {"cdcg": "Cdcg", "vdcg": "Vdcg0"}


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _prereg_path(ind):
    return os.path.join(FROZEN, f"calce_csd_{ind}_prereg.json")


def _result_path(ind):
    return os.path.join(HERE, "logs", f"calce_csd_{ind}_result.json")


def _cell_from_sheet(sh):
    return int(re.match(r"S(\d+)", sh).group(1))


def _manifest():
    """Frozen provenance: per-day sheet (cell) names present -- structure only,
    no waveform values read (cheap: openpyxl sheet listing, not full parse)."""
    out = {}
    for day, fn in DAY_FILES.items():
        xl = pd.ExcelFile(os.path.join(DATADIR, fn))
        out[day] = list(xl.sheet_names)
    return out


def _checkpoints(manifest=None):
    manifest = manifest or _manifest()
    out = []
    for day in sorted(manifest):
        label = "fresh" if day in FRESH_DAYS else "aged" if day in AGED_DAYS else None
        assert label, f"day {day} not in FRESH_DAYS/AGED_DAYS"
        for sh in manifest[day]:
            out.append({"day": day, "cell": _cell_from_sheet(sh), "sheet": sh, "label": label})
    return sorted(out, key=lambda r: (r["day"], r["cell"]))


def prereg(ind="cdcg"):
    assert ind in INDICATORS
    os.makedirs(FROZEN, exist_ok=True)
    manifest = _manifest()
    cps = _checkpoints(manifest)
    n_fresh = sum(1 for c in cps if c["label"] == "fresh")
    n_aged = sum(1 for c in cps if c["label"] == "aged")
    spec = {
        "prereg": f"CALCE-CSD-{ind.upper()} (critical-slowing-down early-warning on NMC "
                  "calendar-aging, directed, theory-anchor, same anchor as IMS-CSD)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "domain": "OSF j2sn4 NCM_Calendar_and_Cycling_Dataset, per-checkpoint charge/discharge test",
        "indicator": ind,
        "slow_variable": f"HI[t] = {COL_NAME[ind]} in-test chronological trace at one (cell, day) checkpoint",
        "life_tercile_split": "reuses battery_pipeline.FRESH_DAYS={0,10,20} (early/first) vs AGED_DAYS={40,70,90} (late/last)",
        "params": CSD_PARAMS,
        "statistic": "ims_csd._csd_bearing(hi) UNCHANGED: Kendall tau of rolling ar1(t)/logvar(t) of Gaussian-detrended residual",
        "significance": "per checkpoint, Fourier phase-randomized surrogate of residual x N_surr (ims_csd params); 1-sided p",
        "primary": "per checkpoint PASS = tau_ar1>0 & p_ar1<alpha & tau_var>0 & p_var<alpha; headline = CSD+ rate AGED vs FRESH",
        "specificity": "Fisher exact (CSD+/CSD-) x (aged/fresh), alternative=greater; fresh(early-life) must not fire at aged rate",
        "confound_guard": "ar1-rise reported separately from var-rise; CONFIRMED needs BOTH",
        "orthogonality": "directed dynamical-systems statistic, no learning/no selection; same anchor+method as IMS-CSD",
        "claim_scope": f"NMC calendar-aging fluctuation dynamics; n_fresh={n_fresh} n_aged={n_aged} (severe cell dropout by day70/90, documented)",
        "checkpoints_sha256": _sha_text(json.dumps(cps)),
        "manifest_sha256": _sha_text(json.dumps(manifest)),
        "n_checkpoints": {"fresh": n_fresh, "aged": n_aged},
    }
    with open(_prereg_path(ind), "w") as f:
        json.dump({"spec": spec, "checkpoints": cps}, f, indent=1)
    tip = log_receipt(f"CALCE-CSD-{ind.upper()}-PREREG", spec)
    print(f"prereg[{ind}] written. n_fresh={n_fresh} n_aged={n_aged}")
    print(f"ledger tip = {tip}")


def _pre_check(ind="cdcg"):
    assert os.path.exists(_prereg_path(ind)), "run prereg first (ledger before load!)"
    with open(_prereg_path(ind)) as f:
        pre = json.load(f)
    assert _sha_text(json.dumps(pre["checkpoints"])) == pre["spec"]["checkpoints_sha256"], "checkpoints tampered"
    assert _sha_text(json.dumps(_manifest())) == pre["spec"]["manifest_sha256"], "file manifest changed since freeze"
    return pre


def _find_header_row(df, want="Cdcg", max_scan=10):
    for i in range(min(max_scan, len(df))):
        if want in df.iloc[i].tolist():
            return i
    raise ValueError(f"header row with '{want}' not found in first {max_scan} rows")


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    manifest = _manifest()
    for day, fn in DAY_FILES.items():
        xl = pd.ExcelFile(os.path.join(DATADIR, fn))
        n_done = n_new = 0
        for sh in manifest[day]:
            cell = _cell_from_sheet(sh)
            out = os.path.join(CACHE, f"{day}_{cell}.npz")
            if os.path.exists(out):
                n_done += 1
                continue
            df = xl.parse(sh, header=None)
            hdr = _find_header_row(df)
            row = df.iloc[hdr].tolist()
            ci, vi = row.index("Cdcg"), row.index("Vdcg0")
            data = df.iloc[hdr + 1:]
            cdcg = pd.to_numeric(data[ci], errors="coerce").to_numpy()
            vdcg = pd.to_numeric(data[vi], errors="coerce").to_numpy()
            cdcg = cdcg[np.isfinite(cdcg)].astype(np.float32)
            vdcg = vdcg[np.isfinite(vdcg)].astype(np.float32)
            np.savez_compressed(out, cdcg=cdcg, vdcg=vdcg, day=day, cell=cell)
            n_new += 1
        print(f"day={day:2d}: {n_new} new, {n_done} cached", flush=True)
    print("fetch complete")


def run(ind="cdcg"):
    pre = _pre_check(ind)
    a = CSD_PARAMS["alpha"]
    rows = []
    for cp in pre["checkpoints"]:
        z = np.load(os.path.join(CACHE, f"{cp['day']}_{cp['cell']}.npz"))
        hi = z[ind]
        t_ar1, t_var, p_ar1, p_var = _csd_bearing(hi)
        ar1_ok = t_ar1 > 0 and p_ar1 < a
        var_ok = t_var > 0 and p_var < a
        row = {"day": cp["day"], "cell": cp["cell"], "label": cp["label"],
               "n": int(len(hi)), "tau_ar1": round(t_ar1, 3), "p_ar1": round(p_ar1, 4),
               "tau_var": round(t_var, 3), "p_var": round(p_var, 4),
               "ar1_rise": ar1_ok, "var_rise": var_ok, "csd_pass": bool(ar1_ok and var_ok)}
        rows.append(row)
        print(f"day={cp['day']:2d} cell={cp['cell']:2d} {cp['label']:5s} N={row['n']:5d} "
              f"tau_ar1={t_ar1:+.3f}(p={p_ar1:.3f}) tau_var={t_var:+.3f}(p={p_var:.3f}) "
              f"{'CSD+' if row['csd_pass'] else '.'}", flush=True)

    aged = [r for r in rows if r["label"] == "aged"]
    fresh = [r for r in rows if r["label"] == "fresh"]
    fp = sum(r["csd_pass"] for r in aged)
    hp = sum(r["csd_pass"] for r in fresh)
    odds, fisher_p = fisher_exact([[fp, len(aged) - fp], [hp, len(fresh) - hp]],
                                  alternative="greater")
    directed = fp / max(len(aged), 1) > hp / max(len(fresh), 1)
    verdict = ("CSD-CONFIRMED" if directed and fisher_p < 0.05 else
               "CSD-PARTIAL" if directed and fisher_p < 0.10 else "CSD-NULL")
    summary = {
        "csd_pass_aged": f"{fp}/{len(aged)}",
        "csd_pass_fresh": f"{hp}/{len(fresh)}",
        "fisher_p_aged_gt_fresh": round(float(fisher_p), 4),
        "ar1_rise_aged": f"{sum(r['ar1_rise'] for r in aged)}/{len(aged)}",
        "var_rise_aged": f"{sum(r['var_rise'] for r in aged)}/{len(aged)}",
        "verdict": verdict,
        "indicator": ind,
        "note": "n_aged=29 (vs IMS n_failed=4); verdict thresholds documented in code (directed + p<0.05 CONFIRMED, p<0.10 PARTIAL)",
    }
    print(f"\n== SUMMARY [{ind}] ==")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    rp = _result_path(ind)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    tip = log_receipt(f"CALCE-CSD-{ind.upper()}-RESULT", summary)
    print(f"ledger tip = {tip}")


if __name__ == "__main__":
    fn, ind = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "cdcg")
    {"prereg": lambda: prereg(ind), "fetch": fetch, "run": lambda: run(ind)}[fn]()
