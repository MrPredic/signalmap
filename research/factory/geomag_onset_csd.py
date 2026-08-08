"""ZEIT-FAKTOR / CRITICAL SLOWING DOWN on GEOMAGNETIC STORM ONSET (Kp>=5, G1+).

Same theory anchor as ims_csd.py / calce_csd.py (Scheffer/Dakos: approaching a
critical transition, AR1 AND variance of the slow-variable fluctuations should
rise). Storm onset is a genuine candidate bifurcation-like transition (magneto-
spheric substorm/ring-current threshold); this asks whether ground-magnetometer
X-component fluctuation dynamics show the CSD signature 12-24h BEFORE onset --
deliberately excluding the last 12h so the test is a real early-warning window,
not just "the storm ramp itself looks different."

DATA:
  - Historical Kp (3h cadence), GFZ Potsdam JSON API (same endpoint as
    geomag_fresh.py): https://kp.gfz.de/app/json/?start=...&end=...&index=Kp
  - Ground magnetometer 1Hz X component, USGS geomag web service (same source/
    URL pattern as geomag_fresh.py's _fetch, generalized to arbitrary start/end
    instead of only full UTC days): observatories BOU + FRD (both, for
    cross-station robustness; reuses OBS-pair convention from geomag_fresh.py).

EVENT DEFINITION (label-only, acquired from Kp BEFORE any waveform is loaded --
same discipline as geomag_fresh.py's pickdays(), which is what makes label
acquisition here, not readout):
  - ONSET (t): Kp[t] >= 5 AND Kp[t-1] < 5 AND max(Kp[t-8:t]) < 5 -- i.e. the
    first 3h bin of a G1+ storm, preceded by >=24h with no Kp>=5 (isolation:
    a clean pre-onset baseline, not a second bin of an already-running storm).
  - QUIET control: a run of >=16 consecutive 3h bins (>=48h) with Kp<3; pick
    the bin 30h into the run as a pseudo-onset reference (>=24h clean margin
    before it, matching the onset windowing exactly).
  - Period: 2019-01-01 to 2026-07-01 (fixed, reproducible; start pinned to
    where USGS 1Hz X coverage for BOU/FRD via this endpoint is reliable --
    spot-checked BEFORE prereg, 2015-2018 came back all-NaN in test pulls,
    a data-availability fact, not a readout/tuning decision). Subsampled
    (spread evenly, same stride idiom as geomag_fresh.pickdays) to ~20 onsets
    and ~20 quiet controls -- targeted comfortably above the "at least ~15"
    floor even after fetch drops.

PRE-REGISTERED DESIGN (ledger GEOMAG-ONSET-CSD-PREREG, frozen BEFORE any
magnetometer waveform is loaded):
  - WINDOW per event = [reference_time - 24h, reference_time - 12h) -- a 12h,
    43200-sample (1Hz) block strictly BEFORE the storm/pseudo-storm reference.
  - Slow variable HI[t] = raw 1Hz X (nT) over that window, per observatory.
  - Per-window CSD statistic = ims_csd._csd_bearing(HI) UNCHANGED (same
    Gaussian detrend sigma=0.05*N, rolling ar1/logvar window=0.5*N step=0.005*N,
    Kendall tau, N_surr=500 Fourier phase-randomized surrogate, seed 20260708,
    alpha=0.05) -- ims_csd.py is imported (pure functions only) and not edited.
  - PRIMARY, directed, per observatory: CSD+ rate (tau_ar1>0 & p_ar1<alpha &
    tau_var>0 & p_var<alpha) among ONSET windows > QUIET windows -- Fisher
    exact, alternative='greater'.
  - SPECIFICITY control: QUIET windows must not fire at the onset rate (this IS
    the "healthy/quiet periods must not fire" requirement).
  - CONFOUND guard: ar1-rise reported separately from var-rise; CONFIRMED
    needs BOTH.
  - HONEST-N gate: if n_onset_valid < 15 (after fetch drops) for an
    observatory, that observatory's verdict is forced to "INSUFFICIENT-N",
    never silently treated as a NULL or a PASS.
  - Claim scope: ground-based X-component fluctuation dynamics, 12-24h pre-
    onset window, BOU+FRD only (no claim beyond these two mid-latitude
    observatories).

USAGE (order enforces prereg-before-readout):
  python geomag_onset_csd.py pickevents   # derive onset/quiet times from Kp (labels only)
  python geomag_onset_csd.py prereg       # freeze design + event list + ledger
  python geomag_onset_csd.py fetch        # cache per-event 1Hz X windows (BOU+FRD)
  python geomag_onset_csd.py run          # CSD taus + surrogates + verdict, per obs
"""
import hashlib, json, os, sys, urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.stats import fisher_exact

from ims_csd import _csd_bearing, PARAMS as CSD_PARAMS
from receipt_ledger import log_receipt

ROOT = "<local-path>/signalmap"
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "frozen")
CACHE = os.path.join(HERE, "cache", "geomag_onset")
EVENTS_PATH = os.path.join(FROZEN, "geomag_onset_events.json")
PREREG_PATH = os.path.join(FROZEN, "geomag_onset_csd_prereg.json")
RESULT_PATH = os.path.join(HERE, "logs", "geomag_onset_csd_result.json")

KP_URL = ("https://kp.gfz.de/app/json/?start=2019-01-01T00:00:00Z"
          "&end=2026-07-01T00:00:00Z&index=Kp")
# period start 2019-01-01: USGS 1Hz X (this web-service endpoint) verified all-NaN
# for BOU/FRD in 2015-2018 spot checks, reliable from 2019-01 onward (checked
# BEFORE prereg -- data-availability logistics, not a readout/tuning decision).
OBS = ("BOU", "FRD")
WINDOW_START_H, WINDOW_END_H = -24, -12   # [-24h, -12h) before reference
N_TARGET = 20
MIN_N = 15


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


# ---------- pickevents (label acquisition, no waveform read) ----------
def pickevents():
    d = json.load(urllib.request.urlopen(KP_URL, timeout=180))
    times = [_parse(t) for t in d["datetime"]]
    kp = np.array([v if v is not None else np.nan for v in d["Kp"]])
    n = len(kp)

    onset_idx = [t for t in range(8, n)
                 if kp[t] >= 5 and kp[t - 1] < 5 and np.nanmax(kp[t - 8:t]) < 5]
    k = max(1, len(onset_idx) // N_TARGET)
    onset_pick = onset_idx[::k][:N_TARGET]

    below = kp < 3
    quiet_idx = []
    i = 0
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if j - i >= 16 and i + 10 < j:
                quiet_idx.append(i + 10)
            i = j
        else:
            i += 1
    k2 = max(1, len(quiet_idx) // N_TARGET)
    quiet_pick = quiet_idx[::k2][:N_TARGET]

    events = {
        "onset": [_iso(times[t]) for t in onset_pick],
        "quiet": [_iso(times[t]) for t in quiet_pick],
        "n_onset_raw": len(onset_idx), "n_quiet_raw": len(quiet_idx),
        "kp_period": [d["datetime"][0], d["datetime"][-1]],
    }
    os.makedirs(FROZEN, exist_ok=True)
    with open(EVENTS_PATH, "w") as f:
        json.dump(events, f, indent=1)
    print(f"onsets: {len(onset_idx)} raw -> picked {len(onset_pick)}", flush=True)
    print(f"quiet runs: {len(quiet_idx)} raw -> picked {len(quiet_pick)}", flush=True)
    print(f"written {EVENTS_PATH}")


# ---------- prereg ----------
def prereg():
    assert os.path.exists(EVENTS_PATH), "run pickevents first"
    events = json.load(open(EVENTS_PATH))
    spec = {
        "prereg": "GEOMAG-ONSET-CSD (critical-slowing-down early-warning, 12-24h pre-storm-onset, directed, theory-anchor, same anchor as IMS-CSD/CALCE-CSD)",
        "frozen_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "domain": "USGS geomag 1Hz X component, observatories BOU+FRD",
        "event_definition": {
            "onset": "Kp[t]>=5 & Kp[t-1]<5 & max(Kp[t-8:t])<5 (24h clean baseline, G1+)",
            "quiet": ">=48h consecutive Kp<3 run, reference = 30h into run",
            "period": "2015-01-01 to 2026-07-01 (fixed)",
            "subsample": f"stride pick, target N={N_TARGET} each (geomag_fresh.pickdays idiom)",
        },
        "window": f"[reference{WINDOW_START_H:+d}h, reference{WINDOW_END_H:+d}h) -- 12h, 1Hz X, per obs",
        "slow_variable": "HI[t] = raw 1Hz X (nT) in window",
        "params": CSD_PARAMS,
        "statistic": "ims_csd._csd_bearing(hi) UNCHANGED: Kendall tau of rolling ar1(t)/logvar(t) of Gaussian-detrended residual",
        "significance": "per window, Fourier phase-randomized surrogate x N_surr (ims_csd params); 1-sided p",
        "primary": "per obs: CSD+ rate (tau_ar1>0 & p_ar1<alpha & tau_var>0 & p_var<alpha) onset > quiet; Fisher exact greater",
        "specificity": "quiet windows must not fire at the onset rate",
        "confound_guard": "ar1-rise reported separately from var-rise; CONFIRMED needs BOTH",
        "honest_n_gate": f"if n_onset_valid < {MIN_N} after fetch drops -> verdict forced INSUFFICIENT-N (per obs)",
        "claim_scope": "ground X-component fluctuation dynamics, 12-24h pre-onset, BOU+FRD only",
        "n_onset_target": len(events["onset"]), "n_quiet_target": len(events["quiet"]),
        "events_sha256": _sha_text(json.dumps(events)),
    }
    with open(PREREG_PATH, "w") as f:
        json.dump({"spec": spec, "events": events}, f, indent=1)
    tip = log_receipt("GEOMAG-ONSET-CSD-PREREG", spec)
    print(f"prereg written. n_onset={len(events['onset'])} n_quiet={len(events['quiet'])}")
    print(f"ledger tip = {tip}")


def _pre_check():
    assert os.path.exists(PREREG_PATH), "run prereg first (ledger before load!)"
    pre = json.load(open(PREREG_PATH))
    assert _sha_text(json.dumps(pre["events"])) == pre["spec"]["events_sha256"], "events tampered"
    return pre


# ---------- fetch ----------
def _fetch_window(obs, start_dt, end_dt, tries=3):
    u = (f"https://geomag.usgs.gov/ws/data/?id={obs}"
         f"&starttime={_iso(start_dt)}&endtime={_iso(end_dt)}"
         f"&elements=X&sampling_period=1&format=json")
    for a in range(tries):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "signalmap/1.0"})
            d = json.load(urllib.request.urlopen(req, timeout=120))
            x = np.array([v if v is not None else np.nan
                          for v in d["values"][0]["values"]], float)
            x = x[np.isfinite(x)]
            return x.astype(np.float32)
        except Exception as e:
            if a == tries - 1:
                print(f"  {obs} {_iso(start_dt)}: FETCH-FAIL {e}", flush=True)
                return np.array([], dtype=np.float32)


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    events = json.load(open(EVENTS_PATH))
    for obs in OBS:
        for label, times in (("onset", events["onset"]), ("quiet", events["quiet"])):
            for t in times:
                ref = _parse(t)
                start = ref + timedelta(hours=WINDOW_START_H)
                end = ref + timedelta(hours=WINDOW_END_H)
                tag = t.replace(":", "").replace("-", "")
                out = os.path.join(CACHE, f"{obs}_{label}_{tag}.npz")
                if os.path.exists(out):
                    continue
                x = _fetch_window(obs, start, end)
                np.savez_compressed(out, X=x)
        print(f"{obs}: fetch done", flush=True)
    print("fetch complete")


# ---------- run ----------
def run():
    pre = _pre_check()
    events = pre["events"]
    a = CSD_PARAMS["alpha"]
    all_summaries = {}
    for obs in OBS:
        rows = []
        for label, times in (("onset", events["onset"]), ("quiet", events["quiet"])):
            for t in times:
                tag = t.replace(":", "").replace("-", "")
                fp = os.path.join(CACHE, f"{obs}_{label}_{tag}.npz")
                if not os.path.exists(fp):
                    rows.append({"t": t, "label": label, "status": "MISSING"})
                    continue
                x = np.load(fp)["X"]
                if len(x) < 43200 * 0.9:   # expect ~43201 samples for 12h@1Hz
                    rows.append({"t": t, "label": label, "status": "SHORT", "n": int(len(x))})
                    continue
                t_ar1, t_var, p_ar1, p_var = _csd_bearing(x)
                ar1_ok = t_ar1 > 0 and p_ar1 < a
                var_ok = t_var > 0 and p_var < a
                rows.append({"t": t, "label": label, "status": "OK", "n": int(len(x)),
                             "tau_ar1": round(t_ar1, 3), "p_ar1": round(p_ar1, 4),
                             "tau_var": round(t_var, 3), "p_var": round(p_var, 4),
                             "ar1_rise": ar1_ok, "var_rise": var_ok,
                             "csd_pass": bool(ar1_ok and var_ok)})
                print(f"{obs} {label:5s} {t} N={len(x):5d} "
                      f"tau_ar1={t_ar1:+.3f}(p={p_ar1:.3f}) tau_var={t_var:+.3f}(p={p_var:.3f}) "
                      f"{'CSD+' if rows[-1]['csd_pass'] else '.'}", flush=True)

        onset_rows = [r for r in rows if r["label"] == "onset" and r["status"] == "OK"]
        quiet_rows = [r for r in rows if r["label"] == "quiet" and r["status"] == "OK"]
        n_on, n_q = len(onset_rows), len(quiet_rows)
        if n_on < MIN_N or n_q < MIN_N:
            summary = {"n_onset_valid": n_on, "n_quiet_valid": n_q,
                       "verdict": "INSUFFICIENT-N",
                       "note": f"need >={MIN_N} valid onset AND quiet windows"}
        else:
            fp_ = sum(r["csd_pass"] for r in onset_rows)
            hp_ = sum(r["csd_pass"] for r in quiet_rows)
            odds, fisher_p = fisher_exact([[fp_, n_on - fp_], [hp_, n_q - hp_]],
                                          alternative="greater")
            directed = fp_ / n_on > hp_ / n_q
            verdict = ("CSD-CONFIRMED" if directed and fisher_p < 0.05 else
                       "CSD-PARTIAL" if directed and fisher_p < 0.10 else "CSD-NULL")
            summary = {
                "n_onset_valid": n_on, "n_quiet_valid": n_q,
                "csd_pass_onset": f"{fp_}/{n_on}", "csd_pass_quiet": f"{hp_}/{n_q}",
                "fisher_p_onset_gt_quiet": round(float(fisher_p), 4),
                "ar1_rise_onset": f"{sum(r['ar1_rise'] for r in onset_rows)}/{n_on}",
                "var_rise_onset": f"{sum(r['var_rise'] for r in onset_rows)}/{n_on}",
                "verdict": verdict,
            }
        print(f"\n== SUMMARY [{obs}] ==")
        for kk, v in summary.items():
            print(f"  {kk}: {v}")
        all_summaries[obs] = {"summary": summary, "rows": rows}

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(all_summaries, f, indent=1)
    tip = log_receipt("GEOMAG-ONSET-CSD-RESULT",
                      {obs: all_summaries[obs]["summary"] for obs in OBS})
    print(f"\nledger tip = {tip}")


if __name__ == "__main__":
    {"pickevents": pickevents, "prereg": prereg, "fetch": fetch, "run": run}[sys.argv[1]]()
