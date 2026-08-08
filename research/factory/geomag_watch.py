"""Zero-token GEOMAG-storm watcher (launchd, hourly) -- companion to
usgs_ep51_watch.py, same pattern applied to the GEOMAG-Onset time-factor
line (Zeit-Faktor Prio 2b(c), after IMS-RUL CSD-NULL).

Source: NOAA SWPC planetary K-index JSON (public, no auth, no LLM calls):
  https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
Rolling ~few-day window of 3h-cadence Kp values. An ONSET = a 3h-bin
transition from Kp<5 (below storm threshold) to Kp>=5 (G1+ storm) --
exactly the kind of prospectively-caught, tamper-evident transition event
the CSD-style time-factor tests need (mirrors "onset" in IMS-RUL failure
and Kilauea episode-start). Building this catalog NOW, before any CSD
prereg is frozen for GEOMAG, keeps the eventual test genuinely prospective
for any onset caught after the watcher started -- same prereg-before-
readout discipline as the rest of the project, just applied to data
collection instead of analysis.

Result: session start reads one local log instead of a web fetch; new
onsets accumulate for free while the watcher runs. Zero tokens, zero LLM.

Install (once):  .venv-research/bin/python geomag_watch.py install
Run once (test): .venv-research/bin/python geomag_watch.py run
Status:          tail logs/geomag_watch.jsonl
"""
import json, os, sys, time, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
FEED = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
LOG = os.path.join(HERE, "logs", "geomag_watch.jsonl")
PLIST = os.path.expanduser("~/Library/LaunchAgents/com.signalmap.geomagwatch.plist")
PY = "<local-path>/signalmap/.venv-research/bin/python"
ONSET_KP = 5.0  # G1+ storm threshold, matches prior GEOMAG-fresh Kp>=7 convention (loosened to catch more onsets)
RETRIES = 2
BACKOFF_S = 5


def _fetch_retry():
    last_err = None
    for i in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(FEED, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            last_err = e
            if i < RETRIES:
                time.sleep(BACKOFF_S)
    raise last_err


def _known_onsets():
    """Onset timestamps already seen in the log (dedup across polls)."""
    if not os.path.exists(LOG):
        return set()
    seen = set()
    with open(LOG) as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            for o in e.get("onsets_seen", []):
                seen.add(o)
    return seen


def _blackout_streak():
    if not os.path.exists(LOG):
        return 0
    with open(LOG) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    n = 0
    for e in reversed(lines):
        if e.get("fetch_error"):
            n += 1
        else:
            break
    return n


def run():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {"utc": now}
    try:
        rows = _fetch_retry()  # list of {"time_tag":..., "Kp":..., "a_running":..., "station_count":...}
        pts = [(r["time_tag"], float(r["Kp"])) for r in rows if r.get("Kp") is not None]
        pts.sort()
        onsets = [t for (t, kp), (pt, pkp) in zip(pts[1:], pts[:-1])
                  if pkp < ONSET_KP <= kp]
        known = _known_onsets()
        new_onsets = [t for t in onsets if t not in known]
        entry.update({
            "n_points": len(pts),
            "latest": {"time_tag": pts[-1][0], "kp": pts[-1][1]} if pts else None,
            "onset_kp_threshold": ONSET_KP,
            "onsets_seen": onsets,          # all onsets currently visible in the feed window
            "new_onsets": new_onsets,        # onsets not logged in any prior poll -> act on these
        })
    except Exception as e:
        entry["fetch_error"] = str(e)
    entry["blackout_streak"] = _blackout_streak() + 1 if entry.get("fetch_error") else 0
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    out = {k: entry.get(k) for k in ("utc", "latest", "new_onsets", "blackout_streak")}
    print(json.dumps(out))
    if entry["blackout_streak"] >= 6:
        print(f"WARN: geomag_watch blackout streak = {entry['blackout_streak']} polls", file=sys.stderr)


def install():
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
 <key>Label</key><string>com.signalmap.geomagwatch</string>
 <key>ProgramArguments</key><array>
  <string>{PY}</string>
  <string>{os.path.join(HERE, 'geomag_watch.py')}</string>
  <string>run</string>
 </array>
 <key>StartInterval</key><integer>3600</integer>
 <key>RunAtLoad</key><true/>
 <key>Nice</key><integer>19</integer>
 <key>StandardErrorPath</key><string>{os.path.join(HERE, 'logs', 'geomag_watch.err')}</string>
</dict></plist>"""
    with open(PLIST, "w") as f:
        f.write(plist)
    os.system(f"launchctl unload {PLIST} 2>/dev/null; launchctl load {PLIST}")
    print(f"installed + loaded: {PLIST} (hourly)")


if __name__ == "__main__":
    {"run": run, "install": install}[sys.argv[1]]()
