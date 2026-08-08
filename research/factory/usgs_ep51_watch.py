"""Zero-token USGS/HVO Kilauea watcher (launchd, hourly).

Two sources per run, appended as one JSONL line to logs/ep51_watch.jsonl:
  1. HANS CAP feed (structured): alert_level/color_code/synopsis/pubDate —
     episode starts show up as WATCH/ORANGE-RED + "Episode 51 ... began".
  2. volcano-updates HTML page via curl (urllib gets 403): full HVO wording
     (episode start/end times!), snapshotted to data/volcano/ep51_watch/
     whenever the text changes.
Heuristic flag marks runs whose text suggests Episode 51 started; the
authoritative call stays with the human/next session reading the snapshot.
Result: session start = read one local log instead of web fetches.

Install (once):  .venv-research/bin/python usgs_ep51_watch.py install
Run once (test): .venv-research/bin/python usgs_ep51_watch.py run
Status:          tail logs/ep51_watch.jsonl
"""
import hashlib, html, json, os, re, subprocess, sys, time, urllib.request
from datetime import datetime, timezone

RETRIES = 2
BACKOFF_S = 5


def _retry(fn):
    last_err = None
    for i in range(RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < RETRIES:
                time.sleep(BACKOFF_S)
    raise last_err


HERE = os.path.dirname(os.path.abspath(__file__))
CAP = "https://volcanoes.usgs.gov/hans-public/api/volcano/getCapElevated"
PAGE = "https://www.usgs.gov/volcanoes/kilauea/volcano-updates"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
TARGET_EPISODE = os.environ.get("SIGNALMAP_WATCH_EPISODE", "51")
LOG = os.environ.get("SIGNALMAP_WATCH_LOG",
                     os.path.join(HERE, "logs", f"ep{TARGET_EPISODE}_watch.jsonl"))
SNAP = os.environ.get("SIGNALMAP_WATCH_SNAP",
                      f"<local-path>/signalmap/data/volcano/ep{TARGET_EPISODE}_watch")
PLIST = os.path.expanduser("~/Library/LaunchAgents/com.signalmap.ep51watch.plist")
PY = "<local-path>/signalmap/.venv-research/bin/python"
STARTED_RE = re.compile(
    rf"episode {TARGET_EPISODE}[^.]*?(began|started|commenced|ended|is (currently )?(occurring|erupting))",
    re.I)
FORECAST_RE = re.compile(r"likely|forecast|expect|anticipat|\bmay\b|could|between",
                         re.I)


def _started(blob):
    """Past-tense Ep-51 wording outside forecast context, per sentence-ish span."""
    return any(not FORECAST_RE.search(m.group(0)) for m in STARTED_RE.finditer(blob))


def _cap_once():
    d = json.load(urllib.request.urlopen(CAP, timeout=60))
    k = next(x for x in d if "ilauea" in x.get("volcano_name_appended", ""))
    return {f: k.get(f) for f in ("alert_level", "color_code", "synopsis",
                                  "pubDate", "notice_identifier", "notice_url")}


def _cap():
    return _retry(_cap_once)


def _page_text_once():
    raw = subprocess.run(
        ["curl", "-s", "--compressed", "-A", UA, "-H",
         "Accept: text/html,application/xhtml+xml", PAGE],
        capture_output=True, text=True, timeout=120).stdout
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\s+", " ", text)
    assert len(text) > 2000, f"page too short ({len(text)} chars)"
    return text


def _page_text():
    return _retry(_page_text_once)


def _blackout_streak(log_path):
    if not os.path.exists(log_path):
        return 0
    with open(log_path) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    n = 0
    for e in reversed(lines):
        if e.get("cap_error") and e.get("page_error"):
            n += 1
        else:
            break
    return n


def run():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    os.makedirs(SNAP, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {"utc": now}
    try:
        entry["cap"] = _cap()
    except Exception as e:
        entry["cap_error"] = str(e)
    try:
        text = _page_text()
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        prev = None
        if os.path.exists(LOG):
            with open(LOG) as f:
                lines = [json.loads(x) for x in f if x.strip()]
            prev = next((x.get("page_sha") for x in reversed(lines)
                         if "page_sha" in x), None)
        if sha != prev:
            with open(os.path.join(SNAP, f"{now.replace(':', '')}_{sha}.txt"), "w") as f:
                f.write(text)
        m = re.search(rf".{{200}}episode {TARGET_EPISODE}.{{400}}", text, re.I)
        entry.update({"page_sha": sha, "page_changed": sha != prev,
                      "excerpt": m.group(0) if m else None})
    except Exception as e:
        entry["page_error"] = str(e)
        text = ""
    blob = (text + " " + json.dumps(entry.get("cap", {}))).lower()
    started_key = f"ep{TARGET_EPISODE}_started_heuristic"
    entry[started_key] = _started(blob) or \
        entry.get("cap", {}).get("alert_level") in ("WATCH", "WARNING")
    entry["blackout_streak"] = _blackout_streak(LOG) + 1 \
        if (entry.get("cap_error") and entry.get("page_error")) else 0
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps({k: entry.get(k) for k in
                      ("utc", "page_changed", started_key, "blackout_streak")} |
                     {"alert": entry.get("cap", {}).get("alert_level")}))
    if entry["blackout_streak"] >= 6:
        print(f"WARN: ep51_watch blackout streak = {entry['blackout_streak']} polls", file=sys.stderr)


def install():
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
 <key>Label</key><string>com.signalmap.ep51watch</string>
 <key>ProgramArguments</key><array>
  <string>{PY}</string>
  <string>{os.path.join(HERE, 'usgs_ep51_watch.py')}</string>
  <string>run</string>
 </array>
 <key>StartInterval</key><integer>3600</integer>
 <key>RunAtLoad</key><true/>
 <key>Nice</key><integer>19</integer>
 <key>StandardErrorPath</key><string>{os.path.join(HERE, 'logs', 'ep51_watch.err')}</string>
</dict></plist>"""
    with open(PLIST, "w") as f:
        f.write(plist)
    os.system(f"launchctl unload {PLIST} 2>/dev/null; launchctl load {PLIST}")
    print(f"installed + loaded: {PLIST} (hourly)")


if __name__ == "__main__":
    {"run": run, "install": install}[sys.argv[1]]()
