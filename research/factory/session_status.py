"""One-command session status — replaces N exploratory tool calls at session
start (token saver). Prints: git state, ledger tip, Ep-51 watch (latest local
USGS state incl. started-heuristic), cache counts, running factory processes.

Usage: .venv-research/bin/python session_status.py
"""
import glob, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "<local-path>/signalmap"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          cwd=HERE).stdout.strip()


print("== git:", sh("git branch --show-current"), "|",
      sh("git log --oneline -1"), "| dirty:",
      sh("git status --short | wc -l").strip())
print("== ledger tip:", sh(f"{ROOT}/.venv-research/bin/python receipt_ledger.py tip"))

wl = os.path.join(HERE, "logs", "ep51_watch.jsonl")
if os.path.exists(wl):
    lines = [json.loads(x) for x in open(wl) if x.strip()]
    last = lines[-1]
    started = [x for x in lines if x.get("ep51_started_heuristic")]
    cap = last.get("cap", {})
    print(f"== ep51 watch: {len(lines)} polls, last {last['utc']}, "
          f"alert={cap.get('alert_level')}/{cap.get('color_code')}, "
          f"started_heuristic={last.get('ep51_started_heuristic')}, "
          f"blackout_streak={last.get('blackout_streak', 0)}")
    print("   synopsis:", (cap.get("synopsis") or "?")[:160])
    if last.get("blackout_streak", 0) >= 6:
        print(f"   !! WATCHER BLIND for {last['blackout_streak']} polls — check network/launchd")
    if started:
        print(f"   !! STARTED-FLAG at {started[0]['utc']} — read snapshot in "
              f"data/volcano/ep51_watch/, then ep51_prereg.py apply + "
              f"ep51_prereg2.py apply with HVO UTC times")
else:
    print("== ep51 watch: NO LOG (launchd agent not running?)")

gl = os.path.join(HERE, "logs", "geomag_watch.jsonl")
if os.path.exists(gl):
    glines = [json.loads(x) for x in open(gl) if x.strip()]
    glast = glines[-1]
    all_onsets = {o for x in glines for o in (x.get("new_onsets") or [])}
    print(f"== geomag watch: {len(glines)} polls, last {glast['utc']}, "
          f"latest Kp={(glast.get('latest') or {}).get('kp')}, "
          f"blackout_streak={glast.get('blackout_streak', 0)}, "
          f"onsets_caught_total={len(all_onsets)}")
    if glast.get("new_onsets"):
        print(f"   !! NEW ONSET(S) this poll: {glast['new_onsets']} (Kp>={glast.get('onset_kp_threshold')})")
    if glast.get("blackout_streak", 0) >= 6:
        print(f"   !! WATCHER BLIND for {glast['blackout_streak']} polls — check network/launchd")
else:
    print("== geomag watch: NO LOG (launchd agent not running?)")

for d, pat in (("fresh_oot", "*.npz"), ("precursor", "*.npz"), ("ep51_watch", "*.txt")):
    p = os.path.join(ROOT, "data", "volcano", d)
    n = len(glob.glob(os.path.join(p, pat))) if os.path.isdir(p) else 0
    print(f"== cache {d}: {n}")

procs = sh("ps aux | grep 'venv-research/bin/python' | grep -v grep | "
           "awk '{for(i=11;i<=NF;i++) printf $i\" \"; print \"\"}'")
print("== running factory procs:", procs or "none")
for lg in sorted(glob.glob(os.path.join(HERE, "logs", "*fetch*.log"))):
    print(f"== tail {os.path.basename(lg)}:", sh(f"tail -1 '{lg}'")[:120])
