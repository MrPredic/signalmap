"""Current HVO watcher for Kīlauea episode 52.

Run once with:
  ../../.venv-research/bin/python3 usgs_ep52_watch.py run
"""
import os
import sys

os.environ.setdefault("SIGNALMAP_WATCH_EPISODE", "52")
os.environ.setdefault("SIGNALMAP_WATCH_LOG",
                       os.path.join(os.path.dirname(__file__), "logs", "ep52_watch.jsonl"))
os.environ.setdefault("SIGNALMAP_WATCH_SNAP",
                       "<local-path>/signalmap/data/volcano/ep52_watch")

from usgs_ep51_watch import run  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "run":
        raise SystemExit("usage: usgs_ep52_watch.py run")
    run()
