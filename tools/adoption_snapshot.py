#!/usr/bin/env python3
"""Snapshot public adoption signals for signalmap into a durable local file.

    python tools/adoption_snapshot.py            # fetch, merge, print report
    python tools/adoption_snapshot.py --report   # print the stored series only

Why a script: GitHub's traffic API keeps only the last 14 days, so any day not
snapshotted is gone for good. Runs are merged by date into
``research/adoption/adoption.json``; re-running on the same day overwrites that
day's numbers rather than double-counting.

Signals, and what each is worth:
  * PyPI downloads (pypistats, mirrors excluded) — still counts CI and bots.
  * Clones — a GitHub Actions checkout is a clone, so compare against the
    workflow-run count of the same day before reading anything into it.
  * Views / unique visitors and referrers — the closest thing to a human.
  * Stars / forks / watchers / open issues — deliberate acts, hardest to fake.

Nothing is subtracted automatically: the report prints CI runs next to clones
and leaves the attribution to the reader.
"""
import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = "signalmap"
REPO = "MrPredic/signalmap"
STORE = Path(__file__).resolve().parent.parent / "research" / "adoption" / "adoption.json"
UA = {"User-Agent": f"{PACKAGE}-adoption-snapshot"}


def _get_json(url):
    """GET a JSON document, or return None with a note on failure."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as exc:
        print(f"  ! {url} -> {exc}", file=sys.stderr)
        return None


def _gh_json(path):
    """Call the GitHub API through the gh CLI (uses the operator's own token)."""
    try:
        out = subprocess.run(
            ["gh", "api", path], capture_output=True, text=True, timeout=60, check=True
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        detail = getattr(exc, "stderr", "") or exc
        print(f"  ! gh api {path} -> {str(detail).strip()[:200]}", file=sys.stderr)
        return None
    try:
        return json.loads(out.stdout)
    except ValueError:
        return None


def merge_series(existing, incoming):
    """Merge day-keyed dicts, newer values winning. Pure; the unit under test."""
    merged = dict(existing or {})
    merged.update(incoming or {})
    return {day: merged[day] for day in sorted(merged)}


def pypi_days(payload):
    """{date: downloads} from a pypistats overall payload, mirrors excluded."""
    rows = (payload or {}).get("data") or []
    return {
        r["date"]: r["downloads"]
        for r in rows
        if isinstance(r, dict) and r.get("category") == "without_mirrors"
    }


def traffic_days(payload, key):
    """{date: {"count": n, "uniques": n}} from a clones/views traffic payload."""
    rows = (payload or {}).get(key) or []
    return {
        r["timestamp"][:10]: {"count": r["count"], "uniques": r["uniques"]}
        for r in rows
        if isinstance(r, dict) and r.get("timestamp")
    }


def runs_days(payload):
    """{date: workflow runs started} — the CI share of the clone count."""
    runs = (payload or {}).get("workflow_runs") or []
    return dict(Counter(r["created_at"][:10] for r in runs if r.get("created_at")))


def fetch():
    """Collect one snapshot from PyPI and GitHub. Missing sources stay empty."""
    print("fetching ...", file=sys.stderr)
    return {
        "pypi_downloads": pypi_days(
            _get_json(f"https://pypistats.org/api/packages/{PACKAGE}/overall?mirrors=false")
        ),
        "clones": traffic_days(_gh_json(f"repos/{REPO}/traffic/clones"), "clones"),
        "views": traffic_days(_gh_json(f"repos/{REPO}/traffic/views"), "views"),
        "ci_runs": runs_days(_gh_json(f"repos/{REPO}/actions/runs?per_page=100")),
        "referrers": _gh_json(f"repos/{REPO}/traffic/popular/referrers") or [],
        "paths": _gh_json(f"repos/{REPO}/traffic/popular/paths") or [],
        "repo": _gh_json(f"repos/{REPO}") or {},
    }


def update(store, snap, now):
    """Fold a snapshot into the store, keeping every day ever observed."""
    out = dict(store)
    for key in ("pypi_downloads", "clones", "views", "ci_runs"):
        out[key] = merge_series(store.get(key, {}), snap.get(key, {}))
    repo = snap.get("repo") or {}
    marks = out.setdefault("marks", {})
    if repo:
        marks[now[:10]] = {
            "stars": repo.get("stargazers_count"),
            "forks": repo.get("forks_count"),
            "watchers": repo.get("subscribers_count"),
            "open_issues": repo.get("open_issues_count"),
        }
    for key in ("referrers", "paths"):
        if snap.get(key):
            out.setdefault(key, {})[now[:10]] = snap[key]
    out["last_snapshot"] = now
    return out


def report(store, days=21):
    """Render the stored series as a fixed-width table plus deliberate acts."""
    seen = set()
    for key in ("pypi_downloads", "clones", "views", "ci_runs"):
        seen |= set(store.get(key, {}))
    window = sorted(seen)[-days:]
    lines = [
        "date        pypi  clones/uniq  views/uniq  ci_runs",
        "-" * 52,
    ]
    for day in window:
        clones = store.get("clones", {}).get(day) or {}
        views = store.get("views", {}).get(day) or {}
        lines.append(
            f"{day}  {store.get('pypi_downloads', {}).get(day, 0):>4}"
            f"  {clones.get('count', 0):>6}/{clones.get('uniques', 0):<4}"
            f"  {views.get('count', 0):>5}/{views.get('uniques', 0):<3}"
            f"  {store.get('ci_runs', {}).get(day, 0):>7}"
        )
    marks = store.get("marks", {})
    if marks:
        latest = marks[sorted(marks)[-1]]
        lines += [
            "-" * 52,
            "deliberate acts: "
            + "  ".join(f"{k}={v}" for k, v in latest.items()),
        ]
    refs = store.get("referrers", {})
    if refs:
        latest = refs[sorted(refs)[-1]]
        lines.append(
            "referrers: "
            + (", ".join(f"{r['referrer']}({r['uniques']}u)" for r in latest) or "none")
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="print stored data, fetch nothing")
    ap.add_argument("--store", type=Path, default=STORE, help=f"data file (default: {STORE})")
    args = ap.parse_args(argv)

    store = json.loads(args.store.read_text()) if args.store.exists() else {}
    if not args.report:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        store = update(store, fetch(), now)
        args.store.parent.mkdir(parents=True, exist_ok=True)
        args.store.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")
        print(f"stored -> {args.store}", file=sys.stderr)
    print(report(store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
