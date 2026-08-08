"""Factory scheduler -- the single RAM-bounded serial driver for the
zero-LLM-token research factory (regression / retro-sweep / family-queue).

DESIGN (why): multiple other local projects share this machine's RAM budget.
This driver guarantees AT MOST ONE heavy factory job runs at any time (flock
single-slot lock) and refuses to even start a heavy job unless free memory
clears a floor (default 3.0GB), so total factory RAM is bounded by the
largest single job, never the sum of concurrent ones. Cheap jobs (pytest,
readout_screen --flags) use a much lower floor since they barely touch RAM.

Priority per tick (checkpointed, resumable, at most ONE unit of work):
  (a) nightly regression, if not already run today            [cheap]
  (b) one retro-sweep step (cursor-checkpointed, resumable)    [heavy]
  (c) one family-queue step, IF prereg gate passes             [heavy, gated]
      -- currently REFUSES: no committed PREREG_FACTORY_FAMILY_QUEUE.md.

Every decision is appended as one JSONL line to logs/factory_scheduler.jsonl
(ts, job, decision, ram_free_gb, result). State (cursor/last-run-date) lives
in logs/factory_scheduler_state.json, so a tick can be killed at any point
and the next tick resumes correctly.

CLI:
  factory_scheduler.py --tick       do one unit of work (or defer/refuse/busy)
  factory_scheduler.py --status     print current state + live RAM reading
  factory_scheduler.py --dry-run    decide + log, never launch anything

Install (hourly, NOT auto-loaded by this script -- see launchd/README below):
  launchctl load ~/Library/LaunchAgents/com.signalmap.factory.plist
"""
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import receipt_ledger  # noqa: E402

LOGS = os.path.join(HERE, "logs")
LOCK_PATH = os.path.join(LOGS, "factory_scheduler.lock")
STATE_PATH = os.path.join(LOGS, "factory_scheduler_state.json")
JSONL_PATH = os.path.join(LOGS, "factory_scheduler.jsonl")
BASELINE_PATH = os.path.join(LOGS, "regression_baseline.json")
ALERT_PATH = os.path.join(LOGS, "regression_alert.txt")
RETRO_CANDIDATES_PATH = os.path.join(LOGS, "retro_candidates.jsonl")
PREREG_FAMILY_QUEUE_PATH = os.path.join(HERE, "PREREG_FACTORY_FAMILY_QUEUE.md")

PY = "<local-path>/signalmap/.venv-research/bin/python"

# Configurable RAM floors (env override). Heavy: don't start a ~1-2GB job
# unless >=3GB is free, so job+headroom stays inside the ~4-5GB shared
# budget. Cheap: pytest/readout_screen barely touch RAM, low floor is enough.
RAM_FLOOR_HEAVY_GB = float(os.environ.get("RAM_FLOOR_GB", 3.0))
RAM_FLOOR_CHEAP_GB = float(os.environ.get("RAM_FLOOR_CHEAP_GB", 1.0))

# Configurable wall-clock timeouts (env override). Every heavy-job subprocess
# launch (family-queue distill) and every cheap-job subprocess launch
# (regression's pytest/screen runs) goes through _run_subprocess_with_timeout
# below -- NOTHING launched by this scheduler can run past its bound. This is
# the direct fix for the observed failure mode: a hand-launched heavy script
# with no timeout of its own ran ~19h unbounded (see LOOP_SAFETY.md).
HEAVY_JOB_TIMEOUT_SEC = float(os.environ.get("HEAVY_JOB_TIMEOUT_SEC", 5400))   # 90min
CHEAP_JOB_TIMEOUT_SEC = float(os.environ.get("CHEAP_JOB_TIMEOUT_SEC", 1800))   # 30min
JOB_KILL_GRACE_SEC = float(os.environ.get("JOB_KILL_GRACE_SEC", 10))
MAX_JOB_FAILURES = int(os.environ.get("MAX_JOB_FAILURES", 3))

RETRO_DEFAULT_BANK = "UWE-volcano-precursor"  # one small archived bank, wired below
RETRO_STEP = 4  # windows scanned per tick

# (family, bank) pairs a finished, committed prereg would authorize. Inert
# until PREREG_FACTORY_FAMILY_QUEUE.md is finalized -- see _prereg_committed.
FAMILY_QUEUE = [("envelope", "hydcooler"), ("coherence", "hydcooler")]

TODO_MARKER = "<!-- TODO:"
DRAFT_MARKER = "STATUS: DRAFT"

_PASS_RE = re.compile(r"(\d+) passed")
_FAIL_RE = re.compile(r"(\d+) failed")
_VMSTAT_FREE_RE = re.compile(r"Pages free:\s+(\d+)\.")
_VMSTAT_INACTIVE_RE = re.compile(r"Pages inactive:\s+(\d+)\.")
_VMSTAT_PAGESIZE_RE = re.compile(r"page size of (\d+) bytes")


# --------------------------------------------------------------------- RAM
def _vm_stat_text():
    return subprocess.run(["vm_stat"], capture_output=True, text=True,
                          timeout=10).stdout


def free_mem_gb(text=None):
    """Free memory estimate = page_size * (Pages free + Pages inactive).
    Fail-safe: any parse failure returns 0.0 (defers every heavy job)."""
    text = text if text is not None else _vm_stat_text()
    m_free = _VMSTAT_FREE_RE.search(text)
    m_inact = _VMSTAT_INACTIVE_RE.search(text)
    if not (m_free and m_inact):
        return 0.0
    m_page = _VMSTAT_PAGESIZE_RE.search(text)
    page_size = int(m_page.group(1)) if m_page else 4096
    free_pages = int(m_free.group(1)) + int(m_inact.group(1))
    return free_pages * page_size / 1e9


# ------------------------------------------------------------------- lock
def _acquire_lock():
    os.makedirs(LOGS, exist_ok=True)
    fd = open(LOCK_PATH, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fd.close()
        return None
    return fd


# ------------------------------------------------------------------ state
def _default_state():
    return {"last_regression_date": None,
            "retro_cursor": {"bank": None, "offset": 0},
            "retro_done": False,
            "family_queue_idx": 0}


def _load_state():
    if not os.path.exists(STATE_PATH):
        return _default_state()
    with open(STATE_PATH) as f:
        state = json.load(f)
    for k, v in _default_state().items():
        state.setdefault(k, v)
    return state


def _save_state(state):
    os.makedirs(LOGS, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=1)


def log_decision(job, decision, ram_free_gb=None, result=None):
    os.makedirs(LOGS, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "job": job, "decision": decision,
             "ram_free_gb": round(ram_free_gb, 3) if ram_free_gb is not None else None,
             "result": result}
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------- timeout-bounded launch
def _run_subprocess_with_timeout(cmd, cwd, timeout_sec, grace_sec=None, job_name="job",
                                 capture=False):
    """The ONE place any heavy or cheap subprocess gets launched from. Hard
    wall-clock bound: SIGTERM at timeout_sec, grace period, then SIGKILL. On
    a kill, appends a `killed_timeout` JSONL line (pid + elapsed) so any
    runaway is provably bounded and provably logged -- never silent, never
    unbounded.

    Returns dict: returncode (None if killed before exit), timed_out (bool),
    elapsed (float sec), pid (int), stdout (str, only if capture=True).
    """
    grace_sec = JOB_KILL_GRACE_SEC if grace_sec is None else grace_sec
    start = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=cwd,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None,
                            text=True if capture else False)
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
        return {"returncode": proc.returncode, "timed_out": False,
               "elapsed": time.monotonic() - start, "pid": proc.pid, "stdout": out or ""}
    except subprocess.TimeoutExpired:
        proc.terminate()  # SIGTERM -- give it a chance to exit cleanly first
        try:
            out, _ = proc.communicate(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            proc.kill()  # SIGKILL -- did not honor SIGTERM within the grace window
            out, _ = proc.communicate()
        elapsed = time.monotonic() - start
        log_decision(job_name, "killed_timeout", None,
                    {"pid": proc.pid, "elapsed_sec": round(elapsed, 1),
                     "timeout_sec": timeout_sec, "grace_sec": grace_sec})
        return {"returncode": proc.returncode, "timed_out": True,
               "elapsed": elapsed, "pid": proc.pid, "stdout": out or ""}


# --------------------------------------------------- retry / quarantine gate
def _job_failures(state):
    return state.setdefault("job_failures", {})


def _quarantined(state):
    return state.setdefault("quarantined_jobs", {})


def is_quarantined(state, key):
    return key in state.get("quarantined_jobs", {})


def quarantine_job(state, key, reason):
    """Mark `key` quarantined: STOP scheduling it (caller must check
    is_quarantined before running) until a human calls clear_quarantine."""
    _quarantined(state)[key] = {"reason": reason,
                                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    log_decision(key, "quarantined", None, reason)


def record_job_failure(state, key, reason, max_failures=None):
    """Bump key's consecutive-failure count; quarantine at the threshold.
    Returns True iff this call caused quarantine."""
    max_failures = MAX_JOB_FAILURES if max_failures is None else max_failures
    failures = _job_failures(state)
    failures[key] = failures.get(key, 0) + 1
    if failures[key] >= max_failures:
        quarantine_job(state, key, f"{reason} ({failures[key]} consecutive failures)")
        return True
    return False


def record_job_success(state, key):
    _job_failures(state)[key] = 0


def clear_quarantine(state, key):
    """The human-clears-it escape hatch: drop quarantine + failure count for
    key so the next tick will retry it fresh."""
    _quarantined(state).pop(key, None)
    _job_failures(state).pop(key, None)


# ------------------------------------------------------------ (a) regression
def _default_pytest_runner():
    res = _run_subprocess_with_timeout(["nice", "-n", "19", PY, "-m", "pytest", "-q"],
                                       cwd=REPO_ROOT, timeout_sec=CHEAP_JOB_TIMEOUT_SEC,
                                       job_name="regression_pytest", capture=True)
    out = res["stdout"]
    mp, mf = _PASS_RE.search(out), _FAIL_RE.search(out)
    passed, failed = (int(mp.group(1)) if mp else 0), (int(mf.group(1)) if mf else 0)
    if res["timed_out"]:
        failed = max(failed, 1)  # a timeout-kill must never read as a clean green run
    return (passed, failed, out)


def _default_screen_runner():
    res = _run_subprocess_with_timeout(["nice", "-n", "19", PY, "readout_screen.py", "--flags"],
                                       cwd=HERE, timeout_sec=CHEAP_JOB_TIMEOUT_SEC,
                                       job_name="regression_screen", capture=True)
    return res["stdout"]


def run_regression(pytest_runner=None, screen_runner=None):
    """pytest -q + readout_screen.py --flags vs a stored green baseline.
    First run seeds the baseline. A pass-count drop, a failure-count rise,
    or a drift in the readout_screen flag matrix (hashed) raises an ALERT."""
    pytest_runner = pytest_runner or _default_pytest_runner
    screen_runner = screen_runner or _default_screen_runner
    passed, failed, _out = pytest_runner()
    screen_text = screen_runner()
    screen_hash = hashlib.sha256(screen_text.encode()).hexdigest()[:16]

    baseline = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)

    if baseline is None:
        os.makedirs(LOGS, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump({"passed": passed, "failed": failed, "screen_hash": screen_hash,
                      "seeded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                     f, indent=1)
        return {"decision": "seeded_baseline", "result": f"{passed} passed, {failed} failed"}

    alerts = []
    if passed < baseline["passed"]:
        alerts.append(f"pytest passed dropped {baseline['passed']} -> {passed}")
    if failed > baseline.get("failed", 0):
        alerts.append(f"pytest failed rose {baseline.get('failed', 0)} -> {failed}")
    if screen_hash != baseline.get("screen_hash"):
        alerts.append(f"readout_screen --flags drifted ({baseline.get('screen_hash')} -> {screen_hash})")

    if alerts:
        msg = "; ".join(alerts)
        with open(ALERT_PATH, "w") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} ALERT: {msg}\n")
        return {"decision": "ALERT", "result": msg}

    # ratchet counts upward on green; screen_hash stays pinned so any future
    # drift (in either direction) is still caught.
    with open(BASELINE_PATH, "w") as f:
        json.dump({"passed": passed, "failed": failed, "screen_hash": screen_hash,
                  "seeded_utc": baseline.get("seeded_utc")}, f, indent=1)
    return {"decision": "ok", "result": f"{passed} passed, {failed} failed"}


# ---------------------------------------------------------- (b) retro-sweep
def ar1(x):
    """Lag-1 autocorrelation -- mirrored from ep52_retro.py (verbatim logic),
    kept local so importing this module never pulls in volcano_precursor's
    network-fetch import chain just to reuse two pure-math functions."""
    import numpy as np
    v = np.asarray(x, float) - np.mean(x)
    d = (v * v).sum()
    return float((v[1:] * v[:-1]).sum() / d) if d > 0 else 0.0


def perm_ent(x):
    """Permutation entropy (3-symbol) -- mirrored from ep52_retro.py verbatim."""
    import numpy as np
    a, b, c = x[:-2], x[1:-1], x[2:]
    pat = (a < b).astype(int) * 2 + (b < c).astype(int)
    h = np.bincount(pat, minlength=4) / max(len(pat), 1)
    h = h[h > 0]
    return float(-(h * np.log(h)).sum() / np.log(4)) if len(h) else 0.0


def _default_retro_loader(bank):
    """Real wiring, NOT exercised by this build/test session (tests inject
    their own tiny synthetic loader). Cache-only per volcano_precursor._fetch
    comment ("cache-only in practice after fetch mode") -- no network call
    once the ep52_retro/precursor_sampling cache is warm."""
    import numpy as np
    from volcano_precursor import _fetch
    X = _fetch("UWE", "2026-07-11T00:00")  # any cached reference timestamp
    return [np.asarray(w, float) for w in X] if X is not None else []


def _default_retro_scorer(chunk, bank, offset):
    """Candidate = window whose |ar1| or perm_ent clears a loose exploratory
    threshold. NEVER auto-promoted -- output feeds logs/retro_candidates.jsonl
    for a human to look at, same discipline as ep52_retro's ranked list."""
    cands = []
    for i, w in enumerate(chunk):
        a1, pe = ar1(w), perm_ent(w)
        if abs(a1) > 0.8 or pe < 0.3:
            cands.append({"bank": bank, "index": offset + i,
                         "ar1": a1, "perm_ent": pe,
                         "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return cands


def retro_sweep_step(state, loader=None, scorer=None):
    """One RAM-bounded chunk of a cursor-checkpointed scan. Resumable: the
    (bank, offset) cursor lives in the scheduler state file, so a killed
    process picks up exactly where it left off on the next tick."""
    loader = loader or _default_retro_loader
    scorer = scorer or _default_retro_scorer
    cursor = state.setdefault("retro_cursor", {"bank": None, "offset": 0})
    bank = cursor.get("bank") or RETRO_DEFAULT_BANK
    offset = cursor.get("offset", 0)

    arr = loader(bank)
    total = len(arr)
    if offset >= total:
        state["retro_done"] = True
        return {"decision": "bank_exhausted", "result": f"{bank} exhausted at {total}"}

    chunk = arr[offset: offset + RETRO_STEP]
    candidates = scorer(chunk, bank, offset)
    if candidates:
        os.makedirs(LOGS, exist_ok=True)
        with open(RETRO_CANDIDATES_PATH, "a") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")

    new_offset = offset + len(chunk)
    state["retro_cursor"] = {"bank": bank, "offset": new_offset}
    if new_offset >= total:
        state["retro_done"] = True
    return {"decision": "step",
           "result": f"{bank}:{offset}->{new_offset}/{total} ({len(candidates)} candidates)"}


# ------------------------------------------------------- (c) family-queue
def _prereg_committed():
    """Fail-closed integrity gate: refuses unless a FINALIZED (no TODO/DRAFT
    marker) PREREG_FACTORY_FAMILY_QUEUE.md exists AND is referenced by a
    logged ledger receipt (i.e. actually committed, not just sitting on
    disk). Currently always refuses -- see PREREG_FACTORY_FAMILY_QUEUE.md."""
    if not os.path.exists(PREREG_FAMILY_QUEUE_PATH):
        return False, "missing: PREREG_FACTORY_FAMILY_QUEUE.md"
    text = open(PREREG_FAMILY_QUEUE_PATH).read()
    if TODO_MARKER in text or DRAFT_MARKER in text:
        return False, "prereg still DRAFT (unresolved TODO block, not finalized)"
    ledger_path = receipt_ledger.LEDGER
    ref = False
    if os.path.exists(ledger_path):
        with open(ledger_path) as f:
            ref = any("PREREG_FACTORY_FAMILY_QUEUE.md" in ln for ln in f)
    if not ref:
        return False, "prereg not referenced by any committed ledger receipt"
    return True, "ok"


def _prereg_covers(family, bank):
    if not os.path.exists(PREREG_FAMILY_QUEUE_PATH):
        return False
    text = open(PREREG_FAMILY_QUEUE_PATH).read()
    return f"{family}:{bank}" in text


def _launch_distill(family, bank):
    """The one real heavy-job subprocess launch in this file -- hard-bounded
    at HEAVY_JOB_TIMEOUT_SEC (SIGTERM->grace->SIGKILL, see
    _run_subprocess_with_timeout). Returns the timeout-wrapper's result dict
    (not a bare int) so family_queue_step can see timed_out."""
    cmd = ["nice", "-n", "19", PY, "distill_premium_case.py", f"--family={family}", bank]
    return _run_subprocess_with_timeout(cmd, cwd=HERE, timeout_sec=HEAVY_JOB_TIMEOUT_SEC,
                                        job_name="family_queue")


def family_queue_step(state, queue=None, launch=None):
    """Retry/quarantine gate: a (family, bank) unit that fails
    MAX_JOB_FAILURES times in a row, or times out even once, is quarantined
    -- the queue STOPS retrying that exact unit (idx does not advance past
    it) until a human calls clear_quarantine(state, key). This deliberately
    also pauses the rest of the queue behind it: a unit that keeps failing
    or timing out usually signals an environment problem that would break
    later units too, so silently skipping ahead is not safe-by-default."""
    ok, why = _prereg_committed()
    if not ok:
        return {"decision": "refused_no_prereg", "result": why}
    queue = queue if queue is not None else FAMILY_QUEUE
    idx = state.get("family_queue_idx", 0)
    if idx >= len(queue):
        return {"decision": "queue_exhausted", "result": None}
    family, bank = queue[idx]
    key = f"family_queue:{family}:{bank}"

    if is_quarantined(state, key):
        return {"decision": "skipped_quarantined", "result": _quarantined(state)[key]["reason"]}

    if not _prereg_covers(family, bank):
        return {"decision": "refused_not_covered", "result": f"{family}/{bank} not declared in prereg"}

    launch = launch or _launch_distill
    launch_result = launch(family, bank)
    if isinstance(launch_result, dict):
        rc = launch_result.get("returncode")
        timed_out = launch_result.get("timed_out", False)
        elapsed = launch_result.get("elapsed", 0.0)
    else:  # back-compat: a plain int returncode from a test-injected launch
        rc, timed_out, elapsed = launch_result, False, 0.0

    receipt_ledger.log_receipt(f"FACTORY-FAMILY-QUEUE-{bank.upper()}-{family.upper()}",
                               {"family": family, "bank": bank, "returncode": rc,
                                "timed_out": timed_out,
                                "prereg": "PREREG_FACTORY_FAMILY_QUEUE.md"})

    if timed_out:
        # Doomed unit: quarantine immediately (a single timeout-kill is
        # sufficient, matching spec point 2) instead of retrying the same
        # unbounded job again next tick.
        quarantine_job(state, key, f"timeout-killed after {elapsed:.1f}s")
        return {"decision": "quarantined", "result": f"{family}/{bank} timed out after {elapsed:.1f}s"}

    if rc != 0:
        quarantined_now = record_job_failure(state, key, f"launch rc={rc}")
        decision = "quarantined" if quarantined_now else "failed"
        return {"decision": decision, "result": f"{family}/{bank} rc={rc}"}

    record_job_success(state, key)
    state["family_queue_idx"] = idx + 1
    return {"decision": "ran", "result": f"{family}/{bank} rc={rc}"}


# ------------------------------------------------------------------- tick
def tick(dry_run=False):
    fd = _acquire_lock()
    if fd is None:
        log_decision("scheduler", "busy_skip", None, "lock held by another run")
        print("busy, skip")
        return 0
    try:
        state = _load_state()
        free_gb = free_mem_gb()
        today = _today()

        # (a) nightly regression
        if state.get("last_regression_date") != today:
            if free_gb < RAM_FLOOR_CHEAP_GB:
                log_decision("regression", "deferred_low_mem", free_gb)
                print(f"deferred: low mem ({free_gb:.2f}GB < {RAM_FLOOR_CHEAP_GB}GB floor)")
                return 0
            if dry_run:
                log_decision("regression", "would_run(dry)", free_gb)
                print(f"dry-run: would run regression (free {free_gb:.2f}GB)")
                return 0
            res = run_regression()
            state["last_regression_date"] = today
            _save_state(state)
            log_decision("regression", res["decision"], free_gb, res.get("result"))
            print(f"regression: {res['decision']} -- {res.get('result')}")
            return 0

        # (b) retro-sweep
        if not state.get("retro_done"):
            if free_gb < RAM_FLOOR_HEAVY_GB:
                log_decision("retro_sweep", "deferred_low_mem", free_gb)
                print(f"deferred: low mem ({free_gb:.2f}GB < {RAM_FLOOR_HEAVY_GB}GB floor)")
                return 0
            if dry_run:
                log_decision("retro_sweep", "would_run(dry)", free_gb)
                print(f"dry-run: would run retro-sweep step (free {free_gb:.2f}GB)")
                return 0
            res = retro_sweep_step(state)
            _save_state(state)
            log_decision("retro_sweep", res["decision"], free_gb, res.get("result"))
            print(f"retro_sweep: {res['decision']} -- {res.get('result')}")
            return 0

        # (c) family-queue -- integrity-gated first, RAM checked only if it would run
        ok, why = _prereg_committed()
        if not ok:
            log_decision("family_queue", "refused_no_prereg", free_gb, why)
            print(f"refused: {why}")
            return 0
        if free_gb < RAM_FLOOR_HEAVY_GB:
            log_decision("family_queue", "deferred_low_mem", free_gb)
            print(f"deferred: low mem ({free_gb:.2f}GB < {RAM_FLOOR_HEAVY_GB}GB floor)")
            return 0
        if dry_run:
            log_decision("family_queue", "would_run(dry)", free_gb)
            print(f"dry-run: would run family-queue step (free {free_gb:.2f}GB)")
            return 0
        res = family_queue_step(state)
        _save_state(state)
        log_decision("family_queue", res["decision"], free_gb, res.get("result"))
        print(f"family_queue: {res['decision']} -- {res.get('result')}")
        return 0
    finally:
        fd.close()


def status():
    state = _load_state()
    free_gb = free_mem_gb()
    prereg_ok, prereg_why = _prereg_committed()
    print(json.dumps({
        "state": state,
        "ram_free_gb": round(free_gb, 3),
        "ram_floor_heavy_gb": RAM_FLOOR_HEAVY_GB,
        "ram_floor_cheap_gb": RAM_FLOOR_CHEAP_GB,
        "heavy_job_timeout_sec": HEAVY_JOB_TIMEOUT_SEC,
        "cheap_job_timeout_sec": CHEAP_JOB_TIMEOUT_SEC,
        "family_queue_gate": {"ok": prereg_ok, "why": prereg_why},
        "quarantined_jobs": state.get("quarantined_jobs", {}),
    }, indent=1))
    return 0


def _clear_quarantine_cli(arg):
    key = arg.split("=", 1)[1] if "=" in arg else None
    if not key:
        print("usage: factory_scheduler.py --clear-quarantine=<job_key>")
        return 1
    state = _load_state()
    if not is_quarantined(state, key):
        print(f"'{key}' is not quarantined (nothing to clear)")
        return 0
    clear_quarantine(state, key)
    _save_state(state)
    log_decision(key, "quarantine_cleared", None, "cleared by human via --clear-quarantine")
    print(f"cleared quarantine for '{key}'")
    return 0


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if "--status" in argv:
        return status()
    if "--dry-run" in argv:
        return tick(dry_run=True)
    if "--tick" in argv:
        return tick(dry_run=False)
    clear_arg = next((a for a in argv if a.startswith("--clear-quarantine")), None)
    if clear_arg is not None:
        return _clear_quarantine_cli(clear_arg)
    print("usage: factory_scheduler.py [--tick|--status|--dry-run|--clear-quarantine=<job_key>]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
