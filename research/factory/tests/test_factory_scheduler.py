"""Targeted tests for factory_scheduler.py -- RAM-bounded single-slot loop
driver. Run: .venv-research/bin/python -m pytest tests/test_factory_scheduler.py -q
(run from research/factory/). Never launches a real distill/family-queue job
-- lock/RAM tests use trivial fake work (no subprocess, no real bank load).
"""
import fcntl
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factory_scheduler as fs  # noqa: E402
import receipt_ledger  # noqa: E402


VMSTAT_LOW = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                      100.
Pages active:                                  300000.
Pages inactive:                                  100.
Pages speculative:                                 10.
Pages wired down:                              100000.
"""

VMSTAT_HIGH = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   200000.
Pages active:                                 300000.
Pages inactive:                               200000.
Pages speculative:                                10.
Pages wired down:                             100000.
"""

# ~1.5GB free (above cheap 1.0GB floor, below heavy 3.0GB floor)
VMSTAT_CHEAP_ONLY = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    50000.
Pages active:                                 300000.
Pages inactive:                                41553.
Pages speculative:                                10.
Pages wired down:                             100000.
"""


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    """Redirect every scheduler path constant into a throwaway tmp dir so
    tests never touch the real logs/ or the real ledger."""
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(fs, "LOGS", str(logs))
    monkeypatch.setattr(fs, "LOCK_PATH", str(logs / "factory_scheduler.lock"))
    monkeypatch.setattr(fs, "STATE_PATH", str(logs / "factory_scheduler_state.json"))
    monkeypatch.setattr(fs, "JSONL_PATH", str(logs / "factory_scheduler.jsonl"))
    monkeypatch.setattr(fs, "BASELINE_PATH", str(logs / "regression_baseline.json"))
    monkeypatch.setattr(fs, "ALERT_PATH", str(logs / "regression_alert.txt"))
    monkeypatch.setattr(fs, "RETRO_CANDIDATES_PATH", str(logs / "retro_candidates.jsonl"))
    monkeypatch.setattr(fs, "PREREG_FAMILY_QUEUE_PATH", str(tmp_path / "PREREG_FACTORY_FAMILY_QUEUE.md"))
    # family_queue_step() writes a ledger receipt on every real attempt --
    # redirect the ledger too so tests never touch the real LEDGER.jsonl.
    monkeypatch.setattr(receipt_ledger, "LEDGER", str(tmp_path / "LEDGER.jsonl"))
    yield tmp_path


def _committed_prereg(family="envelope", bank="hydcooler"):
    """Write a finalized (no DRAFT/TODO) prereg covering (family, bank) and
    reference it from the (already-redirected) ledger, so
    _prereg_committed() returns True -- mirrors a real finalized+committed
    prereg without touching any real file."""
    with open(fs.PREREG_FAMILY_QUEUE_PATH, "w") as f:
        f.write(f"STATUS: FINAL\ncovers: {family}:{bank}\n")
    receipt_ledger.log_receipt("PREREG-COMMIT", {"note": "PREREG_FACTORY_FAMILY_QUEUE.md finalized"})


def _jsonl_lines():
    if not os.path.exists(fs.JSONL_PATH):
        return []
    with open(fs.JSONL_PATH) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ---------------------------------------------------------------- (a) lock
def test_lock_prevents_concurrent_tick(monkeypatch):
    """Holding the lock file open (as another process would) makes a second
    --tick log 'busy' and exit 0 without touching state or running anything."""
    monkeypatch.setattr(fs, "free_mem_gb", lambda: 10.0)
    os.makedirs(fs.LOGS, exist_ok=True)
    held = open(fs.LOCK_PATH, "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        rc = fs.main(["--tick"])
    finally:
        held.close()
    assert rc == 0
    lines = _jsonl_lines()
    assert len(lines) == 1
    assert lines[0]["job"] == "scheduler"
    assert lines[0]["decision"] == "busy_skip"
    assert not os.path.exists(fs.STATE_PATH)  # never got past the lock check


# ------------------------------------------------------------- (b) RAM guard
def test_ram_guard_defers_heavy_job_when_low_mem(monkeypatch):
    """Regression already ran today -> next in line is retro-sweep (heavy).
    Free mem below the 3.0GB heavy floor -> deferred, logged, nothing run."""
    monkeypatch.setattr(fs, "_vm_stat_text", lambda: VMSTAT_LOW)
    state = fs._default_state()
    state["last_regression_date"] = fs._today()
    fs._save_state(state)

    def _boom(*a, **k):
        raise AssertionError("heavy job must not run under low mem")
    monkeypatch.setattr(fs, "retro_sweep_step", _boom)

    rc = fs.main(["--tick"])
    assert rc == 0
    lines = _jsonl_lines()
    assert lines[-1]["job"] == "retro_sweep"
    assert lines[-1]["decision"] == "deferred_low_mem"
    assert lines[-1]["ram_free_gb"] < fs.RAM_FLOOR_HEAVY_GB


# ------------------------------------------------------ (c) cheap under low mem
def test_cheap_job_runs_under_low_mem(monkeypatch):
    """~1.5GB free clears the cheap 1.0GB floor but not the heavy 3.0GB
    floor -> the nightly regression (priority a, cheap) still runs."""
    monkeypatch.setattr(fs, "_vm_stat_text", lambda: VMSTAT_CHEAP_ONLY)
    free_gb = fs.free_mem_gb()
    assert fs.RAM_FLOOR_CHEAP_GB <= free_gb < fs.RAM_FLOOR_HEAVY_GB

    calls = {"n": 0}

    def fake_pytest_runner():
        calls["n"] += 1
        return (5, 0, "5 passed")

    def fake_screen_runner():
        return "SOME-BANK: no readout flag"

    monkeypatch.setattr(fs, "_default_pytest_runner", fake_pytest_runner)
    monkeypatch.setattr(fs, "_default_screen_runner", fake_screen_runner)

    rc = fs.main(["--tick"])
    assert rc == 0
    assert calls["n"] == 1  # regression actually ran, not deferred
    lines = _jsonl_lines()
    assert lines[-1]["job"] == "regression"
    assert lines[-1]["decision"] == "seeded_baseline"
    state = fs._load_state()
    assert state["last_regression_date"] == fs._today()


# --------------------------------------------------------- (d) family-queue gate
def test_family_queue_refuses_with_no_prereg():
    """Fail-closed: no PREREG_FACTORY_FAMILY_QUEUE.md at all -> refuses."""
    state = fs._default_state()
    res = fs.family_queue_step(state)
    assert res["decision"] == "refused_no_prereg"
    assert "missing" in res["result"]


def test_family_queue_refuses_with_draft_prereg(tmp_path):
    """Even if the template file exists (as shipped), the DRAFT/TODO markers
    keep the gate closed -- exactly today's real repo state."""
    with open(fs.PREREG_FAMILY_QUEUE_PATH, "w") as f:
        f.write("STATUS: DRAFT\n<!-- TODO: fill in -->\n")
    state = fs._default_state()
    res = fs.family_queue_step(state)
    assert res["decision"] == "refused_no_prereg"
    assert "DRAFT" in res["result"]


def test_family_queue_gate_ignores_ram_and_dry_run(monkeypatch):
    """Full tick(): plenty of RAM, regression+retro already done, gate still
    refuses because there is no finalized+ledger-committed prereg."""
    monkeypatch.setattr(fs, "_vm_stat_text", lambda: VMSTAT_HIGH)
    state = fs._default_state()
    state["last_regression_date"] = fs._today()
    state["retro_done"] = True
    fs._save_state(state)

    def _boom(*a, **k):
        raise AssertionError("family-queue must not launch without a prereg")
    monkeypatch.setattr(fs, "family_queue_step", _boom)

    rc = fs.main(["--tick"])
    assert rc == 0
    lines = _jsonl_lines()
    assert lines[-1]["job"] == "family_queue"
    assert lines[-1]["decision"] == "refused_no_prereg"


# ------------------------------------------------------------- (e) retro cursor
def test_retro_cursor_advances_and_resumes():
    tiny = [float(i) for i in range(10)]  # 10 "windows", step=4 -> 3 steps

    def loader(bank):
        return tiny

    def scorer(chunk, bank, offset):
        return []  # no candidates, keep this test about the cursor only

    state = fs._default_state()
    res1 = fs.retro_sweep_step(state, loader=loader, scorer=scorer)
    assert res1["decision"] == "step"
    assert state["retro_cursor"] == {"bank": fs.RETRO_DEFAULT_BANK, "offset": 4}
    assert state["retro_done"] is False
    fs._save_state(state)

    # simulate a fresh process: reload state from disk, resume
    resumed = fs._load_state()
    assert resumed["retro_cursor"]["offset"] == 4
    res2 = fs.retro_sweep_step(resumed, loader=loader, scorer=scorer)
    assert resumed["retro_cursor"]["offset"] == 8
    assert resumed["retro_done"] is False

    res3 = fs.retro_sweep_step(resumed, loader=loader, scorer=scorer)
    assert resumed["retro_cursor"]["offset"] == 10
    assert resumed["retro_done"] is True  # exhausted exactly at len(tiny)

    res4 = fs.retro_sweep_step(resumed, loader=loader, scorer=scorer)
    assert res4["decision"] == "bank_exhausted"


def test_retro_candidates_written_for_flagged_windows():
    def loader(bank):
        return [0.0, 1.0]

    def scorer(chunk, bank, offset):
        return [{"bank": bank, "index": offset, "note": "fake-candidate"}]

    state = fs._default_state()
    fs.retro_sweep_step(state, loader=loader, scorer=scorer)
    assert os.path.exists(fs.RETRO_CANDIDATES_PATH)
    with open(fs.RETRO_CANDIDATES_PATH) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    assert rows == [{"bank": fs.RETRO_DEFAULT_BANK, "index": 0, "note": "fake-candidate"}]


# --------------------------------------------------------------- (f) regression
def test_regression_seeds_baseline_first_run():
    def pytest_runner():
        return (116, 0, "116 passed")

    def screen_runner():
        return "BANK: no readout flag"

    res = fs.run_regression(pytest_runner=pytest_runner, screen_runner=screen_runner)
    assert res["decision"] == "seeded_baseline"
    assert os.path.exists(fs.BASELINE_PATH)
    with open(fs.BASELINE_PATH) as f:
        baseline = json.load(f)
    assert baseline["passed"] == 116
    assert baseline["failed"] == 0
    assert not os.path.exists(fs.ALERT_PATH)


def test_regression_flags_a_simulated_drop():
    def pytest_runner_green():
        return (116, 0, "116 passed")

    def screen_runner():
        return "BANK: no readout flag"

    fs.run_regression(pytest_runner=pytest_runner_green, screen_runner=screen_runner)

    def pytest_runner_regressed():
        return (110, 2, "110 passed, 2 failed")  # simulated regression

    res = fs.run_regression(pytest_runner=pytest_runner_regressed, screen_runner=screen_runner)
    assert res["decision"] == "ALERT"
    assert "110" in res["result"]
    assert os.path.exists(fs.ALERT_PATH)
    with open(fs.ALERT_PATH) as f:
        alert_text = f.read()
    assert "116 -> 110" in alert_text


def test_regression_flags_screen_drift():
    def pytest_runner():
        return (116, 0, "116 passed")

    fs.run_regression(pytest_runner=pytest_runner, screen_runner=lambda: "BANK: flag A")
    res = fs.run_regression(pytest_runner=pytest_runner, screen_runner=lambda: "BANK: flag B (drifted)")
    assert res["decision"] == "ALERT"
    assert "drifted" in res["result"]


# ---------------------------------------------------------------------- CLI
def test_status_and_dry_run_never_create_state(monkeypatch):
    monkeypatch.setattr(fs, "_vm_stat_text", lambda: VMSTAT_HIGH)
    rc = fs.main(["--status"])
    assert rc == 0
    rc = fs.main(["--dry-run"])
    assert rc == 0
    lines = _jsonl_lines()
    assert lines[-1]["decision"] in ("would_run(dry)",)
    assert not os.path.exists(fs.STATE_PATH)  # dry-run never persists state


# =========================================== LOOP SAFETY: timeout + quarantine
# NEVER launches a real distill/heavy job. (a) uses `sleep`/a tiny inline
# python one-liner as the fake stubborn subprocess; the REAL heavy-job launch
# path (_launch_distill) is only exercised via monkeypatched fake `launch`
# callables in the family-queue tests below.

def test_timeout_kills_stubborn_job_and_logs_killed_timeout():
    """(a) A subprocess that ignores SIGTERM and would otherwise run forever
    must still die within ~timeout+grace, and the kill must be logged as a
    `killed_timeout` JSONL line with pid + elapsed -- this is the concrete
    proof that a runaway job cannot run unbounded."""
    cmd = ["python3", "-c",
          "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"]
    t0 = time.monotonic()
    res = fs._run_subprocess_with_timeout(cmd, cwd=fs.HERE, timeout_sec=2, grace_sec=1,
                                          job_name="fake_heavy_stubborn")
    wall_elapsed = time.monotonic() - t0

    assert res["timed_out"] is True
    # bounded well below the job's own 30s sleep -- proves the kill actually fired
    assert wall_elapsed < 2 + 1 + 5

    lines = _jsonl_lines()
    kills = [ln for ln in lines if ln["decision"] == "killed_timeout"]
    assert len(kills) == 1
    assert kills[0]["job"] == "fake_heavy_stubborn"
    assert kills[0]["result"]["pid"] == res["pid"]
    assert kills[0]["result"]["timeout_sec"] == 2
    assert kills[0]["result"]["elapsed_sec"] >= 2  # ran at least until the timeout fired


def test_timeout_kill_within_bound_for_cooperative_job():
    """A job that DOES honor SIGTERM is also bounded -- exits promptly at the
    SIGTERM, well before the grace/SIGKILL escalation is needed."""
    res = fs._run_subprocess_with_timeout(["sleep", "10"], cwd=fs.HERE, timeout_sec=1,
                                          grace_sec=5, job_name="fake_heavy_cooperative")
    assert res["timed_out"] is True
    assert res["elapsed"] < 1 + 5  # never reaches the full 10s sleep


def test_family_queue_launch_timeout_quarantines_immediately(monkeypatch):
    """(part of A) A single timeout-kill on a family-queue unit quarantines
    it immediately (no need to accumulate 3 failures for a timeout) and logs
    why -- the checkpoint does NOT retry the same doomed unit."""
    _committed_prereg("envelope", "hydcooler")
    state = fs._default_state()
    calls = {"n": 0}

    def fake_launch(family, bank):
        calls["n"] += 1
        return {"returncode": None, "timed_out": True, "elapsed": 5401.0, "pid": 4242}

    res = fs.family_queue_step(state, queue=[("envelope", "hydcooler")], launch=fake_launch)
    assert res["decision"] == "quarantined"
    assert calls["n"] == 1
    key = "family_queue:envelope:hydcooler"
    assert fs.is_quarantined(state, key)
    # idx must NOT have advanced -- the doomed unit is not retried
    assert state["family_queue_idx"] == 0

    # next tick: same unit is skipped_quarantined WITHOUT calling launch again
    res2 = fs.family_queue_step(state, queue=[("envelope", "hydcooler")], launch=fake_launch)
    assert res2["decision"] == "skipped_quarantined"
    assert calls["n"] == 1  # launch was NOT invoked again


def test_family_queue_quarantines_after_n_consecutive_failures(monkeypatch):
    """(b) A job that fails (non-zero rc, not a timeout) MAX_JOB_FAILURES
    times in a row becomes quarantined, and is skipped -- without launching
    again -- on the next tick."""
    monkeypatch.setattr(fs, "MAX_JOB_FAILURES", 3)
    _committed_prereg("envelope", "hydcooler")
    state = fs._default_state()
    calls = {"n": 0}

    def failing_launch(family, bank):
        calls["n"] += 1
        return {"returncode": 1, "timed_out": False, "elapsed": 1.0, "pid": 1000 + calls["n"]}

    queue = [("envelope", "hydcooler")]
    res1 = fs.family_queue_step(state, queue=queue, launch=failing_launch)
    assert res1["decision"] == "failed"
    res2 = fs.family_queue_step(state, queue=queue, launch=failing_launch)
    assert res2["decision"] == "failed"
    res3 = fs.family_queue_step(state, queue=queue, launch=failing_launch)
    assert res3["decision"] == "quarantined"
    assert calls["n"] == 3

    key = "family_queue:envelope:hydcooler"
    assert fs.is_quarantined(state, key)
    assert state["family_queue_idx"] == 0  # never advanced past the doomed unit

    # next tick: skipped, launch NOT called a 4th time
    res4 = fs.family_queue_step(state, queue=queue, launch=failing_launch)
    assert res4["decision"] == "skipped_quarantined"
    assert calls["n"] == 3

    # human clears it -> retried fresh (failure count reset)
    fs.clear_quarantine(state, key)
    assert not fs.is_quarantined(state, key)
    res5 = fs.family_queue_step(state, queue=queue, launch=failing_launch)
    assert res5["decision"] == "failed"
    assert calls["n"] == 4


def test_family_queue_success_advances_idx_and_resets_failures(monkeypatch):
    """Sanity counterpart to the quarantine tests: a clean success on the
    first try still advances the queue exactly like before this change."""
    _committed_prereg("envelope", "hydcooler")
    state = fs._default_state()

    def ok_launch(family, bank):
        return {"returncode": 0, "timed_out": False, "elapsed": 1.0, "pid": 1}

    res = fs.family_queue_step(state, queue=[("envelope", "hydcooler")], launch=ok_launch)
    assert res["decision"] == "ran"
    assert state["family_queue_idx"] == 1
    assert fs._job_failures(state).get("family_queue:envelope:hydcooler", 0) == 0


def test_clear_quarantine_cli(monkeypatch):
    """The human-clears-it escape hatch, exercised through the CLI entry
    point (not just the pure function)."""
    state = fs._default_state()
    fs.quarantine_job(state, "family_queue:envelope:hydcooler", "test reason")
    fs._save_state(state)

    rc = fs.main(["--clear-quarantine=family_queue:envelope:hydcooler"])
    assert rc == 0
    reloaded = fs._load_state()
    assert not fs.is_quarantined(reloaded, "family_queue:envelope:hydcooler")


# ------------------------------------------------- (c) tick terminates, one unit
def test_tick_terminates_and_does_exactly_one_unit(monkeypatch):
    """(c) Explicit no-internal-loop proof: a single --tick must return
    control to the caller almost instantly and perform exactly one unit of
    work (here: the regression step), never poll/retry/loop internally."""
    monkeypatch.setattr(fs, "_vm_stat_text", lambda: VMSTAT_HIGH)
    calls = {"regression": 0}

    def fake_pytest_runner():
        calls["regression"] += 1
        return (5, 0, "5 passed")

    monkeypatch.setattr(fs, "_default_pytest_runner", fake_pytest_runner)
    monkeypatch.setattr(fs, "_default_screen_runner", lambda: "BANK: ok")

    t0 = time.monotonic()
    rc = fs.main(["--tick"])
    elapsed = time.monotonic() - t0

    assert rc == 0
    assert calls["regression"] == 1
    assert elapsed < 5.0  # returns near-instantly -- no internal wait/retry loop
