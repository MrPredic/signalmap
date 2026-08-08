# Loop Safety Audit — every loop/automation in `research/factory/`, and its guaranteed termination bound

Trigger: a hand-launched heavy research script (`dcase_valve_readout.py`, PID 20027)
ran for ~19h with no wall-clock bound of its own. This audit inventories every
loop-shaped or recurring piece of automation in this directory and states the
bound that guarantees it cannot run unbounded. Anything without a bound is
flagged explicitly at the bottom.

## 1. `factory_scheduler.py --tick` — BOUNDED
One unit of work, then returns. No internal `while`/retry loop around the
priority ladder (regression -> retro-sweep -> family-queue): each branch
`return`s immediately after logging its decision.
- **Heavy subprocess launch** (`_launch_distill`, the only real heavy-job
  subprocess spawn in this file) is wrapped by `_run_subprocess_with_timeout`:
  hard-bounded at `HEAVY_JOB_TIMEOUT_SEC` (default 5400s/90min). On timeout:
  SIGTERM -> `JOB_KILL_GRACE_SEC` (default 10s) -> SIGKILL, and a
  `killed_timeout` JSONL line is appended (pid + elapsed).
- **Cheap subprocess launches** (`_default_pytest_runner`,
  `_default_screen_runner`) go through the same wrapper, bounded at
  `CHEAP_JOB_TIMEOUT_SEC` (default 1800s/30min).
- Worst-case wall time for one `--tick`: max(timeout+grace) across whichever
  single job it picked this tick, i.e. bounded by
  `HEAVY_JOB_TIMEOUT_SEC + JOB_KILL_GRACE_SEC` ≈ 5410s, never more, then it
  returns control (proved by `test_tick_terminates_and_does_exactly_one_unit`).
- **Retry/quarantine**: a family-queue unit that times out even once, or
  fails (non-zero returncode) `MAX_JOB_FAILURES` times in a row (default 3),
  is marked `quarantined` in scheduler state and is no longer scheduled --
  `is_quarantined()` short-circuits before any subprocess is launched --
  until a human runs `factory_scheduler.py --clear-quarantine=<key>`.
  Proved by `test_family_queue_launch_timeout_quarantines_immediately` and
  `test_family_queue_quarantines_after_n_consecutive_failures`.
- The single-slot flock (`_acquire_lock`) additionally guarantees at most one
  `--tick` process runs at a time; a second invocation logs `busy_skip` and
  exits 0 immediately (pre-existing, `test_lock_prevents_concurrent_tick`).

## 2. `launchd/com.signalmap.factory.plist` — BOUNDED
`StartInterval=3600`. launchd itself (not our code) enforces the recurrence;
each invocation is exactly one `--tick`, bounded per (1). launchd does not
overlap invocations of the same Label by default, so this is a second,
OS-level guarantee against pileup on top of the flock.

## 3. `launchd/com.signalmap.batterytransfer.plist` — BOUNDED (residual noted)
`StartInterval=1800`, runs `battery_pipeline.py run` once per invocation.
Internal `while True` in `_dl_file` (line ~96) is a bounded
read-until-EOF loop over `urllib.request.urlopen(..., timeout=120)` — it
terminates on EOF (`if not b: break`) or a socket timeout exception; retried
up to `tries=4` times with linear backoff (`time.sleep(3*(a+1))`). Worst case
per file ≈ `4 * (120 + 12)` ≈ 530s, finite.
**Residual (not fixed here, flagged for a future session):** there is no
single wall-clock cap wrapping the *entire* `download()`/`run()` call across
all files — a pathological number of slow-but-not-timing-out files could add
up. Not touched in this session (battery_pipeline is out of scope for
Deliverable A and not the trigger incident); recommend the same
`_run_subprocess_with_timeout`-style outer bound if this pipeline grows.

## 4. `geomag_watch.py` / `usgs_ep51_watch.py` ("run" mode) — BOUNDED
One-shot (own `StartInterval=3600` launchd plist per script). Internal fetch
retry is a **finite** `for i in range(RETRIES + 1)` loop with
`time.sleep(BACKOFF_S)` between attempts — no `while True`. Terminates once
`RETRIES` is exhausted or a fetch succeeds.

## 5. `ligo_loader.py` / `harden_1112.py` — BOUNDED
`while True` loops in `_download` (ligo_loader.py) and `_sha` (harden_1112.py)
are stream read-until-EOF loops (`if not chunk/b: break`); the network one
(`ligo_loader._download`) sits behind `urllib.request.urlopen(..., timeout=180)`
so a stalled connection raises rather than hanging forever. The local one
(`harden_1112._sha`, disk file hashing) has no hang risk (no network, no
blocking I/O source).

## 6. `receipt_ledger.py` — N/A
No loop; single append + optional full-chain `verify()` walk (bounded by
ledger length, a linear scan, not a wait-loop).

## ⚠️ Flagged: NOT bounded by this scheduler
Any heavy one-shot research script (`dcase_valve_readout.py`,
`distill_premium_case.py`, etc.) **launched directly** by a human/session
(`python foo.py &`, or via a Task/subagent), rather than through
`factory_scheduler.py`'s `family_queue_step` -> `_launch_distill` path, has
**no wall-clock bound of its own** — this is exactly what happened with
PID 20027 (`dcase_valve_readout.py`, ran ~19h, still running at the time this
audit started, exited on its own partway through this session). Deliverable A
only bounds subprocesses **the scheduler itself** launches.
**Recommendation for future sessions:** launch any new heavy/long research
script either (a) through `factory_scheduler.py`'s family-queue path so it
inherits the timeout+quarantine net, or (b) manually wrapped with
`factory_scheduler._run_subprocess_with_timeout(...)` / the shell `timeout(1)`
utility, rather than bare `python script.py &`.
