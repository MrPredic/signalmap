# Source Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a label-free source profile, deterministic method-family router, and JSON registry without changing the existing Distill validation protocol.

**Architecture:** Create `signalmap/qualification.py` as a dependency-light module over `Bank`. It computes a stable profile, routes hard-compatible families, and persists source qualification metadata. Add a thin CLI command and focused tests; later Distill integration remains a separate change after this contract is validated.

**Tech Stack:** Python 3, NumPy, pytest, existing `signalmap.distill.Bank`.

## Global Constraints

- Profiles are label-free and contain no raw samples.
- Existing nested LOGO, permutation, null, and receipt behavior is unchanged.
- Excluded families are recorded as `not_measured_incompatible`, never as failed evidence.
- Source identity requires `source_id`, `device_id`, and `setup_id`.

### Task 1: Qualification profile and routing

**Files:**
- Create: `signalmap/qualification.py`
- Test: `signalmap/tests/test_qualification.py`

**Interfaces:**
- `profile_bank(bank: Bank, *, source_id: str, device_id: str, setup_id: str) -> SourceProfile`
- `route_method_families(profile: SourceProfile) -> RoutingDecision`
- `SourceProfile.to_dict() -> dict`

- [ ] **Step 1: Write failing tests** for single-channel routing, multi-channel routing, stable signatures, and label-free output.
- [ ] **Step 2: Run** `./.venv-research/bin/python -m pytest -q signalmap/tests/test_qualification.py`; expect import/API failures.
- [ ] **Step 3: Implement** deterministic NumPy-only profile metrics and conservative family predicates.
- [ ] **Step 4: Run** the focused test; expect all tests to pass.
- [ ] **Step 5: Commit** `feat: add label-free source qualification profile`.

### Task 2: Method registry and qualification state

**Files:**
- Modify: `signalmap/qualification.py`
- Modify: `signalmap/tests/test_qualification.py`

**Interfaces:**
- `MethodRegistry.register(profile, families, receipts) -> None`
- `MethodRegistry.status(profile) -> str`
- `MethodRegistry.save(path) -> None`
- `MethodRegistry.load(path) -> MethodRegistry`

- [ ] **Step 1: Write failing tests** for roundtrip, exact match (`confirmed`), changed setup (`qualification_required`), and incompatible family reasons.
- [ ] **Step 2: Run** the focused tests and verify the new assertions fail.
- [ ] **Step 3: Implement** stable JSON serialization and exact identity/signature matching.
- [ ] **Step 4: Run** focused tests; expect pass.
- [ ] **Step 5: Commit** `feat: persist source method qualification registry`.

### Task 3: CLI report

**Files:**
- Modify: `signalmap/cli.py`
- Modify: `signalmap/qualification.py`
- Test: `signalmap/tests/test_qualification.py`

**Interface:**
- `signalmap qualify --bank PATH --source-id ID --device-id ID --setup-id ID --out PATH [--multichannel]`

- [ ] **Step 1: Write failing CLI test** invoking `build_parser()` and checking JSON fields/reasons.
- [ ] **Step 2: Run** the test and verify the command is absent.
- [ ] **Step 3: Implement** the thin command using `load_bank`, `profile_bank`, and registry save.
- [ ] **Step 4: Run** focused CLI tests; expect pass.
- [ ] **Step 5: Commit** `feat: expose source qualification CLI`.

### Task 4: Regression verification and docs

**Files:**
- Modify: `README.md` or the existing CLI docs section only if the command is not discoverable.
- Test: `signalmap/tests/test_qualification.py`

- [ ] **Step 1: Add** a regression assertion that importing and running qualification does not alter `distill` defaults.
- [ ] **Step 2: Run** `./.venv-research/bin/python -m pytest -q signalmap/tests/test_qualification.py signalmap/tests/test_distill.py signalmap/tests/test_multichannel_distill.py`.
- [ ] **Step 3: Run** `./.venv-research/bin/python -m compileall -q signalmap`.
- [ ] **Step 4: Inspect** `git diff --check` and confirm only intended files are changed by this work.
