# Method Composer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, bounded compositional measurement search that can generate candidate recipes, test them with null controls, and export an auditable minimal recipe.

**Architecture:** Keep the existing `distill` grammar unchanged. Add a research/product-side `composer` module with immutable expression nodes, fixed operator registries, deterministic enumeration, and a scorer that uses group-aware held-out evaluation. Receipts remain JSON and are consumable by the existing deploy surface only after the evidence gate passes.

**Tech Stack:** Python 3.9+, NumPy, SciPy/scikit-learn already used by distill, JSON receipts, pytest.

## Global Constraints

- No arbitrary Python/code generation; candidates come only from the fixed v1 grammar.
- Candidate selection occurs only inside training folds; held-out groups are untouched until final scoring.
- A retrospective hit without mechanism-null collapse and independent replication is `candidate`, never `discovery`.
- Hard gates: finite output, coverage, runtime, memory, no future samples, and no unsupported channels.
- EP52 remains untouched evaluation data; EP53 remains unchanged confirmation data.
- Existing `distill` and monitor behavior must remain backward compatible.

### Task 1: Expression grammar and deterministic candidate generation

**Files:**
- Create: `signalmap/composer.py`
- Test: `signalmap/tests/test_composer.py`

**Interfaces:**
- `Expr` dataclass with `kind`, `args`, `params`, `canonical()`, `digest()`, and `cost()`.
- `enumerate_candidates(channels: int, budget: int, seed: int = 0) -> list[Expr]`.
- `evaluate_expr(expr: Expr, windows: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write failing tests** for canonical hashes, deterministic order, finite outputs, and unsupported channel rejection.
- [ ] **Step 2: Run `pytest -q signalmap/tests/test_composer.py` and confirm collection/implementation failures.**
- [ ] **Step 3: Implement only fixed primitives:** raw/diff/envelope/spectrum/phase/lag views; mean/std/slope/entropy/ratio/difference/coherence relations; short/medium/full scales.
- [ ] **Step 4: Implement canonicalization and deduplication so algebraically identical candidate descriptions have one digest.
- [ ] **Step 5: Run the focused tests and `python -m py_compile signalmap/composer.py`.**
- [ ] **Step 6: Commit only `signalmap/composer.py` and its tests with `feat: add bounded method composer grammar`.**

### Task 2: Group-aware evidence scorer and null controls

**Files:**
- Modify: `signalmap/composer.py`
- Test: `signalmap/tests/test_composer.py`

**Interfaces:**
- `score_candidates(candidates, X, y, groups, baseline, seed=0) -> list[dict]`.
- `mechanism_null(X, expr, kind, seed=0) -> np.ndarray`.
- `evidence_gate(record: dict) -> str` returning `candidate`, `supported`, `inconclusive`, or `null`.

- [ ] **Step 1: Add failing tests** for group-disjoint folds, label permutation near chance on synthetic data, phase/lag mechanism nulls, and gate downgrade when CI crosses zero.
- [ ] **Step 2: Run the focused tests and record the expected failures.**
- [ ] **Step 3: Implement leave-one-group-out scoring with a fixed lightweight classifier and paired candidate-vs-baseline deltas.**
- [ ] **Step 4: Implement deterministic group-label permutation and mechanism-specific phase/lag/channel-shuffle nulls.**
- [ ] **Step 5: Implement bootstrap CI with seed 0 and explicit missing/coverage accounting.**
- [ ] **Step 6: Run tests and verify null candidates do not receive `supported`.**

### Task 3: Receipt and minimal recipe export

**Files:**
- Modify: `signalmap/composer.py`
- Create: `signalmap/tests/test_composer_receipt.py`

**Interfaces:**
- `compose(X, y, groups, channel_names=None, budget=128, seed=0) -> dict`.
- `write_recipe(receipt: dict, path: str) -> None`.

- [ ] **Step 1: Write failing tests** for JSON round-trip, expression digest, hashes, metrics, nulls, cost, verdict, and recipe omission when the gate fails.
- [ ] **Step 2: Implement a self-describing receipt containing grammar version, seed, split/group hashes, candidate expression, baseline, CI, p-values, nulls, runtime/cost, and downgrade reason.**
- [ ] **Step 3: Export `recipe.json` only for `supported`; export the receipt for every run.**
- [ ] **Step 4: Run focused receipt tests and validate stable JSON under two identical runs.**

### Task 4: CLI proof command and synthetic reduction-to-practice

**Files:**
- Modify: `signalmap/cli.py`
- Modify: `signalmap/tests/test_e2e_chain.py`
- Create: `signalmap/tests/test_composer_cli.py`
- Modify: `README.md`

**Interfaces:**
- `signalmap prove --dataset ... --label-by ... --group-by ... --out ... --budget 128`.
- The command prints one-line verdict and writes `<out>.receipt.json`; it never claims discovery.

- [ ] **Step 1: Write failing CLI tests** for command parsing, receipt creation, and refusal to export a recipe on a null result.
- [ ] **Step 2: Wire the command to the composer without changing existing commands.**
- [ ] **Step 3: Add a deterministic synthetic benchmark with a planted mixed-scale cross-channel effect and a mechanism null.**
- [ ] **Step 4: Run the CLI test and synthetic proof command twice; compare receipt hashes.**
- [ ] **Step 5: Update README with the proof command, output semantics, and the explicit candidate-vs-discovery boundary.**
- [ ] **Step 6: Run the focused suite plus existing end-to-end tests; commit only composer/CLI/docs changes.**

## Verification commands

```bash
.venv-research/bin/python3 -m pytest -q signalmap/tests/test_composer.py signalmap/tests/test_composer_receipt.py signalmap/tests/test_composer_cli.py
.venv-research/bin/python3 -m pytest -q signalmap/tests/test_e2e_chain.py signalmap/tests/test_pipeline.py signalmap/tests/test_security.py
git diff --check
```

## Self-review

- Spec coverage: grammar, composition, deterministic search, nulls, CI gate,
  receipts, recipe export, EP52/EP53 holdout boundary, edge cost and CLI are
  covered by Tasks 1–4.
- No arbitrary code generation or label leakage is introduced.
- The public `distill` API remains unchanged; the composer is additive.
