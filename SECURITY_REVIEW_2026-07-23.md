# SignalMap — Security Review, 2026-07-23

Trigger: Anthropic's "Claude Security" product launched in public beta (2026-07-22). Ran an
equivalent review against this codebase to see whether the same class of automated review would
catch anything here. Review-only — no code changed as part of this pass.

Scope reviewed: `signalmap/` (36 files, CLI + core product), `research/` (16 files),
`research/factory/` (77 files, maintainer-only research tooling), `sensors/`, `examples/`.
Excluded: `.venv-research/` (vendored virtualenv, not project code), `signalmap/tests/`.

## Finding 1 — Unsafe deserialization via `torch.load` without `weights_only=True`

- **Severity:** High
- **Confidence:** 8/10
- **Files:**
  - `signalmap/detector.py:106` — `Detector.load()`: `torch.load(path, map_location="cpu")`
  - `signalmap/models.py:20` — `AutoencoderModel.__init__()`:
    `self.net.load_state_dict(torch.load(weights, map_location="cpu"))`
  - `signalmap/visualize.py:53` — `build()`:
    `model.load_state_dict(torch.load(model_path, map_location="cpu"))`
- **Description:** all three call `torch.load` with no `weights_only=True`. `torch.load` is
  pickle-based; on PyTorch versions before 2.6 (this project's `pyproject.toml` only pins
  `torch>=2.2`, no upper bound) the default is `weights_only=False`, which unpickles arbitrary
  Python objects and can execute attacker-controlled code via a crafted `__reduce__`/`__setstate__`
  payload in a `.pt` file. Even on torch>=2.6 (default flipped to `weights_only=True`), the code
  doesn't require/pin that version or set the flag explicitly, so safety silently depends on
  whatever torch happens to be installed.
- **Exploit scenario:** this hits the product's core workflow directly. `signalmap monitor
  --detector evil.pt` (`cmd_monitor` in `signalmap/cli.py:115` → `Detector.load(args.detector)`)
  takes a detector file path as a plain CLI argument. Detector/weights files are meant to be
  produced by `fit` and shared/deployed — downloaded from a repo, sent by a colleague, pulled from
  a public model-sharing location, since this is an OSS tool. An attacker who gets a victim to run
  `signalmap monitor --detector <malicious.pt>` (same for `--weights` on `run`, `--model` on
  `map`) achieves arbitrary code execution at load time, before any actual monitoring happens.
- **Suggested fix (not applied — review only):** pass `weights_only=True` explicitly at all three
  call sites; consider pinning `torch>=2.6` if the flag alone isn't considered sufficient
  documentation of intent. Document detector/model files as a trust boundary in the README.

## Everything else checked — no findings ≥7

- **`np.load`** (`ingest.py:64`, `multichannel.py:51`, `distill.py:195,319`) — no
  `allow_pickle=True` anywhere in the actual product surface; numpy's safe default applies.
  (Several `research/factory/*.py` maintainer-only scripts do use `allow_pickle=True`, but only
  against the maintainer's own fixed local cache paths, never a CLI-supplied/attacker-reachable
  path — out of scope as internal research tooling, not the public product.)
- **`DistilledDetector.load`** (`distill.py:822`) — plain `json.load`, safe.
- **pickle / yaml.load / eval / exec** — none found anywhere in scope (the only `.eval()` hits
  are PyTorch's `nn.Module.eval()`, unrelated).
- **Path traversal** — CLI file-path args are used as local filesystem paths in a single-user CLI
  context; no server/multi-tenant boundary crosses them.
- **Command/subprocess injection** — no shell-out in `signalmap/` product code.
  `research/factory/` has a couple of `shell=True`/`os.system` calls, but all with hardcoded
  strings run locally by the maintainer, not built from untrusted input.
- **Hardcoded credentials/keys** — none found.
- **`examples/fetch_cwru.py`** — downloads from a community GitHub mirror pinned to a specific
  commit SHA over HTTPS, parsed with `scipy.io.loadmat` (no code-exec vector there).

## Addendum — CI workflow (`.github/workflows/ci.yml`), checked after initial pass

Not covered in the first scoping (it's not under `signalmap/`/`research/`/`examples/`). Checked
separately: `on: pull_request` (not the risky `pull_request_target`), no untrusted context
(`github.event.pull_request.title`, etc.) interpolated into any `run:` step, no secrets used, no
`pull_request_target` + PR-head-checkout combination. Actions are pinned to major-version tags
(`actions/checkout@v4`, `actions/setup-python@v5`) rather than a full commit SHA — a supply-chain
hygiene gap, but not independently exploitable and excluded under this review's own
"outdated/unpinned dependency" exclusion rule, so not filed as a finding. Clean.

## Bottom line

One real, high-confidence finding: unsafe `torch.load` on detector/model files, reachable
directly from the CLI's main `monitor`/`run`/`map` commands. Everything else — including the
`.npy` ingestion path (the a priori most likely place for a pickle-style issue in a sensor-data
tool) and the CI workflow — came back clean. Not fixed as part of this pass (review-only per
instruction); flagging for a follow-up change.

**Coverage note:** this was a category-scoped review (injection, deserialization, path
traversal, crypto, hardcoded secrets, plus the CI-workflow check added above) via targeted
grep + read, not a line-by-line manual audit of every file.

## Fresh-eyes counter-review (independent second pass)

Run after the initial pass, specifically to check whether anything was missed. Result:
- **Finding 1 confirmed accurate, no discrepancy.** Independently traced the exploit path from
  `cli.py`'s `cmd_monitor`/`cmd_run`/`cmd_map` argument parsing through to all three `torch.load`
  call sites — matches the original description exactly. Grepped for `weights_only` across the
  whole repo: zero hits, so the fix has not been applied anywhere yet and the finding is still
  fully live as of this check.
- **No new findings ≥7 confidence.** Independently re-verified every `torch.load`/`np.load`/
  `pickle`/`yaml.load`/`joblib`/`eval`/`exec`/`subprocess` call site across the same scope — same
  conclusions as the first pass on all of them (the `research/factory/*.py` `allow_pickle=True`
  sites all load from hardcoded maintainer-only cache paths, never a CLI-supplied one).

**Verdict: the review is complete for the categories in scope.** One real, confirmed, still-open
finding (unsafe `torch.load`); nothing else surfaced across two independent passes.

## Follow-up remediation — 2026-07-23

The finding was fixed in the product surface:

- `signalmap/detector.py`
- `signalmap/models.py`
- `signalmap/visualize.py`

All product `torch.load` calls now pass `weights_only=True`. A regression test
scans product call sites, and a runtime test confirms that an executable pickle
payload is rejected without running its reducer. Model artifacts remain a trust
boundary and should only be loaded from trusted sources.

Verification:

```text
3 passed, 13 deselected
```
