# Changelog

## Unreleased

Added
- Published to PyPI: `pip install signalmap`. Releases upload through GitHub
  Actions via PyPI trusted publishing (OIDC), so no API token is stored
  anywhere.
- Test coverage for the seven product modules that had none — `train`,
  `visualize`, `embed`, the sinks, the store, `discover`, the simulator and the
  `fit --dataset` path. 116 -> 172 tests, coverage 78% -> 89%.
- Lint gate in CI (ruff, pinned version, explicitly selected rules).

Fixed
- `signalmap map` crashed with `IndexError` on a single-recording dataset: the
  2D projection returned one column when the data could not span two. It now
  always returns two, and skips UMAP below three points.
- `signalmap train --synthetic 0` reported a missing data source instead of
  naming the real problem, the frame count.
- `causal.py` linear-Granger design matrix used an out-of-scope loop variable
  after a rename; caught by the new lint gate.

## 0.4.0 (2026-07-27)

First release with the distill pipeline as a first-class product surface.

Added
- `signalmap distill`: compositional feature grammar with a capacity gate
  (budget = C × recordings) and a gauntlet receipt — nested LOGO accuracy,
  group-permutation p, label-shuffle null, cost per window. Output is a
  deployable `spec.json`.
- Premium families with a cost-receipted champion rule (`--premium
  rqa,coherence`). A family enters the deploy spec only on a paired-CI-solid
  win over the base selection; the receipt reports the refusal otherwise.
  Verdicts on CWRU, CALCE, UCI hydraulic and UCI gas were preregistered before
  readout — two admissions, three refusals (details in the README).
- Multi-channel banks: `--multichannel` for CSV-with-header and 2-D `.npy`
  recordings, synchronous (C, 1024) windows, `--channel-axis` override.
  Single-channel behaviour is pinned bit-for-bit in the test suite.
- `signalmap fit --spec spec.json --bank dir/` and `signalmap monitor
  --detector det.json --bank dir/`: the distilled detector is now on the CLI,
  with the alert threshold calibrated from the healthy envelope at fit time.
- Text ingest hardening: BOM/CRLF/header tolerant, counts ragged and
  non-finite rows, fails closed when more than 5% of a column is unreadable.

Fixed
- `distill` fails closed with a clear message on a one-recording bank instead
  of leaking a raw scikit-learn error — leave-one-group-out needs a recording
  to hold out.
- Multi-channel 2-D `.npy` ingest warns when the array is near-square, where the
  longer-axis-is-time heuristic is a guess; pass `--channel-axis` to be explicit.
- Every command that writes a file now creates its parent directory first.
  `data/` and `artifacts/` are gitignored, so on a fresh `git clone` the
  quickstart commands (`benchmark`, `universal`, `train`, `distill`, `fit`,
  `ingest-file`, the parquet sink, map export) previously crashed with
  `FileNotFoundError` / "Parent directory does not exist" instead of running.

Notes
- The receipt separates two verdicts on purpose: PASS/FAIL gates the base
  selection, the champion rule gates each premium family. Both can disagree,
  and the report explains why when they do.
- 111 tests. Earlier development (0.1–0.3: ingest/pipeline/Conv-AE platform,
  CWRU fit/monitor validation, causal and coupling discovery) predates this
  changelog; see the git history.
