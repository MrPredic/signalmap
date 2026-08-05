# Changelog

## 0.5.0 (2026-08-05)

Added
- Signed verdict receipts (`signalmap.receipt/1`): every `distill`, `fit` and
  `monitor` run emits versioned JSON — claim, verdict
  (INCLUDED/EXCLUDED/PASS/REFUSED), evidence, input hashes, Ed25519 signature.
  The signing key lives in `~/.signalmap/signing_key` (0600) and never enters
  the repo; only the public key travels in the receipt.
- `tools/verify_receipt.py`: standalone verifier that imports **nothing** from
  signalmap — stdlib plus `cryptography`. Checks signature, schema and internal
  consistency (a REFUSED verdict may not carry a deploy spec).
- Verdict corpus: the eight preregistered premium-family verdicts (rqa,
  coherence, envelope over six banks; 2 included, 6 honest exclusions) shipped
  as signed receipts in `research/factory/receipts/`, each pinning the sha256
  of the report it transcribes and labelled `archive_signature` — signed, not
  re-run. A suite gate fails if a shipped receipt goes stale.
- `signalmap corpus` rebuilds that corpus and prints the traction line.
- `signalmap prove` records the permutation-resolution floor of each split and
  reports `inconclusive` (cannot resolve) instead of `null` (no evidence) when
  a p could never have cleared the threshold at that group count.
- `signalmap qualify` (source profiling and method-family routing) and the
  compositional `composer` grammar (`composer-v2`).

- Published to PyPI: `pip install signalmap`. Releases upload through GitHub
  Actions via PyPI trusted publishing (OIDC), so no API token is stored
  anywhere.
- Test coverage for the seven product modules that had none — `train`,
  `visualize`, `embed`, the sinks, the store, `discover`, the simulator and the
  `fit --dataset` path. 116 -> 172 tests, coverage 78% -> 89%.
- Lint gate in CI (ruff, pinned version, explicitly selected rules).

- README section "Signed verdicts, verifiable without us": the receipt, the
  standalone verifier and the shipped corpus in three commands.

Fixed
- `signalmap corpus` printed the absolute path of every receipt, so a pasted
  listing carried the maintainer's home directory. Paths under the working
  directory now print relative to it.
- The CI job that proves standalone verification ran from the checkout root,
  where the source tree is importable through cwd: its "signalmap must not be
  importable" assertion fired on every run and the verifier was never
  exercised. It now runs outside the checkout.
- `test_training_reduces_reconstruction_loss` compared two independently
  initialised models and failed at random (1 in 3 parallel runs); both runs now
  start from the same seed.
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
