# PREREG-HEART-METHOD-FAMILIES

**Status: FINAL — 2026-07-22**

## Question

On a held-out ECG task with a patient/recording-level split, do cardiac-specific
method families add reproducible information beyond SignalMap's frozen lean
baseline? The first target is AF-vs-NSR because the repository already contains
an historical ECG result for this contrast. This is a method result, not a
clinical validation or diagnostic claim.

## Data and unit of analysis

- Primary data: MIT-BIH Atrial Fibrillation Database (`afdb`) and MIT-BIH
  Normal Sinus Rhythm Database (`nsrdb`) via the existing loader, if the public
  files are fetched after this preregistration.
- Recording/patient is the independent unit. Windows from one recording never
  cross train/test folds and are aggregated only after fold prediction.
- The existing historical 16-recording result is treated as prior/exploratory;
  it is not counted as an independent confirmation.
- If only a short segment is available or R-peak quality is insufficient, the
  beat/RR families are `not_measured`, not silently replaced by generic
  features.

## Frozen families

Four independent, theory-anchored families are registered. Parameters are fixed
before the first label-aware readout:

1. **QRS / beat morphology** — Pan–Tompkins-style fixed band-pass, derivative,
   squaring and moving integration for beat candidates; per-window robust
   beat-template deviation, QRS width proxy, slope/area/asymmetry and
   beat-to-beat morphology dispersion. Detection failures are reported.
2. **RR / HRV dynamics** — RR median, SDNN, RMSSD, pNN50, median absolute
   successive difference, lag-1 autocorrelation and fixed short-range entropy.
   Only windows with at least 10 valid intervals are eligible; no imputation.
3. **Multiscale wavelet** — fixed `db4` decomposition, levels 1–5 where the
   sampling rate permits, scale-energy ratios, normalized wavelet entropy and
   adjacent-scale energy change. No wavelet/grid search.
4. **Cross-signal cardiac coupling** — only if a second cardiac/respiratory
   channel exists in the fetched source: fixed-window coherence and
   phase/respiratory modulation summaries. With single-lead ECG this family is
   `not_measured_incompatible`, not a negative result.

The frozen generic baseline is the existing lean pair
`perm_entropy(order=3, normalize=True) + psd_slope` with the existing model and
split protocol. Family selection is not allowed inside the test fold.

## Null controls

- Patient-group-preserving label permutation.
- Phase-randomized surrogate preserving the linear spectrum.
- Beat-order shuffle preserving beat morphology while destroying rhythm
  organization (QRS/HRV families only).
- Within-recording time reversal as a morphology/dynamics stress control.

The cardiac-specific family must lose its advantage on the null that destroys
its claimed mechanism. A label permutation alone is insufficient.

## Gates and reporting

- Primary: paired per-recording delta against the frozen baseline, 10,000
  recording-bootstrap replicates, and a predeclared permutation p-value.
- A family is **SUPPORTED** only if: (a) real-data CI lower bound for delta is
  above zero, (b) patient-level performance is above chance, (c) its mechanism
  null collapses to chance/no advantage, and (d) the result is deterministic
  under a rerun.
- Otherwise report **EXCLUDED**, **NOT_MEASURED**, or **INCONCLUSIVE** with the
  reason. No family is added to a deploy spec from exploratory performance.
- Report per-recording performance, valid-beat coverage, missingness, runtime,
  memory, and exact hashes of source files, preregistration, code and receipt.

## Exploratory lane

After the primary gates only: beat-synchronous P/QRS/T morphology, DFA,
multiscale RR scaling, and nonlinear coupling may be explored as explicitly
secondary hypotheses. They cannot be promoted without a new preregistration
and an untouched confirmation set.

## Expected result classes

- **Method finding:** a cardiac family adds stable information under the above
  controls.
- **Mechanism finding:** the improvement disappears under its mechanism-null.
- **Engineering finding:** a family is accurate enough but too expensive or
  data-hungry for the small/IoT target.
- **No finding:** generic spectral readout remains the only supported signal;
  this is a valid result.

