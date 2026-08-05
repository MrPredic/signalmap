# Source Qualification and Method Registry

## Goal

Make `signalmap distill` source-aware: a new device/source is fully qualified once, while repeated runs on the same validated source reuse confirmed method families and only run a cheap drift/regression check.

## Non-goals

- Do not use labels, class outcomes, or held-out performance to build the source profile.
- Do not remove the existing nested LOGO, permutation, null, or receipt gates.
- Do not replace the base grammar with an unconstrained LLM or arbitrary program synthesizer.
- Do not claim that an excluded family failed; record it as `not_measured_incompatible`.

## Design

Add a small, dependency-light qualification module with three boundaries:

1. `profile_bank(bank)` computes reproducible, label-free measurements from the loaded bank: recordings, windows, channels, finite values, dynamic range, saturation, rough noise proxy, stationarity, spectral concentration, impulsiveness, and cross-channel availability.
2. `route_method_families(profile)` maps those measurements to compatible method families. Routing is deterministic and conservative; a family is excluded only when a hard precondition is missing.
3. `MethodRegistry` stores source identity, profile, qualified families, receipts, and profile version in JSON. A matching source may use `confirmed` families; a changed device/setup or profile version requires `qualification_required`.

The initial family vocabulary is `time_domain`, `spectral`, `envelope`, `recurrence`, `coherence`, and `causal_lag`. The registry distinguishes `confirmed`, `candidate`, `not_measured_incompatible`, and `qualification_required`.

The first release exposes profile/routing/registry APIs and a CLI report. Distill integration will consume the registry only after these contracts have regression tests; no default benchmark behavior changes in this slice.

## Data integrity

- Source identity is explicit (`source_id`, `device_id`, `setup_id`); no path-only identity is accepted.
- JSON is stable and human-readable.
- Profiles contain no labels or raw samples.
- Matching requires the same source identity, profile version, and measurement signature.

## Tests

- Synthetic single-channel bank: label-free profile and compatible family routing.
- Synthetic multi-channel bank: coherence/causal availability and deterministic routing.
- Registry save/load roundtrip and mismatch requiring qualification.
- CLI emits JSON with status and explicit reasons.
- Existing Distill tests remain unchanged and pass.

## Modular method extension

Method modules use `MethodCapability` and `MethodCatalog`. A capability has a
family, hard technical requirements, cost, evidence status, and extensible
module metadata. The core router returns eligible methods, hard exclusions, and
deterministic priorities. Only hard technical failures exclude a method;
diagnostic weakness is a priority signal, not a scientific negative result.

Evidence status is one of `candidate`, `speculative`, or `confirmed`.
Speculative modules are visible to exploratory callers and cannot silently
enter a confirmed source registry without their required receipts.
