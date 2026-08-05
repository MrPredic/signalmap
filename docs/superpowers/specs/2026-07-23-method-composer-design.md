# Method Composer — Design

**Date:** 2026-07-23  
**Status:** Design approved in principle; implementation pending written review  
**Goal:** make SignalMap able to generate and test novel, mixed measurement
recipes retrospectively without turning selection noise into a discovery claim.

## Product claim

SignalMap can search a bounded space of compositional measurement operators,
identify candidates that carry stable information in a held-out regime, and
emit a small, replayable recipe with evidence and failure controls.

This is a candidate-generation instrument. A surviving recipe is a measurement
hypothesis, not proof of new physics.

## Scope of v1

The first implementation searches single- and multi-channel windows over three
explicit axes:

- **representation:** raw, difference, envelope, signed envelope, spectrum,
  spectral shape, phase, and fixed lag views;
- **relation:** scalar summaries, ratios, differences, cross-channel coherence,
  and fixed lag agreement;
- **scale:** short, medium, and full-window summaries with fixed boundaries.

It produces a canonical expression such as:

```text
ratio(spec_slope(envelope(x, scale=medium)),
      spec_slope(envelope(y, scale=medium)))
```

Expressions are canonicalized, hashed, costed, and deduplicated before scoring.
No arbitrary Python/code generation is allowed in v1.

## Search protocol

1. Load a bank with recording/group identity preserved.
2. Split by the declared independent unit before any label-aware scoring.
3. Generate candidates deterministically from a fixed grammar and seed.
4. Apply hard gates: finite output, minimum coverage, bounded runtime/memory,
   no future samples, no cross-fold state, and no unsupported channel use.
5. Rank only on the training folds using the existing distill scorer.
6. Evaluate the frozen top candidates on untouched groups.
7. Run mechanism-specific nulls and a label/group permutation null.
8. Emit `candidate`, `supported`, `inconclusive`, or `null`; never silently
   promote a candidate into a deploy spec.

## Evidence gates

A candidate is `supported` only when all are true:

- held-out improvement over the frozen baseline has a bootstrap CI above zero;
- the independent-unit permutation p-value passes the preregistered threshold;
- the relevant mechanism null removes the effect;
- the candidate is stable across the declared station/device/time replication;
- the recipe remains below the declared cost and memory budget;
- the full expression and hashes are present in the receipt.

Any failed condition is reported with its reason. A retrospective hit without a
future or cross-group confirmation is `candidate`, never `discovery`.

## Kīlauea reduction-to-practice

The first flagship run uses the frozen EP52 protocol:

- UWE and RIMD HHZ;
- fixed T−1h, T−6h, T−12h offsets;
- deep-pause same-clock controls;
- primary baseline `perm_entropy + psd_slope`;
- candidate families restricted to amplitude-invariant spectral/envelope and
  cross-station expressions;
- EP52 is the first untouched evaluation; EP53 is unchanged confirmation.

The composer may generate candidates retrospectively from EP1–EP51 only. It may
not read EP52 labels or choose candidates using EP52 data.

## Receipt and deploy output

Each run writes:

- canonical candidate expression and expression hash;
- grammar/version/seed and search budget;
- data/group/split hashes;
- baseline and candidate metrics with confidence intervals;
- null-control metrics and p-values;
- runtime, peak memory, and estimated edge cost;
- verdict and explicit downgrade reason;
- optional minimal `recipe.json` containing only the accepted operators.

## Non-goals

- no claim of new physics from retrospective association;
- no unconstrained genetic programming or LLM-generated executable code;
- no automatic clinical, safety, or operational decision;
- no large feature-library dump replacing evidence;
- no method selection on EP52/EP53 after observing their outcomes.

## Success criterion for launch

One reproducible command must generate at least one nontrivial candidate on a
known retrospective bank, show that its signal disappears under its mechanism
null, and export a recipe small enough to run through the existing distill/
monitor surface. A clean `candidate` or `null` is still a successful system
test; unsupported discovery language is not.

