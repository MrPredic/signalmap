# PREREG -- Factory Family-Queue (TEMPLATE, NOT FINALIZED)

STATUS: DRAFT

**This document does NOT authorize any run.** `factory_scheduler.py`'s
family-queue gate (`_prereg_committed`) refuses to run any `(family, bank)`
job as long as this file either (a) does not exist, (b) still contains a
`<!-- TODO: ... -->` marker or the `STATUS: DRAFT` line above, or (c) is not
referenced by a logged `receipt_ledger` entry. All three are currently true
-- the gate is correctly REFUSING right now. To activate the family-queue,
a human must fill in every TODO block below, remove the DRAFT marker, commit,
and log a receipt (e.g. `log_receipt("PREREG-FACTORY-FAMILY-QUEUE", {...})`)
that references this file's name before the scheduler will ever run it.

## Fixed family generator / search space
<!-- TODO: enumerate exactly which entries of signalmap/premium.py
     PREMIUM_FAMILIES the queue is allowed to draw from (e.g. rqa, envelope,
     coherence -- the fixed set that exists today). No family may be added
     to this list after seeing any readout. The queue itself
     (factory_scheduler.FAMILY_QUEUE) must be a closed, enumerated list of
     (family, bank) pairs decided here, not generated dynamically at run
     time. -->

## Champion rule
Unchanged from `distill_premium_case.py` / the existing per-family PREREG_*
documents: PASS = 95%-bootstrap paired-CI of the per-fold (augmented -
base) accuracy difference over LOGO-recording folds, CI-lo > 0 ->
**INCLUDED**; otherwise **EXCLUDED**. No new criterion, no threshold tuning
per bank.

## Bank set
<!-- TODO: enumerate the exact banks (from readout_screen.banks() /
     distill_premium_case.BANKS) this queue may run against, and the
     (family, bank) pairs specifically -- mirrors
     factory_scheduler.FAMILY_QUEUE, which must be kept in lockstep with
     this list. -->

## EXCLUDED is valid
An EXCLUDED verdict on any (family, bank) pair is a complete, reportable
result, not a failure to be retried or reasons to add a new family. This
mirrors PREREG_DISTILL_PREMIUM_ENVELOPE.md's explicit stance: the gate is
the product, not any particular family's win rate.

## No post-hoc family selection
The queue processes pairs in the fixed order declared above (FIFO cursor,
`factory_scheduler_state.json::family_queue_idx`). No pair may be skipped,
reordered, or added because an earlier pair's readout looked promising or
disappointing. Multiplicity (N pairs run = N independent verdicts, no
shared threshold) must be declared here once the pair count is fixed.

## Ledger commitment
<!-- TODO: after finalizing, run `log_receipt("PREREG-FACTORY-FAMILY-QUEUE",
     {...this document's frozen content or a hash of it...})` and paste the
     resulting ledger tip hash here as the commitment record. -->
