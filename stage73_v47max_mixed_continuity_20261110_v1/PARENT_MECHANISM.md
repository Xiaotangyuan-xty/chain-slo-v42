# v47pro mechanism and boundaries

The code origin is frozen Stage30 v47, commit082e98c7c05431110bd6cd74234f909487b963ee.
The historical Stage30 directory and all later release freezes remain unchanged.

## Accepted changes

1. Replan on request/stage membership, completion or abort, prediction revision,
   canonical running/waiting/swapped and prefill state, released GPU or CPU
   capacity, resource configuration, mode epoch,
   50-generated-token service quantum, deadline crossings, or waiting/quantum
   timer expiry. A waiting set's0.5s re-evaluation interval starts when planning
   commits, rather than at its start; a long plan therefore cannot trigger itself.
   Non-urgent prediction/quantum/physical-state updates use a bounded coalescing
   interval derived from the observed busy round. Membership, real deadlines,
   preemption, configuration and capacity release remain urgent. TTFT, next cumulative token
   deadline and root deadline include a lead based on observed busy-round time.
   Ordinary decode KV growth and queue reordering do not cause a replan.
   A 50-token milestone triggers a full plan only when the protected set actually
   changes; an unchanged quantum is only cheap bookkeeping. Fifty ticks is only a fallback. These are internal controls, not new gates or
   hard real-time guarantees. No future arrival or true future output length is
   used to construct these triggers. Decision JSON serialization and file writes
   use a bounded background queue; drops and writer errors are explicitly counted.
   Every submitted plan retains its full rows and decision ID for audit.
2. Each donor scenario includes all unfinished members, unsubmitted siblings and
   all future public stages through cached current/future critical, work and
   terminal-credit components; the hot path does not materialize every future
   stage as fixed waves. The physical first selected batch and serial residual
   members use their actual current service costs, including integer prefill
   chunks. Current resident/foreground backgrounds remain charged throughout the
   current stage's residual work, while future stages use their own isolated
   cached units and allow those residents to release. The selector cannot retain
   full parallel root credit after reducing its first wave.
   Final pure and joint selections are revalued against the complete chosen set
   and every retained running resident; non-protected roots whose credit becomes
   zero are removed from desired selection, with bounded rechecking of the set.
   Infeasible/zero-prediction work goes through full native required execution;
   predicted zero is not proof of deadline expiry. The donor timely fraction is
   an empirical scenario frequency, not a calibrated probability.
3. A deselected resident is retained unless an actual selected waiting/swapped
   admission is blocked on KV or sequence slots. Retained residents are included
   in the service and physical-resource background before execution, rather than
   being omitted from the value estimate. Eviction chooses a coarse
   swap-versus-recompute cost and respects the physical allocator. All native
   and proactive preemptions share one lease/plan invalidation boundary.
   Synchronous recompute may continue scheduling in the same round; actual
   swap-out mappings require a separate dependency round. Foreground in joint admission has the
   same bounded residence protection and recovery/prefill charge as chains.
   The allowed execution mask is declared before invoking the native scheduler,
   including retained residents; desired admission and allowed execution are
   separately logged. Foreground-only mode delegates to native selection and
   explicitly releases the candidate selection filter.
4. A small online model uses bounded batches of actual observations, indexed by
   batch size, average context, prefill chunk and transfer transition. Estimates
   use the larger of the latest observation and rolling P90. Unsupported strata
   use conservative scaling/backoff and observed busy-round time, without a
   0.2s upper clamp. Unit estimates are cached only until new evidence arrives.
   Idle gaps without any continuously live request are excluded. Prefill and
   recompute use integer waves of the configured token budget. Service units use
   executor host boundaries only. Scheduler and planning host time are separate
   diagnostics and are not multiplied by future token counts; busy rounds only
   affect deadline lead and unsupported fallback. This is a coarse safety model,
   not a counterfactual search or a claim that batching/interference are perfectly predicted.
5. Each scheduler owns a copied config and a saved initial pure/joint sequence
   cap. Native foreground adjustment changes only its SchedulingBudget. Expired
   drain passes a temporary private cap into chunked scheduling and restores the
   prior config object. Every mode change clears selected rows, leases, old
   membership/priority snapshots and native bin/pending-swap plans through one
   function and advances the mode epoch. Physical KV is never freed by clearing
   an unexecuted plan. The native idle guard preserves pending requests for
   controller failure handling rather than aborting them after idle iterations.

## Preserved and shared

The 295 FIT DR training asset is byte-identical to v47. The neighbor matching,
observed-prefix survival conditioning, root terminal reward and1.2 incumbent
ordering factor originate in v47. The coexistence target reserves roughly half
the available KV/slots for foreground before chain admission. The score remains
expected root terminal credit divided by remaining service work, now expressed
in estimated seconds. It does not reuse A/Astar's root scoring or sliding-window
pack in the mixed/pure paths. The foreground reward formula and foreground-only
native pack were already in v47 and remain explicitly inherited.

Framed shared QRF transport, prediction revision/cache repair, actual-text
credit settlement, native implicit ToT public[3,1,3,1]/cap256 annotation, and the
executor observer are copied from the shared comparison infrastructure. The
shared_prediction change only adds the v47pro method label. A/Astar comparison
installers use the same infrastructure; no baseline scoring is changed. v47pro
retains v47's stage budget semantics; Astar's cumulative-stage budget adapter is
only installed for Astar. These shared fixes and native execution are not new
candidate contributions. Known ToT structure does not make the DR donor asset a
calibrated ToT predictor.

## Limits and reporting

Future arrivals, prediction error, interrupting work, and unseen service strata
remain uncertain. Fixed-wave and long-lived-background estimates can be
pessimistic; actual subsequent replans can interrupt modeled waves. The online
P90/last estimate and lead timer are heuristics, not hard service bounds. The
work-conserving drain completes expired requests and cannot restore credit
already missed in the fixed hour.

A single development prefix validates exercised behavior and completeness. It
cannot establish full-hour throughput, relative performance, stable superiority
or statistical significance. Keep the original technical failures, interrupted
runs and superseded sources visible. Gate PASS is a technical validity result.
The fixed-hour main metric is settled credit/3600; development horizons use
their explicit denominator and are never called hour results. Full replay and
drain are separate; JCT adds no validity constraint; arrival error<=0.5s remains.

The legacy `future_components` helper remains for traceability, but v47pro's
`estimate`/`revalue` paths do not call it. Their prefill/recompute estimates use
configured integer token-budget chunks and the online service model.

## Stage61 diagnostic override

The current diagnostic workspace intentionally passes `forecaster=None` to the
pure and joint selectors. Initial forecasts remain enabled, but selected-wave,
resident-background and final revaluation are skipped. This is a measurement
arm for separating planning cost from selection semantics, not a final policy.
The bounded deselected-resident turnover remains active: selected requests and
those in the current `last_protected` lease cannot be evicted, while other
deselected residents use the existing bounded native preemption path.

## Stage62 pure-core restoration control

The current workspace routes chain-only operation through the isolated frozen
v47 implementation from commit 082e98c7. Its original forecast, selector,
leases, 50-tick cadence, preemption and reversible expired drain are used
instead of pro event replanning, ServiceModel estimates, selected/final revalue
and the Stage60 turnover logic. Shared prediction, native correctness fixes and
executor observation remain common. The mixed/foreground path retains the
current diagnostic implementation. Entering or leaving pure mode rebuilds the
corresponding policy from live native queues and clears selected, lease and
drain state so no previous-mode selection can contaminate pure execution.

## Stage64 arrival-triggered frozen replanning

The frozen pure scheduler retains its original forecast/select/lease/preemption
and drain behavior, but its original replan condition also triggers when a
previously unseen request ID appears. This covers new independent roots and new
stage/sibling arrivals before the 50-tick fallback. No pro prediction, physical,
deadline or service-quantum event is added; ServiceModel and revalue remain
unused. With no new arrival, the trace remains equivalent to frozen v47.
