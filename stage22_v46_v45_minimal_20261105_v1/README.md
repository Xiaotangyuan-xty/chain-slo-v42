# v46: v45 plus mixed coexistence and remaining-work boundaries only

This is the current v46 candidate, re-scoped on 2026-11-05. It starts from the
frozen v45 implementation in `stage15_stable_chain_v45_parallel_20261101_v2/round01`.
The earlier, broader v46 candidate at commit `8f12c202944b0a01d3c6898991612b04b117eab6`
and `stage21_v46_controlled_ac_20261104_v1/` is preserved as superseded research.
Its 30/200 observation does not describe this implementation.

## Implemented scope

1. **Mixed coexistence.** A live foreground latency/deadline singleton no longer
   switches all chains to the native scheduler. With both classes active, the
   adapter computes the unchanged v45 chain goals, retains native foreground
   reward-density order, and jointly chooses physical requests. Foreground has
   a half-pool/half-slot target after accounting for existing v45 chain leases;
   unused capacity can be borrowed. This is a declared coexistence policy, not
   a guaranteed optimal or measured fair allocation. Large foreground contexts
   may borrow beyond the target. Pure foreground delegates to the original native
   scheduler; pure chain delegates directly to the byte-identical v45 loop.
2. **Remaining-work boundaries.** Declared but unsubmitted current siblings
   contribute donor-estimated work/prefill, completed siblings are not charged,
   and virtual members never consume actual KV or slots. Current-call residual
   backoff is bounded by each call's own public `max_tokens - generated`.
   Future calls use the study-wide public cap (1024 unless explicitly declared
   in metadata), not a current sibling's smaller cap. Fallback donors must
   cover every declared stage. Unsupported structure returns an explicit
   unknown-support/zero-score result and remains available to v45 draining;
   zero score is not proof of infeasibility.

The v45 donor distance/cache formula remains unchanged. A donor with adequate
depth still uses v45's future-width scaling and duration formula; this version
does not introduce capacity-aware parallel waves.

## v45 behavior retained

- `native_chain.py`, `chain_admission.py`, the native-reference fixture, and the
  295-FIT-record `CHAIN_MODEL.json` are byte-identical to frozen v45.
- Same schedule-invocation cadence estimator and clamps, 50-tick replan period,
  empty-selection/start/sibling triggers, and incumbent multiplier.
- Same generated-token lease. No prefill-progress charge or wall-clock expiry.
- Same chain oversized fallback and logical reservation behavior. The physical
  native block manager still controls actual allocation; this release does not
  add a stronger logical-budget guarantee than v45.
- Same zero-positive-goal work-conserving drain: mixed zero-positive planning
  uses the same drain signal, not repeated deadline-sorted zero-reward selection.
- Same native executor. No new batch-execution cadence observer, transfer model,
  event-driven replan key, or large per-plan decision log.
- No A* stage-budget adapter. Original native stage-budget semantics are retained.

Mixed mode necessarily partitions request types and arbitrates their shared pool.
Within that adapter, chain replan triggers, cadence and leases retain the v45
rules. A newly arriving foreground request can wait for the next existing plan;
there is no added foreground/status trigger. Existing foreground-only bin plans
are cancelled when joint control takes ownership; these are unexecuted plans,
not completed transfers or physical KV allocations. No lease reset is introduced
on mode transitions. Small counters and a last-plan summary provide CPU evidence.

## Integration entry point

Use a fresh server process with this directory's `programs/` ahead of the old
stage programs on `PYTHONPATH`. Replace the old chain/mixed installer call with:

```python
from install_v46 import install
install(server, output_dir)
```

Call it before importing `native_chain` or `mixed_joint`. Do not additionally
install `mixed_fallback`, `scheduler_v46`, `observation_v46`, or `baseline_budget`
from the superseded package. The entry point applies the existing v45 minimal
observation transform, then installs the mixed adapter and the original v45
whole-chain deadline wrapper. It writes `V46_MINIMAL_RUNTIME.json` and a final
`MIXED_RUNTIME.json` when an output directory is provided.

This is a source candidate, not a launched experiment matrix. No old study,
authorization, controller, attempt, gate receipt or paper is overwritten.
Protocol binding and complete GPU regression are still required before making
performance claims. The known v45 43--45/200 results retain their v45 attribution.

## Validation

`tools/verify_scope.py` checks exact v45 core/model preservation, allowed forecast
method changes, pure-chain delegation and absence of superseded imports.
`programs/test_boundaries.py` runs with standard Python. In the existing frozen
native Python environment, with `CUDA_VISIBLE_DEVICES=''`, run each of:

```text
programs/test_boundaries.py
programs/controlled_native.py
programs/test_mixed_runtime.py
programs/test_install.py
```

These cover boundary counterexamples, native swap/recompute and prefix retention,
mixed foreground/chain model-scheduling opportunities, complete KV release,
pure-chain execution/selection/lease/replan trace equality, no wall-clock lease,
zero-reward draining, mode transition and the actual minimal-observation installer.
CPU fixtures do not execute a model, predict GPU attainment counts or prove TTFT/TBT
coexistence performance. CPU qualification records are in `evidence/v46_minimal/`.

Primary metric remains complete-chain SLO attainment. JCT is report-only and the
arrival-error threshold remains 0.5 seconds. This release runs no GPU experiment.
