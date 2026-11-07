# v49pro: stage-aware paired terminal-credit control

The completed v49 mixed replay exposed a scheduling regression: fixed-hour native weighted-token goodput was **630.6125/s**, DR attainment was **4/60**, and drain took **1336.625 s**. Its 6336 joint plans consumed **499.953 s** in the recorded planning region (78.907 ms mean, 132.782 ms P95). These planning-region wall-clock durations exclude subsequent JSON serialization; they are not process CPU accounting or GPU execution time. The decision log was 863 MB. In 389 plans at least eight requests were live while only one or two were selected. This is evidence of expensive control and restrictive selection; it does not quantify the isolated causal effect of each component.

GPU1 was stopped at the user's request. v49 had completed with a PASS gate; Astar was interrupted, A did not run. That queue is **not** a completed three-method comparison. The original v49 code, its results and earlier versions remain intact.

## What changes

The implementation is in `stage41_v49pro_stage_rollout_20261107_v1/`.

* **Stage-aware donor scenarios.** `stage_scenarios.py` retains each stage's declared width, predicted prefill, context and remaining decode envelope. Future service is evaluated at future widths, rather than multiplying an entire root by the current batch unit time. With 50 units at width 1 /20 ms and another 50 at width 32 /72 ms, the forecast is 4.6 s; a 3 s slack is rejected. The reverse congestion pattern is also tested. Conditional prefix credit is paid only when the root completes within its real deadline.
* **Paired rolling service windows.** `paired_rollout.py` uses identical donor indices, cost snapshots, planning delay, physical budgets and continuation order for the reference and a proposal. It advances stage barriers, widths, prefill and shared decode service as simulated time progresses. It reports per-action completion times, empirical attainment frequency, resident losses, candidate gains and switching overhead. Completed/temporarily barrier-blocked selections allow common continuation backfill inside the window. It does not read future arrivals, evaluation targets or the benchmark end time.
* **Set search instead of single-add greed.** `mixed_control.py` explicitly seeds the native physical pack, incumbent and empty option. Pools up to eight are enumerated subject to evaluation/time budgets; larger pools use a bounded beam with pair-add, remove, 1-for-1 and 1-for-2 neighborhoods. Losing intermediate sets may survive. Regression cases cover the three-way batching trap and the 6-versus-4+4 KV trap, including an integrated credit rollout with real physical charges. This is bounded search, not a global-optimality claim.
* **Bound planning before expensive prediction.** Cheap public-state pruning precedes donor materialization; the default expensive pool is at most 12 actions and at most eight evenly spaced donor scenarios per root. Resident/reference losses cannot be silently omitted: if they do not fit in the model pool, the controller defers to native execution. Immutable future-stage templates and rollout results are cached; only widths encountered by a rollout are queried. The default planning allowance is 6 ms and is reserved in every evaluated deadline. It is a **soft wall-time guard**, not a hard real-time guarantee: checks occur between events/scenarios, and native planner/serialization/dispatch overhead remains separately observable. An overrun cannot commit a new candidate intervention.
* **Recover native execution on uncertainty.** Unsupported predictions, pool/event/time budget exhaustion and nonpositive paired gain release the candidate selection and invoke the native executor and its admission/planning paths. A real expired root has zero terminal credit, but its current required work remains in the comparison. Only known deadline expiry moves a root to the zero-credit background tier; a model probability of zero is not treated as proof of expiry. No request is deleted or aborted by this mechanism, and emitted prefixes remain subject to the native swap/recompute protocol.
* **Measure execution at the executor boundary.** `execution_observation.py` wraps both native sync and async executor entry/return, recording timestamps, active members, total context, prefill tokens, swap/copy state, preempted/ignored counts and scheduler/planning host time. No CUDA synchronization is added. `step_cost.py` retains long, prefill, transition and planning-associated observations in bounded strata. It no longer fits scheduler-entry intervals or silently discards calls over one second. The measurement is **executor host wall time**, including transfer/wait, and must not be called pure GPU kernel latency.
* **Bound diagnostic output.** All decisions are still streamed, but detailed action summaries are limited to the materialized pool. Per-executor-call features have a separate buffered stream. Complete source/sample counters and eviction/overwrite counters remain available; sparse or filtered strata are explicit.

## Preserved comparison contract

The pure DR v47 core (`native_chain.py`, `chain_forecast.py`, `chain_admission.py`, `token_reward.py`, `CHAIN_MODEL.json`) remains byte-identical. The reserved ToT length bank is unchanged. The native implicit ToT state machine and `[3,1,3,1]` annotation/cap256 are preserved. ToT-only service estimates use the new stage-aware envelope; the all-DR pure path delegates to the original core.

The shared predictor module differs from v49 only by accepting the explicit `v49pro` label. Weights, framed protocol, dynamic snapshots and cache repairs are unchanged. A/Astar continue to use their declared original/corrected budgets. The executor observer must be installed identically for fresh A/Astar/v49pro comparisons; do not silently reuse old baseline runs without that observer.

Use a **fresh process** and the comparison entry:

```python
from install_comparison_runtime import install
install(server, output_dir, method="v49pro", model_key="llama8",
        port=prediction_port, predictor_protocol="length_framed_pickle_v1")
```

Use `method="A"` or `"Astar"` for baselines. Candidate receipts are `V49PRO_RUNTIME.json`, `MIXED_RUNTIME.json`, `V49PRO_DECISIONS.jsonl`. All three methods produce `EXECUTOR_EVENTS.jsonl` and final `EXECUTOR_OBSERVATION.json`. Installation constructs no engine, runs no model and allocates no GPU. The candidate-only installer without shared configuration is explicitly not comparison-ready.

## Evidence and limitations

See `FREEZE_V49PRO.json`, `verify_v49pro_freeze.py` and `evidence/v49pro/` for exact source hashes and native CPU qualification. Verification imports the actual JITServe/vLLM classes, tests native ToT routing/keyword ABI, swap and recompute resume, complete required work, resource release, shared predictor semantics, executor metadata/constructor hooks, scenario counterexamples, planning budgets and unchanged pure-DR decisions. CUDA visibility is empty and CUDA initialization is asserted false. No GPU/model benchmark is part of this release.

The empirical donor frequency is an approximate `P_hat(T <= D)`, **not a calibrated probability guarantee**. A stage uses a conservative barrier envelope: its declared width stays active until its longest member completes; partial physical fit uses service waves. Prefill sharing and future context are approximations, and the continuation policy omits unknown future arrivals. Sparse execution strata extrapolate conservatively. The common rollout window is based on current service cost, not the one-hour evaluation boundary.

CPU budget synthesis is diagnostic only. It does not establish GPU goodput, and pool/time guards may deliberately choose native execution frequently under large backlogs. v49pro has **not** been shown to beat A or Astar. GPU experiments remain stopped until the user explicitly asks to start a new qualified queue. Evaluation inputs, SLOs, arrival tolerance, scoring and JCT reporting rules are not relaxed.
