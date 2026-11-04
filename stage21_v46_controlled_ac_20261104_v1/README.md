# v46: complete-work estimates and joint residency control

This version is a new candidate, not a relabeling of v45 results. Existing v42--v45 snapshots and server studies remain unchanged.

## Changes

- Remaining-work estimates include declared but not yet submitted current-stage siblings. Virtual members affect value estimates, never physical KV admission.
- Residual backoff is bounded by each call's public remaining generation cap. Future-stage caps use the declared study-wide limit, not the smallest current member cap or evaluation output lengths.
- Donors must cover every declared stage. Unsupported structures are explicitly tagged and drained without a fabricated zero-cost tail.
- Duration proxies account for memory/sequence-limited parallel waves. Decode cadence comes from actual model execution batches; transfer-only batches are sampled separately.
- Protected contexts pay for prefill progress and output progress, have a finite wall-clock eligibility window, and are subject to the same logical budget as new admissions.
- Mixed workloads use joint admission. Foreground requests retain native GMAX order within a half-pool/half-slot target. Chains use remaining-work density in the remaining budget. Both lanes can borrow unused capacity; this is a declared coexistence policy, not a demonstrated optimal allocation.
- Every ordinary admission is charged. A single physical-progress exception is explicit, charged in full, and allowed only if its current context is representable in the physical pool.
- Replanning also reacts to new work, status/progress changes and protection/deadline eligibility. Decision records include q/W, support/missing-member state, selected calls, protection, budget charges and proactive preemption. They are buffered with a finite evidence bound.

## Baselines and studies

Each fixed case has three complete new arms on the same physical GPU:

- `A`: original native JITServe scheduler and original online stage-budget implementation.
- `Astar`: the same native scheduler, with an explicit semantic adapter that includes the current stage in cumulative allocation and uses declared depth for its default allocation. Compatible historical matching still uses `max(historical, uniform)`.
- `C`: v46, with the same semantic adapter as Astar.

All arms share the fresh-connection client, per-token lifetime timestamps, actual-token auditing, native executor and physical KV pools. Initial max_num_seqs is32; original native A/Astar adaptive batch behavior is preserved and is not described as a fixed32 hard limit. C's explicit working set is limited to its initial sequence budget.

Pure cases contain the exact original200 DeepResearch prompts,1197 main calls and504695 output tokens, with Stage18 seed42 arrivals. Three blocks repeat these known inputs. Mixed cases preserve the exact Stage20 block1/2/3 WORKLOAD bytes:1800 roots,2399 main calls,120 DR chains,773778 output tokens and3600-second arrivals. Pure and mixed retain their respective historical prompt conventions; comparisons are within a workload, not an isolated test of adding foreground requests to otherwise identical prompts.

The primary outcome is complete-chain SLO attainment. JCT is reported, not constrained. Arrival error remains<=0.5seconds. All admitted work drains, including expired work. A failed validity check preserves the attempt and stops that GPU's remaining queue; there are no automatic retries or success-based continuation gates.

## CPU qualification and reproduction

`programs/test_v46.py` covers missing siblings, completed-member accounting, public cap bounds, unsupported tails, capacity-aware duration, charged protection, native block-manager joint execution and finite prefill protection. `qualify_runtime.py` also runs the preserved native-reference, export-statistics, lifetime-observer and real HTTP connection tests. This repository reuses the frozen external model/runtime paths described by PREPARE/FREEZE; weights and third-party binary runtimes are not bundled.

`build_cases.py` materializes/binds a new six-case matrix. Runtime control is in `case_runner.py` and `dispatch_matrix.py`. Never point them at an existing attempted study, reuse old authorization, or rewrite prior results. The generated study manifest and CPU receipts are required before launch.

The initial preparation error that double-formatted pure prompts was rejected before any GPU run; the rejected CPU inputs remain on the server. Final pure inputs are checked against every Stage18 prompt/token hash, and mixed inputs are checked byte-for-byte against Stage20.

Publication freezes implementation, not experimental success. No new performance claims are made here before complete validated results are collected.
