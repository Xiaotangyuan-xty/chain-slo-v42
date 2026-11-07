# v49: mixed terminal-credit scheduling with compute admission

Parent: v48 `a4aa15e7bc1efb51b390c0fc445596f3547c8161`.
This is a CPU-qualified candidate. No v49 GPU generation, profiling or replay
has been performed. Higher goodput than A/Astar has not yet been demonstrated.

## Trace-backed problem

The completed, jointly repaired Stage37 Llama8B mixed runs gave fixed-hour
weighted-token goodput of 759.81 for v48, 769.69 for A, and 1010.92 for Astar.
All three final gates and independent raw audits passed. Astar attained36/60 DR
roots versus v48's12/60; both attained60/60 ToT roots. Astar had24 exclusive timely
DR roots and95 exclusive timely deadline singletons, with no reverse exclusive
attainments in those two classes. These are one-run observations, not independent
seed confidence intervals.

v48 improved first-token response and nearly eliminated long pauses, but all4096
retained mixed plans selected every live member. Slots and KV never bound those
plans, so the new credit and interruption calculations could not change the
selected set. Already-running zero-credit work was retained and background work
used physical vacancies while competing for decode execution. The full log had
4874 plans;778 early plans were overwritten, which limits retrospective coverage.

v49 separates physical fit from compute opportunity cost. It contains no root
IDs, trace identifiers, arrival rates, benchmark end time or test output labels.
The existing evaluation arrays were used for diagnosis, not to train or retune
historical predictor weights. Future performance claims require held-out paired
GPU runs under the same common prediction repair.

## Online compute cost and terminal-credit curves

`step_cost.py` keeps at most64 observed decode-only scheduler intervals per batch
member count, up to128 counts. Samples outside1ms-1s are excluded, as are scheduling
outputs involving prefill, copy, swap or a planning step. The next scheduler call
supplies the elapsed interval, which includes ordinary execution/loop overhead;
this is a bounded online proxy, not a GPU latency oracle. Bucket medians are made
monotone and interpolated between observed counts. Above the observed range, a
positive slope with a nonnegative fixed-cost component from the last two points
is retained; otherwise a conservative proportional extrapolation is used. This
avoids locking selection at the largest previously seen width when batching
benefits have already been observed.
No speed-up below a cold reference batch is invented. Attention/context effects
are not independently identified, and low-data/extrapolation behavior must be
validated on new hardware. These limits are recorded in every plan.

`credit_curve.py` uses the same historical neighbors, survival conditioning,
prefill/critical-work boundaries, caps, safety factor and complete-chain token
credit as v48's frozen DR/ToT forecasters. It materializes historical scenarios
once, then evaluates their terminal credit over candidate batch costs. CPU tests
compare each curve against the unchanged forecaster across deadlines and unit
costs. Completed compound prefix credit remains conditional on timely root
completion. Latency credit excludes already-settled prefix and preserves the
native cumulative-token deadline rule and late-TTFT recovery.

For a selected set S, the mixed objective proxy is the sum over actions of

    expected terminal credit at predicted step_cost(|S|)
    / expected remaining aggregate work
    * current ready member width
    / predicted step_cost(|S|)

It estimates the credit rate of serving the currently ready packets, rather than
claiming an optimal scheduling theorem. Future sibling/stage work affects the
credit scenarios but is never inserted as a physical request. Every admitted
compound packet contains all its currently live members. A physical packet that
cannot fit delegates to the existing native required-work fallback when no
positive feasible selection exists; it is not dropped.

`mixed_control.py` searches positive marginal additions even when slots/KV are
free. Each addition is evaluated against interference with the other selected
actions. Candidate search is bounded using the native top-k pool convention;
required work outside the search pool remains registered. Physical KV/slots and
the native allocator remain authoritative.

## Continuity, background work and complete drain

Residents have no unconditional right to remain selected. A proposed replacement
is compared with incumbent credit rate, context restoration-work proxy and the
expected conditional credit lost during a bounded pause. Extra conditional
continuity protection expires after50 newly generated tokens. This uses the
existing service quantum, not a benchmark-fitted duration. Restore work uses the
existing256-prefill-token proxy; it is not measured PCIe/NVLink swap latency.
An expired or zero-credit resident does not retain credit protection.

Background decode can join positive work only on a learned non-increasing cost
plateau, with at least8 observations at both widths, no new prefill, and no lost
objective rate. A cold model or unused KV alone never authorizes background fill.
When no positive feasible action remains, required work uses the existing native
admission/execution path. A zero prediction is not a proof of root expiry and
never permits dropping a request. Full outputs and later stages must complete.
Pure DR's reversible deadline-proven fast drain remains unchanged.

Mixed entry, arrivals/new stages, completion/removal, prediction revision, real
credit-deadline changes and continuity-quantum expiry trigger replanning. The
50-tick periodic fallback remains. Actual preemption is limited by the existing
per-iteration swap limit and uses native swap/recompute. No swaps and swaps-in
are combined in the same candidate transition step. Pending work and prefixes
remain under the native scheduler/block manager.

## Pure DR, ToT and common runtime

The original v47 `native_chain.py`, `chain_forecast.py`, `chain_admission.py`,
`token_reward.py` and `assets/CHAIN_MODEL.json` remain byte-identical. Pure DR
continues to delegate to that loop; CPU tests compare actual selected requests,
leases, swap decisions and draining state. Foreground-only operation remains
native. The new online observation wrapper does not change either branch's
selection logic. Shared prediction repair is still a separate experimental
condition and may change results relative to old unrepaired GPU runs.

Native ToT remains implicit[3,1,3,1], cap256. The existing14 reserved calibration
graphs/model and the original295 DR FIT records are unchanged. Unknown model keys
require explicit transfer mode; the calibration asset is not proof of out-of-domain
prediction coverage or of real ToT answer quality.

All methods must use the identical `shared_framed_dynamic_prediction_v1` and
`length_framed_pickle_v1`, explicit model-specific QRF/vectorizer and sampling /
input / SLO / arrival / resource contracts. Relative to the parent shared module,
only the accepted method-label list adds v49; numerical and update semantics are
unchanged. A retains original budgets; Astar retains the qualified DR and native
ToT budget repairs. Do not pair v49 with historical unrepaired A/Astar results.

In a fresh serving process, before engine construction:

```python
from install_comparison_runtime import install
install(server, output_dir, method="v49",  # or "A", "Astar"
        model_key="llama8", port=explicit_local_prediction_port,
        predictor_protocol="length_framed_pickle_v1")
```

The predictor service uses the same package's `framed_predictor.py` with explicit
model, vectorizer and port. Candidate-only `install_v49.install(...,
model_key="llama8")` records `prediction_comparison_ready=false` and is intended
only for CPU/explicit legacy ablations. No installer constructs an engine or
provides GPU experiment authorization. Invalid protocol/output configuration is
rejected before mutating the serving runtime; duplicate installs require a fresh
process.

## Diagnostics and limits

`V49_RUNTIME.json` and compatible `MIXED_RUNTIME.json` record mode counts, compute
filtering, observed cost bins, shared prediction and pure-file hashes.
`V49_DECISIONS.jsonl` streams every plan, including actions/credit, selected width,
cost evidence, physical fit, admission reasons and continuity decisions. A bounded
buffer is flushed at graceful exit; a crash can lose an unflushed tail. The
in-memory cache keeps only64 plans and reports losses when no stream is installed.
There is no per-token decision serialization. Shared prediction has its own
identity/success/fallback event ledger.

CPU qualification checks native swap/recompute/resume, actual keyword API and
implicit ToT progress, all required finite work completion, cache/transport
semantics, parent pure decisions, real-deadline replans and non-oracle attributes.
CPU planner timings on synthetic states are not GPU goodput measurements. The
cost proxy can be inaccurate, greedy selection need not be optimal, and strict
background admission can extend drain time. Those are explicit risks for future
paired GPU validation, not reasons to change metrics or discard losing runs.

Future GPU evaluation must keep complete outputs/no drops, arrival error<=0.5s,
JCT report-only, fixed-hour settlement and full-replay metrics separate, and raw
request/input/token/TBT/resource/final-cleanup gates. No new GPU runs or paper
changes are part of this release.
