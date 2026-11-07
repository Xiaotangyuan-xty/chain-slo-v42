# v48 mixed continuity candidate

Parent: v47 commit `082e98c7c05431110bd6cd74234f909487b963ee`.
No GPU engine, replay, profiling, or performance experiment has been run for v48.
CPU qualification is structural/integration evidence, not evidence of higher goodput.

## Event-driven mixed scheduling

`MixedScheduler` still delegates pure DR scheduling to the unchanged v47 loop and
foreground-only work to the original native scheduler. Returning from mixed to
pure work forces a fresh plan and resets ownership leases before delegating. Mixed entry, added request
IDs (including new roots, stages and siblings), completion/abort, and prediction
revision trigger a plan. Arrivals in a single scheduler step are coalesced. A
50-tick periodic fallback remains. No arrival array, benchmark end time or future
request is consulted.

The mixed planner starts with resident work. Positive expected-credit work uses
free slots/KV before considering eviction. Under a declared admission-budget
conflict, a replacement must exceed the estimated interruption cost: displaced
credit density, context restore-work proxy, and conditional prefix value when
root slack is tight or a stage is close to completion. These costs are explicit
heuristics in decode-work units; they are not measured PCIe/GPU latency. Resident
horizon pressure can still force a safe plan change. The native allocator is
always authoritative, including swap/recompute and chunked prefill.

Remaining slots/KV backfill native-ranked required work, including zero-score
chains. Zero probability and missing model support are not expiry proofs or drop
permissions. With no positive-credit action or an oversized context, native
chunked-prefill admission remains the fallback. The planner does not enlarge
physical KV or bypass native allocation. All outputs and future stages must finish.

## Mixed token objective

Compound actions retain v47 conditional full-chain token reward and separate
completed prefix credit from remaining work. Foreground actions use predicted
remaining length bounded by the public sampling cap, with v47's cap-bounded
survival backoff. Latency reward counts only future eligible decode events under
the original cumulative token deadlines plus any still-unearned prompt credit;
a missed TTFT does not automatically zero all future decode credit. Deadline
reward remains completion-conditioned, including the uncredited generated prefix.
Past irrevocable latency credit is not counted again.

Scores use expected added credit per remaining work proxy. Native priority is
also retained as a backfill ranking. There is no fixed half-pool/half-slot split.
This heuristic is not an optimal batch GPU-time model. Preemption is bounded by
the existing max-swaps-per-iteration limit, and its effect on SLO must be measured.

## Pure DR preservation and native ToT

`native_chain.py`, `chain_forecast.py`, `chain_admission.py`, `token_reward.py` and
`assets/CHAIN_MODEL.json` are byte-identical to v47; see `V47_ORIGIN.json`.
The copied `joint_admission.py` is an unchanged regression reference, not the v48
mixed selection path. Pure DR tests compare actual decisions, not just hashes.
Shared runtime repair below is a separately versioned condition: unchanged pure
candidate code does not imply that repaired common prediction semantics reproduce
old GPU results.

Native implicit ToT omits `requests_per_stage`; known metadata supplies widths
[3,1,3,1] and future sampling cap256 before sequence-group registration. No
explicit DR DAG conversion is performed. Astar's existing DR and separately
qualified native-ToT budget repairs are included in the baseline integration.
Native A gets no Astar budget adapter.

Mixed/ToT forecasting can route to `TOT_CALIBRATION.json` for explicit `llama8`
or `qwen14`. Its14 unique graphs/model are from the previously frozen Stage34
calibration split; evaluation input graphs are excluded before building the asset.
The original DR FIT295 model is untouched. The source has fixed service-replay
lengths from public recorded/reconstructed ToT-style logs, not new GPU-generated
labels or natural answer-quality labels. A leave-one-graph-out check within the
reserved set is archived; it is not formal workload validation or 95% coverage
proof. Unsupported model keys require explicit transfer mode and are reported as
`unqualified_ToT_transfer`. Future models still need their own validation.

## Shared prediction protocol for fair comparisons

`prediction_transport.py` uses a 4-byte length frame, reads the complete body,
limits frames to 64 MiB and matches request/response IDs. QRF/vectorizer computation
and weights remain the caller's explicit assets. Error/EOF/protocol failures keep
the existing 1024 fallback and are logged as `fallback_error`, not valid predictions.
No LLM request is retried or dropped. Predictions remain asynchronous.

`shared_prediction.py` refreshes `predict_output_length` from live `output_len`
before native collective allocation and invalidates cached priorities for all
members of a collection when a prediction revision changes. The native reward
formula is unchanged. The transport and update semantics must be identical across
A/Astar/v48; per-process ports and method identities may differ. Both predictor
and serving client must use the framed protocol. Do not repair only the candidate
and reuse historical unrepaired baselines as if they were the same configuration.

In each fresh serving process, before engine construction:

```python
from install_comparison_runtime import install
receipt = install(
    server, output_dir, method="v48",  # "A" or "Astar" for the paired baselines
    model_key="llama8", port=chosen_local_prediction_port,
    predictor_protocol="length_framed_pickle_v1",
)
```

Use this instead of stacking the old v47/v46/budget installers. Put this package's
`programs/` first on PYTHONPATH. The predictor process uses the same package and
its explicit model-specific assets:

```sh
python3 programs/framed_predictor.py --model QRF_FILE --vectorizer VECTORIZER_FILE --port PORT
```

This CLI is a CPU prediction service, not an experiment launcher. Existing runtime
and dependencies are external. Llama experiments must use their verified 8B asset
binding; a historical global Qwen7B default is not a substitute for per-case model
configuration. A future runner must bind model/KV/SLO/arrival/sampling hashes and
record the shared protocol for every arm.

Candidate-only CPU/ablation entry `install_v48.install(server, output_dir,
model_key="llama8")` leaves legacy common prediction untouched and records
`prediction_comparison_ready=false`. Legacy transport is meaningful only if all
arms deliberately use the same declared legacy condition. No GPU launch or device
authorization is provided by either entry point.

## Observations and validation

`V48_RUNTIME.json` and compatible `MIXED_RUNTIME.json` record version, common
runtime protocol, pure-file hashes and counters. `V48_DECISIONS.jsonl` exports up
to 4096 plan events with triggers, per-action probability/work/credit, selections,
eviction-cost proxies and planning CPU time; dropped-event counts are explicit.
It logs plans, not every token. Shared prediction has separate event/error logs.
No hidden benchmark-specific IDs or thresholds are used.

Pure-Python tests run with ordinary Python. Actual scheduler/block-manager and
native ToT integration tests require the existing frozen Linux runtime with
CUDA_VISIBLE_DEVICES empty. They allocate no engine or model and assert no CUDA
context was initialized. Preserved development attempts and final source/log
hashes are in `evidence/v48/`. Parent v47/v46 freeze verification remains valid.

Future performance evaluation must preserve identical inputs/SLO/arrivals/resources
and shared protocol across methods, complete outputs and drain, actual native
weighted-token scoring, request identity and raw token/TBT audits, arrival error
<=0.5s, explicit failed attempts and JCT as report-only. Fixed-hour and full-replay
metrics stay separate. Faster post-window drain alone does not increase the
fixed-hour numerator. This release authorizes none of those GPU experiments.
