# v47: conditional token reward, expired-chain drain and bounded caches

v47 is an independent candidate based on the minimal v46 at commit
`16c86bb43b1b11f01699ce57cbe5609d69fae31d`. It preserves that release and the
Stage29 runs. CPU qualification is provided; no v47 model execution or GPU
performance experiment has been run. Higher goodput is a hypothesis, not a result.

## Token reward

The score is now the donor mean of full-chain weighted token credit **conditioned
on the same donor finishing on time**, divided by mean remaining work:

`E[R_chain * 1(predicted completion <= slack)] / E[remaining work]`.

`R_chain` includes the completed prefix, emitted current tokens, and predicted
current/future input + 8*output. Completed work earns credit only after the whole
chain meets its existing deadline; it is not charged again as remaining work.
This is not `q * unrelated_average_reward`, nor reward for remaining tokens alone.
The v46 donor support, missing-sibling, per-call cap and full-depth boundaries are
retained, along with the 1.2 incumbent preference and generated-token leases.

Known prompt and completed output text are counted once using the serving model
 tokenizer with `add_special_tokens=False`. The tokenizer is obtained from the
live server graph context. Future inputs/output lengths remain FIT-donor
estimates; they are not exact future tokenizer counts. If a tokenizer is absent
in another integration, public engine-length estimates are used and counted in
`engine_token_fallbacks`; check this counter before interpreting experiments.
No evaluation graph, unsubmitted test prompt, actual future output, fixed trace
target length, Astar adapter, or new training asset is consulted.

## Reversible expired-only fast drain

The root registry retains deadlines across stage barriers. A lazy deadline heap
answers whether any known unfinished root remains unexpired; model zero score
and unknown support are not expiry proofs. Only when there is no such root and
actual work is queued does the pure-chain scheduler bypass forecasting, leases
and value-based planning. Requests and their remaining stages still execute.

The mode is reversible, including before all arrivals have ended: the next live
root immediately exits it, forces a fresh plan and restores the prior sequence
limit. Mixed/foreground entry also restores the limit. There is no assumption
that the benchmark's last-arrival time is known, and no permanent latch that
would strand future requests. Restoring a logical limit does not instantly evict
resident sequences; physical native scheduling/preemption remains responsible
for that transition.

Fast drain uses native chunked prefill, not a complete switch to original
A/GMAX. Every membership change or 50 ticks, it considers running residency and
pending admission in native swapped-before-waiting order, charging current
context plus the existing 50-token horizon. The sequence limit may grow up to
`max(initial_limit, min(128, max_num_batched_tokens))` if the existing KV budget
allows it. It does not enlarge KV, the batch-token budget, generation caps or
GPU allocation; the native block manager remains authoritative. The horizon is
not a guarantee against every subsequent physical preemption.

A new root/foreground can still encounter resident expired work during the
transition. The CPU tests prove dispatch restoration and execution opportunities,
not a universal foreground latency or GPU throughput guarantee.

## Caches and observations

- Existing neighbor reuse additionally binds declared depth and widths.
- A bounded 8192-entry LRU stores immutable donor future work/critical/reward
  suffixes, keyed by donor, stage, widths and public future cap.
- A bounded 2048-entry text-count cache binds tokenizer identity and text hash.
- Physical block charges are evaluated once per member per selection pass.
- Probability, slack, progress, leases and complete actions are not cached across
  changing runtime states. Newly completed prefixes and new members remain live
  inputs to each forecast.
- Expired estimates return zero credit before donor search; unsupported live
  structure is separately reported as unknown.
- Aggregate forecast/cache/drain counters are written to `V47_RUNTIME.json`;
  detailed per-tick decision construction is absent, matching v46's minimal
  runtime observation scope.

The separate legacy native QRF/graph path has deliberately not been removed or
repaired in this release. Its known TCP framing/fallback and Astar snapshot race
must not be hidden by a future performance claim. Mixed foreground still depends
on native policy behavior, and that dependency needs separate auditing.

## Integration

Use a fresh process with this directory's `programs/` first on `PYTHONPATH`.
Call the installer before directly importing `native_chain` or `mixed_joint`:

```python
from install_v47 import install
receipt = install(server, output_dir)
```

The expected mechanism version is
`v47_token_reward_deadline_drain_cached_forecast`; the mixed variant is
`v47_joint_token_reward_v1`. Do not additionally install v46, `low_observation`,
or `baseline_budget`. No standalone launcher or experiment authorization is
provided here. A future case needs its own protocol, hashes and device approval.

## Validation and limits

Run the pure-Python boundary and reward/cache tests with standard Python. Native
scheduler/block-manager tests require the repository's existing frozen Linux
vLLM runtime and dependencies, with `CUDA_VISIBLE_DEVICES=''`. See
`evidence/v47/CPU_QUALIFICATION.json` and its per-file logs for the exact source
binding, counts and outcomes. Historical development failures are preserved.
CPU fixtures allocate no model weights and exercise no CUDA execution.

The future GPU evaluation must retain complete200-chain/1197-call replay, actual
text-retokenized JITServe goodput, complete drain, explicit retry/failure records,
arrival error <=0.5 seconds, the authorized KV/GPU/model/arrival configuration,
and JCT as report-only. These are evaluation requirements, not authorization to
start a run. Token weighting can trade chain count against token credit; both
must be reported. This release does not claim to beat Astar from CPU results.
