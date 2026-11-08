# v50pro: physical terminal-credit control with executable commitments

This version targets timely native weighted-token credit in a heterogeneous
latency/deadline/compound workload. It does not claim that a positive model
estimate guarantees real GPU goodput gains.

| Item | Native A | Astar | v50pro |
|---|---|---|---|
| Priority/selection | JITServe native SLO policy and sliding window | Same A policy, repaired cumulative stage budget | Own conditional terminal-token-credit proposals, exact physical no-change reference, proposal-specific event rollout |
| Compound objective | Native stage/graph allocation and native score | Same score with budget repair | Whole-root conditional prefix and future-stage donor credit, discrete empirical attainment |
| Displacement | Native execution/planning | Same native execution/planning | Bounded displacement compared to every modeled peer, integer transfer/recompute cost, root cooldown |
| Execution obligation | Native mechanism | Native mechanism | 50-output-token OR 1.5-second bounded quantum, non-extendible restoration dispatch, retain restored member until observed token service |
| No positive intervention / expired work | Native scheduler | Native scheduler | Own nonpreemptive credit/compute admission, online cost probes, aging and required-work drain; no call to native score/pack on mixed paths |
| Shared infrastructure | Original vLLM execution, graph state, author Llama8B QRF | Same, plus budget adapter | Same execution and public graph annotations; shared prediction repair and identical observer are infrastructure, not novelty claims |

Astar-equivalent stage-budget correction remains declared in the adapter. It is
not claimed as a new contribution. A retains its original budget. Native A and
Astar source policies are not edited by this version. Pure DR without any prior
foreground arrival delegates to the preserved v47 core; CPU execution traces
check this explicitly.

Important limits:

* The 1.5-second obligation is a restoration **dispatch** deadline subject to a
  scheduler/executor iteration, not a guarantee that the first restored output
  token appears by that exact instant. Pending restoration cannot be hidden by
  another plan; first observed token service is separately recorded.
* Stage/donor rollout has finite scenarios/events and unknown future arrivals.
  Empirical donor frequency is not advertised as a calibrated probability.
* The decode-step target is the existing TBT budget minus the declared 10ms planning and 10ms metadata reserves (80ms for a100ms TBT), not a
  change to the experimental SLO or a new validity gate. Sparse widths get
  bounded online exploration up to eight members before trusting extrapolation.
* Soft Python planning budgets cannot preempt arbitrary noninterruptible work.
  Outer-plan, logging and complete scheduler host time are recorded separately.
* No request is dropped when credit expires. Required work still completes.
* GPU development probes use frozen 5/10-minute arrival prefixes and full drain.
  They establish exercised paths and correctness, not an independent held-out
  benchmark result or proof of publication novelty/performance superiority.

The baseline score is patched to raise in actual native-ABI CPU tests for the
mixed and foreground-only paths. This validates that v50pro does not quietly
replace its mechanism with A/Astar's scoring or native pack.

Admission cues use whole-root terminal credit over all-stage service cost, and renewable next-50-token latency/prompt credit. A point forecast of zero reward is not a certificate of hard SLO expiry: uncertain unexpired work can use safe nonpreemptive spare capacity. Joint pair additions/exchanges precede single-action proposals, with a production 1-for-2 KV counterexample test.
