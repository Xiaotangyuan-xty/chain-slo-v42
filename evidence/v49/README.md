# v49 CPU qualification and trace provenance

Final native CPU qualification: **75 tests PASS**, active attempt08. All commands
used CUDA_VISIBLE_DEVICES empty and CPU28/30/84/86; each test process checked
that no CUDA context was initialized. No LLM engine, GPU generation, GPU profiling
or performance replay was started. External frozen JITServe/vLLM dependencies and
model tokenizers are reused, not modified.

The tests cover compute-sensitive selection despite free KV, interference-induced
expiry, batching benefits, bounded live retention, no unconditional zero-value
background admission, online sample filters, original-forecaster equivalence,
real native swap/recompute/resume and final KV release, full finite required-work
completion, selected-set audit, immediate arrivals/stages/prediction/deadline
replans, pure v47 decision regression, native-ToT keyword API and metadata,
identical shared prediction semantics for A/Astar/v49, fragmentation/errors and
installer validation. Parent v48/v47/v46 freeze checks remain separate.

Preserved development attempts:

- attempt01: CPU failures exposed expired-root curve handling with a legacy stub
  and test fixtures incorrectly finishing a still-waiting foreground request.
  Expired curves now short-circuit before donor/model access; fixtures finish only
  active requests or use explicit cancellation where the scenario is removal.
- attempt02: 70 tests PASS.
- attempt03: 73 tests PASS, adding full mixed completion, real deadline crossing
  and common v49 native-ToT integration through the real keyword API.
- attempt04: 73 tests PASS, with early invalid-protocol/output validation.
- attempt05: 73 tests PASS, including corrected module description.
- attempt06: 74 tests PASS, adding exit-buffer persistence of all5000 streamed plan events.
- attempt07: 74 tests PASS after trailing-blank-line cleanup.
- attempt08: 75 tests PASS on final source; preserve the observed fixed-cost component when extrapolating batch cost and verify learning can pass the largest seen batch.

Every attempt keeps its source archive, logs and source hashes; failed CPU
attempts are not hidden or treated as GPU trials. CPU tests are mechanism and
integration evidence, not a forecast that v49 will outperform A/Astar.

TRACE_DIAGNOSIS.json contains aggregate facts from the completed Stage37 GPU1
runs and retained v48 decisions. It contains no raw user prompts or returned text.
These existing runs motivated the design; predictor weights/calibration banks
are unchanged. LOCAL_CPU_PLANNER_COST.json is a synthetic local CPU microbenchmark,
not a GPU cost curve, held-out experiment or goodput estimate.
