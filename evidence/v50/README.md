# v50 qualification evidence

Active CPU qualification: attempt05. 121 tests pass with CUDA invisible and CUDA initialization asserted false. Exact .py/.json source hashes, full native test logs and the original result object are preserved. This includes inherited mathematical/pure-core regressions, actual JITServe/vLLM integration, v50 policy and shared baseline branches, actual nested native metrics, matrix and preview guards.

PUBLIC_TRACE_REPLAY.json records an open-loop test on 64 deterministically spaced public-state snapshots from the two complete v49pro runs. All membership reconstructions match; only past observed prefixes/completions, current QRF outputs and past executor samples reach the planner. Ordering is a corrected-uniform-budget proxy. This qualifies CPU timing and actual proposal paths, not counterfactual GPU throughput or stability. Runtime code does not load the snapshot fixture.

DEPLOYMENT_DRY_RUN.json and DEPLOYMENT_PREVIEW.log record the actual published preparer producing six CPU-only cases, two GPUs x v50/A/Astar. Input/model/stop/native-ToT/shared prediction checks pass; hashes of publication and deployed programs match, 12 ports are unique, all attempts are empty and no launch receipt exists. Preview start/runner/controller reject GPU execution.

The native metric smoke test reads the complete old Stage42 GPU0 data without writing its files, explicitly uses the correct root and reconstructs the accepted3174614/3215763 numerators. It is not presented as v50 experimental data. Older CPU prototypes, raw traces and original failed gates remain retained in their original work/CPU-candidate locations.

No model/GPU experiment is part of these qualification results. Target hourly goodput1000/1100 remains unmeasured at publication and must not be converted into a gate for selecting or discarding runs.
