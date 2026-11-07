# v48 CPU qualification

Active receipt: `CPU_QUALIFICATION.json`, matching `cpu_attempts/attempt05/`.
All **67 tests passed** on the existing frozen Linux native runtime with
`CUDA_VISIBLE_DEVICES=''`. Each test process also asserts no CUDA context was
initialized. No model engine, weights execution or GPU performance replay ran.

- 16 preserved forecast-boundary/legacy mixed-reference tests.
- 9 preserved token-credit and cache tests.
- 10 native block-manager/pure-reference tests, including swap/recompute prefixes.
- 7 preserved v47 expired-drain/lifecycle tests exercised through v48 delegation.
- 8 mixed marginal-credit/admission/domain tests.
- 4 framed transport and shared-client tests, including fragmented messages >4KiB.
- 9 actual native mixed runtime/event/ownership/resource tests.
- 1 actual native shared-prediction snapshot/cache test.
- 2 fresh-process A/Astar common-runtime tests on the actual implicit ToT state machine.
- 1 fresh v48 installer/native-ToT domain registration and duplicate-install test.

`TOT_CALIBRATION_CHECK.json` is an additional leave-one-graph-out diagnostic on
14 reserved calibration graphs/model. It compares length-work proxies, not
formal evaluation outcomes, natural response quality or GPU throughput. Runtime
assets contain no labels from Stage35 performance runs. Source pool hashes and
selected calibration input-graph fingerprints are retained in the asset.

Development evidence is preserved rather than replaced:

- attempt01: pure/transport tests passed; native integration exposed missing
  `seq_group=` keyword compatibility and a test import-order cycle.
- attempt02: those issues were fixed; an inherited fast-drain exit regression
  exposed a positive model stub on an expired chain. Hard root expiry now zeros
  candidate credit regardless of model score, preserving foreground progress.
- attempt03: all64 tests passed.
- attempt04: all66 passed after adding actual A/Astar integration qualification.
- attempt05: all67 passed, including the explicit mixed-to-pure ownership handoff.

Every attempt binds its exact source archive, per-file hashes and test logs.
Parent v47/v46 snapshots and verifiers remain unchanged. The v48 verifier checks
published source against the final CPU receipt and the five unchanged pure files.
The shared prediction repair is a new comparison condition for every method,
not a retroactive change to old results. No performance superiority is claimed.
