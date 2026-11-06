# v47 CPU qualification

Active receipt: `CPU_QUALIFICATION.json`, matching `cpu_attempts/attempt04/`.
All **51 tests passed**, with `CUDA_VISIBLE_DEVICES=''`, on the existing frozen
Linux native runtime. No model engine or GPU replay was executed.

- 16 preserved work-boundary and mixed-packing tests.
- 9 token-reward, deadline, text-count and cache counterexamples.
- 10 native block-manager/reference tests (including swap/recompute prefixes).
- 8 mixed/foreground coexistence and ownership tests.
- 7 expired-root drain, adaptive execution, stage-gap, recovery and KV tests.
- 1 fresh-interpreter public installer and compatible runtime-receipt test.

Each receipt binds its source snapshot and log hashes. `source.tar.gz` preserves
the uploaded CPU package, including the old pure-Python forecast reference.
The repository's existing runtime/dependencies are external; this is not a
self-contained inference installation or a throughput benchmark.

Development history is retained:

- attempt01: failed first-prefill batch assertion and incorrect pre-installer
  test import order. The former was corrected to require actual concurrent
  execution over subsequent native prefill/decode steps; the latter now uses the
  documented fresh installer first. This attempt also accidentally discovered
  imported reference TestCases twice; final counts exclude that duplication.
- attempt02: 50 tests passed.
- attempt03: 51 passed, adding completed-current-sibling reward accounting and
  clarifying the pure-chain delegation test name.
- attempt04: 51 passed on the published source, including both `V47_RUNTIME.json`
  and compatible `MIXED_RUNTIME.json`. Installer metadata no longer risks implying
  a future actual run did not execute a model.

The tests establish CPU structure, state transitions and resource cleanup.
They do not establish v47 goodput, chain attainment or superiority over Astar.
No experiment controller, frozen Stage29 case, GPU authorization or paper was
changed. Root v46 documentation remains byte-frozen and separately verified.
