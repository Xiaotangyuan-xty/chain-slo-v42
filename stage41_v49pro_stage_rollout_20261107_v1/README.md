# v49pro release package

This package replaces v49's mixed credit/work-times-width heuristic with a bounded stage-aware, paired terminal-credit rollout. The original v49 release remains intact in `../stage39_v49_compute_budget_20261107_v1/`.

Read [V49PRO.md](../V49PRO.md) for the actual v49 trajectory diagnosis, mechanism, comparison contract, qualification and limitations. Run `python3 ../verify_v49pro_freeze.py` for the read-only source/evidence audit.

Fresh serving-process integration:

```python
from install_comparison_runtime import install
install(server, output_dir, method="v49pro", model_key="llama8",
        port=prediction_port, predictor_protocol="length_framed_pickle_v1")
```

The baseline methods are `A` and `Astar`. All three must use the same shared predictor configuration and executor observer. Do not stack older installers. Candidate-only `install_v49pro.install` without an explicit shared prediction configuration is not comparison-ready.

Outputs: `V49PRO_RUNTIME.json`, compatible `MIXED_RUNTIME.json`, all-plan `V49PRO_DECISIONS.jsonl`, common `EXECUTOR_EVENTS.jsonl` and final `EXECUTOR_OBSERVATION.json`. Request input/arrivals/SLO, scoring, resource caps, protocol completion and final-gate requirements remain the experiment harness's responsibility; this package launches no replay or engine.

The historical `V47_ORIGIN.json` and `V48_ORIGIN.json` record inherited provenance, not an assertion that the mixed code remains unchanged from those releases. Current parent identity is `V49_ORIGIN.json`; current exact hashes are in `../FREEZE_V49PRO.json`.

95 CPU/native tests pass with CUDA invisible. Pure DR v47 core, training/ToT length assets and shared prediction operational semantics are preserved. **No GPU performance claim has been validated.** GPU1 was stopped by user request, and this package does not restart it.
