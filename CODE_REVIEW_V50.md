# v50 pre-submission self-review

Scope: production policy/forecast/cost path, fresh-process installers, both baseline branches, published prepare/start/runner/controller, warmup, fixed-hour/native scoring, cleanup and final gates. This is the implementing agent's review plus executable checks, not an independent human/subagent review. No claim that testing proves an absence of all defects.

| Finding | Resolution before submission | Evidence |
|---|---|---|
| Full event continuation consumes budget before a candidate can be compared | Cached stage arrays, analytic native foreground credit, indexed cost cells; bounded set proposals | Native CPU trace replay and explicit score/counterexample tests |
| Exhausting later search discards an already completed gain | Keep last complete best; charge reserve; reject incomplete/unsafe overrun | Manual-clock anytime regression |
| A large foreground reference trips a12-action model-pool guard | Account for all resident/reference foreground losses; cap challengers/compound materialization instead | 32-resident integrated test and actual snapshots |
| Hard sparse scenario steps provide almost no marginal signal | Declared weighted-CDF interpolation with bounded support, conditional prefix credit | Width1→32/32→1, CDF/cumulative-credit tests |
| Switching margin depends on unrelated constant credit | Scale hysteresis only by changed credit at risk | Constant-credit invariance regression |
| Multiple individually feasible evictions collectively exceed CPU swap space | Reserve blocks in the same native-priority eviction order; price overflow replay, including foreground | Joint-budget projection test; native multi-member swap/recompute resume with both pools released |
| Fallback starts from A's known incomplete stage budget | Explicit candidate corrected-budget adapter with Astar-equivalent semantics; A and Astar retain their own declarations | Actual native DR/implicit-ToT install/routing tests |
| Candidate's pure ToT goals did not update chain priorities | Populate priorities from its stage-aware goal; retain all-DR pure-core delegation | Native ToT registration/goals and pure-DR decision equivalence |
| Baseline snapshot assumes cost_model exists | Optional candidate cost model; require observer events for every method, cost samples only for v50 | Extracted actual server snapshot function and all-three warmup branches |
| New per-GPU path breaks native metric root lookup | Required explicit root, protocol binding and native source SHA; both mixed/pure calls pass it | Full old nested-case metric recomputation, no writes; wrong-root rejection |
| Matrix/startup integration can silently change method/order/device/ports | One shared pure matrix contract, published actual harness; guards and six-case CPU preparation | Matrix negative cases and deployment preview |
| CPU preview could accidentally be launched | Marker checked by start/runner and non-formal protocol rejected by controller | Guard tests; preview has no launch receipt or attempts |

Additional checks: all deployment modules parse; no parent-depth root assumptions remain; runtime files do not access true future target/stop attributes or the benchmark end time; source/model/input hashes are bound before launch; common prediction differs from the parent only by accepting labelv50; executor observer bytes are unchanged and no CUDA synchronization is added. All methods run in new processes. The repository's older frozen releases are verified unchanged.

Final qualification artifacts record exact test counts, logs, source SHA, open-loop replay results and the six-case CPU-only preview. The GPU stage remains necessary to validate real-model integration and performance. The pipeline must preserve failed gates and must not retry, alter frozen code, relax arrival tolerance or stop/reselect runs based on goodput.
