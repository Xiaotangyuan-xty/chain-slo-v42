# v50 release package

Read [V50.md](../V50.md) and [CODE_REVIEW_V50.md](../CODE_REVIEW_V50.md). Current exact hashes are in `../FREEZE_V50.json`; verify with `python3 ../verify_v50_freeze.py`.

Production entry: `programs/install_comparison_runtime.py`, method `v50` / `A` / `Astar`, fresh process and explicit model_key/prediction port/protocol. The production mixed planner is `window_control.py` with `window_projection.py` and immutable `planner_config.py`. Older `mixed_control.py`/event-rollout helpers are retained only for inherited mathematical regression and native credit helpers; they are not the installed planner.

The candidate uses explicitly declared corrected stage budgets; native A retains original budgeting and Astar retains its existing correction. Common predictor operational code and executor observer are the same across methods. Pure DR core and frozen length banks are unchanged. Native ToT stays implicit.

`harness/` contains the actual reviewed prepare, start, runner, controller, servers, client, native graph observer and final native metric code. The root is explicit, never inferred from case depth. CPU preview directories are non-executable by all GPU entry points. Experiment code copies these bytes into its qualified per-case programs before launch.

`assets/TRACE_PUBLIC_SNAPSHOTS.json` is a CPU qualification fixture, not a runtime bank or training target file. Production code never loads it. Its current public-state/observed-prefix data qualify timing and action-path execution only; they are not GPU-performance predictions.

Native CPU qualification and a six-case deployment preview pass without CUDA/model execution. Actual v50 goodput1000/1100 and superiority over A/Astar are goals to be tested, not release claims. The experiment must finish every request and full drain with the unchanged native scoring and arrival tolerance.

Historical ORIGIN files record ancestry; they do not assert that mixed algorithms are unchanged from those parents. Current parent is `V49PRO_ORIGIN.json`; original versions and failures remain in place.
