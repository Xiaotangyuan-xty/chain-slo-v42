# v47max: isolated mixed coordination over v47pro3

v47max starts at published v47pro3 commit
`f8c5d84048536861a017adaa53648b6b753487e6`. The implementation is in
`stage73_v47max_mixed_continuity_20261110_v1/`.

All parent programs and assets are preserved byte for byte. New code lives in
`programs/mixed_max.py` and `programs/install_v47max.py`. Pure chain-only queues
use the same `schedule_pure` entry, frozen policy, allocation, preemption,
expired drain and shared helpers as v47pro3. Parent source hashes are bound in
`PURE_PARENT_MANIFEST.json`; CPU golden tests compare selected requests, leases,
queue order, output prefixes, swap counts and KV release against v47pro3.

Only the foreground/joint path changes:

- Keep resident decode groups running across plan changes. Proactive replacement
  requires blocked, positive-value foreground and an expired victim; native
  physical preemption remains responsible for safety.
- Rank new foreground by future reward per service time, excluding already
  settled latency reward. Share free slots and KV by marginal value, without
  the old fixed half-pool admission target.
- Revalue a proposed chain wave with actual residents/foreground in the batch.
  Whole-root reward, training records and future-stage public structure remain
  the inherited model; no oracle outputs or future prompts are introduced.
- Cache admission forecasts by root/stage/prediction/progress, preserve resident
  state across foreground/joint changes, and invalidate mixed caches on policy
  reconstruction. Pure dispatch never invokes these new helpers.
- Adapt mixed admission within 8..64 slots using matching observed decode bins,
  retaining existing residents while lowering the admission target. This is a
  policy choice within the benchmark's explicitly shared runtime ceiling128.

`install_v47max.install` is the entry point. Comparison infrastructure retains
`v47pro` as its transport-family label and records **release=v47max**, the new
installer, scheduler class and `v47max_mixed_continuity_v1` runtime version.
Historical v47pro/v47pro3 files and branches are unchanged.

## Validation and limits

`evidence/v47max/CPU_QUALIFICATION.json` records the exact tested candidate
hashes, native ToT/DR tracking and Astar helper coverage, actual inherited-FD
HTTP serving, and concurrency-contract checks. The 200-root pure golden tests
use the actual frozen1197-call input shapes at1024 and11357 blocks, pinned
predictions and short simulated outputs. They establish deterministic scheduling
agreement in those fixtures, **not GPU goodput equality**. Native resource and
mixed decode-continuity tests are also included. GPU performance superiority is
not claimed by this commit.

The authorized comparison is five fresh mixed600 workloads on **GPU0** with
Llama-3.1-8B, fixed10176 blocks=19.875GiB, nominal1root/s, and methods
v47max/A/Astar. Initial concurrency32 and common runtime ceiling128 are explicit;
A/Astar retain native scoring below that shared ceiling. ToT uses the native
implicit `[3,1,3,1]` path with256 cap; other applications use1024. Astar's DR and
implicit-ToT budget corrections remain separate shared adapters.

Finite-window primary accounting is actual-text native settlement credit/600;
full replay/drain, attainment and report-only JCT are separate. These are not
hour-long results. Arrival gate remains0.5s. No request is dropped, shortened or
retried to improve results. The queue preserves a failed case and continues
independent cases after cleanup. Reserved listening FDs remove the previous
check-close-bind race. GPU1's existing pure-ablation queue is untouched; GPU2 is
outside scope. Publishing source itself does not launch a GPU experiment.
