# v50pro — own terminal-token-credit control with executable commitments

This release keeps the candidate mechanism distinct from native JITServe A and
Astar. See [mechanism and baseline comparison](stage46_v50pro_executable_commitments_20261108_v1/MECHANISM_V50PRO.md).

The mixed path uses a physical no-change reference, bounded proposal-specific
member/stage event simulation, discrete empirical donor credit, executable
service/restoration obligations, round-aware switching cost and its own compute
admission/required-work controller. It does not call A/Astar's scoring or native
pack on the mixed/foreground paths. Actual native-ABI tests replace those calls
with exceptions to verify that separation. Astar-equivalent budget correction,
framed shared QRF repair, public native-ToT metadata and the identical executor
observer remain declared common infrastructure, not new-contribution claims.

Validation records are in `evidence/v50pro/` and bound by `FREEZE_V50PRO.json`.
The complete CPU/native suite contains 119 tests. Eight completed GPU0 component
probes must pass; one earlier liveness failure and a superseded performance regression is explicitly preserved with its
registered-process stop and cleanup evidence. Development prefixes and complete
finite drain test exercised paths/correctness. They do not establish full-hour
performance superiority or independent-run confidence intervals.

Run `python3 verify_v50pro_freeze.py` for a read-only source/evidence audit. It
also checks all previous version freezes, unchanged pure-DR core/assets and the
common prediction/observer contract. It constructs no model/engine/device.

GPU harness preparation and launch are separate. Use the published `harness/`
only with an explicit authorized protocol, matching frozen native JITServe
source/model/QRF/workload assets and fresh experiment directory. Each original
run gate checks identity, generation, arrival error<=0.5s, raw token/TBT coverage,
actual-text native weighted-token credit and cleanup. JCT remains report-only.
No request is dropped or retried when credit expires.

The current formal workload is the fixed one-hour 1800-root mixed stream
(840 latency + 840 deadline + 60 DR + 60 implicit ToT), average0.5 roots/s,
Llama-3.1-8B-Instruct,19.875GiB KV. The hour uses settlement-time credit/3600;
full replay and drain are separately reported. KV stress on GPU1 uses the
preserved v47,A,Astar implementations at1/2/4/8/16GiB and is a separate study.

Soft planning limits, executor-host timing and donor projections retain the
limitations stated in the mechanism document. No claim of reaching1000/1100
weighted-token goodput/s is made before the complete paired formal results.
