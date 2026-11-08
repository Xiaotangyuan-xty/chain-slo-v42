# v47pro: event-driven fixes to the v47 scheduler

The scheduling implementation is rebuilt from frozen v47 commit
`082e98c7c05431110bd6cd74234f909487b963ee`, in
`stage48_v47pro_event_service_20261108_v1/`. It does not import the v48/v49/v50/
v50pro candidate schedulers or their counterfactual search, commitments, or
admission controllers. The Git parent retains those historical releases without
changing their files; Git ancestry is distinct from the scheduling-code origin.

v47pro keeps v47's empirical whole-root terminal-token reward, training-only DR
neighbors, generated-token protection and foreground/chain coexistence target.
It fixes event/deadline replanning, selected-wave completion estimates,
demand-driven eviction, physical preemption invalidation, coarse online service
costs and mode/config isolation. Native foreground-only selection and the
foreground reward formula are inherited from v47; they are not claimed as a new
candidate contribution.

See the stage `MECHANISM.md` for exact behavior, inherited infrastructure and
limits. CPU/native and GPU development evidence is in `evidence/v47pro/`.
Development prefixes are not full-hour measurements and do not establish
superiority or stability over v47/A/Astar. Failed/superseded evidence is retained.

The fixed-hour main metric remains native settlement-time weighted credit/3600.
Complete replay, drain and JCT are reported separately. Arrival error must be
<=0.5 seconds; JCT is report-only. No expired request is dropped, truncated or
retried to improve these metrics.

Publication does not start any GPU work. The bundled harness only permits an
explicitly authorized unpublished development snapshot on GPU0 with a prefix of
at most 600 seconds. It is not a full-hour launch authorization. GPU1's existing
v47 capacity queue is separate, and GPU2 is outside this work's scope.
