# v49pro evidence

Active native CPU qualification: attempt08, 95 tests, CUDA invisible, no CUDA context, no model/GPU execution; actual JITServe/vLLM imports plus controlled CPU fixtures. Exact source hashes are in CPU_QUALIFICATION.json; each full native test log is bound by SHA256. The full original result is also preserved in cpu_attempts/attempt08/CPU_QUALIFICATION_FULL.json. Earlier CPU prototypes and diagnostics remain under the local work/v49pro_build directory and the remote CPU-candidate attempt directories; no GPU experiment was used to select this revision.

V49_RESULT_AND_RUNTIME.json is the completed v49 case's gate/metrics/runtime receipt and first decision. V49_TRACE_DIAGNOSIS.json was calculated by streaming all 6336 decisions and the complete client call record; it includes source byte counts/SHA256, underfill examples, service-gap/TTFT statistics and phase aggregates. These describe the old v49, not GPU measurements of v49pro.

GPU1_USER_STOP.json records that v49 had completed; Astar was interrupted and A did not run. Cleanup checked registered start ticks, process exits, GPU1 memory/compute state and lock release. No complete A/Astar comparison is claimed for that queue.

CPU_BUDGET_SYNTHESIS.json covers 15 synthetic CPU plans each with 64 and 256 candidate actions. Both cases used the 6ms soft guard and all runs returned to native execution on budget exhaustion. This establishes bounded materialization/overhead on these inputs, not scheduling quality or GPU throughput. The test setup and measurements are published; no machine-speed threshold is asserted as a portable real-time guarantee.

Original v49/v48/v47/v46 evidence, assets and code remain preserved and independently verifiable. The raw large remote v49 logs were read, hashed and retained in place; prompts/token streams are not duplicated in this release.
