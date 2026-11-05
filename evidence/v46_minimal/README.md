# CPU qualification for the revised v45-based v46

Active qualification: `cpu_attempts/attempt04/CPU_QUALIFICATION.json`.
All36 tests passed:16 forecast/mixed-packing boundary tests,10 preserved v45
native-reference tests,8 mixed/native runtime tests, and2 fresh-install tests.
Tests used the existing frozen native Python environment with
`CUDA_VISIBLE_DEVICES=''`, no model execution and no GPU replay.

`attempt02` is preserved as a failed development qualification: native dispatch
uses the keyword `seq_group`, while the first adapter draft named the parameter
`group`. The keyword-compatible interface was fixed before publication.
`attempt03` passed the first33 tests; `attempt04` includes the final transition
and fresh-installer checks. Each receipt binds its own source and log hashes.

CPU fixtures exercise the native scheduler and block manager without running
model weights. They verify structural behavior, not chain attainment or TTFT/TBT
performance. The candidate still needs future GPU regression under a newly bound
protocol. No inference attempt, matrix or automatic continuation was started.
