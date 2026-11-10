# v47pro implementation

Release entry: `../README_V47PRO.md`. Design and inherited infrastructure are
specified in `MECHANISM.md`. The code is derived from frozen Stage30 v47.

The fresh-process integration entry is `programs/install_comparison_runtime.py`.
Use `method='v47pro'`, an existing fresh output directory, and an explicit shared
framed predictor endpoint. Model/engine construction belongs to the separately
qualified serving harness. Calling an installer alone constructs no engine or
GPU context. This snapshot depends on the frozen native JITServe/vLLM/runtime
and external model/QRF assets; it is not a self-contained model installation.

`programs/test_service_forecast.py` runs without native/GPU dependencies.
`test_native_runtime.py` exercises the actual frozen native scheduler and KV
allocator on CPU. `test_install.py` checks the real ABI and implicit ToT bridge.
Their 48 tests must pass in the qualified runtime before GPU development tests.
No unit test is a performance claim.

`tools/make_development_packet.py` prepares an explicitly labeled unpublished
GPU0 prefix packet; it does not start work. Preparation, starting, raw auditing
and publication remain distinct operations. The development harness permits
one candidate prefix per fresh directory, at most 600s, no retries and no GPU1/2.
It uses the fixed full-mixed stream's exact requested prefix, not a resampled
workload. Every admitted arrival must finish its complete outputs and drain.
`tools/independent_raw_audit.py` is read-only and refuses a partial/failed queue.

Do not use a stored historical authorization as new permission to launch. This
release's user explicitly requested no GPU experiment after GitHub publication.
