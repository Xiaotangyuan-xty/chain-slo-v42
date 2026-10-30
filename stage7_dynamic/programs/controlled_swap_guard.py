import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from controlled_zero_guard import Metadata, Request
from fake_backend import FakeBackend
from harness_executor import HarnessAsyncExecutor


def executor(output):
    output.mkdir(parents=True, exist_ok=False)
    value = HarnessAsyncExecutor.__new__(HarnessAsyncExecutor)
    value.backend = FakeBackend()
    value.backend.reset(8192, 8192)
    value.block_owners = {}
    value.computed_by_request = {}
    value.events = []
    value.finalized = False
    value.cache_block_bytes = 0
    value.output_dir = output
    value.cache_config = SimpleNamespace(num_gpu_blocks=512, num_cpu_blocks=0)
    value.event_handle = (output / "executor_events.jsonl").open("x")
    return value


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only gate")
    base = root / "stage6_mechanism_repair/controlled" / attempt / "swap_guard"
    stale = executor(base / "stale")
    request = Request()
    request.blocks_to_swap_out = [(492, 127)]
    stale.execute_model(request)
    stale.finalize()
    stale_events = [json.loads(line) for line in (base / "stale/executor_events.jsonl").read_text().splitlines()]
    stale_pass = any(event["kind"] == "stale_swap_out_ignored" for event in stale_events)

    live = executor(base / "live")
    live_metadata = Metadata()
    live_metadata.block_tables = {0: [492]}
    live_request = Request()
    live_request.seq_group_metadata_list = [live_metadata]
    live_request.blocks_to_swap_out = [(492, 127)]
    live.execute_model(live_request)
    live.finalize()
    live_events = [json.loads(line) for line in (base / "live/executor_events.jsonl").read_text().splitlines()]
    live_audited = any(event["kind"] == "stale_swap_out_ignored"
                       and event["active_in_current_metadata"] for event in live_events)
    passed = stale_pass and live_audited
    result = {
        "schema": "swap-guard-gate/v1",
        "gate": "PASS" if passed else "FAIL",
        "stale_ignored": stale_pass,
        "live_missing_owner_audited": live_audited,
        "gpu_used": False,
    }
    with (root / "stage6_mechanism_repair/receipts" / f"{attempt}_SWAP_GUARD_GATE.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
