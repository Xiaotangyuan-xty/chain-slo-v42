import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from fake_backend import FakeBackend
from harness_executor import HarnessAsyncExecutor
from vllm.sampling_params import SamplingParams


class Data:
    def get_token_ids(self):
        return [1, 2, 3, 4]

    def get_prompt_len(self):
        return 4

    def get_num_computed_tokens(self):
        return 4


class Metadata:
    request_id = "zero-work"
    seq_data = {0: Data()}
    token_chunk_size = 0
    is_prompt = True
    do_sample = False
    computed_block_nums = []
    block_tables = {}
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1)


class Request:
    seq_group_metadata_list = [Metadata()]
    finished_requests_ids = []
    blocks_to_copy = []
    blocks_to_swap_out = []
    blocks_to_swap_in = []


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only gate")
    output = root / "stage6_mechanism_repair/controlled" / attempt / "zero_guard"
    output.mkdir(parents=True, exist_ok=False)
    executor = HarnessAsyncExecutor.__new__(HarnessAsyncExecutor)
    executor.backend = FakeBackend()
    executor.backend.reset(8192, 8192)
    executor.block_owners = {}
    executor.computed_by_request = {}
    executor.events = []
    executor.finalized = False
    executor.cache_block_bytes = 0
    executor.output_dir = output
    executor.cache_config = SimpleNamespace(num_gpu_blocks=512, num_cpu_blocks=0)
    executor.event_handle = (output / "executor_events.jsonl").open("x")
    result = executor.execute_model(Request())
    executor.finalize()
    backend = executor.backend.summary()
    events = [json.loads(line) for line in (output / "executor_events.jsonl").read_text().splitlines()]
    skipped = [event for event in events if event["kind"] == "zero_token_prefill_skipped"]
    passed = (len(result) == 1 and len(result[0].outputs) == 1
              and not result[0].outputs[0].samples and skipped
              and backend["actual_forward_query_tokens"] == 0)
    receipt = {
        "schema": "zero-prefill-guard-gate/v1",
        "gate": "PASS" if passed else "FAIL",
        "skipped": skipped,
        "output_samples": result[0].outputs[0].samples,
        "actual_forward_query_tokens": backend["actual_forward_query_tokens"],
        "gpu_used": False,
    }
    with (root / "stage6_mechanism_repair/receipts" / f"{attempt}_ZERO_PREFILL_GATE.json").open("x") as handle:
        handle.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
