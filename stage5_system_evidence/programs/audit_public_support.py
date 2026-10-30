import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

from trace.trace_deepresearch import DeepResearchTrace
from v2_predictor import Predictor


ASSET_ROOT = Path("/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_finite_h4_confirm_gpu01_20261018T114337Z_resume_v1")
EXPECTED_CHAIN0_PROMPTS = [375, 284, 2933, 214, 4662, 4510]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    attempt = sys.argv[2]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This audit is CPU-only")
    trace_path = root / "repo/traces/deepresearch_filter.jsonl"
    predictor_path = ASSET_ROOT / "PREDICTOR.json"
    prepare = json.loads((root / "PREPARE.json").read_text())
    pred = Predictor.from_dict(json.loads(predictor_path.read_text()))
    tokenizer = AutoTokenizer.from_pretrained(prepare["model"], local_files_only=True)
    chains = DeepResearchTrace.load_trace(str(trace_path))
    rows = []
    for chain in chains:
        stages = []
        for ordinal, stage in enumerate(chain.stages):
            member_prompts = [len(tokenizer.encode(request.prompt)) for request in stage.requests]
            prompt = sum(member_prompts)
            requested = sum(request.output_tokens for request in stage.requests)
            supported_positions = 0
            initial_supported = True
            try:
                pred.ready_base(prompt, ordinal, 0)
            except ValueError:
                initial_supported = False
            if initial_supported:
                for emitted in range(requested):
                    try:
                        pred.ready_base(prompt, ordinal, emitted)
                        supported_positions += 1
                    except ValueError:
                        break
            stages.append({
                "ordinal": ordinal,
                "members": len(stage.requests),
                "member_prompt_tokens": member_prompts,
                "compound_prompt_tokens": prompt,
                "requested_output_tokens": requested,
                "initial_supported": initial_supported,
                "supported_emitted_positions": supported_positions,
                "requested_position_coverage": supported_positions / max(1, requested),
            })
        total_requested = sum(stage["requested_output_tokens"] for stage in stages)
        total_supported = sum(stage["supported_emitted_positions"] for stage in stages)
        rows.append({
            "chain_index": chain.collection_id,
            "stage_num": chain.stage_num,
            "requests_per_stage": [len(stage.requests) for stage in chain.stages],
            "max_compound_slots": max(len(stage.requests) for stage in chain.stages),
            "stages": stages,
            "all_stages_initial_supported": all(stage["initial_supported"] for stage in stages),
            "all_stages_atomic_at_four_slots": all(len(stage.requests) <= 4 for stage in chain.stages),
            "initial_supported_stages": sum(stage["initial_supported"] for stage in stages),
            "requested_output_tokens": total_requested,
            "supported_emitted_positions": total_supported,
            "requested_position_coverage": total_supported / max(1, total_requested),
        })

    eligible = [row["chain_index"] for row in rows
                if row["all_stages_initial_supported"] and row["all_stages_atomic_at_four_slots"]]
    coverage80 = [row["chain_index"] for row in rows
                  if row["chain_index"] in eligible and row["requested_position_coverage"] >= 0.8]
    full = [row["chain_index"] for row in rows
            if row["chain_index"] in eligible and row["requested_position_coverage"] == 1.0]
    np.random.seed(42)
    arrivals = []
    offset = 0.0
    for _ in range(8):
        offset += np.random.poisson(1000) / 1000
        arrivals.append(offset)
    result = {
        "schema": "public-support-audit/v1",
        "attempt": attempt,
        "scope": "Static tokenization and frozen-predictor support only; no inference, performance measurement or chain outcome selection",
        "trace_sha256": sha(trace_path),
        "predictor_sha256": sha(predictor_path),
        "predictor_version": pred.version,
        "q_cal": pred.q_cal,
        "chains": len(rows),
        "stages": sum(len(row["stages"]) for row in rows),
        "individual_calls": sum(sum(row["requests_per_stage"]) for row in rows),
        "chain0_prompt_token_contract": {
            "expected": EXPECTED_CHAIN0_PROMPTS,
            "actual": [stage["compound_prompt_tokens"] for stage in rows[0]["stages"]],
            "pass": [stage["compound_prompt_tokens"] for stage in rows[0]["stages"]] == EXPECTED_CHAIN0_PROMPTS,
        },
        "aggregate": {
            "all_stages_initial_supported_chains": len(eligible),
            "all_stages_initial_supported_and_coverage80_chains": len(coverage80),
            "all_stages_initial_supported_and_full_requested_coverage_chains": len(full),
            "all_initial_supported_stage_fraction": sum(row["initial_supported_stages"] for row in rows) / sum(len(row["stages"]) for row in rows),
            "requested_position_coverage": sum(row["supported_emitted_positions"] for row in rows) / sum(row["requested_output_tokens"] for row in rows),
            "chains_with_any_compound_stage": sum(any(len(stage.requests) > 1 for stage in chain.stages) for chain in chains),
            "chains_with_stage_over_four_slots": sum(row["max_compound_slots"] > 4 for row in rows),
        },
        "candidate_lists_support_only": {
            "all_initial_and_atomic": eligible,
            "all_initial_atomic_and_coverage80": coverage80,
            "all_initial_atomic_and_full_requested_coverage": full,
        },
        "official_arrival_plan": {
            "seed": 42,
            "arrival_rate": 1.0,
            "poisson_lambda_ms": 1000,
            "first_8_target_offsets_seconds": arrivals,
        },
        "rows": rows,
        "gpu_used": False,
    }
    receipt = root / "stage5_system_evidence/receipts" / f"{attempt}_PUBLIC_SUPPORT_AUDIT.json"
    with receipt.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(receipt), "chain0_token_contract": result["chain0_prompt_token_contract"],
                      "aggregate": result["aggregate"],
                      "first_all_initial_atomic": eligible[:12],
                      "first_coverage80": coverage80[:12],
                      "arrival_offsets": arrivals}, indent=2))
    return 0 if result["chain0_prompt_token_contract"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
