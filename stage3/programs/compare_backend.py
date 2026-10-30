import json
import os
from pathlib import Path
import sys


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def semantic_records(case):
    return [record["groups"] for record in case["execute_records"]]


def first_divergence(a_tokens, b_tokens):
    for index, (a, b) in enumerate(zip(a_tokens, b_tokens)):
        if a != b:
            return {"index": index, "A": a, "B": b}
    return None


def main():
    root = Path(sys.argv[1]).resolve()
    a = json.loads((root / "stage3/attempts/backend_A_02/backend_results.json").read_text())
    b = json.loads((root / "stage3/attempts/backend_B_01/backend_results.json").read_text())
    a_receipt = json.loads((root / "stage3/attempts/backend_A_02/controller_receipt.json").read_text())
    b_receipt = json.loads((root / "stage3/attempts/backend_B_01/controller_receipt.json").read_text())
    failures = []
    if a_receipt["failure"] is not None or b_receipt["failure"] is not None:
        failures.append("controller failure")
    if not a_receipt["gpu_released"] or not b_receipt["gpu_released"]:
        failures.append("study-owned GPU process remains")
    cases = []
    for ca, cb in zip(a["cases"], b["cases"]):
        actions_equal = semantic_records(ca) == semantic_records(cb)
        if ca["name"] != cb["name"] or not actions_equal:
            failures.append(f"action mismatch: {ca['name']}")
        outputs = []
        for oa, ob in zip(ca["outputs"], cb["outputs"]):
            if oa["request_id"] != ob["request_id"] or oa["output_tokens"] != ob["output_tokens"]:
                failures.append(f"output mismatch: {ca['name']}")
            outputs.append({
                "request_id": oa["request_id"],
                "output_tokens": oa["output_tokens"],
                "first_token_divergence": first_divergence(oa["token_ids"], ob["token_ids"]),
                "token_ids_equal": oa["token_ids"] == ob["token_ids"],
            })
        cases.append({
            "name": ca["name"],
            "A_wall_seconds": ca["wall_seconds"],
            "B_wall_seconds": cb["wall_seconds"],
            "B_over_A_wall": cb["wall_seconds"] / ca["wall_seconds"],
            "A_execute_model_seconds": ca["execute_model_seconds"],
            "B_execute_model_seconds": cb["execute_model_seconds"],
            "B_over_A_execute_model": cb["execute_model_seconds"] / ca["execute_model_seconds"],
            "engine_steps_A": ca["engine_steps"],
            "engine_steps_B": cb["engine_steps"],
            "actions_equal": actions_equal,
            "outputs": outputs,
        })
    total_a = sum(case["A_wall_seconds"] for case in cases)
    total_b = sum(case["B_wall_seconds"] for case in cases)
    result = {
        "calibration_complete": not failures,
        "failures": failures,
        "cases": cases,
        "total_main_wall_seconds": {"A": total_a, "B": total_b, "B_over_A": total_b / total_a},
        "relative_backend_slowdown_percent": (total_b / total_a - 1) * 100,
        "interpretation": "Small fixed-action fixed-output backend calibration. It is not a JITServe policy, C mechanism, statistical-equivalence or general-system conclusion.",
        "next_step": "Implement and semantically accept C on the exact B backend, then run the small B/C comparison" if not failures else "Stop and preserve the backend mismatch",
        "formal_matrix_allowed": False,
    }
    dump(root / "stage3/receipts/BACKEND_COMPARISON.json", result)
    print(json.dumps({"calibration_complete": result["calibration_complete"], "failures": failures,
                      "B_over_A_total": result["total_main_wall_seconds"]["B_over_A"],
                      "case_B_over_A": {case["name"]: case["B_over_A_wall"] for case in cases},
                      "receipt": str(root / "stage3/receipts/BACKEND_COMPARISON.json")}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
