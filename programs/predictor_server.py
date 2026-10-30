import json
import os
from pathlib import Path
import sys
import time


def main():
    root = Path(sys.argv[1]).resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    prepare = json.loads((root / "PREPARE.json").read_text())
    repo = Path(prepare["repo"])
    sys.path[:0] = [
        str(repo),
        str(repo / "third_party/vllm"),
        prepare["sklearn_deps"],
        prepare["support_deps"],
    ]
    from jitserve.request_analyzer import prediction

    original_load = prediction.load_model

    def audited_load(model_path, tokenizer_path):
        print(json.dumps({"phase": "qrf_load_start", "pid": os.getpid(), "wall": time.time()}), flush=True)
        loaded = original_load(model_path, tokenizer_path)
        print(json.dumps({"phase": "qrf_load_ready", "pid": os.getpid(), "wall": time.time()}), flush=True)
        return loaded

    prediction.load_model = audited_load
    prediction.start_client(prepare["qrf_model"], prepare["qrf_vectorizer"])


if __name__ == "__main__":
    main()
