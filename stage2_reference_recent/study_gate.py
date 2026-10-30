import os
GPU=2
EXPECTED_UUID='GPU-e085e0e1-a283-82f2-325b-01981b51f984'
if os.environ.get("STUDY_GPU")!="2":raise RuntimeError("only physical GPU2 authorized")
