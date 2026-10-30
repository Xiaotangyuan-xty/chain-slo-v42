import os
GPU=0
EXPECTED_UUID='GPU-ff8c7d5c-f75c-26e1-43de-b40ca283f938'
if os.environ.get("STUDY_GPU")!="0":raise RuntimeError("only physical GPU0 authorized")
