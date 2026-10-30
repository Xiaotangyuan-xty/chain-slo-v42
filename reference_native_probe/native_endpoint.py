"""Original socket predictor service, configured with model-matched official assets."""
import os,sys,json,time
from pathlib import Path
root=Path(sys.argv[1]);os.environ['CUDA_VISIBLE_DEVICES']=''
reuse=json.loads((root/'NATIVE_RUNTIME_REUSE.json').read_text())
sys.path[:0]=[reuse['repo'],reuse['repo']+'/third_party/vllm','/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_finite_h4_confirm_gpu01_20261018T114337Z_resume_v1/deps']
from jitserve.request_analyzer.prediction import start_client
print(json.dumps({'phase':'native_predictor_load','pid':os.getpid(),'model':'official_qwen25_7b','UTC':time.time()}),flush=True)
start_client(str(root/'native_assets/qrf_model/0_qrf_lmsys_chat_qwen25_7b.pkl'),str(root/'native_assets/qrf_vectorizer/0_qrf_lmsys_chat_qwen25_7b.pkl'))
