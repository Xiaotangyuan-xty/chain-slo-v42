import os,sys,json
from pathlib import Path
def main():
 root,stage=map(Path,sys.argv[1:3]);arm=sys.argv[3]
 from model_contract import prepare as bound_prepare
 p=bound_prepare(root,stage);repo=Path(p['repo'])
 sys.path[:0]=[str(repo),str(repo/'third_party/vllm'),p['sklearn_deps'],p['support_deps']]
 protocol=json.loads((stage/'PROTOCOL.json').read_text())
 assert protocol['predictor_protocol']=='length_framed_pickle_v1'and os.environ['CUDA_VISIBLE_DEVICES']==''
 sys.argv=[sys.argv[0],'--model',p['qrf_model'],'--vectorizer',p['qrf_vectorizer'],'--port',str(protocol['prediction_ports'][arm])]
 from framed_predictor import main as serve
 serve()
if __name__=='__main__':main()
