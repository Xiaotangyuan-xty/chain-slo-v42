"""CPU-only, explicit-asset QRF server; never chooses a model or starts replay."""
import argparse
import json
import os
import socket
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'common_runtime'))
from prediction_transport import handle,PROTOCOL


def main():
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    root,stage=map(Path,sys.argv[1:3]);arm=sys.argv[3]
    from model_contract import prepare
    bound=prepare(root,stage)
    repo=Path(bound['repo'])
    sys.path[:0]=[str(repo),str(repo/'third_party/vllm'),bound['sklearn_deps'],bound['support_deps']]
    protocol=json.loads((stage/'PROTOCOL.json').read_text())
    assert protocol['predictor_protocol']==PROTOCOL
    args=SimpleNamespace(model=bound['qrf_model'],vectorizer=bound['qrf_vectorizer'],port=protocol['prediction_ports'][arm])
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from jitserve.request_analyzer import prediction
    model,vectorizer=prediction.load_model(args.model,args.vectorizer)
    with socket.socket(fileno=int(os.environ['STUDY_PREDICTOR_FD']))as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.listen(64)
        (Path(os.environ['STUDY_OUTPUT_DIR'])/'PREDICTOR_READY.json').write_text(json.dumps({'pid':os.getpid(),'port':args.port,'model_loaded':True})+'\n')
        print(json.dumps({'protocol':PROTOCOL,'model':args.model,'vectorizer':args.vectorizer,'port':args.port,'CPU_only':True}),flush=True)
        while True:
            connection,_=listener.accept()
            try:
                with connection:handle(connection,lambda prompts:prediction.predict(vectorizer,model,prompts))
            except Exception as error:
                print(json.dumps({'kind':'predictor_error','type':type(error).__name__,'error':str(error)}),flush=True)


if __name__=='__main__':main()
