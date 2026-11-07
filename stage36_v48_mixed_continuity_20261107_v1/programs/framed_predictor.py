"""CPU-only, explicit-asset QRF server; never chooses a model or starts replay."""
import argparse
import json
import os
import socket
from prediction_transport import handle,PROTOCOL


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True)
    parser.add_argument('--vectorizer',required=True);parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from jitserve.request_analyzer import prediction
    model,vectorizer=prediction.load_model(args.model,args.vectorizer)
    with socket.socket()as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',args.port));listener.listen(64)
        print(json.dumps({'protocol':PROTOCOL,'model':args.model,'vectorizer':args.vectorizer,'port':args.port,'CPU_only':True}),flush=True)
        while True:
            connection,_=listener.accept()
            try:
                with connection:handle(connection,lambda prompts:prediction.predict(vectorizer,model,prompts))
            except Exception as error:
                print(json.dumps({'kind':'predictor_error','type':type(error).__name__,'error':str(error)}),flush=True)


if __name__=='__main__':main()
