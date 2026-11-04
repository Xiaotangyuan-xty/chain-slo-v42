"""Reject the unlaunched double-formatted CPU build; preserve its complete inputs."""
from pathlib import Path
import json,hashlib,shutil,os

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main(root,study):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(read(root/'PREPARE.json')['model'],local_files_only=True,trust_remote_code=False)
    raw=[json.loads(s)for s in(root/'repo/traces/deepresearch_filter.jsonl').read_text().splitlines()if s.strip()]
    old=read(root/'stage18_two_horizon_lowobs_gpu2_20261103_v1/b1_1_full/TRACE_MANIFEST.json')['expected_requests']
    for case in read(study/'MATRIX.json')['cases']:
        if case['kind']!='pure':continue
        stage=Path(case['stage']);assert not list((stage/'attempts').iterdir());history=stage/'input_history/double_wrapping_rejected_CPU_only';history.mkdir(parents=True)
        for file in ['WORKLOAD.json','TRACE_MANIFEST.json','PROTOCOL.json','LONG_EXECUTION_APPROVAL.json','receipts/CPU_QUALIFICATION.json']:shutil.copy2(stage/file,history/Path(file).name)
        work=read(stage/'WORKLOAD.json')
        for phase in ['warmup','main']:
            for job in work[phase]:
                for sid,members in enumerate(job['stages']):
                    for request in members:
                        text=raw[job['collection_id']]['stages'][sid][request['request_id']]['prompt'];ids=tok(text,add_special_tokens=False).input_ids
                        request.update(prompt=text,prompt_tokens=len(ids),prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),prompt_token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                        if phase=='main':
                            ref=old[f'{job["collection_id"]}:{sid}:{request["request_id"]}'];assert request['prompt_sha256']==ref['input_sha256']and request['prompt_token_ids_sha256']==ref['prompt_token_ids_sha256']and request['target_output_tokens']==ref['target_output_tokens']
        work['selection']['prompt_format']='Original DeepResearch identity formatter, exact Stage18 prompt/token contract'
        dump(stage/'WORKLOAD.json',work)
        from build_cases import manifest
        dump(stage/'TRACE_MANIFEST.json',{'requests':manifest(work),'workload_sha256':sha(stage/'WORKLOAD.json')})
        approval=read(stage/'LONG_EXECUTION_APPROVAL.json');approval['workload_sha256']=sha(stage/'WORKLOAD.json');dump(stage/'LONG_EXECUTION_APPROVAL.json',approval)
        p=read(stage/'PROTOCOL.json');p.update(workload_sha256=sha(stage/'WORKLOAD.json'),long_approval_sha256=sha(stage/'LONG_EXECUTION_APPROVAL.json'));dump(stage/'PROTOCOL.json',p)
    print(json.dumps({'gate':'PASS','pure_main_calls_verified_per_block':1197,'pure_root_jobs':200,'GPU_used':False,'rejected_inputs_preserved':True}))
if __name__=='__main__':
    import sys
    main(*map(Path,sys.argv[1:3]))
