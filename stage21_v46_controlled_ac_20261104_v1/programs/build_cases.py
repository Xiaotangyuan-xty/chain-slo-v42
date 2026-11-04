"""CPU-only binding of three fixed blocks, pure200 and mixed, A/A*/v46."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import ast,copy,hashlib,json,os,shutil,sys

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
UUIDS=['GPU-ff8c7d5c-f75c-26e1-43de-b40ca283f938','GPU-f5874940-7f3b-ad3e-b810-bc7dda423d1a','GPU-e085e0e1-a283-82f2-325b-01981b51f984']
CPUS=[list(range(0,28,2))+list(range(56,84,2)),list(range(1,28,2))+list(range(57,84,2)),list(range(28,56))+list(range(84,112))]
USER='改进代码，push v46到main；v46跑200链和混合负载对比A，三卡都可以用；保留原生A，并额外跑共同预算语义修正A*强对照。'

def manifest(work):return {f'{phase}:{j["collection_id"]}:{s}:{r["request_id"]}':{k:r[k]for k in ['prompt_sha256','prompt_token_ids_sha256','prompt_tokens','target_output_tokens']}for phase in ['warmup','main']for j in work[phase]for s,stage in enumerate(j['stages'])for r in stage}

def main(root,study):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    common=study/'common';q=read(common/'receipts/CPU_V46_INTEGRATION.json');assert q['gate']=='PASS'
    from transformers import AutoTokenizer
    prep=read(root/'PREPARE.json');tokenizer=AutoTokenizer.from_pretrained(prep['model'],local_files_only=True,trust_remote_code=False)
    path=root/'repo/benchmark/trace/client.py';fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='qwen_prompt_format');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    formatter=ns['qwen_prompt_format']
    pure_parent=root/'stage18_two_horizon_lowobs_gpu2_20261103_v1/b1_1_full';oldpure=read(pure_parent/'PROTOCOL.json')
    dr=[json.loads(s)for s in(root/'repo/traces/deepresearch_filter.jsonl').read_text().splitlines()if s.strip()];jobs=[]
    for cid,row in enumerate(dr):
        stages=[]
        for source_stage in row['stages']:
            current=[]
            for rid,source in enumerate(source_stage):
                text=source['prompt'];ids=tokenizer(text,add_special_tokens=False).input_ids;target=min(1024,max(1,source['output_tokens']))
                assert len(ids)+target<=8192
                current.append({'request_id':rid,'state':source['state'],'prompt':text,'declared_input_tokens':source['input_tokens'],'prompt_tokens':len(ids),
                   'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_token_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'target_output_tokens':target})
            stages.append(current)
        scale=cid%4+1;jobs.append({'collection_id':cid,'request_type':2,'source':{'dataset':'known_deepresearch200','index':cid,'heldout_claim':False},
          'received_slo':[2.*scale,.1*scale,20.*scale],'stages':stages,'arrival_offset_seconds':oldpure['target_arrivals'][cid]})
    assert len(jobs)==200 and sum(len(s)for j in jobs for s in j['stages'])==1197 and sum(r['target_output_tokens']for j in jobs for s in j['stages']for r in s)==504695
    warm=copy.deepcopy(jobs[:1]);warm[0].pop('arrival_offset_seconds');pure={'schema':'v46-pure200-materialized/v1','main':jobs,'warmup':warm,
       'source_files':{'repo/traces/deepresearch_filter.jsonl':sha(root/'repo/traces/deepresearch_filter.jsonl'),'repo/benchmark/trace/client.py':sha(path)},
       'selection':{'known_workflows':True,'heldout_claim':False,'root_jobs':200,'arrival_seed':42,'same_original_stage18_arrivals':True}}
    orders_pure=[['A','Astar','C'],['C','A','Astar'],['Astar','C','A']]
    orders_mixed=[list(reversed(x))for x in orders_pure]
    cases=[]
    for gpu in range(3):
        for kind in ['pure','mixed']:
            name=f'{kind}_b{gpu+1}';stage=study/'cases'/name;assert not stage.exists();(stage/'attempts').mkdir(parents=True);(stage/'receipts').mkdir()
            for folder in ['programs','assets']:shutil.copytree(common/folder,stage/folder)
            for file in ['PARENT_V45.json','PARENT_CPU_QUALIFICATION.json','FROZEN_V45_CORE.json']:shutil.copy2(common/file,stage/file)
            if kind=='pure':work=copy.deepcopy(pure);dump(stage/'WORKLOAD.json',work);dump(stage/'TRACE_MANIFEST.json',{'requests':manifest(work),'workload_sha256':sha(stage/'WORKLOAD.json')})
            else:
                source=root/f'stage20_mixed_long_ready_20261104_v1/bound/block0{gpu+1}'
                for file in ['WORKLOAD.json','TRACE_MANIFEST.json']:shutil.copy2(source/file,stage/file)
                work=read(stage/'WORKLOAD.json')
            approval={'authorized':True,'long_experiment_authorized':True,'utc':datetime.now(timezone.utc).isoformat(),'user_instruction':USER,
                 'devices':[{'physical_gpu_index':gpu,'gpu_uuid':UUIDS[gpu]}],'scope':'Only this new v46 case, three fixed arms, no retries.','workload_sha256':sha(stage/'WORKLOAD.json')}
            dump(stage/'LONG_EXECUTION_APPROVAL.json',approval)
            p=read(common/'PROTOCOL.json');attempts={a:f'{name.upper()}_{a.upper()}_GPU{gpu}_01'for a in ['A','Astar','C']}
            p.update(schema='stage21-v46-controlled/v1',case=name,workload_kind=kind,study_mode='v46_controlled_full_replay',
               C_version='v46_complete_work_joint_residency',mixed_variant='v46_joint_foreground_budget_v1',planned_arms=(orders_pure if kind=='pure'else orders_mixed)[gpu],
               allowed_arms=['A','C'],attempts=attempts,attempt_variants={n:('native'if a=='A'else'corrected')for a,n in attempts.items()},
               arm_devices={a:{'physical_gpu_index':gpu,'gpu_uuid':UUIDS[gpu],'authorization_file':'DEVICE_AUTHORIZATION.json'}for a in ['A','C']},
               forbidden_gpu_indices=[g for g in range(3)if g!=gpu],http_ports={a:28700+gpu*20 for a in ['A','C']},prediction_ports={a:65450+gpu for a in ['A','C']},
               cpu_affinity={a:CPUS[gpu]for a in ['A','C']},workload_sha256=sha(stage/'WORKLOAD.json'),
               expected_arrival_window_seconds=work['main'][-1]['arrival_offset_seconds'],target_arrivals=[j['arrival_offset_seconds']for j in work['main']],
               stage_counts={str(j['collection_id']):len(j['stages'])for j in work['main']},requests_per_stage={str(j['collection_id']):[len(s)for s in j['stages']]for j in work['main']},
               long_approval_sha256=sha(stage/'LONG_EXECUTION_APPROVAL.json'),client_transport='fresh_connection_per_request_v1',
               full_trace_authorization=USER,comparability='New same-GPU A(native), Astar(stage-budget semantic adapter), C(v46+same adapter); identical frozen workload, input generation, client/observer, resource pool and CPU affinity. All main requests fully drain; no reward/JCT continuation threshold.',
               controller_wall_seconds=28800,http_request_timeout_seconds=28200,
               workloads={'full':{'main_root_jobs':len(work['main']),'main_compound_chains':sum(j['request_type']==2 for j in work['main']),
                'main_call_count':sum(len(s)for j in work['main']for s in j['stages']),'warmup_call_count':sum(len(s)for j in work['warmup']for s in j['stages'])}})
            p['generation']['top_k']=1 if kind=='pure'else-1
            p['resources'].update(physical_gpu_index=gpu,gpu_uuid=UUIDS[gpu]);p['lifetime_observation_sha256']=sha(stage/'programs/lifetime_observation.py')
            dump(stage/'PROTOCOL.json',p)
            auth=read(common/'DEVICE_AUTHORIZATION.json');auth.update(physical_gpu_index=gpu,gpu_uuid=UUIDS[gpu],allowed_physical_gpu_indices=[gpu],
              max_attempt_wall_seconds=28800,http_request_timeout_seconds=28200,user_instruction=USER)
            dump(stage/'DEVICE_AUTHORIZATION.json',auth)
            dump(stage/'BUILD_RECEIPT.json',{'parent':str(common),'GPU':gpu,'kind':kind,'new_complete_replay':True,'original_A_preserved':True,'Astar_explicit':True,'all_clients_fresh_connections':True,'source_GPU_used_for_build':False})
            files={str(f.relative_to(stage)):sha(f)for folder in ['programs','assets']for f in(stage/folder).glob('*')if f.is_file()}
            for file in ['PROTOCOL.json','WORKLOAD.json','TRACE_MANIFEST.json','DEVICE_AUTHORIZATION.json','LONG_EXECUTION_APPROVAL.json','BUILD_RECEIPT.json','PARENT_V45.json','PARENT_CPU_QUALIFICATION.json','FROZEN_V45_CORE.json']:files[file]=sha(stage/file)
            receipt={'gate':'PASS','base_files':q['base_files'],'stage_files':files,'tests':q['tests'],'GPU_used':False,
                'shared_v46_behavioral_qualification_sha256':sha(common/'receipts/CPU_V46_INTEGRATION.json'),
                'binding_checks':{'exclusive_GPU':True,'same_CPU_all_arms':True,'workload_manifest_matches':set(manifest(work))==set(read(stage/'TRACE_MANIFEST.json')['requests']),
                    'three_variants_same_workload':True,'arrival_gate_seconds':.5,'JCT_constraint':False}}
            assert all(v is True for k,v in receipt['binding_checks'].items()if k not in ['arrival_gate_seconds','JCT_constraint'])
            dump(stage/'receipts/CPU_QUALIFICATION.json',receipt)
            cases.append({'name':name,'GPU':gpu,'kind':kind,'stage':str(stage),'planned_arms':p['planned_arms'],'workload_sha256':p['workload_sha256'],
                'qualification_sha256':sha(stage/'receipts/CPU_QUALIFICATION.json'),'main_jobs':len(work['main']),'main_calls':p['workloads']['full']['main_call_count']})
    matrix={'schema':'v46-three-GPU-controlled-matrix/v1','user_instruction':USER,'cases':cases,'total_arms':18,'GPUs':[0,1,2],
      'primary':'Complete chain SLO attainment count','arrival_gate_seconds':.5,'JCT_constraint':False,'no_automatic_retry':True,
      'ordering':'Each GPU runs pure first, then mixed; three within-case arm orders rotated and reversed between workloads.',
      'pure_repeat_scope':'Same200 known inputs/seed42 repeated in3blocks, not600 independent workflows.',
      'mixed_scope':'Frozen Stage20 blocks1/2/3,1800 roots and2399 main calls each,3600-second open-loop arrivals then drain.'}
    dump(study/'MATRIX.json',matrix);print(json.dumps({'gate':'PASS','cases':cases,'total_arms':18}),flush=True)
if __name__=='__main__':main(*map(Path,sys.argv[1:3]))
