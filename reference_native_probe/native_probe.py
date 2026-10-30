"""Native HTTP->prediction->SequenceGroup->COLLECTIVE integration probe.

Decision source files remain byte-identical. Runtime wrappers observe inputs and
outputs; only an explicitly named diagnostic may refresh a snapshot at read.
"""
from pathlib import Path
from collections import Counter
import argparse,asyncio,functools,hashlib,inspect,json,logging,os,socket,sys,threading,time,traceback

def dump(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)

async def run(root,gpu,variant):
 dest=root/'gate1'/f'GPU{gpu}'/variant;dest.mkdir(exist_ok=False)
 reuse=json.loads((root/'NATIVE_RUNTIME_REUSE.json').read_text());repo=Path(reuse['repo'])
 sys.path[:0]=[str(repo),str(repo/'third_party/vllm'),str(root/'programs'),'/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_finite_h4_confirm_gpu01_20261018T114337Z_resume_v1/deps']
 from runtime_adapter import install
 install(dest.parent)
 events=[];lock=threading.Lock();clock=time.monotonic;beg=clock();phasefile=dest/'STATUS.json'
 def put(kind,**kw):
  with lock:events.append({'n':len(events),'kind':kind,'mono':clock()-beg,'wall':time.time(),**kw})
 def phase(x):dump(phasefile,{'phase':x,'UTC':time.time(),'pid':os.getpid(),'gpu':gpu,'variant':variant});print(json.dumps({'phase':x,'gpu':gpu,'variant':variant}),flush=True)
 originals={};native_end=None;httpserver=None;http_task=None;engine=None;results=[];groups={};ready_at={};all_calls={};fault=None
 try:
  phase('IMPORT_NATIVE')
  import torch,uvicorn,httpx
  from vllm.engine.arg_utils import AsyncEngineArgs
  from vllm.engine.async_llm_engine import AsyncLLMEngine
  from vllm.sequence import SequenceGroup
  from jitserve.scheduler.policy import SLOPolicy
  from jitserve import server
  from jitserve.request_analyzer.graph_context import GraphMatchContext
  from transformers import AutoTokenizer
  assert torch.__version__=='2.5.0+cu121'
  assert str(inspect.getfile(SequenceGroup)).startswith(str(repo))
  logging.getLogger('vllm').setLevel(logging.ERROR)
  phase('MODEL_INIT_WARMUP')
  resources=dict(dtype='half',tensor_parallel_size=1,pipeline_parallel_size=1,max_model_len=2048,
   gpu_memory_utilization=.48,swap_space=4,max_num_seqs=4,max_num_batched_tokens=256,
   enable_chunked_prefill=True,enable_prefix_caching=False,enforce_eager=True,
   scheduling_policy='jitserve',penalty_factor=100,top_k_selection=1,max_swaps=3,search_strategy='sliding_window',seed=20261024)
  args=AsyncEngineArgs(model='/home/taig/models/Qwen2.5-7B-Instruct',**resources)
  engine=AsyncLLMEngine.from_engine_args(args)
  assert torch.cuda.device_count()==1
  phase('NATIVE_HTTP_INIT')
  # Populate the same defaults as the native CLI, then configure this probe.
  parser=AsyncEngineArgs.add_cli_args(argparse.ArgumentParser())
  cli=parser.parse_args(['--model','/home/taig/models/Qwen2.5-7B-Instruct','--scheduling-policy','jitserve'])
  for k,v in dict(root_path='',disable_prediction=False,disable_graph_matching=False,use_default_length=False,
   graph_structure_type='deepresearch',graph_matching_mode='online',use_total_deadline=False,use_all_node=False,stage_ratio_method='execution_time').items():setattr(cli,k,v)
  app=await server.init_app(cli,llm_engine=engine)
  assert server.use_prediction and not server.use_default_length
  put('native_paths',server=inspect.getfile(server),prediction_transport=inspect.getfile(server.send_and_update_request_info),sequence=inspect.getfile(SequenceGroup),policy=inspect.getfile(SLOPolicy),resources=resources,context_clusters=server.graph_ctx.dynamic_clustering.get_clustering_info() if hasattr(server.graph_ctx.dynamic_clustering,'get_clustering_info') else {'medoids':len(server.graph_ctx.dynamic_clustering.medoids),'finished':len(server.graph_ctx.dynamic_clustering.finished_requests)})
  def wrap(obj,name,fn):
   orig=getattr(obj,name);originals[(obj,name)]=orig;setattr(obj,name,functools.wraps(orig)(fn(orig)))
  def request_fields(info):return {'collection':info.collection_id,'stage':info.stage_id,'cache':info.output_len,'info_object':id(info)}
  wrap(server,'_maybe_start_prediction',lambda orig:lambda info,prompt:(put('prediction_submit',**request_fields(info)),orig(info,prompt),put('prediction_submit_return',**request_fields(info)))[1])
  def updater(orig):
   def f(info,prompt):
    put('prediction_worker_begin',**request_fields(info));result=orig(info,prompt);ready_at[id(info)]=clock()-beg
    put('prediction_ready',**request_fields(info));return result
   return f
  wrap(server,'send_and_update_request_info',updater)
  def ctor(orig):
   def f(self,*a,**kw):
    result=orig(self,*a,**kw);groups[self.request_id]=self;all_calls[(self.collection_id,self.request_info.stage_id)]=self
    put('native_assign_snapshot',id=self.request_id,collection=self.collection_id,stage=self.request_info.stage_id,cache=self.request_info.output_len,snapshot=self.predict_output_length,info_object=id(self.request_info),ready_before_assign=id(self.request_info) in ready_at,arrival=self.arrival_time,deadline=self.deadline)
    return result
   return f
  wrap(SequenceGroup,'__init__',ctor)
  def allocation(orig):
   def f(self,sg):
    peers=list(self.seq_group_dict[sg.collection_id]);before=[{'id':g.request_id,'cache':g.request_info.output_len,'snapshot':g.predict_output_length,'decode':g.first_seq.get_decode_len(),'info_object':id(g.request_info)} for g in peers]
    if variant=='snapshot_refresh_sensitivity':
     for g in peers:g.predict_output_length=g.request_info.output_len
    value=orig(self,sg)
    put('COLLECTIVE_read',id=sg.request_id,collection=sg.collection_id,before=before,after=[{'id':g.request_id,'snapshot':g.predict_output_length} for g in peers],unit_time=self._get_interval_time(),remain=value,schedule_count=self.num_schedule_count)
    return value
   return f
  wrap(SLOPolicy,'allocation_control',allocation)
  def share(orig):
   def f(self,*a,**kw):
    started=clock();v=orig(self,*a,**kw);put('native_graph_share',share=v,stage=kw.get('stage_id',0),collection=a[3] if len(a)>3 else kw.get('collection_id'),seconds=clock()-started,medoids=len(self.dynamic_clustering.medoids));return v
   return f
  wrap(GraphMatchContext,'calculate_stage_ratio',share)
  for s in engine.engine.scheduler:
   assert type(s).__name__=='SLOScheduler'
   old=s.schedule
   def schedule(*a,_s=s,_orig=old,**kw):
    queues={name:[{'id':g.request_id,'arrival':g.arrival_time,'prompt':g.first_seq.get_prompt_len(),'output':g.first_seq.get_output_len(),'computed':g.first_seq.data.get_num_computed_tokens()} for g in getattr(_s,name)] for name in ['waiting','running','swapped']}
    step_before=_s.policy.num_schedule_count;res=_orig(*a,**kw);md,out,_=res
    actual=[z.seq_group.request_id for z in out.scheduled_seq_groups]
    held=[x['id'] for x in queues['running']];ordered=sorted(queues['waiting']+queues['swapped'],key=lambda z:z['arrival']);shim=(held+[x['id'] for x in ordered if x['id'] not in held])[:4]
    put('native_schedule',count=_s.policy.num_schedule_count,official_frame=_s.policy.num_schedule_count%50==0,
     queues=queues,scheduled=actual,chunks=[{'id':z.seq_group.request_id,'chunk':z.token_chunk_size,'prefill':z.seq_group.is_prefill()} for z in out.scheduled_seq_groups],shared_keep_fill_shadow=shim,
     ignored=[g.request_id for g in out.ignored_seq_groups],preempted=out.preempted)
    return res
   s.schedule=schedule
  tokenizer=AutoTokenizer.from_pretrained('/home/taig/models/Qwen2.5-7B-Instruct',local_files_only=True)
  prompts=[tokenizer.decode(([100+i]*n),skip_special_tokens=True) for i,n in enumerate([32,48,96,128,512,768])]
  fixture=[]
  for i,prompt in enumerate(prompts):
   fixture.append({'collection':9000+i,'prompt':prompt,'outputs':[48,48] if i<4 else [96,48],'arrival_offset':0. if i<5 else .15,'stages':2,'chain_deadline':60.})
  dump(dest/'INPUT_FIXTURE.json',fixture)
  listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(('127.0.0.1',0));listener.listen(64);port=listener.getsockname()[1]
  httpserver=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning',access_log=False))
  http_task=asyncio.create_task(httpserver.serve(sockets=[listener]))
  while not httpserver.started:
   if http_task.done():await http_task
   await asyncio.sleep(.02)
  phase('NATIVE_COHORT')
  base=clock()
  async with httpx.AsyncClient(timeout=240) as client:
   async def job(item):
    await asyncio.sleep(item['arrival_offset']);start=clock();collection=item['collection']
    for ordinal,nout in enumerate(item['outputs']):
     # Upstream DeepResearch client uses a zero-indexed stage_id.
     prompt=item['prompt']+'\nContinue with the next step.'*ordinal
     body={'stream':False,'client_id':collection,'num_stages':2,'requests_per_stage':[1,1],
      'accumulate_stage_ratio':(ordinal+1)/2,'served_time':clock()-start,
      'slo_constraint':[.8,.08,item['chain_deadline']/2],
      'request_info':{'request_type':2,'collection_id':collection,'deadline':999,'output_len':nout,'prompt':prompt,'stage_id':ordinal,'request_id':0},
      'sampling_params':{'temperature':0.,'ignore_eos':True,'max_tokens':nout,'min_tokens':nout}}
     put('HTTP_arrival',collection=collection,stage=ordinal,relative=clock()-base)
     response=await client.post(f'http://127.0.0.1:{port}/generate',json=body);response.raise_for_status()
     response.json();g=all_calls[(collection,ordinal)]
     results.append({'collection':collection,'stage':ordinal,'required_outputs':nout,'actual_outputs':g.first_seq.get_output_len(),'snapshot_at_finish':g.predict_output_length,'cache_at_finish':g.request_info.output_len,'id':g.request_id})
   await asyncio.wait_for(asyncio.gather(*(job(x) for x in fixture)),timeout=300)
  phase('DRAIN_PREDICTIONS')
  limit=clock()+30
  while len(ready_at)<len(results) and clock()<limit:await asyncio.sleep(.05)
  if len(ready_at)!=len(results):raise RuntimeError('prediction ready events incomplete')
 except BaseException as exc:
  fault={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()};dump(dest/'FAILURE.json',fault)
 finally:
  phase('CLEANUP')
  if httpserver is not None:
   httpserver.should_exit=True
   if http_task:
    try:await asyncio.wait_for(http_task,timeout=10)
    except BaseException:pass
  if engine is not None:engine.shutdown_background_loop()
  for (obj,name),orig in reversed(list(originals.items())):setattr(obj,name,orig)
  with (dest/'EVENTS.jsonl').open('x') as f:
   for e in events:f.write(json.dumps(e,allow_nan=False)+'\n')
  dump(dest/'CALL_RESULTS.json',results)
  assigns=[x for x in events if x['kind']=='native_assign_snapshot'];reads=[x for x in events if x['kind']=='COLLECTIVE_read'];schedules=[x for x in events if x['kind']=='native_schedule'];shares=[x for x in events if x['kind']=='native_graph_share']
  between=[x for x in schedules if not x['official_frame']]
  summary={'complete':fault is None,'failure':fault,'variant':variant,'gpu':gpu,'calls_complete':len(results),'all_outputs_complete':bool(results) and all(x['actual_outputs']==x['required_outputs'] for x in results),
   'assignments':len(assigns),'assign_snapshot_1024':sum(x['snapshot']==1024 for x in assigns),'prediction_ready_before_assign':sum(x['ready_before_assign'] for x in assigns),
   'COLLECTIVE_reads':len(reads),'COLLECTIVE_peer_observations':sum(len(x['before']) for x in reads),'COLLECTIVE_snapshot_1024':sum(y['snapshot']==1024 for x in reads for y in x['before']),
   'COLLECTIVE_snapshot_differs_cache':sum(y['snapshot']!=y['cache'] for x in reads for y in x['before']),
   'native_graph_share_histogram':dict(Counter(str(x['share']) for x in shares)),'native_graph_warm_medoids_max':max((x['medoids'] for x in shares),default=0),
   'native_schedule_ticks':len(schedules),'official_frames':sum(x['official_frame'] for x in schedules),
   'between_frame_ticks':len(between),'between_frame_keep_fill_membership_differs':sum(set(x['scheduled'])!=set(x['shared_keep_fill_shadow']) for x in between),
   'all_event_types':dict(Counter(x['kind'] for x in events)),'seconds_including_init':clock()-beg,
   'scope':'Actual native /generate HTTP + official socket QRF + native GraphMatchContext + actual AsyncLLMEngine/COLLECTIVE. Small integration fixture, not paired SLO performance.'}
  dump(dest/'SUMMARY_LIGHT.json',summary);phase('COMPLETE' if fault is None else 'FAILED_PRESERVED')
 return 0 if fault is None else 1

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--gpu',type=int,choices=[0,1],required=True);ap.add_argument('--variant',choices=['native_unmodified','snapshot_refresh_sensitivity'],default='native_unmodified');a=ap.parse_args()
 sys.exit(asyncio.run(run(Path(a.root),a.gpu,a.variant)))
