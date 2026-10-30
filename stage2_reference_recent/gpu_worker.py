"""Fixed64 live GPU2 episodes, one QRF inference-parallelism intervention."""
from pathlib import Path
import gzip,hashlib,json,os,pickle,sys,time,traceback
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def main(root):
    root=Path(root);plan=read(root/'PLAN_SERVER_ONLY.json');parent=Path(plan['engine_parent'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2' and os.environ['STUDY_GPU']=='2'
    sys.path[:0]=[str(root/'programs'),str(parent/'src'),str(parent/'deps')]
    freeze=read(root/'FREEZE.json')
    def verify():
        for p,h in freeze['bindings'].items():
            if sha(p)!=h:raise RuntimeError('historical binding changed '+p)
        for p,h in freeze['new_files'].items():
            if sha(root/p)!=h:raise RuntimeError('study source changed '+p)
    verify();target=root/'episodes';target.mkdir(exist_ok=True);attempts=root/'attempts';attempts.mkdir(exist_ok=True)
    def status(phase,**kw):write(root/'WORKER_STATUS.json',{'phase':phase,'GPU':2,'pid':os.getpid(),'committed':len(list(target.glob('*.commit.json'))),'total':64,'UTC':time.time(),**kw})
    from serial_qrf import load_serial
    from qrf_study import ARMS,MeasuredPredictor,policy_class,default_h4,validate_original_gmax_trace
    from accept_bridge import validate_policy_trace
    from bridge_policies import ARMS as BRIDGES
    from pool_worker import validate
    from pool_backend import Pool
    from pool_policies import Performance
    from v2_predictor import Predictor
    import live_engine,h4_policy
    original=live_engine.OfficialGMAX
    references={b:{tuple(r['features']):r['prediction'] for r in rr} for b,rr in read(plan['prediction_reference_path']).items()}
    assets={};identities={}
    for batch,path in plan['asset_parents'].items():
        p=Path(path)
        with (p/'JIT_QRF.pkl').open('rb') as f:q4=pickle.load(f)
        assert q4.n_jobs==4
        q1,identity=load_serial(p/'JIT_QRF.pkl');identities[batch]=identity
        # Both implementations warm on the same public feature before any episode.
        feature=next(iter(references[batch]))
        assert float(q4.predict([feature],quantiles=.95)[0])==float(q1.predict([feature],quantiles=.95)[0])==references[batch][feature]
        assets[batch]=(read(p/'PERFORMANCE.json'),read(p/'PREDICTOR.json'),MeasuredPredictor(q4),MeasuredPredictor(q1),read(p/'JIT_GRAPHS.json'),read(p/'JIT_PREDICTION_REGISTER.json'))
    write(root/'RUNTIME_QRF_IDENTITIES.json',identities)
    status('model_load');pool=Pool();status('numeric_startup')
    from pool_numeric import verify as numeric_verify
    numeric=numeric_verify(pool);pool.reset();write(root/f'STARTUP_NUMERIC_{os.getpid()}.json',numeric)
    for job in plan['jobs']:
        ident=job['id'];path=target/(ident+'.json.gz');commit=target/(ident+'.commit.json')
        if commit.exists():
            if sha(path)!=read(commit)['episode_sha256']:raise RuntimeError('committed episode corrupted')
            continue
        verify();inp=read(job['input_path']);arm=job['arm'];pa,pr,q4,q1,graphs,reg=assets[job['batch']]
        perf=Performance(pa);pred=Predictor.from_dict(pr)
        measured=q4 if arm==ARMS[0] else q1;measured.reset()
        assert measured.model.n_jobs==(4 if arm==ARMS[0] else 1)
        live_engine.OfficialGMAX=policy_class(original,arm,perf.decode_seconds);h4_policy.fixed_h4_action=default_h4
        wire='v2_fixed_h4' if arm==ARMS[2] else 'JITServe_official_GMAX_core'
        attempt=1+len(list(attempts.glob(ident+'_*.json')));begin=time.monotonic();status('episode',current=ident)
        try:
            e=live_engine.run_episode(inp,wire,pool,perf,pred,measured,graphs,reg['median_output'],router='cache_load')
            acceptance=validate(inp,e,2)
            if arm in ARMS[:2]:policy_accept=validate_original_gmax_trace(e)
            else:policy_accept=validate_policy_trace(e,BRIDGES[1] if arm==ARMS[2] else BRIDGES[3],perf.decode_seconds)
            ncalls=sum(len(p['calls']) for p in inp['programs'])
            if arm==ARMS[2]:assert not measured.records and e['official_gmax'] is None;numeric_qrf={'passed':True,'calls':0,'prediction_path_disabled':True}
            else:
                numeric_qrf=measured.validate(references[job['batch']],ncalls)
                assert e['official_gmax']['counts']['prediction']==ncalls and e['official_gmax']['counts'].get('prediction_failure_default_1024',0)==0
                assert e['official_gmax']['prediction_lifecycle_counts']['calls']==ncalls
            cm={(c['program'],c['ordinal']):c for c in e['call_metrics']}
            for p in inp['programs']:
                for ordinal,s in enumerate(p['calls']):
                    release=p['arrival'] if ordinal==0 else cm[p['id'],ordinal-1]['finish']+p['calls'][ordinal-1]['away']
                    assert abs(cm[p['id'],ordinal]['release']-release)<1e-8
            assert len(e['trace'])==len(e['batches'])
            for t in e['trace']:
                assert t['forced'] is None or t['forced'] in t['selected']
                assert set(t['selected'])<=set(t['legal'])|({t['forced']} if t['forced'] else set())
            assert all(r['audit']['registered_common_router']=='cache_load' and r['audit']['common_router_length_prior']==1024 for r in e['routes'])
            e.update(arm=arm,wire_arm=wire,job_binding=job,experiment_identity='gmax_qrf_njobs1_gpu2_v1',
                     qrf_runtime=numeric_qrf,qrf_prediction_records=measured.records,
                     acceptance={**acceptance,'policy_replay':policy_accept,'QRF_exact_original_values':True,'endogenous_release':True})
            tmp=target/(ident+f'.attempt{attempt}.partial.gz')
            with gzip.open(tmp,'xt',compresslevel=3) as f:json.dump(e,f,separators=(',',':'),allow_nan=False)
            os.link(tmp,path);tmp.unlink()
            row={k:job[k] for k in ['id','batch','index','seed','arm','source_GPU','source_GPU_UUID','input_sha256']}
            row.update(GPU=2,GPU_UUID=plan['GPU_UUID'],metrics=e['metrics'],episode_sha256=sha(path),path=str(path),seconds=time.monotonic()-begin,
                       passed=True,UTC=time.time(),qrf_runtime=numeric_qrf,
                       lifecycle_counts=e['official_gmax']['prediction_lifecycle_counts'] if e['official_gmax'] else {},
                       cohorts={k:{'chains':sum(j['kind']==k for j in e['jobs']),'attained':sum(j['attained'] for j in e['jobs'] if j['kind']==k)} for k in ['short','background','long']})
            write(commit,row);write(attempts/(ident+f'_{attempt}.json'),{'complete':True,'seconds':row['seconds']})
            print(json.dumps(row),flush=True);del e
        except BaseException as exc:
            oom=isinstance(exc,pool.instances[0].torch.cuda.OutOfMemoryError)
            write(attempts/(ident+f'_{attempt}.json'),{'complete':False,'OOM':oom,'seconds':time.monotonic()-begin,'error':str(exc),'traceback':traceback.format_exc()})
            if oom:return 75
            raise
    verify();status('complete');write(root/'COMPLETE.json',{'complete':True,'committed':64,'GPU':2,'UTC':time.time()});return 0
if __name__=='__main__':
    try:sys.exit(main(sys.argv[1]))
    except Exception as exc:
        write(Path(sys.argv[1])/f'WORKER_FAILURE_{os.getpid()}.json',{'error':str(exc),'traceback':traceback.format_exc(),'UTC':time.time()})
        if 'out of memory' in str(exc).lower():sys.exit(75)
        raise
