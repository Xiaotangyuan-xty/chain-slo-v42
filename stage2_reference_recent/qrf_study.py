"""Single QRF runtime change; common observation of prediction lifecycle."""
import gc,time
from frozen_membership import ARMS as HISTORICAL,wrapper,default_h4
from bridge_policies import ARMS as BRIDGES,make_policy

ARMS=('A_GMAX_threads4','A_GMAX_threads1','D_default_h4','F_shadow_threads1')

class MeasuredPredictor:
    def __init__(self,model):self.model=model;self.records=[]
    def reset(self):self.records=[]
    def predict(self,rows,quantiles=.95):
        started=time.perf_counter();cpu=time.process_time();thread=time.thread_time()
        before=sum(v['collections'] for v in gc.get_stats())
        result=self.model.predict(rows,quantiles=quantiles)
        self.records.append({'features':[list(x) for x in rows],'quantiles':quantiles,
                             'prediction':[float(x) for x in result],
                             'wall_seconds':time.perf_counter()-started,
                             'thread_CPU_seconds':time.thread_time()-thread,
                             'process_CPU_during_prediction_seconds':time.process_time()-cpu,
                             'process_GC_collections_during_prediction':sum(v['collections'] for v in gc.get_stats())-before,
                             'n_jobs':self.model.n_jobs})
        return result

    def validate(self,reference,expected_calls):
        assert len(self.records)==expected_calls
        for row in self.records:
            assert row['quantiles']==.95
            for feat,value in zip(row['features'],row['prediction']):assert reference[tuple(feat)]==value
        return {'passed':True,'calls':len(self.records),'n_jobs':self.model.n_jobs,'exact_reference_predictions':True,
                'wall_seconds':sum(r['wall_seconds'] for r in self.records),
                'thread_CPU_seconds':sum(r['thread_CPU_seconds'] for r in self.records),
                'process_CPU_during_prediction_seconds':sum(r['process_CPU_during_prediction_seconds'] for r in self.records),
                'process_GC_collections_during_prediction':sum(r['process_GC_collections_during_prediction'] for r in self.records)}

def lifecycle_base(original):
    class LifecycleObserved(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.lifecycle=[]
        def assign(self,key,instance):
            sg=self.groups[key];ri=sg.request_info
            entered=self.clock();ready_before=ri.prediction_ready
            super().assign(key,instance)
            self.lifecycle.append(({'key':key,'instance':instance,'assign_enter_time':entered,'assign_return_time':self.clock(),
                                    'snapshot_at_assign':sg.predict_output_length,
                                    'prediction_ready_before_assign':ready_before,
                                    'prediction_ready_after_assign':ri.prediction_ready,
                                    'current_prediction_after_assign':ri.output_len},ri))
        def snapshot(self):
            out=super().snapshot()
            rows=[{**r,'prediction_ready_time':ri.prediction_ready_time,'prediction_current_final':ri.output_len,
                   'prediction_ready_final':ri.prediction_ready} for r,ri in self.lifecycle]
            out['prediction_lifecycle']=rows
            out['prediction_lifecycle_counts']={'calls':len(rows),
                'ready_before_assign':sum(r['prediction_ready_before_assign'] for r in rows),
                'ready_after_assign':sum(r['prediction_ready_after_assign'] for r in rows),
                'snapshot_1024':sum(r['snapshot_at_assign']==1024 for r in rows)}
            return out
    return LifecycleObserved

def policy_class(original,arm,decode):
    if arm==ARMS[2]:return original
    base=lifecycle_base(original)
    if arm in ARMS[:2]:return wrapper(base,HISTORICAL[0])
    assert arm==ARMS[3]
    return make_policy(base,BRIDGES[3],decode)

def validate_original_gmax_trace(e):
    release={f'{c["program"]}:{c["ordinal"]}':c['release'] for c in e['call_metrics']}
    previous=[[],[]];frames=0
    for t,b in zip(e['trace'],e['batches']):
        a=t['audit'];i=t['instance'];legal=set(t['legal']);raw=a['raw_official_selection']
        assert a['official_frame']==(a['official_schedule_count']%50==0)
        assert a['membership_selection']==raw
        if not a['official_frame']:
            ordered=sorted((g['ready']['key'] for g in t['view']['goals'] if g['ready'] and g['ready']['key'] in legal),key=lambda k:release[k])
            held=[k for k in previous[i] if k in legal]
            assert raw==(held+[k for k in ordered if k not in held])[:4]
        forced=t['forced'];actual=([forced]+raw)[:4] if forced and forced not in raw else raw
        assert actual==t['selected']==b['keys'];previous[i]=actual;frames+=bool(a['official_frame'])
    return {'passed':True,'original_frame_and_interframe_semantics':True,'ticks':len(e['trace']),'frames':frames}
