"""Causal state adapter to directly executed official GMAX priority/batch code."""
import collections,math,time,types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from official_loader import load_official
from pool_policies import features

@dataclass
class PublicSequence:
    prompt:int
    emitted:int=0
    def get_prompt_len(self):return self.prompt
    def get_output_len(self):return self.emitted
    def get_decode_len(self):return self.emitted

class OfficialGMAX:
    def __init__(self,qrf,graphs,clock,penalty=100,top_k=1,async_prediction=True):
        self.qrf=qrf;self.graphs=graphs;self.clock=clock
        self.async_prediction=async_prediction
        self.executor=ThreadPoolExecutor(max_workers=1) if async_prediction else None
        self.futures=[]
        self.raw=load_official(clock);self.groups={};self.timings=collections.Counter();self.calls=collections.Counter()
        self.policies=[self.raw.policy.SLOPolicy(schedule_interval=50,penalty_factor=penalty) for _ in range(2)]
        self.schedulers=[]
        for p in self.policies:
            s=object.__new__(self.raw.scheduler.SLOScheduler)
            s.policy=p;s.scheduler_config=types.SimpleNamespace(top_k_selection=top_k,max_num_seqs=4)
            s.max_batch_size=4;s.waiting=collections.deque();s.swapped=collections.deque();s.running=collections.deque()
            s._current_bin_queue=collections.deque();s._pending_swap_out_queue=collections.deque()
            self.schedulers.append(s)

    def prepare(self,g,release):
        """Only observed arrival/history; prediction and graph matching once/call.

        The released upstream server starts prediction/graph handling on arrival.
        Official priority frames consume the cached predicted TOTAL length.
        """
        c=g.ready
        if c.key in self.groups:return self.groups[c.key]
        observed_features=features(g)
        ri=types.SimpleNamespace(output_len=1024,prediction_ready=False,prediction_ready_time=None)
        def predict_once():
            started=time.perf_counter()
            try:
                estimated=max(1,int(self.qrf.predict([observed_features],quantiles=.95)[0]))
                ri.output_len=c.emitted+estimated;ri.prediction_ready=True;ri.prediction_ready_time=self.clock()
            except Exception:
                self.calls['prediction_failure_default_1024']+=1
            finally:
                self.timings['prediction_task_seconds']+=time.perf_counter()-started;self.calls['prediction']+=1
        begin=time.perf_counter()
        if self.executor:self.futures.append(self.executor.submit(predict_once))
        else:predict_once()
        self.timings['prediction_submit_or_block_seconds']+=time.perf_counter()-begin
        begin=time.perf_counter();best=None
        for graph in self.graphs:
            if len(graph['calls'])<=c.ordinal:continue
            score=((graph['calls'][c.ordinal]['prompt']-c.prompt_tokens)/256)**2
            for j,h in enumerate(g.history):
                if j>=len(graph['calls']):score+=1000;break
                score+=((graph['calls'][j]['outputs']-h.outputs)/64)**2
            if best is None or score<best[0]:best=(score,graph)
        share=best[1]['shares'][c.ordinal] if best else 1.
        self.timings['graph_seconds']+=time.perf_counter()-begin;self.calls['graph']+=1
        begin=time.perf_counter()
        # Same absolute stage deadline, expressed in the official call-arrival
        # coordinate: original server multiplies by stage share then subtracts
        # the already served chain time before creating the SequenceGroup.
        stage_budget=(g.deadline-g.arrival)*share-(release-g.arrival)
        seq=PublicSequence(c.prompt_tokens,c.emitted)
        sg=types.SimpleNamespace(request_id=c.key,collection_id=g.program,
            request_type=self.raw.info.RequestType.COLLECTIVE,request_info=ri,
            first_seq=seq,seqs=[seq],arrival_time=release,deadline=stage_budget,
            predict_output_length=ri.output_len,
            concord_metrics=types.SimpleNamespace(TTFT=None,service_gain=0.),
            observed_stage_share=share,observed_release=release,instance=None)
        self.groups[c.key]=sg
        self.timings['adapter_seconds']+=time.perf_counter()-begin
        return sg

    def assign(self,key,instance):
        sg=self.groups[key];sg.instance=instance
        # Upstream SequenceGroup.__init__ snapshots the prediction (line746).
        # Later upstream prediction callbacks update request_info only.
        sg.predict_output_length=sg.request_info.output_len
        self.policies[instance].add_seq_group(sg)

    def finish(self,key):
        sg=self.groups.pop(key);self.policies[sg.instance].delete_seq_group(sg)
        self.policies[sg.instance].seq_group_slo_dict.pop(key,None)

    def choose(self,view,instance,legal,previous):
        began=time.perf_counter();p=self.policies[instance];s=self.schedulers[instance]
        ordered=sorted((g.ready.key for g in view.goals if g.ready and g.ready.key in legal),key=lambda k:self.groups[k].observed_release)
        for g in view.goals:
            c=g.ready
            if c is not None and c.key in self.groups:
                self.groups[c.key].first_seq.emitted=c.emitted
        held=[k for k in previous if k in legal]
        s.running=collections.deque(self.groups[k] for k in held)
        s.waiting=collections.deque(self.groups[k] for k in ordered if k not in held);s.swapped=collections.deque()
        self.timings['adapter_seconds']+=time.perf_counter()-began
        core=time.perf_counter();due=p.update_schedule_count()
        if due:
            s._recompute_bins_sliding_window()
            selected=[sg.request_id for sg in s._current_bin_queue][:4]
            self.calls['gmax_frame']+=1
        else:
            # The official scheduler delegates intervening iterations to its
            # ordinary chunked-prefill engine; the shared backend retains and
            # fills the running batch under the same slot limit.
            selected=(held+[k for k in ordered if k not in held])[:4]
        elapsed=time.perf_counter()-core;self.timings['official_core_seconds']+=elapsed;self.calls['engine_tick']+=1
        audit={'official_frame':due,'official_core_seconds':elapsed,'official_schedule_count':p.num_schedule_count,
            'interval_time':p.interval_time,'priority_cache_entries':len(p.seq_group_slo_dict),
            'original_source_file':s._recompute_bins_sliding_window.__code__.co_filename,
            'original_source_line':s._recompute_bins_sliding_window.__code__.co_firstlineno,
            'selected':selected,'upstream_cached_total_length':{k:self.groups[k].request_info.output_len for k in selected},
            'upstream_sequence_prediction_snapshot':{k:self.groups[k].predict_output_length for k in selected},
            'prediction_ready':{k:self.groups[k].request_info.prediction_ready for k in selected},
            'upstream_stage_deadline':{k:self.groups[k].arrival_time+self.groups[k].deadline for k in selected}}
        return selected,audit

    def wait_predictions(self):
        for f in self.futures:f.result()

    def snapshot(self):
        if self.executor:self.executor.shutdown(wait=True)
        return {'timings':dict(self.timings),'counts':dict(self.calls),'async_prediction':self.async_prediction,
        'upstream_commit':self.raw.manifest['commit'],'algorithm_functions_modified':False,
        'scope':'official priority/frame/sliding-window source executed directly; common native GPU backend and metadata predictor adapter'}
