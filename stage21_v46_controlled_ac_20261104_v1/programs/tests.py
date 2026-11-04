"""CPU qualification of native fallback, frozen-chain equivalence and transport."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

class MixedCPU(unittest.TestCase):
    def scheduler(self,kind,cpu=32):
        from controlled_native import DummyModel
        import native_chain
        from mixed_fallback import MixedScheduler
        from jitserve.scheduler.slo_scheduler import SLOScheduler
        from vllm.config import CacheConfig,SchedulerConfig
        native_chain.MODEL=DummyModel()
        cfg=SchedulerConfig('generate',64,32,512,enable_chunked_prefill=True,policy='jitserve',penalty_factor=100,top_k_selection=3,max_swaps_per_iter=3)
        cache=CacheConfig(16,.9,1,'auto');cache.num_gpu_blocks=20;cache.num_cpu_blocks=cpu
        cls={'mixed':MixedScheduler,'chain':native_chain.NativeChainScheduler,'native':SLOScheduler}[kind]
        return cls(cfg,cache,None)
    def group(self,rid,kind):
        from jitserve.request_info import RequestInfo,RequestType
        from vllm.sequence import Sequence,SequenceGroup
        from vllm import SamplingParams
        info=RequestInfo(RequestType(kind),(2.,.1,20.),0,int(rid),120,160,64)
        if kind==2:info.chain_deadline=2000.;info.chain_stage_count=1;info.chain_stage_widths=[1]
        seq=Sequence(int(rid),inputs={'prompt_token_ids':list(range(160)),'prompt':'CPU'},block_size=16)
        return SequenceGroup(rid,[seq],100.,info,SamplingParams(temperature=0,max_tokens=256))
    def simulate(self,kind,types,abort_at=None,cpu=32):
        from controlled_native import NativeRuntime
        fixture=NativeRuntime();s=self.scheduler(kind,cpu);rows=[]
        for rid,t in enumerate(types,1):s.add_seq_group(self.group(str(rid),t))
        for tick in range(120):
            if abort_at is not None and tick==abort_at:s.abort_seq_group('2')
            out=fixture.step(s)
            rows.append(([x.seq_group.request_id for x in out.scheduled_seq_groups],list(out.blocks_to_swap_in),list(out.blocks_to_swap_out),
                         s.block_manager.get_num_free_gpu_blocks(),s.block_manager.get_num_free_cpu_blocks(),
                         [(g.request_id,g.first_seq.status.name,g.first_seq.get_output_len())for g in list(s.running)+list(s.waiting)+list(s.swapped)]))
        for i in range(1,len(types)+1):s.abort_seq_group(str(i))
        self.assertEqual(s.block_manager.get_num_free_gpu_blocks(),20)
        self.assertEqual(s.block_manager.get_num_free_cpu_blocks(),cpu)
        return rows,s
    def test_foreground_only_matches_actual_native_scheduler(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            a,_=self.simulate('native',[0,1]);c,s=self.simulate('mixed',[0,1])
        self.assertEqual(a,c);self.assertEqual(s.policy.chains,{})
    def test_native_fallback_matches_native_with_foreground_and_chain(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            a,_=self.simulate('native',[2,0]);c,s=self.simulate('mixed',[2,0])
        self.assertEqual(a,c);self.assertGreater(s.mixed_counts['main_mixed_native_calls'],0)
    def test_compound_only_matches_frozen_v45(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            for cpu in [1,32]:
                a,_=self.simulate('chain',[2,2],cpu=cpu);c,s=self.simulate('mixed',[2,2],cpu=cpu)
                self.assertEqual(a,c);self.assertEqual(s.mixed_counts['main_native_fallback_calls'],0)
    def test_native_to_chain_transition_preserves_state_and_kv(self):
        with patch('time.time',return_value=100.),patch('time.perf_counter',return_value=1000.):
            _,s=self.simulate('mixed',[2,0],abort_at=20)
        self.assertGreater(s.mixed_counts['main_native_fallback_calls'],0);self.assertGreater(s.mixed_counts['main_v45_chain_calls'],0)
        self.assertNotIn(2,s.policy.chains)
    def test_stream_parser_handles_chunk_boundaries_and_final_metrics(self):
        from client import StreamParser
        p=StreamParser('PROMPT')
        data=json.dumps({'text':['PROMPTx'],'ttft':.2,'tbt':[]}).encode()+b'\0'
        p.feed(data[:7],1.);self.assertIsNone(p.first_text);p.feed(data[7:],2.)
        end=json.dumps({'text':['PROMPTxy'],'ttft':.2,'tbt':[.1],'service_gain':5}).encode()+b'\0'
        p.feed(end,3.);self.assertEqual(p.first_text,2.);self.assertEqual(p.finish()['tbt'],[.1])
        q=StreamParser('P');q.feed(b'{',1.)
        with self.assertRaises(ValueError):q.finish()
    def test_long_protocol_scope(self):
        stage=Path(__file__).resolve().parents[1]
        p=json.loads((stage/'PROTOCOL.json').read_text());w=json.loads((stage/'WORKLOAD.json').read_text())
        self.assertEqual(p['study_mode'],'long_execution_bound')
        self.assertEqual(len(w['main']),1800)
        self.assertEqual(sum(j['request_type']==2 for j in w['main']),120)
        self.assertEqual(p['mixed_metrics']['arrival_gate_seconds'],.5)
        self.assertFalse(p['mixed_metrics']['JCT_constraint'])
        self.assertEqual(w['main'][-1]['arrival_offset_seconds'],3600.)
        self.assertTrue((stage/'LONG_EXECUTION_APPROVAL.json').exists())
        self.assertEqual({d['physical_gpu_index']for d in p['arm_devices'].values()},{2})
        self.assertEqual(p['forbidden_gpu_indices'],[0, 1])
        self.assertEqual(__import__('hashlib').sha256((stage/'WORKLOAD.json').read_bytes()).hexdigest(),p['workload_sha256'])
        for n,h in json.loads((stage/'FROZEN_V45_CORE.json').read_text())['files'].items():
            self.assertEqual(__import__('hashlib').sha256((stage/n).read_bytes()).hexdigest(),h)

if __name__=='__main__':unittest.main(verbosity=2)
