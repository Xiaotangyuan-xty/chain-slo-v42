"""CPU golden comparison on a supplied frozen200-root shape, not GPU evidence."""
from copy import copy
import hashlib,json,sys
from pathlib import Path
from unittest.mock import patch

from mixed_joint import MixedScheduler as Original
from mixed_max import MixedScheduler as Candidate
from test_pure_core_restore import PureCoreTraceTests
from vllm.config import CacheConfig,SchedulerConfig
from vllm.sequence import SequenceStatus


def run(work, blocks):
    assert len(work['main'])==200 and all(j['request_type']==2 for j in work['main'])
    timer=[1000.]
    with patch('time.perf_counter',side_effect=lambda:timer[0]),patch('time.time',side_effect=lambda:timer[0]-900.):
        config=SchedulerConfig('generate',2048,32,8192,enable_chunked_prefill=True,
            policy='jitserve',top_k_selection=3,max_swaps_per_iter=3)
        def cache():
            c=CacheConfig(16,.85,1,'auto');c.num_gpu_blocks=blocks;c.num_cpu_blocks=512;return c
        schedulers=[Original(copy(config),cache(),None),Candidate(copy(config),cache(),None)]
        groups=[{},{}];active={};arrived=0;framehash=hashlib.sha256();preemptions=0
        fixture=PureCoreTraceTests()
        def add(index,job,sid):
            widths=list(map(len,job['stages']));s=schedulers[index]
            for member,r in enumerate(job['stages'][sid]):
                rid=f"{job['collection_id']}:{sid}:{member}"
                g=fixture.group(rid,2,prompt=r['prompt_tokens'],cid=job['collection_id'],stage=sid,width=widths[sid],deadline=1000.+job['arrival_offset_seconds']+job['received_slo'][2]*len(widths))
                g.request_info.request_id=member;g.request_info.chain_stage_count=len(widths);g.request_info.chain_stage_widths=widths;g.request_info.output_len=192
                s.add_seq_group(g);groups[index][rid]=g
        for tick in range(40000):
            timer[0]=1000.+tick*.05
            while arrived<200 and work['main'][arrived]['arrival_offset_seconds']<=timer[0]-1000.:
                j=work['main'][arrived];active[j['collection_id']]=(j,0)
                for index in range(2):add(index,j,0)
                arrived+=1
            frames=[]
            for index,s in enumerate(schedulers):
                out=fixture.step(s)
                frames.append({'selected':sorted(s.selected)if s.selected is not None else None,
                    'leases':s.leases,'draining':s.policy.draining,'last_plan':s.last_plan,
                    'running':[g.request_id for g in s.running],'waiting':[g.request_id for g in s.waiting],
                    'swapped':[g.request_id for g in s.swapped],
                    'outputs':[(x.seq_group.request_id,x.token_chunk_size,x.seq_group.first_seq.get_output_len())for x in out.scheduled_seq_groups],
                    'preempted':out.preempted,'swap_in':len(out.blocks_to_swap_in),'swap_out':len(out.blocks_to_swap_out),
                    'free':(s.block_manager.get_num_free_gpu_blocks(),s.block_manager.get_num_free_cpu_blocks()),
                    'max_seqs':s.scheduler_config.max_num_seqs})
                for item in out.scheduled_seq_groups:
                    g=item.seq_group;target=3+sum(g.request_id.encode())%5
                    if g.first_seq.get_output_len()>=target:g.first_seq.status=SequenceStatus.FINISHED_LENGTH_CAPPED
                s.free_finished_seq_groups()
            assert frames[0]==frames[1],(blocks,tick,frames)
            framehash.update(json.dumps(frames[0],sort_keys=True).encode());preemptions+=frames[0]['preempted']
            for cid,(j,sid)in list(active.items()):
                ids=[f'{cid}:{sid}:{i}'for i in range(len(j['stages'][sid]))]
                done=[all(groups[k][rid].is_finished()for rid in ids)for k in range(2)]
                assert done[0]==done[1]
                if done[0]:
                    if sid+1<len(j['stages']):
                        for k in range(2):add(k,j,sid+1)
                        active[cid]=(j,sid+1)
                    else:del active[cid]
            if arrived==200 and not active:break
        assert arrived==200 and not active
        expected=sum(len(rs)for j in work['main']for rs in j['stages'])
        assert all(len(gs)==expected and all(g.is_finished()for g in gs.values())for gs in groups)
        for s in schedulers:
            assert s.block_manager.get_num_free_gpu_blocks()==blocks and s.block_manager.get_num_free_cpu_blocks()==512
        assert schedulers[1].max_counts['mixed_plans']==0
        return {'GPU_blocks':blocks,'roots':200,'calls':expected,'frames_equal':tick+1,
            'preemptions':preemptions,'trace_sha256':framehash.hexdigest(),'all_KV_free':True}


if __name__=='__main__':
    work=json.loads(Path(sys.argv[1]).read_text())
    results=[run(work,blocks)for blocks in [1024,11357]]
    print(json.dumps({'gate':'PASS','actual_frozen200_root_shapes':True,
        'simulated_equal_predictions_and_short_outputs_not_performance_measurement':True,
        'profiles':results,'GPU_used':False}))
