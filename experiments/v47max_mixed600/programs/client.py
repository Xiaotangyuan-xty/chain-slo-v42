"""One frozen global arrival stream, shared by A/C; finite mixed calibration."""
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def dump(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(q,p)

class StreamParser:
    def __init__(self,prompt):self.prompt=prompt;self.buffer=b'';self.last=None;self.first_text=None;self.messages=0
    def feed(self,chunk,stamp):
        frames=(self.buffer+chunk).split(b'\0');self.buffer=frames.pop()
        for frame in frames:
            if not frame.strip():continue
            self.last=frame;self.messages+=1
            if self.first_text is None:
                data=json.loads(frame);text=data.get('text',[''])[0]
                if not text.startswith(self.prompt):raise ValueError('Response prompt prefix differs')
                if text[len(self.prompt):]:self.first_text=stamp
    def finish(self):
        if self.buffer.strip():raise ValueError('Truncated stream frame')
        if self.last is None:raise ValueError('Empty stream')
        return json.loads(self.last)

async def run(root,stage,out):
    import aiohttp
    from client_recording import CallJSONLWriter
    protocol=json.loads((stage/'PROTOCOL.json').read_text());work=json.loads((stage/'WORKLOAD.json').read_text())
    arm=os.environ['STUDY_ARM'];url='http://127.0.0.1:'+str(protocol['http_ports'][arm])+'/generate'
    writer=CallJSONLWriter(out/'client_calls.jsonl');calls=[]
    timeout=aiohttp.ClientTimeout(total=protocol['http_request_timeout_seconds'])
    connector=aiohttp.TCPConnector(limit=0,force_close=True)
    async with aiohttp.ClientSession(timeout=timeout,connector=connector)as session:
        async def job(job,phase,origin=None):
            started=time.perf_counter();stages=[]
            for sid,requests in enumerate(job['stages']):
                async def request(r):
                    from stage_replay import payload_for
                    st=time.perf_counter();payload=payload_for(job,sid,r,phase);payload['sampling_params']['top_k']=protocol['generation']['top_k']
                    if job['request_type']==2:payload['served_time']=st-started
                    record={'phase':phase,'collection_id':job['collection_id'],'stage_id':sid,'request_id':r['request_id'],'request_type':job['request_type'],
                            'success':False,'target_output_tokens':r['target_output_tokens'],'input_prompt_sha256':r['prompt_sha256'],'received_slo':job['received_slo'],
                            'client_request_start_monotonic':st}
                    try:
                        parser=StreamParser(r['prompt'])
                        async with session.post(url,json=payload)as response:
                            if response.status!=200:raise RuntimeError('HTTP '+str(response.status)+' '+(await response.text())[:1000])
                            async for chunk in response.content.iter_any():parser.feed(chunk,time.perf_counter())
                        data=parser.finish();record.update(success=True,server_ttft=data.get('ttft'),server_tbt=data.get('tbt',[]),logged_service_gain=data.get('service_gain',0),
                            client_first_text_monotonic=parser.first_text,stream_messages=parser.messages,generated_text=data.get('text',[''])[0][len(r['prompt']):])
                    except Exception as e:record['error']=repr(e)
                    record['client_request_end_monotonic']=time.perf_counter()
                    return record
                result=await asyncio.gather(*(request(r)for r in requests))
                stages.append({'stage_id':sid,'success':all(r['success']for r in result),'requests':result})
                if not stages[-1]['success']:break
            end=time.perf_counter();summary={'phase':phase,'collection_id':job['collection_id'],'request_type':job['request_type'],'application':job.get('application','dr'),'source':job['source'],
                'received_slo':job['received_slo'],'success':len(stages)==len(job['stages'])and all(s['success']for s in stages),
                'client_chain_start_monotonic':started,'client_chain_end_monotonic':end,'arrival_offset_seconds':None if origin is None else started-origin,
                'planned_arrival_offset_seconds':job.get('arrival_offset_seconds')if phase=='main'else None,'stages':stages}
            calls.append(summary);writer.record(summary)
            return summary
        try:
            for warm in work['warmup']:
                if not(await job(warm,'warmup'))['success']:raise RuntimeError('Warmup failed')
            from native_graph_observation import qualify_warmup
            dump(out/'WARMUP_NATIVE_TOT_GATE.json',qualify_warmup(out,work,protocol,calls))
            origin=time.perf_counter();dump(out/'MAIN_START.json',{'origin_monotonic':origin,'jobs':len(work['main']),'window_seconds':protocol.get('observation_horizon_seconds'),'last_planned_arrival_seconds':work['main'][-1]['arrival_offset_seconds']})
            tasks=[]
            for item in work['main']:
                await asyncio.sleep(max(0.,origin+item['arrival_offset_seconds']-time.perf_counter()))
                tasks.append(asyncio.create_task(job(item,'main',origin)))
            results=await asyncio.gather(*tasks)
            if not all(r['success']for r in results):raise RuntimeError('At least one main job failed')
            last_end=max(r['client_chain_end_monotonic']for r in results)
            await asyncio.sleep(max(0.,origin+(protocol.get('observation_horizon_seconds')or 0.)-time.perf_counter()))
            dump(out/'CLIENT_DONE.json',{'complete':True,'main_jobs':len(results),'main_end_monotonic':last_end,'observation_end_monotonic':time.perf_counter(),'horizon_seconds':protocol.get('observation_horizon_seconds')})
        finally:
            writer.finish();dump(out/'client_calls.json',calls)

if __name__=='__main__':asyncio.run(run(*map(Path,sys.argv[1:4])))
