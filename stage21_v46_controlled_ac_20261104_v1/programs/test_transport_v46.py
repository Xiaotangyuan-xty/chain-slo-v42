"""CPU socket tests of actual old/new clients, including no-retry and full trace."""
import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

STAGE=Path(__file__).resolve().parents[1]
PARENT=STAGE.parent.parent/'stage20_gpu2_lifetime_observer_resume_20261104_v2'

class HTTPFixture:
    def __init__(self,abort_reuse=False,abort_cid=None,idle_timeout=.1):
        self.abort_reuse=abort_reuse;self.abort_cid=abort_cid;self.idle_timeout=idle_timeout
        self.connections=0;self.received=[];self.handled=[];self.tasks=set();self.writers=set()
    async def start(self):
        self.server=await asyncio.start_server(self.accept,'127.0.0.1',0)
        self.port=self.server.sockets[0].getsockname()[1];return self
    async def stop(self):
        self.server.close();await self.server.wait_closed()
        for writer in list(self.writers):writer.close()
        for task in list(self.tasks):task.cancel()
        if self.tasks:await asyncio.gather(*list(self.tasks),return_exceptions=True)
    async def accept(self,reader,writer):
        self.tasks.add(asyncio.current_task());self.writers.add(writer)
        self.connections+=1;connection=self.connections;count=0
        try:
            while True:
                try:header=await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'),timeout=self.idle_timeout)
                except(asyncio.TimeoutError,asyncio.IncompleteReadError,ConnectionError):return
                fields={k.lower():v.strip()for k,v in(line.split(':',1)for line in header.decode().split('\r\n')[1:]if ':'in line)}
                payload=json.loads(await reader.readexactly(int(fields['content-length'])))
                count+=1;entry={'connection':connection,'number':count,'payload':payload};self.received.append(entry)
                if(self.abort_reuse and count>1)or payload['request_info']['collection_id']==self.abort_cid:
                    writer.transport.abort();return
                self.handled.append(entry);prompt=payload['request_info']['prompt'];n=payload['target_output_length']
                first=json.dumps({'text':[prompt+'x'],'ttft':.002,'tbt':[]}).encode()+b'\0'
                final=json.dumps({'text':[prompt+'x'*n],'ttft':.002,'tbt':[.001]*max(0,n-1),'service_gain':0}).encode()+b'\0'
                body=first+final
                close=fields.get('connection','').lower()=='close'
                response=('HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: '+str(len(body))+'\r\nConnection: '+('close'if close else'keep-alive')+'\r\n\r\n').encode()
                writer.write(response+first[:7]);await writer.drain();await asyncio.sleep(.001)
                writer.write(first[7:]+final);await writer.drain()
                if close:return
        except(ConnectionError,asyncio.IncompleteReadError):return
        finally:
            writer.close();self.writers.discard(writer);self.tasks.discard(asyncio.current_task())

def load(path,label):
    spec=importlib.util.spec_from_file_location(label,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def one_job(cid,offset=0.):
    prompt='CPU stable '+str(cid)
    return {'collection_id':cid,'request_type':0,'source':{'dataset':'CPU fixture'},'received_slo':[2.,.1,20.],
        'arrival_offset_seconds':offset,'stages':[[{'request_id':0,'state':0,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'declared_input_tokens':4,'target_output_tokens':8}]]}

class ConnectionCPU(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Match production asyncio.run(debug=False), avoiding unittest tracing overhead.
        asyncio.get_running_loop().set_debug(False)
    async def exercise(self,source,work,fixture,expect_success=True):
        fixture=await fixture.start()
        try:
            with tempfile.TemporaryDirectory(prefix='stage20-http-')as directory:
                path=Path(directory);out=path/'out';out.mkdir()
                protocol=json.loads((STAGE/'PROTOCOL.json').read_text());protocol['http_ports']={'A':fixture.port,'C':fixture.port}
                (path/'PROTOCOL.json').write_text(json.dumps(protocol));(path/'WORKLOAD.json').write_text(json.dumps(work))
                module=load(source,'client_'+str(time.time_ns()))
                error=None
                with patch.dict(os.environ,{'STUDY_ARM':protocol['planned_arms'][0]}):
                    try:await module.run(STAGE.parent,path,out)
                    except RuntimeError as e:error=str(e)
                calls=json.loads((out/'client_calls.json').read_text())
                done=(out/'CLIENT_DONE.json').exists()
                self.assertEqual(error is None,expect_success,(error,calls[:1]))
                self.assertEqual(done,expect_success)
                flat=[r for j in calls for s in j['stages']for r in s['requests']]
                return {'calls':calls,'flat':flat,'received':list(fixture.received),'handled':list(fixture.handled),'connections':fixture.connections,'error':error}
        finally:await fixture.stop()
    async def test_01_old_pooled_connection_failure_reproduced_new_client_passes(self):
        work={'warmup':[one_job(-1)],'main':[one_job(i,.03*i)for i in range(4)]}
        old=await self.exercise(PARENT/'programs/client.py',work,HTTPFixture(abort_reuse=True),False)
        self.assertTrue(any(not r['success']for r in old['flat']))
        new=await self.exercise(STAGE/'programs/client.py',work,HTTPFixture(abort_reuse=True))
        self.assertTrue(all(r['success']for r in new['flat']))
        self.assertEqual(len(new['received']),5);self.assertEqual(new['connections'],5)
        self.assertTrue(all(r['number']==1 for r in new['received']))
    async def test_02_connection_failure_propagates_once_without_retry(self):
        work={'warmup':[one_job(-1)],'main':[one_job(i,.03*i)for i in range(4)]}
        out=await self.exercise(STAGE/'programs/client.py',work,HTTPFixture(abort_cid=2),False)
        ids=Counter(r['payload']['request_info']['collection_id']for r in out['received'])
        self.assertEqual(ids,Counter({-1:1,0:1,1:1,2:1,3:1}))
        self.assertEqual([r['collection_id']for r in out['flat']if not r['success']],[2])
    async def test_03_actual_five_second_idle_close_boundary(self):
        work={'warmup':[one_job(-1)],'main':[one_job(0,0.),one_job(1,4.99),one_job(2,5.02),one_job(3,10.05)]}
        out=await self.exercise(STAGE/'programs/client.py',work,HTTPFixture(idle_timeout=5.))
        self.assertEqual(out['connections'],5);self.assertTrue(all(x['success']for x in out['flat']))
    async def test_04_full_frozen_request_contract_no_loss_or_duplicate(self):
        work=json.loads((STAGE/'WORKLOAD.json').read_text())
        # CPU-only compressed-arrival fixture; production WORKLOAD bytes never change.
        for row in work['main']:row['arrival_offset_seconds']*=24./3600.
        old=await self.exercise(PARENT/'programs/client.py',work,HTTPFixture(idle_timeout=5.))
        new=await self.exercise(STAGE/'programs/client.py',work,HTTPFixture(abort_reuse=True,idle_timeout=5.))
        def signatures(out):
            result={}
            for row in out['received']:
                payload=json.loads(json.dumps(row['payload']));info=payload['request_info']
                key=(payload['client_id'],info['collection_id'],info['stage_id'],info['request_id'])
                self.assertNotIn(key,result)
                payload.pop('served_time',None)
                result[key]=payload
            return result
        self.assertEqual(signatures(old),signatures(new))
        self.assertEqual(len(new['flat']),2408)
        self.assertEqual(new['connections'],2408)
        self.assertEqual(sum(r['phase']=='main'for r in new['calls']),1800)
        self.assertEqual(sum(r['phase']=='main'and r['request_type']==2 for r in new['calls']),120)
        self.assertTrue(all(r['success']for r in new['flat']))
        errors=[abs(r['arrival_offset_seconds']-r['planned_arrival_offset_seconds'])for r in new['calls']if r['phase']=='main']
        self.assertLessEqual(max(errors),.5)
        for row in new['flat']:
            self.assertEqual(len(row['server_tbt']),row['target_output_tokens']-1)
            self.assertEqual(row['generated_text'],'x'*row['target_output_tokens'])
        print(json.dumps({'CPU_full_trace_main_roots':1800,'CPU_calls_including_warmup':2408,'connections':new['connections'],
             'max_arrival_error_seconds':max(errors),'all_payload_contracts_match_old_client':True,'attempts_per_request':1,
             'arrival_window_seconds':24.,'production_arrival_window_unchanged':3600.}),flush=True)
    def test_05_source_delta_and_production_workload_bytes(self):
        old=(PARENT/'programs/client.py').read_text();new=(STAGE/'programs/client.py').read_text()
        self.assertEqual(new,old.replace('aiohttp.TCPConnector(limit=0)','aiohttp.TCPConnector(limit=0,force_close=True)').replace("'top_k':-1","'top_k':protocol['generation']['top_k']"))
        self.assertEqual((STAGE/'WORKLOAD.json').read_bytes(),(PARENT/'WORKLOAD.json').read_bytes())
        self.assertEqual((STAGE/'programs/lifetime_observation.py').read_bytes(),(PARENT/'programs/lifetime_observation.py').read_bytes())

if __name__=='__main__':unittest.main(verbosity=2)
