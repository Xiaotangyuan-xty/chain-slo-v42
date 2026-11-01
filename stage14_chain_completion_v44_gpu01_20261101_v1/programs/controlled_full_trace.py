"""Check full-trace evaluation changes without allocating any GPU."""
import ast
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest

from abc_observation import install_observation
from client import CallJSONLWriter, dump, make_optimized_single_request, transport_call
from gpu0_platform import adapt_tree as adapt_gpu0_tree
from gpu1_platform import adapt_tree as adapt_gpu1_tree
from offline_compare import endpoint_metrics

STAGE = Path(__file__).resolve().parents[1]


class FakeRequestOutput:
    def __init__(self):
        self.request_output = ''
        self.request_ttft = 0.0
        self.request_tbt = []
        self.request_service_gain = 0
        self.stop_reason = ''
        self.success = False
        self.error = ''


def official_parse(buffer, is_stream, request_prompt):
    output = FakeRequestOutput()
    generated_text = ''
    if is_stream:
        while '\0' in buffer:
            part, buffer = buffer.split('\0', 1)
            data = json.loads(part)
            output.request_tbt = data.get('tbt', [])
            output.request_service_gain = data.get('service_gain', 0)
            output.stop_reason = data.get('finish_reason', '')
            generated_text = data.get('text', [''])[0]
            if output.request_ttft == 0.0 and data.get('ttft') is not None:
                output.request_ttft = data['ttft']
    else:
        data = json.loads(buffer)
        output.request_tbt = data.get('tbt', [])
        output.request_service_gain = data.get('service_gain', 0)
        output.stop_reason = data.get('finish_reason', '')
        generated_text = data.get('text', [''])[0]
        if output.request_ttft == 0.0 and data.get('ttft') is not None:
            output.request_ttft = data['ttft']
    output.request_output = generated_text[len(request_prompt):]
    return output


def fake_client_module(chunks):
    sessions = []

    class Content:
        async def iter_any(self):
            for chunk in chunks:
                yield chunk

    class Response:
        status = 200

        def __init__(self):
            self.content = Content()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    class Session:
        def __init__(self, timeout):
            self.timeout = timeout
            self.posts = []
            sessions.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, url, json):
            self.posts.append((url, json))
            return Response()

    class Timeout:
        def __init__(self, total):
            self.total = total

    module = S(
        RequestOutput=FakeRequestOutput,
        get_prompt_format_func=lambda model_name: lambda prompt: prompt,
        aiohttp=S(ClientTimeout=Timeout, ClientSession=Session),
    )
    return module, sessions


def fake_request_info():
    request = S(
        prompt='PROMPT:', request_type=S(value=7), collection_id=3, stage_id=2,
        request_id=1, state='queued', output_len=5,
    )
    request.to_dict = lambda: {'prompt': request.prompt, 'request_marker': 'preserved'}
    return S(
        request=request, api_url='http://127.0.0.1/generate', slo_constraint=(2., .1, 20.),
        sampling_params=S(to_dict=lambda: {'seed': 42, 'temperature': 0}),
        client_id=9, num_stages=2, requests_per_stage=[1, 2], accumulate_stage_ratio=.5,
    )


class FullTraceChecks(unittest.TestCase):
    def test_complete_trace_and_original_arrival_prefix(self):
        p = json.loads((STAGE / 'PROTOCOL.json').read_text())
        m = json.loads((STAGE / 'TRACE_MANIFEST.json').read_text())
        self.assertEqual(m['chain_count'], 200)
        self.assertEqual(len(m['expected_requests']), 1197)
        self.assertEqual(m['main_target_output_tokens'], 504695)
        self.assertEqual(len(p['target_arrivals']), 200)
        self.assertEqual(set(p['stage_counts']), {str(i) for i in range(200)})
        self.assertLessEqual(m['max_actual_context'], 8192)
        self.assertFalse(m['invalid_contexts'])
        for actual, expected in zip(p['target_arrivals'], [.988,2.010,2.973,3.982,5.017,5.984]):
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertTrue(all(b > a for a,b in zip(p['target_arrivals'],p['target_arrivals'][1:])))

    def test_http_timeout_does_not_change_slo_or_served_time(self):
        def send(request_info, model_name, client_deadline=200, is_stream=True, served_time=0.):
            return request_info, model_name, client_deadline, is_stream, served_time
        info = S(slo_constraint=(2.,.1,20.))
        bound = transport_call(send, info, ('model',1200,True,62.), {}, 27000)
        result = send(*bound.args, **bound.kwargs)
        self.assertIs(result[0], info)
        self.assertEqual(result[2], 27000)
        self.assertEqual(result[4], 62.)
        self.assertEqual(info.slo_constraint, (2.,.1,20.))

    def test_received_deadline_and_failed_chain_metrics(self):
        c = dict(phase='main', collection_id=199, success=True,
                 client_chain_start_monotonic=100.,client_chain_end_monotonic=579.,
                 arrival_offset_seconds=1.,stages=[{'requests':[{'received_slo':[8.,.4,80.]}]}]*6)
        self.assertEqual(endpoint_metrics([c],{'199':6})['official_scaled_attained'],1)
        c['client_chain_end_monotonic']=581.
        self.assertEqual(endpoint_metrics([c],{'199':6})['official_scaled_attained'],0)
        c.update(success=False,client_chain_end_monotonic=101.)
        self.assertEqual(endpoint_metrics([c],{'199':6})['official_scaled_attained'],0)

    def test_output_ids_recorded_without_changing_outputs(self):
        output=S(finished=True,prompt_token_ids=[3,4],outputs=[S(token_ids=[7,8],finish_reason='stop',stop_reason=151645)])
        class Engine:
            async def generate(self,*args,**kwargs):
                yield output
        with tempfile.TemporaryDirectory() as d:
            obs=install_observation(S(),Engine,Path(d),'A',Path(d),{'physical_gpu_index':0})
            async def consume():
                return [x async for x in Engine().generate('p',S(collection_id=0,stage_id=0,request_id=0,slo_constraint=(2.,.1,20.)),S(),'id')]
            result=asyncio.run(consume())
            self.assertIs(result[0],output)
            obs.finalize()
            row=json.loads((Path(d)/'engine_token_ids.jsonl').read_text())
            self.assertEqual(row['prompt_token_ids'],[3,4])
            self.assertEqual(row['output_token_ids'],[[7,8]])
            self.assertEqual(row['output_token_ids_sha256'],hashlib.sha256(json.dumps([[7,8]]).encode()).hexdigest())

    def test_gpu0_and_gpu1_adapters_only_change_device_functions(self):
        source='''import os
def device_id_to_physical_device_id(device_id):
    return 99
def warn_if_different_devices():
    pass
class Scheduler:
    def choose(self):
        return 123
'''
        for physical,adapter in [(0,adapt_gpu0_tree),(1,adapt_gpu1_tree)]:
            with self.subTest(physical=physical):
                tree,changed=adapter(source,physical,f'GPU-{physical}')
                self.assertEqual(ast.dump(ast.parse(source).body[-1]),ast.dump(tree.body[-1]))
                self.assertEqual(sorted(changed),['device_id_to_physical_device_id','warn_if_different_devices'])
        controller=(STAGE/'programs/controller.py').read_text()
        self.assertIn('"nvidia-smi", "-i", physical',controller)
        self.assertIn('env["STUDY_GPU"] = str(physical)',controller)
        self.assertIn('device["authorization_file"]',controller)
        self.assertIn('if arm == "C":',controller)
        self.assertIn('previous.get("gate") != "PASS"',controller)
        self.assertNotIn('final_gpu0',controller)

    def test_arm_authorizations_launchers_and_sequential_order(self):
        protocol=json.loads((STAGE/'PROTOCOL.json').read_text())
        expected={
            'A':(0,'GPU-ff8c7d5c-f75c-26e1-43de-b40ca283f938',[1,2]),
            'C':(1,'GPU-f5874940-7f3b-ad3e-b810-bc7dda423d1a',[0,2]),
        }
        self.assertEqual(protocol['attempts'],{
            'compat':'COMPAT_C_V44_GPU1_01','A':'FULL_TRACE_A_GPU0_V44_01','C':'FULL_TRACE_C_V44_GPU1_01'})
        for arm,(physical,gpu_uuid,forbidden) in expected.items():
            with self.subTest(arm=arm):
                mapping=protocol['arm_devices'][arm]
                self.assertEqual(mapping['physical_gpu_index'],physical)
                self.assertEqual(mapping['gpu_uuid'],gpu_uuid)
                authorization=json.loads((STAGE/mapping['authorization_file']).read_text())
                self.assertEqual(authorization['schema'],'gpu-execution-authorization/stage14')
                self.assertEqual(authorization['physical_gpu_index'],physical)
                self.assertEqual(authorization['gpu_uuid'],gpu_uuid)
                self.assertEqual(authorization['forbidden_gpu_indices'],forbidden)
                self.assertEqual(authorization['max_attempt_wall_seconds'],28800)
                self.assertEqual(authorization['predictor_cold_start_seconds'],900)
                self.assertEqual(authorization['http_request_timeout_seconds'],27000)
                self.assertTrue(authorization['preflight_requires_no_compute_process_and_zero_MiB'])
                self.assertTrue(authorization['only_registered_processes_may_be_signaled'])
        self.assertIn('from gpu0_platform import install',(STAGE/'programs/a_server.py').read_text())
        c_server=(STAGE/'programs/c_server.py').read_text()
        self.assertIn('from gpu1_platform import install',c_server)
        self.assertNotIn('from gpu0_platform import install',c_server)
        runner=(STAGE/'programs/run_experiments.py').read_text()
        self.assertIn("[('compat', 'C'), ('full', 'A'), ('full', 'C')]",runner)
        offline=(STAGE/'programs/offline_compare.py').read_text()
        self.assertIn("check('only_authorized_physical_gpu'",offline)
        self.assertIn('max(errors) <= .5',offline)

    def test_incremental_call_jsonl_matches_final_json(self):
        calls=[
            {'phase':'warmup','collection_id':0,'success':True,'stages':[{'requests':[{'request_id':0}]}]},
            {'phase':'main','collection_id':1,'success':True,'arrival_offset_seconds':2.01,'stages':[]},
            {'phase':'main','collection_id':2,'exception':'RuntimeError: injected','traceback':'preserved'},
        ]
        with tempfile.TemporaryDirectory() as d:
            jsonl=Path(d)/'client_calls.jsonl'
            final=Path(d)/'client_calls.json'
            writer=CallJSONLWriter(jsonl)
            writer.record(calls[0])
            writer.drain()
            self.assertFalse(final.exists())
            self.assertEqual([json.loads(line) for line in jsonl.read_text().splitlines()],calls[:1])
            for call in calls[1:]:
                writer.record(call)
            writer.finish()
            writer.finish()
            streamed=[json.loads(line) for line in jsonl.read_text().splitlines()]
            self.assertEqual(streamed,calls)
            dump(final,calls)
            self.assertEqual(json.loads(final.read_text()),streamed)
            completed=jsonl.read_text()
            with self.assertRaises(RuntimeError):
                writer.record({'phase':'main','collection_id':3})
            self.assertEqual(jsonl.read_text(),completed)

    def test_optimized_response_matches_official_stream_and_nonstream(self):
        prompt='PROMPT:'
        stream_parts=[
            {'text':[''],'ttft':.125,'tbt':[],'service_gain':0,'finish_reason':''},
            {'text':['PROMPT:AB'],'tbt':[.1,.2],'service_gain':2.5,'finish_reason':'stop'},
        ]
        ignored={'text':['PROMPT:IGNORED'],'ttft':9,'tbt':[9],'service_gain':9,'finish_reason':'length'}
        stream_buffer='\0'.join(json.dumps(part) for part in stream_parts)+'\0'+json.dumps(ignored)
        nonstream_buffer=json.dumps({'text':['PROMPT:XYZ'],'ttft':.5,'tbt':[.4],
                                     'service_gain':3.5,'finish_reason':'length'})
        for is_stream,buffer in [(True,stream_buffer),(False,nonstream_buffer)]:
            with self.subTest(is_stream=is_stream):
                chunks=[buffer[:13].encode(),b'  ',buffer[13:41].encode(),b'\n',
                        buffer[41:97].encode(),buffer[97:].encode()]
                module,sessions=fake_client_module(chunks)
                optimized=make_optimized_single_request(module)
                info=fake_request_info()
                actual=asyncio.run(optimized(info,'model',200,is_stream,62.))
                expected=official_parse(buffer,is_stream,prompt)
                self.assertTrue(actual.success,actual.error)
                for field in ['request_output','request_ttft','request_tbt','request_service_gain','stop_reason']:
                    with self.subTest(field=field):
                        self.assertEqual(getattr(actual,field),getattr(expected,field))
                self.assertEqual(actual.request_input,prompt)
                self.assertEqual((actual.request_type,actual.collection_id,actual.stage_id,actual.request_id),(7,3,2,1))
                self.assertEqual(actual.state,'queued')
                self.assertEqual(len(sessions),1)

    def test_optimized_single_request_transport_signature_and_payload(self):
        body=json.dumps({'text':['PROMPT:Z'],'ttft':.25,'tbt':[.1],
                         'service_gain':1.5,'finish_reason':'stop'})+'\0'
        module,sessions=fake_client_module([body[:17].encode(),body[17:].encode()])
        optimized=make_optimized_single_request(module)
        info=fake_request_info()
        bound=transport_call(optimized,info,('model',1200,True,62.),{},27000)
        self.assertEqual(bound.arguments['client_deadline'],27000)
        self.assertEqual(bound.arguments['served_time'],62.)
        result=asyncio.run(optimized(*bound.args,**bound.kwargs))
        self.assertTrue(result.success,result.error)
        self.assertEqual(result.request_output,'Z')
        self.assertEqual(len(sessions),1)
        session=sessions[0]
        self.assertEqual(session.timeout.total,27000)
        self.assertEqual(len(session.posts),1)
        url,payload=session.posts[0]
        self.assertEqual(url,'http://127.0.0.1/generate')
        self.assertEqual(payload,{
            'request_info':{'prompt':'PROMPT:','request_marker':'preserved'},
            'slo_constraint':(2.,.1,20.),
            'sampling_params':{'seed':42,'temperature':0},
            'client_id':9,'stream':True,'target_output_length':5,
            'num_stages':2,'requests_per_stage':[1,2],
            'accumulate_stage_ratio':.5,'served_time':62.,
        })
        self.assertEqual(info.request.prompt,'PROMPT:')
        self.assertEqual(info.slo_constraint,(2.,.1,20.))

    def test_writer_errors_are_recorded_and_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            existing=Path(d)/'existing.jsonl'
            existing.write_text('keep\n')
            with self.assertRaises(RuntimeError):
                CallJSONLWriter(existing)
            self.assertEqual(existing.read_text(),'keep\n')
            jsonl=Path(d)/'client_calls.jsonl'
            writer=CallJSONLWriter(jsonl)
            good={'phase':'warmup','collection_id':0,'success':True}
            writer.record(good)
            writer.drain()
            writer.record({'phase':'main','invalid':float('nan')})
            with self.assertRaises(RuntimeError):
                writer.drain()
            self.assertIsNotNone(writer.error)
            self.assertEqual(writer.error['type'],'ValueError')
            with self.assertRaises(RuntimeError):
                writer.finish()
            with self.assertRaises(RuntimeError):
                writer.finish()
            self.assertEqual([json.loads(line) for line in jsonl.read_text().splitlines()],[good])


if __name__=='__main__':
    unittest.main(verbosity=2)
