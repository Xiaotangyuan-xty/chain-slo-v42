"""Deployment contracts, baseline observer branches and real nested-path metrics."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,json,sys,tempfile,unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1];H=PACKAGE/'harness'
sys.path.insert(0,str(H))
from protocol_paths import experiment_root
import native_graph_observation as graph


class Harness(unittest.TestCase):
    def test_all_harness_modules_parse_and_do_not_infer_root_from_parent_depth(self):
        for p in H.glob('*.py'):
            ast.parse(p.read_text(),str(p))
            self.assertNotIn('stage.parents[2]',p.read_text(),p.name)

    def test_actual_server_snapshot_handles_baseline_without_cost_model(self):
        for file in ['a_server.py','c_server.py']:
            fn=next(n for n in ast.walk(ast.parse((H/file).read_text()))if isinstance(n,ast.FunctionDef)and n.name=='snapshot')
            for candidate in [False,True]:
                scheduler=NS(cost_model=NS(stats={'accepted_samples':14}))if candidate else NS()
                ns={'server':NS(engine=NS(engine=NS(model_executor=NS(_v49pro_observed=True),scheduler=[scheduler]))),
                    'runtime_receipt':{'execution_observer':{'events':14}}}
                exec(compile(ast.Module(body=[fn],type_ignores=[]),file,'exec'),ns)
                r=ns['snapshot']();self.assertTrue(r['executor_hooked']);self.assertEqual(r['cost_accepted_samples'],14 if candidate else 0)
                self.assertEqual(r['candidate_cost_models_present'],candidate)

    def test_warmup_observer_and_budget_receipt_contract_all_three_methods(self):
        for method in ['v50','A','Astar']:
            with self.subTest(method=method),tempfile.TemporaryDirectory()as directory:
                p=Path(directory);arm='C'if method=='v50'else'A'
                protocol={'candidate_version':method,'workload_kind':'mixed','expected_resource_caps':{},'model_path':'model',
                    'resources':{'physical_gpu_index':0,'gpu_uuid':'GPU0'},'C_version':'v50_anytime_stage_window_credit',
                    'predictor_protocol':'length_framed_pickle_v1','prediction_ports':{arm:12345},'runtime_arm':arm}
                work={'warmup':[{'application':app,'stages':[[{}]]}for app in ['latency','deadline','dr','tot_style']]}
                calls=[{'phase':'warmup','success':True,'stages':[{'requests':[{'success':True,'server_tbt':[.01],'target_output_tokens':2}]}]}for _ in range(4)]
                write=lambda name,data:(p/name).write_text(json.dumps(data))
                write('common_events.jsonl',{})
                events=[{'kind':'engine_ready','model_path':'model'},{'kind':'device_identity','physical_gpu_index':0,'actual_uuid':'GPU0'}]
                (p/'common_events.jsonl').write_text('\n'.join(json.dumps(r)for r in events)+'\n')
                write('LIVE_EXECUTOR_OBSERVATION.json',{'executor_hooked':True,'execution_observer':{'version':'executor_host_boundary_v1','events':14,'failed_calls':0},'cost_accepted_samples':14 if method=='v50'else 0,'candidate_cost_models_present':method=='v50'})
                write('SHARED_PREDICTION_RUNTIME.json',{'version':'shared_framed_dynamic_prediction_v1','protocol':'length_framed_pickle_v1','method':method,'port':12345})
                (p/'SHARED_PREDICTION_EVENTS.jsonl').write_text('\n'.join(json.dumps({'kind':'prediction_success'})for _ in range(4))+'\n')
                if method=='v50':
                    write('V50_RUNTIME.json',{'version':protocol['C_version'],'prediction_comparison_ready':True});write('V50_BUDGET_ADAPTER.json',{})
                else:write('COMPARISON_RUNTIME.json',{'method':method,'candidate_scheduler_installed':False})
                if method=='Astar':write('BASELINE_BUDGET_ADAPTER.json',{})
                with patch.object(graph,'route_checks',return_value=({'native_graph_route':True},[])):
                    result=graph.qualify_warmup(p,work,protocol,calls)
                self.assertEqual(result['gate'],'PASS',result)

    def test_explicit_root_is_required_and_protocol_mismatch_fails_early(self):
        with tempfile.TemporaryDirectory()as d:
            C=Path(d)/'study/gpu0/cases/case';C.mkdir(parents=True)
            (C/'PROTOCOL.json').write_text(json.dumps({'experiment_root':str(Path(d)/'right')}))
            with self.assertRaisesRegex(ValueError,'Explicit'):experiment_root(C,None)
            with self.assertRaisesRegex(ValueError,'mismatch'):experiment_root(C,Path(d)/'wrong')

    def test_full_native_metric_on_completed_nested_GPU_case_without_writes(self):
        # Uses *old* accepted raw data only to qualify the calculation path.
        # It is not a v50 GPU result and supplies no labels to the scheduler.
        R=Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1')
        C=R/'stage42_v49pro_llama8_nativeToT_mixed_gpu01_20261107_v1/gpu0/cases/01_llama8_mixed_v49pro_gpu0'
        if not C.exists():self.skipTest('Native read-only dataset available on CPU qualification host')
        A=next((C/'attempts').iterdir());work=json.loads((C/'WORKLOAD.json').read_text());calls=json.loads((A/'client_calls.json').read_text())
        origin=json.loads((A/'MAIN_START.json').read_text())['origin_monotonic'];duration=max(c['client_chain_end_monotonic']for c in calls if c['phase']=='main')-origin
        before={p.name:(p.stat().st_size,p.stat().st_mtime_ns)for p in A.iterdir()if p.is_file()}
        from alignment_metrics import analyze
        result=analyze(R,C,A.name,work,calls,duration,emit_outputs=False)
        self.assertTrue(all(result['checks'].values()),result['checks'])
        self.assertEqual(result['metrics']['fixed_hour_credited_weighted_tokens'],3174614)
        self.assertEqual(result['metrics']['jitserve_credited_weighted_tokens'],3215763)
        self.assertEqual(before,{p.name:(p.stat().st_size,p.stat().st_mtime_ns)for p in A.iterdir()if p.is_file()})
        with self.assertRaisesRegex(ValueError,'lacks native metric'):analyze(C.parents[2],C,A.name,work,calls,duration,emit_outputs=False)


if __name__=='__main__':unittest.main(verbosity=2)
