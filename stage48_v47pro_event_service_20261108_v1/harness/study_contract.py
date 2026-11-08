"""Pure launch-matrix contract; no filesystem/device/process operations."""
from pathlib import Path

METHODS=('v47pro','A','Astar')
VERSION='v47pro_event_wave_service_v1'
VARIANT='v47pro_v47_reward_event_wave_v1'


def case_specs(study,gpu):
    if gpu not in [0,1]:raise ValueError('Only GPU0/GPU1 are authorized')
    S=Path(study)/f'gpu{gpu}'
    return [{'name':f'{pos:02d}_llama8_mixed_{method}_gpu{gpu}','GPU':gpu,
             'version':method,'model_key':'llama8','position':pos,'order':1,
             'stage':str(S/'cases'/f'{pos:02d}_llama8_mixed_{method}_gpu{gpu}'),
             'arm':'C'if method=='v47pro'else'A','attempt':f'{pos:02d}_LLAMA8_MIXED_{method.upper()}_GPU{gpu}_01',
             'http_port':30200+gpu*10+pos,'prediction_port':64500+gpu*10+pos,
             'gpu_blocks':10176,'KV_GiB':19.875,'root_rate':.5,'main_roots':1800,'main_calls':2527,'workload_kind':'mixed'}
            for pos,method in enumerate(METHODS,1)]


def validate_matrix(matrix,gpu):
    cases=matrix['cases']
    if matrix['GPU']!=gpu or len(cases)!=3 or [c['version']for c in cases]!=list(METHODS):raise ValueError('Wrong case/method matrix')
    if any(c['GPU']!=gpu or c['model_key']!='llama8'or c['gpu_blocks']!=10176 or c['KV_GiB']!=19.875 for c in cases):raise ValueError('Model/resource/device binding mismatch')
    if len({c['stage']for c in cases})!=3 or len({c['attempt']for c in cases})!=3:raise ValueError('Duplicate case identities')
    return True
