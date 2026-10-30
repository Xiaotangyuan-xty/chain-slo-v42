"""Authorized stage0/DEV only; no TEST inputs or outcomes are generated."""
import hashlib,random
from slo_design import workload as v1_workload
from contracts import BETAS,digest

ARMS=tuple(f'beta_{b:g}' for b in BETAS)+('EDF','slack_first','deadline_LTR','full_V2','no_conformal','no_lookahead','fixed_h4','V1_representation')
def seed(split,index):return int.from_bytes(hashlib.sha256(f'idea3_slo_v2/{split}/{index}'.encode()).digest()[:8],'big')
FIT=tuple(seed('FIT',i) for i in range(128))
CAL=tuple(seed('CAL',i) for i in range(64))
DEV=tuple(seed('DEV',i) for i in range(32))

def workload(s,d,condition='normal'):
    inp=v1_workload(s,d);inp['schema']='away-timer-v2-dev-input/v1';inp['condition']=condition
    if condition not in ('normal','tight'):raise ValueError('condition')
    if condition=='tight':
        for p in inp['programs']:
            if p['kind']=='short':p['slo_budget']*=.5;p['deadline']=p['arrival']+p['slo_budget']
    return inp

def jobs():
    out=[]
    for i,s in enumerate(DEV):
        conditions=['normal','tight'] if i<8 else ['normal']
        random.Random(seed('DEV-condition-order',i)).shuffle(conditions)
        for c in conditions:
            arms=list(ARMS);random.Random(seed('DEV-arm-order-'+c,i)).shuffle(arms)
            out.extend((s,c,a) for a in arms)
    assert len(out)==520
    return out

def plan():
    return {'schema':'away-timer-v2-authorized-dev/v1','authorization':'current user stages 0 and 1 only',
      'change':'explicit overlapping AWAY timers and all-arrived-unfinished-goal reference; no credit or bound-tightening change',
      'arms':ARMS,'FIT':FIT,'CAL':CAL,'DEV':DEV,'TEST_generated':False,'namespace_index_origin':0,
      'conditions':{'normal_seeds':32,'tight_preselected_first_seeds':8,'tight_short_budget_multiplier':.5},
      'episodes':520,'primary_purpose':'DEV only; no confirmatory labels',
      'CPU':{'max_core_seconds':28800,'new_episodes':0,'fixed_old_multi_action_states':64,'FIT_cluster_bootstraps':32,
        'bootstrap_calibration':'same fixed fresh CAL, same formula recalibrated per FIT bootstrap; both representations share each snapshot',
        'state_selection':'stratified by old current/future AWAY and READY count; SHA ordering, no V2 score/outcome selection'},
      'predictor':{'alpha':.1,'scale':'max(16,mean/4)','unit':'whole workload, READY current/chain and supported continuous AWAY chain phases',
         'AWAY_support':'empirical strict survival T>elapsed, conditional on observed completed output length, prompt bucket and ordinal',
         'calibration_enumeration':'zero, every FIT duration breakpoint on both sides and true CAL return left-limit; unsupported intervals registered',
         'no_JCT_coverage_claim':True},
      'reference':{'future_policy':'fixed EDF of modeled ready compute phases','timer_prediction':'deterministic conditional FIT mean',
        'no_timer_GPU_billing':True,'same_reference_all_actions':True,'current_AWAY_in_goal_denominator':True,
        'future_unknown_arrivals':'not supplied','future_guard_simulation':'same fluid EDF proxy as V1; actual engine enforces common guards each tick',
        'zero_AWAY':'shared exact V1 kernel for bitwise degenerate equivalence'},
      'execution':{'q_decode_equivalents':[64,128,256,512],'tick_outputs':4,'hmax_decode_equivalents':64,
        'GPU_KV_tokens':2048,'host_KV_tokens':2048,'cache':'identical common LRU/prefetch','guard':'identical common true-history guard',
        'selector_timeout_seconds':.02,'nonpreemptible_overshoot':'charged in full'},
      'GPU':{'only':2,'supervision_window_seconds':21600,'automatic_busy_retry':False,'foreign_PID':'stop only verified own worker'},
      'profile':'reuse byte-identical deployment/backend profile; new GPU native/speed sentinel does not change d, deadlines or prediction snapshot',
      'DEV_analysis':['all arms attainment/JCT and G_sbeta','old V1 DEV endpoint comparison, distributions differ, no pooled inference',
        'adoption/no_model_gain/identical-goodput/all-zero rates','SD(G), whole-seed bootstrap power for N<=320','simple-envelope DEV selection'],
      'family_for_future_power_only':16,'alpha_for_future_power_only':.05,'automations':False,'old_results_labels_immutable':True}
