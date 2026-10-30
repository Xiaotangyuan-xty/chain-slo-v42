"""New independently frozen SLO study; old objectives are not relabeled."""
import random
from contracts import BETAS

FIT=tuple(range(8161001,8161129))
CAL=tuple(range(8162001,8162065))
DEV=(8163001,8163002)
TEST=tuple(range(8164001,8164013))
ARMS=tuple(f'beta_{b:g}' for b in BETAS)+('slo_candidate','slo_laxity')

def workload(seed,d):
    rng=random.Random(seed);ps=[];total=0
    kinds=['long']*4+['short']*4+['background']*4;rng.shuffle(kinds)
    for kind in kinds:
        if kind=='long':
            outs=[rng.choice((16,32)),rng.choice((48,64,80)),rng.choice((16,32))]
            base=rng.choice((192,256));add=64
        elif kind=='background':
            outs=[rng.choice((8,16,24)) for _ in range(rng.choice((3,4,5)))];base=128;add=32
        else:outs=[rng.choice((8,16,24))];base=128;add=0
        calls=[];prompt=base
        for i,out in enumerate(outs):
            away=rng.uniform(8,24)*d if i+1<len(outs) else 0.
            calls.append({'ordinal':i,'prompt_tokens':prompt,'outputs':out,'away':away,
                          'output_token_seed':rng.getrandbits(32)})
            prompt+=out+add;total+=out
        # Requested latency budget is independent of actual hidden length/type.
        budget=rng.choice((128,256,512))*d
        ps.append({'id':f'{rng.getrandbits(96):024x}','kind':kind,'calls':calls,
                   'arrival':0.,'token_seed':rng.getrandbits(32),'slo_budget':budget,'weight':1.})
    horizon=total*d/.9
    for p,a in zip(ps,sorted(rng.uniform(0,horizon) for _ in ps)):
        p['arrival']=a;p['deadline']=a+p['slo_budget']
    return {'schema':'chain-slo-fixed-replay/v1','seed':seed,'programs':ps,'total_decode':total,
            'arrival_horizon_seconds':horizon,'decode_anchor_seconds':d}

def plan():
    return {'schema':'chain-slo-gpu2/v1','betas':list(BETAS),'arms':ARMS,
      'primary':'equal-weight complete-chain deadline attainment; all late chains fully completed',
      'success':{'relative_attainment_gain':.10,'weighted_mean_JCT_ratio_max':1.05,
                 'against':'every one of five beta arms, simultaneous one-sided family=10 alpha=.05',
                 'point_and_CI_both_required':True},
      'inference':'paired episode/seed contrasts g_candidate-1.1*g_beta; paired JCT ratios; Student t, exploratory synthetic-domain inference',
      'baseline_selection':'DEV best goodput then mean JCT; all five TEST comparisons reported; no TEST winner used as a frozen comparator',
      'candidate':'group-split conformal workload-unit intrinsic-work upper estimates; one finite action then common EDF fluid reference; no balance',
      'diagnostic':'same-information least-laxity feasible-first, finite h=4 outputs; candidate ablation not external baseline family',
      'information':{'visible':['opaque ID','arrival','requested deadline','weight=1','current prompt size','ordinal','emitted progress',
          'actual past wait/service','READY/RUNNING/AWAY','cache occupancy'],
          'forbidden':['hidden future calls','true remaining length','future tool return time','role labels','token seeds']},
      'conformal':{'unit':'entire independent workload seed; max residual across all chains and token progress states',
          'alpha':.1,'scope':'intrinsic outputs plus modeled necessary prefill plus AWAY, not wall-clock JCT or SLO guarantee',
          'caveat':'marginal exchangeability under fixed synthetic generator; cache restoration and timing drift outside coverage'},
      'fit':FIT,'calibration':CAL,'dev':DEV,'test':TEST,'jobs_per_seed':12,
      'deadline_rule':'arrival + UniformChoice(128,256,512)*frozen deployment decode_seconds, independent of hidden length/type',
      'arrival_rule':'uniform over total decode seconds/.9; exogenous, identical across arms; not measured total utilization',
      'execution':{'model':'Qwen2.5-7B-Instruct BF16 SDPA','single_active':True,'teacher_forced_fixed_history':True,
          'quantum_decode_equivalents':[64,128,256,512],'tick_outputs':4,'hcap_outputs':64,
          'gpu_KV_quota':2048,'host_KV_quota':2048,'cache':'unchanged common LRU with common no-eviction prefetch',
          'fairness':'unchanged common virtual lead 2q, oldest wait 8q, continuous 8q; overshoot charged'},
      'budgets':{'GPU_active_wall_seconds':10800,'CPU_preparation_hours':2,'data_bytes':5*1024**3,
          'resource_busy_retries':0,'automatic_followup':False},
      'stages':{'technical':'recheck native correctness and profile','dev_episodes':14,'test_episodes':84,
          'test_gate':'correctness, profile support and projected remaining budget; no DEV performance tuning'},
      'limitations':['synthetic deadlines and fixed replay','no production batching','no natural Agent quality',
          'fluid lookahead compresses future call and AWAY structure; must beat diagnostic to attribute lookahead contribution'],
      'old_results_failures_immutable':True,'GPU_allowed':[2],'automation':False}
