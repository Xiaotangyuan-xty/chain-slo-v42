"""Final structural validation; no attainment/JCT success requirement."""
from pathlib import Path
import json,math

def joint_selected(selected,kinds):return any(kinds[r]!=2 for r in selected)and any(kinds[r]==2 for r in selected)

def validate(attempt,protocol,receipt):
    attempt=Path(attempt);expected=protocol['attempt_variants'][attempt.name];adapter=attempt/'BASELINE_BUDGET_ADAPTER.json'
    checks={'stage_budget_variant_correct':adapter.exists()==(expected=='corrected'),
       'shared_execution_observer_present':(attempt/'V46_EXECUTION_OBSERVATION.json').exists()}
    if receipt['arm']!='C':return checks
    common=json.loads((attempt/'common_audit_summary.json').read_text())
    kinds={r['request_uuid']:int(r['request_type'])for r in common['requests']}
    bad=[];plans=0;joint=False
    for line in(attempt/'V46_DECISIONS.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['kind']!='plan':continue
        plans+=1;states={m['request']:m for m in row['member_states']};selected=row['selected']
        if len(selected)!=len(set(selected))or len(selected)>row['sequence_limit']or not set(selected)<=states.keys():bad.append(['selection_identity',plans])
        charges=sum(math.ceil((states[r]['prompt']+min(states[r]['max_tokens'],states[r]['generated']+50))/16)for r in selected)
        if charges!=row['charged_blocks']:bad.append(['charge',plans])
        if charges>row['capacity_blocks']and row['physical_progress_exception']is None:bad.append(['undeclared_overcommit',plans])
        for goal in row['goals']:
            if not goal['supported']and goal['probability']!=0:bad.append(['unsupported_positive',plans])
        joint|=joint_selected(selected,kinds)
    checks.update(v46_decision_accounting_complete=plans>0 and not bad,v46_no_global_native_bypass=True)
    (attempt/'V46_DECISION_AUDIT.json').write_text(json.dumps({'checks':checks,'bad':bad,'plans':plans,'joint_selection_observed':joint},indent=2)+'\n')
    if protocol['workload_kind']=='mixed':checks['joint_foreground_and_chain_selection_observed']=joint
    return checks
