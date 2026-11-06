"""One reward per predicted timely chain, with short-horizon native KV packing."""
import math

def blocks(member,block_size=16,horizon=50):
    future=min(member['max_tokens'],member['generated']+horizon)
    return math.ceil((member['prompt']+future)/block_size)

def select(goals,capacity,max_seqs,block_size=16,protected=()):
    charges={m['request']:blocks(m,block_size) for g in goals for m in g['members']}
    positive=[g for g in goals if g['probability']>0]
    def record(g,chosen,need):
        return {'chain':g['chain'],'stage':g['stage'],'deadline':g['deadline'],'probability':g['probability'],
                'remaining_work':g['remaining_work'],'critical_mean_seconds':g['critical_mean_seconds'],'score':g['score'],
                'reservation_blocks':need,'members':[m['request'] for m in g['members']],'selected':[m['request'] for m in chosen]}
    negative=[record(g,[],sum(charges[m['request']] for m in g['members'])) for g in goals if g['probability']<=0]
    if not positive:return None,negative
    ordered=sorted(positive,key=lambda g:(-g['score'],g['deadline'],g['arrival']))
    # Existing execution leases consume the same slots and KV budget as new work.
    # Native pressure can still preempt when already-resident work exceeds capacity.
    protected=set(protected)
    selected={m['request'] for g in goals for m in g['members'] if m['request'] in protected}
    used=sum(charges[m['request']] for g in goals for m in g['members'] if m['request'] in selected)
    rows=[]
    for goal in ordered:
        members=sorted(goal['members'],key=lambda x:x['arrival']);need=sum(charges[m['request']] for m in members)
        chosen=[m for m in members if m['request'] in selected]
        pending=[m for m in members if m['request'] not in selected]
        incremental=sum(charges[m['request']] for m in pending)
        if used+incremental<=capacity and len(selected)+len(pending)<=max_seqs:chosen+=pending;used+=incremental
        elif not selected:
            for member in pending:
                b=charges[member['request']]
                if (not chosen or used+b<=capacity) and len(chosen)<max_seqs:chosen.append(member);used+=b
        selected.update(m['request'] for m in chosen)
        rows.append({'chain':goal['chain'],'stage':goal['stage'],'deadline':goal['deadline'],'probability':goal['probability'],
                     'remaining_work':goal['remaining_work'],'critical_mean_seconds':goal['critical_mean_seconds'],'score':goal['score'],
                     'reservation_blocks':need,'members':[m['request'] for m in members],'selected':[m['request'] for m in chosen]})
    return selected,rows+negative
