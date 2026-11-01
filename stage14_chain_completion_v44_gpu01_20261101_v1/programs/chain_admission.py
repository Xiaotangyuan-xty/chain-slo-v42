"""One reward per predicted timely chain, with short-horizon native KV packing."""
import math

def blocks(member,block_size=16,horizon=50):
    future=min(member['max_tokens'],member['generated']+horizon)
    return math.ceil((member['prompt']+future)/block_size)

def select(goals,capacity,max_seqs,block_size=16):
    positive=[g for g in goals if g['probability']>0]
    def record(g,chosen,need):
        return {'chain':g['chain'],'stage':g['stage'],'deadline':g['deadline'],'probability':g['probability'],
                'remaining_work':g['remaining_work'],'critical_mean_seconds':g['critical_mean_seconds'],'score':g['score'],
                'reservation_blocks':need,'members':[m['request'] for m in g['members']],'selected':[m['request'] for m in chosen]}
    negative=[record(g,[],sum(blocks(m,block_size) for m in g['members'])) for g in goals if g['probability']<=0]
    if not positive:return None,negative
    ordered=sorted(positive,key=lambda g:(-g['score'],g['deadline'],g['arrival']))
    selected=set();used=0;rows=[]
    for goal in ordered:
        members=sorted(goal['members'],key=lambda x:x['arrival']);need=sum(blocks(m,block_size) for m in members);chosen=[]
        if used+need<=capacity and len(selected)+len(members)<=max_seqs:chosen=members;used+=need
        elif not selected:
            for member in members:
                b=blocks(member,block_size)
                if (not chosen or used+b<=capacity) and len(chosen)<max_seqs:chosen.append(member);used+=b
        selected.update(m['request'] for m in chosen)
        rows.append({'chain':goal['chain'],'stage':goal['stage'],'deadline':goal['deadline'],'probability':goal['probability'],
                     'remaining_work':goal['remaining_work'],'critical_mean_seconds':goal['critical_mean_seconds'],'score':goal['score'],
                     'reservation_blocks':need,'members':[m['request'] for m in members],'selected':[m['request'] for m in chosen]})
    return selected,rows+negative
