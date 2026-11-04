"""Capacity-accounted two-lane admission. Unsubmitted calls never occupy KV."""
import math

def blocks(member,block_size=16,horizon=50):return math.ceil((member['prompt']+min(member['max_tokens'],member['generated']+horizon))/block_size)

def choose(goals,capacity,max_seqs,block_size=16,protected=()):
    selected=set();charges={m['request']:blocks(m,block_size)for g in goals for m in g['members']};rows=[]
    used=0;limit=capacity
    protected=set(protected)
    ordered=sorted(goals,key=lambda g:(not g.get('supported',True),g['probability']<=0,-g['score'],g['deadline'],g['arrival']))
    for g in ordered:
        for m in sorted(g['members'],key=lambda v:v['arrival']):
            rid=m['request']
            if rid in protected and rid not in selected and used+charges[rid]<=limit and len(selected)<max_seqs:
                selected.add(rid);used+=charges[rid]
    positive=[g for g in ordered if g['probability']>0]
    for g in ordered:
        if positive and g['probability']<=0:continue
        pending=[m for m in g['members']if m['request']not in selected];need=sum(charges[m['request']]for m in pending)
        added=[]
        if used+need<=limit and len(selected)+len(pending)<=max_seqs:added=pending
        elif not selected:
            for m in pending:
                cost=charges[m['request']]
                if used+sum(charges[v['request']]for v in added)+cost<=limit and len(added)<max_seqs:added.append(m)
        for m in added:selected.add(m['request']);used+=charges[m['request']]
        rows.append({'chain':g['chain'],'probability':g['probability'],'remaining_work':g['remaining_work'],'score':g['score'],
          'supported':g.get('supported',True),'unsubmitted_members':g.get('estimated_unsubmitted_members',0),
          'selected':[m['request']for m in g['members']if m['request']in selected]})
    return selected,used,rows
