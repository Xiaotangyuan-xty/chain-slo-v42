"""v47 reward density with explicit first-wave repricing and no oversized claims."""
from service_model import member_blocks


def blocks(member, block_size=16, horizon=50):
    return member_blocks(member,block_size,horizon)


def row(goal, chosen, charge):
    names=['chain','stage','deadline','probability','remaining_work','critical_mean_seconds','score',
           'expected_credited_tokens','stage_waves_mean']
    return {**{k:goal.get(k) for k in names},'reservation_blocks':charge,
            'members':[m['request'] for m in goal['members']], 'selected':sorted(chosen),
            'probability_recomputed_for_selected_wave':True}


def _members(goals,residents,background):
    members={m['request']:m for g in goals for m in g['members']}
    for m in list(residents)+list(background):members[m['request']]=m
    return members


def _revalue(forecaster,goal,chosen,selected,members,residents,background):
    own={m['request'] for m in goal['members']}
    actual=set(selected)|set(chosen)
    actual.update(m['request'] for m in list(residents)+list(background))
    first=own&actual
    other=[members[r] for r in sorted(actual-own)]
    return forecaster.revalue(goal,first,other)


def reprice(goals,selected,protected,forecaster,block_size=16,residents=(),background=()):
    if forecaster is None:return selected,None
    selected=set(selected or());protected=set(protected)
    members=_members(goals,residents,background)
    while True:
        values={};remove=set()
        for goal in goals:
            own={m['request'] for m in goal['members']};chosen=own&selected
            if chosen:
                valued=_revalue(forecaster,goal,chosen,selected,members,residents,background)
                if valued['probability']<=0 and not chosen&protected:remove.update(chosen)
            else:
                valued={**goal,'probability':0.,'score':0.,'expected_credited_tokens':0.}
            values[goal['chain']]=valued
        if not remove:break
        selected-=remove
    rows=[row(values[g['chain']],{m['request'] for m in g['members']}&selected,
              sum(blocks(m,block_size) for m in g['members'])) for g in goals]
    if not any(v['probability']>0 and v['expected_credited_tokens']>0 for v in values.values()):
        return None,rows
    return (selected or None),rows


def select(goals,capacity,max_seqs,block_size=16,protected=(),*,forecaster=None,background=(),residents=(),final_reprice=True):
    members=_members(goals,residents,background)
    selected=set(protected)&members.keys()
    used=sum(blocks(members[r],block_size) for r in selected)
    rows=[]
    for goal in sorted(goals,key=lambda g:(-g['score'],g['deadline'],g['arrival'])):
        own={m['request'] for m in goal['members']}
        chosen=own&selected
        charge=0
        for member in sorted(goal['members'],key=lambda m:m['arrival']):
            rid=member['request'];b=blocks(member,block_size)
            if rid not in selected and used+charge+b<=capacity and len(selected)+len(chosen-selected)<max_seqs:
                chosen.add(rid);charge+=b
        if forecaster and chosen:
            valued=_revalue(forecaster,goal,chosen,selected,members,residents,background)
        else:valued=goal
        # Empty first wave cannot earn whole-root credit. Required work is
        # executed through the explicit native drain fallback, without that claim.
        if chosen and valued['probability']>0:
            selected.update(chosen);used+=charge
        else:
            chosen=own&selected
            if not chosen:valued={**valued,'probability':0.,'score':0.,'expected_credited_tokens':0.}
        rows.append(row(valued,chosen,sum(blocks(m,block_size) for m in goal['members'])))
    if forecaster and final_reprice:
        return reprice(goals,selected,protected,forecaster,block_size,residents,background)
    if not selected:return None,rows
    return selected,rows
