"""v47 coexistence target, with exact selected-wave root repricing."""
from chain_admission import select,blocks


def select_joint(goals,foreground,capacity,max_seqs,block_size=16,protected=(),*,forecaster=None):
    all_members={m['request']:m for g in goals for m in g['members']}
    all_members.update({m['request']:m for m in foreground})
    protected=set(protected)&all_members.keys()
    chain_protected=protected&{m['request'] for g in goals for m in g['members']}
    fg_selected=protected-chain_protected
    used=sum(blocks(all_members[r],block_size) for r in protected)
    target=min(capacity//2,max(0,capacity-used));target_slots=min(max(1,max_seqs//2),max(0,max_seqs-len(protected)))
    borrowed=0;added=0
    for m in foreground:
        b=blocks(m,block_size);rid=m['request']
        if rid not in protected and borrowed+b<=target and added<target_slots:
            fg_selected.add(rid);borrowed+=b;added+=1
    if not fg_selected and max_seqs>len(protected):
        for m in foreground:
            if used+blocks(m,block_size)<=capacity:
                fg_selected.add(m['request']);break
    fg=[all_members[r] for r in fg_selected]
    fg_cost=sum(blocks(m,block_size) for m in fg)
    chains,rows=select(goals,capacity-fg_cost,max_seqs-len(fg),block_size,chain_protected,
                       forecaster=forecaster,background=())
    # Reprice against the whole resource pool including fixed foreground.
    # select() uses the reduced pool, so its waves already reserve those slots
    # and KV. Service estimates additionally include foreground batch/context.
    if forecaster:
        final=[]
        chosen=set(chains or ())
        for g in goals:
            own={m['request'] for m in g['members']};first=own&chosen
            if not first:continue
            # The resource pool is restored for this call because background is
            # now explicit. No resource is subtracted twice.
            valued=forecaster.revalue(g,first,[all_members[r] for r in chosen-own]+fg)
            if valued['probability']<=0 and not(first&chain_protected):chosen-=own
            final.append(valued)
        chains=chosen
        from chain_admission import row
        by={g['chain']:g for g in final}
        rows=[row(by.get(g['chain'],g) if set(chains)&{m['request'] for m in g['members']} else {**g,'probability':0.,'expected_credited_tokens':0.,'score':0.},set(chains)&{m['request'] for m in g['members']},sum(blocks(m,block_size) for m in g['members'])) for g in goals]
    if not chains:
        # As in v47, zero predicted opportunity is not expiry: preserve full
        # required execution and let native allocation make forward progress.
        return None,rows,fg_selected
    return set(chains)|fg_selected,rows,fg_selected
