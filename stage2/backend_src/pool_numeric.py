"""Independent matched-shape batch witness and corruption negative control."""
def verify(pool):
    checks=[];negatives=[]
    for replica,b in enumerate(pool.instances):
        t=b.torch
        for B in [1,2,3,4]:
            b.reset(8192,8192);items=[]
            for j,n in enumerate([64,96,128,192][:B]):
                p=[100+j]*n;b.step(str(j),p,[]);items.append((str(j),p,[500+j]))
            for step in range(2):
                prior={k:tuple(tuple(x.clone() for x in pair) for pair in b.gpu[k]) for k,p,e in items}
                P=max(len(p)+len(e)-1 for k,p,e in items)+1
                got=b.step_batch(items)
                from transformers.cache_utils import DynamicCache
                for k,p,e in items:
                    n=len(p)+len(e)-1
                    dup=tuple(tuple(t.cat([t.nn.functional.pad(x,(0,0,0,P-n))]*B,0) for x in pair) for pair in prior[k])
                    with t.inference_mode():
                        ref=b.model.model(input_ids=t.tensor([[e[-1]]]*B,device='cuda:0'),
                            attention_mask=t.tensor([[1]*n+[0]*(P-n)+[1]]*B,device='cuda:0'),
                            position_ids=t.tensor([[n]]*B,device='cuda:0'),past_key_values=DynamicCache.from_legacy_cache(dup),use_cache=True,return_dict=True)
                        expected=b.model.lm_head(ref.last_hidden_state[:,-1,:])[0:1]
                    same=bool(t.allclose(got[k],expected,atol=.05,rtol=.01))
                    prefix=all(bool(t.equal(old,new[...,:n,:])) for before,after in zip(prior[k],b.gpu[k]) for old,new in zip(before,after))
                    checks.append({'replica':replica,'B':B,'step':step,'key':k,'logits_equal':same,'old_prefix_bitwise_preserved':prefix,
                        'max_abs_error':float((got[k]-expected).abs().max())})
                    if not same or not prefix:raise RuntimeError('batch numeric witness '+str(checks[-1]))
                    e.append(700+step)
        cache=b.gpu['0'];bad=tuple((k,v+1) for k,v in cache)
        _,normal,_,_=b._forward([800],cache);_,corrupt,_,_=b._forward([800],bad)
        rejected=not bool(t.allclose(normal,corrupt,atol=.05,rtol=.01));negatives.append(rejected)
        if not rejected:raise RuntimeError('corruption control undetected')
    return {'passed':True,'checks':checks,'corrupt_controls_rejected':negatives,'atol':.05,'rtol':.01,
        'reference':'independent duplicated-row matched B/physical padded length, prior KV only'}
