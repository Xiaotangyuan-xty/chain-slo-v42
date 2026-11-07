"""Reserved-calibration leave-one-graph-out length-proxy diagnostic; no GPU."""
from copy import deepcopy
import hashlib,json,statistics,sys
from pathlib import Path
from chain_forecast import StageForecaster


def main(output):
    stage=Path(__file__).resolve().parents[1]
    bank=json.loads((stage/'assets/TOT_CALIBRATION.json').read_text())
    old=json.loads((stage/'assets/CHAIN_MODEL.json').read_text());results={}
    for key,entry in bank['models'].items():
        records=entry['model']['records'];rows=[]
        for target in records:
            model=deepcopy(entry['model']);model['records']=[r for r in records if r['id']!=target['id']]
            assert target['id']not in {r['id']for r in model['records']}
            views=[{'request':str(i),'member':i,'declared_input':n,'prompt':n,'generated':0,'max_tokens':256,'prefill_tokens':n,'reward_input':n,'arrival':0.}for i,n in enumerate(target['stages'][0]['inputs'])]
            def estimate(data):
                meta={'observed':{},'neighbor_cache':{},'widths':[3,1,3,1],'stages':4,'future_max_tokens':256}
                return StageForecaster(data).estimate(meta,0,views,1e9,.04)
            new=estimate(model);baseline=estimate(old)
            truth=sum(sum(s['outputs'])+sum(s['inputs'])/256. for s in target['stages'])
            rows.append({'held_out_graph':target['id'],'donors':new['donors'],
                         'reserved_ToT_work_ratio':new['remaining_work']/truth,
                         'DR_transfer_work_ratio':baseline['remaining_work']/truth})
        results[key]={'graphs':len(rows),'median_ToT_work_ratio':statistics.median(r['reserved_ToT_work_ratio']for r in rows),'median_DR_transfer_work_ratio':statistics.median(r['DR_transfer_work_ratio']for r in rows),'rows':rows}
    report={'CPU_only':True,'GPU_used':False,'gate':'PASS','scope':'Leave-one-graph-out within14 reserved calibration input graphs per model. Fixed replay-length work proxy, not formal evaluation, not natural response prediction coverage or GPU performance. No parameter selection from these scores.',
            'asset_sha256':hashlib.sha256((stage/'assets/TOT_CALIBRATION.json').read_bytes()).hexdigest(),'results':results}
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{n:v for n,v in z.items()if n!='rows'}for k,z in results.items()}))


if __name__=='__main__':main(sys.argv[1])
