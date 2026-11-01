"""Post-run CPU audit/report. Does not modify any experimental source."""
import collections
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import zipfile

ROOT=Path('/mnt/sda1/xuetianyou-a40/agent-dispatch-experiments/idea1_continuity/idea3_full_jitserve_compare_20261027T121314Z_v1')
STAGE=ROOT/'stage14_chain_completion_v44_gpu01_20261101_v1'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def key(row):
    return (row['phase'],row['collection_id'],row['stage_id'],row['request_id'])

def raw_tokens(path,common):
    expected={key(r):r for r in common['requests']}
    records={};bad=[]
    for line in path.open():
        row=json.loads(line);k=key(row)
        prompt=hashlib.sha256(json.dumps(row['prompt_token_ids']).encode()).hexdigest()
        output=hashlib.sha256(json.dumps(row['output_token_ids']).encode()).hexdigest()
        if (k in records or k not in expected or prompt!=row['prompt_token_ids_sha256']
            or output!=row['output_token_ids_sha256'] or output!=expected[k]['token_ids_sha256']
            or prompt!=expected[k]['prompt_token_ids_sha256']):bad.append(k)
        records[k]={'prompt_sha':prompt,'output_sha':output,'tokens':row['output_token_ids'][0]}
    if set(records)!=set(expected):bad.append('raw/common request identities differ')
    return records,bad

def execution_stats(path):
    stats={p:{'calls':0,'groups':0,'await_wall_seconds':0.,'swap_in_blocks':0,'swap_out_blocks':0,'max_batch':0} for p in ['warmup','main']}
    phase='warmup'
    for line in path.open():
        event=json.loads(line)
        if event['kind']=='request_start':phase=event['phase']
        if event['kind']!='model_execute':continue
        d=stats[phase];d['calls']+=1;d['groups']+=event['batch_size'];d['await_wall_seconds']+=event['elapsed_seconds'];d['max_batch']=max(d['max_batch'],event['batch_size'])
        d['swap_in_blocks']+=event['swap_in_blocks'];d['swap_out_blocks']+=event['swap_out_blocks']
    for d in stats.values():d['mean_batch']=d['groups']/max(1,d['calls'])
    return stats

def reservation_alignment(attempt):
    def snapshots():
        for line in (attempt/'reservation_events.jsonl').open():yield json.loads(line)
    it=iter(snapshots());nxt=next(it,None);current=None;count=verified=outside=0;examples=[]
    for line in (attempt/'common_events.jsonl').open():
        event=json.loads(line)
        if event['kind']!='model_execute':continue
        begin=event['monotonic']-event['elapsed_seconds']
        while nxt is not None and nxt['monotonic']<=begin:
            current=nxt;count+=1;nxt=next(it,None)
        if current is None:continue
        unexpected=set(event['request_ids'])-set(current['selected'])
        if unexpected:
            outside+=1
            if len(examples)<12:examples.append({'time':begin,'requests':sorted(unexpected),'snapshot_free_cpu_blocks':current['free_cpu']})
        else:verified+=1
    return {'snapshots_consumed':count,'executor_calls_within_selection':verified,
            'executor_calls_outside_selection':outside,'first_examples':examples,
            'interpretation':'New C uses owned native recomputation when CPU swap capacity is unavailable. '
            'Out-of-selection execution is reported; it is not silently relabeled as enforced admission.'}

def package():
    paths=[STAGE/'PROTOCOL.json',STAGE/'GPU0_AUTHORIZATION.json',STAGE/'GPU1_AUTHORIZATION.json',STAGE/'TRACE_MANIFEST.json',STAGE/'PARENT_V43.json',STAGE/'RUNNER_LAUNCH.json']+list((STAGE/'assets').glob('*.json'))
    paths+=list((STAGE/'programs').glob('*.py'))+list((STAGE/'receipts').glob('*'))
    paths+=list((STAGE/'postprocess').glob('*.py'))
    paths+=list((STAGE/'results').glob('*'))
    for attempt in (STAGE/'attempts').iterdir():
        for name in ['common_audit_summary.json','server_audit_summary.json','controller_receipt.json','CLIENT_CONTRACT.json','official_metrics.json','processes.json','controller_status.json','CONTROLLER_FAILURE.json']:
            if (attempt/name).exists():paths.append(attempt/name)
    target=STAGE/'FULL_TRACE_LIGHT_DELIVERY.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p.is_file():z.write(p,str(p.relative_to(STAGE)))
    return {'archive':str(target),'archive_sha256':sha(target),'bytes':target.stat().st_size}

def main():
    result_dir=STAGE/'results';result_dir.mkdir(exist_ok=False)
    comparison_path=STAGE/'receipts/AC_COMPARISON.json'
    if not comparison_path.exists():
        failures={p.parent.name:read(p) for p in (STAGE/'attempts').glob('*/CONTROLLER_FAILURE.json')}
        status={'state':'FAILED_PRESERVED','failure_receipts':failures,'comparison_valid':False,
                'message':'The gated pair stopped before a valid complete comparison; no result is claimed.'}
        (result_dir/'EXPERIMENT_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
        log=(STAGE/'RUNNER.log').read_text()
        (result_dir/'完整trace_GPU01_A_v44C实验报告.md').write_text('# 完整 trace GPU0 实验未形成有效对照\n\n运行被验收门停止，原始失败记录已保留。不能将不完整结果作为 A/C 胜负。\n\n```\n'+log[-4500:]+'\n```\n')
        print(json.dumps(package()),flush=True);return
    protocol=read(STAGE/'PROTOCOL.json');comparison=read(comparison_path);qualification=read(STAGE/'receipts/CPU_QUALIFICATION.json');bad=[]
    for prefix,hashes in [(ROOT,qualification['base_files']),(STAGE,qualification['stage_files'])]:
        for name,digest in hashes.items():
            if not (prefix/name).is_file() or sha(prefix/name)!=digest:bad.append(str(prefix/name))
    common={};raw={};stats={};controllers={};raw_bad={};hashes={}
    for arm in ['A','C']:
        attempt=STAGE/'attempts'/protocol['attempts'][arm]
        common[arm]=read(attempt/'common_audit_summary.json');controllers[arm]=read(attempt/'controller_receipt.json')
        raw[arm],raw_bad[arm]=raw_tokens(attempt/'engine_token_ids.jsonl',common[arm])
        stats[arm]=execution_stats(attempt/'common_events.jsonl')
        hashes[arm]={name:{'sha256':sha(attempt/name),'bytes':(attempt/name).stat().st_size} for name in ['client_calls.json','common_events.jsonl','server_events.jsonl','engine_token_ids.jsonl']}
    token={'all':{'requests':0,'identical_sequences':0,'positionwise_matches':0,'tokens_compared':0},'by_phase':{},'first_differences':[],'bad_raw_records':raw_bad}
    for k,a in raw['A'].items():
        c=raw['C'].get(k)
        if c is None:token['bad_raw_records']['C'].append(list(k));continue
        if a['prompt_sha']!=c['prompt_sha']:token['bad_raw_records']['C'].append(['input_ids_differ',*k])
        eq=a['tokens']==c['tokens'];same=sum(x==y for x,y in zip(a['tokens'],c['tokens']))
        d=token['by_phase'].setdefault(k[0],{'requests':0,'identical_sequences':0,'positionwise_matches':0,'tokens_compared':0})
        for item in [d,token['all']]:
            item['requests']+=1;item['identical_sequences']+=eq;item['positionwise_matches']+=same;item['tokens_compared']+=max(len(a['tokens']),len(c['tokens']))
        if not eq:
            index=next((i for i,(x,y) in enumerate(zip(a['tokens'],c['tokens'])) if x!=y),min(len(a['tokens']),len(c['tokens'])))
            token['first_differences'].append({'identity':list(k),'first_different_index_zero_based':index,'A_token':a['tokens'][index] if index<len(a['tokens']) else None,'C_token':c['tokens'][index] if index<len(c['tokens']) else None,'A_length':len(a['tokens']),'C_length':len(c['tokens']),'positionwise_matches':same})
    token['interpretation']='Complete-sequence equality, common prefixes and positionwise ID equality are not semantic-quality measurements. No logits were recorded.'
    align=reservation_alignment(STAGE/'attempts'/protocol['attempts']['C'])
    gpu_state={}
    for index in ['0','1']:
        gpu=subprocess.check_output(['nvidia-smi','-i',index,'--query-gpu=index,uuid,memory.used','--format=csv,noheader'],text=True).strip()
        processes=subprocess.check_output(['nvidia-smi','-i',index,'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        gpu_state[index]={'gpu':gpu,'processes':processes}
    ports={}
    for port in [18080,65433]:
        try:
            with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))
            ports[str(port)]=True
        except OSError:ports[str(port)]=False
    valid=(comparison['gate']=='PASS' and not bad and not any(raw_bad.values()))
    audit={'gate':'PASS' if valid else 'FAIL','source_mismatches':bad,'base_files':len(qualification['base_files']),
           'stage_files':len(qualification['stage_files']),'gpu_state':gpu_state,'ports_free':ports,
           'final_kv_free':{a:c['final_kv_free'] for a,c in common.items()},'controllers':controllers,
           'C_reservation_alignment':align,'raw_trace_inventory':hashes,'token_record_errors':raw_bad}
    a=comparison['metrics']['A'];c=comparison['metrics']['C'];chain_rows=[]
    for i in range(200):
        k=str(i);chain_rows.append({'chain':i,'stages':protocol['stage_counts'][k],'deadline_seconds':20*(i%4+1)*protocol['stage_counts'][k],'A_jct':a['jct'][k],'C_jct':c['jct'][k],'A_attained':a['official_scaled_per_chain'][k],'C_attained':c['official_scaled_per_chain'][k]})
    cases=collections.Counter(('both' if x['A_attained'] and x['C_attained'] else 'A_only' if x['A_attained'] else 'C_only' if x['C_attained'] else 'neither') for x in chain_rows)
    summary={'comparison_valid':valid,'metrics':comparison['metrics'],'attainment_overlap':dict(cases),'chains':chain_rows,'execution_stats':stats,'scope':'All200 public trace chains in original order, one load/seed pair, includes inspected DEV prefix. Fixed input/output lengths; no semantic-equivalence claim.'}
    status={'state':'COMPLETED' if valid else 'COMPLETED_INVALID','comparison_valid':valid,'A_attained':a['official_scaled_attained'],'C_attained':c['official_scaled_attained'],'C_strictly_more':c['official_scaled_attained']>a['official_scaled_attained'],'completed_at_Asia_Shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),'server_stage':str(STAGE)}
    for name,data in [('FINAL_AUDIT.json',audit),('TOKEN_COMPARISON.json',token),('FULL_TRACE_SUMMARY.json',summary),('EXPERIMENT_STATUS.json',status)]:
        (result_dir/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    verdict='C 的达标链数严格多于 A。' if status['C_strictly_more'] else '本轮 C 的达标链数没有严格多于 A。'
    report=f'''# GPU0/GPU1 完整公开 trace：原生 JITServe A / v44 C

状态：{status['state']}。统一离线对照有效：{valid}。{verdict}

| 指标 | 原生 A | 新机制 v44 C |
|---|---:|---:|
| 实际下发 SLO 达标链数 | {a['official_scaled_attained']}/200 | {c['official_scaled_attained']}/200 |
| 严格未缩放达标链数 | {a['strict_attained']}/200 | {c['strict_attained']}/200 |
| 平均 JCT，秒 | {a['mean_jct']:.3f} | {c['mean_jct']:.3f} |
| JCT P95，秒 | {a['jct_p95']:.3f} | {c['jct_p95']:.3f} |
| 主负载排空时长，秒 | {a['episode_duration']:.3f} | {c['episode_duration']:.3f} |
| 达标链请求 goodput，req/s | {a['request_goodput']:.6f} | {c['request_goodput']:.6f} |
| 主负载平均实际 batch | {stats['A']['main']['mean_batch']:.3f} | {stats['C']['main']['mean_batch']:.3f} |
| 实际最大 batch（含热身） | {common['A']['stats']['max_executed_batch']} | {common['C']['stats']['max_executed_batch']} |
| 主负载执行器等待墙钟，秒 | {stats['A']['main']['await_wall_seconds']:.3f} | {stats['C']['main']['await_wall_seconds']:.3f} |

完整输入为原顺序全部 200 条链、1197 个主请求，每臂另有相同7请求热身；主目标输出504695 tokens。A使用物理GPU0、C使用物理GPU1（同型号A40，顺序运行），BF16 Qwen2.5-7B-Instruct、原生vLLM执行器、GPU512/CPU1170 KV块、最大32序列、每步2048 tokens、模型长度8192、1GiB swap。v43冻结及既有stage13完整保留；v44使用独立训练图、完成密度、短期KV预留与执行量保护。A启动器与stage13逐字节一致，A调度语义不变。

到达间隔沿用seed42的1链/秒公开生成规则，两臂使用同一累计绝对计划（最后一链计划199.780秒）。仅将HTTP传输等待和控制器墙钟上限扩展至27000/28800秒，实际SLO不变。SLO以客户端收到的值乘实际阶段数，未缩放20×阶段数另列；没有为结果改链、deadline、seed、负载或资源。

达标重合：{dict(cases)}。200条链逐条JCT和达标结果见FULL_TRACE_SUMMARY.json。执行器等待墙钟不是CUDA event时间，批次内各请求时间重叠。

完整输出token序列：{token['all']['identical_sequences']}/{token['all']['requests']}个请求逐序列一致。该比例不等于逐token一致率。本次保存了服务器端真实输入/输出token数组，并在TOKEN_COMPARISON.json记录每个不一致请求的首次分叉位置与token ID。未记录logits，不据此断言差异原因或语义质量相同。

源码审计：{audit['gate']}，{audit['base_files']}个基础文件及{audit['stage_files']}个本轮文件；最终KV：{audit['final_kv_free']}。两卡最终状态：{gpu_state}。端口释放：{ports}。

C准入执行对齐记录见FINAL_AUDIT.json，集合外执行调用数为{align['executor_calls_outside_selection']}。v44在CPU换出池不足时使用原生重算；集合外执行如出现仍单独报告。

这是完整公开trace的一次固定负载/seed对照，包含此前查看过的DEV前缀；不属于未接触过的独立HOLDOUT，也不证明泛化优势或真实agent语义等价。所有迟到请求均按原机制排空，不能只截取达标前缀解释结果。

原始client_calls、common/server事件、reservation事件和engine_token_ids保留在服务器：
`{STAGE}/attempts/`

本报告和轻量证据不会改动GitHub冻结标签或既有论文初稿。
'''
    (result_dir/'完整trace_GPU01_A_v44C实验报告.md').write_text(report)
    print(json.dumps({**package(),'state':status['state'],'A':status['A_attained'],'C':status['C_attained']}),flush=True)

if __name__=='__main__':main()
