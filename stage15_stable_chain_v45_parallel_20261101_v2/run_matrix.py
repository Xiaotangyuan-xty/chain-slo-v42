"""Run the predeclared fixed candidate on three parallel counterbalanced pairs."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,traceback
from pathlib import Path

def dump(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(q,p)

def main():
    root,matrix=map(Path,sys.argv[1:3]);spec=json.loads((matrix/'MATRIX.json').read_text())
    python=json.loads((root/'PREPARE.json').read_text())['python']
    result={'state':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),'rounds':[],
            'stable_win_proven':False,'scope':'Fixed200 public DEV-informed replay, same arrival seed42; not generalization or semantic equivalence'}
    def status(**kw):
        result.update(kw);dump(matrix/'MATRIX_STATUS.json',result);print(json.dumps(kw),flush=True)
    try:
        fingerprints=[]
        for row in spec['rounds']:
            stage=matrix/row['directory'];q=json.loads((stage/'receipts/CPU_QUALIFICATION.json').read_text())
            if q['gate']!='PASS':raise RuntimeError('Every CPU gate must pass before GPU launch')
            fingerprints.append({n:h for n,h in q['stage_files'].items() if n.startswith(('programs/','assets/'))})
        if any(f!=fingerprints[0]for f in fingerprints):raise RuntimeError('Candidate differs between rounds')
        status(candidate_source_fingerprints=fingerprints[0],matrix_sha256=hashlib.sha256((matrix/'MATRIX.json').read_bytes()).hexdigest())
        for row in spec['rounds']:
            stage=matrix/row['directory'];status(active_round=row['round'],phase='COMPAT_THEN_PARALLEL')
            with (stage/'RUNNER.log').open('x')as log:
                code=subprocess.run([python,str(stage/'programs/run_experiments.py'),str(root),str(stage)],cwd=root,stdout=log,stderr=subprocess.STDOUT).returncode
            with (stage/'POSTPROCESS.log').open('x')as log:
                post=subprocess.run([sys.executable,str(matrix/'postprocess_round.py'),str(root),str(stage)],cwd=root,stdout=log,stderr=subprocess.STDOUT).returncode
            receipt=stage/'receipts/AC_COMPARISON.json'
            c=json.loads(receipt.read_text()) if receipt.exists() else {}
            audit=json.loads((stage/'results/FINAL_AUDIT.json').read_text()) if (stage/'results/FINAL_AUDIT.json').exists() else {}
            valid=code==post==0 and c.get('comparison_valid') is True and audit.get('gate')=='PASS'
            metrics=c.get('metrics',{})
            result['rounds'].append({**row,'valid':valid,'controller_runner_returncode':code,'postprocess_returncode':post,
                'A':metrics.get('A',{}).get('official_scaled_attained'),'C':metrics.get('C',{}).get('official_scaled_attained'),
                'C_win':c.get('C_strictly_more_scaled_chains',False) if valid else None})
            status(rounds=result['rounds'])
            if not valid:raise RuntimeError(f'Round {row["round"]} invalid; preserve and stop, do not widen gates')
        status(state='COMPLETED',phase='DONE',active_round=None,
               stable_win_proven=len(result['rounds'])==3 and all(r['valid']and r['C_win']for r in result['rounds']),
               completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status(state='FAILED_PRESERVED',phase='STOPPED',error=str(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
