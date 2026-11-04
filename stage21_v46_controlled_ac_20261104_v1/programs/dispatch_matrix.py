"""Three independent per-GPU sequential workers. Never repeat a failed case."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import json,os,subprocess,sys,threading,traceback

def read(p):return json.loads(p.read_text())
def main(root,study):
 matrix=read(study/'MATRIX.json');lock=threading.RLock();status={'state':'RUNNING','utc':datetime.now(timezone.utc).isoformat(),'workers':{gpu:{'state':'PENDING','active_case':None}for gpu in matrix['GPUs']}}
 def save():
  with lock:
   temp=study/'MATRIX_STATUS.tmp';temp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');os.replace(temp,study/'MATRIX_STATUS.json')
 save()
 def worker(gpu):
  cases=[c for c in matrix['cases']if c['GPU']==gpu];status['workers'][gpu]={'state':'RUNNING','active_case':None};save()
  for index,c in enumerate(cases):
   stage=Path(c['stage']);status['workers'][gpu].update(active_case=c['name']);save()
   with(stage/'OUTER_RUNNER.log').open('x')as output:
    result=subprocess.run([sys.executable,str(study/'common/programs/case_runner.py'),str(root),str(stage)],cwd=root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
   if result.returncode:
    status['workers'][gpu].update(state='FAILED_PRESERVED',failed_case=c['name'])
    for rest in cases[index+1:]:
     Path(rest['stage'],'CALIBRATION_STATUS.json').write_text(json.dumps({'state':'SKIPPED_AFTER_FAILURE','GPU':gpu,'active_arm':None,'completed_arms':[],'reason':c['name']})+'\n')
    save();return
  status['workers'][gpu].update(state='COMPLETED',active_case=None);save()
 with ThreadPoolExecutor(max_workers=3)as pool:list(pool.map(worker,matrix['GPUs']))
 status['state']='COMPLETED'if all(w['state']=='COMPLETED'for w in status['workers'].values())else'FAILED_PRESERVED';status['completed_utc']=datetime.now(timezone.utc).isoformat();save()
if __name__=='__main__':main(*map(Path,sys.argv[1:3]))
