"""One compatibility-gated parallel pair; never omit a losing complete arm."""
import json
from pathlib import Path
import subprocess
import sys
import traceback

def parallel_controllers(commands,handles,cwd):
    children={arm:subprocess.Popen(command,stdout=handles[arm],stderr=subprocess.STDOUT,cwd=cwd)
              for arm,command in commands.items()}
    return {arm:child.wait() for arm,child in children.items()}

def main():
    root,stage=map(Path,sys.argv[1:3]);p=json.loads((stage/'PROTOCOL.json').read_text())
    python=json.loads((root/'PREPARE.json').read_text())['python']
    def state(phase,**extra):
        data={'phase':phase,'round':p['round'],**extra}
        (stage/'ROUND_STATUS.json').write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps(data),flush=True)
    def run(command,log):
        with (stage/log).open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=root)
    base=[python,str(stage/'programs/controller.py'),str(root),str(stage)]
    compare=[python,str(stage/'programs/offline_compare.py'),str(root),str(stage)]
    try:
        state('COMPATIBILITY')
        run(base+['C',p['attempts']['compat'],'compat'],'COMPAT_CONTROLLER.log')
        run(compare+['arm',p['attempts']['compat'],'compat'],'COMPAT_GATE.log')
        state('PARALLEL_FULL_RUNNING',devices=p['arm_devices'])
        handles={arm:(stage/f'FULL_{arm}_CONTROLLER.log').open('x') for arm in ['A','C']}
        try:codes=parallel_controllers({arm:base+[arm,p['attempts'][arm],'full'] for arm in ['A','C']},handles,root)
        finally:
            for handle in handles.values():handle.close()
        if any(codes.values()):raise RuntimeError(f'Controller failure; both awaited and preserved: {codes}')
        state('OFFLINE_VALIDATION')
        for arm in ['A','C']:run(compare+['arm',p['attempts'][arm],'full'],f'FULL_{arm}_GATE.log')
        run(compare+['ac',p['attempts']['A'],p['attempts']['C']],'AC_GATE.log')
        c=json.loads((stage/'receipts/AC_COMPARISON.json').read_text())
        state('VALID_COMPLETED',A=c['metrics']['A']['official_scaled_attained'],C=c['metrics']['C']['official_scaled_attained'])
    except BaseException as e:
        state('FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
