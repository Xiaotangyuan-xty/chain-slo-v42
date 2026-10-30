"""Execute unchanged upstream source; isolate vLLM type/runtime boundaries.

No priority, sorting, caching or sliding-window algorithm is reimplemented here.
The complete official modules are compiled verbatim. Import adapters supply
logging/types only; the real GPU execution remains the common native backend.
"""
import builtins,hashlib,json,logging,sys,types,enum
from pathlib import Path

SOURCE=Path(__file__).parent/'official_jitserve'
def verify_sources():
    m=json.loads((SOURCE/'MANIFEST.json').read_text())
    for n,x in m['files'].items():
        if hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()!=x['sha256']:raise RuntimeError('official source hash changed: '+n)
    return m

class RuntimeBoundary:
    """The full vLLM allocator/engine constructor is intentionally not invoked."""
    def __init__(self,*a,**k):raise RuntimeError('full vLLM runtime is outside the shared-backend kernel adapter')
class SequenceStatus(enum.IntEnum):
    WAITING=0;RUNNING=1;SWAPPED=2;FINISHED_ABORTED=3

def load_official(clock):
    manifest=verify_sources();prefix='_pinned_official_jitserve_'+str(id(clock))
    modules={}
    def execute(name,filename):
        mod=types.ModuleType(prefix+'.'+name);mod.__file__=str(SOURCE/filename)
        sys.modules[mod.__name__]=mod;ns=mod.__dict__
        def scoped_import(target,globals=None,locals=None,fromlist=(),level=0):
            if target in ('jitserve.request_info','vllm.slo_tracker.request_info'):return modules['request_info']
            if target=='vllm.logger':return types.SimpleNamespace(init_logger=logging.getLogger)
            if target=='vllm.sequence':return types.SimpleNamespace(SequenceGroup=object,SequenceStatus=SequenceStatus)
            if target in ('vllm.config','vllm.core.interfaces','vllm.utils','vllm.core.scheduler'):
                names=set(fromlist or ())
                return types.SimpleNamespace(**{n:RuntimeBoundary for n in names})
            result=builtins.__import__(target,globals,locals,fromlist,level)
            if target=='time':return types.SimpleNamespace(time=clock,perf_counter=result.perf_counter)
            return result
        ns['__builtins__']={**vars(builtins),'__import__':scoped_import}
        text=(SOURCE/filename).read_text()
        exec(compile(text,str(SOURCE/filename),'exec'),ns)
        modules[name]=mod;return mod
    info=execute('request_info','request_info.py')
    policy=execute('policy','policy.py');scheduler=execute('slo_scheduler','slo_scheduler.py')
    return types.SimpleNamespace(info=info,policy=policy,scheduler=scheduler,manifest=manifest)
