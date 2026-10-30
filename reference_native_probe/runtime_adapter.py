"""UUID compatibility only; original source files/native libraries stay byte-identical."""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys


def install(study):
    study=Path(study);m=json.loads((study/'PREPARE.json').read_text())
    expected_uuid=m['gpu_uuid'];physical=m['physical_gpu_index']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==expected_uuid
    source=(study/'repo/third_party/vllm/vllm/platforms/cuda.py').resolve()
    frozen=json.loads((study/'ENVIRONMENT_ADAPTATION.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen['cuda_platform_source_sha256']

    class Loader(importlib.abc.Loader):
        def create_module(self,spec):return None
        def exec_module(self,module):
            tree=ast.parse(source.read_text());changed=[]
            for node in tree.body:
                if isinstance(node,ast.FunctionDef) and node.name=='device_id_to_physical_device_id':
                    code=f"assert os.environ.get('CUDA_VISIBLE_DEVICES') == {expected_uuid!r}\nif device_id != 0:\n    raise RuntimeError('Single authorized GPU only')\nreturn {physical}"
                    node.body=ast.parse(code).body;changed.append(node.name)
                elif isinstance(node,ast.FunctionDef) and node.name=='warn_if_different_devices':
                    # A one-visible-card run has no heterogeneity warning to compute.
                    node.body=[ast.Return(value=ast.Constant(None))];changed.append(node.name)
            assert sorted(changed)==['device_id_to_physical_device_id','warn_if_different_devices']
            module.__file__=str(source)
            exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),module.__dict__)
            module.__dict__['_study_UUID_adapter']={'uuid':expected_uuid,'physical_index':physical,'changed':changed}

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='vllm.platforms.cuda':return importlib.util.spec_from_loader(fullname,Loader(),origin=str(source))
            return None
    sys.meta_path.insert(0,Finder())

