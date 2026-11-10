"""Device identity/UUID adaptation only; no serving or scheduling changes."""
import ast
import ctypes
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import uuid


def driver_uuid():
    lib = ctypes.CDLL('libcuda.so.1')
    if lib.cuInit(0) != 0:
        raise RuntimeError('CUDA driver initialization failed')
    device = ctypes.c_int()
    if lib.cuDeviceGet(ctypes.byref(device), 0) != 0:
        raise RuntimeError('Masked logical CUDA0 is unavailable')
    class U(ctypes.Structure):
        _fields_ = [('bytes', ctypes.c_ubyte * 16)]
    result = U()
    fn = getattr(lib, 'cuDeviceGetUuid_v2', lib.cuDeviceGetUuid)
    if fn(ctypes.byref(result), device) != 0:
        raise RuntimeError('Cannot read CUDA UUID')
    return 'GPU-' + str(uuid.UUID(bytes=bytes(result.bytes)))


def adapt_tree(text, physical, mask):
    tree = ast.parse(text)
    changed = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'device_id_to_physical_device_id':
            node.body = ast.parse(
                f"assert os.environ.get('CUDA_VISIBLE_DEVICES') == {mask!r}\n"
                "if device_id != 0:\n    raise RuntimeError('Single visible CUDA device only')\n"
                f"return {physical}").body
            changed.append(node.name)
        elif isinstance(node, ast.FunctionDef) and node.name == 'warn_if_different_devices':
            node.body = [ast.Return(value=ast.Constant(None))]
            changed.append(node.name)
    if sorted(changed) != ['device_id_to_physical_device_id', 'warn_if_different_devices']:
        raise RuntimeError('Unexpected official CUDA platform source')
    return ast.fix_missing_locations(tree), changed


def install(root, stage, shared_backend):
    root, stage = Path(root), Path(stage)
    protocol=json.loads((stage/'PROTOCOL.json').read_text())
    arm=os.environ['STUDY_ARM']
    authorization = json.loads((stage / protocol['arm_devices'][arm]['authorization_file']).read_text())
    from transport_isolation import isolate_prediction_port
    isolate_prediction_port(protocol['prediction_ports'][arm],'server')
    physical = authorization['physical_gpu_index']
    expected = authorization['gpu_uuid']
    if physical not in authorization['allowed_physical_gpu_indices'] or os.environ.get('CUDA_VISIBLE_DEVICES') != expected or os.environ.get('STUDY_GPU') != str(physical):
        raise RuntimeError('Only the authorized physical GPU mask is allowed')
    actual = driver_uuid()
    if actual != expected:
        raise RuntimeError(f'Logical CUDA0 UUID differs from authorized GPU: {actual}')
    source = root / 'repo/third_party/vllm/vllm/platforms/cuda.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = json.loads((root / 'ENVIRONMENT_ADAPTATION.json').read_text())
    if digest != original['cuda_platform_source_sha256']:
        raise RuntimeError('Official CUDA platform SHA mismatch')
    if 'vllm.platforms.cuda' in sys.modules:
        raise RuntimeError('CUDA platform imported before device adaptation')
    tree, changed = adapt_tree(source.read_text(), physical, expected)
    receipt = {'physical_gpu_index': physical, 'expected_uuid': expected, 'actual_uuid': actual,
               'mask': expected, 'cuda_platform_sha256': digest, 'adapted_functions': changed,
               'scheduling_or_prediction_semantics_changed': False}
    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            module.__file__ = str(source)
            exec(compile(tree, str(source), 'exec'), module.__dict__)
            module._stage8_device_adapter = receipt
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'vllm.platforms.cuda':
                return importlib.util.spec_from_loader(fullname, Loader(), origin=str(source))
            return None
    sys.meta_path.insert(0, Finder())
    if shared_backend:
        # Rebind only the frozen backend's device authorization constants.
        gate = types.ModuleType('study_gate')
        gate.__file__ = __file__
        gate.GPU, gate.EXPECTED_UUID = expected, expected
        sys.modules['study_gate'] = gate
    import torch
    torch.cuda._device_count_nvml = lambda: -1
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU is required')
    return receipt
