"""Skip vLLM's vendor all-card NVML probe; expose only masked CUDA logical0."""
import os,sys,types,importlib.util,importlib.metadata,importlib.machinery,ctypes,uuid
from pathlib import Path

def scoped_platform(torch_module,Platform,PlatformEnum):
 class GPU0Platform(Platform):
  _enum=PlatformEnum.CUDA
  @staticmethod
  def _logical(device_id):
   if device_id!=0:raise RuntimeError('only logical CUDA0 / physicalGPU0 authorized')
  @staticmethod
  def get_device_capability(device_id=0):GPU0Platform._logical(device_id);return torch_module.cuda.get_device_capability(0)
  @staticmethod
  def get_device_name(device_id=0):GPU0Platform._logical(device_id);return torch_module.cuda.get_device_name(0)
  @staticmethod
  def is_full_nvlink(physical_device_ids):
   if len(physical_device_ids)>1 or any(i!=2 for i in physical_device_ids):raise RuntimeError('multiGPU topology query forbidden')
   return True
 return GPU0Platform

def bootstrap():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID':raise RuntimeError('GPU0 mask/order not installed before imports')
 if 'vllm' in sys.modules or 'vllm.platforms' in sys.modules:raise RuntimeError('vLLM imported before scoped platform')
 import torch
 # Force real CUDA runtime enumeration under the mask, not NVML global count.
 if hasattr(torch.cuda,'_device_count_nvml'):torch.cuda._device_count_nvml=lambda:-1
 vendor=Path(importlib.metadata.distribution('vllm').locate_file('vllm'));name='vllm.platforms.interface';spec=importlib.util.spec_from_file_location(name,vendor/'platforms/interface.py');module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 pack=types.ModuleType('vllm.platforms');pack.__path__=[str(vendor/'platforms')];pack.__file__=str(vendor/'platforms/__init__.py');pack.__package__='vllm.platforms';pack.__spec__=importlib.machinery.ModuleSpec('vllm.platforms',loader=None,is_package=True);pack.Platform=module.Platform;pack.PlatformEnum=module.PlatformEnum;pack.UnspecifiedPlatform=module.UnspecifiedPlatform
 cls=scoped_platform(torch,module.Platform,module.PlatformEnum);pack.current_platform=cls();sys.modules['vllm.platforms']=pack
 cuda=types.ModuleType('vllm.platforms.cuda');cuda.__file__=__file__;cuda.CudaPlatform=cls;sys.modules['vllm.platforms.cuda']=cuda
 return {'all_device_platform_enumeration_bypassed':True,'authorized_physical_gpu':2,'visible_mask':'2'}

def cuda_driver_uuid():
 lib=ctypes.CDLL('libcuda.so.1')
 if lib.cuInit(0)!=0:raise RuntimeError('CUDA driver initialization failed')
 device=ctypes.c_int()
 if lib.cuDeviceGet(ctypes.byref(device),0)!=0:raise RuntimeError('logical CUDA0 unavailable')
 class U(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
 u=U();fn=getattr(lib,'cuDeviceGetUuid_v2',lib.cuDeviceGetUuid)
 if fn(ctypes.byref(u),device)!=0:raise RuntimeError('driver UUID unavailable')
 return 'GPU-'+str(uuid.UUID(bytes=bytes(u.bytes)))
