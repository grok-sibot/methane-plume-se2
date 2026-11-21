"""Fake-GPU device checker for CPU-only boxes.

Pretends a second device exists: FAKE_DEV = torch.device('cpu', 0) plays the role of 'cuda:0'. A TorchFunctionMode
tags every tensor that is moved or created there (.to(FAKE_DEV), device=FAKE_DEV, or computed from a tagged tensor),
reports FAKE_DEV as the `.device` of tagged tensors (so `device=x.device` keeps working), and raises DeviceMismatch
whenever an op mixes tagged and untagged tensors the way CUDA would refuse:
  * any op with a tagged and an untagged tensor that is not 0-dim (CUDA accepts CPU scalars),
  * indexing an untagged tensor with a tagged index, or assigning tagged values into an untagged tensor,
  * .numpy() / __array__ on a tagged tensor (CUDA tensors cannot become numpy arrays without .cpu()),
  * torch.Generator(device=...) mismatches are checked for rand/randint/randn/randperm with generator=.
Allowed like on CUDA: .to()/.cpu()/.copy_() across devices, tagged[cpu_index], tagged[...] = cpu_values.
Usage: enable(); DEV = FAKE_DEV; ... run the pipeline ...
"""
import torch, traceback
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten

FAKE_DEV = torch.device('cpu', 0)
_DEVICE_DESC = torch._C.TensorBase.__dict__['device']; _ISCUDA_DESC = torch._C.TensorBase.__dict__['is_cuda']
_TAG = '_fake_cuda'
_GEN_DEV = {}           # id(generator) -> fake?
class DeviceMismatch(RuntimeError): pass

def _is_fake(d):
    if d is None: return False
    if isinstance(d, str): d = torch.device(d)
    return isinstance(d, torch.device) and d.type == 'cpu' and d.index == 0
def tagged(t): return isinstance(t, torch.Tensor) and getattr(t, _TAG, False)
def _tag(o):
    for t in tree_flatten(o)[0]:
        if isinstance(t, torch.Tensor):
            try: setattr(t, _TAG, True)
            except Exception: pass
    return o

_orig_gen = torch.Generator
class _Gen:  # wraps torch.Generator(device=...) to remember the device
    def __new__(cls, device='cpu'):
        g = _orig_gen(device='cpu'); _GEN_DEV[id(g)] = _is_fake(device); return g

MOVES = {torch.Tensor.to, torch.Tensor.cpu, torch.Tensor.cuda, torch.Tensor.copy_, torch.Tensor.detach, torch.Tensor.clone}
FACTORY_KW = ('device',)
GEN_FUNCS = {torch.rand, torch.randn, torch.randint, torch.randperm, torch.normal, torch.Tensor.uniform_, torch.Tensor.normal_}

class FakeCudaMode(TorchFunctionMode):
    def __init__(self): super().__init__(); self.ops = 0; self.tagged_ops = 0
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, '__name__', str(func))
        # .device of a tagged tensor reports the fake GPU
        desc = getattr(func, '__self__', None)
        if desc is _DEVICE_DESC:
            return FAKE_DEV if tagged(args[0]) else func(*args, **kwargs)
        if desc is _ISCUDA_DESC:
            return tagged(args[0])
        flat = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        self.ops += 1
        if func in (torch.Tensor.numpy, torch.Tensor.__array__) and tagged(args[0]):
            raise DeviceMismatch(f'{name}() on a (fake) CUDA tensor: needs .cpu() first')
        g = kwargs.get('generator')
        if g is not None and func in GEN_FUNCS:
            want = _is_fake(kwargs.get('device')) or (func in (torch.Tensor.uniform_, torch.Tensor.normal_) and tagged(args[0]))
            if _GEN_DEV.get(id(g), False) != want:
                raise DeviceMismatch(f'{name}: generator on {"cuda" if _GEN_DEV.get(id(g)) else "cpu"} but output on {"cuda" if want else "cpu"}')
        # moves
        if func in (torch.Tensor.to,):
            dst = kwargs.get('device', None)
            for a in args[1:]:
                if isinstance(a, (torch.device, str)): dst = a
                if isinstance(a, torch.Tensor): dst = FAKE_DEV if tagged(a) else torch.device('cpu')
            out = func(*args, **kwargs)
            if dst is None: return _tag(out) if tagged(args[0]) and out is not args[0] else out
            if _is_fake(dst):
                if out is args[0] and not isinstance(out, torch.nn.Parameter): out = out.view_as(out)
                return _tag(out)
            if tagged(args[0]) and out is args[0]: out = out.view_as(out)   # fake GPU -> CPU: a distinct untagged object
            if out is not args[0] and hasattr(out, _TAG): delattr(out, _TAG)
            return out
        if func is torch.Tensor.cpu:
            out = func(*args, **kwargs)
            if out is args[0] and tagged(out): out = out.view_as(out)
            if hasattr(out, _TAG): delattr(out, _TAG)
            return out
        if func is torch.Tensor.copy_:
            return func(*args, **kwargs)   # cross-device copy is legal; keeps self's device
        if func is torch.Tensor.__setitem__:
            self_, idx, val = args[0], args[1], args[2]
            idx_t = [t for t in tree_flatten(idx)[0] if isinstance(t, torch.Tensor)]
            if not tagged(self_) and (any(tagged(t) for t in idx_t) or (tagged(val) and val.dim() > 0)):
                raise DeviceMismatch('__setitem__: CPU tensor indexed/assigned with (fake) CUDA tensors')
            return func(*args, **kwargs)
        if func is torch.Tensor.__getitem__:
            self_ = args[0]; idx_t = [t for t in tree_flatten(args[1])[0] if isinstance(t, torch.Tensor)]
            if not tagged(self_) and any(tagged(t) for t in idx_t):
                raise DeviceMismatch('__getitem__: CPU tensor indexed with a (fake) CUDA tensor')
            out = func(*args, **kwargs)
            return _tag(out) if tagged(self_) else out
        fake_in = [t for t in flat if tagged(t)]
        cpu_in = [t for t in flat if not tagged(t) and t.dim() > 0]
        if fake_in and cpu_in:
            shapes = f'cuda {[tuple(t.shape) for t in fake_in][:3]} vs cpu {[tuple(t.shape) for t in cpu_in][:3]}'
            raise DeviceMismatch(f'{name}: tensors on (fake) cuda and cpu: {shapes}')
        out = func(*args, **kwargs)
        if fake_in or _is_fake(kwargs.get('device')):
            self.tagged_ops += 1; _tag(out)
        return out

_MODE = None
def enable():
    """Enter the mode for the rest of this thread (e.g. a Jupyter kernel) and patch torch.Generator."""
    global _MODE
    torch.Generator = _Gen
    _MODE = FakeCudaMode(); _MODE.__enter__(); return _MODE

_mod_to = torch.nn.Module.to
def _module_to(self, *args, **kwargs):
    with torch._C.DisableTorchFunction():   # moving a module is legal; re-tag its tensors afterwards
        out = _mod_to(self, *args, **kwargs)
    dst = kwargs.get('device') or next((a for a in args if isinstance(a, (torch.device, str))), None)
    if dst is not None:
        for t in list(self.parameters()) + list(self.buffers()):
            if _is_fake(dst): setattr(t, _TAG, True)
            elif hasattr(t, _TAG): delattr(t, _TAG)
    return out
torch.nn.Module.to = _module_to
