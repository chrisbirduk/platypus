"""Shared imports, the device, and the helpers that keep module-level names consistent across modules."""
import os as _os0
_os0.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')   # less fragmentation (only takes effect before CUDA starts)
import math
import torch

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
_PKG_MODULES=[]          # the package modules in load order (filled by platypus/__init__.py); empty in the single-file export
_PKG_ORDER=[]            # (module, name) of every module-level name in the order of the original single file
def _set_global(name, value):
    """rebind a module-level name everywhere it is held (the defining module and every module that imported it)"""
    for d_ in ([m_.__dict__ for m_ in _PKG_MODULES] or [globals()]):
        if name in d_: d_[name]=value
def _pkg_globals():
    """all module-level names and their current values, in the order of the original single file"""
    if not _PKG_MODULES: return globals()
    return {n_:m_.__dict__[n_] for m_,n_ in _PKG_ORDER if n_ in m_.__dict__}
LOG2=math.log(2)
