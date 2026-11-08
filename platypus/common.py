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
# CLEAN EVALUATION (inf_clean): wordings the system never writes (conclusions are written as "So A is connected to C.", typed
#   "A r1 someone who r2 C."); the fit wording measures schema fit with a prompt different from every test prompt
_HO_TWO=("{A} is tied to ","The one tied to {A} is ")
_HO_TYPED="{A} {r0} a person who {r1} "
_FIT_W="{A} keeps company with "
def _nslots(cfg):
    """slots the router may open (infer_slots; default: one per domain)"""
    return max(len(PILE_DOMAINS),int(getattr(cfg,"infer_slots",0) or 0))
LOG2=math.log(2)
_FAILS=[]                                    # issue #28: every print-and-continue exception handler records here; the run summary prints FAILURES: n
def _fail(where):
    import sys as _s9
    e_=_s9.exc_info()[1]; _FAILS.append((where,type(e_).__name__ if e_ is not None else '?',str(e_)[:160] if e_ is not None else ''))

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    'PILE_DOMAINS': 'platypus.config',
}
