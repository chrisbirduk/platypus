"""Named regions for the one-off GPU profile."""
import contextlib
import torch
from platypus.common import (
    _pkg_globals, _set_global,
)


# ---------------- PROFILING LABELS: named regions for the one-off GPU profile (profile_visit); no effect otherwise ----------------
_PRF={'on':False}
def _rf(name):
    """v9.8: a named region in the one-off GPU profile (no effect otherwise)"""
    return torch.profiler.record_function('R:'+name) if _PRF['on'] else contextlib.nullcontext()
def _labelled(fn,name):
    import functools
    @functools.wraps(fn)
    def g(*a,**k):
        if not _PRF['on']: return fn(*a,**k)
        with torch.profiler.record_function('R:'+name): return fn(*a,**k)
    return g
def _label_regions():
    """label the named regions (called by platypus/__init__.py once every module is loaded)"""
    for _nm in ['_cd_gen', '_cd_encode', '_cd_mis', '_cd_rows', '_cd_cat', '_recon_loss_mark', '_rs_rowloss', '_rs_pick', '_fgram', '_kw_par', '_delta_core', 'muon_prep', 'varpro_old_refit', 'varpro_refit', 'pu_term', 'film_mod', 'ngram_embed', 'ngp_vec', 'priv', 'bank_dream_batch', '_kf_hook', 'apply_update_gate', 'dctl_rate', '_kf_apply', '_ns5_batch', '_ns5_gram_batch', 'muon_apply', 'ngp_step', 'head_center_']:
        if _nm in _pkg_globals(): _set_global(_nm,_labelled(_pkg_globals()[_nm],_nm))


# ---------------- cg26q FUNCTION TIMES (PLATYPUS_FNTIME=1): wall-clock seconds per sleep and readout function, to find what to run less often ----------------
_FT={'on':False,'t':{},'stk':[],'tot':{}}
_FT_MODULES=('inference.audit','inference.closed_form','inference.decay','inference.evaluate','inference.infer','inference.jepa','inference.ledger',
             'inference.mixing','inference.need','inference.readouts','inference.rem','inference.selftaught','inference.sentences','inference.sleep',
             'diagnostics','importance','meta','dream_bank','orth')
def _timed(fn,name):
    import functools, time
    @functools.wraps(fn)
    def g(*a,**k):
        t0=time.perf_counter(); _FT['stk'].append(0.0)
        try: return fn(*a,**k)
        finally:
            if torch.cuda.is_available():
                try: torch.cuda.synchronize()
                except Exception: pass
            dt=time.perf_counter()-t0; ch=_FT['stk'].pop()
            v=_FT['t'].setdefault(name,[0,0.0,0.0]); v[0]+=1; v[1]+=dt; v[2]+=dt-ch
            if _FT['stk']: _FT['stk'][-1]+=dt
    g._ft_inner=fn
    return g
def _time_functions():
    """wrap every module-level function of the sleep, readout and diagnostic modules (called once at import when PLATYPUS_FNTIME=1)"""
    import sys, inspect
    _FT['on']=True; n_=0
    for m in _FT_MODULES:
        mod=sys.modules.get('platypus.'+m)
        if mod is None: continue
        for name,obj in list(vars(mod).items()):
            if inspect.isfunction(obj) and obj.__module__==mod.__name__ and not hasattr(obj,'_ft_inner'):
                _set_global(name,_timed(obj,name)); n_+=1
    return n_
def _ft_report(tag,top=14):
    """print and reset the function times since the last report (exclusive = without the wrapped functions it calls)"""
    if not _FT['on'] or not _FT['t']: return
    T=_FT['t']; tot=sum(v[2] for v in T.values())
    for k,v in T.items(): w=_FT['tot'].setdefault(k,[0,0.0]); w[0]+=v[0]; w[1]+=v[2]
    it=sorted(T.items(),key=lambda kv:-kv[1][2])[:top]
    print(f"  [FUNCTION TIMES | {tag}] {tot:.1f} s in wrapped functions | exclusive s (calls, inclusive s): "
          +" | ".join(f"{k} {v[2]:.1f} ({v[0]}, {v[1]:.1f})" for k,v in it),flush=True)
    _FT['t']={}
def _ft_total(top=25):
    if not _FT['on']: return
    _ft_report('last visit')
    T=_FT['tot']; tot=sum(v[1] for v in T.values())
    it=sorted(T.items(),key=lambda kv:-kv[1][1])[:top]
    print(f"  [FUNCTION TIMES | WHOLE RUN] {tot/60:.1f} min in wrapped functions | exclusive min (calls): "+" | ".join(f"{k} {v[1]/60:.2f} ({v[0]})" for k,v in it),flush=True)
