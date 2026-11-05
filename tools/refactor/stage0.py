"""Stage 0: behaviour-preserving edits to the single file that make a split into real modules safe.
python3 stage0.py IN OUT
1. functions defined twice: the dead first _cka is dropped; the wrapped first bpc / ngp_step get the names their wrappers call (_bpc_w0, _ngp_step_0)
2. module globals rebound at run time (_INF_NAME, _INF_TRIP, the profiling labels) go through _set_global, which updates every module holding the name
3. the memory ledger's scan of module-level stores goes through _pkg_globals (every module, in the original definition order)
4. run() asks the data module whether the Hugging Face source is in use (_using_hf_stream) instead of comparing a name it would import"""
import sys
src = open(sys.argv[1]).read()
def rep(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, count, old[:120])
    src = src.replace(old, new)

# 1. duplicates
rep("@torch.no_grad()\ndef _cka(X,Y):\n    return float((X*Y).sum()/(X.norm()*Y.norm()+1e-30))\n", "")
rep("    return P\ndef ngp_step(model,cfg,lr):\n    O3=model.__dict__.get('_ngpi_opt')", "    return P\ndef _ngp_step_0(model,cfg,lr):\n    O3=model.__dict__.get('_ngpi_opt')")
rep("_ngp_step_0=ngp_step\ndef ngp_step(model,cfg,lr):", "def ngp_step(model,cfg,lr):")
rep("    model.train(); return float(np.mean(L))/LOG2, float(np.mean(A))\n\ndef bpc(model,get,domain,n=15,skip=0):",
    "    model.train(); return float(np.mean(L))/LOG2, float(np.mean(A))\n\ndef _bpc_w0(model,get,domain,n=15,skip=0):")
rep("Head.forward=_head_fwd_m\n_bpc_w0=bpc\ndef bpc(", "Head.forward=_head_fwd_m\ndef bpc(")

# helpers, right after the module-level imports
rep('DEVICE="cuda" if torch.cuda.is_available() else "cpu"\n',
    'DEVICE="cuda" if torch.cuda.is_available() else "cpu"\n'
    '_PKG_MODULES=[]          # the package modules in load order (filled by platypus/__init__.py); empty in the single-file export\n'
    '_PKG_ORDER=[]            # (module, name) of every module-level name in the order of the original single file\n'
    'def _set_global(name, value):\n'
    '    """rebind a module-level name everywhere it is held (the defining module and every module that imported it)"""\n'
    '    for d_ in ([m_.__dict__ for m_ in _PKG_MODULES] or [globals()]):\n'
    '        if name in d_: d_[name]=value\n'
    'def _pkg_globals():\n'
    '    """all module-level names and their current values, in the order of the original single file"""\n'
    '    if not _PKG_MODULES: return globals()\n'
    '    return {n_:m_.__dict__[n_] for m_,n_ in _PKG_ORDER if n_ in m_.__dict__}\n')

# 2. run-time rebinding
rep("    global _INF_NAME\n    if _units_on(cfg):                                                          # cg20s: RULE-FREE: names and facts come from the learned lexicon\n"
    "        global _INF_TRIP; _INF_NAME=_UnitName(); _INF_TRIP=_UnitTrip(); _UNITS['on']=True\n",
    "    if _units_on(cfg):                                                          # cg20s: RULE-FREE: names and facts come from the learned lexicon\n"
    "        _set_global('_INF_NAME',_UnitName()); _set_global('_INF_TRIP',_UnitTrip()); _UNITS['on']=True\n")
rep("        _INF_NAME=_re_inf.compile(rb'(?<![A-Za-z\\x80-\\xff])[A-Z][a-z]{3,11}(?![A-Za-z\\x80-\\xff])')\n",
    "        _set_global('_INF_NAME',_re_inf.compile(rb'(?<![A-Za-z\\x80-\\xff])[A-Z][a-z]{3,11}(?![A-Za-z\\x80-\\xff])'))\n")
rep("        global _INF_NAME,_INF_TRIP\n        if not isinstance(_INF_NAME,_UnitName): _INF_NAME=_UnitName(); _INF_TRIP=_UnitTrip(); _UNITS['on']=True",
    "        if not isinstance(_INF_NAME,_UnitName): _set_global('_INF_NAME',_UnitName()); _set_global('_INF_TRIP',_UnitTrip()); _UNITS['on']=True")
rep("    if _nm in globals(): globals()[_nm]=_labelled(globals()[_nm],_nm)\n",
    "    if _nm in _pkg_globals(): _set_global(_nm,_labelled(_pkg_globals()[_nm],_nm))\n")

# 3. memory ledger
rep("    for k_,v_ in list(globals().items()):\n        if k_.startswith('_') and k_[1:2].isupper() and isinstance(v_,dict): add_(k_,v_)",
    "    for k_,v_ in list(_pkg_globals().items()):\n        if k_.startswith('_') and k_[1:2].isupper() and isinstance(v_,dict): add_(k_,v_)")

# 4. data source
rep("_hf_stream_bytes=_stream_bytes\n",
    "_hf_stream_bytes=_stream_bytes\n"
    "def _using_hf_stream(): return _stream_bytes is _hf_stream_bytes\n"
    "def set_stream_source(fn):\n"
    "    \"\"\"replace the text source (local stand-in data for tests): fn(domain, seed, n) -> bytes\"\"\"\n"
    "    global _stream_bytes\n"
    "    _stream_bytes=fn\n")
rep("    if _stream_bytes is _hf_stream_bytes:            # new domains", "    if _using_hf_stream():                           # new domains")
open(sys.argv[2], 'w').write(src)
print('stage 0 written', len(src.splitlines()), 'lines')
