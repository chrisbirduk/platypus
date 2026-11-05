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
