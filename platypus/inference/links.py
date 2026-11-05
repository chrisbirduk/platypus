"""Link memory and cue repair."""
import math
import numpy as np
from platypus.inference.store import (
    _INFM,
)
from platypus.inference.probes import (
    _INF,
)

# ---- LINK MEMORY (inf_graph 2): links are not kept as a list. Every link read (a, b) is written into an associative matrix memory between sparse random
#      entity codes (k signed units of D); writes ACCUMULATE, so querying with b returns each partner with a strength equal to how often the link was read
#      (a count sketch: crosstalk ~ sqrt(load), small against k*k); a slow decay fades links that are never read again. Linking during sleep is a QUERY:
#      recall a bridge's partners, propose A-C for partner pairs, promote into the fast store when the value in bits clears the threshold.
_LM={'M':None}
def _lm_init(cfg):
    if _LM['M'] is None:
        D_=int(getattr(cfg,'inf_lm_dim',4096)); _LM.update(M=np.zeros((D_,D_),np.float32),D=D_,k=int(getattr(cfg,'inf_lm_k',16)),codes={},names=[],idx=[],sgn=[],nw=0,
                                                       fz=float(getattr(cfg,'inf_lm_fuzzy',0.0)),kn=int(getattr(cfg,'inf_lm_kn',48)))
def _lm_raw(n_):
    """(units, signed weights) of a name's code, not registered. inf_lm_fuzzy a > 0: the exact random code plus a CHARACTER-TRIGRAM code of
    weight a (lower-cased, padded): near-spellings share most trigram units, so a damaged or variant name recalls partly; total energy k"""
    import zlib as _zl
    r_=np.random.RandomState(_zl.crc32(n_)&0x7fffffff); ix_=r_.choice(_LM['D'],_LM['k'],replace=False).astype(np.int64); sg_=r_.choice([-1.0,1.0],_LM['k']).astype(np.float32)
    a_=_LM.get('fz',0.0)
    if a_<=0: return ix_,sg_
    w_=b'#'+n_.lower()+b'#'; tg_=sorted(set(w_[i_:i_+3] for i_ in range(len(w_)-2))); m_=max(1,_LM['kn']//max(1,len(tg_)))
    ui_=[]; us_=[]
    for t_ in tg_:
        rt_=np.random.RandomState(_zl.crc32(b'tri:'+t_)&0x7fffffff); ui_.extend(rt_.choice(_LM['D'],m_,replace=False).tolist()); us_.extend(rt_.choice([-1.0,1.0],m_).tolist())
    ui_=np.array(ui_[:_LM['kn']]+[0]*max(0,_LM['kn']-len(ui_)),np.int64); us_=np.array(us_[:_LM['kn']]+[0.0]*max(0,_LM['kn']-len(us_)),np.float32)
    nz_=max(1,int((us_!=0).sum())); us_*=a_*math.sqrt(_LM['k']/nz_)
    return np.concatenate([ix_,ui_]),np.concatenate([sg_,us_])/math.sqrt(1.0+a_*a_)
def _lm_code(n_):
    c_=_LM['codes'].get(n_)
    if c_ is None:
        ix_,sg_=_lm_raw(n_)
        c_=(ix_,sg_,len(_LM['names'])); _LM['codes'][n_]=c_; _LM['names'].append(n_); _LM['idx'].append(ix_); _LM['sgn'].append(sg_); _LM['arr']=None
    return c_
def _osa(a_,b_):
    """optimal string alignment distance (edit distance where swapping two neighbours costs 1)"""
    la_,lb_=len(a_),len(b_); d_=[[0]*(lb_+1) for _ in range(la_+1)]
    for i_ in range(la_+1): d_[i_][0]=i_
    for j_ in range(lb_+1): d_[0][j_]=j_
    for i_ in range(1,la_+1):
        for j_ in range(1,lb_+1):
            c_=0 if a_[i_-1]==b_[j_-1] else 1; d_[i_][j_]=min(d_[i_-1][j_]+1,d_[i_][j_-1]+1,d_[i_-1][j_-1]+c_)
            if i_>1 and j_>1 and a_[i_-1]==b_[j_-2] and a_[i_-2]==b_[j_-1]: d_[i_][j_]=min(d_[i_][j_],d_[i_-2][j_-2]+1)
    return d_[la_][lb_]
_CR={'n':-1,'idx':{},'low':{}}
def _cue_fix_word(cfg,w_):
    """cg26m CUE REPAIR of one word (bytes): the stored name it most plausibly is, or None (known, not name-like, too far, or a tie)"""
    nm_=_INFM.get('names') or {}
    if w_ in nm_ or len(w_)<4: return None
    if _CR['n']!=len(nm_):                                                      #   index of stored names by lower-case first letter, and by lower-case form
        ix_={}; lw_={}
        for n_ in nm_:
            if not n_[:1].isalpha(): continue
            l_=n_.lower(); ix_.setdefault(l_[:1],[]).append((l_,n_)); lw_.setdefault(l_,[]).append(n_)
        _CR.update(n=len(nm_),idx=ix_,low=lw_)
    wl_=w_.lower()
    if w_==wl_:                                                                 #   a lower-case word: only a name it equals ignoring case, and only if the
        if (_CR.get('df') or {}).get(wl_,0)>int(getattr(cfg,'inf_df',40)): return None   #   lower-case form is not itself a common word in the text
        m_=_CR['low'].get(wl_); return m_[0] if m_ and len(m_)==1 else None
    if not w_[:1].isupper(): return None
    D_=int(getattr(cfg,'inf_cue_repair_d',2)) if len(w_)>=6 else 1; b0_=None; d0_=99; d1_=99
    for l_,n_ in _CR['idx'].get(wl_[:1],()):
        if abs(len(l_)-len(wl_))>D_: continue
        d_=_osa(wl_,l_)
        if d_<d0_: d1_=d0_; d0_=d_; b0_=n_
        elif d_<d1_: d1_=d_
    return b0_ if (b0_ is not None and d0_<=D_ and d1_>d0_) else None
def _cue_fix(cfg,p_):
    """cg26m CUE REPAIR of a question (str): unknown name-like words replaced by the stored name they most plausibly are"""
    if not int(getattr(cfg,'inf_cue_repair',0)) or _INF.get('cr_off'): return p_
    import re as _re
    def f_(m_):
        w_=m_.group(0).encode('latin-1'); n_=_cue_fix_word(cfg,w_); return n_.decode('latin-1') if n_ is not None else m_.group(0)
    return _re.sub(r'[A-Za-z]{4,}',f_,p_)
def _lev(a_,b_):
    """edit distance of two byte strings"""
    p0_=list(range(len(b_)+1))
    for i_,x_ in enumerate(a_,1):
        c0_=[i_]
        for j_,y_ in enumerate(b_,1): c0_.append(min(p0_[j_]+1,c0_[j_-1]+1,p0_[j_-1]+(x_!=y_)))
        p0_=c0_
    return p0_[-1]
def _lm_sim(a_,b_):
    """code overlap of two names (1.0 = same name): exact part 0 for different names, trigram part ~ shared trigrams"""
    ia_,sa_=_LM['codes'][a_][:2] if a_ in _LM['codes'] else _lm_raw(a_); ib_,sb_=_LM['codes'][b_][:2] if b_ in _LM['codes'] else _lm_raw(b_)
    va_=np.zeros(_LM['D'],np.float32); np.add.at(va_,ia_,sa_); return float(va_[ib_]@sb_)/_LM['k']
def _lm_write(a_,c_,w_=1.0):
    ia_,sa_,_=_lm_code(a_); ic_,sc_,_=_lm_code(c_); M_=_LM['M']
    if _LM.get('fz',0.0)>0: np.add.at(M_,(ia_[:,None],ic_[None,:]),w_*np.outer(sa_,sc_)); np.add.at(M_,(ic_[:,None],ia_[None,:]),w_*np.outer(sc_,sa_))   # (units may repeat)
    else: M_[np.ix_(ia_,ic_)]+=w_*np.outer(sa_,sc_); M_[np.ix_(ic_,ia_)]+=w_*np.outer(sc_,sa_)
    _LM['nw']+=1
def _lm_pair(a_,c_):
    ia_,sa_,_=_lm_code(a_); ic_,sc_,_=_lm_code(c_); return float(sa_@_LM['M'][np.ix_(ia_,ic_)]@sc_)/(_LM['k']**2)
def _lm_partners(b_):
    """strength of every known entity as a partner of b_ (one query: k rows of the memory, then each entity's k units)"""
    if _LM.get('arr') is None: _LM['arr']=(np.stack(_LM['idx']),np.stack(_LM['sgn']))
    I_,S0_=_LM['arr']; ib_,sb_,_=_lm_code(b_)
    if _LM.get('arr') is None or I_.shape[0]!=len(_LM['names']): _LM['arr']=(np.stack(_LM['idx']),np.stack(_LM['sgn'])); I_,S0_=_LM['arr']
    v_=sb_@_LM['M'][ib_,:]; return (v_[I_]*S0_).sum(1)/(_LM['k']**2)
