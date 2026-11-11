"""Conduit dreams: replay rows, hints, CUDA-graph and slot-memory generation, NREM losses."""
from platypus.common import _fail
import contextlib
import math
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
from platypus.protection import (
    _PJ, _STF,
)
from platypus.private import (
    _BUFDET,
    _NGPD,
    _XDG,
)
from platypus.layers import (
    _CARRY, _SlotF,
)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _carry_save(): return {k_:_CARRY[k_] for k_ in ('on','nr','st','new','rows')}
def _carry_load(sv): _CARRY.update(sv)
def _uh_save(model): return [(b.__dict__.get('_uh'),b.__dict__.get('_um')) for b in model.blocks]
def _uh_load(model,sv):
    for b,(u_,m_) in zip(model.blocks,sv):
        if u_ is not None: b.__dict__['_uh']=u_; b.__dict__['_um']=m_
@contextlib.contextmanager
def _rp_tags(model):
    """replay passes: every row is the CURRENT domain (the training batch's per-row tags _s0_dom have another row count; a mismatch
    silently switched the private trunk units / FiLM codes off for those rows, so encoder and decoder ran different models)"""
    sv_=model.__dict__.get('_s0_dom'); model.__dict__['_s0_dom']=None
    try: yield
    finally: model.__dict__['_s0_dom']=sv_
def recon_loss(model,cfg,x,y,dom):
    with _rp_tags(model): return _recon_loss(model,cfg,x,y,dom)
def _recon_loss(model,cfg,x,y,dom):
    """REPLAY TRAINING: each window regenerated starting from its own (differentiable) end-of-window memory."""
    n=min(int(cfg.recon_rows),x.shape[0]); xr,yr=x[:n],y[:n]; sv=_carry_save(); suh=_uh_save(model)
    try:
        _CARRY.update(on=True,nr=n,st={},new={},rows=None,grad=True); model.features(xr); snap=dict(_CARRY['new'])
        _CARRY.update(on=True,nr=n,st=snap,new={},rows=None,grad=False)
        l_=F.cross_entropy(model(xr,dom).reshape(-1,256),yr.reshape(-1))
    finally:
        _CARRY.pop('grad',None); _carry_load(sv); _uh_load(model,suh)
    st=model.__dict__.setdefault('_rc_stat',[0.0,0]); st[0]+=float(l_.detach()); st[1]+=1
    return l_
def _cd_g(model,dev,cfg):
    if not hasattr(model,'_cd_gen'):
        g_=torch.Generator(device=dev); g_.manual_seed(int(cfg.seed)+41); object.__setattr__(model,'_cd_gen',g_)
    return model._cd_gen
def _cd_slices(model,cfg,x,n,L=None):
    B,T=x.shape; L=min(int(cfg.cd_seg) if L is None else int(L),T); g_=_cd_g(model,x.device,cfg)
    r_=torch.randint(0,B,(n,),generator=g_,device=x.device); s_=torch.randint(0,T-L+1,(n,),generator=g_,device=x.device)
    if L==T and n<=B: r_=torch.randperm(B,generator=g_,device=x.device)[:n]          # FULL WINDOWS: distinct rows of the batch
    return x[r_[:,None],s_[:,None]+torch.arange(L,device=x.device)[None]].long()        # v9d: one gather (was 2 syncs per window)
_RSS={'pe':[],'pool':[]}                                                   # v8: p_exact of the chosen pieces / of the pool
def _rs_rowloss(model,cfg,X,dom):
    """per-piece teacher-forced replay loss (no gradient): [mark] + piece[:-1] -> piece from the piece's own memory, live model"""
    n=X.shape[0]; sv=_carry_save(); suh=_uh_save(model)
    try:
        with torch.no_grad():
            _CARRY.update(on=True,nr=n,st={},new={},rows=None); model.features(X); snap=dict(_CARRY['new'])
            _CARRY.update(on=True,nr=n,st=snap,new={},rows=None)
            Xin=torch.cat([torch.full((n,1),int(cfg.cd_mark),dtype=torch.long,device=X.device),X[:,:-1]],1)
            l_=F.cross_entropy(model(Xin,dom).reshape(-1,256).float(),X.reshape(-1),reduction='none').view(n,-1).mean(1)
    finally: _carry_load(sv); _uh_load(model,suh)
    return torch.nan_to_num(l_,nan=1e3,posinf=1e3)
def _rs_pick(model,cfg,Cd,k,dom,g_):
    """v8 SELECTIVE REPLAY TRAINING: k of the candidate pieces Cd: round(rs_frac x k) sampled in proportion to p(1 - p) (borderline), the rest
    uniformly from the remaining candidates"""
    m=Cd.shape[0]; k=min(int(k),m)
    if k<=0 or m<=k: return Cd[:k]
    ks=int(round(float(cfg.rs_frac)*k)); pe=torch.exp(-Cd.shape[1]*_rs_rowloss(model,cfg,Cd,dom)); w=pe*(1-pe)+1e-6
    sel=torch.multinomial(w,ks,replacement=False,generator=g_) if ks>0 else torch.zeros(0,dtype=torch.long,device=Cd.device)
    if cfg.cd_nosync:                                                       # v9d: uniform among the rest via random keys (no nonzero)
        rk_=torch.rand(m,generator=g_,device=Cd.device); rk_[sel]=-1.0; ri=rk_.topk(k-ks).indices
        if ks>0: _RSS['pe'].append(pe[sel].mean())
        _RSS['pool'].append(pe.mean())
    else:
        rest=torch.ones(m,dtype=torch.bool,device=Cd.device); rest[sel]=False; ri=rest.nonzero().flatten()
        ri=ri[torch.randperm(len(ri),generator=g_,device=Cd.device)[:k-ks]]
        if ks>0: _RSS['pe'].append(float(pe[sel].mean()))
        _RSS['pool'].append(float(pe.mean()))
    return Cd[torch.cat([sel,ri])].contiguous()
def _rs_msg():
    if not _RSS['pool']: return ""
    m_=f" | selective: borderline pieces p_exact {_fm(_RSS['pe']) if _RSS['pe'] else float('nan'):.2f} (pool {_fm(_RSS['pool']):.2f})"
    _RSS['pe']=[]; _RSS['pool']=[]; return m_
def recon_loss_mark(model,cfg,x,dom):
    with _rp_tags(model): return _recon_loss_mark(model,cfg,x,dom)
def _recon_loss_mark(model,cfg,x,dom,X=None):
    """REPLAY TRAINING v2: segments decoded as [mark] + segment[:-1] -> segment from their own (differentiable) end-of-segment memory.
    X: the segments themselves (v6: exact replays in conduit passes); otherwise recon_rows slices of the batch."""
    if X is None:
        n=min(int(cfg.recon_rows),x.shape[0])
        if cfg.rs_sel: X=_rs_pick(model,cfg,_cd_slices(model,cfg,x,max(n,int(cfg.rs_pool)*n)),n,dom,_cd_g(model,x.device,cfg))   # v8
        else: X=_cd_slices(model,cfg,x,n)
    else: n=X.shape[0]
    sv=_carry_save(); suh=_uh_save(model)
    try:
        _CARRY.update(on=True,nr=n,st={},new={},rows=None,grad=True); model.features(X); snap=dict(_CARRY['new'])
        _CARRY.update(on=True,nr=n,st=snap,new={},rows=None,grad=False)
        Xin=torch.cat([torch.full((n,1),int(cfg.cd_mark),dtype=torch.long,device=X.device),X[:,:-1]],1)
        l_=F.cross_entropy(model(Xin,dom).reshape(-1,256).float(),X.reshape(-1))
        if not bool(torch.isfinite(l_.detach())):                        # NON-FINITE replay loss: skip this step's replay term (counted)
            model.__dict__['_rc_bad']=model.__dict__.get('_rc_bad',0)+1; l_=torch.zeros((),device=X.device)
    finally:
        _CARRY.pop('grad',None); _carry_load(sv); _uh_load(model,suh)
    st=model.__dict__.setdefault('_rc_stat',[0.0,0]); st[0]=st[0]+(l_.detach() if cfg.cd_nosync else float(l_.detach())); st[1]+=1
    return l_
@contextlib.contextmanager
def _cd_weights(model):
    """the frozen conduit teacher: visit-start weights swapped in (data pointers only, restored on exit)"""
    P=model.__dict__.get('_cd_P')
    if P is None:
        yield; return
    ps=dict(model.named_parameters())
    for k_ in P: ps[k_].data,P[k_]=P[k_],ps[k_].data
    try: yield
    finally:
        for k_ in P: ps[k_].data,P[k_]=P[k_],ps[k_].data
_ENC_STOP={'k':None}
_V97={'vp':[],'pn':[],'pd':[],'ag':[],'agd':[],'rep':0,'imp':[],'chk':[],'dis':[],'fw':[],'skip':0,'calls':0}
_HINT={'calls':0,'bytes':0,'tot':0,'wrong':0,'wcov':0,'pieces':0}               # cg12b hint statistics (device tensors, read at the visit end)
@torch.no_grad()
def _cd_hints(model,cfg,st_,m_,L,dom,M,Xc):
    """cg12b HINTS for m_ pieces: one teacher-forced pass with each piece's trace over its TRUE bytes (written with the trace, while the
    text is at hand); margin = log p(true byte) - best other; the lowest cd_hint_frac share of margins (<= cd_hint_piece per piece) are hinted.
    Returns (mask (m_,L) bool, bytes (m_,L))."""
    dev=Xc.device; sv_=_carry_save()
    try:
        Yin=torch.cat([torch.full((m_,1),M,dtype=torch.long,device=dev),Xc[:,:-1]],1)
        _CARRY.update(on=True,nr=m_,st=st_,new={},rows=None)
        lp=F.log_softmax(model(Yin,dom).float(),-1)
    finally: _carry_load(sv_)
    tl=lp.gather(-1,Xc[...,None]).squeeze(-1); ot=lp.scatter(-1,Xc[...,None],float('-inf')).max(-1).values; mg=tl-ot
    rk=torch.argsort(torch.argsort(mg,1),1); mgc=torch.where(rk<int(cfg.cd_hint_piece),mg,torch.full_like(mg,float('inf')))
    wr_=mg<0
    if float(cfg.cd_hint_adapt)>0:                                            # cg13 ADAPTIVE BUDGET: cd_hint_adapt x the teacher-forced wrong bytes,
        k=int(math.ceil(float(cfg.cd_hint_adapt)*float(wr_.sum())))          #   between cd_hint_min and cd_hint_max of the bytes (cg12b at 3 percent:
        k=min(max(k,int(round(float(cfg.cd_hint_min)*m_*L))),int(round(float(cfg.cd_hint_max)*m_*L)))   #   Wikipedia 1.95 percent wrong, 91 percent covered;
    else: k=int(round(float(cfg.cd_hint_frac)*m_*L))                          #   DM Mathematics 4.04 percent wrong, 74 percent covered)
    k=max(1,k); idx=torch.topk(-mgc.flatten(),min(k,m_*L)).indices
    hm=torch.zeros(m_*L,dtype=torch.bool,device=dev); hm[idx]=True; hm=hm.view(m_,L)&torch.isfinite(mgc)
    _HINT['calls']+=1; _HINT['bytes']=_HINT['bytes']+hm.sum(); _HINT['tot']+=m_*L
    _HINT['wrong']=_HINT['wrong']+wr_.sum(); _HINT['wcov']=_HINT['wcov']+(wr_&hm).sum(); _HINT['pieces']=_HINT['pieces']+hm.any(1).sum()
    return hm,Xc
def _cd_fb(model,snap):
    fb_=model.__dict__.get('_cd_fb')
    if fb_ is None:
        ix_=[i for i,b in enumerate(model.blocks) if id(b) in snap]; fb_=(ix_[0],id(model.blocks[ix_[0]])); model.__dict__['_cd_fb']=fb_
    return fb_
def _cd_encode_chk(model,cfg,X):
    if cfg.cd_first_block:
        _ENC_STOP['k']=model.__dict__['_cd_fb'][0]
        try: return _cd_encode(model,X)
        finally: _ENC_STOP['k']=None
    return _cd_encode(model,X)
def _cd_kdist(model,enc,ref,m,L):
    """(m, L): relative distance of the replay's first-delta-block write-key rows from the stored ones (max over heads)"""
    bid_=model.__dict__['_cd_fb'][1]; Ke=enc[bid_][1].float(); Kr=ref[bid_][1].float()
    d_=(Ke-Kr).norm(dim=-1)/Kr.norm(dim=-1).clamp_min(1e-12); return d_.reshape(m,-1,d_.shape[-1])[:,:,-L:].amax(1)
def _cd_vcheck(model,cfg,Y,ref,m,L,eps):
    """v9e: (mismatch e - exact iff e <= eps, verified prefix length j) of m replays Y against their stored traces ref"""
    enc=_cd_encode_chk(model,cfg,Y); bad_=_cd_kdist(model,enc,ref,m,L)>float(model.__dict__.get('_cd_ktol',cfg.cd_keytol))
    j_=torch.where(bad_.any(1),bad_.float().argmax(1),torch.full((m,),L,device=Y.device,dtype=torch.float32)).long()
    if cfg.cd_first_block: return (j_<L).float(),j_
    e_=_cd_mis(enc,ref,m); ok_=e_<=eps; _V97['dis'].append((~ok_&(j_>=L)).sum())
    j_=torch.where(ok_,torch.full_like(j_,L),torch.where(j_>=L,torch.zeros_like(j_),j_)); return e_,j_
_CBUF={'v':0,'ldom':None,'d':{},'mix':0,'calls':0,'src':{}}
def _cbuf_take(model,cfg,dom,n0,dev):
    """v9f: visit bookkeeping + the buffered windows for this step: list of (source domain, index)"""
    if _CBUF['ldom']!=dom:
        _CBUF['v']+=1; _CBUF['ldom']=dom; _CBUF['d'][dom]={'x':[],'t':[],'m':[],'visit':_CBUF['v']}
        for d_ in [d_ for d_,e_ in _CBUF['d'].items() if e_['visit']<_CBUF['v']-int(cfg.cd_buf_visits)]: del _CBUF['d'][d_]
    return []
    pool_=[(d_,i_) for d_,e_ in _CBUF['d'].items() if d_!=dom for i_ in range(len(e_['x']))]
    k_=min(int(round(float(cfg.cd_buf)*n0)),len(pool_),n0-1)
    if k_<=0: return []
    pk_=torch.randperm(len(pool_),generator=_cd_g(model,dev,cfg),device=dev)[:k_].tolist()
    return [pool_[i_] for i_ in pk_]
def _cd_encode(model,X):
    _CARRY.update(on=True,nr=X.shape[0],st={},new={},rows=None); model.features(X)
    return {k_:(tuple(t_.clone() for t_ in v_) if isinstance(v_,tuple) else v_.clone()) for k_,v_ in _CARRY['new'].items() if not isinstance(k_,tuple)}
def _cd_rows(st,idx,n):
    f_=lambda v_: v_.reshape(n,-1,*v_.shape[1:])[idx].reshape(-1,*v_.shape[1:])
    return {k_:(tuple(f_(t_) for t_ in v_) if isinstance(v_,tuple) else f_(v_)) for k_,v_ in st.items()}
def _cd_cat(sts):
    return {k_:(tuple(torch.cat([s_[k_][i] for s_ in sts]) for i in range(len(sts[0][k_]))) if isinstance(sts[0][k_],tuple) else torch.cat([s_[k_] for s_ in sts])) for k_ in sts[0]}
def _fgram(Ua,Ka,Ub,Kb): return (torch.bmm(Ua,Ub.transpose(1,2))*torch.bmm(Ka,Kb.transpose(1,2))).sum((1,2))   # <Ua^T Ka, Ub^T Kb>_F
_MIS32={'on':False}
def _fgram32(Ua,Ka,Ub,Kb):
    """<Ua^T Ka, Ub^T Kb>_F with the (L x L) Grams in full fp32 (no TF32) and their elementwise product + sum in fp64"""
    pv_=torch.backends.cuda.matmul.allow_tf32 if Ua.is_cuda else None
    if pv_ is not None: torch.backends.cuda.matmul.allow_tf32=False
    try: G_=torch.bmm(Ua,Ub.transpose(1,2)).double(); H_=torch.bmm(Ka,Kb.transpose(1,2)).double()
    finally:
        if pv_ is not None: torch.backends.cuda.matmul.allow_tf32=pv_
    return (G_*H_).sum((1,2))
def _cd_mis(a,b,n):
    num=0.0; den=0.0
    for k_ in b:
        if isinstance(b[k_],tuple):                                     # FACTORED: |Sa - Sb|^2 = <Sa,Sa> + <Sb,Sb> - 2 <Sa,Sb> (float64 Grams)
            if _MIS32['on']:                                            # v9c: Gram matrices in fp32 (TF32 off), their products + sums in fp64
                Ua,Ka=(t_.float() for t_ in a[k_]); Ub,Kb=(t_.float() for t_ in b[k_])
                sbb=_fgram32(Ub,Kb,Ub,Kb); d_=(_fgram32(Ua,Ka,Ua,Ka)+sbb-2.0*_fgram32(Ua,Ka,Ub,Kb)).clamp_min(0)
                num=num+d_.reshape(n,-1).sum(1).float(); den=den+sbb.reshape(n,-1).sum(1).float()
                continue
            Ua,Ka=(t_.double() for t_ in a[k_]); Ub,Kb=(t_.double() for t_ in b[k_])
            sbb=_fgram(Ub,Kb,Ub,Kb); d_=(_fgram(Ua,Ka,Ua,Ka)+sbb-2.0*_fgram(Ua,Ka,Ub,Kb)).clamp_min(0)
            num=num+d_.reshape(n,-1).sum(1).float(); den=den+sbb.reshape(n,-1).sum(1).float()
        else:
            num=num+((a[k_].float()-b[k_].float()).reshape(n,-1)**2).sum(1); den=den+(b[k_].float().reshape(n,-1)**2).sum(1)
    return num/den.clamp_min(1e-12)
_INC={'on':False,'k':1}
_RESET={'m':None}
FOKS={'skip':[],'est':[],'true':[],'k':[],'runs':[]}                                                          # v7: (B,T) bool, memory reset positions for the student pass
_CDCONF={'c':None}                                                         # v7: per-row confidence of the last _cd_gen call
_FOKD={}                                                                   # v7: per-domain gate calibration
def _fok_dom(d):
    return _FOKD.setdefault(int(d),{'c':[],'e':[],'thr':float('inf'),'exr':0.0,'n':0,'nc':0,'exrT':None,'ncall':0})
def _fm(v):
    """mean of a list of floats and/or 0-d tensors (one read-back)"""
    if not v: return 0.0
    return float(torch.stack([t_.detach().float().reshape(()).cpu() if torch.is_tensor(t_) else torch.tensor(float(t_)) for t_ in v]).mean())
def _fsum(v):
    return float(sum(float(t_) for t_ in v)) if v else 0.0
def _fok_thr_gpu(fd_,cfg):
    """v9d: the gate threshold from the GPU calibration buffer (lowest confidence with <= cd_fok_far inexact tries above it, >= 200 above)"""
    c_=torch.cat(fd_['c']); e_=torch.cat(fd_['e'])
    if c_.numel()>40000: c_=c_[-40000:]; e_=e_[-40000:]
    fd_['c']=[c_]; fd_['e']=[e_]
    o_=torch.argsort(-c_); cs_=c_[o_]; rk_=torch.arange(1,c_.numel()+1,device=c_.device,dtype=torch.float32)
    bad_=torch.cumsum(1.0-e_[o_].float(),0)/rk_; ok_=(bad_<=float(cfg.cd_fok_far))&(rk_>=200)
    li_=(ok_.float()*rk_).argmax()
    fd_['thr']=torch.where(ok_.any(),cs_[li_],torch.full((),float('inf'),device=c_.device))
_CDT={'on':False,'t':0.0,'acc':{},'n':0,'calls':0}                 # conduit timing (every 20th call, synchronised)
def _cdt(tag):
    if not _CDT['on']: return
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_=time.time(); _CDT['acc'][tag]=_CDT['acc'].get(tag,0.0)+t_-_CDT['t']; _CDT['t']=t_
# ---------------- CUDA-GRAPH CONDUIT GENERATION (cd_graph) ----------------
_CDG={'g':{},'seen':{},'dom':None,'off':None,'calls':{},'n_rep':0,'n_eag':0,'n_cap':0,'t_cap':0.0,'n_chk':0,'n_bit':0,'maxd':0.0,'minag':1.0,'pad_rows':0,'rows':0}
class _CdgFake:
    """CPU test stand-in: 'replay' re-runs the recorded function and writes into the same output tensors (static-buffer semantics)."""
    def __init__(s,fn,lg,new): s.fn=fn; s.lg=lg; s.new=new
    def replay(s):
        lg_,new_=s.fn(); s.lg.copy_(lg_)
        for k_ in s.new:
            if isinstance(s.new[k_],tuple):
                for a_,b_ in zip(s.new[k_],new_[k_]): a_.copy_(b_)
            else: s.new[k_].copy_(new_[k_])
def _cdg_sig(st,m): return tuple((k_,tuple((tuple(t_.shape[1:]),t_.shape[0]//m,str(t_.dtype)) for t_ in (st[k_] if isinstance(st[k_],tuple) else (st[k_],)))) for k_ in sorted(st))
def _cdg_fill(dst,src,m):
    """copy src (m rows, or already padded) into the static buffer dst (mb rows); extra rows repeat row 0"""
    if src.shape[0]==dst.shape[0]: dst.copy_(src); return
    r_=src.shape[0]//m; mb=dst.shape[0]//r_; dv=dst.view(mb,r_,*dst.shape[1:]); sv=src.view(m,r_,*src.shape[1:])
    dv[:m].copy_(sv)
    if mb>m: dv[m:].copy_(sv[:1].expand(mb-m,*sv.shape[1:]))
def _cdg_fill_st(dst,src,m):
    for k_ in dst:
        if isinstance(dst[k_],tuple):
            for a_,b_ in zip(dst[k_],src[k_]): _cdg_fill(a_,b_,m)
        else: _cdg_fill(dst[k_],src[k_],m)
def _cdg_slice(st,m,mb):
    """the first m rows of a padded state (for the eager fallback / self-check)"""
    if mb==m: return st
    f_=lambda t_: t_.view(mb,t_.shape[0]//mb,*t_.shape[1:])[:m].reshape(m*(t_.shape[0]//mb),*t_.shape[1:])
    return {k_:(tuple(f_(t_) for t_ in v_) if isinstance(v_,tuple) else f_(v_)) for k_,v_ in st.items()}
def _cdg_once(model,dom,Yw,st,nr):
    """ONE generation forward, exactly as the eager loop in _cd_gen does it (incremental mode)"""
    _CARRY.update(on=True,nr=nr,st=st,new={},rows=None); _INC['on']=True
    try: lg=model(Yw,dom)[:,-1].float()
    finally: _INC['on']=False
    return lg,{k_:v_ for k_,v_ in _CARRY['new'].items() if not isinstance(k_,tuple)}
def _cdg_fp(model):
    """parameters by storage (the conduit teacher is swapped in by data pointer) + the forward's global switches. (Buffers are left out: the
    training statistics among them are re-created every step; a buffer the forward reads that is replaced would be caught by the self-check.)"""
    return hash((tuple((p_.data_ptr(),tuple(p_.shape)) for p_ in model.parameters()),
                 bool(_BUFDET['on']),bool(_STF['on']),bool(_XDG['on']),_XDG.get('cur'),bool(_PJ['on']),_ENC_STOP['k'],bool(_CARRY.get('fac')),
                 bool(_NGPD['on']),int(_INC['k']),bool(model.training),torch.is_grad_enabled()))
def _cdg_plan(model,cfg,m,dom,W_,st0,dev):
    """which graph set this call uses (None: eager)"""
    if _CDG['off'] is not None or not (dev.type=='cuda' or cfg.cd_graph_fake): return None
    if _CDG['dom']!=dom:                                                       # one domain per visit: free the previous visit's graphs
        _CDG['g'].clear(); _CDG['seen'].clear(); _CDG['dom']=dom; _CDG['pool']=None
        if dev.type=='cuda': torch.cuda.synchronize(); torch.cuda.empty_cache()
    if cfg.cd_graph_pad:
        mb=64
        while mb<m: mb*=2
    else: mb=m
    key=(dom,mb,int(W_),_cdg_sig(st0,m),_cdg_fp(model))
    if key not in _CDG['g']:
        if not cfg.cd_graph_pad:
            _CDG['seen'][key]=_CDG['seen'].get(key,0)+1
            if _CDG['seen'][key]<int(cfg.cd_graph_min): return None
        if len(_CDG['g'])>=48: _CDG['g'].pop(next(iter(_CDG['g'])))                        # bound the number of graph sets (memory)
        if dev.type=='cuda':
            fr_=(torch.cuda.mem_get_info()[0]+torch.cuda.memory_reserved()-torch.cuda.memory_allocated())/2**30   # free + PyTorch's unused cache
            if fr_<float(cfg.cd_graph_free_gb):                                   # MEMORY GUARD: this batch size stays eager
                _CDG['n_skip']=_CDG.get('n_skip',0)+1
                if _CDG['n_skip']==1: print(f"  [cd graph] only {fr_:.1f} GB free (< {float(cfg.cd_graph_free_gb):.0f}): new graph sets skipped while memory is short",flush=True)
                return None
            if _CDG.get('pool') is None: _CDG['pool']=torch.cuda.graph_pool_handle()   # ONE memory pool shared by every graph (each graph's outputs
        _CDG['g'][key]={'t':{},'pool':_CDG.get('pool'),'calls':0,'st':None}        #   are read before any other graph is replayed)
    G_=_CDG['g'][key]; G_['calls']+=1
    return {'G':G_,'mb':mb,'chk':(G_['calls']==1 or G_['calls']%max(1,int(cfg.cd_graph_check))==0)}
def _cdg_capture(model,cfg,G_,t,Yw,cur,m,mb,dom,padded):
    if _CDG['n_cap']>=int(getattr(cfg,'cd_graph_max_rec',800)):              # SAFETY: recording should be rare; runaway re-recording -> eager
        raise RuntimeError(f"{_CDG['n_cap']} graphs recorded so far: re-recording too often")
    t0_=time.time(); dev=Yw.device
    Ys=torch.empty((mb,Yw.shape[1]),dtype=Yw.dtype,device=dev); _cdg_fill(Ys,Yw,m)
    n_=mb if padded else m                                                     # rows of the incoming state (padded after the first byte)
    mk1=lambda t_: torch.empty(((t_.shape[0]//n_)*mb,*t_.shape[1:]),dtype=t_.dtype,device=dev)
    sk_=tuple((k_,tuple(tuple(mk1(x_).shape) for x_ in (v_ if isinstance(v_,tuple) else (v_,)))) for k_,v_ in sorted(cur.items()))
    if G_.get('st') is None: G_['st']={}
    if sk_ not in G_['st']: G_['st'][sk_]={k_:(tuple(mk1(x_) for x_ in v_) if isinstance(v_,tuple) else mk1(v_)) for k_,v_ in cur.items()}
    Ss=G_['st'][sk_]; _cdg_fill_st(Ss,cur,m)                                   # static state buffers shared by the byte positions with the same shapes
    fn=lambda: _cdg_once(model,dom,Ys,Ss,mb)
    if dev.type!='cuda':                                                       # CPU test: same plumbing, forward re-run at every 'replay'
        lg_,new_=fn(); gr=_CdgFake(fn,lg_.clone(),{k_:(tuple(x_.clone() for x_ in v_) if isinstance(v_,tuple) else v_.clone()) for k_,v_ in new_.items()})
        G_['t'][t]={'g':gr,'Y':Ys,'st':Ss,'lg':gr.lg,'new':gr.new}; _CDG['n_cap']+=1; _CDG['t_cap']+=time.time()-t0_; return
    if _CDG['n_cap']==0:                                                       # first recording of the run: a forward that synchronises with the CPU
        prev_=torch.cuda.get_sync_debug_mode(); torch.cuda.set_sync_debug_mode(2)   #   cannot be recorded -> find out BEFORE recording (raises here)
        try: fn()
        finally: torch.cuda.set_sync_debug_mode(prev_)
    s_=torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for _ in range(2): fn()                                                #   warm-up (lazy initialisation, workspaces) off the recording
    torch.cuda.current_stream().wait_stream(s_)
    gr=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr,pool=G_['pool']): lg_,new_=fn()
    G_['t'][t]={'g':gr,'Y':Ys,'st':Ss,'lg':lg_,'new':new_}; _CDG['n_cap']+=1
    torch.cuda.synchronize(); torch.cuda.empty_cache()                         # hand fragmented cache back (recording is rare: once per set and position)
    _CDG['t_cap']+=time.time()-t0_
def _cdg_step(model,cfg,P_,t,Yw,cur,m,dom,padded):
    """generation forward at byte position t through the recorded graph -> (logits (m, V), state, state padded?)"""
    G_,mb=P_['G'],P_['mb']
    try:
        if t not in G_['t']: _cdg_capture(model,cfg,G_,t,Yw,cur,m,mb,dom,padded)
        e_=G_['t'][t]
        if e_['Y'].shape[1]!=Yw.shape[1]: raise RuntimeError(f'window length changed ({e_["Y"].shape[1]} -> {Yw.shape[1]})')
        _cdg_fill(e_['Y'],Yw,m); _cdg_fill_st(e_['st'],cur,m); e_['g'].replay(); _CDG['n_rep']+=1; _CDG['rows']+=m; _CDG['pad_rows']+=mb-m
        lg=e_['lg'][:m]
        if P_['chk']:                                                         # SELF-CHECK: the eager forward on the same inputs
            sv_=_carry_save(); lge,_=_cdg_once(model,dom,Yw,_cdg_slice(cur,m,mb) if padded else cur,m); _carry_load(sv_)
            d_=float((lge-lg).abs().max()); sc_=float(lge.abs().max())+1e-6; ag_=float((lge.argmax(-1)==lg.argmax(-1)).float().mean())
            _CDG['n_chk']+=1; _CDG['n_bit']+=int(torch.equal(lge,lg)); _CDG['maxd']=max(_CDG['maxd'],d_/sc_); _CDG['minag']=min(_CDG['minag'],ag_)
            if d_/sc_>float(cfg.cd_graph_tol) or ag_<0.95:                    #   (a stale or broken graph gives unrelated logits; padding only rounding)
                raise RuntimeError(f'self-check failed at byte {t}: max logit difference {d_/sc_:.2e} of the largest logit, same top byte {ag_:.4f}')
        return lg,e_['new'],True
    except Exception as ex_:
        _fail('conduit:386')
        _CDG['off']=repr(ex_)[:300]; _CDG['g'].clear(); _CDG['pool']=None
        if Yw.is_cuda: torch.cuda.synchronize(); torch.cuda.empty_cache()          # free the graphs' memory before carrying on eagerly
        print(f"  [cd graph] OFF from here on (eager generation, unchanged results): {_CDG['off']}",flush=True)
        lg_,new_=_cdg_once(model,dom,Yw,_cdg_slice(cur,m,mb) if padded else cur,m); return lg_,new_,False
def _cdg_rows(st,a,b,m):
    """rows a..b of a state dict whose tensors have m row groups"""
    f_=lambda t_: t_.view(m,t_.shape[0]//m,*t_.shape[1:])[a:b].reshape((b-a)*(t_.shape[0]//m),*t_.shape[1:])
    return {k_:(tuple(f_(x_) for x_ in v_) if isinstance(v_,tuple) else f_(v_)) for k_,v_ in st.items()}
def _cdg_plans(model,cfg,m,dom,W_,st0,dev):
    """ONE plan per chunk of at most cd_graph_rows rows, made once per generation call from the starting state (the state's shapes grow
    with the byte position, so planning per position would give every position its own graph set)"""
    R_=max(64,int(getattr(cfg,'cd_graph_rows',512))); out=[]
    for a_ in range(0,m,R_):
        b_=min(m,a_+R_); out.append((a_,b_,_cdg_plan(model,cfg,b_-a_,dom,W_,_cdg_rows(st0,a_,b_,m) if (a_>0 or b_<m) else st0,dev)))
    return out
def _cdg_gen_step(model,cfg,t,Yw,cur,m,dom,plans):
    """one byte position for m rows through the graphs, chunk by chunk (state passed between positions unpadded)"""
    if len(plans)==1:
        a_,b_,P_=plans[0]
        if P_ is None: return None
        lg,new,pd_=_cdg_step(model,cfg,P_,t,Yw,cur,m,dom,False)
        return lg,(_cdg_slice(new,m,P_['mb']) if pd_ else new)
    lgs=[]; news=[]
    for a_,b_,P_ in plans:
        ca_=_cdg_rows(cur,a_,b_,m)
        if P_ is None or _CDG['off'] is not None:                             # (eager for this chunk: memory guard or graphs off)
            lg_,new_=_cdg_once(model,dom,Yw[a_:b_],ca_,b_-a_)
        else:
            lg_,new_,pd_=_cdg_step(model,cfg,P_,t,Yw[a_:b_],ca_,b_-a_,dom,False)
            if pd_: new_=_cdg_slice(new_,b_-a_,P_['mb'])
        lgs.append(lg_.clone()); news.append({k_:(tuple(x_.clone() for x_ in v_) if isinstance(v_,tuple) else v_.clone()) for k_,v_ in new_.items()})
    def catst(ds):
        out={}
        for k_ in ds[0]:
            if isinstance(ds[0][k_],tuple): out[k_]=tuple(torch.cat([d_[k_][i] for d_ in ds],0) for i in range(len(ds[0][k_])))
            else: out[k_]=torch.cat([d_[k_] for d_ in ds],0)
        return out
    return torch.cat(lgs,0),catst(news)
def _cdg_msg():
    if not (_CDG['n_rep'] or _CDG['off']): return ''
    return (f" || cuda graphs: {_CDG['n_rep']} graph forwards ({_CDG['n_cap']} recorded, {_CDG['t_cap']:.1f}s), padding {_CDG['pad_rows']/max(1,_CDG['rows']+_CDG['pad_rows']):.0%} of rows"
            f" | self-checks {_CDG['n_chk']}: bit-identical {_CDG['n_bit']}, max logit diff {_CDG['maxd']:.1e}, same top byte >= {_CDG['minag']:.4f}"+(f" | {torch.cuda.mem_get_info()[0]/2**30:.1f} GB free, peak allocated {torch.cuda.max_memory_allocated()/2**30:.1f} GB" if torch.cuda.is_available() else '')+(f" | sets skipped for memory {_CDG.get('n_skip',0)}" if _CDG.get('n_skip') else '')+(f" | OFF: {_CDG['off'][:120]}" if _CDG['off'] else ''))
# ---------------- v9.8 cg5: SLOT MEMORY GENERATION (cd_slot), with CUDA graphs bound to one persistent buffer ----------------
_CG5={'key':None,'cap':0,'B':None,'Y':None,'LG':None,'g':{},'pool':None,'nchk':0,'chk_calls':0}
def release_gen_graphs(why='end of training'):
    """10 Nov: free every recorded generation graph, its static buffers and memory pools, and switch graph capture off (generation runs eager
    from here). Called when training ends: the end-of-run readouts need memory more than graph speed (seed 3 ran out with ~80 GiB allocated)"""
    n1_=len(_CDG['g']); n2_=len(_CG5['g'])
    _CDG['g'].clear(); _CDG['pool']=None; _CDG['off']=why
    _CG5['g'].clear(); _CG5['B']=None; _CG5['Y']=None; _CG5['LG']=None; _CG5['pool']=None; _CG5['key']=None; _CG5['cap']=0
    return n1_,n2_

def _slot_ok(st0,m):
    """the starting state is a set of factored memories (U, K) for m rows (no other carried statistics)"""
    why=None
    if not st0: why='empty state'
    for k_,v_ in (st0 or {}).items():
        if why: break
        if isinstance(k_,tuple): why=f'extra carried statistic {k_[1:]}'
        elif not (isinstance(v_,tuple) and len(v_)==2): why=f'memory not factored ({type(v_).__name__})'
        elif not (v_[0].dim()==3 and v_[0].shape[0]%m==0): why=f'factor shape {tuple(v_[0].shape)} for {m} rows'
        elif v_[0].dtype!=torch.float32 or v_[1].dtype!=torch.float32: why=f'factor type {v_[0].dtype}'
    if why and not _CG5.get('why'): _CG5['why']=why; print(f"  [slot memory] not used for this state: {why} (old generation path)",flush=True)
    return why is None
def _slot_alloc(st0,m,cap,R,dev):
    return {k_:(torch.zeros((cap*(v_[0].shape[0]//m),R,v_[0].shape[2]),device=dev),torch.zeros((cap*(v_[0].shape[0]//m),R,v_[1].shape[2]),device=dev),
                torch.zeros((cap*(v_[0].shape[0]//m),R),device=dev),torch.zeros((cap*(v_[0].shape[0]//m),),device=dev)) for k_,v_ in st0.items()}
def _slot_fill(B_,st0,m):
    for k_,(U,K,Gs,Gt) in B_.items():
        U0,K0=st0[k_]; n_=U0.shape[0]; r0=U0.shape[1]
        U[:n_,:r0].copy_(U0); U[:n_,r0:].zero_(); K[:n_,:r0].copy_(K0); K[:n_,r0:].zero_(); Gs[:n_].zero_(); Gt[:n_].zero_()
def _slot_view(B_,a,b,m_tot,r0,i,mh_of):
    return {k_:_SlotF(U[a*mh_of[k_]:b*mh_of[k_]],K[a*mh_of[k_]:b*mh_of[k_]],Gs[a*mh_of[k_]:b*mh_of[k_]],Gt[a*mh_of[k_]:b*mh_of[k_]],r0,i) for k_,(U,K,Gs,Gt) in B_.items()}
def _slot_fwd(model,dom,Yv,sl,nr):
    _CARRY.update(on=True,nr=nr,st=sl,new={},rows=None); _INC['on']=True
    try: return model(Yv,dom)[:,-1].float()
    finally: _INC['on']=False
def _slot_clone(sl): return {k_:_SlotF(v_.U.clone(),v_.K.clone(),v_.Gs.clone(),v_.Gt.clone(),v_.r0,v_.i) for k_,v_ in sl.items()}
def _slot_copy(dst,src):
    for k_ in dst: dst[k_].U.copy_(src[k_].U); dst[k_].K.copy_(src[k_].K); dst[k_].Gs.copy_(src[k_].Gs); dst[k_].Gt.copy_(src[k_].Gt)
def _cd_gen_slot(model,cfg,st0,m,L,dom,M,temp,g_,forced=None,flen=None,ban=None,hint=None):
    """as _cd_gen (same bytes, same log-probabilities up to rounding), with the memories in fixed-size slot buffers written in place"""
    dev=next(model.parameters()).device; W_=int(cfg.cd_inc_win); _INC['k']=next((i for i,b in enumerate(model.blocks) if not getattr(b,'loc',False)),1)
    k0_=next(iter(st0)); r0=st0[k0_][0].shape[1]; R=r0+L; mh_of={k_:v_[0].shape[0]//m for k_,v_ in st0.items()}
    gr_=bool(getattr(cfg,'cd_graph',False) and (dev.type=='cuda' or getattr(cfg,'cd_graph_fake',False)) and _CDG['off'] is None)
    RB=max(64,int(getattr(cfg,'cd_graph_rows',512)))
    if gr_:
        key=(dom,R,r0,L,W_,tuple(sorted((k_,mh_of[k_],tuple(v_[0].shape[2:]),tuple(v_[1].shape[2:])) for k_,v_ in st0.items())),_cdg_fp(model))
        need=-(-m//RB)*RB
        if _CG5['key']!=key or _CG5['cap']<need:
            _CG5['g'].clear(); _CG5['B']=None; _CG5['Y']=None; _CG5['LG']=None; _CG5['pool']=None   # (a pool whose graphs are all freed must not be reused)
            if dev.type=='cuda': torch.cuda.synchronize(); torch.cuda.empty_cache()
            cap=max(need,_CG5['cap'] if _CG5['key'] is not None and _CG5['key'][:6]==key[:6] else 0)
            ok_=True
            if dev.type=='cuda':
                byt=sum(cap*mh_of[k_]*R*(v_[0].shape[2]+v_[1].shape[2]+1)*4 for k_,v_ in st0.items())
                fr_=(torch.cuda.mem_get_info()[0]+torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
                ok_=fr_-byt>float(cfg.cd_graph_free_gb)*2**30
                if not ok_:
                    _CDG['n_skip']=_CDG.get('n_skip',0)+1
                    if _CDG['n_skip']==1: print(f"  [cd graph] slot buffers need {byt/2**30:.1f} GB, {fr_/2**30:.1f} GB free: eager slot generation for this call",flush=True)
            if ok_:
                _CG5['B']=_slot_alloc(st0,m,cap,R,dev); _CG5['Y']=torch.full((cap,L+1),M,dtype=torch.long,device=dev); _CG5['LG']=torch.zeros((cap,256),device=dev)
                _CG5['key']=key; _CG5['cap']=cap
                if dev.type=='cuda' and _CG5.get('pool') is None: _CG5['pool']=torch.cuda.graph_pool_handle()
            else: gr_=False; _CG5['key']=None; _CG5['cap']=0
    if gr_: B_=_CG5['B']; Ybuf=_CG5['Y']; LG=_CG5['LG']; cap=_CG5['cap']
    else: cap=m; B_=_slot_alloc(st0,m,m,R,dev); Ybuf=torch.full((m,L+1),M,dtype=torch.long,device=dev); LG=None
    _slot_fill(B_,st0,m); Ybuf[:m,0]=M
    nch=-(-m//RB); lps=[]
    for t in range(L):
        y0=max(0,t+1-W_) if W_>0 else 0
        if gr_ and _CDG['off'] is None:
            for j in range(nch):
                a=j*RB; b=a+RB; Yv=Ybuf[a:b,y0:t+1]; sl=_slot_view(B_,a,b,cap,r0,r0+t,mh_of)
                def fn(Yv=Yv,sl=sl,a=a,b=b): LG[a:b].copy_(_slot_fwd(model,dom,Yv.contiguous(),sl,RB))
                gk=(j,t)
                try:
                    if gk not in _CG5['g']:
                        if _CDG['n_cap']>=int(getattr(cfg,'cd_graph_max_rec',800)): raise RuntimeError(f"{_CDG['n_cap']} graphs recorded so far: re-recording too often")
                        t0_=time.time(); i_=r0+t
                        sv_={k_:(v_.U[:,i_].clone(),v_.K[:,i_].clone(),v_.Gs[:,i_].clone(),v_.Gt.clone()) for k_,v_ in sl.items()}
                        def rst():
                            for k_,v_ in sl.items(): v_.U[:,i_].copy_(sv_[k_][0]); v_.K[:,i_].copy_(sv_[k_][1]); v_.Gs[:,i_].copy_(sv_[k_][2]); v_.Gt.copy_(sv_[k_][3])
                        if dev.type!='cuda': _CG5['g'][gk]=fn
                        else:
                            if _CDG['n_cap']==0:                                       # a forward that synchronises cannot be recorded: find out first
                                prev_=torch.cuda.get_sync_debug_mode(); torch.cuda.set_sync_debug_mode(2)
                                try: fn()
                                finally: torch.cuda.set_sync_debug_mode(prev_); rst()
                            s_=torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(s_):
                                for _ in range(2): fn(); rst()                         # warm-up runs write slot i: undone
                            torch.cuda.current_stream().wait_stream(s_)
                            g5=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g5,pool=_CG5['pool']): fn()           # (recording executes nothing)
                            _CG5['g'][gk]=g5; torch.cuda.synchronize(); torch.cuda.empty_cache()
                        _CDG['n_cap']+=1; _CDG['t_cap']+=time.time()-t0_; del sv_
                    _CG5['chk_calls']+=1; ce_=int(cfg.cd_graph_check) if _CDG['n_chk']<40 else int(getattr(cfg,'cd_graph_check2',250))
                    chk=(_CG5['chk_calls']%max(1,ce_)==0)
                    if chk: slc=_slot_clone(sl); Yc=Yv.contiguous().clone()
                    g5=_CG5['g'][gk]
                    (g5() if dev.type!='cuda' else g5.replay()); _CDG['n_rep']+=1; nv_=max(0,min(b,m)-a); _CDG['rows']+=nv_; _CDG['pad_rows']+=RB-nv_
                    if chk:                                                        # SELF-CHECK: the eager forward on copies of the same inputs
                        lge=_slot_fwd(model,dom,Yc,slc,RB); lgg=LG[a:a+nv_]; lge_=lge[:nv_]
                        d_=float((lge_-lgg).abs().max()) if nv_ else 0.0; sc_=float(lge_.abs().max())+1e-6 if nv_ else 1.0
                        ag_=float((lge_.argmax(-1)==lgg.argmax(-1)).float().mean()) if nv_ else 1.0
                        ds_=max(float((slc[k_].Gt-sl[k_].Gt).abs().max()) for k_ in sl)
                        _CDG['n_chk']+=1; _CDG['n_bit']+=int(torch.equal(lge_,lgg)); _CDG['maxd']=max(_CDG['maxd'],d_/sc_); _CDG['minag']=min(_CDG['minag'],ag_)
                        if d_/sc_>float(cfg.cd_graph_tol) or ag_<0.95 or ds_>1e-4:
                            _slot_copy(sl,slc); LG[a:b].copy_(lge)
                            raise RuntimeError(f'self-check failed at byte {t}: logit difference {d_/sc_:.2e}, same top byte {ag_:.4f}, memory difference {ds_:.1e}')
                        del slc,Yc
                except Exception as ex_:
                    _fail('conduit:532')
                    _CDG['off']=repr(ex_)[:300]; _CG5['g'].clear(); _CG5['key']=None; _CG5['pool']=None
                    if dev.type=='cuda': torch.cuda.synchronize(); torch.cuda.empty_cache()
                    print(f"  [cd graph] OFF from here on (eager slot generation, unchanged results): {_CDG['off']}",flush=True)
                    if 'self-check failed' not in _CDG['off']:                    # (a failed self-check already restored the correct result)
                        for jj in range(j,nch):
                            aa=jj*RB; bb=min(aa+RB,m)
                            if bb>aa: LG[aa:bb].copy_(_slot_fwd(model,dom,Ybuf[aa:bb,y0:t+1].contiguous(),_slot_view(B_,aa,bb,cap,r0,r0+t,mh_of),bb-aa))
                        break
            lg=LG[:m].clone()
        else:
            lg=_slot_fwd(model,dom,Ybuf[:m,y0:t+1].contiguous(),_slot_view(B_,0,m,cap,r0,r0+t,mh_of),m)
        lps.append(F.log_softmax(lg,-1))
        if ban is not None: lg=lg.masked_fill((flen==t)[:,None]&(torch.arange(lg.shape[-1],device=dev)[None]==ban[:,None]),float('-inf'))
        nx_=lg.argmax(-1,keepdim=True) if temp<=0 else torch.multinomial(F.softmax(lg/float(temp),-1),1,generator=g_)
        if forced is not None: nx_=torch.where((t<flen)[:,None],forced[:,t:t+1],nx_)
        if hint is not None: nx_=torch.where(hint[0][:,t:t+1],hint[1][:,t:t+1],nx_)   # cg12b: hinted bytes forced
        cf_=lps[-1].gather(1,nx_).squeeze(1); _CDCONF['c']=cf_ if t==0 else _CDCONF['c']+cf_
        Ybuf[:m,t+1]=nx_[:,0]
    return Ybuf[:m,1:L+1].clone(), torch.stack(lps,1)
_SLOTCHK={'n':0}
def _cd_gen(model,cfg,st0,m,L,dom,M,temp,g_,forced=None,flen=None,ban=None,hint=None):
    if getattr(cfg,'cd_slot',False) and bool(cfg.cd_inc) and _slot_ok(st0,m):
        if int(getattr(cfg,'cd_slot_check',0))>_SLOTCHK['n']:                  # TEST: old path vs slot path on the same inputs (same random state)
            _SLOTCHK['n']+=1; gs_=g_.get_state(); c0_=_CDCONF.get('c')
            Yo,lo=_cd_gen_old(model,cfg,st0,m,L,dom,M,temp,g_,forced,flen,ban,hint); g_.set_state(gs_)
            Yn,ln=_cd_gen_slot(model,cfg,st0,m,L,dom,M,temp,g_,forced,flen,ban,hint)
            same_=float((Yo==Yn).float().mean()); fin_=torch.isfinite(lo)&torch.isfinite(ln)
            print(f"  [slot check {_SLOTCHK['n']}] {m} rows x {L} bytes: same bytes {same_:.4f} | largest log-probability difference {float((lo-ln).abs()[fin_].max()):.2e}",flush=True)
            return Yn,ln
        return _cd_gen_slot(model,cfg,st0,m,L,dom,M,temp,g_,forced,flen,ban,hint)
    return _cd_gen_old(model,cfg,st0,m,L,dom,M,temp,g_,forced,flen,ban,hint)
def _cd_gen_old(model,cfg,st0,m,L,dom,M,temp,g_,forced=None,flen=None,ban=None,hint=None):
    """m segments generated from their traces st0 (temp 0: greedy). Returns the bytes (m, L) and the teacher's log-probabilities at every step
    (m, L, 256): lp[:,j] predicts byte j from [mark] + bytes[:j] with the trace (the same as the parallel teacher pass over the result)."""
    dev=next(model.parameters()).device; Y=torch.full((m,1),M,dtype=torch.long,device=dev); lps=[]; cur=st0
    inc_=bool(cfg.cd_inc); W_=int(cfg.cd_inc_win)
    if inc_: _INC['k']=next((i for i,b in enumerate(model.blocks) if not getattr(b,'loc',False)),1)
    gr_=bool(inc_ and getattr(cfg,'cd_graph',False) and (dev.type=='cuda' or getattr(cfg,'cd_graph_fake',False)))   # CUDA GRAPHS (cd_graph)
    pl_=_cdg_plans(model,cfg,m,dom,W_,st0,dev) if (gr_ and _CDG['off'] is None) else None
    for t in range(L):
        o_=_cdg_gen_step(model,cfg,t,Y[:,-W_:] if W_>0 else Y,cur,m,dom,pl_) if (pl_ is not None and _CDG['off'] is None) else None
        if o_ is not None: lg,cur=o_
        else:
          _CARRY.update(on=True,nr=m,st=cur,new={},rows=None)
          if inc_:
            _INC['on']=True
            try: lg=model(Y[:,-W_:] if W_>0 else Y,dom)[:,-1].float()
            finally: _INC['on']=False
            cur={k_:v_ for k_,v_ in _CARRY['new'].items() if not isinstance(k_,tuple)}    # memory after this byte (factored)
          else: lg=model(Y,dom)[:,-1].float()
        lps.append(F.log_softmax(lg,-1))
        if ban is not None: lg=lg.masked_fill((flen==t)[:,None]&(torch.arange(lg.shape[-1],device=dev)[None]==ban[:,None]),float('-inf'))   # v9e repair
        nx_=lg.argmax(-1,keepdim=True) if temp<=0 else torch.multinomial(F.softmax(lg/float(temp),-1),1,generator=g_)
        if forced is not None: nx_=torch.where((t<flen)[:,None],forced[:,t:t+1],nx_)                # v9e: the verified prefix kept
        if hint is not None: nx_=torch.where(hint[0][:,t:t+1],hint[1][:,t:t+1],nx_)                 # cg12b: hinted bytes forced
        cf_=lps[-1].gather(1,nx_).squeeze(1); _CDCONF['c']=cf_ if t==0 else _CDCONF['c']+cf_   # confidence: sum log p(chosen byte)
        Y=torch.cat([Y,nx_],1)
    return Y[:,1:], torch.stack(lps,1)
@torch.no_grad()
def _store_score(tq,lpy,vm,C,L):
    """per window: the mean over its pieces of the POSITIVE part of each piece's reducible signal (log q(y) - log p(y) at the verified bytes)"""
    T1=tq.shape[1]; pg=((torch.arange(T1,device=tq.device)+1)//max(1,L)).clamp(max=C-1)
    r=(tq-lpy).float()*vm; sm=torch.zeros(tq.shape[0],C,device=tq.device).index_add_(1,pg,r); ct=torch.zeros_like(sm).index_add_(1,pg,vm.float())
    return (sm/ct.clamp_min(1)).clamp_min(0).mean(1)
def _nrem_loss(model,cfg,dom):
    """NREM: re-replay the long-term store's windows (any domain), highest remaining signal first; re-teach, re-measure, retire."""
    SB_=model.__dict__.get('_sbuf'); z0_=next(model.parameters()).sum()*0.0
    if not SB_ or not SB_['e']: return z0_
    ST_=SB_['st']; L=int(cfg.cd_seg); M=int(cfg.cd_mark); dev=model.embed.weight.device
    sel_=sorted(SB_['e'],key=lambda e_:-e_['v'])[:int(cfg.cd_store_k)]; drop_=set(); KS_=0.0; NS_=0
    for d_ in sorted({e_['dom'] for e_ in sel_}):
        sd_=[e_ for e_ in sel_ if e_['dom']==d_]; k_l=len(sd_); C=sd_[0]['x'].numel()//L
        with torch.no_grad():
            trs_=_cd_cat([e_['tr'] for e_ in sd_]) if k_l>1 else sd_[0]['tr']
            xb_=torch.stack([e_['x'] for e_ in sd_]); pcs_=xb_.reshape(k_l*C,L)
            Yl_=torch.cat([torch.full((k_l*C,1),M,dtype=torch.long,device=dev),pcs_[:,:-1]],1)
            svl_=_carry_save(); wasl_=model.training; model.eval(); fsl_=model.feat_scale.clone(); _CARRY['fac']=bool(cfg.cd_fac)
            try:
                with _cd_weights(model):
                    _CARRY.update(on=True,nr=k_l*C,st=trs_,new={},rows=None); tgl_=F.log_softmax(model(Yl_,d_).float(),-1)
            finally: _CARRY.pop('fac',None); _carry_load(svl_); model.feat_scale.copy_(fsl_); model.train(wasl_)
            tgl_=tgl_.reshape(k_l,C*L,-1)[:,1:]; ybl_=xb_[:,1:]; wl_=torch.stack([e_['w'] for e_ in sd_]).float(); vml_=(wl_>0).float()
            tqyl_=tgl_.gather(-1,ybl_[...,None]).squeeze(-1)
            if float(cfg.cd_evid_w)>0:
                dgp_=(tgl_.argmax(-1)!=ybl_).float()*vml_; ptl_=tgl_.exp(); lam2_=(float(cfg.cd_evid_w)*dgp_).unsqueeze(-1)
                tgl_=torch.log(((1-lam2_)*ptl_+lam2_*F.one_hot(ybl_,ptl_.shape[-1]).to(ptl_.dtype)).clamp_min(1e-12))
        sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
        try: lpl_=F.log_softmax(model(xb_[:,:-1],d_).float(),-1)
        finally: _carry_load(sv4_); _uh_load(model,suh4_); _NGPD['on']=False
        kll_=(tgl_.exp()*(tgl_-lpl_)).sum(-1); KS_=KS_+(kll_*wl_).sum(); NS_+=wl_.numel()
        with torch.no_grad(): rn_=_store_score(tqyl_,lpl_.gather(-1,ybl_[...,None]).squeeze(-1),vml_,C,L).tolist()
        for e_,r_ in zip(sd_,rn_):
            e_['n']+=1; ST_['nrem']=ST_.get('nrem',0)+1
            if r_<=max(1e-4,float(cfg.cd_store_retire)*e_['v0']): drop_.add(id(e_)); ST_['ret']=ST_.get('ret',0)+1; ST_['ret_n']=ST_.get('ret_n',0)+e_['n']; ST_['nrem_ret']=ST_.get('nrem_ret',0)+1; continue
            e_['v']=r_
    SB_['e']=[e_ for e_ in SB_['e'] if id(e_) not in drop_]
    return KS_/max(1,NS_) if NS_>0 else z0_
def lora_merge(model,cfg,olds,cur):
    """LOW-RANK DREAM MERGE: per old domain, the trunk update that restores its dreams (stored targets) -> rank-r -> merged -> no-harm check -> applied."""
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); ol=[d for d in olds if d in BK and d in TV and d!=cur]
    if not ol: return
    t0=time.time(); dev=model.embed.weight.device; kb=int(cfg.bank_targets); T=min(int(cfg.seq_len),BK[ol[0]].shape[1]); V_=int(cfg.lm_val)
    PR=[(n_,p_) for n_,p_ in model.named_parameters() if p_.ndim==2 and n_.startswith('blocks.') and '.pu_' not in n_ and '.film_' not in n_]
    rs=np.random.RandomState(int(cfg.seed)+211+int(model.__dict__.get('_visit',0)))
    def tg_(d,idx):
        X=torch.stack([BK[d][i][:T] for i in idx]).long().to(dev); Vv=torch.stack([TV[d][0][i][:T] for i in idx]).to(dev).float(); Ii=torch.stack([TV[d][1][i][:T] for i in idx]).to(dev).long()
        rest=(1-Vv.exp().sum(-1,keepdim=True)).clamp(min=1e-8)/(256-kb); G=rest.log().expand(X.shape[0],X.shape[1],256).clone(); G.scatter_(-1,Ii,Vv); return X,G
    def kl_(d,X,G): lp=F.log_softmax(model(X,d).float(),-1); return (G.exp()*(G-lp)).sum(-1).mean()
    SBk_=(None)
    STW=[e_ for e_ in (SBk_['e'] if SBk_ else []) if e_['dom'] in ol][:32]
    def store_ce():                                                        # real-text check: the student's loss on the stored VERIFIED bytes of old domains
        if not STW: return None
        tot=0.0; cnt=0.0
        for d_ in sorted({e_['dom'] for e_ in STW}):
            xs=torch.stack([e_['x'] for e_ in STW if e_['dom']==d_]); ws=torch.stack([e_['w'] for e_ in STW if e_['dom']==d_]).float()
            lp=F.log_softmax(model(xs[:,:-1],d_).float(),-1); ce=-lp.gather(-1,xs[:,1:,None]).squeeze(-1)
            tot+=float((ce*(ws>0).float()).sum()); cnt+=float((ws>0).float().sum())
        return tot/max(1.0,cnt)
    sv_=_carry_save(); suh_=_uh_save(model); was_=model.training; fs_=model.feat_scale.clone(); _CARRY.update(on=False,st={},new={}); model.eval()
    LR_=model.__dict__.get('_last_rep')
    try:
        VAL={}; base={}
        with torch.no_grad():
            for d in ol: n_=BK[d].shape[0]; VAL[d]=tg_(d,list(range(min(V_,n_//4)))); base[d]=float(kl_(d,*VAL[d]))
            cur_p=F.log_softmax(model(LR_[0][:8,:-1],cur).float(),-1) if (LR_ is not None and LR_[1]==cur) else None
            sce0_=store_ce()
        P0={n_:p_.detach().clone() for n_,p_ in PR}; DL={}; fit={}; lowr={}
        for d in ol:
            op_=torch.optim.Adam([p_ for _,p_ in PR],lr=float(cfg.lm_lr)); n_=BK[d].shape[0]; lo_=min(V_,n_//4)
            for _ in range(int(cfg.lm_steps)):
                X,G=tg_(d,rs.randint(lo_,n_,size=int(cfg.lm_rows)).tolist()); op_.zero_grad(set_to_none=True); kl_(d,X,G).backward(); op_.step()
            with torch.no_grad():
                fit[d]=float(kl_(d,*VAL[d])); D={}
                for n_,p_ in PR:
                    dd=p_.detach()-P0[n_]; p_.data.copy_(P0[n_])
                    if int(cfg.lm_rank)>0 and min(dd.shape)>int(cfg.lm_rank):
                        U,S,Vh=torch.linalg.svd(dd.float(),full_matrices=False); r=int(cfg.lm_rank); dd=(U[:,:r]*S[:r])@Vh[:r]
                    D[n_]=dd.to(p_.dtype)
                for n_,p_ in PR: p_.data.add_(D[n_])
                lowr[d]=float(kl_(d,*VAL[d]))
                for n_,p_ in PR: p_.data.copy_(P0[n_])
            DL[d]=D
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            MG={}; conf=[]
            for n_,_ in PR:
                S_=torch.stack([DL[d][n_].float() for d in ol])
                if cfg.lm_merge=='ties':
                    k_=max(1,int(0.2*S_[0].numel())); th_=S_.abs().flatten(1).kthvalue(S_[0].numel()-k_+1,dim=1).values.view(-1,*([1]*(S_.dim()-1)))
                    St_=torch.where(S_.abs()>=th_,S_,torch.zeros_like(S_)); sg_=torch.sign(St_.sum(0)); ag_=(torch.sign(St_)==sg_)&(St_!=0)
                    MG[n_]=(St_*ag_).sum(0)/ag_.sum(0).clamp_min(1); nz_=(St_!=0).sum(0); conf.append(float(((nz_>ag_.sum(0))&(nz_>0)).float().mean()))
                elif cfg.lm_merge=='sum': MG[n_]=S_.sum(0)
                else: MG[n_]=S_.mean(0)
            res=[]; acc=None
            for sc in cfg.lm_scales:
                for n_,p_ in PR: p_.data.copy_(P0[n_]+float(sc)*MG[n_].to(p_.dtype))
                ks={d:float(kl_(d,*VAL[d])) for d in ol}
                cd_=float((cur_p.exp()*(cur_p-F.log_softmax(model(LR_[0][:8,:-1],cur).float(),-1))).sum(-1).mean()) if cur_p is not None else 0.0
                ok_=all(ks[d]<=base[d]*(1+float(cfg.lm_eps)) for d in ol) and np.mean(list(ks.values()))<np.mean(list(base.values())) and cd_<=float(cfg.lm_cur_eps)
                if sce0_ is not None:
                    sce_=store_ce(); ok_=ok_ and sce_<=sce0_; cd_=(cd_,sce_-sce0_)
                res.append((sc,ks,cd_,ok_))
                if ok_ and acc is None: acc=sc; break
            if acc is None:
                for n_,p_ in PR: p_.data.copy_(P0[n_])
    finally: _carry_load(sv_); _uh_load(model,suh_); model.feat_scale.copy_(fs_); model.train(was_)
    mb_=np.mean(list(base.values())); mf_=np.mean(list(fit.values())); ml_=np.mean(list(lowr.values())); last_=res[-1]
    cap_=(mb_-ml_)/max(1e-9,mb_-mf_)
    print(f"  [dream merge] {len(ol)} old domains, rank {cfg.lm_rank if int(cfg.lm_rank)>0 else 'full'}, {cfg.lm_merge}: held-out dream KL {mb_:.4f} -> own full fit {mf_:.4f}, own rank-r {ml_:.4f}"
          f" (rank-r keeps {cap_:.0%} of the restoring effect) | sign conflicts {np.mean(conf) if conf else 0:.2f} | "
          +(f"ACCEPTED scale {acc:g}: merged KL {np.mean(list(last_[1].values())):.4f}, visited-domain drift {last_[2] if not isinstance(last_[2],tuple) else last_[2][0]:.4f}"+(f", stored real windows {last_[2][1]:+.4f} nats" if isinstance(last_[2],tuple) else "") if acc is not None else
            "REJECTED at every scale ("+"; ".join(f"x{r_[0]:g}: KL {np.mean(list(r_[1].values())):.4f}, worst x{max(r_[1][d]/max(1e-9,base[d]) for d in ol):.2f}, drift {r_[2] if not isinstance(r_[2],tuple) else r_[2][0]:.4f}"+(f", stored real {r_[2][1]:+.4f}" if isinstance(r_[2],tuple) else "") for r_ in res)+")")
          +f" | {time.time()-t0:.0f}s")
