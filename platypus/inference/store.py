"""The fast store: name-indexed sparse logit corrector, and evaluation with it."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from platypus.layers import (
    Head,
)
from platypus.model import (
    Model,
)
from platypus.heads import (
    _bpc_w0,
)

def _ng_zero(model):
    """clear the SPARSE gradients of the hashed n-gram tables (kept in model.__dict__, so not model.parameters()): any extra backward
    (consolidation, sleep phases) otherwise ACCUMULATES them until the next n-gram optimiser step - a huge stale update, and on the GPU an
    int32 overflow in coalesce (the illegal-address crash after the first closed-loop consolidation)"""
    for k_ in ('_ngpi_opt','_ngp_opt','_ngo_opt','_fcp_opt'):
        O_=model.__dict__.get(k_)
        if O_ is not None: O_.zero_grad(set_to_none=True)
def _inf_sync(cfg,tag_):
    """DEBUG (inf_debug_sync): force the GPU to finish everything queued so far; an illegal memory access then surfaces HERE, naming the stage"""
    if not int(getattr(cfg,'inf_debug_sync',0)) or not torch.cuda.is_available(): return
    torch.cuda.synchronize(); n_=_INF.setdefault('dbgn',{}); n_[tag_]=n_.get(tag_,0)+1
    if n_[tag_]<=3 or n_[tag_]%50==0: print(f"  [inference debug] GPU state OK after {tag_} (#{n_[tag_]})",flush=True)
# ---- cg20h: CONSOLIDATION INTO A FAST STORE (inf_cons='mem') instead of the weights -----------------------------------------------
#   a sparse error-memory corrector keyed on the model's PRE-HEAD features z (centred, normalised, random expansion to inf_mem_dx units,
#   top inf_mem_act share kept): logits += gain * phi(z) M. Written in sleep by the delta rule with the residual (onehot - q) on the
#   conclusion bytes, and 'do-no-harm' anchor writes (p_model - q) on recent verified windows. Active only in no-grad evaluation
#   (bpc, probe recall): the weights and the closed loop never see it.
_INFM={'M':None,'on':False,'X':None}
_INFM_TAUS=(8.0,32.0,128.0)
def _infm_init(model,cfg):
    if _INFM['M'] is not None: return
    dev_=model.embed.weight.device; d_=int(model.embed.weight.shape[1]); key_=str(getattr(cfg,'inf_mem_key','z')); g_=torch.Generator().manual_seed(int(cfg.seed)+2222)
    Dk_={'name':d_,'nameb':d_,'namemix':d_,'z':d_,'slow':4*d_,'slowgate':4*d_,'latch':2*d_}[key_]
    DX_=int(getattr(cfg,'inf_mem_dx',0) or (4096 if key_ in ('name','nameb','namemix','z') else 16384)); DXB_=int(getattr(cfg,'inf_mem_dxb',65536)) if key_.startswith('name') else DX_
    _INFM['bpc_on']=str(getattr(cfg,'inf_mem_bpc','on'))!='off'
    _INFM['bank_first']=bool(int(getattr(cfg,'inf_bank_first',0)))
    _INFM['bank_span']=int(getattr(cfg,'inf_bank_span',0))
    _INFM['gap_dbg']=bool(int(getattr(cfg,'inf_gap_debug',0)))
    _INFM.update(M=torch.zeros(DXB_,256,device=dev_),WX=(torch.randn(DX_,Dk_,generator=g_)/math.sqrt(Dk_)).to(dev_),mu=None,key=key_,DXB=DXB_,names={},d=d_,
                 gates=torch.ones(len(_INFM_TAUS),device=dev_),lthr=None,model=model,dream=str(getattr(cfg,'inf_xfer','none'))=='dream',
                 ka=max(1,int(DX_*float(getattr(cfg,'inf_mem_act',0.02)))),g=float(getattr(cfg,'inf_mem_gain',1.0)),wz=float(getattr(cfg,'inf_mem_wz',0.5)),gthr=float(getattr(cfg,'inf_mem_gate',0.0)))
def _infm_ids(X_):
    """key 'name': per position, the hashed id of the last complete name before it if that name is in the store (else -1: no correction)"""
    Xc_=X_.detach().to('cpu',torch.uint8).numpy(); out_=np.full(Xc_.shape,-1,np.int64); nm_=_INFM['names']
    for r_ in range(Xc_.shape[0]):
        if not _UNITS.get('on'):
            for m_ in _INF_NAME.finditer(bytes(Xc_[r_])): out_[r_,m_.end():]=nm_.get(m_.group(0),-1)
            continue
        L_=_UNITS['lex']; dfu_=_UNITS.get('df') or {}; best_=None; sp_=int(_INFM.get('bank_span',0)); k_=0
        for m_ in _UEVT.finditer(bytes(Xc_[r_])):                             # rule-free: a stored unit opens its bank; another unit or a sentence end closes it
            g_=m_.group(0)
            if _UNITS.get('all'):                                              # word-level layer: the bank of the RAREST stored word in the sentence so far
                if len(g_)==1 and g_ in b'.!?\n': best_=None; out_[r_,m_.end():]=-1; continue
                if g_ in nm_ and (best_ is None or (not _INFM.get('bank_first') and dfu_.get(g_,0)<dfu_.get(best_,0))): best_=g_; k_=0; out_[r_,m_.end():]=nm_[g_]; continue
                if best_ is not None and sp_>0:                                #   BANK SPAN: the store was only ever taught clauses of at most sp_ words
                    k_+=1                                                      #   after the bank word, so it stays silent beyond them
                    if k_>sp_: best_=None; out_[r_,m_.end():]=-1
                continue                                                       #   (bank_first: the FIRST stored word keeps the bank - stable as frequencies drift)
            if g_ in nm_: out_[r_,m_.end():]=nm_[g_]
            elif len(g_)==1 and g_ in b'.!?\n' or (g_ in L_ and not _UNITS.get('all')): out_[r_,m_.end():]=-1
    return torch.from_numpy(out_).to(X_.device)
def _infm_nrm(v_): return v_/(v_.norm(dim=-1,keepdim=True)+1e-6)
def _infm_bytes(X_):
    """DRIFT-FREE key features: a FIXED random code of the last 8 bytes (order-aware, decaying); never trained, so it cannot drift with the trunk"""
    E_=_INFM.get('EB')
    if E_ is None or E_.device!=X_.device:
        g_=torch.Generator().manual_seed(4321); E_=(torch.randn(8,256,_INFM['d'],generator=g_)/math.sqrt(_INFM['d'])).to(X_.device); _INFM['EB']=E_
    B_,T_=X_.shape; Xl_=X_.long(); out_=torch.zeros(B_,T_,E_.shape[2],device=X_.device)
    for j_ in range(min(8,T_)): out_[:,j_:]+=(0.8**j_)*E_[j_][Xl_[:,:T_-j_]]
    return out_
def _infm_latch(z_,X_,lg_):
    """SURPRISE LATCH: a word whose first two bytes were surprising (above the running 85th percentile of ordinary text) is latched (mean
    feature over the word) and held until the next latched word; positions before any latch get id -1 (no correction)"""
    B_,T_,d_=z_.shape; out_=torch.zeros(B_,T_,d_,device=z_.device); id_=torch.full((B_,T_),-1,dtype=torch.long,device=z_.device)
    lp_=F.log_softmax(lg_.float(),-1); su_=torch.zeros(B_,T_,device=z_.device); su_[:,1:]=-lp_[:,:-1].gather(-1,X_[:,1:,None]).squeeze(-1)/math.log(2)
    Xc_=X_.detach().to('cpu',torch.uint8).numpy(); suc_=su_.detach().cpu().numpy(); thr_=_INFM['lthr'] if _INFM['lthr'] is not None else 1e9; ws_=[]
    isl_=lambda c: (65<=c<=90) or (97<=c<=122)
    for r_ in range(B_):
        st_=-1; cur_=None
        for t_ in range(T_):
            c_=int(Xc_[r_,t_]); 
            if isl_(c_):
                if st_<0: st_=t_
            elif st_>=0:
                w_=float(suc_[r_,st_]+(suc_[r_,st_+1] if st_+1<t_ else 0.0)); ws_.append(w_)
                if w_>thr_ and st_>0: cur_=z_[r_,st_:t_].mean(0)
                st_=-1
            if cur_ is not None: out_[r_,t_]=cur_; id_[r_,t_]=0
    return out_,id_,ws_
def _infm_keys(z_,X_,lg_=None):
    """per-position store keys K (centred, per-piece normalised) and ids (-1: inactive). z (B,T,d), X (B,T) bytes, lg (B,T,256) logits"""
    k_=_INFM['key']; zc_=z_.float()-_INFM['mu']
    if k_=='name': return zc_,_infm_ids(X_)
    if k_=='nameb': return _infm_bytes(X_),_infm_ids(X_)                                    # name bank + byte-context units: no trunk features at all
    if k_=='namemix': return _infm_nrm(_infm_bytes(X_))+_INFM['wz']*_infm_nrm(zc_),_infm_ids(X_)   # mostly stable, partly trunk (rewording)
    if k_=='z': return zc_,torch.zeros(X_.shape,dtype=torch.long,device=X_.device)
    if k_ in ('slow','slowgate'):
        B_,T_,d_=zc_.shape; st_=torch.zeros(B_,len(_INFM_TAUS),d_,device=zc_.device); al_=torch.tensor([1.0/t for t in _INFM_TAUS],device=zc_.device)[None,:,None]; S_=[]
        for t_ in range(T_): st_=(1-al_)*st_+al_*zc_[:,t_,None,:]; S_.append(st_)
        S_=torch.stack(S_,1)                                                        # (B,T,ntau,d)
        K_=torch.cat([_infm_nrm(zc_)]+[_INFM['gates'][j_]*_infm_nrm(S_[:,:,j_]) for j_ in range(len(_INFM_TAUS))],-1)
        return K_,torch.zeros(X_.shape,dtype=torch.long,device=X_.device)
    L_,id_,_=_infm_latch(zc_,X_,lg_)
    return torch.cat([_infm_nrm(zc_),_infm_nrm(L_)],-1),id_
def _infm_sparse(k_,id_):
    y_=_infm_nrm(k_.float())@_INFM['WX'].T
    v_,i_=y_.topk(_INFM['ka'],-1); v_=v_.clamp(min=0)+1e-6; v_=v_/v_.norm(dim=-1,keepdim=True)
    if _INFM['key'].startswith('name'): i_=(i_+((id_[:,None]%1000003)*2654435761)%_INFM['DXB'])%_INFM['DXB']      # conjunction: the name picks the bank, z the units (no int64 overflow)
    i_=i_.clamp_(0,_INFM['M'].shape[0]-1)
    return i_,v_
def _infm_corr(z_,id_):
    sh_=z_.shape; zf_=z_.reshape(-1,sh_[-1]); idf_=id_.reshape(-1); out_=torch.zeros(zf_.shape[0],256,device=z_.device)
    if idf_.numel()!=zf_.shape[0]: return (0.0*out_).view(*sh_[:-1],256)            # misaligned ids: no correction (never index out of range)
    sel_=(idf_>=0).nonzero().squeeze(-1); Mw_=_INFM.get('Mw') if _INFM.get('wm_on') else None; lt_=not _INFM.get('lt_off')
    for c_ in range(0,sel_.numel(),512):
        s_=sel_[c_:c_+512]; i_,v_=_infm_sparse(zf_[s_],idf_[s_])
        if lt_: out_[s_]=(v_[...,None]*_INFM['M'][i_]).sum(1)
        if Mw_ is not None: out_[s_]+=(v_[...,None]*Mw_[i_]).sum(1)                 # WORKING MEMORY (question-time scratch store)
    return (_INFM['g']*out_).view(*sh_[:-1],256)
def _infm_write(zf_,idf_,R_,beta_,gate_lr=0.0,tmask=None):
    if idf_.numel()!=zf_.shape[0] or R_.shape[0]!=zf_.shape[0]: return
    sel_=((idf_>=0)&(R_.abs().sum(-1)>0)).nonzero().squeeze(-1)
    for c_ in range(0,sel_.numel(),512):
        s_=sel_[c_:c_+512]; i_,v_=_infm_sparse(zf_[s_],idf_[s_])
        if gate_lr>0:                                                               # SLOWGATE: timescale gates follow the retrieval error
            dk_=(((_INFM['M'][i_]@R_[s_][:,:,None]).squeeze(-1)*v_)[...,None]*_INFM['WX'][i_]).sum(1); d_=_INFM['d']
            for j_ in range(len(_INFM_TAUS)):
                pc_=zf_[s_][:,d_*(j_+1):d_*(j_+2)]; _INFM['gates'][j_]+=gate_lr*float((dk_[:,d_*(j_+1):d_*(j_+2)]*pc_).sum(-1).mean())
            _INFM['gates'].clamp_(0.0,3.0)
        _INFM['M'].index_add_(0,i_.reshape(-1),(beta_*v_[...,None]*R_[s_][:,None,:]).reshape(-1,256))
        if _INFM.get('ds_on'):                                                  # cg26i: rows keyed by a write this sleep count as rehearsed (fast-store downscaling)
            if _INFM.get('touch') is None or _INFM['touch'].shape[0]!=_INFM['M'].shape[0]: _INFM['touch']=torch.zeros(_INFM['M'].shape[0],dtype=torch.bool,device=_INFM['M'].device)
            ti_=i_ if tmask is None else i_[tmask[s_]]                         #   (only positions of CONCLUSION rows count, when the caller says which: anchor rows
            if ti_.numel(): _INFM['touch'][ti_.reshape(-1)]=True                #   key nearly every row, so they would make every row look rehearsed)
_Model_fwd0=Model.forward
def _model_fwd_m(s,x,domain=0):
    if _INFM['M'] is not None: _INFM['X']=x
    return _Model_fwd0(s,x,domain)
Model.forward=_model_fwd_m
def _infm_head_corr(x,lg_=None):
    X_=_INFM['X']
    if X_ is None or x.dim()!=3 or X_.dim()!=2 or X_.shape[0]!=x.shape[0] or X_.shape[1]<x.shape[1] or X_.device!=x.device: return 0.0   # only full (rows, positions, features) reads aligned with the stashed bytes
    K_,id_=_infm_keys(x,X_[:,-x.shape[1]:],lg_); x=K_
    if _INF.get('corr') and not _INFM.get('dbg_pr'):
        _INFM['dbg_pr']=1; print(f"  [inference eval | fast store debug] stashed row 0: {bytes(X_[0].to('cpu',torch.uint8).tolist())!r} | x {tuple(x.shape)} X {tuple(X_.shape)} | ids row 0 {id_[0].tolist()[:40]} | first store names {list(_INFM['names'])[:5]}",flush=True)
    c_=_infm_corr(x,id_)
    gt_=float(_INFM.get('gthr',0.0))
    if gt_>0: c_=c_*(c_.amax(-1,keepdim=True)>=gt_).float()                       # CONFIDENCE GATE: a correction is applied only if it backs some byte by >= gthr logits
    d_=_INFM.setdefault('dbg',[0,0,0.0]); d_[0]+=int(id_.numel()); d_[1]+=int((id_>=0).sum()); d_[2]+=float(c_.abs().amax(-1).sum())
    return c_
_Head_fwd0=Head.forward
def _head_fwd_m(s,x):
    z_=_Head_fwd0(s,x)
    if _INFM['M'] is not None and _INFM['mu'] is not None and not torch.is_grad_enabled():
        act_=_INFM['force'] if 'force' in _INFM else (_INFM['on'] or (_INFM['dream'] and float(_INFM['model'].__dict__.get('_real_w',1.0))==0))
        if act_: z_=z_+_infm_head_corr(x,z_)
    return z_
Head.forward=_head_fwd_m
def bpc(model,get,domain,n=15,skip=0):
    """evaluation with the fast store active (cg20h); identical to the original when no store exists"""
    if _INFM['M'] is None: return _bpc_w0(model,get,domain,n,skip)
    sv_=_INFM['on']; _INFM['on']=bool(_INFM['bpc_on'])
    try: return _bpc_w0(model,get,domain,n,skip)
    finally: _INFM['on']=sv_

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_INF': 'platypus.inference.probes',
    '_INF_NAME': 'platypus.inference.units',
    '_UEVT': 'platypus.inference.units',
    '_UNITS': 'platypus.inference.units',
}
