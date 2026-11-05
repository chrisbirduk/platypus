"""Weight protection: replay stiffness, projections, Kalman/Fisher factors, protected Muon, varpro head refits, probes."""
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref as _weakref
from platypus.common import (
    LOG2,
)
from platypus.gating import (
    _f32,
)


# ---------------- REPLAY-DRIVEN STIFFNESS (split weight-gradient multiply) ----------------
_STF={'on':False,'B':None,'ns':None,'dream_call':False}
_lin_fwd0=nn.Linear.forward
def stiff_mods(model): return [m for m in model.blocks.modules() if isinstance(m,nn.Linear) and type(m) is nn.Linear]
@torch.no_grad()
def stiff_post(sv):
    for p,p0,f in sv: p.copy_(p0+(p-p0)*f.to(p.dtype))

# ---------------- BACKPROP REDUCTION: closed-form protection, selective steps, prioritised bank rows, freezing, closed-form heads ----------------
_PJ={'on':False,'ns':None,'B':None,'dream_call':False}
def _pj_hook(mod,inp):
    if not _PJ['on'] or _PJ['ns'] is None: return
    x=inp[0]; B=_PJ['B']
    if x.dim()>=3 and x.shape[0]==B: xd=x[_PJ['ns']:]
    elif x.dim()==2 and B and x.shape[0]%B==0: xd=x[(x.shape[0]//B)*_PJ['ns']:]
    else: return
    xd=xd.detach().reshape(-1,x.shape[-1]).float()
    if xd.shape[0]==0: return
    C=xd.t().matmul(xd)/xd.shape[0]; d_=mod.__dict__; C0=d_.get('_pjC'); e=_PJ['eps']
    d_['_pjC']=C if C0 is None else C0.mul_(1-e).add_(C,alpha=e)
def proj_setup(model,cfg):
    _PJ['on']=bool(cfg.proj_lambda>0); _PJ['eps']=float(cfg.proj_eps)
    if not _PJ['on']: return
    for m in stiff_mods(model):
        if not m.__dict__.get('_pj_h'): m.register_forward_pre_hook(_pj_hook); m.__dict__['_pj_h']=True
    print(f"  [projection] closed-form protection on {len(stiff_mods(model))} trunk Linear layers, lambda {cfg.proj_lambda}"+(f" (+ dream KL every {max(1,cfg.dream_every)} steps)"))
@torch.no_grad()
def proj_post(sv,model,step):
    sh=[]
    for W,W0,P in sv:
        dW=W-W0; dP=dW.matmul(P.to(dW.dtype)); W.copy_(W0+dP)
        if step%90==0: sh.append(float(dP.norm()/dW.norm().clamp(min=1e-12)))
    if sh: print(f"      [projection] update norm kept after projection: mean {np.mean(sh):.3f} (min {np.min(sh):.3f}) over {len(sh)} layers")
def _ns5(G,steps=5):
    """Newton-Schulz orthogonalisation (Muon's quintic iteration); zero rows / columns are preserved."""
    a,b,c=3.4445,-4.7750,2.0315; X=G.to(torch.bfloat16) if G.is_cuda else G.float(); tr_=X.shape[0]>X.shape[1]   # bf16 on GPU (as in Muon)
    if tr_: X=X.t()
    X=X/(X.norm()+1e-7)
    for _ in range(steps): A=X@X.t(); X=a*X+(b*A+c*(A@A))@X
    X=X.float(); return X.t() if tr_ else X
def _ns5_batch(Zs,steps=5):
    """_ns5 for a list of SAME-SHAPE matrices in one batched call (per-matrix normalisation, identical iteration)."""
    a,b,c=3.4445,-4.7750,2.0315; X=torch.stack(Zs); X=X.to(torch.bfloat16) if X.is_cuda else X.float(); tr_=X.shape[1]>X.shape[2]
    if tr_: X=X.transpose(1,2)
    X=X/(X.norm(dim=(1,2),keepdim=True)+1e-7)
    for _ in range(steps): A=X@X.transpose(1,2); X=a*X+(b*A+c*(A@A))@X
    X=X.float(); X=X.transpose(1,2) if tr_ else X
    return list(X.unbind(0))
_MU={'gram':False}
def _ns5_gram_batch(Zs,steps=5,restart=3):
    """GRAM NEWTON-SCHULZ (Dao lab 2026): the same quintic iteration run on the small n x n Gram matrix R = X X^T (n = short side):
    Z = b R + c R^2, Q <- Q Z + a Q, R <- Z R Z (+ a terms in fp32), polar(X) = Q X; ONE RESTART before iteration `restart` (X <- Q X,
    R <- X X^T, Q <- I) keeps rounding's spurious negative eigenvalues bounded. fp16 GEMMs (values stay O(1)), fp32 accumulation and
    a-terms. Rectangular products: 4 instead of 10."""
    a,b,c=3.4445,-4.7750,2.0315; X=torch.stack(Zs).float(); tr_=X.shape[1]>X.shape[2]
    if tr_: X=X.transpose(1,2)
    dt=torch.float16 if X.is_cuda else torch.float32
    mm=lambda p,q: torch.matmul(p.to(dt),q.to(dt)).float()               # small n x n products: fp16 GEMM, fp32 result
    X=(X/(X.norm(dim=(1,2),keepdim=True)+1e-7)).to(dt)                     # the big matrix is cast ONCE and stays in fp16
    R=torch.matmul(X,X.transpose(1,2)).float(); Q=None; I=torch.eye(X.shape[1],device=X.device)
    for t in range(1,steps+1):
        if t==restart and Q is not None: X=torch.matmul(Q.to(dt),X); R=torch.matmul(X,X.transpose(1,2)).float(); Q=None
        Z=b*R+c*mm(R,R)
        Q=(Z+a*I) if Q is None else mm(Q,Z)+a*Q
        RZ=mm(R,Z)+a*R; R=mm(Z,RZ)+a*RZ
    X=torch.matmul(Q.to(dt),X).float(); X=X.transpose(1,2) if tr_ else X
    return list(X.unbind(0))
_KF={'on':False,'acc':None}
def _kf_hook(mod,inp,out):
    if not _KF['on'] or not out.requires_grad: return
    A_=_KF['acc']; k_=id(mod); nr_=_KF.get('nreal'); md_=_KF['maxdim']
    shc_=_KF.get('share')                                                 # SHARED INPUT: one covariance per input tensor (kf_share_a)
    def _cov(rows_,tag_):
        if shc_ is not None:
            e0_=shc_.get((id(inp[0]),tag_))
            if e0_ is not None and e0_[0]() is inp[0]: return e0_[1]
        a0_=rows_.detach().reshape(-1,inp[0].shape[-1]).float(); c0_=(a0_.t()@a0_) if a0_.shape[1]<=md_ else (a0_*a0_).sum(0)
        if shc_ is not None: shc_[(id(inp[0]),tag_)]=(_weakref.ref(inp[0]),c0_)
        return c0_
    if _KF.get('dream') and nr_ and inp[0].shape[0]>nr_:                   # LIVE OLD-DOMAIN KEYS: input factor of the dream rows (current network)
        ad_=inp[0][nr_:].detach().reshape(-1,inp[0].shape[-1]).float()
        sd_=_cov(inp[0][nr_:],'d')
        D_=_KF['dacc'].setdefault(k_,{'A':None,'n':0}); D_['A']=sd_ if D_['A'] is None else D_['A']+sd_; D_['n']+=ad_.shape[0]
    if _KF.get('real_off'): return
    a_=(inp[0][:nr_] if nr_ else inp[0]).detach().reshape(-1,inp[0].shape[-1]).float()
    e_=A_.setdefault(k_,{'A':None,'nA':0,'G':None,'nG':0,'sA':None})
    if not _KF.get('no_a'):                                               # v9.8 (g3): stored input factors unused at kf_dream 1 -> not collected
        s_=_cov(inp[0][:nr_] if nr_ else inp[0],'r')
        e_['A']=s_ if e_['A'] is None else e_['A']+s_
    e_['nA']+=a_.shape[0]
    if a_.shape[1]>md_ and _KF.get('sketch',0)>0:                         # LOW-RANK SKETCH of a wide input factor: Y += a' (a Omega)
        r_=min(int(_KF['sketch']),a_.shape[1]); OM_=_KF.setdefault('om',{}); ky_=(a_.shape[1],r_,str(a_.device)); Om_=OM_.get(ky_)
        if Om_ is None:
            g_=torch.Generator(device='cpu'); g_.manual_seed(7919+a_.shape[1])
            Om_=OM_[ky_]=torch.randn(a_.shape[1],r_,generator=g_).to(torch.bfloat16).float().to(a_.device)   # bf16-exact: same Omega in both precisions
        ns_=int(_KF.get('sk_tok',4096)); as_=a_[::-(-a_.shape[0]//ns_)] if a_.shape[0]>ns_ else a_
        if as_.is_cuda: ab_=as_.to(torch.bfloat16); y_=(ab_.t()@(ab_@Om_.to(torch.bfloat16))).float()
        else: y_=as_.t()@(as_@Om_)
        e_['Y']=y_ if e_.get('Y') is None else e_['Y']+y_; e_['nY']=e_.get('nY',0)+as_.shape[0]; e_['Om']=Om_
    if getattr(mod,'_kf_last',False): _KF.setdefault('lasta',{})[k_]=a_
    cs_=a_.sum(0); e_['sA']=cs_ if e_.get('sA') is None else e_['sA']+cs_
    def gh_(g):
        with torch.autocast(device_type=g.device.type, enabled=False):                 # fp32 statistics (never bf16)
            g_=(g[:nr_] if nr_ else g).detach().reshape(-1,g.shape[-1]).float(); t_=(g_.t()@g_) if g_.shape[1]<=md_ else (g_*g_).sum(0)
            e_['G']=t_ if e_['G'] is None else e_['G']+t_; e_['nG']+=g_.shape[0]
    out.register_hook(gh_)
def _kf_mods(model):
    return [m for b in model.blocks for m in b.modules() if isinstance(m,nn.Linear) and m.weight.requires_grad]
def kf_register(model):
    for m_ in _kf_mods(model):
        if not getattr(m_,'_kf_h',False): m_.register_forward_hook(_f32(_kf_hook)); m_._kf_h=True
def kf_main_on(model,cfg,nreal):
    kf_register(model); _KF['acc']=model.__dict__.setdefault('_kf_cur',{}); _KF['maxdim']=int(cfg.kf_maxdim); _KF['nreal']=int(nreal); _KF['on']=True
    _KF['no_a']=bool(float(getattr(cfg,'kf_dream',0.0))>=1.0 and getattr(cfg,'kf_drop_a',True))
    _KF['share']={} if cfg.kf_share_a else None
    _KF['sketch']=0; _KF['sk_tok']=int(cfg.kf_sketch_tok)
def kf_collect(model,cfg,x,d):
    """Real batch, labels sampled from the model: Kronecker factors of the true Fisher of every block weight (accumulated over the visit)."""
    M_=_kf_mods(model)
    for m_ in M_:
        if not getattr(m_,'_kf_h',False): m_.register_forward_hook(_f32(_kf_hook)); m_._kf_h=True
    cur_=model.__dict__.setdefault('_kf_cur',{}); _KF['acc']=cur_; _KF['maxdim']=int(cfg.kf_maxdim); _KF['nreal']=None; _KF['sketch']=0
    was_=model.training; model.eval(); fs_=model.feat_scale.clone(); _KF['on']=True
    try:
        z_=model(x,d).float().reshape(-1,256)
        with torch.no_grad(): ys_=torch.multinomial(F.softmax(z_,-1),1).squeeze(-1)
        torch.autograd.grad(F.cross_entropy(z_,ys_,reduction='sum'),[m_.weight for m_ in M_],allow_unused=True)
    finally:
        _KF['on']=False; model.feat_scale.copy_(fs_); model.train(was_)
def sim_diag(model,cfg,seen,tag):
    """Data-free cross-domain similarity from the dream banks (pass ends): output affinity, representation CKA, Fisher overlap."""
    BK=model.__dict__.get('_bank',{}); ds=[d for d in seen if d in BK]
    if len(ds)<2: return
    dev=model.embed.weight.device; Pp=min(int(cfg.dream_stream_prompt),8); n_=int(cfg.sim_rows); OUT={}; ST={}
    sv_=model.__dict__.pop('_kf_cur',None)
    for d in ds:
        X_=BK[d][:n_].long().to(dev); x_,y_=X_[:,:-1],X_[:,1:]
        ch_=max(1,min(x_.shape[0],int(cfg.batch_size)//2))            # chunks of half a training batch (GPU memory: the Fisher pass keeps a graph)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        with torch.no_grad():
            was_=model.training; model.eval(); fs_=model.feat_scale.clone()
            for e in ds:
                tot_=0.0
                for c0 in range(0,x_.shape[0],ch_):
                    z_=model(x_[c0:c0+ch_],e).float()[:,Pp:]; tot_+=float(F.cross_entropy(z_.reshape(-1,256),y_[c0:c0+ch_,Pp:].reshape(-1),reduction='sum'))
                OUT[(d,e)]=tot_/(x_.shape[0]*(x_.shape[1]-Pp))/LOG2
            model.feat_scale.copy_(fs_); model.train(was_)
        model.__dict__['_kf_cur']={}
        for c0 in range(0,x_.shape[0],ch_): kf_collect(model,cfg,x_[c0:c0+ch_],d)
        C_=model.__dict__.pop('_kf_cur',{}); S_={}
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        for k_,e_ in C_.items():
            if e_['G'] is None or e_['nG']==0: continue
            m_=e_['sA']/e_['nA']; A_=e_['A']/e_['nA']; A_=(A_-torch.outer(m_,m_)) if A_.dim()==2 else (A_-m_*m_)
            S_[k_]=(A_,e_['G']/e_['nG'])
        ST[d]=S_
    if sv_ is not None: model.__dict__['_kf_cur']=sv_
    REP={}; FIS={}
    for i,d in enumerate(ds):
        for e in ds[i+1:]:
            ks_=[k for k in ST[d] if k in ST[e]]
            ca_=[_cka(ST[d][k][0],ST[e][k][0]) for k in ks_]; cg_=[_cka(ST[d][k][1],ST[e][k][1]) for k in ks_]
            REP[(d,e)]=float(np.mean(ca_)) if ca_ else float('nan'); FIS[(d,e)]=float(np.mean([a*b for a,b in zip(ca_,cg_)])) if ca_ else float('nan')
    import json as _js
    print("  [simjson] "+_js.dumps({'tag':tag,'doms':ds,'out':{f"{d},{e}":v for (d,e),v in OUT.items()},
                                     'rep':{f"{d},{e}":v for (d,e),v in REP.items()},'fis':{f"{d},{e}":v for (d,e),v in FIS.items()}}))
_FILM={'mask':None}
_DP2={'on':False}
@torch.no_grad()
def aff_weights(model,cfg,x,y,olds,tau):
    """Old heads' bits on a real batch -> weights n_old softmax(-bits / tau) (mean 1)."""
    was_=model.training; model.eval(); fs_=model.feat_scale.clone()
    b_=torch.tensor([float(F.cross_entropy(model(x,d).float().reshape(-1,256),y.reshape(-1)))/LOG2 for d in olds])
    model.feat_scale.copy_(fs_); model.train(was_)
    w_=torch.softmax(-(b_-b_.min())/tau,0)*len(olds); return {d:float(w_[i]) for i,d in enumerate(olds)},{d:float(b_[i]) for i,d in enumerate(olds)}
def film_mod(b,hn,ns_=None):
    C=_FILM.get('codes')
    if C is None or getattr(b,'film_g',None) is None: return hn
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if tag is None: return hn
    if isinstance(tag,(list,tuple)):
        RI_=_XDG.get('rows_ix')
        if RI_ is not None and RI_.shape[0]==hn.shape[0]: ix=RI_                    # SAMPLER: row index built once per sampling call
        else:
            cur=_XDG['cur'] if _XDG['cur'] is not None else 0
            ix=torch.tensor([d if d is not None else cur for d in tag],device=hn.device)
        if ix.shape[0]!=hn.shape[0]: return hn
        c_=C[ix]; g_=(c_@b.film_g.t())[:,None,:]; o_=(c_@b.film_b.t())[:,None,:]
    else:
        c_=C[tag]; g_=c_@b.film_g.t(); o_=c_@b.film_b.t()
    return hn*(1.0+g_.to(hn.dtype))+o_.to(hn.dtype)
def kf_store(model,d):
    cur_=model.__dict__.pop('_kf_cur',None)
    if not cur_: return
    D_=model.__dict__.setdefault('_kf_dom',{}).setdefault(d,{})
    for k_,e_ in cur_.items():
        if e_['G'] is None or e_['nG']==0 or not bool((e_['G']!=0).any()): continue   # FIX (review): skip all-zero G (closed-loop visits)
        A_=(e_['A']/e_['nA']) if e_['A'] is not None else None; G_=e_['G']/e_['nG']; ok9_=True
        if k_ in D_:                                                    # revisit: average with the stored factors
            G_=0.5*(D_[k_][1]+G_); A_=(0.5*(D_[k_][0]+A_)) if (A_ is not None and D_[k_][0] is not None) else A_
        D_[k_]=(A_,G_)
    nv_=model.__dict__.setdefault('_kf_nv',{})
    if any(e_['G'] is not None and e_['nG']>0 and bool((e_['G']!=0).any()) for e_ in cur_.values()): nv_[d]=nv_.get(d,0)+1   # FIX (review): only visits with evidence
    else: return
    tr_=lambda X: float(X.diagonal().sum() if X.dim()==2 else X.sum())
    if cur_:                                                               # FLATNESS proxy: sum over weights of tr(A) tr(G) on this visit's real rows
        fl_=sum(tr_(e_['A']/e_['nA'])*tr_(e_['G']/e_['nG']) for e_ in cur_.values() if e_['G'] is not None and e_['nG']>0 and e_['A'] is not None and e_['nA']>0)
        if fl_: model.__dict__.setdefault('_flat',{})[(d,nv_[d])]=fl_; print(f"    [flatness] {_lab(model.cfg,d)}: Fisher-trace proxy sum tr(A) tr(G) {fl_:.4g} (visit {nv_[d]})")
def _proj_apply(PJ,u):
    """u P for a projector: full matrix, coordinate mask (vector) or ('lr', U): u - (u U) U'."""
    if isinstance(PJ,tuple):
        U_=PJ[1]
        if U_.dtype==torch.bfloat16: return u-((u.to(torch.bfloat16)@U_)@U_.t()).float()     # GPU: bf16 matmuls (as Newton-Schulz)
        return u-(u@U_)@U_.t()
    return (u@PJ) if PJ.dim()==2 else u*PJ[None,:]
_KFR={'r':0}
def _isqrt(X,lam):
    """(I + X / (lam mean eig X))^-1/2 for a PSD factor X (vector = diagonal). Eigenvalues of X clamped at 0, so every entry of the
    result is in (0, 1]: the whitening can only shrink a step, never enlarge it."""
    X=X.double()
    if X.dim()==1: X=X.clamp(min=0); return (1.0+X/(lam*float(X.mean())+1e-30)).rsqrt().float()
    w_,V_=torch.linalg.eigh(0.5*(X+X.t())); w_=w_.clamp(min=0)
    r_=int(_KFR['r'])
    if 0<r_<w_.shape[0]: w_=w_.clone(); w_[:-r_]=w_[:-r_].mean()             # LOW-RANK + ISOTROPIC: top r kept (ascending order), rest = its mean
    return ((V_*(1.0+w_/(lam*float(w_.mean())+1e-30)).rsqrt())@V_.t()).float()
def _isqrt_ns_batch(Xs,lams):
    """Batched (I + X / (lam mean eig X))^-1/2 for same-shape PSD factors by the COUPLED NEWTON-SCHULZ iteration (Higham): B scaled by
    1.1 x a power-iteration estimate of its top eigenvalue (all eigenvalues of B are >= 1, so the iteration count follows from that bound);
    Y -> (B/s)^1/2, Z -> (B/s)^-1/2. fp32 products with TF32 off (the factors' condition number reaches ~1e3)."""
    X=torch.stack([x.double() for x in Xs]); n=X.shape[-1]; X=0.5*(X+X.transpose(1,2)); dev=X.device
    c=torch.tensor([float(l) for l in lams],dtype=torch.float64,device=dev)[:,None,None]*(torch.diagonal(X,dim1=1,dim2=2).sum(-1)/n).clamp(min=1e-30)[:,None,None]
    B=(torch.eye(n,dtype=torch.float64,device=dev)+X/c).float(); del X
    pv_=torch.backends.cuda.matmul.allow_tf32 if dev.type=='cuda' else None
    if pv_ is not None: torch.backends.cuda.matmul.allow_tf32=False
    try:
        g_=torch.Generator(device='cpu'); g_.manual_seed(97); v=torch.randn(B.shape[0],n,1,generator=g_).to(dev)
        for _ in range(12): v=B@v; v=v/v.norm(dim=1,keepdim=True)
        s=(1.1*(v.transpose(1,2)@B@v).squeeze(-1).squeeze(-1)).clamp(min=1.0)
        it=int(min(60,math.ceil(math.log(float(s.max()))/math.log(1.5))+6)); s=s[:,None,None]
        Y=B/s; Z=torch.eye(n,device=dev).expand_as(Y).clone(); I3=3.0*torch.eye(n,device=dev)
        for _ in range(it): T=0.5*(I3-Z@Y); Y=Y@T; Z=T@Z
    finally:
        if pv_ is not None: torch.backends.cuda.matmul.allow_tf32=pv_
    return list((Z/s.sqrt()).unbind(0))
def kf_live_warmup(model,cfg,olds):
    """v9.8 (g3 without stored input factors): the dream rows' CURRENT input factors from one forward pass at visit start"""
    if not getattr(model,'_dsq',None): return None
    acc_=model.__dict__.setdefault('_kf_dacc',{}); was_=model.training; fs_=model.feat_scale.clone()
    for _ in range(max(1,int(getattr(cfg,'kf_warm_batches',8)))):           # several dream batches: a stable estimate of the input statistics
        sb_=stream_batch(model,cfg,olds)
        if sb_ is None: break
        xd,_,dm_,_=sb_
        kf_main_on(model,cfg,1); _KF['dream']=True; _KF['real_off']=True; _KF['dacc']=acc_
        model.eval(); model.__dict__['_s0_dom']=list(dm_)+[None]*(xd.shape[0]-len(dm_))
        try:
            with torch.enable_grad(): model.features(xd)                   # (graph discarded: the hooks only read the inputs)
        finally:
            _KF['on']=False; _KF['dream']=False; _KF['real_off']=False; model.__dict__['_s0_dom']=None; model.feat_scale.copy_(fs_); model.train(was_)
    return {k_:e_['A']/max(1,e_['n']) for k_,e_ in acc_.items() if e_['A'] is not None} or None
def kf_build(model,cfg,d,live=None,quiet=False):
    """Visit start: preconditioners from the OTHER domains' stored factors, keyed by weight id. live = {module id: current input factor of the
    dream rows} (old domains' keys as the network sees them NOW): input side (1 - kf_dream) * sum(stored A) + kf_dream * n_old * A_live."""
    _KF['wbf16']=bool(getattr(cfg,'whiten_bf16',False)); t0_=time.time(); nsj_={}; _KFR['r']=int(cfg.kf_rank); ekc_=[]; isc_=[]
    D_=model.__dict__.get('_kf_dom',{}); P_={}
    for m_ in _kf_mods(model):
        W_=model.__dict__.get('_kf_wt') or {}                           # EARLY DRIFT GUARD: flagged domains' factors weighted up
        k_=id(m_); Gs_=[D_[j][k_][1]*float(W_.get(j,1.0)) for j in D_ if j!=d and k_ in D_[j]]
        if not Gs_: continue
        nA_=len(Gs_); As_=[(D_[j][k_][0])*float(W_.get(j,1.0)) for j in D_ if j!=d and k_ in D_[j] and D_[j][k_][0] is not None]
        if not As_:                                                     # v9.8 (g3): no stored input factors -> the live dream factor alone
            if not (live and k_ in live): continue
            As_=[live[k_].float()*float(nA_)]
        elif live and k_ in live and live[k_].shape==As_[0].shape:
            mx_=float(cfg.kf_dream); As_=[a_*(1.0-mx_) for a_ in As_]+[live[k_].to(As_[0].device,As_[0].dtype)*(mx_*len(As_))]
        lam_=cfg.kf_lam
        nv_=sum(v_ for j_,v_ in model.__dict__.get('_kf_nv',{}).items() if j_!=d); nv_=min(nv_,cfg.kf_kalman_max)
        lam_=cfg.kf_lam*cfg.kf_kalman/max(1,nv_)
        PO_=model.__dict__.get('_kf_P',{}).get(id(m_.weight)) if live else None    # live-key rebuild: gradient side unchanged -> reused
        XA_=sum(As_); ZA_=None
        if cfg.kf_share_a:                                          # identical input factor (shared input) -> the same whitening
            ZA_=next((h_[2] for h_ in isc_ if h_[0].shape==XA_.shape and h_[1]==lam_ and torch.equal(h_[0],XA_)),None)
        if ZA_ is None:
            ZA_=_isqrt(XA_,lam_).float()
            if cfg.kf_share_a: isc_.append((XA_,lam_,ZA_))
        P_[id(m_.weight)]=(PO_[0] if PO_ is not None else _isqrt(sum(Gs_),lam_).float(),ZA_)
    if nsj_:
        for jb_ in nsj_.values():
            for (k_,si_,_,_),Z_ in zip(jb_,_isqrt_ns_batch([j_[2] for j_ in jb_],[j_[3] for j_ in jb_])): P_[k_][si_]=Z_.float()
        P_={k_:tuple(v_) for k_,v_ in P_.items()}
    if P_ and next(iter(P_.values()))[-1].is_cuda: torch.cuda.synchronize()
    st_=model.__dict__.setdefault('_kfbt',[0.0,0]); st_[0]+=time.time()-t0_; st_[1]+=1
    model.__dict__['_kf_P']=P_
    model.__dict__.pop('_kf_K',None)
    model.__dict__.pop('_kf_proj',None)
    if P_ and not quiet:
        sh_=[(float(v_[3].mean()) if v_[0]=='ek' else float((v_[0].diagonal() if v_[0].dim()==2 else v_[0]).mean()*(v_[1].diagonal() if v_[1].dim()==2 else v_[1]).mean())) for v_ in P_.values()]
        print(f"    [protected muon] {len(P_)} weights whitened by {len([j for j in D_ if j!=d])} other domains' Fisher factors (mean diagonal shrink {np.mean(sh_):.3f})")
def _kf_apply(P,u):
    if P[0]=='ek':                                                      # EKFAC: rotate, scale per entry, rotate back
        _,UG_,UA_,W_=P; lp_=_KF.get('wbf16') and u.is_cuda; dt_=torch.bfloat16 if lp_ else u.dtype
        x=u.to(dt_)
        if UG_ is not None: x=UG_.t().to(dt_)@x
        if UA_ is not None: x=x@UA_.to(dt_)
        x=(x.float()*W_).to(dt_)
        if UG_ is not None: x=UG_.to(dt_)@x
        if UA_ is not None: x=x@UA_.t().to(dt_)
        return x.float()
    PG_,PA_=P
    if _KF.get('wbf16') and u.is_cuda:                                  # bf16 whitening (tensor cores), fp32 result
        x=u.to(torch.bfloat16); x=(PG_.to(torch.bfloat16)@x) if PG_.dim()==2 else PG_.to(torch.bfloat16)[:,None]*x
        return ((x@PA_.to(torch.bfloat16)) if PA_.dim()==2 else x*PA_.to(torch.bfloat16)[None,:]).float()
    u=(PG_@u) if PG_.dim()==2 else PG_[:,None]*u
    return (u@PA_) if PA_.dim()==2 else u*PA_[None,:]
def muon_prep(model,cfg):
    """Before opt.step(): Muon directions for the blocks' 2-D weights; their grads are cleared so Adam skips them."""
    out=[]; pend_=[]
    if cfg.muon_precise and '_mp_priv' not in model.__dict__:               # private / per-domain matrices keep Muon (PX_PS: Muon private)
        model.__dict__['_mp_priv']={id(p_) for b in model.blocks for n_,p_ in b.named_parameters() if n_.startswith(('pu_','film_'))}
        print(f"  [precise shared] shared block matrices step along the whitened momentum (Muon-sized); {len(model.__dict__['_mp_priv'])} private / FiLM matrices keep Muon", flush=True)
    for p_ in [p for b in model.blocks for p in b.parameters() if p.ndim==2 and p.requires_grad]:
        if p_.grad is None: continue
        g_=p_.grad.float(); st_=model.__dict__.setdefault('_muB',{})
        bf_=st_.get(id(p_))
        if bf_ is None or bf_.shape!=g_.shape: bf_=st_[id(p_)]=torch.zeros_like(g_)
        mu_=cfg.muon_mom
        ud_=None
        bf_.mul_(mu_).add_(g_); u_=g_+mu_*bf_
        if ud_ is not None: u_=u_+ud_
        P_=model.__dict__.get('_kf_P',{}).get(id(p_))
        z_=_kf_apply(P_,u_) if P_ is not None else u_
        PJ_=None
        if PJ_ is not None:                                             # DISJOINT SUBSPACES: Muon inside the free input subspace only
            o_=_proj_apply(PJ_,_ns5(_proj_apply(PJ_,u_)))
            out.append((p_,o_*(0.2*math.sqrt(max(p_.shape))))); p_.grad=None; continue
        K_=None
        if K_ is not None:                                              # KALMAN MUON: left whitening kept, input side = Kalman gain in the polar
            PG_=P_[0]; zl_=(PG_@u_) if PG_.dim()==2 else PG_[:,None]*u_
            R_,lam_=K_; s_=float(zl_.pow(2).sum())/zl_.shape[1]
            o_=_ns5(torch.cat([zl_,R_*math.sqrt(max(s_,1e-30)/lam_)],0))[:zl_.shape[0]]
            o_=(PG_@o_) if PG_.dim()==2 else PG_[:,None]*o_
            o_=o_*(zl_.abs().sum(1,keepdim=True)>0)*(zl_.abs().sum(0,keepdim=True)>0)        # zero rows / columns stay zero (private units)
            out.append((p_,o_*(0.2*math.sqrt(max(p_.shape))))); p_.grad=None; continue
        if cfg.muon_precise and id(p_) not in model.__dict__.get('_mp_priv',set()):   # cg11 PRECISE SHARED: whitened momentum, Muon-sized
            o_=z_*(math.sqrt(min(z_.shape))/z_.norm().clamp_min(1e-12))        #   (tensor norm: no host sync)
            if cfg.muon_precise_cap>0:                                         # cg11b CAP: largest singular value <= cap x Muon's (power iteration,
                PV_=model.__dict__.setdefault('_mpV',{}); v_=PV_.get(id(p_))   #   warm-started per matrix; no host sync). cg11: Frobenius-only sizing
                if v_ is None or v_.shape[0]!=o_.shape[1]: v_=torch.randn(o_.shape[1],device=o_.device,dtype=o_.dtype)   #   let the top directions
                for _ in range(int(cfg.muon_precise_pit)): v_=o_.t()@(o_@v_); v_=v_/v_.norm().clamp_min(1e-30)          #   grow with width (GPU:
                PV_[id(p_)]=v_; sg_=(o_@v_).norm()                             #   learning far worse than Muon from the first domain)
                o_=o_*torch.clamp(float(cfg.muon_precise_cap)/sg_.clamp_min(1e-12),max=1.0)
            if P_ is not None: o_=_kf_apply(P_,o_)
            out.append((p_,o_*(0.2*math.sqrt(max(p_.shape))))); p_.grad=None; continue
        pend_.append((p_,z_,P_)); p_.grad=None; continue
        if P_ is not None: o_=_kf_apply(P_,o_)                                       # Fisher-whitened Muon step
        out.append((p_,o_*(0.2*math.sqrt(max(p_.shape)))))
        p_.grad=None
    if pend_:                                                                         # one Newton-Schulz call per matrix shape
        grp_={}
        for e_ in pend_: grp_.setdefault(tuple(e_[1].shape),[]).append(e_)
        for G_ in grp_.values():
            ns_=_ns5_gram_batch if (cfg.muon_gram and min(G_[0][1].shape)*2<=max(G_[0][1].shape)) else _ns5_batch   # Gram NS for rectangular weights (>= 2:1)
            for (p_,z_,P_),o_ in zip(G_,ns_([e_[1] for e_ in G_],steps=int(cfg.muon_ns_steps))):   # cg12a: muon_ns_steps (5 = Muon default)
                if P_ is not None: o_=_kf_apply(P_,o_)
                out.append((p_,o_*(0.2*math.sqrt(max(p_.shape)))))
    return out
def _cnt_ctx(X):
    """Context hashes ending at each position: order 1 (exact), 2 (8192), 3 (16384)."""
    b=F.pad(X,(1,0))[:,:-1]; c=F.pad(X,(2,0))[:,:-2]; M=4294967296
    return [X,(((b*256+X)*2654435761)%M)>>19,((((c*256+b)*256+X)*2654435761)%M)>>18]
@torch.no_grad()
def _probe_lq(model,cfg,X,d):
    was=model.training; model.eval(); fs=model.feat_scale.clone(); out=[]
    for c0 in range(0,X.shape[0],16):
        Xc=X[c0:c0+16]; out.append(F.log_softmax(model.heads[d](priv(model,Xc,model.features(Xc),d)).float()/cfg.shunt_T,-1))
    model.feat_scale.copy_(fs); model.train(was); return torch.cat(out)
def _kl_topk(lq,V,I):
    """THE stored-target KL, per position: KL(top-k log-probs V at bytes I + one rest bin || student log-probs lq). Every user of the
    stored targets (dream loss, probes, sensors, noise scales, hybrid rows) calls this one function."""
    pk_=V.exp(); r_=(1-pk_.sum(-1)).clamp(min=1e-8)
    mk_=torch.ones_like(lq,dtype=torch.bool).scatter_(-1,I,False); lr_=lq.masked_fill(~mk_,float('-inf')).logsumexp(-1)   # student's log-mass outside the top-k
    return (pk_*(V-lq.gather(-1,I))).sum(-1)+r_*(r_.log()-lr_)
@torch.no_grad()
def _probe_kl(lq,V,I,Pp,k_):
    """per-row KL(stored top-k + rest bin || current) over positions >= Pp"""
    return _kl_topk(lq,V,I)[:,Pp:].mean(-1)
@torch.no_grad()
def _probe_real(model,cfg,X,d):
    """count-table log-likelihood per row (realism of a dream row for domain d)"""
    C=model.__dict__.get('_cnt',{}).get(d)
    if C is None: return None
    Xl=X.long(); hs=_cnt_ctx(Xl[:,:-1]); y=Xl[:,1:]; m_=5.0; lp=torch.zeros_like(y,dtype=torch.float32); done=torch.zeros_like(y,dtype=torch.bool)
    for o in (2,1,0):
        T_=C[o].view(-1,256)[hs[o]]; tot=T_.sum(-1); ok=(tot>=m_)&~done
        lp=torch.where(ok,(T_.gather(-1,y[...,None]).squeeze(-1)+0.5).log()-(tot+128).log(),lp); done|=ok
    return lp.mean(-1)
def probe_pre(model,cfg,olds,t,vals):
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); k_=int(cfg.bank_targets); dev=model.embed.weight.device; P={'olds':[],'t':t}
    xb_=[]; g_=vals[t]
    if hasattr(g_,'reset'): g_.reset()
    for _ in range(4): xb_.append(g_()[0])
    XB=torch.cat(xb_).long().to(dev)[:64]; P['XB']=XB
    for d in olds:
        if d not in BK or d not in TV: continue
        X=BK[d].long().to(dev); V,I=TV[d]; V=V.to(dev).float(); I=I.to(dev).long(); Pp=min(cfg.dream_stream_prompt,X.shape[1]//2)
        e={'d':d,'bpc':bpc(model,vals[d],d),'X':X,'V':V,'I':I,'Pp':Pp,'klA':_probe_kl(_probe_lq(model,cfg,X,d),V,I,Pp,k_)}
        lB=_probe_lq(model,cfg,XB,d); vB,iB=lB.topk(k_,-1); e['B']=(vB,iB)
        XC=dream_sample(model,d,64,X.shape[1],cfg.dream_temp).long(); lC=_probe_lq(model,cfg,XC,d); vC,iC=lC.topk(k_,-1); e['C']=(XC,vC,iC)
        e['real']=_probe_real(model,cfg,X,d)
        P['olds'].append(e)
    return P
def probe_post(model,cfg,P,vi,vals):
    import json as _js
    k_=int(cfg.bank_targets); rows=[]
    for e in P['olds']:
        d=e['d']; b1=bpc(model,vals[d],d); dR=(b1-e['bpc'])*math.log(2)          # real damage in nats / byte
        klA=_probe_kl(_probe_lq(model,cfg,e['X'],d),e['V'],e['I'],e['Pp'],k_); dA=klA-e['klA']
        vB,iB=e['B']; dB=float(_probe_kl(_probe_lq(model,cfg,P['XB'],d),vB,iB,0,k_).mean())
        XC,vC,iC=e['C']; dC=float(_probe_kl(_probe_lq(model,cfg,XC,d),vC,iC,e['Pp'],k_).mean())
        r_=e['real']; dD=dDm=float('nan')
        if r_ is not None:
            o_=torch.argsort(r_,descending=True); h_=len(o_)//2; dD=float(dA[o_[:h_]].mean()); dDm=float(dA[o_[h_:]].mean())
        rows.append({'vi':vi,'d':d,'real':dR,'A':float(dA.mean()),'B':dB,'C':dC,'D':dD,'Dm':dDm})
    for r in rows: print("  [probe] "+_js.dumps(r))
def muon_apply(out,lr):
    with torch.no_grad():
        for p_,u_ in out: p_.add_(u_.to(p_.dtype),alpha=-lr)
def _vp_inv(model,cfg,k_,C_):
    IC_=model.__dict__.setdefault('_vpI',{}); n_=model.__dict__.setdefault('_vpIn',{}); n_[k_]=n_.get(k_,0)+1
    if k_ not in IC_ or (n_[k_]-1)%max(1,int(cfg.varpro_inv_every))==0:
        IC_[k_]=torch.linalg.inv_ex(C_+torch.diagonal(C_).mean()*cfg.varpro_lam*torch.eye(C_.shape[0],device=C_.device))[0]
    return IC_[k_]
@torch.no_grad()
def varpro_old_refit(model,cfg,xd,hfd,spans,Pp,vi_,tgt_d):
    """Gauss-Newton refit of every OLD head on its own dream rows (positions after the prompt) towards their targets."""
    CG_=model.__dict__.setdefault('_vpC',{}); BA_=[]; BR_=[]; BH_=[]
    k_v=max(1,int(model.__dict__.get('_vpo_k',1))); vlr_=cfg.varpro_lr if k_v==1 else 1.0-(1.0-float(cfg.varpro_lr))**k_v   # k damped steps in one (head_dream_k)
    cur_=model.__dict__.get('_live_dom')                                     # the visited domain's head is trained normally this step
    for d,a,b in spans:
        if (cfg.self_dream_nreal or cfg.self_prot_w>0) and cur_ is not None and int(d)==int(cur_): continue
        hd_=model.heads[d]
        Hd_=(priv(model,xd[a:b],hfd[a:b].detach(),d))[:,Pp:].float()
        Hd_=Hd_.reshape(-1,Hd_.shape[-1]); n_=Hd_.shape[0]
        if vi_ is not None:                                                    # stored top-k + rest bin -> dense target
            V_,I_=vi_[0][a:b,Pp:].float(),vi_[1][a:b,Pp:]; pk_=V_.exp(); r_=(1-pk_.sum(-1,keepdim=True)).clamp(min=0)
            T_=(r_/(256-I_.shape[-1])).expand(*I_.shape[:-1],256).clone(); T_.scatter_(-1,I_,pk_); T_=T_.reshape(-1,256)
        elif tgt_d is not None: T_=tgt_d[a:b,Pp:].float().exp().reshape(-1,256)
        else: continue
        Ha_=torch.cat([Hd_,torch.ones(n_,1,device=Hd_.device)],1); Cb_=Ha_.t()@Ha_/n_; k_=('dream',d)
        CG_[k_]=Cb_ if k_ not in CG_ else cfg.varpro_beta*CG_[k_]+(1.0-cfg.varpro_beta)*Cb_
        model.__dict__['_vpo_n']=model.__dict__.get('_vpo_n',0)+1
        E_=F.softmax(hd_(Hd_.to(hd_.weight.dtype)).float()/cfg.shunt_T,-1)-T_; C_=CG_[k_]
        BA_.append(C_+torch.diagonal(C_).mean()*cfg.varpro_lam*torch.eye(C_.shape[0],device=C_.device)); BR_.append(Ha_.t()@E_/n_); BH_.append(hd_)
        continue
        Ai_=_vp_inv(model,cfg,k_,CG_[k_])
        for _ in range(max(1,cfg.varpro_k)):
            E_=F.softmax(hd_(Hd_.to(hd_.weight.dtype)).float()/cfg.shunt_T,-1)-T_
            dW_=-(Ai_@(Ha_.t()@E_)/n_).t()*vlr_
            hd_.weight.add_(dW_[:,:-1].to(hd_.weight.dtype)); hd_.bias.add_(dW_[:,-1].to(hd_.bias.dtype))
    if BA_:                                                                    # ONE batched solve for every old head of this step
        X_=torch.linalg.solve_ex(torch.stack(BA_),torch.stack(BR_))[0]          # _ex: no singularity check = no host sync
        for hd_,x_ in zip(BH_,X_):
            dW_=-x_.t()*vlr_; hd_.weight.add_(dW_[:,:-1].to(hd_.weight.dtype)); hd_.bias.add_(dW_[:,-1].to(hd_.bias.dtype))
@torch.no_grad()
def head_center_(model):
    for hd_ in model.heads:
        hd_.weight.sub_(hd_.weight.mean(0,keepdim=True))
        if hd_.bias is not None: hd_.bias.sub_(hd_.bias.mean())
def varpro_refit(model,cfg,opt,x,y,h,d):
    """VarPro (head first): refit the current head on this batch's detached features before the trunk gradient is taken."""
    hd_=model.heads[d]
    with torch.no_grad(): hv_=(priv(model,x,h.detach(),d)).reshape(-1,h.shape[-1]).float()
    yv_=y.reshape(-1); lr0_=opt.param_groups[0]['lr']
    Ha_=torch.cat([hv_,torch.ones(hv_.shape[0],1,device=hv_.device)],1); n_=Ha_.shape[0]
    CG_=model.__dict__.setdefault('_vpC',{}); Cb_=Ha_.t()@Ha_/n_
    CG_[d]=Cb_ if d not in CG_ else cfg.varpro_beta*CG_[d]+(1.0-cfg.varpro_beta)*Cb_
    Ai_=None
    with torch.no_grad():
        for _ in range(cfg.varpro_k):
            P_=F.softmax(hd_(hv_.to(hd_.weight.dtype)).float(),-1); P_[torch.arange(n_),yv_]-=1.0           # P - Y
            if Ai_ is None:                                                                                    # direct solve (LU)
                C_=CG_[d]; dW_=-torch.linalg.solve_ex(C_+torch.diagonal(C_).mean()*cfg.varpro_lam*torch.eye(C_.shape[0],device=C_.device),Ha_.t()@P_/n_)[0].t()*cfg.varpro_lr
            else: dW_=-(Ai_@(Ha_.t()@P_)/n_).t()*cfg.varpro_lr                                               # (256, D+1)
            hd_.weight.add_(dW_[:,:-1].to(hd_.weight.dtype)); hd_.bias.add_(dW_[:,-1].to(hd_.bias.dtype))
@torch.no_grad()
def smem_reencode(model,cfg,cur):
    """SLOT MEMORY RE-ENCODING: every other slot's memory rebuilt with the current weights from slot_mem_reenc of its bank dreams (first halves)."""
    SM=model.__dict__.get('_smem',{}); BK=model.__dict__.get('_bank',{}); R=int(cfg.slot_mem_reenc); Tq_=min(cfg.dream_len,cfg.seq_len)
    dev=model.embed.weight.device; n_=0
    for d in list(SM):
        if d==cur or d not in BK: continue
        X=BK[d][:R,:Tq_].long().to(dev); was=model.training; fs=model.feat_scale.clone(); s0_=model.__dict__.get('_s0_dom')
        sv_={k_:_CARRY[k_] for k_ in ('on','nr','st','new','rows')}
        try:
            model.eval(); _CARRY.update(on=True,nr=X.shape[0],st={},new={},rows=None); model.__dict__['_s0_dom']=d
            model.features(X); new_=dict(_CARRY['new'])
        finally:
            model.train(was); model.feat_scale.copy_(fs); model.__dict__['_s0_dom']=s0_; _CARRY.update(sv_)
        for k_,v_ in new_.items():
            if isinstance(k_,tuple): continue
            SM[d][k_]=v_.reshape(X.shape[0],v_.shape[0]//X.shape[0],*v_.shape[1:]).mean(0)
        n_+=1
    if n_: print(f"  [slot memory] re-encoded {n_} other slots' memories with the current weights ({R} dreams each)")
@torch.no_grad()
def guard_sense(model,cfg,olds):
    """EARLY DRIFT GUARD sensor: KL(stored targets || current) per old domain on its first guard_rows bank rows (held out: never trained)."""
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); dev=model.embed.weight.device; H=min(int(cfg.guard_rows),int(cfg.bank_hold)); out={}
    for d in olds:
        if d not in BK or d not in TV or H<1: continue
        Xs=BK[d][:H].long().to(dev); V,I=TV[d]; Vs=V[:H].to(dev).float(); Is=I[:H].to(dev).long()
        sv_=model.__dict__.get('_s0_dom'); model.__dict__['_s0_dom']=d
        try: lq=_probe_lq(model,cfg,Xs,d)
        finally: model.__dict__['_s0_dom']=sv_
        out[d]=float(_probe_kl(lq,Vs,Is,min(cfg.dream_stream_prompt,Xs.shape[1]//2),Vs.shape[-1]).mean())
    return out
def noise_diag(model,cfg,x,y,d,olds):
    """GRADIENT NOISE SCALES of the two parts of the step (McCandlish et al. 2018), on the trunk: two independent half batches of each part
    give per-row noise tr(Sigma) = b |g1-g2|^2 / 2 and signal |G|^2 = |g1+g2|^2/4 - tr(Sigma)/(2b); B_noise = tr(Sigma)/|G|^2 is the batch
    beyond which more rows of that part stop paying per step. NEYMAN allocation of a fixed row budget between the real loss and the
    w-weighted dream loss: n_dream / n_real = w sigma_dream / sigma_real * sqrt(c_real / c_dream). Accumulated per visit, printed at its end."""
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); ol=[j for j in olds if j in BK and j in TV]
    if not ol or x.shape[0]<4: return
    dev=model.embed.weight.device; P_=[p for b in model.blocks for p in b.parameters() if p.requires_grad]
    h=x.shape[0]//2; k_=int(cfg.bank_targets); was_=model.training; model.eval(); fs_=model.feat_scale.clone()
    sv_=model.__dict__.get('_s0_dom')
    def grads(loss_fn):
        with torch.enable_grad():
            l_=loss_fn(); g_=torch.autograd.grad(l_,P_,allow_unused=True)
        return [gg.detach().float() if gg is not None else torch.zeros_like(pp,dtype=torch.float32) for gg,pp in zip(g_,P_)]
    def real(i):
        model.__dict__['_s0_dom']=d
        z_=model(x[i*h:(i+1)*h],d).float(); return F.cross_entropy(z_.reshape(-1,256),y[i*h:(i+1)*h].reshape(-1))
    r_=np.random.RandomState(int(model.__dict__.get('_gstep',0))+977)
    dm=sorted(ol[i] for i in r_.randint(len(ol),size=2*h)); ri=[r_.randint(BK[j].shape[0]) for j in dm]
    X=torch.stack([BK[j][i] for j,i in zip(dm,ri)]).long().to(dev)
    V=torch.stack([TV[j][0][i] for j,i in zip(dm,ri)]).to(dev).float(); I=torch.stack([TV[j][1][i] for j,i in zip(dm,ri)]).to(dev).long()
    Pp=min(cfg.dream_stream_prompt,X.shape[1]//2)
    def dream(i):
        sl=slice(i*h,(i+1)*h); Xs,Vs,Is,ds=X[sl],V[sl],I[sl],dm[sl.start:sl.stop]
        model.__dict__['_s0_dom']=list(ds); hf=model.features(Xs); out=[]; a=0
        for j in sorted(set(ds)):
            b_=a+sum(1 for q in ds if q==j); out.append(F.log_softmax(model.heads[j](priv(model,Xs[a:b_],hf[a:b_],j)).float()/cfg.shunt_T,-1)); a=b_
        return _kl_topk(torch.cat(out),Vs,Is)[:,Pp:].mean()
    try:
        R1,R2=grads(lambda: real(0)),grads(lambda: real(1)); D1,D2=grads(lambda: dream(0)),grads(lambda: dream(1))
    finally:
        model.__dict__['_s0_dom']=sv_; model.feat_scale.copy_(fs_); model.train(was_)
    dot=lambda A,B: float(sum((a*b).sum() for a,b in zip(A,B)))
    def stats(A,B):
        dd=dot(A,A)+dot(B,B)-2*dot(A,B); ss=dot(A,A)+dot(B,B)+2*dot(A,B); tr=h*dd/2; G2=ss/4-tr/(2*h); return tr,G2
    trR,GR=stats(R1,R2); trD,GD=stats(D1,D2)
    SR_=[a+b for a,b in zip(R1,R2)]; SD_=[a+b for a,b in zip(D1,D2)]                 # raw cosine of the two mean gradients (bounded; noise
    cos=dot(SR_,SD_)/max(1e-30,(dot(SR_,SR_)*dot(SD_,SD_))**0.5)                       #   biases it towards 0 - the signal estimate can be < 0)
    A_=model.__dict__.setdefault('_ndiag',[]); A_.append((trR,GR,trD,GD,cos))
@torch.no_grad()
def noise_report(model,cfg,dom_id):
    A_=model.__dict__.pop('_ndiag',None)
    if not A_: return
    trR=np.mean([a[0] for a in A_]); GR=np.mean([a[1] for a in A_]); trD=np.mean([a[2] for a in A_]); GD=np.mean([a[3] for a in A_]); cs=np.mean([a[4] for a in A_])
    P_=model.__dict__.get('_sleepP',{}); w=float(cfg.dream_weight)*(float(cfg.sleep_fixed))
    ratio=w*math.sqrt(max(trD,0)/max(trR,1e-30))*math.sqrt(1.0/1.5)                     # dream row ~1.5x the cost of a real row
    print(f"  [noise scales] {len(A_)} probes: per-row gradient noise real {trR:.3e} / dream {trD:.3e} | signal |G|^2 real {GR:.3e} / dream {GD:.3e} | "
          f"B_noise (rows) real {trR/max(GR,1e-30):.0f} / dream {trD/max(GD,1e-30):.0f} | cos(G_real, G_dream) {cs:+.2f} | "
          f"Neyman dream rows for {cfg.batch_size} real rows at weight {w:.2f}: {ratio*cfg.batch_size:.0f}")

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_CARRY': 'platypus.layers',
    '_XDG': 'platypus.private',
    '_cka': 'platypus.diagnostics',
    '_lab': 'platypus.capacity',
    'bpc': 'platypus.inference.store',
    'dream_sample': 'platypus.capacity',
    'priv': 'platypus.private',
    'stream_batch': 'platypus.dream_bank',
}
