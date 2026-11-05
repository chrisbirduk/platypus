"""Inferred heads (domain not given at test time) and BPC evaluation."""
import numpy as np
import time
import torch
import torch.nn.functional as F
from platypus.common import (
    LOG2,
)
from platypus.config import (
    PILE_DOMAINS,
)
from platypus.capacity import (
    apply_delta, domain_delta,
)

# ---------------- INFERRED HEADS (domain identity not given at test time) ----------------
@torch.no_grad()
def head_stats_update(model, x, dom_id, step, cfg):
    """Per-domain summary statistics gathered while training: byte-pair counts (whole domain) and the mean /
    variance of the final representation (second half of the domain, when it has settled)."""
    dev=x.device
    if not hasattr(model,'_bg'):
        object.__setattr__(model,'_bg',torch.zeros(len(PILE_DOMAINS),256*256,device=dev))
        object.__setattr__(model,'_fs',{})
    pair=(x[:,:-1]*256+x[:,1:]).reshape(-1)
    model._bg[dom_id]+=torch.bincount(pair,minlength=256*256).float()
    if step>=cfg.steps_per_task//2 and getattr(model,'_last_h',None) is not None:
        h=model._last_h.reshape(-1,model._last_h.shape[-1]).float()
        s1,s2,n=model._fs.get(dom_id,(0.0,0.0,0))
        model._fs[dom_id]=(s1+h.sum(0), s2+(h*h).sum(0), n+h.shape[0])

@torch.no_grad()
def inferred_logp(model, x, ncand, mode, cfg):
    return inferred_logp_multi(model, x, ncand, (mode,), cfg)[mode]

@torch.no_grad()
def inferred_logp_multi(model, x, ncand, modes, cfg):
    """Mixture of the trained heads weighted by a CAUSAL posterior over domains (token t uses bytes <= t), for several posterior
    modes from ONE forward pass. mode 'bytes': naive-Bayes on byte pairs; 'heads': the heads' own likelihood; 'both';
    'features': diagonal Gaussian on the final representation."""
    h=model.features(x); D=ncand
    if getattr(model,'_spec',None):                     # domain-specific shunts: each head reads its own configuration
        LPl=[]
        for d in range(D):
            dd=domain_delta(model,d) if d!=getattr(model,'_live_dom',None) else {}
            apply_delta(model,dd,+1); LPl.append(F.log_softmax(model.heads[d](model.features(x)),-1)); apply_delta(model,dd,-1)
        LP=torch.stack(LPl)
    else:
        LP=torch.stack([F.log_softmax(model(x,d),-1) for d in range(D)])
    def _bytes_ll():
        cnt=model._bg[:D]+cfg.hs_alpha; cnt=cnt.view(D,256,256)
        logp=torch.log(cnt/cnt.sum(-1,keepdim=True))                                  # log p_d(next | prev)
        prev,nxt=x[:,:-1],x[:,1:]
        l_=logp[:,prev,nxt]                                                            # (D,B,T-1)
        return torch.cat([torch.zeros_like(l_[...,:1]),l_],-1)
    def _heads_ll():                       # HEAD LIKELIHOOD: how well each head predicted the bytes seen so far (Bayesian
        nxt=x[:,1:]                        # model averaging over the heads; causal: position t uses bytes <= t)
        l_=LP[:,:,:-1].gather(-1,nxt[None,...,None].expand(D,-1,-1,1))[...,0]        # (D,B,T-1)
        return torch.cat([torch.zeros_like(l_[...,:1]),l_],-1)
    res={}
    for mode in modes:
        if mode=='bytes': ll=_bytes_ll()
        elif mode=='heads': ll=_heads_ll()
        elif mode=='both': ll=_bytes_ll()+_heads_ll()
        else:
            hs=[]
            for d in range(D):
                s1,s2,n=model._fs[d]; mu=s1/n; var=(s2/n-mu*mu).clamp(min=1e-4)
                hs.append((-0.5*((h-mu)**2/var+torch.log(var))).sum(-1))                    # (B,T)
            ll=torch.stack(hs)
        post=F.log_softmax(torch.cumsum(ll,-1)*cfg.hs_temp,0)                              # (D,B,T)
        res[mode]=(torch.logsumexp(post.unsqueeze(-1)+LP,0), post)
    return res

def bpc_inferred_modes(model, get, domain, ncand, modes, cfg, n=15):
    """bpc_inferred for several modes with one forward pass per batch (identical values)."""
    if hasattr(get,'reset'): get.reset()
    model.eval(); L={m:[] for m in modes}; A={m:[] for m in modes}
    with torch.no_grad():
        for _ in range(n):
            x,y=get()
            Ts_=int(cfg.seq_len)
            for m,(lp,post) in inferred_logp_multi(model,x,ncand,modes,cfg).items():
                L[m].append(float(-lp[:,-Ts_:].gather(-1,y[:,-Ts_:].unsqueeze(-1)).mean())); A[m].append(float((post[...,-Ts_:].argmax(0)==domain).float().mean()))
    model.train(); return {m:(float(np.mean(L[m]))/LOG2, float(np.mean(A[m]))) for m in modes}

def bpc_inferred(model, get, domain, ncand, mode, cfg, n=15):
    if hasattr(get,'reset'): get.reset()
    model.eval(); L=[]; A=[]
    with torch.no_grad():
        for _ in range(n):
            x,y=get(); lp,post=inferred_logp(model,x,ncand,mode,cfg)
            L.append(float(-lp.gather(-1,y.unsqueeze(-1)).mean())); A.append(float((post.argmax(0)==domain).float().mean()))
    model.train(); return float(np.mean(L))/LOG2, float(np.mean(A))

def _bpc_w0(model,get,domain,n=15,skip=0):
    if hasattr(get,'reset'): get.reset()
    for _ in range(int(skip)): get()                                    # FIX (review): skip=k evaluates on batches after the first k (brakes off the reported batches)
    dd=domain_delta(model,domain) if (getattr(model,'_spec',None) and domain!=getattr(model,'_live_dom',None)) else {}
    apply_delta(model,dd,+1)
    model.eval(); L=[]
    eb=int(getattr(getattr(model,'cfg',None),'eval_batch',0) or 0)
    Ts_=int(model.cfg.seq_len)                                        # eval_ctx: longer windows, scored on their last seq_len bytes
    with torch.no_grad():
        if eb>0:                                                     # same batches, stacked: one forward per eb sequences
            xs,ys=zip(*[get() for _ in range(n)]); X=torch.cat(xs); Y=torch.cat(ys); tot=0.0
            for i in range(0,X.shape[0],eb):
                tot+=float(F.cross_entropy(model(X[i:i+eb],domain)[:,-Ts_:].reshape(-1,256),Y[i:i+eb,-Ts_:].reshape(-1),reduction='sum'))
            L=[tot/Y[:,-Ts_:].numel()]
        else:
            for _ in range(n): x,y=get(); L.append(F.cross_entropy(model(x,domain)[:,-Ts_:].reshape(-1,256),y[:,-Ts_:].reshape(-1)).item())
    apply_delta(model,dd,-1)
    model.train(); return float(np.mean(L))/LOG2

def _tmark(model, cfg, step, name):
    """Per-phase wall-clock (GPU synchronised) on every timing_every-th step."""
    e_=max(2,int(cfg.timing_every))
    if not cfg.timing or step%e_ not in (3,(3+e_//2+(1 if (e_//2)%2==0 else 0))%e_): return   # regular steps of BOTH parities (dream steps can be every 2nd step)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    T=model.__dict__.setdefault('_tm',{'last':None,'acc':{},'n':0})
    for k_,v_ in (('sd',[0.0,0]),('sn',[0.0,0]),('t0',None),('d0',0)): T.setdefault(k_,v_)
    now=time.perf_counter()
    if name=='start': T['last']=now; T['n']+=1; T['t0']=now; T['d0']=model.__dict__.get('_dcount',0); return
    if T['last'] is not None: T['acc'][name]=T['acc'].get(name,0.0)+now-T['last']; T['last']=now
    if name=='rest' and T['t0'] is not None:                         # whole-step time, split by whether the step dreamed
        a_=T['sd'] if model.__dict__.get('_dcount',0)>T['d0'] else T['sn']; a_[0]+=now-T['t0']; a_[1]+=1
