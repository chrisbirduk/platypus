"""Inferred heads (domain not given at test time) and BPC evaluation."""
import numpy as np
import time
import torch
import torch.nn.functional as F
from platypus.common import (
    LOG2, _nslots,
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
        object.__setattr__(model,'_bg',torch.zeros(_nslots(cfg),256*256,device=dev))
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

@torch.no_grad()
def mix_text_eval(model, vals, seen_d, slot_of, cfg, tag=''):
    """MIXED-TEXT READOUT (mix_eval N; readout only, training unchanged): can several slots be active at once? N windows each spliced
    from two seen domains in alternating segments of mix_seg bytes (a b a b ...), scored with every slot in use, mixed per position by:
    ROUTED one slot for the window (the first segment's domain, as a visit is routed); ORACLE the true slot at each position; POST-CUM the
    causal head-likelihood posterior over the whole window so far (heads.inferred_logp_multi 'heads'); POST-64 the same over the last 64
    bytes only, so it can switch; TOP2 of POST-64; UNIFORM an equal mixture. PURE: the same modes on unspliced windows of one domain."""
    import random as _r
    if len(seen_d) < 2: return None
    N_=int(getattr(cfg,'mix_eval',0)); sg0_=int(getattr(cfg,'mix_seg',256)); Ts_=int(cfg.seq_len); sg_=sg0_; W_=64
    rr_=_r.Random(int(cfg.seed)*7919+len(seen_d)); was_=model.training; model.eval()
    slots_=sorted(set(int(slot_of(q)) for q in seen_d)); K_=len(slots_); si_={s:i for i,s in enumerate(slots_)}
    acc_={'MIXED':{}, 'PURE':{}}; ws_={'MIXED':{}, 'PURE':{}}; n_=0
    def _modes(LP, Y_, lab, a_):
        D_,B_,T_=LP.shape[:3]
        l_=LP.gather(-1,Y_[None,...,None].expand(D_,-1,-1,1))[...,0]                          # (D,B,T): log-likelihood of each scored byte
        ll_=torch.cat([torch.zeros_like(l_[...,:1]),l_[...,:-1]],-1)                            # the weights at t use bytes before t only
        cum_=torch.cumsum(ll_,-1); win_=cum_-torch.cat([torch.zeros_like(cum_[...,:W_]),cum_[...,:-W_]],-1)
        pc_=F.log_softmax(cum_*float(getattr(cfg,'hs_temp',1.0)),0); pw_=F.log_softmax(win_,0)
        v2_,i2_=pw_.topk(min(2,D_),0); t2_=torch.full_like(pw_,-1e9).scatter(0,i2_,v2_); t2_=t2_-torch.logsumexp(t2_,0,keepdim=True)
        oh_=lambda idx: torch.where(F.one_hot(idx,D_).permute(2,0,1).bool(),0.0,-1e9)
        W={'ROUTED':oh_(torch.full((B_,T_),a_,dtype=torch.long,device=LP.device)),'ORACLE':oh_(lab),'POST-CUM':pc_,'POST-64':pw_,'TOP2':t2_,
           'UNIFORM':torch.full_like(pw_,-float(np.log(D_)))}
        out_={}
        for k,w in W.items():
            lp_=torch.logsumexp(w.unsqueeze(-1)+LP,0)
            out_[k]=(float(-lp_[:,-Ts_:].gather(-1,Y_[:,-Ts_:,None]).mean())/LOG2, float((w[:,:,-Ts_:].exp()>0.1).float().sum(0).mean()))
        return out_
    for i_ in range(max(1,N_//4)):
        a,b=rr_.sample(list(seen_d),2)
        xa,ya=vals[a](); xb,yb=vals[b]()
        B_=min(xa.shape[0],xb.shape[0],4); xa,ya,xb,yb=xa[:B_],ya[:B_],xb[:B_],yb[:B_]; T_=xa.shape[1]
        sg_=max(4,min(sg0_,min(T_,Ts_)//2)); W_=min(64,max(8,sg_//4)); Ts_=min(Ts_,T_)                # short local windows: at least one switch inside the scored part
        segb_=(((torch.arange(T_,device=xa.device)-(T_-Ts_)).clamp(min=0)//sg_)%2==1)                # segments counted from the start of the scored part
        ia_,ib_=si_[int(slot_of(a))],si_[int(slot_of(b))]
        labm_=torch.where(segb_,torch.tensor(ib_,device=xa.device),torch.tensor(ia_,device=xa.device)).expand(B_,-1)
        for kind,(X_,Y_,lab_) in (('MIXED',(torch.where(segb_,xb,xa),torch.where(segb_,yb,ya),labm_)),
                                  ('PURE',(xa,ya,torch.full((B_,T_),ia_,dtype=torch.long,device=xa.device)))):
            LP=torch.stack([F.log_softmax(model(X_,s).float(),-1) for s in slots_])
            for k,(bp,na) in _modes(LP,Y_,lab_,ia_).items():
                acc_[kind].setdefault(k,[]).append(bp); ws_[kind].setdefault(k,[]).append(na)
        n_+=B_
    if was_: model.train()
    for kind in ('MIXED','PURE'):
        r_={k:float(np.mean(v)) for k,v in acc_[kind].items()}
        print(f"  [mixed text{tag}] {kind}: {n_} windows of {Ts_} scored bytes"+(f", two seen domains in alternating {sg_}-byte segments, posterior window {W_} bytes" if kind=='MIXED' else ", one domain")
              +f", {K_} slots in use | BPC (mean slots with weight > 0.1): "+' | '.join(f"{k} {r_[k]:.3f} ({np.mean(ws_[kind][k]):.1f})" for k in r_)
              +(f" || gain over ROUTED: ORACLE {r_['ROUTED']-r_['ORACLE']:+.3f}, POST-64 {r_['ROUTED']-r_['POST-64']:+.3f}, UNIFORM {r_['ROUTED']-r_['UNIFORM']:+.3f}" if kind=='MIXED' else ''),flush=True)
    return acc_

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
