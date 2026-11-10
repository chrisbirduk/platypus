"""Replay and stream sources, slot scoring and slot (domain) inference."""
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import (
    LOG2, _nslots,
)
from platypus.private import (
    priv,
)

@torch.no_grad()
@torch.no_grad()
def _slot_bpc(model,xs,ys,s):
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    v=float(np.mean([float(F.cross_entropy(model(x_,s).reshape(-1,256),y_.reshape(-1))) for x_,y_ in zip(xs,ys)]))/LOG2
    model.feat_scale.copy_(fs); model.train(was); return v
class _ReplayGet:
    """Wraps a visit's batch stream: the batches already drawn to identify the context are trained on first (no text is skipped, so
    the stream is the same as with labels), and the last n batches are kept (the slot's reference BPC is measured on them)."""
    def __init__(s,get,first=(),n=4): s._g=get; s._q=list(first); s.last=__import__('collections').deque(maxlen=max(1,int(n)))
    def __call__(s):
        b=s._q.pop(0) if s._q else s._g(); s.last.append(b); return b
    def __getattr__(s,k): return getattr(s._g,k)
@torch.no_grad()
def _slot_score(model,xs,ys,s):
    """(BPC, mean top-block output) of the batches read through slot s (its codes, private units and head)"""
    was=model.training; model.eval(); fs=model.feat_scale.clone(); sv_=model.__dict__.get('_s0_dom'); v=0.0; m=0.0
    try:
        for x_,y_ in zip(xs,ys):
            model.__dict__['_s0_dom']=int(s); h_=model.features(x_); model.__dict__['_s0_dom']=sv_
            v+=float(F.cross_entropy(model.heads[s](priv(model,x_,h_,int(s))).float().reshape(-1,256),y_.reshape(-1))); m=m+h_.float().mean((0,1))
    finally:
        model.__dict__['_s0_dom']=sv_; model.feat_scale.copy_(fs); model.train(was)
    return v/len(xs)/LOG2, m/len(xs)
def _rbg_add(model,cfg,x,s_):
    B_=model.__dict__.setdefault('_rbg',{})
    if s_ not in B_: B_[s_]=torch.zeros(256*256,device=x.device)
    B_[s_]+=torch.bincount((x[:,:-1].long()*256+x[:,1:].long()).reshape(-1),minlength=256*256).float()
def _rbg_gain(model,xs):
    """bits per byte pair each slot's profile saves over the pooled profile of all slots, on the decision batches (higher = more this slot's text)"""
    B_=model.__dict__.get('_rbg',{})
    if not B_: return {}
    h_=sum(torch.bincount((x_[:,:-1].long()*256+x_[:,1:].long()).reshape(-1),minlength=256*256).float() for x_ in xs); h_=h_/h_.sum()
    al_=0.5; P0_=sum(B_.values()); lp0_=torch.log2((P0_+al_)/(P0_.sum()+al_*P0_.numel())); ce0_=float(-(h_*lp0_).sum())
    return {s_:ce0_-float(-(h_*torch.log2((c_+al_)/(c_.sum()+al_*c_.numel()))).sum()) for s_,c_ in B_.items()}
def infer_slot_rep(model,cfg,get,t_true,vi):
    """cg9: representation candidate + loss check (see Config.infer_rep). Logged against the true domain (diagnostic only)."""
    S=model.__dict__.setdefault('_slots',[]); R=model.__dict__.setdefault('_slot_ref',{}); L=model.__dict__.setdefault('_slot_log',[])
    PR=model.__dict__.setdefault('_slot_proto',{}); TM=model.__dict__.setdefault('_slot_truth',{})
    if not S:
        S.append(0); L.append((vi,t_true,0,'new',None)); TM.setdefault(t_true,0)
        print(f"  [routing] visit {vi}: first context -> slot 0 (true {_lab(cfg,t_true)})"); return 0,[]
    b_=[get() for _ in range(cfg.infer_batches)]; xs=[a for a,_ in b_]; ys=[c for _,c in b_]
    sc={}; ds={}; BK=model.__dict__.get('_bank',{}); dev=model.embed.weight.device; DR_={}
    for s_ in S:
        bp_,m_=_slot_score(model,xs,ys,s_); sc[s_]=bp_
        if s_ in BK:                                                      # FRESH PROTOTYPE: the slot's own dream-bank rows read NOW (a stored
            Xb_=BK[s_][:16,:xs[0].shape[1]].long().to(dev)                 #   prototype goes stale as the trunk drifts: cg9 CPU check, the
            _,pm_=_slot_score(model,[Xb_],[torch.zeros_like(Xb_)],s_)
            if int(getattr(cfg,'infer_drift_fix',0)) and s_ in model.__dict__.get('_bank_ref',{}):   # cg20x: drift = BPC on own bank rows now - then
                Xf_=BK[s_][:16].long().to(dev); bb_,_=_slot_score(model,[Xf_[:,:-1]],[Xf_[:,1:]],s_); DR_[s_]=bb_-model.__dict__['_bank_ref'][s_]
                if int(getattr(cfg,'infer_drift_clamp',0)): DR_[s_]=max(0.0,DR_[s_])   # cg20y: a slot that IMPROVED gets no bonus (drift >= 0)      #   newest slot was always nearest)
            ds[s_]=float(1-F.cosine_similarity(m_,pm_,dim=0))
        else: ds[s_]=float('inf')
    G9_=_rbg_gain(model,xs); bg9_=int(getattr(cfg,'infer_bigram',0))
    cand=min(sc,key=sc.get); rnear=min(ds,key=ds.get)                     # candidate = lowest BPC; the representation must agree
    gap=sc[cand]-R.get(cand,sc[cand]); loss_ok=gap<=cfg.infer_delta; rep_ok=(rnear==cand) or not any(v_<float('inf') for v_ in ds.values())
    second=sorted(ds.values())[1] if len(ds)>1 else float('inf')
    if len(S)>=_nslots(cfg) and float(cfg.infer_gap_eps)>0 and all(s_ in R for s_ in S):   # cg13 GAP RULE (slots full): among slots within
        lo_=min(sc.values()); nr_={s_:sc[s_]-R[s_] for s_ in S if sc[s_]<=lo_+float(cfg.infer_gap_eps)}   #   infer_gap_eps of the lowest BPC, the one
        s_=min(nr_,key=nr_.get); kind='revisit (slots full, smallest gap to own reference)'
        if DR_:                                                               # cg20x: gaps relative to what each slot can still do now
            old_={'s':s_}; nr_={k_:v_-DR_.get(k_,0.0) for k_,v_ in nr_.items()}; s_=min(nr_,key=nr_.get)
            if s_!=old_['s']: kind='revisit (slots full, smallest DRIFT-CORRECTED gap)'             #   fitting the text best RELATIVE TO ITS OWN
        if (ds.get(rnear,float('inf'))<=cfg.infer_margin*second and (sc[rnear]-lo_)<float(cfg.infer_proto_guard) and rnear!=s_   #   reference (recency-biased slots
                and (float(cfg.infer_proto_gap)<=0 or (sc[rnear]-R.get(rnear,sc[rnear])-DR_.get(rnear,0.0))<=float(cfg.infer_proto_gap))):   # cg14d: the prototype's slot must fit its
            s_=rnear; kind='revisit (slots full, prototype clear)'                                     #   score low BPC on everything); clear prototype
        ps_=float(getattr(cfg,'infer_proto_strong',0.0))
        lst_=L[-1][2] if (L and int(getattr(cfg,'infer_proto_recency',0))) else None   # cg20z: not the slot trained at the previous visit (its fresh prototype sits too close to anything)
        if ps_>0 and second<float('inf') and ds.get(rnear,float('inf'))*ps_<=second and (sc[rnear]-lo_)<float(getattr(cfg,'infer_proto_strong_guard',0.25)) and rnear!=s_ and rnear!=lst_:
            s_=rnear; kind='revisit (slots full, STRONG prototype)'             # cg20y: a prototype 3x nearer than the next wins within 0.25 bits (cg20x GPU replay: fixes visit 16, changes nothing else)
        lr_=L[-1][2] if (L and int(getattr(cfg,'infer_proto_recency',0))>=2) else None   # cg20z (2): NO rule may pick the slot trained at the previous visit - no domain
        rp9_=bool(lr_ is not None and rnear==lr_ and ps_>0 and second<float('inf') and ds.get(rnear,float('inf'))*ps_<=second)   # FIX (review): a STRONG prototype for the last slot
        if lr_ is not None and s_==lr_ and len(sc)>1 and not rp9_:            #   is a genuine repeat (shuffled passes: a domain can follow itself) | no domain is visited twice running, and the fresh slot reads close to anything (cg20y GPU: visits
            alt_={k_:v_ for k_,v_ in nr_.items() if k_!=lr_}                    #   14 and 22, the only times the choice equalled the last slot, were both wrong)
            s_=min(alt_,key=alt_.get) if alt_ else (rnear if rnear!=lr_ else min((k_ for k_ in sc if k_!=lr_),key=sc.get))
            kind='revisit (slots full, not the slot trained at the last visit)'
        gap=sc[s_]-R.get(s_,sc[s_])-DR_.get(s_,0.0)                           #   override kept; replayed on cg9 / cg10 routing: 10/10 and 10/10
        if DR_: kind+=f" [old rule: slot {old_['s']}; drift "+' '.join(f"s{k_}:{v_:+.3f}" for k_,v_ in sorted(DR_.items()))+"]"

    elif len(S)>=_nslots(cfg) and not (rep_ok and loss_ok):           # cg10 SLOTS FULL: the lowest-BPC slot is biased towards the slot trained
        lo2_=sorted(sc,key=sc.get)[:2]                                    #   most recently (cg9: Gutenberg -> FreeLaw, Github -> StackExchange,
        clear_=ds[rnear]<=cfg.infer_margin*second and (sc[rnear]-sc[cand])<cfg.infer_delta                         #   the fresh prototype was right both times): follow the prototype when
        close_=bool(cfg.infer_tiebreak) and len(lo2_)>1 and (sc[lo2_[1]]-sc[lo2_[0]])<0.1 and rnear in lo2_   #   it is clearly nearest, or when the two lowest BPCs are within
        s_=rnear if (clear_ or close_) else cand                          #   0.1 bits and it is one of them; else the lowest BPC
        # cg10 guard: a clear prototype overrides only when its slot's BPC is within infer_delta of the lowest (cg10 CPU: prototypes 0.0005 vs 0.0012
        #   apart but 0.69 bits worse -> wrong); replayed on cg9 GPU routing: 10/10 slots-full revisits right (was 8/10), CPU 5/5 (was 4/5)
        kind='revisit (slots full, '+('prototype clear' if clear_ else 'prototype tie-break' if close_ else 'lowest BPC')+')'
        if s_!=cand: gap=sc[s_]-R.get(s_,sc[s_])
    elif bg9_>=2 and G9_:                                                   # BYTE-PAIR RULE (no cap needed): revisit the slot whose profile saves the most bits over
        gb9_=max(G9_,key=G9_.get)                                             #   the pooled profile, if it saves at least infer_bigram_tau bits per pair; else a new slot
        if G9_[gb9_]>=float(getattr(cfg,'infer_bigram_tau',0.3)): s_=gb9_; kind=f'revisit (byte-pair profile, {G9_[gb9_]:.2f} bits)'
        else: s_=len(S); S.append(s_); kind=f'new (best byte-pair gain {G9_[gb9_]:.2f} bits)'
    elif rep_ok and gap<=(float(cfg.infer_delta_new) if float(cfg.infer_delta_new)>0 else cfg.infer_delta): s_=cand; kind='revisit'   # cg13c free-slot threshold
    else: s_=len(S); S.append(s_); kind='new'
    exp_=TM.get(t_true); ok_=(s_==exp_) if exp_ is not None else (kind=='new')
    if exp_ is None: TM[t_true]=s_
    A=model.__dict__.setdefault('_route_acc',[0,0]); A[0]+=int(ok_); A[1]+=1
    L.append((vi,t_true,s_,kind,round(gap,3)))
    print(f"  [routing] visit {vi}: lowest-BPC slot {cand}, nearest fresh prototype slot {rnear} (agree {rep_ok}; distance {ds[rnear]:.4f}, second {second:.4f}) | BPC {sc[cand]:.3f} vs reference "
          f"{R.get(cand,float('nan')):.3f} (gap {gap:+.3f}, ok {loss_ok}) -> {kind} slot {s_} | true {_lab(cfg,t_true)}: {'CORRECT' if ok_ else 'WRONG'} "
          f"(running {A[0]}/{A[1]}) | distances "+' '.join(f"s{k}:{v:.3f}" for k,v in ds.items())+" | BPC "+' '.join(f"s{k}:{v:.2f}" for k,v in sc.items())
          +(" | byte-pair gain "+' '.join(f"s{k}:{v:.2f}" for k,v in sorted(G9_.items())) if G9_ else ""))
    return s_,b_
def infer_slot(model,cfg,get,t_true,vi):
    if getattr(cfg,'infer_rep',False): return infer_slot_rep(model,cfg,get,t_true,vi)
    """LABEL-FREE CONTEXT: (slot, batches drawn to decide it). The batches are returned so the visit trains on them (stream unchanged)."""
    S=model.__dict__.setdefault('_slots',[]); R=model.__dict__.setdefault('_slot_ref',{}); L=model.__dict__.setdefault('_slot_log',[])
    if not S:
        S.append(0); L.append((vi,t_true,0,'new',None)); print(f"  [context] visit {vi}: first context -> slot 0 (true {_lab(cfg,t_true)})"); return 0,[]
    b_=[get() for _ in range(cfg.infer_batches)]; xs=[a for a,_ in b_]; ys=[c for _,c in b_]
    sc={s_:_slot_bpc(model,xs,ys,s_) for s_ in S}; best=min(sc,key=sc.get); gap=sc[best]-R.get(best,sc[best])
    if gap<=cfg.infer_delta or len(S)>=_nslots(cfg): s_=best; kind='revisit'
    else: s_=len(S); S.append(s_); kind='new'
    L.append((vi,t_true,s_,kind,round(gap,3)))
    print(f"  [context] visit {vi}: best slot {best} (BPC {sc[best]:.3f}, reference {R.get(best,float('nan')):.3f}, gap {gap:+.3f}) -> {kind} slot {s_} (true {_lab(cfg,t_true)}) | "
          +' '.join(f"s{k}:{v:.2f}" for k,v in sc.items()))
    return s_,b_
def slot_ref(model,cfg,b_,s):
    """The slot's reference BPC at the visit end, on the visit's last infer_batches training batches (no extra text drawn)."""
    b_=list(b_)
    if getattr(cfg,'infer_rep',False):                                    # cg9: reference BPC and representation PROTOTYPE in one pass
        bp_,m_=_slot_score(model,[a for a,_ in b_],[c for _,c in b_],s)
        model.__dict__.setdefault('_slot_ref',{})[s]=bp_; model.__dict__.setdefault('_slot_proto',{})[s]=m_.detach(); return
    model.__dict__.setdefault('_slot_ref',{})[s]=_slot_bpc(model,[a for a,_ in b_],[c for _,c in b_],s)
def _ema_params(model):
    P=list(model.parameters())
    for k_ in ('_ngp','_ngpi'):
        T=model.__dict__.get(k_)
        if T: P+=[E.weight for d in sorted(T) for E in T[d]]
    return P
def _ngp_step_0(model,cfg,lr):
    O3=model.__dict__.get('_ngpi_opt')
    if O3 is not None:
        for g_ in O3.param_groups: g_['lr']=lr*cfg.ngram_lr_mult
        O3.step(); O3.zero_grad(set_to_none=True)
    O=model.__dict__.get('_ngp_opt')
    if O is not None:
        for g_ in O.param_groups: g_['lr']=lr*cfg.ngram_lr_mult
        if float(model.__dict__.get('_real_w',1.0))>0 or not getattr(cfg,'cd_ngp_frozen',False): O.step()   # FIX (review): closed loop, frozen tables - no momentum drift
        O.zero_grad(set_to_none=True)
    O4=model.__dict__.get('_ngo_opt')
    if O4 is not None:
        for g_ in O4.param_groups: g_['lr']=lr*cfg.ngram_lr_mult*cfg.ngo_lr_mult
        O4.step(); O4.zero_grad(set_to_none=True)
    O2=model.__dict__.get('_fcp_opt')
    if O2 is not None:
        for g_ in O2.param_groups: g_['lr']=lr*cfg.ngram_lr_mult
        O2.step(); O2.zero_grad(set_to_none=True)

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_lab': 'platypus.capacity',
    'bpc': 'platypus.inference.store',
}
