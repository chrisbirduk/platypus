"""Data-free input covariances, importance and revisit bookkeeping."""
import math
import torch
from platypus.gating import (
    _f32,
)
from platypus.capacity import (
    _lab,
)
from platypus.meta import (
    flow_importance, unit_features,
)


# ---- DATA-FREE running input covariances (orthogonal gating without stored samples)
def install_cov_hooks(model, cfg):
    """While training, every cov_every steps, a token subsample of each projectable layer's input is added to a running
    covariance for the CURRENT domain. At domain end it is folded (equal weight per domain) into the old-domain covariance."""
    object.__setattr__(model,'_cur_cov',{}); object.__setattr__(model,'_old_cov',{}); object.__setattr__(model,'_cov_on',False)
    object.__setattr__(model,'_cov_handles',{})
    for li,b in enumerate(model.blocks): cov_hook_block(model, cfg, li)

def cov_hook_block(model, cfg, li):
    """(Re-)register the running-covariance hooks of block li (needed after a widening replaces its layers)."""
    b=model.blocks[li]; H=model._cov_handles
    for key in ORTH_KEYS:
        if (li,key) in H: H[(li,key)].remove()
        def hk(m,inp,out,k=(li,key)):
            if not (model._cov_on and m.training): return
            with torch.no_grad():
                x_=inp[0][:model.__dict__.get('_nstat')].reshape(-1,inp[0].shape[-1]).float(); x_=x_[::max(1,x_.shape[0]//cfg.cov_tokens)]
                c,n=model._cur_cov.get(k,(0.0,0)); model._cur_cov[k]=(c+x_.T@x_, n+x_.shape[0])
        H[(li,key)]=getattr(b,key).register_forward_hook(_f32(hk))

@torch.no_grad()
def fold_domain_covariance(model, dom=None):
    last={}
    for k,(c,n) in model._cur_cov.items():
        C=c/max(n,1); oc,od=model._old_cov.get(k,(0.0,0))
        model._old_cov[k]=(oc+C, od+1)              # stored as (sum of per-domain covariances, number of domains)
        last[k]=C
    lastm={}
    for k,(ms,mn) in (model.__dict__.get('_cur_mean') or {}).items():   # centring: per-domain means, pooled like the covariances
        m_=ms/max(mn,1); om,odn=model.__dict__.setdefault('_old_mean',{}).get(k,(0.0,0)); model._old_mean[k]=(om+m_, odn+1); lastm[k]=m_
    if dom is not None:                             # REPEATED PASSES: each domain's latest covariance kept (CPU) so that a revisit
        model.__dict__.setdefault('_dom_cov',{})[dom]={k:C.detach().cpu() for k,C in last.items()}   # protects only the OTHERS
        if lastm: model.__dict__.setdefault('_dom_mean',{})[dom]={k:m_.detach().cpu() for k,m_ in lastm.items()}
    object.__setattr__(model,'_last_cov',last)      # the newest old domain's own covariance (light protection option)
    model._cur_cov={}; model.__dict__['_cur_mean']={}

@torch.no_grad()
def rebuild_old_cov(model, olds):
    """Old-input statistics = the latest covariance of every OTHER domain seen so far (equal weight per domain)."""
    DC=getattr(model,'_dom_cov',None) or {}
    if not DC: return
    dev=model.embed.weight.device; acc={}
    for d in olds:
        for k,C in DC.get(d,{}).items():
            c,n=acc.get(k,(0.0,0)); acc[k]=(c+C.to(dev), n+1)
    model._old_cov=acc
    DM=getattr(model,'_dom_mean',None) or {}
    if DM:
        am={}
        for d in olds:
            for k,m_ in DM.get(d,{}).items():
                c,n=am.get(k,(0.0,0)); am[k]=(c+m_.to(dev), n+1)
        model.__dict__['_old_mean']=am

def _imp_T(v, cfg):
    """Importance transform of a per-neuron score (normalised to mean 1)."""
    n=v/(v.mean()+1e-8)
    t=cfg.imp_transform
    if t=='relu': return torch.relu(n-1.0)            # D: only above-average neurons gain importance
    if t=='raw':  return n                            # every neuron gains, proportional to its score
    if t=='sqrt': return torch.sqrt(torch.relu(n-1.0))
    if t=='log':  return torch.log1p(torch.relu(n-1.0))
    if t=='rank':                                     # top imp_top fraction get +1, rest 0
        k=max(1,int(round(cfg.imp_top*v.numel()))); out=torch.zeros_like(v); out[torch.topk(v,k).indices]=1.0; return out
    raise ValueError(t)

@torch.no_grad()
def importance_increments(model, cfg):
    """This domain's importance increment per neuron group (no mutation)."""
    inc=[]
    for b in model.blocks:
        ssm_flow, mlp_flow = flow_importance(b)
        su=(b.unit_usage/(b.unit_usage.mean()+1e-8))*ssm_flow; mu=(b.mlp_usage/(b.mlp_usage.mean()+1e-8))*mlp_flow
        if su is None: inc.append((torch.zeros_like(b.unit_imp),torch.zeros_like(b.mlp_imp),torch.zeros_like(b.ch_imp)))
        else: inc.append((_imp_T(su,cfg),_imp_T(mu,cfg),_imp_T(b.ch_usage,cfg)))
    emb=_imp_T(model.emb_usage,cfg)
    return inc, emb

@torch.no_grad()
def update_importance(model, cfg):
    """After a domain: decay old importance (imp_decay) and add this domain's increment, raising
    those neurons' write thresholds for all later domains."""
    inc,emb=importance_increments(model,cfg)
    for b,(u,m,c) in zip(model.blocks,inc):
        b.unit_imp.mul_(cfg.imp_decay).add_(u); b.mlp_imp.mul_(cfg.imp_decay).add_(m)
        b.ch_imp.mul_(cfg.imp_decay).add_(c); b.ch_usage.zero_()
    model.emb_imp.mul_(cfg.imp_decay).add_(emb); model.emb_usage.zero_()

_REV_BUFS=('unit_imp','mlp_imp','ch_imp','unit_sens','mlp_sens','ch_sens')
def _rev_items(model):
    return [(b,n) for b in model.blocks for n in _REV_BUFS]+[(model,'emb_imp'),(model,'emb_sens')]

@torch.no_grad()
def revisit_begin(model, cfg, t):
    """Visit start: take domain t's own stored importance / sensitivity out of the totals; remember the totals without it."""
    D=model.__dict__.setdefault('_imp_dom',{}); own=D.get(t)
    Lf=model.__dict__.setdefault('_rev_left',{}); vi_=getattr(model,'_visit',0); tau=cfg.imp_recency_tau; c0=cfg.imp_recency_floor
    def w_(d):
        if tau<=0: return 1.0
        age=max(0,vi_-Lf.get(d,vi_-1)-1); return c0+(1.0-c0)*math.exp(-age/tau)
    oth=[(w_(d),c) for d,c in D.items() if d!=t]
    for j,(o,n) in enumerate(_rev_items(model)):
        cur=getattr(o,n); m=torch.zeros_like(cur)
        for w,c in oth: m=torch.maximum(m,w*c[j])
        setattr(o,n,m)
    if tau>0 and oth: print('  [recency] protection weight per old domain: '+' '.join(f'{_lab(None,d)} {w_(d):.2f}' for d in sorted(D) if d!=t))
    model.__dict__['_imp_base']=[getattr(o,n).clone() for o,n in _rev_items(model)]
    return own is not None

@torch.no_grad()
def revisit_end(model, cfg, t):
    """Visit end (after importance and sensitivity are updated): this visit's contribution becomes domain t's stored one."""
    D=model.__dict__.setdefault('_imp_dom',{}); base=model.__dict__.pop('_imp_base',None)
    if base is None: return
    new=[getattr(o,n)-b0 for (o,n),b0 in zip(_rev_items(model),base)]
    D[t]=[c.clamp_min(0) for c in new]; model.__dict__.setdefault('_rev_left',{})[t]=getattr(model,'_visit',0)
    for j,(o,n) in enumerate(_rev_items(model)):
        m=torch.zeros_like(getattr(o,n))
        for c in D.values(): m=torch.maximum(m,c[j])
        setattr(o,n,m)
    return
    D[t]=new

def _imp_bufs(model):
    return [(b,'unit_imp') for b in model.blocks]+[(b,'mlp_imp') for b in model.blocks]+\
           [(b,'ch_imp') for b in model.blocks]+[(model,'emb_imp')]

@torch.no_grad()
def online_importance(model, cfg, base, frac):
    """ONLINE importance: during a domain, imp = importance before the domain + frac * the domain's
    increment so far (the current domain's key neurons start protecting themselves mid-domain)."""
    inc,emb=importance_increments(model,cfg)
    flat=[x[0] for x in inc]+[x[1] for x in inc]+[x[2] for x in inc]+[emb]
    for (o,nm),b0,d in zip(_imp_bufs(model),base,flat):
        pad=torch.cat([b0,torch.zeros(d.numel()-b0.numel(),device=d.device)]) if d.numel()>b0.numel() else b0
        setattr(o,nm,pad+frac*d)

def apply_meta_gate(model, meta, prog):
    """HARD importance-freeze (the +0.32 floor: frozen units get gate 0) THEN soft
    meta-gate modulates the non-frozen units. This guarantees the controller's floor
    IS importance-freezing (hard freeze always applied) - it can only improve by
    soft-protecting the long tail. (Fix: the previous version had no hard freeze and
    a sigmoid gate that couldn't reproduce it, so it couldn't even match importance.)"""
    for b in model.blocks:
        ssm_f, mlp_f = unit_features(b, prog)
        gs=meta(ssm_f); gm=meta(mlp_f)                       # soft gate in (0,1)
        gs=gs.clone(); gm=gm.clone()
        gs[b.frozen_units]=0.0; gm[b.mlp_frozen]=0.0         # HARD freeze on top (floor)
        if b.Wv.weight.grad is not None: b.Wv.weight.grad*=gs.unsqueeze(1)
        if b.Wk.weight.grad is not None: b.Wk.weight.grad*=gs.unsqueeze(1)
        if b.Wout.weight.grad is not None: b.Wout.weight.grad*=gs.unsqueeze(0)
        if b.mlp_in.weight.grad is not None: b.mlp_in.weight.grad*=gm.unsqueeze(1)
        if b.mlp_in.bias.grad is not None: b.mlp_in.bias.grad*=gm
        if b.mlp_out.weight.grad is not None: b.mlp_out.weight.grad*=gm.unsqueeze(0)

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    'ORTH_KEYS': 'platypus.orth',
}
