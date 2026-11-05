"""Write gating: thresholds from impedance and importance, drive statistics, polarisation-gated writes with split weight gradients."""
import torch
import torch.nn.functional as F
from platypus.config import (
    Config,
)


_NOISE={'gen':None}
_RAMP={'s':1.0}                     # base-impedance strength (theta0_ramp_domains), set per domain in train_domain
def _theta(theta0, imp, cfg):
    """Effective threshold = base impedance + alpha*importance (+ per-step impedance noise)."""
    th=(theta0 if _RAMP['s']==1.0 else _RAMP['s']*theta0)+imp   # imp arrives alpha-weighted from _imp (per-group strength)
    return th

_PG_KEY={'Wk':'Wk','Wv':'Wk','Wq':'Wk','Wb':'Wk','Wa':'Wk','gate':'gate','mlp_in':'mlp_in','Wout':'Wout','mlp_out':'mlp_out'}

def _f32(fn):
    """Statistics hooks run with autocast OFF: under bf16 mixed precision their matmuls (covariances etc.) would otherwise be
    computed and stored in bf16."""
    def w(m, inp, out):
        with torch.autocast(device_type=inp[0].device.type, enabled=False): return fn(m, inp, out)
    return w
def _alignment(x, U):
    """Per-token polarisation: share of the input vector's energy along the old-domain subspace U
    (cos^2 of its angle to that subspace). x (...,d), U (d,k) -> (...)"""
    xu=x@U; return (xu*xu).sum(-1)/(x*x).sum(-1).clamp(min=1e-12)

_DSPLIT={'on':False,'keys':('Wk','Wv','gate','Wout','mlp_in','mlp_out')}
class _SplitLinear(torch.autograd.Function):
    """Linear layer whose WEIGHT gradient is split by rows: real rows [:ns] and dream rows [ns:] give two products (same FLOPs as one);
    the total is returned as usual and the dream part is stashed on the weight (W._gd) for the optimiser (dream momentum).
    Forward under the caller's autocast; backward reproduces autocast's casts (inputs and weight saved in the forward dtype)."""
    @staticmethod
    def forward(ctx, x, W, b, ns):
        y=F.linear(x,W,b); dt=y.dtype
        ctx.save_for_backward(x.to(dt),W.to(dt)); ctx.ns=ns; ctx.hasb=b is not None; ctx.xdt=x.dtype; ctx.Wdt=W.dtype; ctx.W=W
        return y
    @staticmethod
    def backward(ctx, gy):
        xc,Wc=ctx.saved_tensors; ns=ctx.ns; gy=gy.to(xc.dtype)
        gx=(gy@Wc).to(ctx.xdt) if ctx.needs_input_grad[0] else None
        o_=gy.shape[-1]; i_=xc.shape[-1]
        gr=torch.mm(gy[:ns].reshape(-1,o_).t(),xc[:ns].reshape(-1,i_)).float()
        gd=torch.mm(gy[ns:].reshape(-1,o_).t(),xc[ns:].reshape(-1,i_)).float()
        W=ctx.W; od_=W.__dict__.get('_gd'); W.__dict__['_gd']=gd if od_ is None else od_+gd
        gb=gy.reshape(-1,o_).float().sum(0) if ctx.hasb else None
        return gx,(gr+gd).to(ctx.Wdt),gb,None
def _wl(b, key, x):
    """Linear layer with POLARISATION-GATED weight writes. Forward output is unchanged; only the input
    seen by the WEIGHT gradient is modified, per token, from the angle between the token's input (the
    combined previous-layer activity) and the old domains' input subspace:
      'weight'       : tokens more old-aligned than the batch average write less (w = 2*sigmoid(-kappa*z))
      'proj'         : the old-subspace part of each token's write is removed in proportion to how
                       old-aligned the token is (s = sigmoid(kappa*z))
      'proj_uniform' : removed for every token (s = 1): gradient-space control for 'proj'
    Input gradients are exact: F.linear(xw,W) + F.linear(x-xw, W.detach()) has d/dx = W^T delta."""
    mod=getattr(b,key); pg=getattr(b,'_pg',None)
    if _DSPLIT['on'] and not pg and key in _DSPLIT['keys'] and b.training and torch.is_grad_enabled() and x.dim()==3:
        ns_=getattr(b,'_nstat',None)
        if ns_ is not None and 0<ns_<x.shape[0]:                         # DREAM / REAL weight-gradient split (dream momentum)
            y=_SplitLinear.apply(x,mod.weight,mod.bias,int(ns_))
            for h_ in mod._forward_hooks.values():                       # module hooks (Kalman statistics) as a module call would run them
                r_=h_(mod,(x,),y)
                if r_ is not None: y=r_
            return y
    if not pg or not b.training or not torch.is_grad_enabled(): return mod(x)
    U=pg.get(_PG_KEY[key])
    if U is None or U.shape[0]!=x.shape[-1]: return mod(x)
    with torch.no_grad():
        a=_alignment(x.detach().float(),U); z=(a-a.mean())/(a.std()+1e-6)
    if b._pg_mode=='proj_seqnov':
        # SEQUENCE NOVELTY: whitened (Mahalanobis) distance to the old domains' input distribution, averaged
        # over each sequence; typical sequences (below the batch average) get the old-subspace part removed
        with torch.no_grad():
            mu,V,lam=b._pg_maha[_PG_KEY[key]]
            if mu.shape[0]!=x.shape[-1]: return mod(x)
            nov=((((x.detach().float()-mu)@V)**2)/lam).sum(-1).mean(-1)          # (B,)
            zs=(nov-nov.mean())/(nov.std()+1e-6)
            sc=torch.sigmoid(-b._pg_kappa*zs).view(-1,*([1]*(x.dim()-1)))
        return F.linear(x-sc*((x@U)@U.T),mod.weight,mod.bias)+F.linear(sc*((x@U)@U.T),mod.weight.detach())
    if b._pg_mode=='weight': xw=(2*torch.sigmoid(-b._pg_kappa*z)).unsqueeze(-1)*x
    else:
        sc=1.0 if b._pg_mode=='proj_uniform' else torch.sigmoid(b._pg_kappa*z).unsqueeze(-1)
        xw=x-sc*((x@U)@U.T)
    return F.linear(xw,mod.weight,mod.bias)+F.linear(x-xw,mod.weight.detach())

@torch.no_grad()
def _track(obj, name, drive):
    """Update a neuron group's drive statistics: fast EMA (D), slow EMA, and running peak
    of the fast EMA. name in {'ssm','mlp','ch','emb'} -> buffers <name>_drive_{ema,slow,peak}."""
    a=Config.drive_ema; ema=getattr(obj,name+'_drive_ema')
    ema=torch.where(ema==0, drive, a*ema+(1-a)*drive); setattr(obj,name+'_drive_ema',ema)
    a2=Config.drive_slow; sl=getattr(obj,name+'_drive_slow')
    setattr(obj,name+'_drive_slow', torch.where(sl==0, drive, a2*sl+(1-a2)*drive))
    pk=getattr(obj,name+'_drive_peak')
    setattr(obj,name+'_drive_peak', torch.maximum(pk*Config.peak_decay, ema))

def _ref(obj, name, cfg):
    """SIGNAL 1: the reference a unit's current drive is compared against."""
    ema=getattr(obj,name+'_drive_ema')
    return ema
    raise ValueError(cfg.drive_ref)

_IMP_OF={'ssm':'unit','mlp':'mlp','ch':'ch','emb':'emb'}
def _imp(obj, name, cfg):
    """Effective importance = usage importance (+ SIGNAL 2: sens_alpha * loss sensitivity)."""
    p=_IMP_OF[name]; imp=getattr(obj,p+'_imp')
    a=_grp(cfg,'imp_alpha',name)
    v=a*imp
    fl=getattr(cfg,'imp_floor',0.0)
    if fl>0 and v.numel()>1:                                   # the least-important share gates as if fresh
        k=max(1,int(fl*v.numel())); thr=torch.kthvalue(v.flatten().float(),k).values
        v=torch.where(v<=thr,torch.zeros_like(v),v)
    return v

def _grp(cfg, attr, name):
    """Per-group override: cfg.<attr>_<group> (group in ssm/mlp/ch/emb) if set, else cfg.<attr>."""
    v=getattr(cfg, f'{attr}_{name}', None)
    return getattr(cfg, attr) if v is None else v
