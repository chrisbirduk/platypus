"""Learned meta-controller, differentiable meta helpers and threshold gates."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from platypus.gating import (
    _grp, _imp, _ref, _theta,
)


# ---------------- learned meta-controller ----------------
class MetaController(nn.Module):
    """Takes per-unit features -> per-unit write-gate in [0,1]. Shared across all
    units/blocks (small, generalizes). Features: [usage, grad_mag, flow, frozen_frac]."""
    def __init__(s, n_feat=6, hid=16):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(n_feat,hid),nn.ReLU(),nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,1))
    def forward(s, feats):  # (N_units, n_feat) -> (N_units,) gate
        return torch.sigmoid(s.net(feats)).squeeze(-1)

@torch.no_grad()
def flow_importance(b):
    fi=b.Wk.weight.abs().sum(1)+b.Wv.weight.abs().sum(1); fo=b.Wout.weight.abs().sum(0)
    ssm=(fi/(fi.mean()+1e-8))*(fo/(fo.mean()+1e-8))
    mfi=b.mlp_in.weight.abs().sum(1); mfo=b.mlp_out.weight.abs().sum(0)
    mlp=(mfi/(mfi.mean()+1e-8))*(mfo/(mfo.mean()+1e-8))
    return ssm, mlp

def unit_features(b, prog):
    # per-unit features for SSM units and MLP units (prog = fraction of training done)
    ssm_flow, mlp_flow = flow_importance(b)
    def norm(x): return x/(x.mean()+1e-8)
    ssm_gm = norm(b.Wv.weight.grad.abs().mean(1)) if b.Wv.weight.grad is not None else torch.zeros_like(b.unit_usage)
    mlp_gm = norm(b.mlp_in.weight.grad.abs().mean(1)) if b.mlp_in.weight.grad is not None else torch.zeros_like(b.mlp_usage)
    ssm_imp=norm(b.unit_usage)*ssm_flow; mlp_imp=norm(b.mlp_usage)*mlp_flow
    ssm_pol=(norm(b.unit_usage)-norm(b.unit_home)).abs(); mlp_pol=(norm(b.mlp_usage)-norm(b.mlp_home)).abs()
    ssm=torch.stack([norm(b.unit_usage), ssm_gm, ssm_flow, ssm_imp, ssm_pol, torch.full_like(b.unit_usage,prog)],1)
    mlp=torch.stack([norm(b.mlp_usage), mlp_gm, mlp_flow, mlp_imp, mlp_pol, torch.full_like(b.mlp_usage,prog)],1)
    return ssm, mlp

# ---------------- MAML-style differentiable meta helpers ----------------
def _diff_features(b, grads_by_param, prog):
    """Differentiable per-unit features (uses graph-connected grads, not .grad buffers)."""
    ssm_flow, mlp_flow = flow_importance(b)   # no_grad (weight-based), fine as features
    def norm(x): return x/(x.mean()+1e-8)
    gWv=grads_by_param.get(id(b.Wv.weight)); gmlp=grads_by_param.get(id(b.mlp_in.weight))
    ssm_gm = norm(gWv.abs().mean(1)) if gWv is not None else torch.zeros_like(b.unit_usage)
    mlp_gm = norm(gmlp.abs().mean(1)) if gmlp is not None else torch.zeros_like(b.mlp_usage)
    # IMPORTANCE score (usage*flow - what plain importance-freezing ranks by; the floor)
    ssm_imp = norm(b.unit_usage)*ssm_flow; mlp_imp = norm(b.mlp_usage)*mlp_flow
    # POLARIZATION: |current usage - home usage| (far from home region = high)
    ssm_pol = (norm(b.unit_usage)-norm(b.unit_home)).abs(); mlp_pol=(norm(b.mlp_usage)-norm(b.mlp_home)).abs()
    ssm=torch.stack([norm(b.unit_usage), ssm_gm, ssm_flow, ssm_imp, ssm_pol, torch.full_like(b.unit_usage,prog)],1)
    mlp=torch.stack([norm(b.mlp_usage), mlp_gm, mlp_flow, mlp_imp, mlp_pol, torch.full_like(b.mlp_usage,prog)],1)
    return ssm, mlp

def _virtual_gated_step(model, meta, block_params, grads, prog, lr):
    """Form virtual updated params p' = p - lr*gate*grad, differentiable w.r.t. meta.
    Returns a dict {param_name: virtual_tensor} for functional_call."""
    grads_by_param={id(p): g for p,g in zip(block_params, grads)}
    # per-block gates (graph-connected through meta)
    gates={}
    for b in model.blocks:
        sf,mf=_diff_features(b, grads_by_param, prog)
        gs=meta(sf); gm=meta(mf)
        # apply the HARD freeze in the virtual step too (meta-training sees the floor)
        gs=gs*(~b.frozen_units).float(); gm=gm*(~b.mlp_frozen).float()
        gates[id(b.Wv.weight)]=gs.unsqueeze(1); gates[id(b.Wk.weight)]=gs.unsqueeze(1)
        gates[id(b.Wout.weight)]=gs.unsqueeze(0)
        gates[id(b.mlp_in.weight)]=gm.unsqueeze(1); gates[id(b.mlp_in.bias)]=gm
        gates[id(b.mlp_out.weight)]=gm.unsqueeze(0)
    # build full param dict for functional_call: virtual for gated params, real for rest
    virtual={}
    name_by_id={id(p):n for n,p in model.named_parameters()}
    gp_ids=set(id(p) for p in block_params)
    for n,p in model.named_parameters():
        if id(p) in gp_ids:
            g=grads_by_param[id(p)]
            if g is None: virtual[n]=p; continue
            gate=gates.get(id(p), 1.0)
            virtual[n]=p - lr*gate*g       # differentiable virtual update
        else:
            virtual[n]=p
    return virtual

def _functional_buffer_loss(model, virtual_params, x, y, domain):
    """Forward the model with the VIRTUAL params (differentiable) and return CE loss."""
    from torch.func import functional_call
    out=functional_call(model, virtual_params, (x, domain))
    return F.cross_entropy(out.reshape(-1,256), y.reshape(-1))

def _gate(drive, ema, theta0, imp, cfg, floor=None):
    d=drive/(ema+1e-8)
    return torch.clamp(torch.sigmoid((d-_theta(theta0,imp,cfg))/cfg.theta_temp), min=cfg.theta_floor if floor is None else floor)

def _comp_gate(drive, ema, theta0, imp, cfg, budget=None, count=None):
    """IMPEDANCE COMPETITION: impedance R = random base + alpha*importance. Each step, units are
    ranked by (relative drive - R); the write flows through the lowest-impedance path: the top
    write_budget fraction gets ~full write, the rest fall off sharply to comp_floor."""
    d=drive/(ema+1e-8)
    s=d-(theta0+imp)
    if count is not None: k=max(1,min(int(count),s.numel()))
    else: k=max(1,int(round((cfg.write_budget if budget is None else budget)*s.numel())))
    cut=torch.topk(s,k).values[-1]
    return torch.clamp(torch.sigmoid((s-cut)/cfg.comp_temp), min=cfg.comp_floor)

def _any_gate(drive, ema, theta0, imp, cfg, floor=None):
    # channels & embeddings: their own rule and budget
    return (_comp_gate(drive,ema,theta0,imp,cfg,cfg.channel_budget) if cfg.channel_gate_mode=='competitive'
            else _gate(drive,ema,theta0,imp,cfg,floor))

def threshold_gates(b, cfg):
    """Per-unit write gate: sigmoid((drive - threshold)/temp), threshold = theta0 + alpha*imp.
    Drive is normalized to mean 1 per layer. Floor keeps a trickle so dead units can wake."""
    def g(drive, ema, theta0, imp):
        # drive relative to THIS unit's typical drive: similar content that drives an
        # important unit at its usual (high) level no longer clears its raised threshold
        d=drive/(ema+1e-8)
        theta=_theta(theta0,imp,cfg)
        return torch.clamp(torch.sigmoid((d-theta)/cfg.theta_temp), min=cfg.theta_floor)
    sr,mr=_ref(b,'ssm',cfg),_ref(b,'mlp',cfg)          # SIGNAL 1
    si,mi=_imp(b,'ssm',cfg),_imp(b,'mlp',cfg)          # usage importance (+ SIGNAL 2)
    return (g(b.ssm_drive,sr,b.unit_theta0,si),
            g(b.mlp_drive,mr,b.mlp_theta0,mi))

def _mixer_snaps(b, gs, gc):
    """Write gates for the mixer-upgrade parameters: query like the keys (unit x channel), per-head beta projection by the mean
    unit gate of its head, convolutions by their unit's gate. (All joined to the shunt and block write strengths via snaps.)"""
    out=[]
    if getattr(b,'Wq',None) is not None: out.append((b.Wq.weight,gs[:,None]*gc[None,:],'full'))
    if getattr(b,'Wb',None) is not None:
        gh=gs.reshape(b.mh,-1).mean(1); out+= [(b.Wb.weight,gh[:,None]*gc[None,:],'full'),(b.Wb.bias,gh,'full')]
    if getattr(b,'Wa',None) is not None:
        gh=gs.reshape(b.mh,-1).mean(1); out+= [(b.Wa.weight,gh[:,None]*gc[None,:],'full'),(b.Wa.bias,gh,'full')]
    for nm in ('conv_k','conv_v','conv_q','conv_loc'):
        w=getattr(b,nm,None)
        if w is not None: out.append((w,gs[:,None,None],'full'))
    return out

@torch.no_grad()
def snapshot_threshold(model, cfg, meta_gates=None):
    """UPDATE-SPACE write gating. Adam normalizes gradient scale away, so scaling grads
    does ~nothing (verified: a 0.05 grad gate reduced Adam's update by 0.0%). Instead:
    snapshot gated weights BEFORE opt.step, then after the step move each unit only
    gate*(new-old). Exact write strength, independent of the optimizer."""
    snaps=[]; elig=[]
    GC=model.__dict__.setdefault('_gcache',{}); gs_step=getattr(model,'_gstep',0)
    fresh=cfg.gate_every<=1 or GC.get('step') is None or gs_step-GC['step']>=cfg.gate_every
    if fresh: GC.clear(); GC['step']=gs_step                              # A4: gates recomputed every gate_every steps
    for i,b in enumerate(model.blocks):
        if fresh or ('u',i) not in GC or GC[('u',i)][0].shape[0]!=b.d_state:
            gs,gm=threshold_gates(b,cfg)
            GC[('u',i)]=(gs,gm,(gs>0.5).float().mean())
        gs,gm,el_=GC[('u',i)]; elig.append(el_)                        # tensor: converted only when printed
        gs=torch.ones_like(gs); gm=torch.ones_like(gm)
        if cfg.route_writes:                  # WRITE ROUTING: content writes to units matching its distinctness
            rs=b.write_route(b.unit_aff); rm=b.write_route(b.mlp_aff)
            gs=gs*rs; gm=gm*rm
            if i==0: model._route_mean=float(rs.mean())
        if meta_gates is not None:            # combined write strength = threshold x meta
            gs=gs*meta_gates[i][0]; gm=gm*meta_gates[i][1]
        # ALL LAYERS: each weight's write strength = (its unit's gate) x (its channel's gate)
        if fresh or ('c',i) not in GC: GC[('c',i)]=_any_gate(b.ch_drive,_ref(b,'ch',cfg),b.ch_theta0,_imp(b,'ch',cfg),cfg,_grp(cfg,'theta_floor','ch'))
        gc=GC[('c',i)]
        gc=torch.ones_like(gc)
        if i==0: model._ch_elig=(gc>0.5).float().mean()
        full=[(b.Wv.weight,gs[:,None]*gc[None,:]),(b.Wk.weight,gs[:,None]*gc[None,:]),
              (b.gate.weight,gs[:,None]*gc[None,:]),(b.gate.bias,gs),
              (b.Wout.weight,gc[:,None]*gs[None,:]),
              (b.mlp_in.weight,gm[:,None]*gc[None,:]),(b.mlp_in.bias,gm),
              (b.mlp_out.weight,gc[:,None]*gm[None,:]),(b.mlp_out.bias,gc),
              (b.ln.weight,gc),(b.ln.bias,gc),(b.norm.weight,gc),(b.norm.bias,gc),
              (b.beta,gc.mean()),(b.unit_affinity,gs),(b.mlp_affinity,gm)]
        full+=[(p,gt) for p,gt,_ in _mixer_snaps(b,gs,gc)]
        for p,gt in full:
            if p is None: continue
            snaps.append((p,p.data.clone(),gt,'full'))
    cm=getattr(model,'_og_common',None)
    if cm and id(model.pos.weight) in cm and not (cfg.channel_thresholds and cfg.emb_thresholds):
        snaps.append((model.pos.weight, model.pos.weight.data.clone(), torch.ones(1,device=model.pos.weight.device), 'full'))
    return snaps, torch.stack([e.float() for e in elig]).mean()

@torch.no_grad()
def apply_update_gate(snaps, og=None, og_in=1.0, og_orth='gate', stats=None, orth_boost=1.0, common=None, common_scale=1.0, og_out=None, shunt=None):
    """og: {id(weight): U} old-domain input subspaces (ORTHOGONAL-GATED UPDATE). Each matrix update d
    is split into din = d U U^T (inside the old subspace: moves old-domain outputs) and dorth = d - din.
      din   is written at gate * og_in        (og_in=0: interfering part removed; 1: gated as usual)
      dorth is written at gate (og_orth='gate') or at a fixed strength og_orth (e.g. 1.0 = free).
    Free orthogonal writes are only safe when U holds ~all old energy (og_energy ~0.99): with a 50%
    subspace the complement still carries half the old-domain input energy."""
    for p,old,gt,kind in snaps:
        g = gt.unsqueeze(1) if kind=='row' else (gt.unsqueeze(0) if kind=='col' else gt)   # 'vec'/'full': as given
        if common and id(p) in common: g=g*common_scale          # common-mode params (biases, LayerNorm, beta)
        U = og.get(id(p)) if og else None
        if U is None or U.shape[0]!=p.shape[-1]:
            p.data.copy_(old + g*(p.data-old)); continue
        d=p.data-old; din=(d@U)@U.T; dorth=d-din
        so=(g if orth_boost==1.0 else torch.clamp(g*orth_boost,max=1.0)) if og_orth=='gate' else og_orth
        if stats is not None:
            stats[0]+=(din*din).sum(); stats[1]+=((g*og_in*din)**2).sum(); stats[2]+=(d*d).sum()   # tensors: no GPU sync
        if shunt is not None:                        # SHUNT: interfering part written, but tracked separately
            sh=g*din; prev=shunt.get(id(p))
            shunt[id(p)]=sh if (prev is None or prev.shape!=sh.shape) else prev+sh
            upd=sh + so*dorth
        else: upd=(g*og_in)*din + so*dorth
        Uo=og_out.get(id(p)) if og_out else None
        if Uo is not None and Uo.shape[0]==p.shape[0]: upd=upd-Uo@(Uo.T@upd)     # two-sided: no change along old dominant OUTPUT directions
        p.data.copy_(old + upd)
