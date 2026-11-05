"""Dormant reserve, uniqueness-driven activation, per-domain weight deltas."""
import math
import torch
import torch.nn.functional as F
from platypus.config import (
    PILE_DOMAINS,
)
from platypus.private import (
    priv,
)


# ---------------- DORMANT RESERVE + UNIQUENESS-DRIVEN ACTIVATION ----------------
# The network is allocated at reserve_mult x its base size from the start (state units and MLP units;
# residual channels stay fixed). Only an initial set computes; the rest are DORMANT (fixed random receptive
# fields, no compute, no writes). Each block runs its own uniqueness detector: the whitened (Mahalanobis)
# distance of each sequence's mean input from everything the block has seen, relative to its typical
# novelty. When the ratio passes uniq_thr, the block activates a number of dormant units proportional to
# how unique the content is: the ones whose receptive fields respond most to what is NEW in the input
# (input minus the familiar mean). Activated units start with zero output weights (no disruption).
class Reserve:
    def __init__(s, d, n_state, n_mlp, dev):
        s.Wk=torch.randn(n_state,d,device=dev)*0.02; s.Wv=torch.randn(n_state,d,device=dev)*0.02
        s.gate=torch.randn(n_state,d,device=dev)*0.02; s.gate_b=torch.zeros(n_state,device=dev)
        bd=1/math.sqrt(d); s.min=(torch.rand(n_mlp,d,device=dev)*2-1)*bd; s.min_b=(torch.rand(n_mlp,device=dev)*2-1)*bd
    def take_state(s, idx):
        keep=torch.ones(s.Wk.shape[0],dtype=torch.bool,device=s.Wk.device); keep[idx]=False
        rows={'Wk':s.Wk[idx],'Wv':s.Wv[idx],'gate':s.gate[idx],'gate_b':s.gate_b[idx]}
        s.Wk,s.Wv,s.gate,s.gate_b=s.Wk[keep],s.Wv[keep],s.gate[keep],s.gate_b[keep]; return rows
    def take_mlp(s, idx):
        keep=torch.ones(s.min.shape[0],dtype=torch.bool,device=s.min.device); keep[idx]=False
        w,b=s.min[idx],s.min_b[idx]; s.min,s.min_b=s.min[keep],s.min_b[keep]; return w,b

class Uniq:
    """Per-block uniqueness detector on sequence-mean input features."""
    def __init__(s, d, dev):
        s.mu=torch.zeros(d,device=dev); s.S=torch.zeros(d,d,device=dev); s.mu_m=torch.zeros(d,device=dev)
        s.Sinv=None; s.n=0; s.nov_s=None; s.base=None; s.last=float('nan')
    @torch.no_grad()
    def update(s, h, hm, cfg):
        f=h.float().mean(1); B,d=f.shape
        if s.Sinv is not None:
            dl=f-s.mu; nov=float(((dl@s.Sinv)*dl).sum(-1).mean())/d; s.last=nov
            s.nov_s=nov if s.nov_s is None else 0.9*s.nov_s+0.1*nov
            s.base=s.nov_s if s.base is None else (1-cfg.uniq_base_rate)*s.base+cfg.uniq_base_rate*s.nov_s
        a=max(cfg.uniq_rate,1.0/(s.n+1))
        s.mu=(1-a)*s.mu+a*f.mean(0); dc=f-s.mu; s.S=(1-a)*s.S+a*(dc.T@dc)/B
        s.mu_m=(1-a)*s.mu_m+a*hm.float().mean((0,1)); s.n+=1
        if (s.Sinv is None and s.n>=cfg.uniq_min_stats) or (s.Sinv is not None and s.n%cfg.recruit_every==0):
            S_=torch.nan_to_num(s.S.double()); S_=0.5*(S_+S_.T)                 # double precision (the fp32 GPU solver can fail on near-degenerate S)
            try: lam,V=torch.linalg.eigh(S_)
            except Exception:
                try: lam,V=torch.linalg.eigh(S_.cpu()+1e-9*torch.eye(S_.shape[0],dtype=S_.dtype)); lam,V=lam.to(S_.device),V.to(S_.device)
                except Exception: return                                      # keep the previous inverse
            lam=lam.clamp(min=0)+cfg.uniq_shrink*lam.mean().clamp(min=1e-12)
            s.Sinv=((V/lam)@V.T).to(s.S.dtype)
    def ratio(s): return s.nov_s/s.base if (s.base is not None and s.base>0) else 1.0

@torch.no_grad()
def _select(resp_rows, feats, center, n, cfg, gen):
    """Pick n reserve units: highest mean |response| to the NOVEL part of the input (or random)."""
    X=feats.reshape(-1,feats.shape[-1]).float(); X=X[::max(1,X.shape[0]//2048)]-center
    return torch.topk((X@resp_rows.T).abs().mean(0),n).indices

@torch.no_grad()
def recruit(model, cfg, forced=None):
    """Uniqueness-driven activation (forced: {'state':n,'mlp':n} for the eligibility fallback).
    Returns list of (block, n_state, n_mlp, ratio)."""
    out=[]
    for li,(b,R,U) in enumerate(zip(model.blocks,model._reserve,model._uniq)):
        r=U.ratio()
        if forced is None:
            if U.n<cfg.uniq_warmup or r<cfg.uniq_thr: continue
            amt=min(r-1.0,2.0)                                   # proportional to how far above typical novelty
            ns=int(round(cfg.recruit_rate*cfg.d_state*amt)); nm=int(round(cfg.recruit_rate*2*cfg.d_model*amt))
        else: ns,nm=forced['state'],forced['mlp']
        ns=min(ns,R.Wk.shape[0]); nm=min(nm,R.min.shape[0])
        if ns<=0 and nm<=0: continue
        dev=b.Wk.weight.device
        if ns>0: b.widen(ns,dev,rows=R.take_state(_select(R.Wv,b._uh,U.mu,ns,cfg,model._rgen)))
        if nm>0:
            w,bb=R.take_mlp(_select(R.min,b._um,U.mu_m,nm,cfg,model._rgen)); b.widen_mlp(nm,dev,w,bb)
        out.append((li,ns,nm,r))
    return out

# ---------------- PARALLEL SLOW NETWORK / FAST-SLOW WEIGHTS (replay substitutes) ----------------
_HEAVY=('_sm','_sm_log','_tr','_hyb','_vals_share','_share_res','_bank','_reserve','_uniq','_orth','_og','_og_out','_og_common','_recruit_log','_polar_log','_slow','_slow_list','_fast','_replay_rng','_replay_buffer','_shunt','_recent','_live','_cur_cov','_old_cov','_last_cov','_dom_cov','_mid','_beta_log','_olds','_visit','_cov_handles','_tm','_tm_dream','_timing_log','_dpick','_cov_on','_topics','_klw','_klw_opt','_merge_log','_vals','_recon_stat','_recon_log','_recon_share','_recon_unew','_sleep_log','_stale_log','_mk_gen','_battr_acc','_bw_dyn','_routers','_ropt','_rver','_rcache','_bw_map','_tcombine_log','_tpeak','_tkl','_tdream','_trng','_tmerge_flag','_nb','_surprise_flag','_tswitch_log','_tdom_start','_tuse','_tpi','_tshunt_on','_tlast_merge','_tmerge_log','_tspawn_log','_gstep','_last_nll','_nll_ema','_tlast_surprise','_dworst','_dfilt','_dmax','_dlam','_dal','_snapring','_rb_fire','_rb_last','_rb_log','_spec','_cross','_xcur','_live_dom','_cross_log','_dreams','_dream_rng','_dream_gen','_dreamstat','_klstat','_klam','_bg','_fs','_last_h','_keep_h','_curve','_curve_val','_teach','_hstk','_dream_tgt','_dqueue','_dq_rng','_klacc','_mids','_dstk','_dtstk','_dscore','_prio_ix','_prio_gen','_prio_ess','_ctxacc','_surpacc','_cur_mean','_old_mean','_dom_mean','_onpol','_onpol_gen','_onpol_pick','_opstat','_op_n','_fastat','_upper_ids','_vals_diag','_fidlog','_klpos_log','_diag_gen','_dstream','_dsq','_dsr','_dstream_last','_sabasis','_hconf','_hconf_last','_strandcnt','_strandstat','_srng','_fdrift_log','_sastat','_shbasis','_shdiag','_shstat','_fsplit_log','_vlog','_vbank','_vslopes','_dsteer','_dsteer_acc','_cov_log','_dcls','_iwtab','_spec_log','_vstart','_gcache','_lwfG','_nstat','_ng_touch','_ng_state','_ng_cur','_ng_old','_ng_sc','_cnt_old','_cnt_cur','_ng_batch','_ng_lrms','_ng_lstate','_ng_dream','_ng_lwf','_lwfwacc','_relaxlog')

def _pmap(model): return {id(p_):p_ for p_ in model.parameters()}

def _lab(cfg,d): return PILE_DOMAINS[d][:6] if 0<=int(d)<len(PILE_DOMAINS) else f"slot{int(d)}"

def domain_delta(model, d):
    """Domain d's own configuration relative to the shared main weights: its specific delta + its cross gains x
    the earlier domains' specific deltas."""
    spec=getattr(model,'_spec',None) or {}; out={}
    for k,v in spec.get(d,{}).items(): out[k]=v.clone()
    X=(getattr(model,'_cross',None) or {}).get(d)
    if X:
        for jj,j in enumerate(X['olds']):
            for kk,k in enumerate(X['keys']):
                if k in spec.get(j,{}) and X['A'][jj,kk]!=0:
                    add=X['A'][jj,kk]*spec[j][k]; out[k]=out[k]+add if k in out else add.clone()
    return out

def current_mix(model):
    """The cross-connection contribution currently in the live weights (domain being trained)."""
    X=getattr(model,'_xcur',None); spec=getattr(model,'_spec',None) or {}; out={}
    if not X: return out
    for jj,j in enumerate(X['olds']):
        for kk,k in enumerate(X['keys']):
            a=float(X['A'][jj,kk])
            if a!=0 and k in spec.get(j,{}):
                add=a*spec[j][k]; out[k]=out[k]+add if k in out else add.clone()
    return out

@torch.no_grad()
def apply_delta(model, delta, sign):
    P=_pmap(model)
    for k,v in delta.items():
        if k in P and P[k].shape==v.shape: P[k].add_(sign*v)

@torch.no_grad()
def dream_sample(model, d, n, T, temp, start=10, gen=None):
    """Old head d generates n byte sequences of length T from the CURRENT weights (callers remove the shunt first)."""
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    x=torch.full((n,1),start,dtype=torch.long,device=model.embed.weight.device)
    for _ in range(T-1):
        hd=model.heads[d]
        lg=hd(priv(model,x,model.features(x),d))[:,-1].float()
        lg=lg/temp; x=torch.cat([x,torch.multinomial(F.softmax(lg,-1),1,generator=gen)],1)
    model.feat_scale.copy_(fs); model.train(was); return x
