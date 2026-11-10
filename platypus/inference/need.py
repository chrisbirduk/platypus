"""Store-off helpers, feature-need signals, sleep-only feature learning, private-into-shared consolidation."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from platypus.private import (
    _HCAP, _PUOFF, _XDG,
)
from platypus.slots import (
    _ngp_step_0,
)
from platypus.layers import (
    _CARRY,
)
from platypus.capacity import (
    _lab,
)
from platypus.conduit import (
    _carry_load, _carry_save, _uh_load, _uh_save,
)
from platypus.inference.store import (
    _INFM, _inf_sync, _ng_zero, bpc,
)
from platypus.inference.sentences import (
    _inf_store,
)


# ---- DEBUG (inf_debug_sync): validate the sparse n-gram gradients before every n-gram optimiser step ------------------------------------
def ngp_step(model,cfg,lr):
    if int(getattr(cfg,'inf_debug_sync',0)) and torch.cuda.is_available():
        for on_ in ('_ngpi_opt','_ngp_opt','_ngo_opt'):
            O_=model.__dict__.get(on_)
            if O_ is None: continue
            for g0_ in O_.param_groups:
                for p_ in g0_['params']:
                    g_=p_.grad
                    if g_ is None or not g_.is_sparse: continue
                    torch.cuda.synchronize(); ix_=g_._indices(); bad_=int(((ix_<0)|(ix_>=p_.shape[0])).sum()) if ix_.numel() else 0
                    if bad_ or not torch.isfinite(g_._values()).all():
                        print(f"  [inference debug] BAD SPARSE GRAD in {on_}: param {tuple(p_.shape)}, {bad_} of {ix_.numel()} indices out of range (min {int(ix_.min()) if ix_.numel() else 0}, max {int(ix_.max()) if ix_.numel() else 0}), values finite {bool(torch.isfinite(g_._values()).all())}",flush=True)
        _inf_sync(cfg,'n-gram gradients validated')
    return _ngp_step_0(model,cfg,lr)

# ---- cg20 SLEEP-ONLY FEATURE LEARNING (inf_sofl): no gradient learning while awake (the visit's learning rate is 0; fast learners only:
#      in-context memory, closed-form head refits, the fast store); at the visit's end the SLEEP runs inf_sofl_frac x steps_per_task
#      backprop steps on the buffered (real or verified) windows, interleaved with old domains' windows under KL to the pre-sleep model.
#      FEATURE-NEED SIGNALS, computed forward-only before the sleep, are logged with the sleep's measured BENEFIT (held-out BPC gain).
def _inf_fwd(model,X_,d_):
    sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=d_
    try: return model(X_[:,:-1],d_).float()
    finally: _carry_load(sv4_); _uh_load(model,suh4_); _XDG['cur']=sc_
def _inf_bpc_off(model,vals,d_,s_,n_=4):
    sv_=_INFM.get('force','__none__'); _INFM['force']=False
    try: return float(bpc(model,vals[d_],s_,n=n_))
    finally:
        if sv_=='__none__': _INFM.pop('force',None)
        else: _INFM['force']=sv_
@torch.no_grad()
def _inf_need(model,cfg,buf_,S_slot,vals,t,ts):
    """forward-only FEATURE-NEED signals on the buffered windows: (A) a linear softmax readout of the pre-head features fitted by the delta rule
    (its held-out bits) vs the same readout on a random nonlinear expansion (gap = structure the features miss) and vs the model's own head;
    (B) delta-rule KEY COLLISIONS in the last block (near-identical keys, different next bytes); (C) FAST-SYSTEM LOAD (store-off minus store-on
    held-out BPC); (D) REPEAT vs NOVEL (bits on the buffered windows minus held-out bits)"""
    dev_=model.embed.weight.device; rs_=np.random.RandomState(len(buf_)+7); was_=model.training; model.eval()
    sel_=[buf_[i_] for i_ in rs_.choice(len(buf_),min(16,len(buf_)),replace=False)]; Zs_=[];Ys_=[];Ls_=[];Ks_=[]
    blk_=model.blocks[-1]
    for w_,d0_ in sel_:
        X_=torch.tensor([list(w_)],dtype=torch.long,device=dev_); sh_=_HCAP['on']; _HCAP['on']=True; kv9_={}
        hk9_=blk_.Wk.register_forward_hook(lambda m_,i_,o_: kv9_.__setitem__(id(blk_),(o_.detach(),)))   # the last block's keys (9 Nov: was an undefined _DQ capture)
        try: lg_=_inf_fwd(model,X_,int(S_slot(d0_))); z_=_HCAP['h'].float()
        finally: _HCAP['on']=sh_; hk9_.remove()
        y_=X_[0,1:]; ok_=y_>0; Zs_.append(z_[0][ok_]); Ys_.append(y_[ok_]); Ls_.append(F.log_softmax(lg_[0][ok_],-1).gather(-1,y_[ok_][:,None]).squeeze(-1))
        kv_=kv9_.get(id(blk_))
        if kv_ is not None: Ks_.append((kv_[0].reshape(-1,kv_[0].shape[-1])[:y_.numel()][ok_],y_[ok_]))
    model.train(was_); Z_=torch.cat(Zs_); Y_=torch.cat(Ys_); N_=Z_.shape[0]; out_={}
    if N_>200:
        pm_=torch.randperm(N_,generator=torch.Generator().manual_seed(3)).to(dev_)[:4096]; Z_=Z_[pm_]; Y_=Y_[pm_]; Lh_=torch.cat(Ls_)[pm_]; n1_=int(0.75*Z_.shape[0])
        mu_=Z_[:n1_].mean(0); sd_=Z_[:n1_].std(0)+1e-6; Zn_=(Z_-mu_)/sd_
        def fit(F_):                                                            # softmax readout by the (batch) delta rule: W += lr F^T (onehot - p) / n
            W_=torch.zeros(F_.shape[1],256,device=dev_); b_=torch.zeros(256,device=dev_); oh_=F.one_hot(Y_[:n1_],256).float()
            for _ in range(150):
                P_=F.softmax(F_[:n1_]@W_+b_,-1); W_+=0.5*F_[:n1_].T@(oh_-P_)/n1_; b_+=0.5*(oh_-P_).mean(0)
            return float(F.cross_entropy(F_[n1_:]@W_+b_,Y_[n1_:]))/math.log(2)
        g_=torch.Generator().manual_seed(11); Rr_=(torch.randn(Zn_.shape[1],4*Zn_.shape[1],generator=g_)/math.sqrt(Zn_.shape[1])).to(dev_)
        out_['lin']=fit(Zn_); out_['rf']=fit(torch.relu(Zn_@Rr_)/math.sqrt(4.0)); out_['head']=float(-Lh_[n1_:].mean())/math.log(2)
        out_['A_gap']=out_['lin']-out_['rf']
    if Ks_:
        K_=F.normalize(torch.cat([k_ for k_,_ in Ks_]).float(),dim=-1); Yk_=torch.cat([y_ for _,y_ in Ks_])
        if K_.shape[0]>2000: ix_=torch.randperm(K_.shape[0],generator=torch.Generator().manual_seed(5))[:2000].to(dev_); K_=K_[ix_]; Yk_=Yk_[ix_]
        C_=K_@K_.T; C_.fill_diagonal_(0); hi_=C_>0.95; npair_=int(hi_.sum())
        out_['B_coll']=float((hi_&(Yk_[:,None]!=Yk_[None,:])).sum())/max(1,npair_) if npair_ else 0.0; out_['B_pairs']=npair_/max(1,K_.shape[0])
    sv_=_INFM.get('force','__none__'); _INFM['force']=True
    try: on_=float(bpc(model,vals[t],ts,n=4))
    finally:
        if sv_=='__none__': _INFM.pop('force',None)
        else: _INFM['force']=sv_
    off_=_inf_bpc_off(model,vals,t,ts); out_['C_load']=off_-on_; out_['D_rep']=(float(-torch.cat(Ls_).mean())/math.log(2))-off_
    return out_,off_
def _inf_sofl(model,cfg,t,ts,vals,olds_,S_slot):
    S_=_inf_store(model); ev_=int(getattr(cfg,'inf_sofl_every',1))
    if ev_>1 and (int(model.__dict__.get('_visit',0))+1)%ev_!=0: return     # sleep only every ev_ visits: the buffer keeps filling meanwhile
    buf_=S_.get('sbuf') or []; S_['sbuf']=[]; S_['sbn']=0
    if len(buf_)<4: return
    dev_=model.embed.weight.device; rs_=np.random.RandomState(int(cfg.seed)+5151+int(model.__dict__.get('_visit',0)))
    need_,b0_=_inf_need(model,cfg,buf_,S_slot,vals,t,ts); _inf_sync(cfg,'feature-need signals')
    o0_=float(np.mean([_inf_bpc_off(model,vals,d_,s_) for d_,s_ in olds_])) if olds_ else float('nan')
    st_=max(1,int(round(int(cfg.steps_per_task)*float(getattr(cfg,'inf_sofl_frac',1.0))))); R_=int(getattr(cfg,'inf_sofl_rows',16)); fa_=float(getattr(cfg,'inf_sofl_old',0.25))
    PA_=[p_ for p_ in model.parameters() if p_.requires_grad]; TP_=[p_.detach().clone() for p_ in PA_]
    op_=model.__dict__.get('_sofl_opt')
    if op_ is None: op_=model.__dict__['_sofl_opt']=torch.optim.Adam(PA_,lr=float(cfg.lr)*float(getattr(cfg,'inf_sofl_lr',1.0)))
    old_=[d_ for d_ in (S_.get('res') or {}) if d_!=int(t) and S_['res'][d_]]
    def teacher(X_,d_):
        sv_=[p_.data for p_ in PA_]
        for p_,q_ in zip(PA_,TP_): p_.data=q_
        try:
            with torch.no_grad(): return F.log_softmax(_inf_fwd(model,X_,d_),-1)
        finally:
            for p_,v_ in zip(PA_,sv_): p_.data=v_
    was_=model.training; model.train(); ls_=[]; L0_=len(buf_[0][0])
    with torch.enable_grad():
        for k_ in range(st_):
            grp_={}
            for _ in range(R_):
                if old_ and rs_.rand()<fa_: d0_=old_[rs_.randint(len(old_))]; w0_=S_['res'][d0_][rs_.randint(len(S_['res'][d0_]))]; an_=True
                else: w0_,d0_=buf_[rs_.randint(len(buf_))]; an_=False
                w0_=(bytes(w0_)+b' '*L0_)[:L0_]; grp_.setdefault(int(S_slot(d0_)),[]).append((torch.tensor(list(w0_),dtype=torch.long),an_))
            loss_=0.0
            for d_,rows_ in grp_.items():
                X_=torch.stack([r_[0] for r_ in rows_]).to(dev_); L_=F.log_softmax(_inf_fwd(model,X_,d_),-1); nz_=(X_[:,1:]>0).float()
                an_=torch.tensor([r_[1] for r_ in rows_],device=dev_)[:,None]
                ce_=-L_.gather(-1,X_[:,1:,None]).squeeze(-1)
                if bool(an_.any()): T_=teacher(X_,d_); kl_=(T_.exp()*(T_-L_)).sum(-1)
                else: kl_=torch.zeros_like(ce_)
                loss_=loss_+(torch.where(an_,kl_,ce_)*nz_).sum()/nz_.sum().clamp(min=1)*len(rows_)/R_
            op_.zero_grad(set_to_none=True); loss_.backward(); op_.step(); ls_.append(float(loss_.detach())); _ng_zero(model)
    for p_ in PA_: p_.grad=None
    model.train(was_); _inf_sync(cfg,'sleep feature learning')
    b1_=_inf_bpc_off(model,vals,t,ts); o1_=float(np.mean([_inf_bpc_off(model,vals,d_,s_) for d_,s_ in olds_])) if olds_ else float('nan')
    ben_=b0_-b1_; model.__dict__.setdefault('_need_log',[]).append((need_,ben_,o1_-o0_ if olds_ else 0.0))
    print(f"  [inference sleep feature learning] {st_} backprop steps in SLEEP on {len(buf_)} buffered windows ({int(100*fa_)}% rows from {len(old_)} old domains, KL to the pre-sleep model) | loss {ls_[0]:.3f} -> {np.mean(ls_[-5:]):.3f} | "
          f"held-out BPC (store off) AWAKE {b0_:.3f} -> AFTER SLEEP {b1_:.3f} (benefit {ben_:+.3f}), old domains {o0_:.3f} -> {o1_:.3f} | FEATURE NEED: "
          +" ".join(f"{k0_} {v0_:+.3f}" if isinstance(v0_,float) else f"{k0_} {v0_}" for k0_,v0_ in need_.items()),flush=True)
def inf_need_report(model):
    """end of run: which forward-only signal predicts how much a sleep's backprop helped (Pearson r over the sleeps)"""
    lg_=model.__dict__.get('_need_log') or []
    if len(lg_)<4: return
    ben_=np.array([b_ for _,b_,_ in lg_]); ks_=sorted(set(k_ for n_,_,_ in lg_ for k_ in n_)); seg_=[]
    for k_ in ks_:
        x_=np.array([n_.get(k_,np.nan) for n_,_,_ in lg_],float); ok_=np.isfinite(x_)
        if ok_.sum()>=4 and np.std(x_[ok_])>0 and np.std(ben_[ok_])>0: seg_.append(f"{k_} r {np.corrcoef(x_[ok_],ben_[ok_])[0,1]:+.2f}")
    print(f"  [inference sleep feature learning | FEATURE-NEED CALIBRATION] {len(lg_)} sleeps, benefit (held-out BPC gain) mean {ben_.mean():+.3f} | correlation of each forward-only signal with the benefit: "+" | ".join(seg_),flush=True)

def pu_consolidate(model,cfg,t,ts,vals,olds_,S_):
    """cg19 (1)+(3): PRIVATE-INTO-SHARED CONSOLIDATION on the visit's VERIFIED windows, with trace-level retirement and a capacity cap.
    Teacher = the model frozen at the visit end (private units ON). Student = the live model with slot ts's private units OFF on the visit's
    windows; old domains' kept windows (private units on) are taught to keep the frozen output. Only the blocks' SHARED weights learn."""
    RV_=model.__dict__.get('_nres'); BK_=model.__dict__.get('_nres_bank',{})
    if not RV_ or RV_.get('td')!=int(t) or not RV_.get('e') or not hasattr(model.blocks[0],'pu_out'):
        print(f"  [consolidation] {_lab(cfg,t)}: skipped (no verified windows or no private units)"); return
    own_=[e_ for e_ in RV_['e']]; anc_=[e_ for d_,l_ in BK_.items() if d_!=int(t) for e_ in l_]
    nb_=int(cfg.pu_cons_eval); dev_=model.embed.weight.device
    def _ev(off):
        _PUOFF['d']=(ts if off else None)
        try: b_=float(bpc(model,vals[t],ts,n=nb_,skip=int(getattr(cfg,'brake_eval_skip',0))))   # FIX (review): brake off the reported batches
        finally: _PUOFF['d']=None
        return b_
    def _evo():
        if not olds_: return float('nan')
        return float(np.mean([bpc(model,vals[d_],s_,n=nb_,skip=int(getattr(cfg,'brake_eval_skip',0))) for d_,s_ in olds_]))
    was_=model.training
    on0_=_ev(False); off0_=_ev(True); old0_=_evo(); model.train()
    PA_=[p_ for n_,p_ in model.named_parameters() if p_.requires_grad]
    TP_=[p_.detach().clone() for p_ in PA_]                                       # the frozen teacher
    SH_=[p_ for n_,p_ in model.named_parameters() if p_.requires_grad and n_.startswith('blocks.') and '.pu_' not in n_]
    PU_=[getattr(b_,k_) for b_ in model.blocks for k_ in ('pu_in','pu_b','pu_out')]   # only slot ts's slice is updated (others' grads zeroed)
    SH_=SH_+PU_
    op_=torch.optim.Adam(SH_,lr=float(cfg.lr)*float(cfg.pu_cons_lr))
    rs_=model.__dict__.setdefault('_pucons_rs',np.random.RandomState(int(cfg.seed)+6161))
    R_=int(cfg.pu_cons_rows); ka_=min(len(anc_),int(round(R_*float(cfg.pu_cons_anchor)))) if anc_ else 0; ko_=R_-ka_
    kl_o=[]; kl_a=[]
    def _fwd(X_,d_,off):
        sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=d_; _PUOFF['d']=(d_ if off else None)
        try: return F.log_softmax(model(X_[:,:-1],d_).float()/cfg.shunt_T,-1)
        finally: _carry_load(sv4_); _uh_load(model,suh4_); _XDG['cur']=sc_; _PUOFF['d']=None
    def _teach(X_,d_):
        with torch.no_grad():
            sv_=[p_.data for p_ in PA_]
            for p_,q_ in zip(PA_,TP_): p_.data=q_
            try: return _fwd(X_,d_,False)
            finally:
                for p_,v_ in zip(PA_,sv_): p_.data=v_
    _cs_snap_=True if float(getattr(cfg,'pu_cons_brake',0.0))>0 else None; _cs_rv_=list(RV_['e'])   # brake: TP_ (the frozen copy taken above) is the snapshot
    for e_ in own_: e_.pop('ck0',None); e_.pop('ck',None)
    for st_ in range(int(cfg.pu_cons_steps)):
        live_=[e_ for e_ in own_ if not e_.get('cret')]
        if not live_: break
        so_=[live_[i_] for i_ in rs_.choice(len(live_),min(len(live_),ko_),replace=False)]
        sa_=[anc_[i_] for i_ in rs_.choice(len(anc_),ka_,replace=False)] if ka_>0 else []
        loss_=0.0
        Xo_=torch.stack([e_['x'] for e_ in so_]).to(dev_); To_=_teach(Xo_,ts); Lo_=_fwd(Xo_,ts,True)
        ko2_=(To_.exp()*(To_-Lo_)).sum(-1).mean(-1)                               # per-row KL, private units OFF
        Ln_=_fwd(Xo_,ts,False); kn2_=(To_.exp()*(To_-Ln_)).sum(-1).mean(-1)   # and WITH them (no double counting)
        loss_=loss_+(ko2_.mean()+kn2_.mean())*len(so_)/R_
        with torch.no_grad():
            for e_,k_ in zip(so_,ko2_.tolist()):
                e_.setdefault('ck0',k_); e_['ck']=k_
                if float(cfg.pu_cons_retire)>=0 and k_<=float(cfg.pu_cons_retire)*e_['ck0'] and st_>0: e_['cret']=True
        kl_o.append(float(ko2_.mean()))
        for d_ in sorted(set(e_['d'] for e_ in sa_)):                               # ANCHORS: old domains keep the frozen output
            g_=[e_ for e_ in sa_ if e_['d']==d_]; Xa_=torch.stack([e_['x'] for e_ in g_]).to(dev_)
            Ta_=_teach(Xa_,d_); La_=_fwd(Xa_,d_,False); ka2_=(Ta_.exp()*(Ta_-La_)).sum(-1).mean(-1)
            loss_=loss_+ka2_.sum()/R_; kl_a.append(float(ka2_.mean()))
        op_.zero_grad(set_to_none=True); loss_.backward()
        with torch.no_grad():
            for q_ in PU_:                                                        # private units: only this slot's slice learns
                if q_.grad is not None: k_=q_.grad[ts].clone(); q_.grad.zero_(); q_.grad[ts]=k_
        op_.step()
        for p_ in PA_:
            if p_.grad is not None: p_.grad=None
        _ng_zero(model)                                                         # FIX: the n-gram tables' sparse grads accumulated here (GPU crash)
    del op_
    nret_=sum(1 for e_ in own_ if e_.get('cret'))
    RV_['e']=[e_ for e_ in RV_['e'] if not e_.get('cret')]                       # (3) retired windows are not kept
    cap_=int(cfg.pu_cons_cap); nev_=0
    if cap_>0 and len(RV_['e'])>cap_:                                             # (3) capacity cap: weakest-first eviction
        RV_['e']=sorted(RV_['e'],key=lambda e_:-float(e_.get('ck',1e9)))
        nev_=len(RV_['e'])-cap_; RV_['e']=RV_['e'][:cap_]
    on1_=_ev(False); off1_=_ev(True); old1_=_evo(); _cs_rb_=''
    _cs_b_=float(getattr(cfg,'pu_cons_brake',0.0))
    if _cs_snap_ is not None and (old1_-old0_>_cs_b_ or on1_-on0_>_cs_b_):     # BRAKE: the phase hurt old domains or the domain itself -> undo it
        with torch.no_grad():
            for q_,s0_ in zip(PA_,TP_): q_.data.copy_(s0_)
        RV_['e']=_cs_rv_
        for e_ in RV_['e']: e_.pop('cret',None)
        _cs_rb_=f" | ROLLED BACK (old domains {old1_-old0_:+.4f}, own {on1_-on0_:+.4f}; brake {_cs_b_})"; on1_,off1_,old1_=on0_,off0_,old0_; nret_=0; nev_=0
    _cs_snap_=None
    model.train(was_)
    rec_=(off0_-off1_)/max(1e-6,off0_-on0_)
    CL_=model.__dict__.setdefault('_pucons_log',[]); CL_.append((int(t),on0_,on1_,off0_,off1_,rec_,old0_,old1_,nret_,nev_,len(RV_['e'])))
    print(f"  [consolidation] {_lab(cfg,t)}: {len(kl_o)} steps on {len(own_)} verified windows ({ko_} rows/step) + {ka_} anchor rows/step from {len(set(e_['d'] for e_ in anc_))} old domains"
          f" | held-out WITH its private units {on0_:.3f} -> {on1_:.3f} | WITHOUT {off0_:.3f} -> {off1_:.3f} (gap recovered {100*rec_:.0f}%)"
          f" | old domains mean {old0_:.3f} -> {old1_:.3f} ({old1_-old0_:+.4f})"
          f" | KL own (private off) {kl_o[0] if kl_o else float('nan'):.4f} -> {kl_o[-1] if kl_o else float('nan'):.4f}, anchors {np.mean(kl_a) if kl_a else float('nan'):.4f}"
          f" | retired {nret_} (absorbed), evicted {nev_} (cap {cap_ or 'none'}), kept {len(RV_['e'])}"
          +((""))+_cs_rb_)
