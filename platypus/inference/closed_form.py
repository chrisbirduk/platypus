"""Closed-form edits of the last block MLP: layer states, drift, application."""
from platypus.common import _fail, _CONC_CF
import numpy as np
import torch
import torch.nn.functional as F
from platypus.private import (
    _XDG,
)
from platypus.layers import (
    _CARRY,
)
from platypus.ngrams import (
    _COMPILED,
)
from platypus.conduit import (
    _carry_load, _carry_save,
)
from platypus.inference.store import (
    _INFM, bpc,
)
from platypus.inference.evaluate import (
    _inf_lps,
)

def _ws_eager(model):
    """take the compiled forward out of the way (forward hooks on submodules must fire); returns a restore function"""
    cmp_=_COMPILED.pop(id(model),None) if '_COMPILED' in globals() else None
    def back_():
        if cmp_ is not None: _COMPILED[id(model)]=cmp_
    return back_
def _ws_batch(model,txts_):
    L0_=int(cfg_seq_(model)); dev_=model.embed.weight.device; bs_=[t_[:L0_] for t_ in txts_]; L1_=max(8,max(len(b_) for b_ in bs_))
    X_=torch.full((len(bs_),L1_),32,dtype=torch.long,device=dev_); M_=torch.zeros(len(bs_),L1_,device=dev_)
    for j_,b_ in enumerate(bs_): X_[j_,:len(b_)]=torch.tensor(list(b_),device=dev_); M_[j_,:len(b_)]=1.0
    return X_,M_
def _ws_run(model,X_,slot_,grad_=False):
    """logits of X_[:, :-1] through slot_, store off, no carried context"""
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot_; fz_=_INFM.get('force'); _INFM['force']=False; was_=model.training; model.eval()
    try:
        with torch.set_grad_enabled(grad_): return model(X_[:,:-1],slot_).float()
    finally:
        _carry_load(sv4_); _XDG['cur']=sc_; model.train(was_)
        if fz_ is None: _INFM.pop('force',None)
        else: _INFM['force']=fz_
def _ws_bpc(model,cfg,vals,t,ts,olds_):
    nb_=int(cfg.pu_cons_eval) or 4; sk_=int(getattr(cfg,'brake_eval_skip',0))
    b_=bpc(model,vals[t],ts,n=nb_,skip=sk_); o_=float(np.mean([bpc(model,vals[d_],s_,n=nb_,skip=sk_) for d_,s_ in olds_])) if olds_ else float('nan')
    return b_,o_
def _ws_twohop(model,cfg,L_,Cp_,S_slot,rs_,pt_="So {A} is connected to ",dec_=False):
    ok_=[]
    for c_ in L_:
        o_=[x_ for x_ in Cp_ if x_!=c_['C']]; rs_.shuffle(o_)
        if len(o_)<4: continue
        lp_=_inf_lps(model,cfg,pt_.format(A=c_['A']),[c_['C']]+o_[:9],int(S_slot(c_['d'])),False,dec_=dec_); ok_.append(float(int(np.argmax(lp_))==0))
    return float(np.mean(ok_)) if ok_ else float('nan')
_CF_VARY=("{A} has a connection with ","One of the people connected to {A} is ")
@torch.no_grad()
def _cf_layer_states(model,c_,S_slot):
    """mean hidden state over the answer positions at the output of every block, for 'So A is connected to C.' (eager, store off)"""
    pr_=f"So {c_['A']} is connected to ".encode('latin-1','replace'); tg_=(c_['C']+'.').encode('latin-1','replace'); X_,M_=_ws_batch(model,[pr_+tg_])
    T1_=X_.shape[1]-1; ps_=torch.arange(len(pr_)-1,min(T1_,len(pr_)+len(tg_)-1),device=X_.device)
    if ps_.numel()==0: return None
    H_={}; hs_=[]
    for i_,b_ in enumerate(model.blocks):
        def hk_(m_,inp_,out_,i_=i_): H_[i_]=(out_[0] if isinstance(out_,tuple) else out_)
        hs_.append(b_.register_forward_hook(hk_))
    try: _ws_run(model,X_,int(S_slot(c_['d'])))
    finally:
        for h_ in hs_: h_.remove()
    return [H_[i_][0][ps_].float().mean(0).cpu() if i_ in H_ and H_[i_].dim()==3 else None for i_ in range(len(model.blocks))]
def _cf_layer_drift(model,cfg,lv_,S_slot,vis_):
    """STEP 0 (readout, inf_cf_layerdrift 1): how far each block's representation of edited conclusions has moved since they were fitted"""
    D_=[c_ for c_ in lv_ if c_.get('_cfL') is not None]
    if not D_: return
    D_=sorted(D_,key=lambda c_: int(c_['_cfLv'])); ix_=np.unique(np.linspace(0,len(D_)-1,min(24,len(D_))).astype(int)); G_={}
    for i_ in ix_:
        c_=D_[int(i_)]; L1_=_cf_layer_states(model,c_,S_slot)
        if L1_ is None: continue
        cs_=[float(F.cosine_similarity(a_[None],b_[None]).item()) if (a_ is not None and b_ is not None) else float('nan') for a_,b_ in zip(c_['_cfL'],L1_)]
        ag_=vis_-int(c_['_cfLv']); g_='1' if ag_<=1 else '2-3' if ag_<=3 else '4+'; G_.setdefault(g_,[]).append(cs_)
    if G_: print(f"  [inference LAYER DRIFT | visit {vis_}] cosine of each block's state (answer positions) now vs when fitted, blocks 0..{len(model.blocks)-1}: "
                 +" | ".join(f"fitted {g_} visit(s) ago (n {len(L_)}): "+" ".join(f"{np.nanmean([x_[j_] for x_ in L_]):.2f}" for j_ in range(len(L_[0])))
                             for g_,L_ in sorted(G_.items(),key=lambda kv_: {'1':0,'2-3':1,'4+':2}[kv_[0]])),flush=True)
@torch.no_grad()
def _cf_diag(model,cfg,lv_,mo_,cap_,S_slot,vis_):
    """CF DECAY DIAGNOSTIC (readout only, inf_cf_diag 1): for each edited conclusion, the output change its own edit made on its own key
    (its signal, 1.00 when written) and the change along that signal since, split exactly into ordinary training (the rest of the matrix),
    later edits (the accumulated edits) and key drift (the layers below now give another key): k1 W1 - k0 W0 = (k1-k0) W1 + k0 (B1-B0) + k0 (E1-E0)"""
    D_=[c_ for c_ in lv_ if c_.get('_cfd') is not None]
    if not D_: return
    W_=mo_.weight.detach().float(); E0_=model.__dict__.get('_cf_E'); E_=E0_.to(W_.device).float() if E0_ is not None else torch.zeros_like(W_); B_=W_-E_
    D_=sorted(D_,key=lambda c_: int(c_['_cfd']['v'])); ix_=np.unique(np.linspace(0,len(D_)-1,min(48,len(D_))).astype(int)); G_={}
    for i_ in ix_:
        c_=D_[int(i_)]; d_=c_['_cfd']; k0_=d_['k'].to(W_.device).float(); se_=d_['se'].to(W_.device).float(); s2_=float((se_*se_).sum())
        if s2_<=1e-12: continue
        pr_=f"So {c_['A']} is connected to ".encode('latin-1','replace'); tg_=(c_['C']+'.').encode('latin-1','replace'); X_,M_=_ws_batch(model,[pr_+tg_])
        T1_=X_.shape[1]-1; ps_=torch.arange(len(pr_)-1,min(T1_,len(pr_)+len(tg_)-1),device=X_.device)
        if ps_.numel()!=k0_.shape[0]: continue
        cap_['on']=True; cap_['k']=[]; _ws_run(model,X_,int(S_slot(c_['d']))); cap_['on']=False; k1_=cap_['k'][-1][0][ps_].float()
        pj_=lambda Y_: float((Y_*se_).sum())/s2_
        tr_=pj_(k0_@B_.T-d_['Bk'].to(W_.device).float()); ed_=pj_(k0_@E_.T-d_['Ek'].to(W_.device).float()); ky_=pj_((k1_-k0_)@W_.T)
        cs_=float(F.cosine_similarity(k1_,k0_,dim=-1).mean()); ag_=vis_-int(d_['v']); g_='1' if ag_<=1 else '2-3' if ag_<=3 else '4+'
        G_.setdefault(g_,[]).append((tr_,ed_,ky_,cs_,1.0+tr_+ed_+ky_))
    if not G_: return
    mm_=lambda L_,j_: float(np.mean([x_[j_] for x_ in L_]))
    print(f"  [inference CF DECAY DIAGNOSTIC | visit {vis_}] each edited conclusion's own edit signal on its own key (1.00 when written), change along it since, by cause: "
          +" | ".join(f"edited {g_} visit(s) ago (n {len(L_)}): ordinary training {mm_(L_,0):+.2f}, later edits {mm_(L_,1):+.2f}, key drift {mm_(L_,2):+.2f} (key cosine {mm_(L_,3):.2f}) -> left {mm_(L_,4):.2f}"
                     for g_,L_ in sorted(G_.items(),key=lambda kv_: {'1':0,'2-3':1,'4+':2}[kv_[0]])),flush=True)
def _cf_apply(model,cfg,S_,vals,olds_,t,ts,S_slot):
    """cg26l CLOSED-FORM CONSOLIDATION APPLIED (see _ws_closed_form for the edit): conclusions not yet edited (newest first, up to inf_cf_n)
    are written in one least-squares step dW = R^T K (K^T K + lam n C0)^-1 into block inf_cf_layer's MLP output; the edit is KEPT at scale
    inf_cf_scale if visited and old-domain BPC rise by at most inf_cf_brake, else tried at half and a quarter, else rolled back"""
    allc_=S_.get('conc',[]); lv_=[c_ for c_ in allc_ if not c_.get('freed') and isinstance(c_.get('C'),str) and isinstance(c_.get('A'),str)]
    vis_=int(model.__dict__.get('_visit',0)); Cp_=[c_['C'] for c_ in lv_]; rs_=np.random.RandomState(8282+vis_); ne_=int(getattr(cfg,'inf_cf_n',32))
    tw_=lambda L_,pt_="So {A} is connected to ": _ws_twohop(model,cfg,L_,Cp_,S_slot,np.random.RandomState(3),pt_)
    tv_=lambda L_: float(np.nanmean([tw_(L_,p_) for p_ in _CF_VARY])) if L_ else float('nan')
    ag_={}
    for c_ in lv_:
        if c_.get('cfe') is not None: a9_=vis_-int(c_['cfe']); ag_.setdefault('1' if a9_<=1 else '2-3' if a9_<=3 else '4+',[]).append(c_)
    pers_=" | ".join(f"edited {k_} visit(s) ago (n {min(24,len(L_))} of {len(L_)}): fitted wording {tw_(L_[:24]):.2f}, other wordings {tv_(L_[:24]):.2f}" for k_,L_ in sorted(ag_.items())) or "no earlier edits"
    cum_=S_.setdefault('_cfcum',[0.0,0.0,0,0])
    rd_=float(getattr(cfg,'inf_cf_decay',0.0)); Eh_=model.__dict__.get('_cf_E'); dinfo_=''
    if rd_>0 and Eh_ is not None:                                               # cg26n EDIT DECAY: a share of the accumulated edits is taken back every sleep
        mo9_=model.blocks[int(getattr(cfg,'inf_cf_layer',-1))].mlp_out
        with torch.no_grad(): mo9_.weight.sub_((rd_*Eh_).to(mo9_.weight.dtype)); Eh_.mul_(1.0-rd_)
        dinfo_=f" | DECAY {rd_:g}: accumulated edits |E| {float(Eh_.norm()):.2f} after taking back {rd_:.0%}"
    un_=sorted([c_ for c_ in lv_ if c_.get('cfe') is None],key=lambda c_: -int(c_.get('v',0)))
    mo9n_=int(getattr(cfg,'inf_cf_min_others',0)) if int(getattr(cfg,'inf_cf_center',0))>0 else 0
    ne9_=min(ne_,max(0,len(lv_)-mo9n_)) if mo9n_>0 else ne_                    # cg26n: keep at least inf_cf_min_others live conclusions out of the edit to centre the keys on
    rf9n_=int(getattr(cfg,'inf_cf_refresh',0)); RF_=[]; rinfo_=''              # cg26o REFRESH: re-fit a few earlier edited conclusions in each new edit so later edits refresh rather than overwrite them
    ro9_=0
    if rf9n_>0:                                                                 # the refreshed conclusions take their share of the edit batch (never of the conclusions kept back for centring)
        ned9_=sum(1 for c_ in lv_ if c_.get('cfe') is not None); ro9_=max(0,min(rf9n_,ned9_,ne9_-8))
    E_=un_[:ne9_-ro9_]; N_=un_[ne9_-ro9_:ne9_-ro9_+24]
    EN9_=None; ED9_=[]
    if int(getattr(cfg,'inf_cf_resolve',0)):                                    # RE-SOLVE: at every sleep EVERY edited conclusion (live or released) is re-fitted with its CURRENT key,
        ED9_=[c_ for c_ in lv_ if c_.get('cfe') is not None]                    #   together with up to inf_cf_n new ones (none if fewer than 8), so key drift never builds up
        EN9_=un_[:ne9_] if len(un_[:ne9_])>=8 else []; N_=un_[len(EN9_):len(EN9_)+24]; E_=EN9_+ED9_; ro9_=0
    bud_=float(getattr(cfg,'inf_cf_budget',0.0))
    if bud_>0 and cum_[1]>=bud_: print(f"  [inference CLOSED-FORM CONSOLIDATION APPLIED] skipped: cumulative old-domain cost {cum_[1]:+.4f} of {cum_[2]} kept edits reached the budget {bud_:g} | PERSISTENCE: {pers_}",flush=True); return
    kb9_=(f"; {len(lv_)} live, {mo9n_} kept back to centre the keys on" if mo9n_>0 else "")
    if len(E_)<8: print(f"  [inference CLOSED-FORM CONSOLIDATION APPLIED] skipped: {len(E_)} conclusions to edit (need 8{kb9_}) | PERSISTENCE: {pers_}",flush=True); return
    no9_=len(lv_)-len(E_ if EN9_ is None else EN9_)
    if mo9n_>0 and no9_<mo9n_: print(f"  [inference CLOSED-FORM CONSOLIDATION APPLIED] skipped: {no9_} other conclusions to centre the keys on (need {mo9n_}) | PERSISTENCE: {pers_}",flush=True); return   # cg26n: too few keys to find the frame direction (GPU cg26m visits 3-4 rolled back at +5.9 / +7.9 BPC)
    if rf9n_>0:
        if ro9_>0:
            ed9_=sorted([c_ for c_ in lv_ if c_.get('cfe') is not None],key=lambda c_: int(c_.get('cfr',c_['cfe'])))[:3*ro9_]
            if ed9_:
                rm9_=str(getattr(cfg,'inf_cf_refresh_mode','weak'))
                sc9_=[(_ws_twohop(model,cfg,[c_],Cp_,S_slot,np.random.RandomState(3),dec_=True),i_) for i_,c_ in enumerate(ed9_)]   # (refresh order: a decision)
                if rm9_=='weak': sc9_.sort()                                 # 'weak': the weakest of the least recently fitted; 'old': simply the least recently fitted
                RF_=[ed9_[i_] for _,i_ in sc9_[:ro9_]]
                rinfo_=f" | REFRESH ({rm9_}): {len(RF_)} earlier edited conclusions re-fitted ({'weakest of the '+str(len(ed9_))+' ' if rm9_=='weak' else ''}least recently fitted; weights-only before {sum(v_ for v_,_ in sc9_[:ro9_])/max(1,len(RF_)):.2f})"
        rr9_=int(getattr(cfg,'inf_rel_refresh',0))
        if rr9_>0 and ro9_>0:                                                   # HANDOVER THAT HOLDS: released conclusions (they depend on the weights alone) and restored ones
            P9_=sorted([c_ for c_ in lv_ if c_.get('cfe') is not None and (c_.get('cf_refit') or c_.get('released'))],   #   take up to inf_rel_refresh of the refresh slots
                       key=lambda c_: (not c_.get('cf_refit'),int(c_.get('cfr',c_['cfe']))))[:min(rr9_,ro9_)]
            if P9_:
                pid9_={id(x_) for x_ in P9_}; RF_=P9_+[x_ for x_ in RF_ if id(x_) not in pid9_][:ro9_-len(P9_)]
                rinfo_+=f" | HANDOVER: {len(P9_)} of the re-fitted are released or restored ({sum(1 for x_ in P9_ if x_.get('cf_refit'))} restored); weights-only before {tw_(P9_):.2f}"
        E_=E_+RF_
    if EN9_ is not None: RF_=list(ED9_); rinfo_=f" | RE-SOLVE: {len(ED9_)} edited conclusions re-fitted with their current keys, {len(EN9_)} new"
    li_=int(getattr(cfg,'inf_cf_layer',-1)); mo_=model.blocks[li_].mlp_out; back_=_ws_eager(model); W0_=mo_.weight.detach().clone(); cap_={'on':False,'k':[],'delta':None}
    def hook_(m_,inp_,out_):
        if cap_['on']: cap_['k'].append(inp_[0].detach().float())
        if cap_['delta'] is not None: return out_+cap_['delta'].to(out_.dtype)
    hk_=mo_.register_forward_hook(hook_); kept_=None
    try:
        if int(getattr(cfg,'inf_cf_diag',0)):                                   # CF DECAY DIAGNOSTIC (readout only), before this sleep's edit
            try: _cf_diag(model,cfg,lv_,mo_,cap_,S_slot,vis_)
            except Exception as e8_: _fail('inference.closed_form:170'); print(f"  [inference CF DECAY DIAGNOSTIC] unavailable ({type(e8_).__name__}: {e8_})"[:240],flush=True)
        if int(getattr(cfg,'inf_cf_layerdrift',0)):                             # STEP 0: per-block drift of edited conclusions (readout only)
            try: _cf_layer_drift(model,cfg,lv_,S_slot,vis_)
            except Exception as e8_: _fail('inference.closed_form:173'); print(f"  [inference LAYER DRIFT] unavailable ({type(e8_).__name__}: {e8_})"[:240],flush=True)
        e0_=tw_(E_); v0_=tv_(E_); n0_=tw_(N_) if N_ else float('nan'); b0_,o0_=_ws_bpc(model,cfg,vals,t,ts,olds_)
        rl_=[x_ for x_ in S_['it'] if x_.get('k')=='real' and not x_.get('gone') and len(x_['s'])>=40]; gk_=[]
        for j_ in rs_.choice(len(rl_),min(64,len(rl_)),replace=False) if rl_ else []:
            X_,M_=_ws_batch(model,[rl_[int(j_)]['s']]); cap_['on']=True; cap_['k']=[]; _ws_run(model,X_,int(S_slot(rl_[int(j_)]['d']))); cap_['on']=False; gk_.append(cap_['k'][-1][0][M_[0,:-1]>0])
        Kg_=torch.cat(gk_) if gk_ else None; Ks_=[]; Rs_=[]; kd9_=[]; lr_=float(getattr(cfg,'inf_cf_lr',0.5)); st_=int(getattr(cfg,'inf_cf_steps',25))
        nw9_=max(1,min(len(_CONC_CF),int(getattr(cfg,'inf_cf_wordings',1)))); ct9w_=float(getattr(cfg,'inf_cf_contrast',0.0)); ctk9_=int(getattr(cfg,'inf_cons_contrast_k',8))
        if ct9w_>0:                                                             # cg27a COMPETITOR CONTRAST in the edit: other live conclusions' C (never one of the same A) as negatives
            byA9_={}
            for x_ in lv_: byA9_.setdefault(x_['A'],set()).add(x_['C'])
            aC9_=sorted(set(x_['C'] for x_ in lv_)); rn9_=np.random.RandomState(int(cfg.seed)+9127+vis_); cm9_=[]
        for c_ in E_:
          for wi9_ in range(nw9_):                                              # cg27a: the edit is fitted on the keys of nw9_ wordings (_CONC_CF; the first is the original)
            pr_=_CONC_CF[wi9_].format(A=c_['A']).encode('latin-1','replace'); tg_=(c_['C']+'.').encode('latin-1','replace'); X_,M_=_ws_batch(model,[pr_+tg_])
            T1_=X_.shape[1]-1; ps_=torch.arange(len(pr_)-1,min(T1_,len(pr_)+len(tg_)-1),device=X_.device)
            if ps_.numel()==0: continue
            sl_=int(S_slot(c_['d'])); cap_['on']=True; cap_['k']=[]; _ws_run(model,X_,sl_); cap_['on']=False; K1_=cap_['k'][-1][0][ps_]
            D_=mo_.weight.shape[0]; dl_=torch.zeros(len(ps_),D_,device=X_.device,requires_grad=True); op_=torch.optim.Adam([dl_],lr=lr_*0.1); yt_=X_[0,ps_+1]
            dv9_=[]
            if ct9w_>0:                                                         # divergence byte of each negative: same prefix up to j, so the logits there are shared
                ng9_=[n9z_ for n9z_ in aC9_ if n9z_ not in byA9_.get(c_['A'],())]
                for nz7_ in ([ng9_[q9z_] for q9z_ in rn9_.choice(len(ng9_),min(ctk9_,len(ng9_)),replace=False)] if ng9_ else []):
                    nbz7_=(nz7_+'.').encode('latin-1','replace'); jz7_=next((q9z_ for q9z_ in range(min(len(nbz7_),len(tg_))) if nbz7_[q9z_]!=tg_[q9z_]),None)
                    if jz7_ is not None and jz7_<len(ps_): dv9_.append((jz7_,int(tg_[jz7_]),int(nbz7_[jz7_])))
            for s9_ in range(st_):
                full_=torch.zeros(1,T1_,D_,device=X_.device).index_put((torch.zeros_like(ps_),ps_),dl_); cap_['delta']=full_
                lg9_=_ws_run(model,X_,sl_,grad_=True)[0,ps_]; l_=F.cross_entropy(lg9_,yt_)+1e-3*(dl_**2).sum()
                if dv9_:
                    jj9_=torch.tensor([d_[0] for d_ in dv9_],device=X_.device); mg9_=lg9_[jj9_,[d_[1] for d_ in dv9_]]-lg9_[jj9_,[d_[2] for d_ in dv9_]]
                    l_=l_+ct9w_*F.softplus(-mg9_).mean()
                    if s9_ in (0,st_-1): cm9_.append((s9_==0,float((mg9_>0).float().mean())))
                op_.zero_grad(); l_.backward(); op_.step(); cap_['delta']=None
            Ks_.append(K1_); Rs_.append(dl_.detach().float())
            if wi9_==0: kd9_.append((c_,K1_))
        if ct9w_>0 and cm9_:
            print(f"  [inference CF COMPETITOR CONTRAST] x{ct9w_:g}, up to {ctk9_} other conclusions' C per edit: the answer's byte beats the competitor's at the divergence byte "
                  f"{np.mean([v_ for f_,v_ in cm9_ if f_]):.2f} before fitting, {np.mean([v_ for f_,v_ in cm9_ if not f_]):.2f} after | wordings {nw9_}",flush=True)
        cap_['delta']=None
        if not Ks_: print("  [inference CLOSED-FORM CONSOLIDATION APPLIED] no edit positions",flush=True); return
        K_=torch.cat(Ks_); R_=torch.cat(Rs_); H_=K_.shape[1]; lam_=float(getattr(cfg,'inf_cf_lam',1.0))
        C0_=(Kg_.T@Kg_)/max(1,len(Kg_)) if Kg_ is not None else torch.eye(H_,device=K_.device)
        rc_=int(getattr(cfg,'inf_cf_center',0)); cinfo_=''
        if rc_>0:                                                               # cg26m KEY CENTRING: remove the sentence frame from the keys and the edit
            Eid_={id(x_) for x_ in (E_ if EN9_ is None else EN9_)}; oth_=[x_ for x_ in lv_ if id(x_) not in Eid_][:48]; tk_=[]
            for c_ in oth_:
              for wi9_ in range(nw9_):                                          # cg27a: the frame is removed from the keys of every wording used
                pr_=_CONC_CF[wi9_].format(A=c_['A']).encode('latin-1','replace'); tg_=(c_['C']+'.').encode('latin-1','replace'); X_,M_=_ws_batch(model,[pr_+tg_])
                T1_=X_.shape[1]-1; ps_=torch.arange(len(pr_)-1,min(T1_,len(pr_)+len(tg_)-1),device=X_.device)
                if ps_.numel()==0: continue
                cap_['on']=True; cap_['k']=[]; _ws_run(model,X_,int(S_slot(c_['d']))); cap_['on']=False; tk_.append(cap_['k'][-1][0][ps_])
            if len(tk_)>=4:
                Kt_=torch.cat(tk_); Ut_,St_,_=torch.linalg.svd(Kt_.T,full_matrices=False); Ut_=Ut_[:,:rc_]
                Pm_=torch.eye(H_,device=K_.device)-Ut_@Ut_.T; Kc_=K_@Pm_; tf_=float(((K_@Ut_)**2).sum()/(K_**2).sum())
                A_=Kc_.T@Kc_+1e-2*float((K_.T@K_).diagonal().mean())*torch.eye(H_,device=K_.device)+lam_*len(K_)*(Pm_@C0_@Pm_)
                dW_=(torch.linalg.solve(A_,Kc_.T@R_).T)@Pm_; cinfo_=f" | KEYS CENTRED: top-{rc_} frame direction(s) from {len(tk_)} other conclusions removed ({tf_:.0%} of the edit keys' energy)"
            else: rc_=0
        if rc_<=0: dW_=torch.linalg.solve(K_.T@K_+lam_*len(K_)*C0_+1e-4*torch.eye(H_,device=K_.device),K_.T@R_).T
        if int(getattr(cfg,'inf_cf_null',0)):                                   # NULL-SPACE EDITS (after AlphaEdit): the new edit is solved in the null space of the keys of every
            Eid9_={id(x_) for x_ in E_}                                         #   earlier edited conclusion (live or released), so it leaves their outputs unchanged
            ko_=[x_['_cfk'] for x_ in allc_ if x_.get('_cfk') is not None and id(x_) not in Eid9_]
            if ko_:
                Pm9_=Pm_ if rc_>0 else torch.eye(H_,device=K_.device)
                Ko_=torch.cat([k_.to(K_.device).float() for k_ in ko_])@Pm9_
                U9_,S9_,_=torch.linalg.svd(Ko_.T,full_matrices=False); en9_=(S9_**2).cumsum(0)/(S9_**2).sum().clamp(min=1e-12)
                r9_=int((en9_<float(getattr(cfg,'inf_cf_null_e',0.9))).sum())+1; r9_=max(1,min(r9_,H_//2,U9_.shape[1])); U9_=U9_[:,:r9_]
                Pn_=torch.eye(H_,device=K_.device)-U9_@U9_.T; Kb9_=K_@Pm9_; Kn_=Kb9_@Pn_; kl9_=float((Kn_**2).sum()/(Kb9_**2).sum().clamp(min=1e-12))
                An_=Kn_.T@Kn_+1e-2*float((K_.T@K_).diagonal().mean())*torch.eye(H_,device=K_.device)+lam_*len(K_)*(Pn_@Pm9_@C0_@Pm9_@Pn_)
                dW_=(torch.linalg.solve(An_,Kn_.T@R_).T)@Pm9_@Pn_
                cinfo_+=f" | NULL-SPACE EDIT: {r9_} key directions of {len(ko_)} earlier edited conclusions protected ({float(en9_[r9_-1]):.0%} of their key energy, {H_} dims); the new keys keep {kl9_:.0%} of theirs"
        bk_=float(getattr(cfg,'inf_cf_brake',0.03)); sc0_=float(getattr(cfg,'inf_cf_scale',1.0)); tr_=[]
        E9_=model.__dict__.get('_cf_E'); E9_=torch.zeros_like(W0_,dtype=torch.float32) if E9_ is None else E9_.to(W0_.device)
        mg_=float(getattr(cfg,'inf_cf_maxgrow',0.0)); bs9_=W0_.float()-E9_; nb9_=float(bs9_.norm()); En_={}
        def wnew_(al9_):                                                        # cg26n GROWTH CAP: accumulated edits at most maxgrow x the rest of the matrix
            En9_=E9_+al9_*dW_.float(); nn9_=float(En9_.norm())
            if mg_>0 and nn9_>mg_*nb9_: En9_=En9_*(mg_*nb9_/nn9_)
            En_[al9_]=En9_; return (bs9_+En9_).to(W0_.dtype)
        for al_ in (sc0_,sc0_/2,sc0_/4):
            with torch.no_grad(): mo_.weight.copy_(wnew_(al_))
            b1_,o1_=_ws_bpc(model,cfg,vals,t,ts,olds_); dB_,dO_=b1_-b0_,(o1_-o0_ if o1_==o1_ else 0.0); tr_.append(f"x{al_:g}: visited {dB_:+.4f}, old {dO_:+.4f}")
            if dB_<=bk_ and dO_<=bk_: kept_=(al_,dB_,dO_); break
        if kept_ is None:
            with torch.no_grad(): mo_.weight.copy_(W0_)
            cum_[3]+=1
            print(f"  [inference CLOSED-FORM CONSOLIDATION APPLIED] ROLLED BACK (brake {bk_}): "+" | ".join(tr_)+f" | before: edited {e0_:.2f} | PERSISTENCE: {pers_}",flush=True); return
        e1_=tw_(E_); v1_=tv_(E_); n1_=tw_(N_) if N_ else float('nan')
        rfid9_={id(x_) for x_ in RF_}
        for c_ in E_:
            if id(c_) in rfid9_: c_['cfr']=vis_
            else: c_['cfe']=vis_; c_['cfr']=vis_
            c_.pop('cf_refit',None)
        if RF_: rinfo_+=f", after {tw_(RF_):.2f}"
        cum_[0]+=kept_[1]; cum_[1]+=kept_[2]; cum_[2]+=1
        model.__dict__['_cf_E']=En_[kept_[0]].detach()                         # issue #28: record the accumulated edit right after the weights carry it, so a later failure
        if int(getattr(cfg,'inf_cf_diag',0)):                                   # CF DECAY DIAGNOSTIC: each edited conclusion's key, its own signal, and the matrix on its key now
            Wn9_=mo_.weight.detach().float(); En8_=En_[kept_[0]].float(); Bn9_=Wn9_-En8_; dE9_=En8_-E9_.float()
            for c9_,K9_ in kd9_:
                K9f_=K9_.float(); c9_['_cfd']={'k':K9f_.cpu(),'se':(K9f_@dE9_.T).cpu(),'Bk':(K9f_@Bn9_.T).cpu(),'Ek':(K9f_@En8_.T).cpu(),'v':vis_}
        if int(getattr(cfg,'inf_cf_null',0)) or float(getattr(cfg,'inf_key_anchor',0.0))>0:   # NULL-SPACE EDITS / KEY ANCHOR: remember each edited conclusion's keys
            for c9_,K9_ in kd9_: c9_['_cfk']=K9_.float().cpu()
        if int(getattr(cfg,'inf_cf_layerdrift',0)):                             # STEP 0: per-block states of up to 24 newly edited conclusions
            try:
                for c9_,_ in [x_ for x_ in kd9_ if id(x_[0]) not in rfid9_][:24]:
                    L9_=_cf_layer_states(model,c9_,S_slot)
                    if L9_ is not None: c9_['_cfL']=L9_; c9_['_cfLv']=vis_
            except Exception as e8_: _fail('inference.closed_form:254'); print(f"  [inference LAYER DRIFT] states unavailable ({type(e8_).__name__}: {e8_})"[:240],flush=True)
        model.__dict__['_cf_E']=En_[kept_[0]].detach(); cinfo_+=f" | ACCUMULATED EDITS |E| {float(En_[kept_[0]].norm()):.2f} vs rest of the matrix {nb9_:.2f}"+(f" (cap {mg_:g}x)" if mg_>0 else "")+dinfo_
        print(f"  [inference CLOSED-FORM CONSOLIDATION APPLIED] KEPT at x{kept_[0]:g} (brake {bk_}; tries: "+" | ".join(tr_)+f") | {len(E_)} conclusions edited (lambda {lam_:g}, |dW| {float(kept_[0]*dW_.norm()):.2f} vs |W| {float(W0_.norm()):.1f}) | "
              f"two-hop from the weights, fitted wording: edited {e0_:.2f} -> {e1_:.2f}, control (not edited) {n0_:.2f} -> {n1_:.2f} | OTHER WORDINGS: edited {v0_:.2f} -> {v1_:.2f} | "
              f"cumulative cost of {cum_[2]} kept edits (rolled back {cum_[3]}): visited {cum_[0]:+.4f}, old domains {cum_[1]:+.4f}"+(f" (budget {bud_:g})" if bud_>0 else "")+cinfo_+rinfo_+f" | PERSISTENCE: {pers_}",flush=True)
    except Exception as e9_:
        _fail('inference.closed_form:259')
        if kept_ is None:
            with torch.no_grad(): mo_.weight.copy_(W0_)
        print(f"  [inference CLOSED-FORM CONSOLIDATION APPLIED] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
    finally:
        cap_['delta']=None; hk_.remove(); back_()
def _ws_closed_form(model,cfg,S_,vals,olds_,t,ts,S_slot):
    """cg26k CLOSED-FORM CONSOLIDATION, tried on a copy and restored (after the weight-space-programming idea, done the MEMIT way and
    without an outside model): verified conclusions are WRITTEN into one block's MLP output weights in one least-squares step instead of
    being replayed. Keys = the MLP hidden activations at the positions that predict each conclusion's answer bytes; target changes = a small
    added output found by gradient steps on those positions only (store off); the edit dW = R^T K (K^T K + lam n C0)^-1 keeps generic text
    (C0 = the MLP-key covariance of stored real text) almost unchanged. Readout: two-hop from the weights on the edited conclusions and on
    others, BPC of the visited and old domains - against what this sleep's replay achieved."""
    S0_=S_; allc_=S_.get('conc',[]); lv_=[c_ for c_ in allc_ if not c_.get('freed') and isinstance(c_.get('C'),str) and isinstance(c_.get('A'),str)]
    ne_=int(getattr(cfg,'inf_cf_n',32))
    if len(lv_)<ne_+8: print(f"  [inference CLOSED-FORM CONSOLIDATION] skipped: {len(lv_)} live conclusions (need {ne_+8})",flush=True); return
    vis_=int(model.__dict__.get('_visit',0)); rs_=np.random.RandomState(8181+vis_); ix_=rs_.permutation(len(lv_)); E_=[lv_[int(i_)] for i_ in ix_[:ne_]]; N_=[lv_[int(i_)] for i_ in ix_[ne_:ne_+ne_]]
    Cp_=[c_['C'] for c_ in lv_]; li_=int(getattr(cfg,'inf_cf_layer',-1)); blk_=model.blocks[li_]; mo_=blk_.mlp_out
    back_=_ws_eager(model); W0_=mo_.weight.detach().clone(); cap_={'on':False,'k':[],'delta':None}
    def hook_(m_,inp_,out_):
        if cap_['on']: cap_['k'].append(inp_[0].detach().float())
        if cap_['delta'] is not None: return out_+cap_['delta'].to(out_.dtype)
    hk_=mo_.register_forward_hook(hook_)
    try:
        e0_=_ws_twohop(model,cfg,E_,Cp_,S_slot,np.random.RandomState(1)); n0_=_ws_twohop(model,cfg,N_,Cp_,S_slot,np.random.RandomState(2)); b0_,o0_=_ws_bpc(model,cfg,vals,t,ts,olds_)
        it_=S_['it']; rl_=[x_ for x_ in it_ if x_.get('k')=='real' and not x_.get('gone') and len(x_['s'])>=40]
        gk_=[]
        for j_ in rs_.choice(len(rl_),min(64,len(rl_)),replace=False) if rl_ else []:   # C0: MLP keys of stored real text (what must not change)
            X_,M_=_ws_batch(model,[rl_[int(j_)]['s']]); cap_['on']=True; cap_['k']=[]
            _ws_run(model,X_,int(S_slot(rl_[int(j_)]['d'])))
            cap_['on']=False; k9_=cap_['k'][-1][0]; gk_.append(k9_[M_[0,:-1]>0])
        Kg_=torch.cat(gk_) if gk_ else None
        Ks_=[]; Rs_=[]; lr_=float(getattr(cfg,'inf_cf_lr',0.5)); st_=int(getattr(cfg,'inf_cf_steps',25))
        for c_ in E_:                                                           # per conclusion: keys and the target change at its answer positions
            pr_=f"So {c_['A']} is connected to ".encode('latin-1','replace'); tg_=(c_['C']+'.').encode('latin-1','replace'); X_,M_=_ws_batch(model,[pr_+tg_])
            T1_=X_.shape[1]-1; ps_=torch.arange(len(pr_)-1,min(T1_,len(pr_)+len(tg_)-1),device=X_.device)
            if ps_.numel()==0: continue
            sl_=int(S_slot(c_['d'])); cap_['on']=True; cap_['k']=[]; _ws_run(model,X_,sl_); cap_['on']=False; K1_=cap_['k'][-1][0][ps_]
            D_=mo_.weight.shape[0]; dl_=torch.zeros(len(ps_),D_,device=X_.device,requires_grad=True); op_=torch.optim.Adam([dl_],lr=lr_*0.1)
            yt_=X_[0,ps_+1]
            for _ in range(st_):
                full_=torch.zeros(1,T1_,D_,device=X_.device); full_=full_.index_put((torch.zeros_like(ps_),ps_),dl_); cap_['delta']=full_
                lg_=_ws_run(model,X_,sl_,grad_=True); l_=F.cross_entropy(lg_[0,ps_],yt_)+1e-3*(dl_**2).sum()
                op_.zero_grad(); l_.backward(); op_.step(); cap_['delta']=None
            Ks_.append(K1_); Rs_.append(dl_.detach().float())
        cap_['delta']=None
        if not Ks_: print("  [inference CLOSED-FORM CONSOLIDATION] no edit positions",flush=True); return
        K_=torch.cat(Ks_); R_=torch.cat(Rs_); H_=K_.shape[1]; lam_=float(getattr(cfg,'inf_cf_lam',1.0))
        C0_=(Kg_.T@Kg_)/max(1,len(Kg_)) if Kg_ is not None else torch.eye(H_,device=K_.device)
        A_=K_.T@K_+lam_*len(K_)*C0_+1e-4*torch.eye(H_,device=K_.device); dW_=torch.linalg.solve(A_,K_.T@R_).T   # (d x H)
        res9_={}
        for al_ in [float(x_) for x_ in str(getattr(cfg,'inf_cf_alpha','0.5,1')).split(',')]:
            with torch.no_grad(): mo_.weight.copy_(W0_+al_*dW_.to(W0_.dtype))
            e1_=_ws_twohop(model,cfg,E_,Cp_,S_slot,np.random.RandomState(1)); n1_=_ws_twohop(model,cfg,N_,Cp_,S_slot,np.random.RandomState(2)); b1_,o1_=_ws_bpc(model,cfg,vals,t,ts,olds_)
            res9_[al_]=(e1_,n1_,b1_-b0_,o1_-o0_)
            with torch.no_grad(): mo_.weight.copy_(W0_)
        print(f"  [inference CLOSED-FORM CONSOLIDATION] (tried on a copy, restored) block {li_ % len(model.blocks)} MLP output, {len(E_)} conclusions written in one least-squares step "
              f"({len(K_)} answer positions, preservation lambda {lam_:g} on {0 if Kg_ is None else len(Kg_)} real-text keys, |dW| {float(dW_.norm()):.3f} vs |W| {float(W0_.norm()):.1f}) | before: "
              f"two-hop from the weights edited {e0_:.2f}, not edited {n0_:.2f} | "+" | ".join(f"scale {a_:g}: edited {r_[0]:.2f}, not edited {r_[1]:.2f}, visited BPC {r_[2]:+.4f}, old domains {r_[3]:+.4f}" for a_,r_ in res9_.items()),flush=True)
    except Exception as e9_: _fail('inference.closed_form:318'); print(f"  [inference CLOSED-FORM CONSOLIDATION] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
    finally:
        cap_['delta']=None; hk_.remove()
        with torch.no_grad(): mo_.weight.copy_(W0_)
        back_()
def _ws_direction(model,cfg,S_,vals,olds_,t,ts,S_slot):
    """cg26k DOMINANT-DIRECTION DIAGNOSTIC (after the weight-space-programming Qwen fix, where one direction written by two MLP layers caused
    looping): the top principal direction u of the trunk's passage codes; how much of it each block's mixer and MLP write; then, on a copy and
    restored, that direction removed from what the largest writer writes (W <- (I - a u u^T) W): code rank, top-direction share and BPC"""
    it_=S_['it']; rl_=[x_ for x_ in it_ if x_.get('k')=='real' and not x_.get('gone') and len(x_['s'])>=40]
    if len(rl_)<64: print(f"  [inference DIRECTION DIAGNOSTIC] skipped: {len(rl_)} stored real sentences",flush=True); return
    rs_=np.random.RandomState(9393); tx_=[rl_[int(j_)]['s'] for j_ in rs_.choice(len(rl_),min(128,len(rl_)),replace=False)]
    back_=_ws_eager(model); caps_={}; hks_=[]
    def mk_(key_):
        def h_(m_,inp_,out_): caps_.setdefault(key_,[]).append(out_.detach().float())
        return h_
    for i_,b_ in enumerate(model.blocks):
        for nm_ in ('Wout','mlp_out'):
            if hasattr(b_,nm_): hks_.append(getattr(b_,nm_).register_forward_hook(mk_((i_,nm_))))
    try:
        def codes_(collect_):
            out_=[]; parts_={}
            for o_ in range(0,len(tx_),32):
                X_,M_=_ws_batch(model,tx_[o_:o_+32]); caps_.clear()
                sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); fz_=_INFM.get('force'); _INFM['force']=False
                try:
                    with torch.no_grad(): h_=model.features(X_)
                finally:
                    _carry_load(sv4_)
                    if fz_ is None: _INFM.pop('force',None)
                    else: _INFM['force']=fz_
                h_=(h_[0] if isinstance(h_,(tuple,list)) else h_).float(); mm_=M_[:,:h_.shape[1],None]
                out_.append((h_*mm_).sum(1)/mm_.sum(1).clamp(min=1.0))
                if collect_:
                    for k_,L9_ in caps_.items():
                        y_=L9_[-1]; m2_=M_[:,:y_.shape[1],None]; parts_.setdefault(k_,[]).append((y_*m2_).sum(1)/m2_.sum(1).clamp(min=1.0))
            return torch.cat(out_),{k_:torch.cat(v_) for k_,v_ in parts_.items()}
        Z_,P_=codes_(True); Zc_=Z_-Z_.mean(0); U_,S9_,Vt_=torch.linalg.svd(Zc_,full_matrices=False); u_=Vt_[0]; sh0_=float(S9_[0]**2/(S9_**2).sum())
        er0_,_=_jx_effrank(Z_); vz_=float((Zc_@u_).var())
        wr_={k_:float(((v_-v_.mean(0))@u_).var())/max(vz_,1e-12) for k_,v_ in P_.items()}
        top_=sorted(wr_.items(),key=lambda kv_: -kv_[1])[:4]; b0_,o0_=_ws_bpc(model,cfg,vals,t,ts,olds_); res_=[]
        (bi_,nm_),_=top_[0]; mod_=getattr(model.blocks[bi_],nm_); W0_=mod_.weight.detach().clone(); uu_=u_.to(W0_.device)
        for hk_ in hks_: hk_.remove()
        hks_=[]
        try:
            for al_ in (0.5,1.0):
                with torch.no_grad(): mod_.weight.copy_(W0_-al_*torch.outer(uu_,uu_@W0_.float()).to(W0_.dtype))
                Z1_,_=codes_(False); er1_,_=_jx_effrank(Z1_); Z1c_=Z1_-Z1_.mean(0); S1_=torch.linalg.svdvals(Z1c_); sh1_=float(S1_[0]**2/(S1_**2).sum())
                b1_,o1_=_ws_bpc(model,cfg,vals,t,ts,olds_); res_.append((al_,er1_,sh1_,b1_-b0_,o1_-o0_))
        finally:
            with torch.no_grad(): mod_.weight.copy_(W0_)
        print(f"  [inference DIRECTION DIAGNOSTIC] top direction of the passage codes: {sh0_:.0%} of variance, effective rank {er0_:.1f} | written along it (variance share): "
              +", ".join(f"block {k_[0]} {'mixer' if k_[1]=='Wout' else 'MLP'} {v_:.2f}" for k_,v_ in top_)
              +f" | removed from block {bi_} {'mixer' if nm_=='Wout' else 'MLP'} output (on a copy, restored): "+" | ".join(f"scale {a_:g}: rank {e_:.1f}, top share {s_:.0%}, visited BPC {bv_:+.4f}, old domains {ov_:+.4f}" for a_,e_,s_,bv_,ov_ in res_),flush=True)
    except Exception as e9_: _fail('inference.closed_form:372'); print(f"  [inference DIRECTION DIAGNOSTIC] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
    finally:
        for hk_ in hks_: hk_.remove()
        back_()
def _snapcpu(model):
    """cg26d: a restore-only weight snapshot kept in CPU memory (frees one model's worth of GPU memory at peak; load_state_dict copies it back)"""
    return {k_:(v_.detach().to('cpu',copy=True) if v_.is_cuda else v_.detach().clone()) for k_,v_ in model.state_dict().items()}
def _gpu_gb():
    """cg26d: (peak allocated, reserved) GPU memory in GB since the last reset, or None off-GPU"""
    try:
        if torch.cuda.is_available(): return (torch.cuda.max_memory_allocated()/2**30, torch.cuda.memory_reserved()/2**30)
    except Exception: pass
    return None

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_jx_effrank': 'platypus.inference.jepa',
    'cfg_seq_': 'platypus.inference.jepa',
}
