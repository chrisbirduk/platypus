"""JEPA dream selector and latent readouts."""
import math
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
    _INFM, _ng_zero, bpc,
)
from platypus.inference.units import (
    _INF_REL, _INF_SYL, _UWORD,
)
from platypus.inference.probes import (
    inf_probes,
)
from platypus.inference.closed_form import (
    _snapcpu,
)
from platypus.inference.readouts import (
    _SCORE_LEN, _inf_score1,
)
from platypus.inference.selftaught import (
    _inf_cond_bytes, _mix_mask,
)

def _jepa_emb(model,txts_,grad_=False):
    """cg22: in chunks of 64 passages, each chunk padded only to its longest passage (same codes, bounded memory)"""
    if len(txts_)<=64: return _jepa_emb1(model,txts_,grad_)
    return torch.cat([_jepa_emb1(model,txts_[o_:o_+64],grad_) for o_ in range(0,len(txts_),64)])
def _jepa_emb1(model,txts_,grad_=False):
    """passage representation: the trunk's hidden states (store off, no carried context) averaged over the passage's own bytes"""
    L0_=int(cfg_seq_(model)); dev_=model.embed.weight.device; bs_=[t_[:L0_] for t_ in txts_]; L0_=max(8,max((len(b_) for b_ in bs_),default=8))
    X_=torch.full((len(bs_),L0_),32,dtype=torch.long,device=dev_); M_=torch.zeros(len(bs_),L0_,device=dev_)
    for j_,b_ in enumerate(bs_):
        if b_: X_[j_,:len(b_)]=torch.tensor(list(b_),device=dev_); M_[j_,:len(b_)]=1.0
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); fz_=_INFM.get('force'); _INFM['force']=False
    try:
        with torch.set_grad_enabled(grad_): h_=model.features(X_)
    finally:
        _carry_load(sv4_)
        if fz_ is None: _INFM.pop('force',None)
        else: _INFM['force']=fz_
    h_=(h_[0] if isinstance(h_,(tuple,list)) else h_).float()
    Q9_=model.__dict__.get('_jepaQ')
    if Q9_ is not None and int(getattr(model.cfg,'inf_jepa_attpool',0)):    # LEARNED ATTENTION POOLING: a trained query decides which positions make up the passage code
        a9_=(h_@Q9_)/math.sqrt(h_.shape[-1]); a9_=a9_.masked_fill(M_<=0,-1e9); w9_=torch.softmax(a9_,dim=1)   #   (starts uniform = mean pooling)
        return (h_*w9_[:,:,None]).sum(1)
    k9_=int(getattr(model.cfg,'inf_jepa_k',0))
    if k9_>0:                                                                   # SALIENT POOLING (v2): average the hidden states at the passage's k MOST SURPRISING
        sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); _INFM['force']=False   #   bytes (the trunk's own error picks them - names, rare content), not over every byte
        try:
            with torch.no_grad(): lp_=F.log_softmax(model(X_[:,:-1],_XDG['cur']).float(),-1); bt_=-lp_.gather(-1,X_[:,1:,None]).squeeze(-1)   # bits of byte j+1
        finally:
            _carry_load(sv4_)
            if fz_ is None: _INFM.pop('force',None)
            else: _INFM['force']=fz_
        sc_=torch.full_like(M_,-1e9); sc_[:,1:]=torch.where(M_[:,1:]>0,bt_,torch.full_like(bt_,-1e9))   # surprise of byte j, at position j
        W_=torch.zeros_like(M_); tk_=sc_.topk(min(k9_,sc_.shape[1]),dim=1).indices; W_.scatter_(1,tk_,1.0); W_=W_*M_
        return (h_*W_[:,:,None]).sum(1)/W_.sum(1,keepdim=True).clamp(min=1.0)
    return (h_*M_[:,:,None]).sum(1)/M_.sum(1,keepdim=True).clamp(min=1.0)
def cfg_seq_(model): return int(getattr(model.cfg,'seq_len',64))
def _jepa_pred(model,D_):
    P_=model.__dict__.get('_jepaP')
    if P_ is None or P_[0].in_features!=D_:
        P_=torch.nn.Sequential(torch.nn.Linear(D_,D_),torch.nn.GELU(),torch.nn.Linear(D_,D_)).to(model.embed.weight.device)
        with torch.no_grad(): P_[2].weight.mul_(0.1); P_[2].bias.zero_()
        Q_=torch.nn.Parameter(torch.zeros(D_,device=model.embed.weight.device)); model.__dict__['_jepaQ']=Q_
        model.__dict__['_jepaP']=P_; model.__dict__['_jepaO']=torch.optim.Adam(list(P_.parameters())+[Q_],lr=1e-3)
    return P_
def _inf_jepa(model,cfg,S_,ts):
    """LATENT SPLICE DREAMS (JEPA 'good views', cg20z lab): pairs of passages that are two observations of one underlying link - the premises
    of each committed conclusion (and mixing-accepted pairs) - train a small predictor to map one passage's REPRESENTATION to the other's,
    both ways, with a stop-gradient target (no bytes are predicted, so copying a shared name earns nothing), plus language-model anchor rows;
    rolled back if the anchors' bits rise by more than inf_jepa_brake. Readout: two-step latent retrieval on brand-new chains."""
    it_=S_['it']; rs_=np.random.RandomState(4242+int(model.__dict__.get('_visit',0))); was_=model.training
    pr_=[(it_[c_['src'][0]]['s'],it_[c_['src'][1]]['s']) for c_ in S_.get('conc',[]) if not c_.get('freed') and isinstance(c_.get('src'),tuple)]
    pr_+=[(ta_,tb_) for ta_,tb_,*_ in S_.get('mixtxt',[])]
    ring_=S_.get('ring',[]); sl_=int(ts); D_=None; st9_={}
    sc_=_XDG['cur']; _XDG['cur']=sl_
    try:
        if int(getattr(cfg,'inf_jepa',0)) and len(pr_)>=8 and len(ring_)>=16:
            with torch.no_grad(): D_=_jepa_emb(model,[pr_[0][0]]).shape[-1]
            P_=_jepa_pred(model,D_); oP_=model.__dict__['_jepaO']
            ps_=[q_ for q_ in model.parameters() if q_.requires_grad]; oT_=torch.optim.Adam(ps_,lr=float(cfg.lr)*float(getattr(cfg,'inf_jepa_lr',0.01)))
            dev_=model.embed.weight.device; Ln_=len(ring_[0][0])
            def lmb_(idx_):
                X_=torch.stack([torch.tensor(list(ring_[i_][0][:Ln_]),dtype=torch.long) for i_ in idx_]).to(dev_); return X_
            hold_=lmb_(rs_.choice(len(ring_),16,replace=False))
            def bits_():
                sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); model.eval()
                try:
                    with torch.no_grad(): lg_=model(hold_[:,:-1],sl_).float(); return float(F.cross_entropy(lg_.reshape(-1,lg_.shape[-1]),hold_[:,1:].reshape(-1)))/math.log(2)
                finally: _carry_load(sv4_)
            b0_=bits_(); sd_=_snapcpu(model); ls_=[]
            model.train()
            for _ in range(int(getattr(cfg,'inf_jepa_steps',30))):
                bt_=[pr_[int(i_)] for i_ in rs_.choice(len(pr_),min(int(getattr(cfg,'inf_jepa_batch',16)),len(pr_)),replace=False)]
                zA_=_jepa_emb(model,[a_ for a_,_ in bt_],True); zB_=_jepa_emb(model,[b_ for _,b_ in bt_],True)
                if int(getattr(cfg,'inf_jepa_nce',0)):                          # CONTRASTIVE variant: each passage's prediction must pick ITS partner out of the batch
                    tq_=float(getattr(cfg,'inf_jepa_tau',0.1)); yl_=torch.arange(len(bt_),device=zA_.device)   #   (the other pairs are negatives - a predictor cannot collapse
                    l1_=F.normalize(P_(zA_),dim=-1)@F.normalize(zB_.detach(),dim=-1).T/tq_; l2_=F.normalize(P_(zB_),dim=-1)@F.normalize(zA_.detach(),dim=-1).T/tq_   #   to one direction)
                    ja_=0.5*(F.cross_entropy(l1_,yl_)+F.cross_entropy(l2_,yl_))
                else: ja_=-(F.cosine_similarity(P_(zA_),zB_.detach(),dim=-1).mean()+F.cosine_similarity(P_(zB_),zA_.detach(),dim=-1).mean())/2   # both ways (symmetry)
                vr_=F.relu(1.0-torch.sqrt(torch.cat([zA_,zB_]).var(0)+1e-4)).mean()   # keep the representation from collapsing
                if int(getattr(cfg,'inf_jepa_iso',0)):                          # ISOTROPY (after LeJEPA's SIGReg idea): the standardised batch covariance pulled towards
                    zc_=torch.cat([zA_,zB_]); zc_=(zc_-zc_.mean(0))/(zc_.std(0)+1e-4); C9_=zc_.T@zc_/max(1,zc_.shape[0]-1)   #   the identity (decorrelated dimensions)
                    vr_=vr_+((C9_-torch.diag(torch.diag(C9_)))**2).sum()/zc_.shape[1]
                Xa_=lmb_(rs_.choice(len(ring_),8,replace=False)); sv4_=_carry_save(); _CARRY.update(on=False,st={},new={})
                try: lg_=model(Xa_[:,:-1],sl_).float()
                finally: _carry_load(sv4_)
                la_=F.cross_entropy(lg_.reshape(-1,lg_.shape[-1]),Xa_[:,1:].reshape(-1))
                loss_=float(getattr(cfg,'inf_jepa_w',1.0))*ja_+0.04*vr_+la_
                oT_.zero_grad(); oP_.zero_grad(); loss_.backward(); oT_.step(); oP_.step(); ls_.append(float(ja_))
                try: _ng_zero(model)
                except Exception: pass
            b1_=bits_(); rb_=(b1_-b0_)>float(getattr(cfg,'inf_jepa_brake',0.01))
            if rb_:
                with torch.no_grad(): model.load_state_dict(sd_)
            model.__dict__['_jepaN']=model.__dict__.get('_jepaN',0)+len(ls_)*16; del oT_
            with torch.no_grad():                                               # collapse check: spread of the predictor's outputs (1 = as varied as the inputs)
                zz_=_jepa_emb(model,[a_ for a_,_ in pr_[:64]]); pz_=F.normalize(P_(zz_),dim=-1); nz_=F.normalize(zz_,dim=-1)
                spr_=float((1-(pz_@pz_.T).mean())/max(1e-6,float(1-(nz_@nz_.T).mean())))
            st9_=dict(n=len(pr_),l0=ls_[0],l1=ls_[-1],b0=b0_,b1=b1_,rb=rb_,spr=spr_)
        # ---- readout: TWO-STEP LATENT RETRIEVAL on brand-new chains A-B, B-C, C-D (never stored): from the A-B passage, retrieve a neighbour of
        #      predictor(z), then a neighbour of predictor(that) - does it land on the C-D passage (which shares no name with A-B)?
        rq_=np.random.RandomState(777); used_=set(n_ for c_ in inf_probes(cfg) for n_ in c_['names']); fa_=[]
        def nm_():
            while True:
                n_=''.join(rq_.choice(_INF_SYL,rq_.randint(2,4))); n_=n_[0].upper()+n_[1:]
                if 5<=len(n_)<=10 and any(x_ in n_.lower() for x_ in 'qxz') and n_ not in used_: used_.add(n_); return n_
        for _ in range(48):
            a_,b_,c_,d_=nm_(),nm_(),nm_(),nm_(); r_=[_INF_REL[rq_.randint(len(_INF_REL))] for _ in range(3)]
            fa_.append([r_[0].format(a=a_,b=b_).encode('latin-1'),r_[1].format(a=b_,b=c_).encode('latin-1'),r_[2].format(a=c_,b=d_).encode('latin-1')])
        okr_=[i_ for i_ in range(len(it_)) if it_[i_].get('k')=='real' and len(it_[i_]['s'])>=24]
        dis_=[it_[int(i_)]['s'] for i_ in rq_.choice(okr_,min(400,len(okr_)),replace=False)] if okr_ else []
        mem_=[f_ for tr_ in fa_ for f_ in tr_]+dis_; model.eval()
        with torch.no_grad():
            Zr_=_jepa_emb(model,mem_); P_=model.__dict__.get('_jepaP')
            mu9_=Zr_.mean(0); sd9_=Zr_.std(0)+1e-4; wh_=lambda z_: F.normalize((z_-mu9_)/sd9_,dim=-1)   # audit fix: WHITENED retrieval (the raw codes all point one way)
            Z_=wh_(Zr_); Zn_=F.normalize(Zr_,dim=-1)
            def hop_(z_,ex_,usep_):                                             # raw codes: whitened cosine; the predictor: the cosine it was TRAINED with (audit fix 2:
                if usep_ and P_ is not None: s_=Zn_@F.normalize(P_(z_),dim=-1)  #   scoring its output under whitening scored it in a metric it never saw)
                else: s_=Z_@wh_(z_)
                for e_ in ex_: s_[e_]=-9
                return int(s_.argmax())
            res_={}
            if (int(getattr(cfg,'inf_jepa_cca',0)) or int(getattr(cfg,'inf_mix_latent',0))) and len(pr_)>=16:              # CLOSED-FORM LINEAR PREDICTOR (the linear JEPA optimum, cf. the paper's whitened cross-covariance):
                ZA_=_jepa_emb(model,[a_ for a_,_ in pr_[:512]]); ZB_=_jepa_emb(model,[b_ for _,b_ in pr_[:512]]); Zall_=torch.cat([ZA_,ZB_,Zr_])   #   shared shrinkage
                mu8_=Zall_.mean(0); C8_=torch.cov((Zall_-mu8_).T); C8_=0.9*C8_+0.1*C8_.diagonal().mean()*torch.eye(len(C8_),device=C8_.device)   #   whitener, ridge map
                ev8_,eV8_=torch.linalg.eigh(C8_); Wh8_=_pca_wh(ev8_,eV8_,int(getattr(cfg,'inf_jepa_pca',0)),len(ZA_),int(getattr(cfg,'inf_jepa_drop',0))); w8_=lambda z_: (z_-mu8_)@Wh8_   #   fitted on BOTH
                X8_=torch.cat([w8_(ZA_),w8_(ZB_)]); Y8_=torch.cat([w8_(ZB_),w8_(ZA_)]); lam_=0.1*len(X8_)   #   directions (symmetric)
                M8_=torch.linalg.solve(X8_.T@X8_+lam_*torch.eye(X8_.shape[1],device=X8_.device),X8_.T@Y8_); Zw8_=F.normalize(w8_(Zr_),dim=-1)
                model.__dict__['_jepaLin']=(mu8_.detach(),Wh8_.detach(),M8_.detach())   # kept for the latent proposer (preferred to the MLP, which memorised)
                h1_=h2_=0
                for k_ in range(len(fa_)):
                    i0_=3*k_
                    s1_=Zw8_@F.normalize(w8_(Zr_[i0_:i0_+1])@M8_,dim=-1).squeeze(0); s1_[i0_]=-9; j1_=int(s1_.argmax()); h1_+=int(j1_==i0_+1)
                    s2_=Zw8_@F.normalize(w8_(Zr_[j1_:j1_+1])@M8_,dim=-1).squeeze(0); s2_[i0_]=-9; s2_[j1_]=-9; h2_+=int(int(s2_.argmax())==i0_+2)
                res_['lin']=(h1_/len(fa_),h2_/len(fa_))
            for usep_ in ([False,True] if P_ is not None else [False]):
                h1_=h2_=0
                for k_ in range(len(fa_)):
                    i0_=3*k_; j1_=hop_(Zr_[i0_],[i0_],usep_); h1_+=int(j1_==i0_+1); j2_=hop_(Zr_[j1_],[i0_,j1_],usep_); h2_+=int(j2_==i0_+2)
                res_[usep_]=(h1_/len(fa_),h2_/len(fa_))
        model.train(was_)
        tx_=(f"probe only, no training ({len(pr_)} linked pairs for the predictor)") if not st9_ else (f"trained {st9_['n']} linked pairs: latent loss {st9_['l0']:.3f} -> {st9_['l1']:.3f}, prediction spread {st9_['spr']:.2f}, anchor bits {st9_['b0']:.4f} -> {st9_['b1']:.4f}"+(" ROLLED BACK" if st9_['rb'] else " kept")) if st9_ else ("readout only (no latent training)" if not int(getattr(cfg,'inf_jepa',0)) else "too few linked pairs to train")
        print(f"  [inference LATENT SPLICE DREAMS] {tx_} | TWO-STEP LATENT RETRIEVAL on 48 new chains (+{len(dis_)} distractors, chance {1/len(mem_):.3f}): raw representation hop1 {res_[False][0]:.2f} hop2 {res_[False][1]:.2f}"
              +(f" | with the predictor hop1 {res_[True][0]:.2f} hop2 {res_[True][1]:.2f}" if True in res_ else "")+(f" | closed-form linear predictor hop1 {res_['lin'][0]:.2f} hop2 {res_['lin'][1]:.2f}" if 'lin' in res_ else ""),flush=True)
        if any(int(getattr(cfg,k_,0)) for k_ in ('inf_jepa_views','inf_jepa_geom','inf_jepa_next','inf_jepa_shadow','inf_jepa_consol')):
            try: _jx_readouts(model,cfg,S_,ts,fa_,mem_,Zr_)
            except Exception as e_: print(f"  [inference LATENT READOUT] FAILED: {type(e_).__name__}: {str(e_)[:200]}",flush=True)
    finally:
        _XDG['cur']=sc_; model.train(was_)
def _pca_wh(ev_,eV_,k_,n_,drop_=0):
    """whitener: full (k_ 0) or onto the top-k principal directions (cg23: GPU codes have 1088 dims but only ~5 carry most variance, and the
    predictor is fitted on 128-512 pairs - in full width the ridge map overfits; next-span top-1 0.01 with the predictor vs 0.18 without)"""
    dr_=max(0,min(int(drop_),len(ev_)-8))                                       # cg24 ALL-BUT-THE-TOP: the drop_ largest directions (shared by every passage) are left out
    if k_<=0 and dr_==0: return eV_@torch.diag(ev_.clamp(min=1e-6).rsqrt())@eV_.T
    if k_<=0: ix_=torch.argsort(ev_,descending=True)[dr_:]; return eV_[:,ix_]@torch.diag(ev_[ix_].clamp(min=1e-6).rsqrt())
    k_=max(4,min(k_,len(ev_)-dr_,max(4,n_//4))); ix_=torch.argsort(ev_,descending=True)[dr_:dr_+k_]
    return eV_[:,ix_]@torch.diag(ev_[ix_].clamp(min=1e-6).rsqrt())
_JXK={'k':0,'drop':0}
def _jx_lin_fit(ZA_,ZB_,Zx_):
    """closed-form linear JEPA predictor (whitened ridge map, both directions, shared shrinkage whitener) -> (w(z), M)"""
    Zall_=torch.cat([ZA_,ZB_,Zx_]); mu_=Zall_.mean(0); C_=torch.cov((Zall_-mu_).T); C_=0.9*C_+0.1*C_.diagonal().mean()*torch.eye(len(C_),device=C_.device)
    ev_,eV_=torch.linalg.eigh(C_); Wh_=_pca_wh(ev_,eV_,_JXK['k'],len(ZA_),_JXK['drop']); w_=lambda z_: (z_-mu_)@Wh_
    X_=torch.cat([w_(ZA_),w_(ZB_)]); Y_=torch.cat([w_(ZB_),w_(ZA_)]); lam_=0.1*len(X_)
    M_=torch.linalg.solve(X_.T@X_+lam_*torch.eye(X_.shape[1],device=X_.device),X_.T@Y_); return w_,M_
def _jx_hops(q_,k_,n_):
    """two-step retrieval on chains stored as rows 3i, 3i+1, 3i+2 of the memory: q_ (memory x D) = what each row predicts, k_ = keys"""
    h1_=h2_=0
    for i_ in range(n_):
        i0_=3*i_; s1_=k_@q_[i0_]; s1_[i0_]=-9; j1_=int(s1_.argmax()); h1_+=int(j1_==i0_+1)
        s2_=k_@q_[j1_]; s2_[i0_]=-9; s2_[j1_]=-9; h2_+=int(int(s2_.argmax())==i0_+2)
    return h1_/max(1,n_),h2_/max(1,n_)
def _jx_sigreg(z_,nd_=64,tg_=None):
    """SIGReg (after LeJEPA): random 1-D projections of the embeddings compared with a standard normal through the empirical characteristic
    function (Epps-Pulley statistic, Gaussian-weighted) - pulls the embedding towards an isotropic Gaussian, the optimum the theory gives"""
    U_=F.normalize(torch.randn(z_.shape[1],nd_,device=z_.device),dim=0); x_=z_@U_
    t_=tg_ if tg_ is not None else torch.linspace(0.2,3.0,15,device=z_.device)
    tx_=x_[:,:,None]*t_[None,None,:]; re_=torch.cos(tx_).mean(0); im_=torch.sin(tx_).mean(0); gt_=torch.exp(-t_**2/2)
    return (((re_-gt_[None,:])**2+im_**2)*gt_[None,:]).mean()
def _sigreg_le(z_,ns_=256,nk_=17,tm_=5.0,gen_=None):
    """cg26j SIGReg exactly as LeJEPA (Balestriero & LeCun 2025, arXiv 2511.08544) defines it: the Epps-Pulley statistic
    N * integral over t in [-tm, tm] of |ecf(t) - exp(-t^2/2)|^2 * exp(-t^2/2) dt (trapezoid rule, nk knots), for each of ns random unit
    directions drawn afresh every call, averaged over directions. The integrand is even in t, so [0, tm] is integrated and doubled."""
    N_=z_.shape[0]; U_=F.normalize(torch.randn(z_.shape[1],ns_,generator=gen_,device='cpu' if gen_ is not None else z_.device,dtype=z_.dtype).to(z_.device),dim=0); x_=z_@U_
    t_=torch.linspace(0.0,tm_,(nk_+1)//2,device=z_.device,dtype=z_.dtype); tx_=x_[:,:,None]*t_[None,None,:]
    re_=torch.cos(tx_).mean(0); im_=torch.sin(tx_).mean(0); cf_=torch.exp(-t_**2/2)
    err_=((re_-cf_[None,:])**2+im_**2)*cf_[None,:]
    return N_*2.0*torch.trapezoid(err_,t_,dim=-1).mean()
def _jx_effrank(Z_):
    Zc_=Z_-Z_.mean(0); ev_=torch.linalg.eigvalsh(torch.cov(Zc_.T)).clamp(min=0); return float(ev_.sum()**2/(ev_**2).sum().clamp(min=1e-12)),float(ev_.max()/ev_.sum().clamp(min=1e-12))
def _jx_emb_sal(model,txts_,k_=16):
    k0_=int(getattr(model.cfg,'inf_jepa_k',0))
    try:
        model.cfg.inf_jepa_k=k_
        with torch.no_grad(): return _jepa_emb(model,txts_)
    finally: model.cfg.inf_jepa_k=k0_
def _jx_emb_mid(model,txts_):
    """mean-pooled hidden states of the middle block ((n-1)//2), eager forward with a hook (the compiled forward bypassed meanwhile)"""
    cmp_=_COMPILED.pop(id(model),None) if '_COMPILED' in globals() else None; cap_={}; hk_=None
    try:
        bl_=model.blocks[(len(model.blocks)-1)//2]
        def hook_(m_,i_,o_): cap_['h']=(o_[0] if isinstance(o_,(tuple,list)) else o_)
        hk_=bl_.register_forward_hook(hook_); L0_=int(cfg_seq_(model)); dev_=model.embed.weight.device; out_=[]
        for o9_ in range(0,len(txts_),64):
            bs_=[t_[:L0_] for t_ in txts_[o9_:o9_+64]]; L1_=max(8,max(len(b_) for b_ in bs_))
            X_=torch.full((len(bs_),L1_),32,dtype=torch.long,device=dev_); M_=torch.zeros(len(bs_),L1_,device=dev_)
            for j_,b_ in enumerate(bs_): X_[j_,:len(b_)]=torch.tensor(list(b_),device=dev_); M_[j_,:len(b_)]=1.0
            sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); fz_=_INFM.get('force'); _INFM['force']=False
            try:
                with torch.no_grad(): model.features(X_)
            finally:
                _carry_load(sv4_)
                if fz_ is None: _INFM.pop('force',None)
                else: _INFM['force']=fz_
            h9_=cap_['h'].float(); out_.append((h9_*M_[:,:,None]).sum(1)/M_.sum(1,keepdim=True).clamp(min=1.0))
        return torch.cat(out_)
    finally:
        if hk_ is not None: hk_.remove()
        if cmp_ is not None: _COMPILED[id(model)]=cmp_
def _jx_readouts(model,cfg,S_,ts,fa_,mem_,Zr_):
    """cg22 LATENT (JEPA) READOUTS - nothing here changes the model (weights copied and restored where training is tried):
    VIEWS (which pairs teach links: verified premises, accepted splices, random same-domain pairs, neighbouring sentences) | GEOMETRY (effective
    rank and drift of a fixed passage set across sleeps) | NEXT-SPAN PREDICTABILITY (closed-form JEPA from a window's first half to its second)
    | SHADOW JEPA (trainable head + EMA target + predictor + SIGReg on frozen trunk embeddings, continual across sleeps) | LATENT CONSOLIDATION
    tried on a copy (trunk taught to make the linear JEPA relation hold on verified links; two-hop, latent retrieval and BPC cost; restored)"""
    it_=S_['it']; rs_=np.random.RandomState(5151+int(model.__dict__.get('_visit',0))); out_=[]; nC_=len(fa_); _JXK['k']=int(getattr(cfg,'inf_jepa_pca',0)); _JXK['drop']=int(getattr(cfg,'inf_jepa_drop',0))
    prem_=[(it_[c_['src'][0]]['s'],it_[c_['src'][1]]['s']) for c_ in S_.get('conc',[]) if not c_.get('freed') and isinstance(c_.get('src'),tuple)]
    accp_=[(a_,b_) for a_,b_,*_ in S_.get('mixtxt',[])]
    real_=[i_ for i_ in range(len(it_)) if it_[i_].get('k')=='real' and not it_[i_].get('gone') and len(it_[i_]['s'])>=24]
    # (1) VIEWS
    if int(getattr(cfg,'inf_jepa_views',0)) and len(real_)>=64:
        try:
            byd_={}
            for i_ in real_: byd_.setdefault(it_[i_]['d'],[]).append(i_)
            rnd_=[]
            for _ in range(256):
                d_=list(byd_)[rs_.randint(len(byd_))]; L_=byd_[d_]
                if len(L_)>=2: a_,b_=rs_.choice(len(L_),2,replace=False); rnd_.append((it_[L_[a_]]['s'],it_[L_[b_]]['s']))
            adj_=[(it_[i_]['s'],it_[i_+1]['s']) for i_ in real_[-4000:] if i_+1<len(it_) and it_[i_+1].get('k')=='real' and it_[i_+1].get('d')==it_[i_].get('d') and it_[i_+1].get('v')==it_[i_].get('v')]
            adj_=[adj_[int(j_)] for j_ in rs_.choice(len(adj_),min(256,len(adj_)),replace=False)] if adj_ else []
            vs_=[]
            for nm_,P_ in (('verified premises',prem_),('accepted splices',accp_),('random same-domain',rnd_),('neighbouring sentences',adj_)):
                P_=P_[-512:]
                if len(P_)<16: vs_.append(f"{nm_} n {len(P_)} (too few)"); continue
                with torch.no_grad():
                    ZA_=_jepa_emb(model,[a_ for a_,_ in P_]); ZB_=_jepa_emb(model,[b_ for _,b_ in P_]); w_,M_=_jx_lin_fit(ZA_,ZB_,Zr_)
                    h1_,h2_=_jx_hops(F.normalize(w_(Zr_)@M_,dim=-1),F.normalize(w_(Zr_),dim=-1),nC_)
                vs_.append(f"{nm_} (n {len(P_)}) hop1 {h1_:.2f} hop2 {h2_:.2f}")
            out_.append("VIEWS (closed-form predictor fitted on each kind of pair): "+" | ".join(vs_))
        except Exception as e_: out_.append(f"VIEWS failed ({type(e_).__name__}: {str(e_)[:120]})")
    # (2) GEOMETRY of a fixed passage set across sleeps
    if int(getattr(cfg,'inf_jepa_geom',0)):
        try:
            if not model.__dict__.get('_jxref') and len(real_)>=256: model.__dict__['_jxref']=[it_[real_[int(j_)]]['s'] for j_ in rs_.choice(len(real_),256,replace=False)]
            ref_=model.__dict__.get('_jxref')
            if ref_:
                with torch.no_grad(): Z_=_jepa_emb(model,ref_)
                er_,top_=_jx_effrank(Z_); Zp_=model.__dict__.get('_jxrefZ'); dr_=''
                if Zp_ is not None and Zp_.shape==Z_.shape:
                    Zp_=Zp_.to(Z_.device); A_=Z_-Z_.mean(0); B_=Zp_-Zp_.mean(0)
                    cka_=float((B_.T@A_).norm()**2/((B_.T@B_).norm()*(A_.T@A_).norm()).clamp(min=1e-12))
                    An_=A_/A_.norm(); Bn_=B_/B_.norm(); U_,_,Vt_=torch.linalg.svd(An_.T@Bn_); R_=U_@Vt_; pr_=float((An_@R_-Bn_).norm())
                    dr_=f" | since last sleep: linear CKA {cka_:.3f}, Procrustes residual {pr_:.3f} (0 = only rotated)"
                model.__dict__['_jxrefZ']=Z_.detach().cpu()
                out_.append(f"GEOMETRY (256 fixed passages): effective rank {er_:.1f} of {Z_.shape[1]} dims, top direction {100*top_:.0f}% of variance"+dr_)
                if int(getattr(cfg,'inf_jepa_geom',0))>=2:                       # cg23 CORRECTED GEOMETRY + POOLING CHOICES: is the information there but hidden under a few
                    try:                                                        #   shared directions, or missing? (rank, and next-span retrieval with no predictor)
                        def tops_(Zx_,k_):
                            Zc_=Zx_-Zx_.mean(0)
                            if k_>0: U_,S_,Vh_=torch.linalg.svd(Zc_,full_matrices=False); Zc_=Zc_-(Zc_@Vh_[:k_].T)@Vh_[:k_]
                            return Zc_
                        std_=lambda Zx_: (Zx_-Zx_.mean(0))/(Zx_.std(0)+1e-6)
                        g_=[f"centred {_jx_effrank(tops_(Z_,0))[0]:.1f}",f"top-1 removed {_jx_effrank(tops_(Z_,1))[0]:.1f}",f"top-3 removed {_jx_effrank(tops_(Z_,3))[0]:.1f}",f"per-dimension standardised {_jx_effrank(std_(Z_))[0]:.1f}"]
                        k0_=int(getattr(model.cfg,'inf_jepa_k',0))
                        try:
                            model.cfg.inf_jepa_k=16
                            with torch.no_grad(): Zk_=_jepa_emb(model,ref_)
                        finally: model.cfg.inf_jepa_k=k0_
                        g_.append(f"salient pooling (16 most surprising bytes) {_jx_effrank(Zk_)[0]:.1f}")
                        Zm_=None; cmp_=_COMPILED.pop(id(model),None) if '_COMPILED' in globals() else None; hk_=None; cap_={}
                        try:
                            bl_=model.blocks[(len(model.blocks)-1)//2]
                            def hook_(m_,i_,o_): cap_['h']=(o_[0] if isinstance(o_,(tuple,list)) else o_)
                            hk_=bl_.register_forward_hook(hook_)
                            L0_=int(cfg_seq_(model)); dev_=model.embed.weight.device; Zs9_=[]
                            for o9_ in range(0,len(ref_),64):
                                bs_=[t_[:L0_] for t_ in ref_[o9_:o9_+64]]; L1_=max(8,max(len(b_) for b_ in bs_))
                                X_=torch.full((len(bs_),L1_),32,dtype=torch.long,device=dev_); M_=torch.zeros(len(bs_),L1_,device=dev_)
                                for j_,b_ in enumerate(bs_): X_[j_,:len(b_)]=torch.tensor(list(b_),device=dev_); M_[j_,:len(b_)]=1.0
                                sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); fz_=_INFM.get('force'); _INFM['force']=False
                                try:
                                    with torch.no_grad(): model.features(X_)
                                finally:
                                    _carry_load(sv4_)
                                    if fz_ is None: _INFM.pop('force',None)
                                    else: _INFM['force']=fz_
                                h9_=cap_['h'].float(); Zs9_.append((h9_*M_[:,:,None]).sum(1)/M_.sum(1,keepdim=True).clamp(min=1.0))
                            Zm_=torch.cat(Zs9_); g_.append(f"middle block {(len(model.blocks)-1)//2+1} of {len(model.blocks)} {_jx_effrank(Zm_)[0]:.1f}")
                        except Exception as e9_: g_.append(f"middle block unavailable ({type(e9_).__name__})")
                        finally:
                            if hk_ is not None: hk_.remove()
                            if cmp_ is not None: _COMPILED[id(model)]=cmp_
                        out_.append("GEOMETRY CORRECTED (effective rank): "+" | ".join(g_))
                        model.__dict__['_jxgeoZ']={'mean':Z_,'salient':Zk_,'middle':Zm_}
                    except Exception as e_: out_.append(f"GEOMETRY CORRECTED failed ({type(e_).__name__}: {str(e_)[:120]})")
        except Exception as e_: out_.append(f"GEOMETRY failed ({type(e_).__name__}: {str(e_)[:120]})")
    # (3) NEXT-SPAN PREDICTABILITY (the daytime JEPA question, read out: does a window's first half predict its second half in latent space?)
    ring_=S_.get('ring',[])
    if int(getattr(cfg,'inf_jepa_next',0)) and len(ring_)>=64:
        try:
            ix_=rs_.choice(len(ring_),min(256,len(ring_)),replace=False); W_=[bytes(ring_[int(i_)][0]) for i_ in ix_]; h_=[len(w_)//2 for w_ in W_]
            with torch.no_grad():
                Zc_=_jepa_emb(model,[w_[:k_] for w_,k_ in zip(W_,h_)]); Zs_=_jepa_emb(model,[w_[k_:] for w_,k_ in zip(W_,h_)]); n2_=len(W_)//2
                w_,M_=_jx_lin_fit(Zc_[:n2_],Zs_[:n2_],Zs_[:0]); K_=F.normalize(w_(Zs_[n2_:]),dim=-1)
                acc_=float((F.normalize(w_(Zc_[n2_:])@M_,dim=-1)@K_.T).argmax(1).eq(torch.arange(len(K_),device=K_.device)).float().mean())
                acc0_=float((F.normalize(w_(Zc_[n2_:]),dim=-1)@K_.T).argmax(1).eq(torch.arange(len(K_),device=K_.device)).float().mean())
            ex9_=""
            if int(getattr(cfg,'inf_jepa_geom',0))>=2:                           # cg23: no-predictor retrieval under each correction (functional check of the geometry)
                with torch.no_grad():
                    def r9_(A_,B_):
                        A_=F.normalize(A_,dim=-1); B_=F.normalize(B_,dim=-1); return float((A_@B_.T).argmax(1).eq(torch.arange(len(B_),device=B_.device)).float().mean())
                    C9_=Zc_[n2_:]; S9_=Zs_[n2_:]; mu9_=torch.cat([Zc_,Zs_]).mean(0); sd9_=torch.cat([Zc_,Zs_]).std(0)+1e-6
                    Zall9_=torch.cat([Zc_,Zs_])-mu9_; U9_,_,V9_=torch.linalg.svd(Zall9_,full_matrices=False); rm_=lambda Z9_,k9_: (Z9_-mu9_)-((Z9_-mu9_)@V9_[:k9_].T)@V9_[:k9_]
                    ex9_=(f" | no predictor, raw cosine {r9_(C9_,S9_):.2f}, centred {r9_(C9_-mu9_,S9_-mu9_):.2f}, standardised {r9_((C9_-mu9_)/sd9_,(S9_-mu9_)/sd9_):.2f}, "
                          f"top-3 removed {r9_(rm_(C9_,3),rm_(S9_,3)):.2f}, top-10 removed {r9_(rm_(C9_,10),rm_(S9_,10)):.2f}")
            if int(getattr(cfg,'inf_jepa_geom',0))>=3:                           # cg25 POOLING CHOICES, judged by use: no-predictor retrieval with salient / middle-block codes
                try:
                    cw_=[w_[:k_] for w_,k_ in zip(W_,h_)][n2_:]; sw_=[w_[k_:] for w_,k_ in zip(W_,h_)][n2_:]
                    def r8_(A_,B_):
                        mu_=torch.cat([A_,B_]).mean(0); A_=F.normalize(A_-mu_,dim=-1); B_=F.normalize(B_-mu_,dim=-1); return float((A_@B_.T).argmax(1).eq(torch.arange(len(B_),device=B_.device)).float().mean())
                    ex9_+=f" | POOLING (centred cosine): mean {r8_(Zc_[n2_:],Zs_[n2_:]):.2f}, salient-16 {r8_(_jx_emb_sal(model,cw_),_jx_emb_sal(model,sw_)):.2f}, middle block {r8_(_jx_emb_mid(model,cw_),_jx_emb_mid(model,sw_)):.2f}"
                except Exception as e8_: ex9_+=f" | POOLING failed ({type(e8_).__name__})"
            out_.append(f"NEXT-SPAN (first half of a recent window -> its second half, {len(K_)} test windows, chance {1/len(K_):.3f}): closed-form predictor top-1 {acc_:.2f} | no predictor (whitened) {acc0_:.2f}"+ex9_)
        except Exception as e_: out_.append(f"NEXT-SPAN failed ({type(e_).__name__}: {str(e_)[:120]})")
    # (4) SHADOW JEPA on frozen trunk embeddings: trainable head g, EMA target head, predictor, squared loss on normalised codes, SIGReg
    P2_=(prem_+accp_)[-512:]
    if int(getattr(cfg,'inf_jepa_shadow',0)) and len(P2_)>=16:
        try:
            with torch.no_grad(): ZA_=_jepa_emb(model,[a_ for a_,_ in P2_]); ZB_=_jepa_emb(model,[b_ for _,b_ in P2_])
            Zal_=torch.cat([ZA_,ZB_,Zr_]); mu_=Zal_.mean(0); sd_=Zal_.std(0)+1e-4; st_=lambda z_: (z_-mu_)/sd_
            D_=ZA_.shape[1]; d_=int(getattr(cfg,'inf_jepa_shadow_dim',128)); SH_=model.__dict__.get('_jxS')
            if SH_ is None or SH_[0].in_features!=D_:
                import copy as _cp
                g_=torch.nn.Linear(D_,d_).to(ZA_.device); gt_=_cp.deepcopy(g_); P_=torch.nn.Sequential(torch.nn.Linear(d_,d_),torch.nn.GELU(),torch.nn.Linear(d_,d_)).to(ZA_.device)
                for q_ in gt_.parameters(): q_.requires_grad_(False)
                SH_=(g_,gt_,P_,torch.optim.Adam(list(g_.parameters())+list(P_.parameters()),lr=1e-3)); model.__dict__['_jxS']=SH_
            g_,gt_,P_,op_=SH_; xa_,xb_=st_(ZA_),st_(ZB_); ls_=[]; lam_=float(getattr(cfg,'inf_jepa_sigreg',0.1)); ema_=float(getattr(cfg,'inf_jepa_ema',0.99))
            with torch.enable_grad():
                for _ in range(int(getattr(cfg,'inf_jepa_shadow_steps',200))):
                    b_=torch.as_tensor(rs_.randint(len(P2_),size=min(64,len(P2_))),device=ZA_.device); a1_,b1_=xa_[b_],xb_[b_]
                    lp_=((F.normalize(P_(g_(a1_)),dim=-1)-F.normalize(gt_(b1_),dim=-1))**2).sum(-1).mean()+((F.normalize(P_(g_(b1_)),dim=-1)-F.normalize(gt_(a1_),dim=-1))**2).sum(-1).mean()
                    loss_=lp_/2+lam_*_jx_sigreg(g_(torch.cat([a1_,b1_])))
                    op_.zero_grad(); loss_.backward(); op_.step(); ls_.append(float(lp_/2))
                    with torch.no_grad():
                        for p_,q_ in zip(gt_.parameters(),g_.parameters()): p_.mul_(ema_).add_(q_,alpha=1-ema_)
            with torch.no_grad():
                xr_=st_(Zr_); h1_,h2_=_jx_hops(F.normalize(P_(g_(xr_)),dim=-1),F.normalize(gt_(xr_),dim=-1),nC_); er_,_=_jx_effrank(g_(xr_))
            model.__dict__['_jxSn']=model.__dict__.get('_jxSn',0)+1
            out_.append(f"SHADOW JEPA (head {D_}->{d_} + EMA target + predictor + SIGReg, on frozen trunk codes, {len(P2_)} linked pairs, sleep {model.__dict__['_jxSn']}): loss {ls_[0]:.3f} -> {ls_[-1]:.3f}, code effective rank {er_:.1f} of {d_} | new chains hop1 {h1_:.2f} hop2 {h2_:.2f}")
        except Exception as e_: out_.append(f"SHADOW JEPA failed ({type(e_).__name__}: {str(e_)[:120]})")
    # (5) LATENT CONSOLIDATION, tried on a copy of the weights
    vo_=model.__dict__.get('_jx_vo')
    if int(getattr(cfg,'inf_jepa_consol',0)) and len(prem_)>=16 and vo_ is not None and len(ring_)>=16:
        sd0_=None
        try:
            vals_,olds_,t_=vo_; nb_=int(cfg.pu_cons_eval) or 4; dev_=model.embed.weight.device; sl_=int(ts); was_=model.training
            P3_=prem_[-256:]
            with torch.no_grad():
                ZA_=_jepa_emb(model,[a_ for a_,_ in P3_]); ZB_=_jepa_emb(model,[b_ for _,b_ in P3_]); w_,M_=_jx_lin_fit(ZA_,ZB_,Zr_)
            def lat_():                                                          # in-sample latent link: from each premise A-B, find its partner B-C among all partners
                with torch.no_grad():
                    za_=_jepa_emb(model,[a_ for a_,_ in P3_]); zb_=_jepa_emb(model,[b_ for _,b_ in P3_])
                    return float((F.normalize(w_(za_)@M_,dim=-1)@F.normalize(w_(zb_),dim=-1).T).argmax(1).eq(torch.arange(len(P3_),device=dev_)).float().mean())
            lv_=[c_ for c_ in S_.get('conc',[]) if not c_.get('freed') and isinstance(c_.get('C'),str)]; Cp_=[c_['C'] for c_ in lv_]; sm_=[]
            for i_ in rs_.choice(len(lv_),min(24,len(lv_)),replace=False) if lv_ else []:
                c_=lv_[int(i_)]; o_=[x_ for x_ in Cp_ if x_!=c_['C']]; rs_.shuffle(o_)
                if len(o_)>=4: sm_.append((c_['A'],[c_['C']]+o_[:9],int(model.__dict__.get('_t2s',{}).get(c_['d'],c_['d']))))
            def two_():
                if not sm_: return float('nan')
                v_=float(np.mean([_inf_score1(model,cfg,f"So {a_} is connected to ",cd_,s_,False)==0 for a_,cd_,s_ in sm_])); _INFM.pop('force',None); return v_
            def bp_(): return bpc(model,vals_[t_],sl_,n=nb_,skip=int(getattr(cfg,'brake_eval_skip',0))),(float(np.mean([bpc(model,vals_[d_],s_,n=nb_,skip=int(getattr(cfg,'brake_eval_skip',0))) for d_,s_ in olds_])) if olds_ else float('nan'))
            model.eval(); l0_,t0_=lat_(),two_(); v0_,o0_=bp_()
            sd0_=_snapcpu(model)
            ps_=[q_ for q_ in model.parameters() if q_.requires_grad]; op_=torch.optim.Adam(ps_,lr=float(cfg.lr)*float(getattr(cfg,'inf_jepa_consol_lr',0.01)))
            Ln_=len(ring_[0][0]); ls_=[]; model.train()
            for _ in range(int(getattr(cfg,'inf_jepa_consol_steps',30))):
                bt_=[P3_[int(i_)] for i_ in rs_.choice(len(P3_),min(16,len(P3_)),replace=False)]
                za_=_jepa_emb(model,[a_ for a_,_ in bt_],True); zb_=_jepa_emb(model,[b_ for _,b_ in bt_],True)
                lj_=(2-F.cosine_similarity(w_(za_)@M_,w_(zb_).detach(),dim=-1).mean()-F.cosine_similarity(w_(zb_)@M_,w_(za_).detach(),dim=-1).mean())/2
                Xa_=torch.stack([torch.tensor(list(ring_[int(i_)][0][:Ln_]),dtype=torch.long) for i_ in rs_.choice(len(ring_),8,replace=False)]).to(dev_)
                sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); _INFM['force']=False
                try: lg_=model(Xa_[:,:-1],sl_).float()
                finally: _carry_load(sv4_); _INFM.pop('force',None)
                la_=F.cross_entropy(lg_.reshape(-1,lg_.shape[-1]),Xa_[:,1:].reshape(-1))
                loss_=float(getattr(cfg,'inf_jepa_w',1.0))*lj_+la_; op_.zero_grad(set_to_none=True); loss_.backward(); op_.step(); ls_.append(float(lj_))
                try: _ng_zero(model)
                except Exception: pass
            model.eval(); l1_,t1_=lat_(),two_(); v1_,o1_=bp_(); del op_
            out_.append(f"LATENT CONSOLIDATION (tried on a copy: {len(ls_)} steps at lr x{float(getattr(cfg,'inf_jepa_consol_lr',0.01))}, {len(P3_)} verified links, trunk taught the linear JEPA relation + LM anchors; RESTORED): "
                        f"latent loss {ls_[0]:.3f} -> {ls_[-1]:.3f} | in-sample latent link {l0_:.2f} -> {l1_:.2f} | two-hop from the weights {t0_:.2f} -> {t1_:.2f} (n {len(sm_)}) | BPC visited {v0_:.4f} -> {v1_:.4f} ({v1_-v0_:+.4f}), old domains {o0_:.4f} -> {o1_:.4f} ({o1_-o0_:+.4f})")
        except Exception as e_: out_.append(f"LATENT CONSOLIDATION failed ({type(e_).__name__}: {str(e_)[:160]})")
        finally:
            if sd0_ is not None:
                with torch.no_grad(): model.load_state_dict(sd0_)
                try: _ng_zero(model)
                except Exception: pass
            for q_ in model.parameters(): q_.grad=None
            model.train(True)
    for o_ in out_: print(f"  [inference LATENT READOUT] {o_}",flush=True)
def _tfidf_sim(S_,qa_,ca_,dfmax_,names_=False):
    """TF-IDF cosine between cue passages and candidate passages: words of 3+ letters seen in 2..(2 pct of the store) sentences, weight
    log(N / store frequency); binary term counts (rare shared words = names dominate)"""
    df_=S_.get('df',{}); N_=max(1,len(S_.get('it',[]))); mx_=max(5,int(0.02*N_))
    def vec_(t_):
        v_={}
        for w_ in set(m_.group(0) for m_ in _UWORD.finditer(t_) if len(m_.group(0))>=3 and (not names_ or (len(m_.group(0))>=4 and m_.group(0)[:1].isupper() and m_.group(0)[1:].isalpha() and m_.group(0)[1:].islower()))):
            c_=df_.get(w_,0)
            if names_ and df_.get(w_.lower(),0)>dfmax_: continue                  # cg24: a capitalised COMMON word ('First', 'Category') is not a name
            if 2<=c_<=mx_: v_[w_]=math.log(N_/c_)
        n_=math.sqrt(sum(x_*x_ for x_ in v_.values())) or 1.0; return {k_:x_/n_ for k_,x_ in v_.items()}
    Vq_=[vec_(t_) for t_ in qa_]; Vc_=[vec_(t_) for t_ in ca_]; inv_={}
    for j_,v_ in enumerate(Vc_):
        for w_,x_ in v_.items(): inv_.setdefault(w_,[]).append((j_,x_))
    M_=torch.zeros(len(Vq_),len(Vc_))
    for i_,v_ in enumerate(Vq_):
        for w_,x_ in v_.items():
            for j_,y_ in inv_.get(w_,()): M_[i_,j_]+=x_*y_
    return M_
def _inf_ceiling2(model,cfg,nm,sl0_,acc,rs_):
    """COMPOSITION CEILING, LONG (after Wang et al. 2024, grokked implicit reasoning): an entity graph where every entity has two outgoing facts;
    two-hop paths A -> B -> C. ID entities: half of their paths are taught as two-hop statements, the other half held out (in-distribution
    composition); OOD entities: no two-hop statement ever taught (their facts are known only atomically). Long training with weight decay and a
    learning curve, in the weights and in context (worked examples); weights restored after each arm."""
    NE_=int(getattr(cfg,'inf_ceiling2_ent',240)); K_=int(getattr(cfg,'inf_ceiling2_steps',8000)); ev_=[k_ for k_ in (250,500,1000,2000,4000,6000,8000,12000,16000) if k_<=K_]
    E_=[nm() for _ in range(NE_)]; R2_=[_INF_REL[rs_.randint(len(_INF_REL))] for _ in range(2)]
    out_={e_:[(R2_[k_],E_[rs_.randint(NE_)]) for k_ in range(2)] for e_ in E_}
    for e_ in E_: out_[e_]=[(r_,t_ if t_!=e_ else E_[(E_.index(e_)+1)%NE_]) for r_,t_ in out_[e_]]
    atom_=[r_.format(a=e_,b=t_) for e_ in E_ for r_,t_ in out_[e_]]
    paths_={e_:[(e_,b_,c_,r1_.format(a=e_,b=b_),r2_.format(a=b_,b=c_)) for r1_,b_ in out_[e_] for r2_,c_ in out_[b_] if c_!=e_] for e_ in E_}
    idn_=set(E_[:int(0.7*NE_)]); trn_=[];idt_=[];ood_=[]
    for e_ in E_:
        P_=paths_[e_]
        if e_ in idn_: h_=len(P_)//2; trn_+=P_[:h_]; idt_+=P_[h_:]
        else: ood_+=P_
    tgt_={c_ for a_,b_,c_,_,_ in trn_}                                         # FAIR DISTRACTORS: only entities that are themselves trained two-hop answers (each has been
    def items(L_,ctx_):                                                         #   seen after 'is connected to'), so a learned answer prior gives no edge
        it_=[]
        for a_,b_,c_,f0_,f1_ in L_[:64]:
            bad_={x_ for _,_,x_,_,_ in paths_[a_]}; o_=[x_ for x_ in tgt_ if x_ not in bad_ and x_!=c_]; rs_.shuffle(o_)
            if ctx_:                                                            # IN CONTEXT with a DECOY fact: a third name in the prompt is a candidate too, so copying an
                d_=o_[0]; dq_=E_[rs_.randint(NE_)]; dec_=_INF_REL[rs_.randint(len(_INF_REL))].format(a=dq_,b=d_)   #   in-context name is right only half the time
                it_.append((f"{f1_} {dec_} {f0_} So {a_} is connected to ",[c_]+o_[:9],sl0_))
            else: it_.append((f"So {a_} is connected to ",[c_]+o_[:9],sl0_))
        return it_
    oneh_=[(f"{r_.format(a=e_,b='').rsplit(' ',1)[0]} ",[t_]+[x_ for x_ in rs_.choice(E_,9,replace=False) if x_!=t_][:9],sl0_) for e_ in E_[:64] for r_,t_ in out_[e_][:1]]
    dev_=model.embed.weight.device; sd_=_snapcpu(model); wd_=float(getattr(cfg,'inf_ceiling2_wd',0.1))
    _SCORE_LEN['n']=max(_SCORE_LEN['n'],200); res_={}
    def run(txt_,ctx_):
        opt_=torch.optim.AdamW([q_ for n_,q_ in model.named_parameters() if q_.requires_grad and (q_.is_cuda or ('ngram' not in n_ and q_.numel()<4_000_000))],lr=float(getattr(cfg,'inf_cls_lr',3e-4)),weight_decay=wd_); cur_=[]; ls_=[]   # (no hashed n-gram tables or huge embeddings: local memory)
        for st_ in range(1,K_+1):
            model.train(); bt_=[txt_[i_] for i_ in rs_.choice(len(txt_),min(16,len(txt_)),replace=False)]; L0_=max(len(t_) for t_ in bt_)+1
            X_=torch.full((len(bt_),L0_),32,dtype=torch.long,device=dev_)
            for j_,t_ in enumerate(bt_): X_[j_,:len(t_)]=torch.tensor(list(t_.encode('latin-1')),device=dev_)
            lg_=model(X_[:,:-1],sl0_); loss_=F.cross_entropy(lg_.float().reshape(-1,lg_.shape[-1]),X_[:,1:].reshape(-1))
            opt_.zero_grad(); loss_.backward(); opt_.step(); ls_.append(float(loss_))
            if st_ in ev_:
                model.eval(); cur_.append((st_,np.mean(ls_[-50:]),acc(oneh_,False) if not ctx_ else float('nan'),acc(items(trn_,ctx_),False),acc(items(idt_,ctx_),False),acc(items(ood_,ctx_),False)))
        del opt_; return cur_
    sv_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl0_; _INFM['force']=False
    try:
        res_['W']=run(atom_+[f"So {a_} is connected to {c_}." for a_,b_,c_,_,_ in trn_],False)
        with torch.no_grad(): model.load_state_dict(sd_)
        res_['C']=run([f"{f0_} {f1_} So {a_} is connected to {c_}." for a_,b_,c_,f0_,f1_ in trn_]+[f"{f1_} {f0_} So {a_} is connected to {c_}." for a_,b_,c_,f0_,f1_ in trn_],True)
    finally:
        with torch.no_grad(): model.load_state_dict(sd_)
        try: _ng_zero(model)
        except Exception: pass
        _carry_load(sv_); _XDG['cur']=sc_; _INFM.pop('force',None)
    for k_,nm9_ in (('W','IN THE WEIGHTS (atomic facts of all entities + two-hop statements of half the ID paths)'),('C','IN CONTEXT (worked examples of the trained ID paths; test prompts carry both premises)')):
        print(f"  [inference CEILING LONG | {nm9_}] {NE_} entities, {len(trn_)} trained paths, held out ID {len(idt_)} / OOD {len(ood_)}, AdamW wd {wd_}, chance 0.10 (distractors are trained two-hop answers; in context a decoy fact puts a second candidate name in the prompt) | "
              +" | ".join(f"step {s_}: loss {l_:.2f}"+(f", one-hop {o_:.2f}" if o_==o_ else "")+f", trained {a_:.2f}, ID held-out {b_:.2f}, OOD {c_:.2f}" for s_,l_,o_,a_,b_,c_ in res_[k_]),flush=True)
def _inf_isotropy(model,cfg,S_,ts):
    """cg24 ISOTROPY IN THE TRUNK (LeJEPA's prescription as a sleep phase): a few steps of SIGReg on the trunk's pooled passage codes of stored
    real sentences (scale removed, so only the SHAPE is pulled towards an isotropic Gaussian - no single shared direction dominating), with
    language-model anchors on recent windows; the brake rolls the phase back if the visited domain or old domains get worse by more than
    inf_iso_brake. Readout: effective rank of 256 fixed passages before / after, and the BPC change."""
    vo_=model.__dict__.get('_jx_vo'); it_=S_['it']; ring_=S_.get('ring',[])
    if vo_ is None or len(ring_)<16: return
    vals_,olds_,t_=vo_; sl_=int(ts); dev_=model.embed.weight.device; nb_=int(cfg.pu_cons_eval) or 4; rs_=np.random.RandomState(6161+int(model.__dict__.get('_visit',0)))
    real_=[i_ for i_ in range(len(it_)) if it_[i_].get('k')=='real' and not it_[i_].get('gone') and len(it_[i_]['s'])>=24]
    if len(real_)<256: return
    if not model.__dict__.get('_jxref'): model.__dict__['_jxref']=[it_[real_[int(j_)]]['s'] for j_ in rs_.choice(len(real_),256,replace=False)]
    ref_=model.__dict__['_jxref']; was_=model.training
    def bp_(): return bpc(model,vals_[t_],sl_,n=nb_,skip=int(getattr(cfg,'brake_eval_skip',0))),(float(np.mean([bpc(model,vals_[d_],s_,n=nb_,skip=int(getattr(cfg,'brake_eval_skip',0))) for d_,s_ in olds_])) if olds_ else float('nan'))
    model.eval()
    with torch.no_grad(): r0_,tp0_=_jx_effrank(_jepa_emb(model,ref_))
    v0_,o0_=bp_(); sd_=_snapcpu(model)
    ps_=[q_ for n_,q_ in model.named_parameters() if q_.requires_grad and 'ngram' not in n_]
    op_=torch.optim.Adam(ps_,lr=float(cfg.lr)*float(getattr(cfg,'inf_iso_lr',0.003))); lam_=float(getattr(cfg,'inf_iso_w',1.0)); Ln_=len(ring_[0][0]); ls_=[]
    _XDG_sc_=_XDG['cur']; _XDG['cur']=sl_
    try:
        model.train()
        for _ in range(int(getattr(cfg,'inf_iso_steps',20))):
            tx_=[it_[real_[int(j_)]]['s'] for j_ in rs_.choice(len(real_),min(64,len(real_)),replace=False)]
            z_=_jepa_emb(model,tx_,True); z_=(z_-z_.mean(0))/(z_.pow(2).mean().sqrt()+1e-6)/ (1.0/math.sqrt(z_.shape[1]))**0   # scale removed (one number), shape kept
            z_=(z_-z_.mean(0))/((z_-z_.mean(0)).pow(2).sum(1).mean().sqrt()+1e-6)*math.sqrt(z_.shape[1])                 #   -> mean squared norm = dims (isotropic N(0, I) scale)
            lsg_=_jx_sigreg(z_,nd_=128)
            Xa_=torch.stack([torch.tensor(list(ring_[int(i_)][0][:Ln_]),dtype=torch.long) for i_ in rs_.choice(len(ring_),8,replace=False)]).to(dev_)
            sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); _INFM['force']=False
            try: lg_=model(Xa_[:,:-1],sl_).float()
            finally: _carry_load(sv4_); _INFM.pop('force',None)
            la_=F.cross_entropy(lg_.reshape(-1,lg_.shape[-1]),Xa_[:,1:].reshape(-1)); loss_=lam_*lsg_+la_
            op_.zero_grad(set_to_none=True); loss_.backward(); op_.step(); ls_.append(float(lsg_))
            try: _ng_zero(model)
            except Exception: pass
        model.eval()
        with torch.no_grad(): r1_,tp1_=_jx_effrank(_jepa_emb(model,ref_))
        v1_,o1_=bp_(); bk_=float(getattr(cfg,'inf_iso_brake',getattr(cfg,'inf_cons_brake',0.03)))
        rb_=(v1_-v0_)>bk_ or (olds_ and (o1_-o0_)>bk_)
        if rb_:
            with torch.no_grad(): model.load_state_dict(sd_)
        A_=model.__dict__.setdefault('_isoK',[0,0]); A_[0]+=int(not rb_); A_[1]+=1
        print(f"  [inference ISOTROPY (SIGReg on the trunk)] {len(ls_)} steps at lr x{float(getattr(cfg,'inf_iso_lr',0.003))}, weight {lam_} | SIGReg {ls_[0]:.4f} -> {ls_[-1]:.4f} | effective rank {r0_:.1f} -> {r1_:.1f} of {Z_dim_(model,ref_)}, top direction {100*tp0_:.0f}% -> {100*tp1_:.0f}% | BPC visited {v0_:.4f} -> {v1_:.4f} ({v1_-v0_:+.4f}), old domains {o0_:.4f} -> {o1_:.4f} ({o1_-o0_:+.4f})"+(" | ROLLED BACK" if rb_ else " | KEPT")+f" | kept so far {A_[0]}/{A_[1]}",flush=True)
    finally:
        _XDG['cur']=_XDG_sc_; del op_
        for q_ in model.parameters(): q_.grad=None
        try: _ng_zero(model)
        except Exception: pass
        model.train(was_)
def Z_dim_(model,ref_):
    try: return int(model.__dict__['_jxrefZ'].shape[1])
    except Exception: return -1
def _jd_codes(model,cfg,txts_):
    """encoder input for JEPA dreams: trunk codes with salient pooling (the passage's most surprising bytes - names carry the link) or mean pooling"""
    if str(getattr(cfg,'inf_jd_pool','salient'))=='salient': return _jx_emb_sal(model,txts_,int(getattr(cfg,'inf_jd_k',16)))
    with torch.no_grad(): return _jepa_emb(model,txts_)
def _jd_pairs(S_,rs_,n_=1024):
    """linked VIEWS for training: premise pairs of verified conclusions + pairs of recorded facts that chain through a shared entity (X r Y, Y r Z)"""
    it_=S_['it']; P_=[(it_[c_['src'][0]]['s'],it_[c_['src'][1]]['s']) for c_ in S_.get('conc',[]) if not c_.get('freed') and isinstance(c_.get('src'),tuple)]
    F_=[f_ for f_ in S_.get('facts',[]) if not f_.get('freed') and isinstance(f_.get('i'),int) and f_['i']<len(it_)]
    byA_={}
    for f_ in F_: byA_.setdefault(f_['A'],[]).append(f_)
    ch_=[(f_,g_) for f_ in F_ for g_ in byA_.get(f_['B'],[]) if g_ is not f_ and g_['i']!=f_['i']]
    if len(ch_)>n_: ch_=[ch_[int(j_)] for j_ in rs_.choice(len(ch_),n_,replace=False)]
    P_+=[(it_[f_['i']]['s'],it_[g_['i']]['s']) for f_,g_ in ch_]
    return P_
def _jd_pair_gain(model,cfg,S_,recs_):
    """cg26k: the both-ways gain in bits of each pair, measured as the dream judge measures it (bits of B given A against bits of B given
    baseline sentences, and the reverse; copied common words excluded; the weaker side) but with the baseline sentences FIXED in the record,
    so a measurement at a later sleep is comparable with the one taken when the pair was dreamed"""
    df_=S_.get('df',{}); dm_=int(cfg.inf_df); kp_=(lambda w_: 0<df_.get(w_,0)<=dm_) if int(getattr(cfg,'inf_mix_rarecopy',0)) else None
    res_={}; by_={}
    for i_,r_ in enumerate(recs_):
        if r_.get('rb') and r_.get('ra'): by_.setdefault(int(r_['sl']),[]).append(i_)
    for sl_,ix_ in by_.items():
        jobs_=[]
        for i_ in ix_: r_=recs_[i_]; jobs_+=[(r_['a'],r_['b']),(r_['b'],r_['a'])]+[(x_,r_['b']) for x_ in r_['rb']]+[(x_,r_['a']) for x_ in r_['ra']]
        bb_=_inf_cond_bytes(model,cfg,jobs_,sl_); o_=0
        for i_ in ix_:
            r_=recs_[i_]; n1_,n2_=len(r_['rb']),len(r_['ra']); (tB_,bAB_),(tA_,bBA_)=bb_[o_],bb_[o_+1]
            bRB_=np.mean([bb_[o_+2+j_][1] for j_ in range(n1_)],0); bRA_=np.mean([bb_[o_+2+n1_+j_][1] for j_ in range(n2_)],0); o_+=2+n1_+n2_
            mB_=_mix_mask(r_['a'],tB_,kp_); mA_=_mix_mask(r_['b'],tA_,kp_)
            res_[i_]=min(float((bRB_-bAB_)[mB_].sum()),float((bRA_-bBA_)[mA_].sum()))
    return [res_.get(i_,float('nan')) for i_ in range(len(recs_))]
def _jd_negs(S_,P_,k_,rs_,cfg,extra_=None,nr_=0):
    """cg26f HARD NEGATIVES for the JEPA selector: for each linked pair (A, B), up to k_ stored sentences that share a rare name with A but are
    neither A nor B - the same candidates the name index offers at dream time, so training teaches WHICH same-name sentence fits"""
    it_=S_['it']; ix_=S_.get('idx',{}); df_=S_.get('df',{}); dm_=int(cfg.inf_df); out_=[]
    kn9_={}                                                                     # FIX (review): every KNOWN partner of a sentence (either direction) is excluded, not only
    for a9_,b9_ in P_: kn9_.setdefault(a9_.strip(),set()).add(b9_.strip()); kn9_.setdefault(b9_.strip(),set()).add(a9_.strip())   #   this pair's B
    rare_=lambda w_: len(w_)>=4 and w_[:1].isupper() and w_[1:].isalpha() and w_[1:].islower() and 0<df_.get(w_,0)<=dm_ and df_.get(w_.lower(),0)<=dm_
    RP9_=[i_ for i_,t_ in enumerate(it_) if not t_.get('gone') and t_.get('k') in ('real','ver') and len(t_['s'])>=24] if nr_>0 else []
    sq9_=int(getattr(cfg,'inf_jd_seqneg',0)); TX9_={}; S_['_seqneg']=[0,0]
    if sq9_>0:                                                                  # cg26l SAME-DOCUMENT NEGATIVES: JEPAs latch onto slow features (Sobal et al. 2022) - here the
        for i9_,t9_ in enumerate(it_):                                          #   topic of a document - so the sentences stored next to the cue (same visit and domain:
            if not t9_.get('gone'): TX9_.setdefault(t9_['s'].strip(),i9_)        #   same document, same topic, almost never a real link) are its first negatives
    for a_,b_ in P_:
        nm_={w_.group(0) for w_ in _UWORD.finditer(a_) if rare_(w_.group(0))}; C_=[]
        ex9_=kn9_.get(a_.strip(),set())|{a_.strip(),b_.strip()}
        J9_=[x_ for x_ in ((extra_ or {}).get(a_.strip()) or []) if x_.strip() not in ex9_][:k_]   # cg26j: partners the dream judge REJECTED for this cue come first
        for n_ in nm_:
            for i_ in (ix_.get(n_) or [])[-200:]:
                t_=it_[i_]
                if t_.get('gone') or t_.get('k') not in ('real','ver') or len(t_['s'])<24: continue
                if t_['s'].strip() in ex9_: continue
                C_.append(t_['s'])
        C_=[x_ for x_ in dict.fromkeys(C_) if x_ not in set(J9_)]; kr9_=max(0,k_-len(J9_))
        L9_=J9_+([C_[int(j_)] for j_ in rs_.choice(len(C_),min(kr9_,len(C_)),replace=False)] if C_ and kr9_ else [])
        if nr_>0 and RP9_:                                                      # cg26k RANDOM NEGATIVES as well: with only same-name negatives the selector learns that a
            for _ in range(3*nr_):                                              #   shared name means 'not the partner', and in a wide pool it then avoids shared names
                if len(L9_)>=k_+nr_: break                                      #   (local: jepa-wide picks sharing a name fell 11 -> 10 -> 2 of 48)
                x_=it_[RP9_[rs_.randint(len(RP9_))]]['s']
                if x_.strip() not in ex9_ and x_ not in L9_: L9_.append(x_)
        if sq9_>0:
            i9_=TX9_.get(a_.strip()); NB9_=[]
            if i9_ is not None:
                for j9_ in (i9_+1,i9_-1,i9_+2,i9_-2,i9_+3,i9_-3):
                    if len(NB9_)>=sq9_: break
                    if 0<=j9_<len(it_):
                        t9_=it_[j9_]
                        if not t9_.get('gone') and t9_.get('d')==it_[i9_].get('d') and t9_.get('v',-1)==it_[i9_].get('v',-1) and len(t9_['s'])>=24 and t9_['s'].strip() not in ex9_ and t9_['s'] not in L9_: NB9_.append(t9_['s'])
            L9_=NB9_+L9_; S_['_seqneg'][0]+=int(bool(NB9_)); S_['_seqneg'][1]+=len(NB9_)
        out_.append(L9_)
    return out_
def _jd_train(model,cfg,S_):
    """cg26 JEPA DREAM SELECTOR, trained every sleep (continual): encoder head g on frozen trunk codes, EMA target head, predictor P, squared loss on
    normalised codes (both directions) + SIGReg towards an isotropic Gaussian. Codes are standardised and the top inf_jd_drop shared directions removed.
    Readout: held-out linked pairs - does P(g(A)) retrieve B among all held-out partners? (chance 1/n)"""
    rs_=np.random.RandomState(7171+int(model.__dict__.get('_visit',0))); P2_=_jd_pairs(S_,rs_); njp_=0; jrej_={}
    if int(getattr(cfg,'inf_jd_judge',0)):                                     # cg26j TRAINED ON WHAT THE DREAM JUDGE ACCEPTS: splice pairs that passed the both-ways compression
        JG_=S_.get('jd_judged',[]); seen9_={(a_.strip(),b_.strip()) for a_,b_ in P2_}   #   test become linked training pairs, and the partners it REJECTED for the same cue
        nlk_=0; lk9_=int(getattr(cfg,'inf_jd_judge_link',1))
        for e9_ in JG_:                                                         #   become its first hard negatives (what makes a splice useful, not only a shared name);
            a_,b_,ok_,src_=e9_[:4]; ln9_=bool(e9_[4]) if len(e9_)>4 else True   #   sequence pairs (same document) are left out; FIX: an accepted pair counts as a positive
            if ok_ and lk9_ and not ln9_: nlk_+=1; continue                     #   only with a sign of a real link (a shared rare name) - most accepted pairs are only on the
            if ok_ and src_!='sequence' and (a_.strip(),b_.strip()) not in seen9_ and len(a_)>=24 and len(b_)>=24:   #   same topic, and training on them would teach topic
                P2_.append((a_,b_)); seen9_.add((a_.strip(),b_.strip())); njp_+=1
            elif not ok_ and src_ not in ('control+','sequence'): jrej_.setdefault(a_.strip(),[]).append(b_)
        for k9_ in jrej_: jrej_[k9_]=list(dict.fromkeys(jrej_[k9_]))[-16:]
    nop_=0; non_=0; omsg_=''
    if int(getattr(cfg,'inf_jd_outcome',0)):                                   # cg26k TRAINED ON MEASURED OUTCOMES: a pair dreamed at an earlier sleep is measured again now (same
        vis9_=int(model.__dict__.get('_visit',0)); PD_=S_.get('jd_pend',[])     #   test, same baseline sentences). The change in its both-ways gain is what the dreaming taught
        due_=[r_ for r_ in PD_ if r_['v']<vis9_][:int(getattr(cfg,'inf_jd_out_n',48))*2]   #   the weights about that link, lasting over a visit of training: a pair whose gain rose by
        if due_:                                                                #   inf_jd_out_pos bits becomes a training pair (no shared name needed - the outcome is the
            try:                                                                #   evidence), one whose gain did not rise becomes a hard negative for its cue
                g1_=_jd_pair_gain(model,cfg,S_,due_); OU_=S_.setdefault('jd_out',[]); n09_=len(OU_)
                for r_,g9_ in zip(due_,g1_):
                    if np.isfinite(g9_) and np.isfinite(r_.get('g0',float('nan'))): OU_.append((r_['a'],r_['b'],float(g9_-r_['g0']),r_['src'],r_['v']))
                ds9_=[(x_[2],x_[3]) for x_ in OU_[n09_:]]; del OU_[:-3000]; dn9_={id(r_) for r_ in due_}; S_['jd_pend']=[r_ for r_ in PD_ if id(r_) not in dn9_]
                if ds9_:
                    bys9_={}
                    for d9_,s9_ in ds9_: bys9_.setdefault(s9_,[]).append(d9_)
                    omsg_=(f"outcomes measured for {len(ds9_)} pairs dreamed at earlier sleeps: gain change median {np.median([d_ for d_,_ in ds9_]):+.1f} bits, rose by "
                           f"{float(getattr(cfg,'inf_jd_out_pos',1.0)):g}+ bits {np.mean([d_>=float(getattr(cfg,'inf_jd_out_pos',1.0)) for d_,_ in ds9_]):.2f} | by source: "
                           +", ".join(f"{k_} {np.median(v_):+.1f} (n {len(v_)})" for k_,v_ in sorted(bys9_.items())))
            except Exception as e9_: omsg_=f"outcome measurement unavailable ({type(e9_).__name__}: {e9_})"[:160]
        op9_=float(getattr(cfg,'inf_jd_out_pos',1.0)); seen8_={(a_.strip(),b_.strip()) for a_,b_ in P2_}
        OD9_=[x_[2] for x_ in S_.get('jd_out',[])]
        if int(getattr(cfg,'inf_jd_out_rel',0)) and len(OD9_)>=12:             # FIX (local cg26j: median change -0.2 bits, none rose by 1 bit - an absolute bar turned EVERY
            hi9_=max(0.25,float(np.quantile(OD9_,2/3))); lo9_=min(-1.0,float(np.quantile(OD9_,1/3)))   #   dreamed pair, judge-accepted links included, into a negative): RELATIVE labels -
        else: hi9_,lo9_=op9_,0.0                                                #   the top third that also rose (>= 0.25 bits) are positives, the bottom third that fell
        for a_,b_,d9_,src_,_ in S_.get('jd_out',[]):                            #   by 1+ bit are negatives, the rest teach nothing
            if d9_>=hi9_ and (a_.strip(),b_.strip()) not in seen8_ and len(a_)>=24 and len(b_)>=24: P2_.append((a_,b_)); seen8_.add((a_.strip(),b_.strip())); nop_+=1
            elif d9_<=lo9_: jrej_.setdefault(a_.strip(),[]).insert(0,b_); non_+=1
        for k9_ in jrej_: jrej_[k9_]=list(dict.fromkeys(jrej_[k9_]))[:16]
    ne9_=int(getattr(cfg,'inf_jd_entpairs',0)); nep_=0
    if ne9_>0:                                                                  # cg26l ENTITY-SHARING FACT PAIRS: two stored fact sentences with the same subject or object
        it9_=S_['it']; by9_={}; seen7_={(a_.strip(),b_.strip()) for a_,b_ in P2_}   #   are two views of one entity (more linked pairs: the selector is data-limited -
        for f9_ in S_.get('facts',[]):                                          #   about 390 pairs for an effective input of about 128 dimensions)
            if f9_.get('freed') or not isinstance(f9_.get('i'),int) or f9_['i']>=len(it9_) or it9_[f9_['i']].get('gone'): continue
            for n9_ in (f9_['A'],f9_['B']): by9_.setdefault(n9_,[]).append(f9_['i'])
        ex9_=[]
        for n9_,L9_ in by9_.items():
            L9_=sorted(set(L9_))[-12:]
            for x9_ in range(len(L9_)):
                for y9_ in range(x9_+1,len(L9_)): ex9_.append((it9_[L9_[x9_]]['s'],it9_[L9_[y9_]]['s']))
        for j9_ in rs_.permutation(len(ex9_)):
            if nep_>=ne9_: break
            a9_,b9_=ex9_[int(j9_)]
            if a9_.strip()!=b9_.strip() and (a9_.strip(),b9_.strip()) not in seen7_ and (b9_.strip(),a9_.strip()) not in seen7_: P2_.append((a9_,b9_)); seen7_.add((a9_.strip(),b9_.strip())); nep_+=1
    pn_={n_.encode('latin-1') for c_ in inf_probes(cfg) for n_ in c_['names']}
    isp_=lambda pr_: any(w_.group(0) in pn_ for w_ in _UWORD.finditer(pr_[0]+b' '+pr_[1]))
    npl0_=sum(1 for pr_ in P2_ if isp_(pr_)); bal_=''
    if int(getattr(cfg,'inf_jd_real',0)):                                       # cg26b REAL-TEXT BALANCE: planted probe pairs capped at the real-text count (min = training minimum) - cg26 GPU
        Pr_=[pr_ for pr_ in P2_ if not isp_(pr_)]; Pp_=[pr_ for pr_ in P2_ if isp_(pr_)]   #   trained on 141 planted / 63 real but dreams over real text only
        cap_=max(int(getattr(cfg,'inf_jd_min',32)),len(Pr_))                     #   (floor = the training minimum, so a small real-text count never starves training)
        if int(getattr(cfg,'inf_jd_plantw',0)): bal_=f", planted kept and WEIGHTED (equal total weight to {len(Pr_)} real-text pairs)" if Pp_ and Pr_ else ''   # cg26l: weighted, not capped
        elif len(Pp_)>cap_: Pp_=[Pp_[int(j_)] for j_ in rs_.choice(len(Pp_),cap_,replace=False)]; bal_=f", planted capped at {cap_}"
        P2_=Pr_+Pp_
    if len(P2_)<int(getattr(cfg,'inf_jd_min',32)): print(f"  [inference JEPA DREAMS | TRAIN] {len(P2_)} linked pairs (need {int(getattr(cfg,'inf_jd_min',32))})",flush=True); return
    import zlib as _zl9                                                         # FIX: a FIXED held-out set - a pair is held out for ever if the hash of its two sentences
    hk_=lambda pr_: _zl9.crc32(b'\x00'.join(sorted((pr_[0].strip(),pr_[1].strip()))))%5==0   #   (either order) falls in 1 of 5 buckets. The split was re-drawn every
    H_=[pr_ for pr_ in P2_ if hk_(pr_)]; T_=[pr_ for pr_ in P2_ if not hk_(pr_)]  #   sleep while the heads persist, so held-out pairs had been trained on
    rs_.shuffle(H_); rs_.shuffle(T_); P2_=H_+T_; nh_=len(H_)                   #   in earlier sleeps and the readouts were optimistic
    if len(T_)<16: print(f"  [inference JEPA DREAMS | TRAIN] {len(T_)} training pairs after the fixed held-out split (need 16)",flush=True); return
    hr_=[j_ for j_,pr_ in enumerate(H_) if not isp_(pr_)]                       # held-out REAL-TEXT pairs (retrieval among themselves)
    hn_=int(getattr(cfg,'inf_jd_hn',0)); NI_=None; NGT_=None; fl_=[]; nrn_=int(getattr(cfg,'inf_jd_negrand',0)) if hn_>0 else 0
    if hn_>0:                                                                   # cg26f: hard negatives (name-sharing, unlinked) for every pair
        try:
            NGT_=_jd_negs(S_,P2_,hn_,np.random.RandomState(9191+int(model.__dict__.get('_visit',0))),cfg,jrej_,nrn_); fl_=[t_ for L_ in NGT_ for t_ in L_]
            if fl_:
                NI_=torch.full((len(P2_),max(len(L_) for L_ in NGT_)),-1,dtype=torch.long); o_=0   # (cg26l: same-document negatives can widen the rows)
                if int(getattr(cfg,'inf_jd_seqneg',0)): print(f"  [inference JEPA DREAMS | SAME-DOCUMENT NEGATIVES] {S_['_seqneg'][0]} of {len(P2_)} training pairs got {S_['_seqneg'][1]} neighbouring sentences of the cue as first negatives",flush=True)
                for j_,L_ in enumerate(NGT_):
                    for h_ in range(len(L_)): NI_[j_,h_]=o_; o_+=1
            else: hn_=0
        except Exception as e_: print(f"  [inference JEPA DREAMS | TRAIN] hard negatives unavailable ({type(e_).__name__}: {e_})"[:200],flush=True); hn_=0
    ZA_=_jd_codes(model,cfg,[a_ for a_,_ in P2_]); ZB_=_jd_codes(model,cfg,[b_ for _,b_ in P2_]); kd_=int(getattr(cfg,'inf_jd_drop',3))
    sm9_=float(getattr(cfg,'inf_jd_statm',0.0))
    if sm9_>0:                                                                  # cg26j RUNNING INPUT STATISTICS: mean, variance and covariance are averaged across sleeps
        Ztr_=torch.cat([ZA_[nh_:],ZB_[nh_:]]).double(); m1_=Ztr_.mean(0); S1_=(Ztr_.T@Ztr_)/len(Ztr_)   #   (the heads persist, so their input frame must not jump each sleep), from the
        RS9_=model.__dict__.get('_jdstat')                                      #   TRAINING pairs only (the held-out pairs stay unseen); averaged as RAW mean and raw second
        if RS9_ is None or RS9_[0].shape!=m1_.shape: mR_,SR_=m1_,S1_            #   moment, so every sleep is in one frame (FIX: the covariance of standardised codes mixed
        else: mR_=sm9_*RS9_[0]+(1-sm9_)*m1_; SR_=sm9_*RS9_[1]+(1-sm9_)*S1_     #   each sleep's own standardisation)
        model.__dict__['_jdstat']=(mR_.detach(),SR_.detach())
        Cr_=SR_-torch.outer(mR_,mR_); va_=Cr_.diagonal().clamp(min=0); sd_=(va_.sqrt()+1e-4)
        Cv_=Cr_/torch.outer(sd_,sd_); mu_=mR_.float(); sd_=sd_.float()
        ev9_=torch.linalg.eigh(Cv_)[1].flip(1)                                  # eigenvectors, largest first
        V_=ev9_[:,:kd_].T.float() if kd_>0 else None
        if int(getattr(cfg,'inf_jd_pca',0))>0: V_=ev9_[:,kd_:kd_+int(cfg.inf_jd_pca)].T.float()   # cg26l: PROJECT on the next K principal directions instead
    else:
        Zal_=torch.cat([ZA_,ZB_]); mu_=Zal_.mean(0); sd_=Zal_.std(0)+1e-4; X_=(Zal_-mu_)/sd_
        Vt9_=torch.linalg.svd(X_-X_.mean(0),full_matrices=False)[2]; V_=Vt9_[:kd_] if kd_>0 else None
        if int(getattr(cfg,'inf_jd_pca',0))>0: V_=Vt9_[kd_:kd_+int(cfg.inf_jd_pca)]
    pca9_=int(getattr(cfg,'inf_jd_pca',0))>0; model.__dict__['_jdpca']=pca9_
    def enc_(Z_):
        Y_=(Z_-mu_)/sd_
        if pca9_: return Y_@V_.T
        return Y_-(Y_@V_.T)@V_ if V_ is not None else Y_
    model.__dict__['_jdenc']=(mu_.detach(),sd_.detach(),None if V_ is None else V_.detach())
    D_=int(V_.shape[0]) if pca9_ else ZA_.shape[1]; d_=int(getattr(cfg,'inf_jd_dim',128)); SH_=model.__dict__.get('_jdS')
    if SH_ is None or SH_[0].in_features!=D_:
        import copy as _cp
        g_=torch.nn.Linear(D_,d_).to(ZA_.device); gt_=_cp.deepcopy(g_); Pq_=torch.nn.Sequential(torch.nn.Linear(d_,d_),torch.nn.GELU(),torch.nn.Linear(d_,d_)).to(ZA_.device)
        for q_ in gt_.parameters(): q_.requires_grad_(False)
        SH_=(g_,gt_,Pq_,torch.optim.Adam(list(g_.parameters())+list(Pq_.parameters()),lr=1e-3)); model.__dict__['_jdS']=SH_
    g_,gt_,Pq_,op_=SH_; nT_=len(T_); xa_=enc_(ZA_[nh_:]); xb_=enc_(ZB_[nh_:]); ha_=enc_(ZA_[:nh_]); hb_=enc_(ZB_[:nh_])
    obj_=str(getattr(cfg,'inf_jd_obj','hybrid')); model.__dict__['_jdobj']=obj_
    if obj_=='lejepa': qf_=lambda x_: g_(x_); kf_=lambda x_: g_(x_)            # cg26k LeJEPA OBJECTIVE: one embedding for both views, no predictor, no EMA teacher
    else: qf_=lambda x_: Pq_(g_(x_)); kf_=lambda x_: gt_(x_)                   #   (hybrid: predictor on the online head against the EMA target, as before)
    if hn_>0:
        xn_=enc_(_jd_codes(model,cfg,fl_)); NI_=NI_.to(xn_.device); tau_=float(getattr(cfg,'inf_jd_tau',0.1)); whn_=float(getattr(cfg,'inf_jd_hnw',0.5))
    def heldhn_():                                                              # held-out: does the JEPA pick the partner among it + its name-sharing negatives?
        if hn_<=0 or nh_<1: return float('nan'),0,float('nan')
        ok_=[];ch_=[]
        with torch.no_grad():
            for j_ in range(nh_):
                v_=[int(x_) for x_ in NI_[j_].tolist() if x_>=0]
                if not v_: continue
                q_=F.normalize(qf_(ha_[j_:j_+1]),dim=-1); K_=F.normalize(kf_(torch.cat([hb_[j_:j_+1],xn_[v_]])),dim=-1)
                ok_.append(float((q_@K_.T).argmax(1).item()==0)); ch_.append(1.0/(1+len(v_)))
        return (float(np.mean(ok_)) if ok_ else float('nan')),len(ok_),(float(np.mean(ch_)) if ch_ else float('nan'))
    def held_():
        if nh_<4: return float('nan')
        with torch.no_grad():
            q_=F.normalize(qf_(ha_),dim=-1); k_=F.normalize(kf_(hb_),dim=-1); q2_=F.normalize(qf_(hb_),dim=-1); k2_=F.normalize(kf_(ha_),dim=-1)
            ar_=torch.arange(len(ha_),device=ha_.device)
            return 0.5*(float((q_@k_.T).argmax(1).eq(ar_).float().mean())+float((q2_@k2_.T).argmax(1).eq(ar_).float().mean()))
    def heldr_():
        if len(hr_)<4: return float('nan')
        with torch.no_grad():
            ix_=torch.as_tensor(hr_,device=ha_.device); a8_,b8_=ha_[ix_],hb_[ix_]; ar_=torch.arange(len(hr_),device=ha_.device)
            q_=F.normalize(qf_(a8_),dim=-1); k_=F.normalize(kf_(b8_),dim=-1); q2_=F.normalize(qf_(b8_),dim=-1); k2_=F.normalize(kf_(a8_),dim=-1)
            return 0.5*(float((q_@k_.T).argmax(1).eq(ar_).float().mean())+float((q2_@k2_.T).argmax(1).eq(ar_).float().mean()))
    hh0_=heldhn_(); h0_=held_(); hr0_=heldr_(); lam_=float(getattr(cfg,'inf_jd_sigreg',1.0)); ema_=float(getattr(cfg,'inf_jd_ema',0.99)); ls_=[]
    lmode_=str(getattr(cfg,'inf_jd_sigreg_mode','old')); llam_=float(getattr(cfg,'inf_jd_lambda',0.05)); sgl_=[]
    tid_={}; TB_=torch.as_tensor([tid_.setdefault(b9_.strip(),len(tid_)) for _,b9_ in T_],device=ZA_.device)   # partner text ids (same text = same partner)
    TA_=torch.as_tensor([tid_.setdefault(a9_.strip(),len(tid_)) for a9_,_ in T_],device=ZA_.device)   # cue text ids in the same numbering
    KL_={(tid_[a9_.strip()],tid_[b9_.strip()]) for a9_,b9_ in P2_ if a9_.strip() in tid_ and b9_.strip() in tid_}; KL_|={(y_,x_) for x_,y_ in KL_}
    nT9_=max(1,len(tid_)); KLc_=torch.as_tensor(sorted(x_*nT9_+y_ for x_,y_ in KL_),dtype=torch.long,device=ZA_.device)   # every known link (both directions), encoded
    es_=int(getattr(cfg,'inf_jd_es',0)); vb_=[j_ for j_,pr_ in enumerate(T_) if _zl9.crc32(b'\x00'.join(sorted((pr_[0].strip(),pr_[1].strip()))))%5==1] if es_ else []
    if len(vb_)<8 or len(T_)-len(vb_)<16: vb_=[]
    trn_=np.array([j_ for j_ in range(nT_) if j_ not in set(vb_)]); vbt_=torch.as_tensor(vb_,dtype=torch.long,device=ZA_.device)
    pw9_=None
    if int(getattr(cfg,'inf_jd_plantw',0)) and int(getattr(cfg,'inf_jd_real',0)):   # cg26l: batches drawn with planted and real-text pairs at equal total weight
        ip9_=np.array([isp_(T_[j_]) for j_ in trn_],dtype=bool); npp_,nrr_=int(ip9_.sum()),int((~ip9_).sum())
        if npp_ and nrr_: w9_=np.where(ip9_,0.5/npp_,0.5/nrr_); pw9_=w9_/w9_.sum()
    wd_=float(getattr(cfg,'inf_jd_wd',0.0)); best_=(float('inf'),None,0); bad_=0; stp_=0
    def vloss_():                                                               # validation loss: the prediction loss on the validation pairs (both directions)
        with torch.no_grad():
            a9_,b9_=xa_[vbt_],xb_[vbt_]
            if obj_=='lejepa':                                                  # FIX: under LeJEPA the validation score is the whole objective (prediction + SIGReg, fixed
                za9_,zb9_=g_(a9_),g_(b9_); gv_=torch.Generator().manual_seed(4242)   #   directions so checks are comparable) - the prediction term alone grows as SIGReg spreads
                return float((1-llam_)*((za9_-zb9_)**2).mean(-1).mean()/4+llam_*(_sigreg_le(za9_,gen_=gv_)+_sigreg_le(zb9_,gen_=torch.Generator().manual_seed(4243)))/2)   #   the codes, so it froze training
            return float(((F.normalize(Pq_(g_(a9_)),dim=-1)-F.normalize(gt_(b9_),dim=-1))**2).sum(-1).mean()+((F.normalize(Pq_(g_(b9_)),dim=-1)-F.normalize(gt_(a9_),dim=-1))**2).sum(-1).mean())/2
    snap_=lambda: [{k9_:v9_.detach().clone() for k9_,v9_ in m9_.state_dict().items()} for m9_ in (g_,gt_,Pq_)]
    v00_=vloss_() if vb_ else float('nan')
    if vb_: best_=(v00_,snap_(),0)                                              # the heads as they came in are a candidate too (training may not help this sleep)
    tol9_=float(getattr(cfg,'inf_jd_es_tol',0.02)); mn9_=v00_ if vb_ else float('inf')
    with torch.enable_grad():
        for stp_ in range(1,int(getattr(cfg,'inf_jd_steps',300))+1):
            b_=torch.as_tensor(rs_.choice(trn_,min(128,len(trn_),int((pw9_>0).sum()) if pw9_ is not None else len(trn_)),replace=False,p=pw9_),device=ZA_.device); a1_,b1_=xa_[b_],xb_[b_]   # FIX (review): no repeats in a batch
            if obj_=='lejepa':                                                  # cg26k LeJEPA: centre of the two views mu = (zA + zB)/2, prediction loss = mean over views and
                za_=g_(a1_); zb_=g_(b1_); lp_=2*((za_-zb_)**2).mean(-1).mean()/4  #   dimensions of (mu - z)^2 (= (zA - zB)^2 / 4; x2 here so that lp_/2 below is that loss);
                sg_=(_sigreg_le(za_)+_sigreg_le(zb_))/2; loss_=(1-llam_)*lp_/2+llam_*sg_; sgl_.append(float(sg_.detach()))   #   SIGReg on each view, averaged
            else:
                lp_=((F.normalize(Pq_(g_(a1_)),dim=-1)-F.normalize(gt_(b1_),dim=-1))**2).sum(-1).mean()+((F.normalize(Pq_(g_(b1_)),dim=-1)-F.normalize(gt_(a1_),dim=-1))**2).sum(-1).mean()
                if lmode_=='lejepa':                                            # cg26j LeJEPA weighting: (1 - lambda) x prediction + lambda x SIGReg (statistic scaled by N),
                    sg_=(_sigreg_le(g_(a1_))+_sigreg_le(g_(b1_)))/2; loss_=(1-llam_)*lp_/2+llam_*sg_; sgl_.append(float(sg_.detach()))   #   FIX: per view, averaged (pooled doubled N)
                else: loss_=lp_/2+lam_*_jx_sigreg(g_(torch.cat([a1_,b1_])))
            if hn_>0:                                                           # cg26f contrastive term: partner vs its name-sharing negatives vs other partners in the batch
                qh_=F.normalize(qf_(a1_),dim=-1); kp_=F.normalize(kf_(b1_),dim=-1); Ni_=NI_[b_+nh_]; vm_=Ni_>=0
                kn_=F.normalize(kf_(xn_[Ni_.clamp_min(0)]),dim=-1); tb9_=TB_[b_]
                ta9_=TA_[b_]; cd9_=ta9_[:,None]*nT9_+tb9_[None,:]                  # FIX (review): an in-batch partner is not a negative when it has the same text as the
                same9_=(tb9_[:,None]==tb9_[None,:])|(ta9_[:,None]==ta9_[None,:])|torch.isin(cd9_,KLc_)   #   positive, belongs to the same cue text, or is a known link of the cue
                lg_=torch.cat([(qh_*kp_).sum(-1,keepdim=True),torch.einsum('bd,bhd->bh',qh_,kn_).masked_fill(~vm_,float('-inf')),
                               (qh_@kp_.T).masked_fill(same9_,float('-inf'))],1)/tau_
                loss_=loss_+whn_*F.cross_entropy(lg_,torch.zeros(len(b_),dtype=torch.long,device=qh_.device))
            op_.zero_grad(); loss_.backward(); op_.step(); ls_.append(float(lp_.detach())/2)
            with torch.no_grad():
                if wd_>0:                                                       # cg26j decoupled weight decay (AdamW style) on the online heads
                    for q_ in list(g_.parameters())+list(Pq_.parameters()): q_.mul_(1.0-1e-3*wd_)
                if obj_!='lejepa':
                    for p_,q_ in zip(gt_.parameters(),g_.parameters()): p_.mul_(ema_).add_(q_,alpha=1-ema_)
            if vb_ and stp_%20==0:                                              # cg26j EARLY STOPPING on a validation split of the training pairs (hash bucket 1 of 5;
                vl_=vloss_(); mn9_=min(mn9_,vl_)                                #   the held-out readout, bucket 0, stays untouched). FIX: the LATEST heads within tol of the
                if vl_<=mn9_*(1+tol9_): best_=(vl_,snap_(),stp_); bad_=0        #   best validation loss so far are kept (new pairs are learned unless that costs the
                else:                                                           #   validation pairs more than tol - the first version froze the selector after sleep 1);
                    bad_+=1                                                     #   stop after 3 checks outside it
                    if bad_>=3: break
    if vb_ and best_[1] is not None:
        for m9_,sd9_ in zip((g_,gt_,Pq_),best_[1]): m9_.load_state_dict(sd9_)
    h1_=held_(); hr1_=heldr_(); hh1_=heldhn_(); tfh_=float('nan')
    if hn_>0:                                                                   # baseline: TF-IDF on names (what the hybrid ranks by) on the same held-out choices
        try:
            ok9_=[]
            for j_ in range(nh_):
                L_=NGT_[j_]
                if not L_: continue
                sm_=_tfidf_sim(S_,[H_[j_][0]],[H_[j_][1]]+L_,int(cfg.inf_df),True); ok9_.append(float(int(sm_[0].argmax())==0))
            tfh_=float(np.mean(ok9_)) if ok9_ else float('nan')
        except Exception: pass
    with torch.no_grad(): er_,_=_jx_effrank(g_(torch.cat([xa_,xb_])))
    npb_=sum(1 for pr_ in P2_ if isp_(pr_))   # how many training pairs are planted probe facts vs real text
    model.__dict__['_jdN']=model.__dict__.get('_jdN',0)+1
    if True:
        print(f"  [inference JEPA DREAMS | TRAIN SETUP] "+(f"early stopping on {len(vb_)} validation pairs (tolerance {tol9_:.0%}): loss {v00_:.3f} -> kept {best_[0]:.3f} at step {best_[2]} of {stp_} | " if vb_ else "")
              +(f"LeJEPA SIGReg (N x integral, 17 knots on [-5, 5], 256 directions), lambda {llam_:g}: statistic {sgl_[0]:.2f} -> {sgl_[-1]:.2f} | " if sgl_ else "")
              +(f"OBJECTIVE {obj_}{' (one embedding, no predictor, no EMA teacher, prediction to the centre of the two views)' if obj_=='lejepa' else ''} | " )
              +(f"OUTCOME-trained: {nop_} pairs whose gain rose added, {non_} whose gain fell as hard negatives" +(f"; {omsg_}" if omsg_ else "")+" | " if int(getattr(cfg,'inf_jd_outcome',0)) else "")
              +(f"random negatives {nrn_} per pair | " if nrn_ else "")
              +(f"judge-trained: {njp_} accepted splice pairs added ({nlk_} accepted without a shared rare name left out), rejected partners as hard negatives for {len(jrej_)} cues | " if int(getattr(cfg,'inf_jd_judge',0)) else "")
              +(f"running input statistics (momentum {sm9_:g}, training pairs only)" if sm9_>0 else ""),flush=True)
    print(f"  [inference JEPA DREAMS | TRAIN] sleep {model.__dict__['_jdN']}: {nT_} linked training pairs ({len([1 for c_ in S_.get('conc',[]) if isinstance(c_.get('src'),tuple)])} verified premises + fact chains through a shared entity), "
          f"of which planted probe facts {npb_} and real text {len(P2_)-npb_}, {str(getattr(cfg,'inf_jd_pool','salient'))} pooling, top {kd_} directions removed | loss {ls_[0]:.3f} -> {ls_[-1]:.3f}, code effective rank {er_:.1f} of {d_} | HELD-OUT linked pairs, fixed set ({nh_}, chance {1/max(1,nh_):.3f}): partner found {h0_:.2f} -> {h1_:.2f}"
          +(f" | REAL-TEXT held-out ({len(hr_)}, chance {1/len(hr_):.3f}): {hr0_:.2f} -> {hr1_:.2f}" if len(hr_)>=4 else f" | REAL-TEXT held-out: {len(hr_)} pairs (too few)")
          +(f" | of all pairs before balancing, planted {npl0_}{bal_}" if bal_ else "")+(f" | ENTITY-SHARING fact pairs added {nep_}" if ne9_>0 else "")+(f" | input PROJECTED on {D_} principal directions" if pca9_ else "")
          +(f" | HARD NEGATIVES (partner vs up to {int(getattr(cfg,'inf_jd_hn',0))} name-sharing sentences, {hh1_[1]} held-out choices, chance {hh1_[2]:.2f}): JEPA {hh0_[0]:.2f} -> {hh1_[0]:.2f}, TF-IDF names {tfh_:.2f}" if hn_>0 else ""),flush=True)
def _time_w(model,cfg,it_,Qi_,Ci_):
    """cg26h TIME WEIGHTING for NREM partners (after Cai et al. 2016 / Rashid et al. 2016: memories encoded close together in time are linked):
    a weight in [0,1] for each cue x candidate from their arrival in the store. Same visit: nothing within inf_mix_time_near stored sentences (the
    same document - daytime training already links those), then 2^-(distance beyond that / half-life), half-life = inf_mix_time_frac x the sentences
    stored in that visit. Different visits: inf_mix_time_floor x 2^-(visit gap - 1). Candidates without a store position (dream-bank traces) get 0.
    inf_mix_time_shuffle: the control - every sentence gets the timestamp of a random other one (same distribution of weights, no real timing)."""
    nr_=int(getattr(cfg,'inf_mix_time_near',20)); fr_=float(getattr(cfg,'inf_mix_time_frac',0.1)); fl_=float(getattr(cfg,'inf_mix_time_floor',0.25))
    pos_=list(range(len(it_)))
    if int(getattr(cfg,'inf_mix_time_shuffle',0)):
        k_=('_tperm',len(it_)); pm_=model.__dict__.get('_tperm')
        if pm_ is None or pm_[0]!=k_: pm_=(k_,np.random.RandomState(7171+len(it_)).permutation(len(it_)).tolist()); model.__dict__['_tperm']=pm_
        pos_=pm_[1]
    nv_={}
    for x_ in it_: v_=x_.get('v'); nv_[v_]=nv_.get(v_,0)+1
    W_=np.zeros((len(Qi_),len(Ci_)),dtype=np.float32)
    for a_,q_ in enumerate(Qi_):
        if q_ is None or q_<0: continue
        pq_=pos_[q_]; vq_=it_[pq_].get('v')
        if vq_ is None: continue
        for b_,j_ in enumerate(Ci_):
            if j_ is None or j_<0 or j_==q_: continue
            pj_=pos_[j_]; vj_=it_[pj_].get('v')
            if vj_ is None: continue
            if vj_==vq_:
                d_=abs(pq_-pj_)
                if d_>=nr_: W_[a_,b_]=2.0**(-(d_-nr_)/max(float(nr_),fr_*nv_.get(vq_,1)))
            else: W_[a_,b_]=fl_*2.0**(-(abs(int(vq_)-int(vj_))-1))
    return W_
def _jd_scores(model,cfg,qt_,ct_):
    """JEPA selector: predicted partner code of each cue against the target codes of the candidates"""
    SH_=model.__dict__.get('_jdS'); E_=model.__dict__.get('_jdenc')
    if SH_ is None or E_ is None: return None
    g_,gt_,Pq_,_=SH_; mu_,sd_,V_=E_
    def enc_(Z_):
        Y_=(Z_-mu_)/sd_
        if model.__dict__.get('_jdpca') and V_ is not None: return Y_@V_.T
        return Y_-(Y_@V_.T)@V_ if V_ is not None else Y_
    lj_=model.__dict__.get('_jdobj')=='lejepa'
    with torch.no_grad():
        if lj_: q_=F.normalize(g_(enc_(_jd_codes(model,cfg,qt_))),dim=-1); k_=F.normalize(g_(enc_(_jd_codes(model,cfg,ct_))),dim=-1)
        else: q_=F.normalize(Pq_(g_(enc_(_jd_codes(model,cfg,qt_)))),dim=-1); k_=F.normalize(gt_(enc_(_jd_codes(model,cfg,ct_))),dim=-1)
    return q_@k_.T

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    'run': 'platypus.runner',
}
