"""Merged-memory (superposed-memory) dreams with an optional contrastive chaining loss; off unless inf_sup > 0.

Each sleep, after the closed-form edits: for chains A-B-C the system concluded itself (never ground truth; chains held out by inf_kgx_hold
are skipped), the two source sentences are read SEPARATELY into each delta block's memory and the two memories are ADDED. The trunk is then
trained to answer 'So A is connected to C.' from that superposed memory (store off), with plain-text anchor rows, under a brake that rolls
the phase back if the visited or old domains get worse by more than inf_sup_brake.
Options: inf_sup_lr_abs (absolute rate; 0 = lr x inf_sup_lr), inf_sup_steps, inf_sup_rows, inf_sup_brake; inf_sup_retry N retries a
rolled-back phase up to N times from the same weights, each time at the rate / inf_sup_retry_div (default 3); inf_sup_wgrad 1 keeps the
gradient through the premise reads (so premises can be WRITTEN in a chainable form); inf_sup_params 'kv' trains only the delta blocks' key
and value projections (all weights otherwise); inf_sup_chain > 0 adds a CHAINING loss (the value the A-B memory returns for A's question,
M_AB k_q, should match the key B produces as a subject, k_B): inf_sup_chain_mode 'nce' (contrastive among the batch's bridges, temperature
inf_sup_chain_tau) or 'cos'.
Readout per sleep, before and after, store off: two-hop from the superposed memory, from no memory, from the B-C premise's memory alone,
from a random memory (another chain's premises) and with the premises as text; the same rows for up to inf_sup_ho_n (64) steps of chains
held out by inf_kgx_hold, which are never concluded or trained (HELD-OUT CHAINS line: transfer of the skill, each sleep); a one-hop control (premise prefix -> B from that premise's memory against none); the chaining
margin and own-bridge retrieval. A training step that runs out of GPU memory is retried with half the rows.
The brake is measured in eager mode before and after (the compiled-forward offset is printed); inf_sup_anc_visit F takes that share of the
anchor rows from the visited slot (off by default); inf_sup_onehop_w W adds one-hop rows (premise prefix -> B from that premise's
memory) at weight W; inf_sup_protect 1|2 whitens each key/value step by every domain's Fisher factors, the visited domain's included
(1 keeps the step's norm, 2 keeps the whitened size)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from platypus.private import (
    _XDG,
)
from platypus.layers import (
    _CARRY, _CHN,
)
from platypus.ngrams import (
    _COMPILED, _opt_params,
)
from platypus.protection import (
    _kf_apply, kf_build,
)
from platypus.conduit import (
    _carry_load, _carry_save, _uh_load, _uh_save,
)
from platypus.inference.store import (
    _INFM,
)
from platypus.inference.units import (
    _UWORD,
)
from platypus.inference.probes import (
    inf_probes,
)
from platypus.inference.infer import (
    _kgx_ho, _kgx_hold,
)
from platypus.inference.closed_form import (
    _ws_batch, _ws_bpc, _ws_eager, _ws_run,
)

def _ctx_mem_state(model,mems_,slot_,grad_=False):
    """each block's delta-rule memory after reading each memory text on its own (left-padded with spaces, eval mode; no gradient unless grad_)"""
    L_=max(len(m_) for m_ in mems_); dev_=model.embed.weight.device
    X_=torch.full((len(mems_),L_),32,dtype=torch.long,device=dev_)
    for j_,m_ in enumerate(mems_): X_[j_,L_-len(m_):]=torch.tensor(list(m_),dtype=torch.long,device=dev_)
    sv_=_carry_save(); was_=model.training; fs_=model.feat_scale.clone(); s0_=model.__dict__.get('_s0_dom'); sc_=_XDG['cur']; suh_=_uh_save(model); gr0_=_CARRY.get('grad')
    try:
        model.eval(); _XDG['cur']=slot_; model.__dict__['_s0_dom']=None
        _CARRY.update(on=True,nr=len(mems_),st={},new={},rows=None,grad=bool(grad_))
        with torch.set_grad_enabled(bool(grad_)): model.features(X_)
        st_={k_:(v_ if grad_ else v_.detach()) for k_,v_ in _CARRY['new'].items() if not isinstance(k_,tuple) and torch.is_tensor(v_)}
    finally:
        model.train(was_); model.feat_scale.copy_(fs_); model.__dict__['_s0_dom']=s0_; _carry_load(sv_); _XDG['cur']=sc_; _uh_load(model,suh_); _CARRY['grad']=gr0_
    return st_
def _sup_rows(S_,c_):
    s9_=c_.get('src')
    if not (isinstance(s9_,tuple) and len(s9_)==2 and all(isinstance(i9_,int) and 0<=i9_<len(S_['it']) for i9_ in s9_)): return None
    p1_,p2_=S_['it'][s9_[0]]['s'],S_['it'][s9_[1]]['s']
    if not p1_ or not p2_ or p1_.strip()==p2_.strip(): return None
    return p1_[:120],p2_[:120]
def _sup_onehop(S_,c_):
    """one-hop control from the A-B premise: (premise, prefix up to B, B) or None"""
    P_=_sup_rows(S_,c_)
    if P_ is None or not isinstance(c_.get('B'),str): return None
    B_=c_['B'].encode('latin-1','replace'); A_=c_['A'].encode('latin-1','replace')
    for p_ in P_:
        j_=p_.find(B_)
        if j_>0 and A_ in p_[:j_]: return p_,p_[:j_],B_
    return None
_SUPCH={}
def _sup_bc(S_,c_):
    """the premise with B before C, and the index of B's last byte in it"""
    P_=_sup_rows(S_,c_)
    if P_ is None or not isinstance(c_.get('B'),str): return None
    B_=c_['B'].encode('latin-1','replace'); C_=c_['C'].encode('latin-1','replace')
    for p_ in P_:
        j_=p_.find(B_)
        if j_>=0 and p_.find(C_,j_+len(B_))>0: return p_,j_+len(B_)-1
    return None
def _sup_keys(model,txts_,pos_,sl_,grad_):
    """per delta block: the (normalised, per head) key at position pos_[j] of text j, read with no memory and the store off"""
    X_,_=_ws_batch(model,txts_); sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl_
    fz_=_INFM.get('force'); _INFM['force']=False; svo_=_INFM['on']; _INFM['on']=False; was_=model.training; model.eval(); _CHN.update(on=True,k={})
    try:
        with torch.set_grad_enabled(grad_): model.features(X_)
        ks_=dict(_CHN['k'])
    finally:
        _CHN.update(on=False,k={}); _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=svo_; model.train(was_)
        if fz_ is None: _INFM.pop('force',None)
        else: _INFM['force']=fz_
    ix_=torch.tensor([min(p_,X_.shape[1]-1) for p_ in pos_],device=X_.device); ar_=torch.arange(len(txts_),device=X_.device)
    return {b_:k_.float()[ar_,ix_] for b_,k_ in ks_.items()}
def _sup_chain(model,cfg,cs_,S_,sl_,grad_=False,wgrad_=False,shuffle_=False):
    """chaining alignment for chains cs_ (all with a one-hop prefix and a B-C premise): mean cos(M_AB k_q, k_B) over blocks and heads"""
    O_=[_sup_onehop(S_,c_) for c_ in cs_]; Bc_=[_sup_bc(S_,c_) for c_ in cs_]
    M_=_ctx_mem_state(model,[o_[0] for o_ in O_],sl_,grad_=wgrad_)
    Kq_=_sup_keys(model,[o_[1] for o_ in O_],[len(o_[1])-1 for o_ in O_],sl_,grad_)
    pb_=[b_ for b_ in Bc_]
    if shuffle_ and len(pb_)>1: pb_=pb_[1:]+pb_[:1]                           # control: another chain's bridge key
    Kb_=_sup_keys(model,[b_[0] for b_ in pb_],[b_[1] for b_ in pb_],sl_,grad_)
    cs9_=[]
    for b_,S9_ in M_.items():
        if b_ not in Kq_ or b_ not in Kb_ or S9_.dim()!=3: continue
        n_=len(cs_); C_=Kq_[b_].shape[-1]; dh_=S9_.shape[-1]; mh_=C_//dh_
        if mh_*dh_!=C_ or S9_.shape[0]!=n_*mh_: continue
        kq_=F.normalize(Kq_[b_].view(n_*mh_,dh_),dim=-1); kb_=F.normalize(Kb_[b_].view(n_*mh_,dh_),dim=-1)
        vh_=torch.bmm(S9_.float(),kq_[:,:,None]).squeeze(-1)                 # (value, key) x key -> value
        if str(getattr(cfg,'inf_sup_chain_mode','cos'))=='nce':
            vh3_=F.normalize(vh_.view(n_,mh_,dh_),dim=-1).transpose(0,1); kb3_=kb_.view(n_,mh_,dh_).transpose(0,1)   # (heads, chains, dh)
            sim_=torch.bmm(vh3_,kb3_.transpose(1,2))                                   # (heads, chains, chains): own bridge on the diagonal
            cs9_.append(sim_)
        else: cs9_.append(F.cosine_similarity(vh_,kb_,dim=-1))
    if not cs9_: return None
    if str(getattr(cfg,'inf_sup_chain_mode','cos'))=='nce':
        sim_=torch.cat(cs9_,0); n_=sim_.shape[-1]; ey_=torch.eye(n_,device=sim_.device,dtype=torch.bool)
        tau_=float(getattr(cfg,'inf_sup_chain_tau',0.1)); tg_=torch.arange(n_,device=sim_.device).repeat(sim_.shape[0])
        ce_=F.cross_entropy((sim_/tau_).reshape(-1,n_),tg_)
        acc_=(sim_.argmax(-1)==torch.arange(n_,device=sim_.device)).float().mean()
        own_=sim_[:,ey_].mean(); oth_=sim_[:,~ey_].mean() if n_>1 else torch.zeros((),device=sim_.device)
        _SUPCH.update(acc=float(acc_),own=float(own_),oth=float(oth_),n=n_)
        return 1.0-ce_ if grad_ else own_-oth_                                         # training: 1 - CE (the caller adds weight x (1 - value)); readout: margin
    return torch.cat(cs9_).mean()
def _sup_heldout(cfg,S_,S_slot,n_):
    """HELD-OUT CHAINS for the per-sleep transfer readout. Probe chains held out by inf_kgx_hold never get a conclusion (NREM, REM, audit and
    reinstatement all skip them), so they are not in S_['conc']: their items are built from the probe definitions and the stored premise
    sentences instead. For each step A-B-C of a held-out chain: the latest live real or verified sentence that names A then B (both within
    the first 120 bytes, as _sup_rows keeps), and the same for B then C; a step is skipped until both premises are stored, or if one sentence
    holds all three. Returns ([(slot, {'A','B','C'}, (A-B premise, B-C premise))], steps still waiting for a premise)."""
    N_=int(getattr(cfg,'inf_kgx_hold',0))
    if N_<=0 or n_<=0: return [],0
    it_=S_['it']; ix_=S_.get('idx',{}); out_=[]; miss_=0
    def prem_(a_,b_):
        A9_=a_.encode('latin-1','replace'); B9_=b_.encode('latin-1','replace')
        for i_ in sorted(set(ix_.get(A9_,()))&set(ix_.get(B9_,())),reverse=True):
            if not (isinstance(i_,int) and 0<=i_<len(it_)): continue
            x_=it_[i_]
            if x_.get('gone') or x_.get('k') not in ('real','ver') or not x_.get('s'): continue
            ja_=x_['s'].find(A9_); jb_=x_['s'].find(B9_)
            if 0<=ja_<jb_ and jb_+len(B9_)<=120: return i_
        return None
    for k_,ch_ in enumerate(inf_probes(cfg)):
        if k_%N_!=N_-1: continue
        nm_=ch_['names']
        for j_ in range(len(nm_)-2):
            A_,B_,C_=nm_[j_],nm_[j_+1],nm_[j_+2]
            ia_=prem_(A_,B_); ic_=prem_(B_,C_)
            if ia_ is None or ic_ is None or ia_==ic_: miss_+=1; continue
            out_.append((int(S_slot(it_[ia_]['d'])),{'A':A_,'B':B_,'C':C_},(it_[ia_]['s'][:120],it_[ic_]['s'][:120])))
            if len(out_)>=n_: return out_,miss_
    return out_,miss_
def _sup_lp2(model,cfg,rows_,sl_,mode_,grad_=False,wgrad_=False):
    """log p per byte of each row's target; rows_ = [(prompt bytes, target bytes, memory texts or None)]; mode_ 'mem' builds the summed
    memory of the row's memory texts (one or two), 'none' uses no memory"""
    X_,M_=_ws_batch(model,[p_+t_ for p_,t_,_ in rows_]); st_=None
    if mode_=='mem':
        nm_=max(len(m_) for _,_,m_ in rows_); parts_=[]
        for i_ in range(nm_): parts_.append(_ctx_mem_state(model,[m_[min(i_,len(m_)-1)] for _,_,m_ in rows_],sl_,grad_=wgrad_))
        st_={}
        for k_ in parts_[0]:
            if all(k_ in p_ and p_[k_].shape==parts_[0][k_].shape for p_ in parts_):
                st_[k_]=parts_[0][k_]
                for i_,p_ in enumerate(parts_[1:],1):
                    msk_=torch.tensor([1.0 if len(m_)>i_ else 0.0 for _,_,m_ in rows_],device=p_[k_].device)   # rows with one memory text add it once
                    msk_=msk_.repeat_interleave(p_[k_].shape[0]//len(rows_)).view(-1,*([1]*(p_[k_].dim()-1)))
                    st_[k_]=st_[k_]+p_[k_]*msk_
    sv4_=_carry_save(); sc_=_XDG['cur']; _XDG['cur']=sl_; fz_=_INFM.get('force'); _INFM['force']=False; svo_=_INFM['on']; _INFM['on']=False; was_=model.training; model.eval()
    if st_: _CARRY.update(on=True,nr=0,new={},rows=torch.arange(len(rows_)),st=st_)
    else: _CARRY.update(on=False,st={},new={})
    try:
        with torch.set_grad_enabled(grad_): lp_=F.log_softmax(model(X_[:,:-1],sl_).float(),-1)
    finally:
        _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=svo_; model.train(was_)
        if fz_ is None: _INFM.pop('force',None)
        else: _INFM['force']=fz_
    out_=[]
    for j_,(p_,t_,_) in enumerate(rows_):
        ps_=torch.arange(len(p_)-1,len(p_)+len(t_)-1,device=X_.device); ps_=ps_[ps_<X_.shape[1]-1]
        out_.append(lp_[j_,ps_].gather(-1,X_[j_,ps_+1,None]).squeeze(-1).mean())
    return torch.stack(out_)
def _sup_dreams(model,cfg,S_,vals,olds_,t,ts,S_slot):
    """SUPERPOSED-MEMORY DREAMS (see the module note in the build): practise inferring A->C from a memory that holds A-B and B-C"""
    lv_=[c_ for c_ in S_.get('conc',[]) if not c_.get('freed') and isinstance(c_.get('C'),str) and isinstance(c_.get('A'),str) and _sup_rows(S_,c_) is not None and not _kgx_ho(cfg,c_)]
    by_={}
    for c_ in lv_: by_.setdefault(int(S_slot(c_['d'])),[]).append(c_)
    sls_=[s_ for s_,L_ in by_.items() if len(L_)>=4]
    if not sls_: print(f"  [inference SUPERPOSED-MEMORY DREAMS] skipped: {len(lv_)} concluded chains with two source sentences (need 4 in one slot)",flush=True); return
    rs_=np.random.RandomState(7717+int(model.__dict__.get('_visit',0))); vis_=int(model.__dict__.get('_visit',0))
    smp_=[]
    for s_ in sls_: smp_+=[(s_,c_) for c_ in by_[s_][:6]]
    q2_=lambda c_: (f"So {c_['A']} is connected to ".encode('latin-1','replace'),(c_['C']+'.').encode('latin-1','replace'))
    def probe():
        with torch.no_grad():
            R_={k_:[] for k_ in ('sup','none','text','one_m','one_n','rand','bc','two')}
            for s_ in sls_:
                cs_=[c_ for s9_,c_ in smp_ if s9_==s_]
                rw_=[q2_(c_)+(list(_sup_rows(S_,c_)),) for c_ in cs_]
                R_['sup'].append(_sup_lp2(model,cfg,rw_,s_,'mem')); R_['none'].append(_sup_lp2(model,cfg,rw_,s_,'none'))
                R_['bc'].append(_sup_lp2(model,cfg,[(p_,t_,[m_[1]]) for p_,t_,m_ in rw_],s_,'mem'))   # the B-C premise's memory alone (src = (A side, C side))
                if len(rw_)>1: R_.setdefault('rand',[]).append(_sup_lp2(model,cfg,[(p_,t_,rw_[(j_+1)%len(rw_)][2]) for j_,(p_,t_,_) in enumerate(rw_)],s_,'mem'))
                if len(rw_)>1: R_['two'].append(_sup_lp2(model,cfg,[(p_,t_,list(m_)+list(rw_[(j_+1)%len(rw_)][2])) for j_,(p_,t_,m_) in enumerate(rw_)],s_,'mem'))   # + another chain's two premises
                L9_=int(cfg_seq_(model)); tx_=[]
                for p_,t_,m_ in rw_:                                            # premises as text, trimmed from the front of each to fit the window
                    h_=max(8,(L9_-len(p_)-len(t_)-4)//2); tx_.append((m_[0][-h_:]+b' '+m_[1][-h_:]+b' '+p_,t_,None))
                R_['text'].append(_sup_lp2(model,cfg,tx_,s_,'none'))
                o_=[x_ for x_ in (_sup_onehop(S_,c_) for c_ in cs_) if x_ is not None]
                if o_:
                    R_['one_m'].append(_sup_lp2(model,cfg,[(pf_,b_,[pm_]) for pm_,pf_,b_ in o_],s_,'mem')); R_['one_n'].append(_sup_lp2(model,cfg,[(pf_,b_,None) for pm_,pf_,b_ in o_],s_,'none'))
            ch_=[];cc_=[]
            if float(getattr(cfg,'inf_sup_chain',0))>0:
                for s_ in sls_:
                    cs_=[c_ for s9_,c_ in smp_ if s9_==s_ and _sup_onehop(S_,c_) is not None and _sup_bc(S_,c_) is not None]
                    if len(cs_)>=2:
                        a9_=_sup_chain(model,cfg,cs_,S_,s_)
                        if a9_ is not None: ch_.append(float(a9_)); cc_.append(float(_SUPCH.get('acc',float('nan'))))
                        if str(getattr(cfg,'inf_sup_chain_mode','cos'))!='nce':
                            b9_=_sup_chain(model,cfg,cs_,S_,s_,shuffle_=True); cc_[-1:]=[float(b9_)] if b9_ is not None else cc_[-1:]
            out9_={k_:(float(torch.cat(v_).mean()) if v_ else float('nan')) for k_,v_ in R_.items()}
            out9_['chain']=float(np.mean(ch_)) if ch_ else float('nan'); out9_['chain_ctl']=float(np.mean(cc_)) if cc_ else float('nan')
            return out9_
    try: HO_,hmiss_=_sup_heldout(cfg,S_,S_slot,int(getattr(cfg,'inf_sup_ho_n',64)))   # HELD-OUT CHAINS (never concluded, never trained): readout only
    except Exception as e8_: HO_,hmiss_=[],0; print(f"  [inference SUPERPOSED-MEMORY DREAMS | HELD-OUT CHAINS] unavailable ({type(e8_).__name__}: {e8_})"[:260],flush=True)
    xs_=[0]
    def hprobe():
        """the same two-hop rows for the held-out chains: superposed, none, B-C premise only, random memory (the next held-out chain's
        premises), premises as text; each item read in the slot of its A-B premise"""
        if not HO_: return None
        with torch.no_grad():
            R_={k_:[] for k_ in ('sup','none','bc','rand','text','two')}; xs_[0]=0
            AL_=[(q2_(c_)+(list(m_),),s_) for s_,c_,m_ in HO_]; L9_=int(cfg_seq_(model))
            for s_ in sorted(set(s9_ for s9_,_,_ in HO_)):
                ix9_=[j_ for j_,(_,s9_) in enumerate(AL_) if s9_==s_]; rw_=[AL_[j_][0] for j_ in ix9_]
                R_['sup'].append(_sup_lp2(model,cfg,rw_,s_,'mem')); R_['none'].append(_sup_lp2(model,cfg,rw_,s_,'none'))
                R_['bc'].append(_sup_lp2(model,cfg,[(p_,t_,[m_[1]]) for p_,t_,m_ in rw_],s_,'mem'))
                if len(ix9_)>1: R_['rand'].append(_sup_lp2(model,cfg,[(p_,t_,rw_[(j_+1)%len(rw_)][2]) for j_,(p_,t_,_) in enumerate(rw_)],s_,'mem'))   # same slot, as for the trained sample
                elif len(AL_)>1: R_['rand'].append(_sup_lp2(model,cfg,[(p_,t_,AL_[(ix9_[0]+1)%len(AL_)][0][2]) for p_,t_,_ in rw_],s_,'mem')); xs_[0]+=1   # a lone item: another slot's
                if len(ix9_)>1: R_['two'].append(_sup_lp2(model,cfg,[(p_,t_,list(m_)+list(rw_[(j_+1)%len(rw_)][2])) for j_,(p_,t_,m_) in enumerate(rw_)],s_,'mem'))   # + another held-out chain's premises
                tx_=[]
                for p_,t_,m_ in rw_:
                    h_=max(8,(L9_-len(p_)-len(t_)-4)//2); tx_.append((m_[0][-h_:]+b' '+m_[1][-h_:]+b' '+p_,t_,None))
                R_['text'].append(_sup_lp2(model,cfg,tx_,s_,'none'))
            return {k_:(float(torch.cat(v_).mean()) if v_ else float('nan')) for k_,v_ in R_.items()}
    def hprobe_safe():
        try: return hprobe()
        except Exception as e8_: print(f"  [inference SUPERPOSED-MEMORY DREAMS | HELD-OUT CHAINS] unavailable ({type(e8_).__name__}: {e8_})"[:260],flush=True); return None
    if str(getattr(cfg,'inf_sup_params','all'))=='kv':
        P_=[p_ for n_,p_ in model.named_parameters() if p_.requires_grad and (n_.endswith(('Wk.weight','Wv.weight','conv_k','conv_v')))]
    else: P_=[p_ for p_ in _opt_params(model) if p_.requires_grad]
    W0_=[p_.detach().clone() for p_ in P_]
    la_=float(getattr(cfg,'inf_sup_lr_abs',0.0)); lr_=la_ if la_>0 else float(cfg.lr)*float(getattr(cfg,'inf_sup_lr',0.0033)); wg_=bool(int(getattr(cfg,'inf_sup_wgrad',0)))
    st_=int(getattr(cfg,'inf_sup_steps',60)); nb_=int(getattr(cfg,'inf_sup_rows',8)); noom_=0; lch_=[]
    nre_=int(getattr(cfg,'inf_sup_retry',0)); dv_=float(getattr(cfg,'inf_sup_retry_div',3.0)); tr_=[]   # RETRY: after a rolled-back try, again from the same weights at rate / dv_
    an_=[x_ for x_ in S_['it'] if x_.get('k')=='real' and not x_.get('gone') and len(x_['s'])>=40]
    fv_=float(getattr(cfg,'inf_sup_anc_visit',0.0))                             # VISITED-SLOT ANCHORS: this share of the anchor rows comes from the visited slot
    av_=[x_ for x_ in S_['it'] if x_.get('k') in ('real','ver') and not x_.get('gone') and len(x_['s'])>=40 and int(S_slot(x_['d']))==int(ts)] if fv_>0 else []
    nv0_=int(round(fv_*nb_)) if av_ else 0
    ow0_=float(getattr(cfg,'inf_sup_onehop_w',0.0))
    KP_=None; pinfo_=[0,[]]; pm_=int(getattr(cfg,'inf_sup_protect',0))
    if pm_:                                                                     # PROTECTED UPDATES: whiten each key/value step by the Fisher factors of ALL
        sv9_={k_:model.__dict__.get(k_) for k_ in ('_kf_P','_kf_K','_kf_proj')}   #   domains, the visited one included (its factors are stored at the end of
        try:                                                                    #   its visit); the run's own preconditioner is left exactly as it was
            kf_build(model,cfg,-1,quiet=True); KP_=model.__dict__.get('_kf_P') or None
        except Exception as e8_: print(f"  [inference SUPERPOSED-MEMORY DREAMS] protection unavailable ({type(e8_).__name__}: {e8_})"[:200],flush=True); KP_=None
        finally:
            for k_,v_ in sv9_.items():
                if v_ is None: model.__dict__.pop(k_,None)
                else: model.__dict__[k_]=v_
    cmp0_=_COMPILED.get(id(model)) is not None and not _COMPILED.get('failed')
    if cmp0_: bc0_,oc0_=_ws_bpc(model,cfg,vals,t,ts,olds_)                       # compiled-forward BPC: for the offset readout only
    back_=_ws_eager(model)                                                       # every measurement below, before and after, in eager mode (the brake
    try:                                                                         #   baseline used to be measured with the compiled forward, the 'after' eager)
        H0_=hprobe_safe()
        P0_=probe(); m0_,n0_=P0_['sup'],P0_['none']; b0_,o0_=_ws_bpc(model,cfg,vals,t,ts,olds_)
        for at_ in range(nre_+1):
            lr9_=lr_/(dv_**at_); ls_=[]; lch_=[]
            if at_>0:
                with torch.no_grad():
                    for p_,w_ in zip(P_,W0_): p_.copy_(w_)
            op_=torch.optim.Adam(P_,lr=lr9_)
            for _ in range(st_):
                s_=sls_[rs_.randint(len(sls_))]; L_=by_[s_]; cs_=[L_[int(i_)] for i_ in rs_.choice(len(L_),min(nb_,len(L_)),replace=False)]
                try:
                    l_=-_sup_lp2(model,cfg,[q2_(c_)+(list(_sup_rows(S_,c_)),) for c_ in cs_],s_,'mem',grad_=True,wgrad_=wg_).mean()
                    anc_=[]; nv_=int(round(fv_*nb_)) if av_ else 0
                    if an_ and nb_-nv_>0: anc_.append([an_[int(i_)] for i_ in rs_.choice(len(an_),min(nb_-nv_,len(an_)),replace=False)])
                    if nv_>0: anc_.append([av_[int(i_)] for i_ in rs_.choice(len(av_),min(nv_,len(av_)),replace=False)])
                    na_=sum(len(a_) for a_ in anc_)
                    for ax_ in anc_:
                        Xa_,Ma_=_ws_batch(model,[x_['s'] for x_ in ax_])
                        lg_=_ws_run(model,Xa_,int(S_slot(ax_[0]['d'])),grad_=True); l_=l_+(len(ax_)/na_)*F.cross_entropy(lg_.reshape(-1,lg_.shape[-1]),Xa_[:,1:].reshape(-1),reduction='none').view(Xa_.shape[0],-1).mul(Ma_[:,1:]).sum()/Ma_[:,1:].sum().clamp(min=1)
                    chw_=float(getattr(cfg,'inf_sup_chain',0))
                    if chw_>0:
                        cq_=[c_ for c_ in cs_ if _sup_onehop(S_,c_) is not None and _sup_bc(S_,c_) is not None]
                        if len(cq_)>=1:
                            a9_=_sup_chain(model,cfg,cq_,S_,s_,grad_=True,wgrad_=True)
                            if a9_ is not None: l_=l_+chw_*(1.0-a9_); lch_.append(float(a9_))
                    ow_=float(getattr(cfg,'inf_sup_onehop_w',0.0))
                    if ow_>0:                                                       # ONE-HOP ROWS: keep reading B back from the A-B premise's own memory
                        oh_=[x_ for x_ in (_sup_onehop(S_,c_) for c_ in cs_) if x_ is not None]
                        if oh_: l_=l_-ow_*_sup_lp2(model,cfg,[(pf_,b_,[pm9_]) for pm9_,pf_,b_ in oh_],s_,'mem',grad_=True,wgrad_=wg_).mean()
                    op_.zero_grad(); l_.backward()
                    if KP_:
                        pw_=[(q_,q_.detach().clone()) for q_ in P_ if q_.ndim==2 and id(q_) in KP_]; op_.step()
                        with torch.no_grad():
                            for q_,q0_ in pw_:
                                d_=(q_-q0_).float(); z_=_kf_apply(KP_[id(q_)],d_)
                                if pm_==1: z_=z_*(d_.norm()/z_.norm().clamp_min(1e-12))
                                pinfo_[1].append(float(F.cosine_similarity(d_.flatten(),z_.flatten(),dim=0))); q_.copy_((q0_.float()+z_).to(q_.dtype))
                        pinfo_[0]=len(pw_); del pw_
                    else: op_.step()
                    ls_.append(float(l_)); del l_
                except RuntimeError as e8_:
                    if 'out of memory' not in str(e8_).lower() or nb_<=2: raise
                    op_.zero_grad(set_to_none=True); nb_=max(2,nb_//2); noom_+=1
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            op_.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True); del op_
            b1_,o1_=_ws_bpc(model,cfg,vals,t,ts,olds_); bk_=float(getattr(cfg,'inf_sup_brake',0.03)); dB_,dO_=b1_-b0_,(o1_-o0_ if o1_==o1_ else 0.0); kept_=dB_<=bk_ and dO_<=bk_
            tr_.append((lr9_,dB_,dO_,kept_))
            if kept_: break
        if not kept_:
            with torch.no_grad():
                for p_,w_ in zip(P_,W0_): p_.copy_(w_)
        P1_=probe(); m1_,n1_=P1_['sup'],P1_['none']; H1_=hprobe_safe() if H0_ is not None else None
        print(f"  [inference SUPERPOSED-MEMORY DREAMS] visit {vis_}: {st_} steps on {len(lv_)} concluded chains in {len(sls_)} slots ({nb_} rows + {nb_ if (an_ or av_) else 0} anchor rows per step{f', {nv0_} of them from the visited slot ({len(av_)} available)' if fv_>0 else ''}{f', one-hop rows x{ow0_:g}' if ow0_>0 else ''}{f', PROTECTED {pinfo_[0]} key/value matrices by all domains incl. the visited one (mode {pm_}), step kept {np.mean(pinfo_[1]):.2f} cosine to Adam' if pinfo_[1] else (', protection requested but no factors' if pm_ else '')}, lr {lr9_:.1e}) | "
              f"loss {ls_[0]:.3f} -> {np.mean(ls_[-5:]):.3f} | visited {dB_:+.4f}, old {dO_:+.4f}: {'KEPT' if kept_ else 'ROLLED BACK (brake '+str(bk_)+')'}{(' after '+str(len(tr_))+' tries: '+', '.join(f'{a_:.1e} visited {b_:+.4f} old {c_:+.4f} '+('kept' if d_ else 'rolled back') for a_,b_,c_,d_ in tr_)) if len(tr_)>1 else ''} | BRAKE measured in eager mode before and after{f' (compiled forward minus eager before training: visited {bc0_-b0_:+.4f}, old {(oc0_-o0_) if oc0_==oc0_ else 0.0:+.4f}; the brake used to count minus that as damage)' if cmp0_ else ' (no compiled forward)'} | "
              f"log p(C) per byte on {len(smp_)} sampled chains (store off): from the SUPERPOSED memory {m0_:.3f} -> {m1_:.3f}, from no memory {n0_:.3f} -> {n1_:.3f}, "
              f"B-C premise ONLY {P0_['bc']:.3f} -> {P1_['bc']:.3f}, premises as TEXT {P0_['text']:.3f} -> {P1_['text']:.3f}, RANDOM memory (another chain's premises) {P0_['rand']:.3f} -> {P1_['rand']:.3f} | ONE-HOP CONTROL (premise prefix -> B): from that premise's memory {P0_['one_m']:.3f} -> {P1_['one_m']:.3f}, "
              f"from no memory {P0_['one_n']:.3f} -> {P1_['one_n']:.3f} | first rate {lr_:.1e}, gradient through the premise reads {'on' if wg_ else 'off'}, trained {str(getattr(cfg,'inf_sup_params','all'))} ({sum(p_.numel() for p_ in P_)/1e6:.2f}M parameters)"
              +(f", {noom_} out-of-memory retries (rows now {nb_})" if noom_ else "")
              +(f" | CHAINING {'(contrastive) cosine margin own bridge minus others' if str(getattr(cfg,'inf_sup_chain_mode','cos'))=='nce' else 'ALIGNMENT cos(M_AB k_q, k_B)'}: {P0_['chain']:.3f} -> {P1_['chain']:.3f} "
                f"({'own bridge retrieved among the sampled chains' if str(getattr(cfg,'inf_sup_chain_mode','cos'))=='nce' else 'control, another chain bridge'}: {P0_['chain_ctl']:.3f} -> {P1_['chain_ctl']:.3f}); "
                f"in training {lch_[0]:.3f} -> {np.mean(lch_[-10:]):.3f}, weight {float(getattr(cfg,'inf_sup_chain',0)):g}" if lch_ else ""),flush=True)
        if H0_ is not None and H1_ is not None:
            f8_=lambda a_,b_: f"{a_:.3f} -> {b_:.3f} ({b_-a_:+.3f})"
            print(f"  [inference SUPERPOSED-MEMORY DREAMS | HELD-OUT CHAINS] visit {vis_}: {len(HO_)} steps of chains held out by inf_kgx_hold (never concluded, never trained), "
                  f"{len(set(s9_ for s9_,_,_ in HO_))} slots, {hmiss_} more waiting for a premise (their premises are read like any fact: stored, and among the anchor rows as plain text; "
                  f"only the conclusion is withheld) | random memory from the same slot{f' ({xs_[0]} lone items from another slot)' if xs_[0] else ''} | log p(C) per byte, store off, before -> after this phase: "
                  f"SUPERPOSED {f8_(H0_['sup'],H1_['sup'])}, NO memory {f8_(H0_['none'],H1_['none'])}, B-C premise ONLY {f8_(H0_['bc'],H1_['bc'])}, "
                  f"RANDOM memory {f8_(H0_['rand'],H1_['rand'])}, premises as TEXT {f8_(H0_['text'],H1_['text'])}, "
                  f"TWO CHAINS (own premises + another chain's, superposed: A must pick its chain) {f8_(H0_['two'],H1_['two'])} (trained sample {f8_(P0_['two'],P1_['two'])}) | "
                  f"SUPERPOSED minus B-C ONLY (composition beyond the memory that names C): held-out {f8_(H0_['sup']-H0_['bc'],H1_['sup']-H1_['bc'])}, "
                  f"trained sample {f8_(P0_['sup']-P0_['bc'],P1_['sup']-P1_['bc'])} | SUPERPOSED minus RANDOM: held-out {f8_(H0_['sup']-H0_['rand'],H1_['sup']-H1_['rand'])}, "
                  f"trained sample {f8_(P0_['sup']-P0_['rand'],P1_['sup']-P1_['rand'])}",flush=True)
        elif int(getattr(cfg,'inf_kgx_hold',0))>0:
            print(f"  [inference SUPERPOSED-MEMORY DREAMS | HELD-OUT CHAINS] visit {vis_}: none yet ({hmiss_} held-out steps waiting for a premise)",flush=True)
    except Exception as e9_:
        with torch.no_grad():
            for p_,w_ in zip(P_,W0_): p_.copy_(w_)
        print(f"  [inference SUPERPOSED-MEMORY DREAMS] unavailable ({type(e9_).__name__}: {e9_})"[:260],flush=True)
    finally:
        del W0_; back_()

def _sup_tracecomp(model,cfg,S_,S_slot):
    """TRACE COMPOSITION (readout only, inf_sup_tc 1; no effect on training or dreams). Can a JEPA learn to compose two memory TRACES?
    Each text is read on its own into the last delta block's memory (store off); an encoder E (one map per head state shared by all heads,
    then one over the heads) gives a dz-number trace. A predictor P maps (E(M_AB), E(M_BC)) to E(M_AC), the trace of the memory left by
    reading the conclusion 'So A is connected to C.' - a LeJEPA objective: (1 - lambda) x squared prediction error + lambda x SIGReg on the
    traces, trained fresh at every sleep on the store's own fact chains (two recorded facts sharing an entity: A-B and B-C). Chains whose
    names belong to probe chains held out by inf_kgx_hold are never trained on; those chains are scored as HELD-OUT, their conclusion text
    written only for scoring. Score: top-1 by cosine among the four conclusions built from the same three names (A->C, C->A, A->B, B->C;
    chance 0.25: needs the chain's ends in the right order, not just its names), and among those plus four other chains' conclusions
    (chance 0.125). Compared with the sum of the two premise traces, either premise's trace alone, and the untrained encoder and predictor."""
    import time as _t9
    t0_=_t9.time(); vis_=int(model.__dict__.get('_visit',0)); tag_="  [inference TRACE COMPOSITION]"
    try:
        it_=S_['it']; N9_=int(getattr(cfg,'inf_kgx_hold',0)); hon_=set()
        if N9_>0:
            for k_,ch_ in enumerate(inf_probes(cfg)):
                if k_%N9_==N9_-1: hon_.update(ch_['names'])
        okf_=lambda f_: (not f_.get('freed') and not f_.get('derived') and isinstance(f_.get('i'),int) and 0<=f_['i']<len(it_) and not it_[f_['i']].get('gone')
                         and isinstance(f_.get('A'),str) and isinstance(f_.get('B'),str) and it_[f_['i']].get('s'))
        F_=[f_ for f_ in S_.get('facts',[]) if okf_(f_)]; byA_={}
        for f_ in F_: byA_.setdefault(f_['A'],[]).append(f_)
        seen_=set(); TR_=[]
        for f_ in F_:
            for g_ in byA_.get(f_['B'],()):
                A_,B_,C_=f_['A'],f_['B'],g_['B']
                if g_ is f_ or g_['i']==f_['i'] or len({A_,B_,C_})<3 or (A_,B_,C_) in seen_ or ({A_,B_,C_}&hon_): continue
                p1_,p2_=it_[f_['i']]['s'][:120],it_[g_['i']]['s'][:120]
                if p1_.strip()==p2_.strip(): continue
                seen_.add((A_,B_,C_)); TR_.append((A_,B_,C_,p1_,p2_,int(S_slot(it_[f_['i']]['d']))))
        for c_ in S_.get('conc',[]):                                              # + the chains the system concluded (their two source sentences)
            if c_.get('freed') or not all(isinstance(c_.get(x_),str) for x_ in 'ABC'): continue
            A_,B_,C_=c_['A'],c_['B'],c_['C']; P9_=_sup_rows(S_,c_)
            if P9_ is None or len({A_,B_,C_})<3 or (A_,B_,C_) in seen_ or ({A_,B_,C_}&hon_): continue
            seen_.add((A_,B_,C_)); TR_.append((A_,B_,C_,P9_[0],P9_[1],int(S_slot(c_['d']))))
        HO_,_=_sup_heldout(cfg,S_,S_slot,int(getattr(cfg,'inf_sup_ho_n',64)))
        HT_=[(c_['A'],c_['B'],c_['C'],m_[0],m_[1],s_) for s_,c_,m_ in HO_]
        if len(TR_)<32: print(f"{tag_} visit {vis_}: skipped, {len(TR_)} fact and concluded chains outside the held-out names (need 32)",flush=True); return
        rs_=np.random.RandomState(8080+vis_); rs_.shuffle(TR_); TR_=TR_[:int(getattr(cfg,'inf_sup_tc_n',512))]
        nv_=max(16,len(TR_)//4); VS_=TR_[:nv_//2]; VA_=TR_[nv_//2:nv_]; TT_=TR_[nv_:]   # VS_ picks the heads, VA_ is reported (unbiased)
        cq_=lambda a_,c_: f"So {a_} is connected to {c_}.".encode('latin-1','replace')
        txt_={}                                                                   # (bytes, slot) -> row
        def tid_(b_,s_):
            k_=(bytes(b_),int(s_))
            if k_ not in txt_: txt_[k_]=len(txt_)
            return txt_[k_]
        def rows_(T_):
            return [(tid_(p1_,s_),tid_(p2_,s_),[tid_(cq_(A_,C_),s_),tid_(cq_(C_,A_),s_),tid_(cq_(A_,B_),s_),tid_(cq_(B_,C_),s_)]) for A_,B_,C_,p1_,p2_,s_ in T_]
        RT_,RV_,RH_,RS_=rows_(TT_),rows_(VA_),rows_(HT_),rows_(VS_)
        keys_=list(txt_.keys()); dev_=model.embed.weight.device; Mx_=[None]*len(keys_); bk_=None
        back_=_ws_eager(model)
        try:
            with torch.no_grad():
                for s_ in sorted(set(k_[1] for k_ in keys_)):
                    ix_=[j_ for j_,k_ in enumerate(keys_) if k_[1]==s_]
                    for a_ in range(0,len(ix_),64):
                        jj_=ix_[a_:a_+64]; st_=_ctx_mem_state(model,[keys_[j_][0] for j_ in jj_],s_)
                        b3_=[b_ for b_,v_ in st_.items() if torch.is_tensor(v_) and v_.dim()==3]
                        if not b3_: continue
                        bk_=b3_[-1] if bk_ is None or bk_ not in b3_ else bk_; v_=st_[bk_].float(); mh_=v_.shape[0]//len(jj_)
                        v_=v_.view(len(jj_),mh_,v_.shape[1]*v_.shape[2])
                        for q_,j_ in enumerate(jj_): Mx_[j_]=v_[q_]
        finally: back_()
        if any(m_ is None for m_ in Mx_): print(f"{tag_} visit {vis_}: unavailable (no delta-block memory for some texts)",flush=True); return
        M_=torch.stack(Mx_)                                                       # (texts, heads, dh*dh)
        trn_=sorted({j_ for a_,b_,cs_ in RT_ for j_ in (a_,b_,cs_[0])})
        mu_=M_[trn_].mean((0,1),keepdim=True); sd_=M_[trn_].std().clamp(min=1e-6); M_=(M_-mu_)/sd_
        dz_=int(getattr(cfg,'inf_sup_tc_dim',32)); r_=16; mh_,dd_=M_.shape[1],M_.shape[2]; lam_=float(getattr(cfg,'inf_sup_tc_lambda',0.05))
        fr9_=torch.random.fork_rng(devices=[dev_] if dev_.type=='cuda' else []); fr9_.__enter__()   # the readout leaves the run's random streams untouched
        torch.manual_seed(8081+vis_)
        E1_=nn.Linear(dd_,r_).to(dev_); E2_=nn.Linear(mh_*r_,dz_).to(dev_); P_=nn.Sequential(nn.Linear(2*dz_,128),nn.GELU(),nn.Linear(128,dz_)).to(dev_)
        enc_=lambda X_: E2_(E1_(X_).flatten(1))
        def score(R_,mode_):
            if not R_: return float('nan'),float('nan')
            with torch.no_grad():
                Z_=enc_(M_); s4_=[];s8_=[]
                for q_,(a_,b_,cs_) in enumerate(R_):
                    za_,zb_=Z_[a_],Z_[b_]
                    pr_={'P':lambda: P_(torch.cat([za_,zb_])[None])[0],'sum':lambda: za_+zb_,'ab':lambda: za_,'bc':lambda: zb_}[mode_]()
                    oth_=[R_[(q_+o_)%len(R_)][2][0] for o_ in range(1,5)] if len(R_)>4 else []
                    C9_=Z_[cs_+oth_]; c9_=F.cosine_similarity(pr_[None],C9_,dim=-1)
                    s4_.append(float(int(torch.argmax(c9_[:4]))==0)); s8_.append(float(int(torch.argmax(c9_))==0))
                return float(np.mean(s4_)),float(np.mean(s8_))
        U_={nm_:score(R_,'P') for nm_,R_ in (('train',RT_),('val',RV_),('ho',RH_))}
        op_=torch.optim.AdamW(list(E1_.parameters())+list(E2_.parameters())+list(P_.parameters()),lr=1e-3,weight_decay=0.05)
        ia_=torch.tensor([a_ for a_,_,_ in RT_],device=dev_); ib_=torch.tensor([b_ for _,b_,_ in RT_],device=dev_); ic_=torch.tensor([cs_[0] for _,_,cs_ in RT_],device=dev_)
        st9_=int(getattr(cfg,'inf_sup_tc_steps',600)); ls_=[]; best_=(-1.0,None,0)
        for k9_ in range(st9_):
            bi_=torch.as_tensor(rs_.choice(len(RT_),min(64,len(RT_)),replace=False),device=dev_)
            za_,zb_,zc_=enc_(M_[ia_[bi_]]),enc_(M_[ib_[bi_]]),enc_(M_[ic_[bi_]])
            l_=(1-lam_)*((P_(torch.cat([za_,zb_],1))-zc_)**2).sum(1).mean()+lam_*_sigreg_le(torch.cat([za_,zb_,zc_]))
            op_.zero_grad(); l_.backward(); op_.step(); ls_.append(float(l_.detach()))
            if (k9_+1)%100==0 and RS_:                                             # keep the heads that score best on the selection chains
                v9_=score(RS_,'P')[0]
                if v9_>best_[0]: best_=(v9_,[{k_:x_.detach().clone() for k_,x_ in m_.state_dict().items()} for m_ in (E1_,E2_,P_)],k9_+1)
        if best_[1] is not None:
            for m_,sd9_ in zip((E1_,E2_,P_),best_[1]): m_.load_state_dict(sd9_)
        R9_={(nm_,md_):score(R_,md_) for nm_,R_ in (('train',RT_),('val',RV_),('ho',RH_)) for md_ in ('P','sum','ab','bc')}
        with torch.no_grad():
            Z_=enc_(M_); dt_=[float(F.cosine_similarity(Z_[cs_[0]],Z_[cs_[1]],dim=0)) for _,_,cs_ in RT_[:64]]
        f2_=lambda x_: f"{x_[0]:.2f} / {x_[1]:.2f}"
        fr9_.__exit__(None,None,None); fr9_=None
        print(f"{tag_} visit {vis_}: {len(TT_)} training, {len(VS_)} selection and {len(VA_)} validation chains (recorded fact chains and concluded chains; names of held-out probe chains excluded), {len(RH_)} HELD-OUT chain steps | "
              f"last delta block memory ({mh_} heads x {dd_}) -> trace {dz_}, predictor MLP, LeJEPA lambda {lam_:g}, {st9_} steps (heads kept from step {best_[2]}), loss {ls_[0]:.3f} -> {np.mean(ls_[-20:]):.3f} | "
              f"TOP-1 among A->C, C->A, A->B, B->C (chance 0.25) / plus 4 other chains (chance 0.125): LEARNED PREDICTOR train {f2_(R9_[('train','P')])}, validation {f2_(R9_[('val','P')])}, HELD-OUT {f2_(R9_[('ho','P')])} | "
              f"SUM of premise traces: validation {f2_(R9_[('val','sum')])}, held-out {f2_(R9_[('ho','sum')])} | A-B trace alone: validation {f2_(R9_[('val','ab')])}, held-out {f2_(R9_[('ho','ab')])} | "
              f"B-C trace alone: validation {f2_(R9_[('val','bc')])}, held-out {f2_(R9_[('ho','bc')])} | untrained: validation {f2_(U_['val'])}, held-out {f2_(U_['ho'])} | "
              f"cos(trace of A->C, trace of C->A) {np.mean(dt_):.2f} | {_t9.time()-t0_:.0f} s",flush=True)
    except Exception as e9_:
        print(f"{tag_} visit {vis_}: unavailable ({type(e9_).__name__}: {e9_})"[:300],flush=True)
    finally:
        f9_=locals().get('fr9_')
        if f9_ is not None: f9_.__exit__(None,None,None)

def _jd_memread(model,cfg,S_,H_,NGT_,nh_):
    """JEPA MEMORY-READ VIEW, readouts only (inf_jd_memread 1; no effect on dream choice or training). A sentence's code is what its own
    delta-block memory returns when probed with the keys of the cue's rare names: z(t|c) = [M_t k_n] for n in N(c), per block, heads kept.
    1. SANITY: |M_t k_n| for sentences that contain the name n against sentences that do not (AUC), split by where n sits in the sentence
       (first half: mostly subject; second half: mostly object), with the name key read ALONE and in a FRAME (' Name is').
    2. LEARNING-FREE BASELINE on the JEPA's held-out hard-negative task (the partner among sentences sharing a rare name with the cue):
       rank candidates by cos(z(t|c), z(c|c)) averaged over the cue's names, top delta block and all blocks; compared with TF-IDF on names
       and the trained JEPA by the caller. The local trunk barely reads names from memory (one-hop control about +0.01), so a weak local
       result means the harness cannot judge this view, not that it fails."""
    import time as _t9
    t0_=_t9.time()
    try:
        df_=S_.get('df',{}); dm_=int(cfg.inf_df); sl_=int(_XDG.get('cur') or 0); cap_=200
        rare_=lambda w_: len(w_)>=4 and w_[:1].isupper() and w_[1:].isalpha() and w_[1:].islower() and 0<df_.get(w_,0)<=dm_ and df_.get(w_.lower(),0)<=dm_
        names_=lambda t_: list(dict.fromkeys(w_.group(0) for w_ in _UWORD.finditer(t_) if rare_(w_.group(0))))[:4]
        J_=[j_ for j_ in range(min(nh_,len(H_))) if NGT_ is not None and j_<len(NGT_) and NGT_[j_] and names_(H_[j_][0])][:int(getattr(cfg,'inf_jd_memread_n',64))]
        if not J_: print("  [inference JEPA DREAMS | MEMORY-READ VIEW] skipped: no held-out cue with a rare name and name-sharing negatives",flush=True); return
        txt_=[]; tix_={}
        def tid_(t_):
            t_=bytes(t_[:cap_])
            if t_ not in tix_: tix_[t_]=len(txt_); txt_.append(t_)
            return tix_[t_]
        tasks_=[]
        for j_ in J_:
            c_=H_[j_][0]; cand_=[H_[j_][1]]+list(NGT_[j_])
            tasks_.append((tid_(c_),[tid_(x_) for x_ in cand_],names_(c_)))
        RN_=[it_['s'] for it_ in S_.get('it',[]) if not it_.get('gone') and len(it_['s'])>=24]
        rsn_=np.random.RandomState(5150+int(model.__dict__.get('_visit',0)))
        nms_=list(dict.fromkeys(n_ for _,_,N_ in tasks_ for n_ in N_))
        ctl_={}                                                                # 4 random stored sentences without the name, per name (sanity)
        for n_ in nms_:
            pool_=[x_ for x_ in (RN_[int(i_)] for i_ in rsn_.choice(len(RN_),min(len(RN_),40),replace=False)) if n_ not in x_][:4] if RN_ else []
            ctl_[n_]=[tid_(x_) for x_ in pool_]
        with torch.no_grad():
            M_={}                                                              # per block: (len(txt_), heads, dh, dh)
            for a_ in range(0,len(txt_),32):
                st_=_ctx_mem_state(model,txt_[a_:a_+32],sl_)
                for b_,v_ in st_.items():
                    if torch.is_tensor(v_) and v_.dim()==3: M_.setdefault(b_,[]).append(v_.float())
            nT_=len(txt_); M_={b_:torch.cat(v_) for b_,v_ in M_.items()}
            M_={b_:v_.view(nT_,v_.shape[0]//nT_,v_.shape[1],v_.shape[2]) for b_,v_ in M_.items() if v_.shape[0]%nT_==0}
            blocks_=[b_ for b_ in M_]; top_=blocks_[-1] if blocks_ else None
            K_={}
            for fr_ in ('alone','frame'):
                tx9_=[n_ if fr_=='alone' else b' '+n_+b' is' for n_ in nms_]; ps9_=[len(n_)-1 if fr_=='alone' else len(n_) for n_ in nms_]
                ks_=_sup_keys(model,tx9_,ps9_,sl_,False)
                for b_,k_ in ks_.items():
                    if b_ in M_:
                        mh_=M_[b_].shape[1]; dh_=M_[b_].shape[-1]
                        K_[(fr_,b_)]=F.normalize(k_.float().view(len(nms_),mh_,dh_),dim=-1)
            ni_={n_:i_ for i_,n_ in enumerate(nms_)}
            def read_(t_,n_,fr_,b_): return torch.einsum('hvk,hk->hv',M_[b_][t_],K_[(fr_,b_)][ni_[n_]])
            # 1. sanity
            out_={}
            for fr_ in ('alone','frame'):
                for role_ in ('subject','object'):
                    pos_=[];neg_=[]
                    for (c9_,_,N_) in tasks_:
                        ct_=txt_[c9_]
                        for n_ in N_:
                            r_=ct_.find(n_)/max(1,len(ct_)); 
                            if (role_=='subject')!=(r_<0.5): continue
                            pos_.append(float(sum(read_(c9_,n_,fr_,b_).norm(dim=-1).mean() for b_ in blocks_)/len(blocks_)))
                            neg_+=[float(sum(read_(x_,n_,fr_,b_).norm(dim=-1).mean() for b_ in blocks_)/len(blocks_)) for x_ in ctl_[n_]]
                    if pos_ and neg_:
                        P9_=np.array(pos_);N9_=np.array(neg_); auc_=float(((P9_[:,None]>N9_[None,:]).mean()+0.5*(P9_[:,None]==N9_[None,:]).mean()))
                        out_[(fr_,role_)]=(auc_,float(P9_.mean()),float(N9_.mean()),len(pos_))
            # 2. learning-free baseline
            acc_={('frame','top'):[],('frame','all'):[],('alone','top'):[],('alone','all'):[]}; ch_=[]
            for (c9_,C9_,N_) in tasks_:
                ch_.append(1.0/len(C9_))
                for fr_ in ('alone','frame'):
                    for sc_ in ('top','all'):
                        bs_=[top_] if sc_=='top' else blocks_
                        sco_=[]
                        for x_ in C9_:
                            v_=[]
                            for n_ in N_:
                                zc_=torch.cat([read_(c9_,n_,fr_,b_).flatten() for b_ in bs_]); zt_=torch.cat([read_(x_,n_,fr_,b_).flatten() for b_ in bs_])
                                v_.append(float(F.cosine_similarity(zc_,zt_,dim=0)))
                            sco_.append(float(np.mean(v_)))
                        acc_[(fr_,sc_)].append(float(int(np.argmax(sco_))==0))
        f9_=lambda k_: (f"AUC {out_[k_][0]:.2f} (|read| {out_[k_][1]:.3f} with the name against {out_[k_][2]:.3f} without, n {out_[k_][3]})" if k_ in out_ else "n 0")
        print(f"  [inference JEPA DREAMS | MEMORY-READ VIEW] {len(tasks_)} held-out cues with rare names, {len(nms_)} names, {len(blocks_)} delta blocks, slot {sl_} | "
              f"SANITY, does a sentence's memory answer its own names: key alone - name early (subject) {f9_(('alone','subject'))}, late (object) {f9_(('alone','object'))}; "
              f"key in a frame - early {f9_(('frame','subject'))}, late {f9_(('frame','object'))} | "
              f"LEARNING-FREE BASELINE, partner among name-sharing candidates (chance {np.mean(ch_):.2f}): memory read in a frame, top block {np.mean(acc_[('frame','top')]):.2f}, all blocks {np.mean(acc_[('frame','all')]):.2f}; "
              f"key alone, top block {np.mean(acc_[('alone','top')]):.2f}, all blocks {np.mean(acc_[('alone','all')]):.2f} | {_t9.time()-t0_:.0f} s",flush=True)
        return {k_:float(np.mean(v_)) for k_,v_ in acc_.items()}
    except Exception as e9_:
        print(f"  [inference JEPA DREAMS | MEMORY-READ VIEW] unavailable ({type(e9_).__name__}: {e9_})"[:300],flush=True)


# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    'cfg_seq_': 'platypus.inference.jepa',
    '_sigreg_le': 'platypus.inference.jepa',
}
