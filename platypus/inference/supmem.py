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
Readout per sleep, before and after, store off: two-hop from the superposed memory, from no memory, from a random memory (another chain's
premises) and with the premises as text; a one-hop control (premise prefix -> B from that premise's memory against none); the chaining
margin and own-bridge retrieval. A training step that runs out of GPU memory is retried with half the rows."""
import numpy as np
import torch
import torch.nn.functional as F
from platypus.private import (
    _XDG,
)
from platypus.layers import (
    _CARRY, _CHN,
)
from platypus.ngrams import (
    _opt_params,
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
from platypus.inference.infer import (
    _kgx_ho,
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
            R_={k_:[] for k_ in ('sup','none','text','one_m','one_n','rand')}
            for s_ in sls_:
                cs_=[c_ for s9_,c_ in smp_ if s9_==s_]
                rw_=[q2_(c_)+(list(_sup_rows(S_,c_)),) for c_ in cs_]
                R_['sup'].append(_sup_lp2(model,cfg,rw_,s_,'mem')); R_['none'].append(_sup_lp2(model,cfg,rw_,s_,'none'))
                if len(rw_)>1: R_.setdefault('rand',[]).append(_sup_lp2(model,cfg,[(p_,t_,rw_[(j_+1)%len(rw_)][2]) for j_,(p_,t_,_) in enumerate(rw_)],s_,'mem'))
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
    P0_=probe(); m0_,n0_=P0_['sup'],P0_['none']; b0_,o0_=_ws_bpc(model,cfg,vals,t,ts,olds_)
    if str(getattr(cfg,'inf_sup_params','all'))=='kv':
        P_=[p_ for n_,p_ in model.named_parameters() if p_.requires_grad and (n_.endswith(('Wk.weight','Wv.weight','conv_k','conv_v')))]
    else: P_=[p_ for p_ in _opt_params(model) if p_.requires_grad]
    W0_=[p_.detach().clone() for p_ in P_]
    la_=float(getattr(cfg,'inf_sup_lr_abs',0.0)); lr_=la_ if la_>0 else float(cfg.lr)*float(getattr(cfg,'inf_sup_lr',0.0033)); wg_=bool(int(getattr(cfg,'inf_sup_wgrad',0)))
    st_=int(getattr(cfg,'inf_sup_steps',60)); nb_=int(getattr(cfg,'inf_sup_rows',8)); noom_=0; lch_=[]
    nre_=int(getattr(cfg,'inf_sup_retry',0)); dv_=float(getattr(cfg,'inf_sup_retry_div',3.0)); tr_=[]   # RETRY: after a rolled-back try, again from the same weights at rate / dv_
    an_=[x_ for x_ in S_['it'] if x_.get('k')=='real' and not x_.get('gone') and len(x_['s'])>=40]
    back_=_ws_eager(model)
    try:
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
                    if an_:
                        ax_=[an_[int(i_)] for i_ in rs_.choice(len(an_),min(nb_,len(an_)),replace=False)]; Xa_,Ma_=_ws_batch(model,[x_['s'] for x_ in ax_])
                        lg_=_ws_run(model,Xa_,int(S_slot(ax_[0]['d'])),grad_=True); l_=l_+F.cross_entropy(lg_.reshape(-1,lg_.shape[-1]),Xa_[:,1:].reshape(-1),reduction='none').view(Xa_.shape[0],-1).mul(Ma_[:,1:]).sum()/Ma_[:,1:].sum().clamp(min=1)
                    chw_=float(getattr(cfg,'inf_sup_chain',0))
                    if chw_>0:
                        cq_=[c_ for c_ in cs_ if _sup_onehop(S_,c_) is not None and _sup_bc(S_,c_) is not None]
                        if len(cq_)>=1:
                            a9_=_sup_chain(model,cfg,cq_,S_,s_,grad_=True,wgrad_=True)
                            if a9_ is not None: l_=l_+chw_*(1.0-a9_); lch_.append(float(a9_))
                    op_.zero_grad(); l_.backward(); op_.step(); ls_.append(float(l_)); del l_
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
        P1_=probe(); m1_,n1_=P1_['sup'],P1_['none']
        print(f"  [inference SUPERPOSED-MEMORY DREAMS] visit {vis_}: {st_} steps on {len(lv_)} concluded chains in {len(sls_)} slots ({nb_} rows + {nb_ if an_ else 0} anchor rows per step, lr {lr9_:.1e}) | "
              f"loss {ls_[0]:.3f} -> {np.mean(ls_[-5:]):.3f} | visited {dB_:+.4f}, old {dO_:+.4f}: {'KEPT' if kept_ else 'ROLLED BACK (brake '+str(bk_)+')'}{(' after '+str(len(tr_))+' tries: '+', '.join(f'{a_:.1e} visited {b_:+.4f} old {c_:+.4f} '+('kept' if d_ else 'rolled back') for a_,b_,c_,d_ in tr_)) if len(tr_)>1 else ''} | "
              f"log p(C) per byte on {len(smp_)} sampled chains (store off): from the SUPERPOSED memory {m0_:.3f} -> {m1_:.3f}, from no memory {n0_:.3f} -> {n1_:.3f}, "
              f"premises as TEXT {P0_['text']:.3f} -> {P1_['text']:.3f}, RANDOM memory (another chain's premises) {P0_['rand']:.3f} -> {P1_['rand']:.3f} | ONE-HOP CONTROL (premise prefix -> B): from that premise's memory {P0_['one_m']:.3f} -> {P1_['one_m']:.3f}, "
              f"from no memory {P0_['one_n']:.3f} -> {P1_['one_n']:.3f} | first rate {lr_:.1e}, gradient through the premise reads {'on' if wg_ else 'off'}, trained {str(getattr(cfg,'inf_sup_params','all'))} ({sum(p_.numel() for p_ in P_)/1e6:.2f}M parameters)"
              +(f", {noom_} out-of-memory retries (rows now {nb_})" if noom_ else "")
              +(f" | CHAINING {'(contrastive) cosine margin own bridge minus others' if str(getattr(cfg,'inf_sup_chain_mode','cos'))=='nce' else 'ALIGNMENT cos(M_AB k_q, k_B)'}: {P0_['chain']:.3f} -> {P1_['chain']:.3f} "
                f"({'own bridge retrieved among the sampled chains' if str(getattr(cfg,'inf_sup_chain_mode','cos'))=='nce' else 'control, another chain bridge'}: {P0_['chain_ctl']:.3f} -> {P1_['chain_ctl']:.3f}); "
                f"in training {lch_[0]:.3f} -> {np.mean(lch_[-10:]):.3f}, weight {float(getattr(cfg,'inf_sup_chain',0)):g}" if lch_ else ""),flush=True)
    except Exception as e9_:
        with torch.no_grad():
            for p_,w_ in zip(P_,W0_): p_.copy_(w_)
        print(f"  [inference SUPERPOSED-MEMORY DREAMS] unavailable ({type(e9_).__name__}: {e9_})"[:260],flush=True)
    finally:
        del W0_; back_()

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
}
