"""End-of-run readouts: dual pathway, CLS probes, ceiling, transfer."""
from platypus.common import _fail
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
from platypus.conduit import (
    _carry_load, _carry_save,
)
from platypus.inference.store import (
    _INFM, _ng_zero, bpc,
)
from platypus.inference.units import (
    _INF_REL, _INF_SYL,
)
from platypus.inference.probes import (
    _INF, inf_probes,
)
from platypus.inference.sentences import (
    _inf_store,
)
from platypus.inference.links import (
    _LM, _cue_fix, _lm_partners, _lm_raw,
)
from platypus.inference.closed_form import (
    _snapcpu,
)


def inf_final_bpc(model,cfg,vals,S_):
    inf_need_report(model)
    """end of run: every domain's held-out BPC with the fast store OFF (weights only) and ON (same model): its direct cost"""
    if _INFM['M'] is None or _INFM.get('mu') is None: return
    r_={}
    try:
        for on_ in (False,True): _INFM['force']=on_; r_[on_]=[bpc(model,vals[d_],S_(d_)) for d_ in range(len(vals))]
    finally: _INFM.pop('force',None)
    print(f"  [inference final BPC] store OFF (weights only) {np.mean(r_[False]):.3f} | store ON {np.mean(r_[True]):.3f} | direct cost of the store {np.mean(r_[True])-np.mean(r_[False]):+.3f} | per domain off/on "
          +" ".join(f"{a_:.3f}/{b_:.3f}" for a_,b_ in zip(r_[False],r_[True]))+f" | store used in BPC during training: {_INFM.get('bpc_on',True)}",flush=True)
    if int(getattr(cfg,'inf_cls_probe',0)):
        try: _inf_cls_probe(model,cfg)
        except Exception as e_: _fail('inference.readouts:47'); print(f"  [CLS PROBES] failed: {type(e_).__name__}: {e_}",flush=True)
    if _INF.get('known_c'):
        try:                                                                    # END OF RUN: weights-only two-hop, known-word vs invented chain ends
            ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd')]; rsx_=np.random.RandomState(5151); sl_=lambda c_: int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0]))
            out_=[]
            for nm0_,grp_ in (('known-word ends',[c_ for c_ in ch_ if c_.get('known')]),('invented ends',[c_ for c_ in ch_ if not c_.get('known')])):
                Cs0_=[c_['names'][2] for c_ in grp_]; a_={True:[],False:[]}
                for c_ in grp_:
                    o_=[x_ for x_ in Cs0_ if x_!=c_['names'][2]]; rsx_.shuffle(o_); cd_=[c_['names'][2]]+o_[:9]
                    for st0_ in (False,True): a_[st0_].append(_inf_score1(model,cfg,f"So {c_['names'][0]} is connected to ",cd_,sl_(c_),st0_)==0)
                out_.append(f"{nm0_} (n {len(grp_)}): weights only {np.mean(a_[False]) if a_[False] else float('nan'):.2f}, with the store {np.mean(a_[True]) if a_[True] else float('nan'):.2f}")
            print("  [inference KNOWN-WORD PROBES | END OF RUN] two-hop, candidates from the same kind (chance 0.10): "+" | ".join(out_),flush=True)
        except Exception as e_: _fail('inference.readouts:59'); print(f"  [inference KNOWN-WORD PROBES] failed: {type(e_).__name__}: {e_}",flush=True)
    if int(getattr(cfg,'inf_dual_probe',0)) and _LM.get('M') is not None:
        try: _inf_dual_probe(model,cfg)
        except Exception as e_: _fail('inference.readouts:62'); print(f"  [inference DUAL PATHWAY] failed: {type(e_).__name__}: {e_}",flush=True)
    if int(getattr(cfg,'inf_lm_probe',0)):
        try: _inf_lm_probe(model,cfg)
        except Exception as e_: _fail('inference.readouts:65'); print(f"  [inference CLS PROBES] failed: {type(e_).__name__}: {e_}",flush=True)
def _inf_dual_probe(model,cfg):
    """DUAL PATHWAY ARBITRATION. Two-hop questions 'So A is connected to' on chains the system concluded, cue intact or damaged. EXACT pathway:
    the model with the fast store (exact byte keys) - a distribution over 10 candidates from its log-probs. OVERLAPPING pathway: the link memory
    (spelling-aware codes) - strengths of the candidates as partners of the cue (direct, plus by path through its strongest partners), turned
    into a distribution. DUAL: (i) product of the two distributions; (ii) whichever pathway is more confident (higher top probability)."""
    S0_=_inf_store(model); rs_=np.random.RandomState(9191); bt_=float(getattr(cfg,'inf_dual_beta',4.0))
    cm_={frozenset((c_['A'],c_['C'])) for c_ in S0_['conc'] if not c_.get('freed')}
    ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd') and frozenset((c_['names'][0],c_['names'][2])) in cm_][:40]
    if len(ch_)<4: print(f"  [inference DUAL PATHWAY] only {len(ch_)} concluded chains",flush=True); return
    Cs_=[c_['names'][2] for c_ in ch_]; slot_=lambda c_: int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0]))
    I_,S0a_=np.stack(_LM['idx']),np.stack(_LM['sgn'])
    def part(n_):
        if n_ in _LM['codes']: return _lm_partners(n_)
        ix_,sg_=_lm_raw(n_); v_=sg_@_LM['M'][ix_,:]; return (v_[I_]*S0a_).sum(1)/(_LM['k']**2)
    def sm(v_): v_=np.asarray(v_,np.float64); v_=v_-v_.max(); e_=np.exp(v_); return e_/e_.sum()
    dmg_={'intact':lambda n_: n_,'last letter dropped':lambda n_: n_[:-1],'inner letters swapped':lambda n_: n_[:1]+n_[2:3]+n_[1:2]+n_[3:] if len(n_)>3 else n_,'lower-cased':lambda n_: n_.lower()}
    seg_=[]
    for k_,f_ in dmg_.items():
        r_={'exact':[],'overlap':[],'product':[],'confident':[],'recognised':[]}; ch0_=[]; nrec_=0
        for c_ in ch_:
            o_=[x_ for x_ in Cs_ if x_!=c_['names'][2]]; rs_.shuffle(o_); cd_=[c_['names'][2]]+o_[:9]; A_=f_(c_['names'][0])
            lp_=_inf_lp(model,cfg,f"So {A_} is connected to ",cd_,slot_(c_),True); pe_=sm(lp_)
            sc_=part(A_.encode('latin-1')); top_=np.argsort(-sc_)[:3]
            ov_=[]
            for x_ in cd_:
                xb_=x_.encode('latin-1')
                if xb_ not in _LM['codes']: ov_.append(0.0); continue
                j_=_LM['codes'][xb_][2]; ov_.append(float(sc_[j_])/9.0+sum(max(0.0,float(sc_[t_]))*float(part(_LM['names'][t_])[j_]) for t_ in top_ if t_!=j_)/9.0)
            po_=sm(bt_*np.array(ov_)); pp_=pe_*po_; pp_=pp_/pp_.sum()
            r_['exact'].append(int(np.argmax(pe_)==0)); r_['overlap'].append(int(np.argmax(po_)==0)); r_['product'].append(int(np.argmax(pp_)==0))
            ce_=pe_ if pe_.max()>=po_.max() else po_; r_['confident'].append(int(np.argmax(ce_)==0)); ch0_.append(pe_.max()>=po_.max())
            rc_=A_.encode('latin-1') in _INFM.get('names',{}); nrec_+=int(rc_)                   # RECOGNITION gate: the cue is a name the exact store
            r_['recognised'].append(int(np.argmax(pe_ if rc_ else po_)==0))                        #   knows -> exact recall; unknown -> completion
        seg_.append(f"{k_}: exact {np.mean(r_['exact']):.2f}, overlapping {np.mean(r_['overlap']):.2f}, DUAL product {np.mean(r_['product']):.2f}, DUAL more-confident {np.mean(r_['confident']):.2f} (exact chosen {np.mean(ch0_):.2f}), DUAL recognition-gated {np.mean(r_['recognised']):.2f} (cue recognised {nrec_/max(1,len(ch_)):.2f})")
    print(f"  [inference DUAL PATHWAY | two-hop, {len(ch_)} concluded chains, 10 candidates, chance 0.10] "+" | ".join(seg_),flush=True)
def _inf_lm_probe(model,cfg):
    """FUZZY RECALL PROBES on the link memory. For two-hop chains A-B-C whose names it holds: (1) one-hop: rank the true partner B among 10
    candidate bridges by memory strength from the cue A; (2) two-hop: rank C among 10 candidates - direct (a promoted conclusion link) and by
    path (strongest partners of the cue, then their partners). Cues: intact and damaged (last letter dropped, inner letters swapped, lower-cased).
    Damaged cues are coded on the fly (not added). Also: VARIANT-BRIDGE chains found as conclusions vs ordinary chains."""
    ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd') and all(n_.encode('latin-1') in _LM.get('codes',{}) for n_ in c_['names'])] if _LM.get('M') is not None else []
    if ch_: _inf_lm_recall(ch_)
    _inf_variant_readout(model,cfg)
def _inf_lm_recall(ch_):
    rs_=np.random.RandomState(4242); Bs_=[c_['names'][1].encode('latin-1') for c_ in ch_]; Cs_=[c_['names'][2].encode('latin-1') for c_ in ch_]
    I_,S0_=np.stack(_LM['idx']),np.stack(_LM['sgn'])
    def part(n_):
        if n_ in _LM['codes']: return _lm_partners(n_)
        ix_,sg_=_lm_raw(n_); v_=sg_@_LM['M'][ix_,:]; return (v_[I_]*S0_).sum(1)/(_LM['k']**2)
    def rank1(sc_,true_,pool_):
        o_=[x_ for x_ in pool_ if x_!=true_]; rs_.shuffle(o_); cd_=[true_]+o_[:9]
        v_=[sc_[_LM['codes'][x_][2]] for x_ in cd_]; return int(np.argmax(v_)==0)
    dmg_={'intact':lambda n_: n_,'last letter dropped':lambda n_: n_[:-1],'inner letters swapped':lambda n_: n_[:1]+n_[2:3]+n_[1:2]+n_[3:] if len(n_)>3 else n_,'lower-cased':lambda n_: n_.lower()}
    seg_=[]
    for k_,f_ in dmg_.items():
        h1_=[];h2d_=[];h2p_=[]
        for c_,B_,C_ in zip(ch_,Bs_,Cs_):
            A_=f_(c_['names'][0].encode('latin-1')); sc_=part(A_)
            h1_.append(rank1(sc_,B_,Bs_)); h2d_.append(rank1(sc_,C_,Cs_))
            top_=np.argsort(-sc_)[:3]; pth_=sum(max(0.0,float(sc_[j_]))*part(_LM['names'][j_]) for j_ in top_)
            h2p_.append(rank1(pth_,C_,Cs_) if not isinstance(pth_,float) else 0)
        seg_.append(f"{k_}: one-hop {np.mean(h1_):.2f}, two-hop direct {np.mean(h2d_):.2f}, two-hop by path {np.mean(h2p_):.2f}")
    print(f"  [inference FUZZY RECALL | link memory, codes exact"+(f" + spelling {_LM['fz']:.2f}" if _LM.get('fz',0)>0 else " only")+f"] {len(ch_)} chains, chance 0.10 | "+" | ".join(seg_),flush=True)
def _inf_variant_readout(model,cfg):
    S0s_=_inf_store(model); cm_=set(frozenset((c_['A'],c_['C'])) for c_ in S0s_['conc'] if not c_.get('freed'))
    va_=[c_ for c_ in inf_probes(cfg) if c_.get('var')]; no_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('var') and not c_.get('upd')]
    if va_:
        fv_=np.mean([frozenset((c_['names'][0],c_['names'][2])) in cm_ for c_ in va_]); fn_=np.mean([frozenset((c_['names'][0],c_['names'][2])) in cm_ for c_ in no_]) if no_ else float('nan')
        print(f"  [inference VARIANT BRIDGES] chains whose bridge is spelled differently in its two facts: concluded {fv_:.2f} (n {len(va_)}) | ordinary two-hop chains concluded {fn_:.2f} (n {len(no_)})",flush=True)
def _gpu_mem_report(tag):
    """cg26q GPU MEMORY: what is still allocated before the heavy end-of-run readouts (two of them ran out of memory on the GPU with about 80 GiB
    held), grouped by tensor shape; tensors under 64 MiB are left out"""
    if not torch.cuda.is_available(): return
    try:
        import gc as _gc9; _gc9.collect(); torch.cuda.empty_cache()
        al_=torch.cuda.memory_allocated()/2**30; rv_=torch.cuda.memory_reserved()/2**30; big_={}
        for o_ in _gc9.get_objects():
            try:
                if torch.is_tensor(o_) and o_.is_cuda:
                    n_=o_.numel()*o_.element_size()
                    if n_>=2**26: k_=(tuple(o_.shape),str(o_.dtype).replace('torch.','')); v_=big_.setdefault(k_,[0,0]); v_[0]+=1; v_[1]+=n_
            except Exception: pass
        top_=sorted(big_.items(),key=lambda kv_:-kv_[1][1])[:10]
        print(f"  [GPU MEMORY | {tag}] allocated {al_:.1f} GiB, reserved {rv_:.1f} GiB | largest live tensors (count x shape dtype: GiB): "
              +" | ".join(f"{v_[0]} x {list(k_[0])} {k_[1]}: {v_[1]/2**30:.2f}" for k_,v_ in top_),flush=True)
    except Exception as e_: _fail('inference.readouts:151'); print(f"  [GPU MEMORY | {tag}] unavailable ({type(e_).__name__})",flush=True)
def _inf_cls_probe(model,cfg):
    """END-OF-RUN CLS PROBES. (1) LESION BY AGE: two-hop with the store ON vs OFF for chains concluded early vs late (complementary learning
    predicts older memories survive the lesion better once consolidation works). (2) PARTIAL CUES: two-hop with the store when the subject's
    name is damaged (last letter dropped, inner letters swapped, lower-cased). (3) SCHEMA TEST: a few gradient steps on the one-hop facts of
    brand-new chains (weights copied and restored afterwards): how fast the trunk absorbs new facts, and whether it then chains them."""
    S0_=_inf_store(model); ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd')]; rs_=np.random.RandomState(777)
    slot_=lambda c_: int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0]))
    Cs_=[c_['names'][2] for c_ in ch_]
    def cands(c_):
        o_=[x_ for x_ in Cs_ if x_!=c_['names'][2]]; rs_.shuffle(o_); return [c_['names'][2]]+o_[:9]
    def acc(items,store_):
        if not items: return float('nan')
        return float(np.mean([_inf_score1(model,cfg,pr_,cd_,sl_,store_)==0 for pr_,cd_,sl_ in items]))
    cm_={(c_['A'],c_['C']):c_ for c_ in S0_['conc'] if not c_.get('freed')}
    found_=[(c_,cm_.get((c_['names'][0],c_['names'][2])) or cm_.get((c_['names'][2],c_['names'][0]))) for c_ in ch_]
    found_=[(c_,k_) for c_,k_ in found_ if k_ is not None]
    if len(found_)>=4:
        md_=float(np.median([k_.get('v',0) for _,k_ in found_])); grp_={'OLDER':[],'RECENT':[]}
        for c_,k_ in found_: grp_['OLDER' if k_.get('v',0)<=md_ else 'RECENT'].append((f"So {c_['names'][0]} is connected to ",cands(c_),slot_(c_)))
        seg_=[f"{g_} (n {len(it_)}): store ON {acc(it_,True):.2f}, store OFF (lesion) {acc(it_,False):.2f}" for g_,it_ in grp_.items()]
        print(f"  [inference CLS PROBES | LESION BY AGE] two-hop on chains the system concluded, split at conclusion visit {md_:.0f}: "+" | ".join(seg_),flush=True)
    dmg_={'intact':lambda n_: n_,'last letter dropped':lambda n_: n_[:-1],'inner letters swapped':lambda n_: n_[0]+n_[2]+n_[1]+n_[3:] if len(n_)>3 else n_,'lower-cased':lambda n_: n_.lower()}
    pc_=[]
    for k0_,f_ in dmg_.items():
        it_=[(f"So {f_(c_['names'][0])} is connected to ",cands(c_),slot_(c_)) for c_,_ in found_[:40]]
        pc_.append(f"{k0_} {acc(it_,True):.2f}")
    print(f"  [inference CLS PROBES | PARTIAL CUES] two-hop with the store, subject name damaged (n {min(40,len(found_))} found chains): "+" | ".join(pc_),flush=True)
    # (3) schema test on brand-new chains
    used_=set(n_ for c_ in inf_probes(cfg) for n_ in c_['names']); nw_=[]
    def nm():
        while True:
            n_=''.join(rs_.choice(_INF_SYL,rs_.randint(2,4))); n_=n_[0].upper()+n_[1:]
            if 5<=len(n_)<=10 and any(x_ in n_.lower() for x_ in 'qxz') and n_ not in used_: used_.add(n_); return n_
    NS_=int(getattr(cfg,'inf_schema_n',16))
    for _ in range(NS_):
        A_,B_,C_=nm(),nm(),nm(); t0_,t1_=_INF_REL[rs_.randint(len(_INF_REL))],_INF_REL[rs_.randint(len(_INF_REL))]
        nw_.append((A_,B_,C_,t0_.format(a=A_,b=B_),t1_.format(a=B_,b=C_)))
    sl0_=int(model.__dict__.get('_t2s',{}).get(0,0)); Cn_=[c_[2] for c_ in nw_]; Bn_=[c_[1] for c_ in nw_]
    def candn(x_,pool_):
        o_=[y_ for y_ in pool_ if y_!=x_]; rs_.shuffle(o_); return [x_]+o_[:9]
    def accn(two_):
        it_=[(f"So {a_} is connected to ",candn(c_,Cn_),sl0_) if two_ else (f"{f0_.rsplit(' ',1)[0]} ",candn(b_,Bn_),sl0_) for a_,b_,c_,f0_,f1_ in nw_]
        return acc(it_,False)
    b1_,b2_=accn(False),accn(True); _SCORE_LEN['n']=int(getattr(cfg,'inf_schema_ctxlen',0))
    ic_=[(f"{f1_} {f0_} So {a_} is connected to ",candn(c_,Cn_),sl0_) for a_,b_,c_,f0_,f1_ in nw_ if len(f"{f1_} {f0_} So {a_} is connected to {c_}.")<=max(int(cfg.seq_len),int(getattr(cfg,'inf_schema_ctxlen',0)))]   # IN CONTEXT (audit fix:
    bic_=acc(ic_,False) if ic_ else float('nan')                                #   only chains whose whole prompt + answer fits the window are scored)
    sd_=_snapcpu(model); dev_=model.embed.weight.device; was_=model.training
    opt_=torch.optim.Adam([q_ for q_ in model.parameters() if q_.requires_grad],lr=float(getattr(cfg,'inf_cls_lr',3e-4))); K_=int(getattr(cfg,'inf_cls_steps',30))
    txt_=[f0_ for _,_,_,f0_,_ in nw_]+[f1_ for *_,f1_ in nw_]
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl0_; _INFM['force']=False
    try:
        model.train()
        for st_ in range(K_):
            bt_=[txt_[i_] for i_ in rs_.choice(len(txt_),min(16,len(txt_)),replace=False)]; L0_=max(len(t_) for t_ in bt_)+1
            X_=torch.full((len(bt_),L0_),32,dtype=torch.long,device=dev_)
            for j_,t_ in enumerate(bt_): X_[j_,:len(t_)]=torch.tensor(list(t_.encode('latin-1')),device=dev_)
            lg_=model(X_[:,:-1],sl0_); loss_=F.cross_entropy(lg_.float().reshape(-1,lg_.shape[-1]),X_[:,1:].reshape(-1))
            opt_.zero_grad(); loss_.backward(); opt_.step()
        model.eval(); a1_,a2_=accn(False),accn(True)
    finally:
        _carry_load(sv4_); _XDG['cur']=sc_; _INFM.pop('force',None)
        with torch.no_grad(): model.load_state_dict(sd_)
        model.train(was_); del opt_
        try: _ng_zero(model)
        except Exception: pass
    if int(getattr(cfg,'inf_ceiling2',0)):
        _gpu_mem_report('before CEILING LONG')
        for lean_ in (0,1):                                                     # cg26q: after running out of GPU memory, retry once without optimizer state for the large tables
            cfg._ceil2_lean=lean_
            try: _inf_ceiling2(model,cfg,nm,sl0_,acc,rs_); break
            except Exception as e_:
                _fail('inference.readouts:222')
                oom_='out of memory' in str(e_).lower() and torch.cuda.is_available()
                import traceback as _tb; print(f"  [inference CEILING LONG] FAILED{' (retrying lean)' if oom_ and not lean_ else ''}: {type(e_).__name__}: {str(e_)[:200]} | {_tb.format_exc()[-300:]!r}",flush=True)
                if not oom_: break
                import gc as _gc9; _gc9.collect(); torch.cuda.empty_cache()
        cfg._ceil2_lean=0
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
    if int(getattr(cfg,'inf_ceiling',0)):
        try: _inf_ceiling(model,cfg,nm,sl0_,acc,candn,rs_)
        except Exception as e_:
            _fail('inference.readouts:233')
            import traceback as _tb; print(f"  [inference CEILING TEST] FAILED: {type(e_).__name__}: {e_} | {_tb.format_exc()[-500:]!r}",flush=True)
    _SCORE_LEN['n']=0
    print(f"  [inference CLS PROBES | SCHEMA TEST, IN CONTEXT] {len(ic_)} of {NS_} brand-new chains fit the window whole, both premises in the prompt, store off: two-hop {bic_:.2f} (chance 0.10)",flush=True)
    print(f"  [inference CLS PROBES | SCHEMA TEST] {NS_} brand-new chains, store off, weights only: before learning one-hop {b1_:.2f} two-hop {b2_:.2f} -> after {K_} steps on their one-hop facts: one-hop {a1_:.2f} two-hop {a2_:.2f} (weights restored)",flush=True)

def _inf_ceiling(model,cfg,nm,sl0_,acc,candn,rs_):
    """COMPOSITION CEILING (can this trunk compose at all?). Brand-new chains, weights copied and restored after each arm.
    (W) IN THE WEIGHTS: train on TRAINING chains' one-hop facts AND their two-hop statements ('So A is connected to C.'), plus the HELD-OUT
        chains' one-hop facts only; then ask the held-out two-hop. Training two-hop = memorising a stated link; held-out two-hop above chance
        = composing two facts met apart (the 'two-hop curse': large transformers mostly fail this without special training).
    (C) IN CONTEXT: train on whole worked examples 'fact fact So A is connected to C.' of training chains (either premise order); then give
        held-out chains' two premises in the prompt and ask. Above chance = the trunk can compose facts it reads together."""
    _SCORE_LEN['n']=max(_SCORE_LEN['n'],160)
    NT_=int(getattr(cfg,'inf_ceiling_train',192)); NH_=int(getattr(cfg,'inf_ceiling_held',64)); K_=int(getattr(cfg,'inf_ceiling_steps',400))
    def chains(n_):
        o_=[]
        for _ in range(n_):
            A_,B_,C_=nm(),nm(),nm(); t0_,t1_=_INF_REL[rs_.randint(len(_INF_REL))],_INF_REL[rs_.randint(len(_INF_REL))]
            o_.append((A_,B_,C_,t0_.format(a=A_,b=B_),t1_.format(a=B_,b=C_)))
        return o_
    TR_,HE_=chains(NT_),chains(NH_); Ct_=[c_[2] for c_ in TR_]; Ch_=[c_[2] for c_ in HE_]; Bh_=[c_[1] for c_ in HE_]
    two_=lambda L_,P_: [(f"So {a_} is connected to ",candn(c_,P_),sl0_) for a_,b_,c_,f0_,f1_ in L_]
    one_=lambda L_,P_: [(f"{f0_.rsplit(' ',1)[0]} ",candn(b_,P_),sl0_) for a_,b_,c_,f0_,f1_ in L_]
    ctx_=lambda L_,P_: [(f"{f1_} {f0_} So {a_} is connected to ",candn(c_,P_),sl0_) for a_,b_,c_,f0_,f1_ in L_]
    dev_=model.embed.weight.device; was_=model.training; sd_=_snapcpu(model)
    def train(txt_):
        opt_=torch.optim.Adam([q_ for q_ in model.parameters() if q_.requires_grad],lr=float(getattr(cfg,'inf_cls_lr',3e-4))); model.train(); l_=[]
        for st_ in range(K_):
            bt_=[txt_[i_] for i_ in rs_.choice(len(txt_),min(16,len(txt_)),replace=False)]; L0_=max(len(t_) for t_ in bt_)+1
            X_=torch.full((len(bt_),L0_),32,dtype=torch.long,device=dev_)
            for j_,t_ in enumerate(bt_): X_[j_,:len(t_)]=torch.tensor(list(t_.encode('latin-1')),device=dev_)
            lg_=model(X_[:,:-1],sl0_); loss_=F.cross_entropy(lg_.float().reshape(-1,lg_.shape[-1]),X_[:,1:].reshape(-1))
            opt_.zero_grad(); loss_.backward(); opt_.step(); l_.append(float(loss_))
        model.eval(); del opt_; return l_
    def restore():
        with torch.no_grad(): model.load_state_dict(sd_)
        try: _ng_zero(model)
        except Exception: pass
    sv_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl0_; _INFM['force']=False; R_={}
    try:
        model.eval(); R_['pre']=(acc(two_(HE_,Ch_),False),acc(ctx_(HE_,Ch_),False))
        txtW_=[f0_ for *_,f0_,f1_ in TR_]+[f1_ for *_,f1_ in TR_]+[f"So {a_} is connected to {c_}." for a_,b_,c_,*_ in TR_]+[f0_ for *_,f0_,f1_ in HE_]+[f1_ for *_,f1_ in HE_]
        lW_=train(txtW_); R_['W']=(acc(two_(TR_,Ct_),False),acc(one_(HE_,Bh_),False),acc(two_(HE_,Ch_),False),np.mean(lW_[:20]),np.mean(lW_[-20:])); restore()
        txtC_=[f"{f0_} {f1_} So {a_} is connected to {c_}." for a_,b_,c_,f0_,f1_ in TR_]+[f"{f1_} {f0_} So {a_} is connected to {c_}." for a_,b_,c_,f0_,f1_ in TR_]
        lC_=train(txtC_); R_['C']=(acc(ctx_(TR_,Ct_),False),acc(ctx_(HE_,Ch_),False),np.mean(lC_[:20]),np.mean(lC_[-20:])); restore()
    finally:
        restore(); _carry_load(sv_); _XDG['cur']=sc_; _INFM.pop('force',None); model.train(was_)
    w_,c_=R_['W'],R_['C']
    print(f"  [inference CEILING TEST] {NT_} training / {NH_} held-out brand-new chains, {K_} steps per arm, store off, chance 0.10 | before: held-out two-hop {R_['pre'][0]:.2f}, in context {R_['pre'][1]:.2f} || "
          f"(W) IN THE WEIGHTS (training chains taught their two-hop, held-out chains only their one-hop): training two-hop {w_[0]:.2f} | held-out one-hop {w_[1]:.2f} | HELD-OUT TWO-HOP {w_[2]:.2f} (loss {w_[3]:.2f} -> {w_[4]:.2f}) || "
          f"(C) IN CONTEXT (worked examples of training chains): training chains in context {c_[0]:.2f} | HELD-OUT IN CONTEXT {c_[1]:.2f} (loss {c_[2]:.2f} -> {c_[3]:.2f})",flush=True)

def _inf_conc_row(c_,Lr_,filler,rs_,fa_):
    """a conclusion row (as the store writes): the sentence alone at the start (share fa_) or planted in same-domain filler; mask = its bytes"""
    sn_=c_['s'] if (c_['typed'] and rs_.rand()<0.5) else (f"So {c_['A']} is connected to {c_['C']}." if rs_.rand()<0.5 else f"So {c_['C']} is connected to {c_['A']}.")
    if rs_.rand()<fa_:
        it_=(sn_+' ').encode('latin-1')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); o_=0
    else:
        w_=bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); it_=(' '+sn_+' ').encode('latin-1')[:Lr_-8]; o_=rs_.randint(4,max(5,Lr_-len(it_)-4)); w_[o_:o_+len(it_)]=it_
    w_=w_[:Lr_]; mk_=torch.zeros(Lr_-1); mk_[max(0,o_-1) if o_>0 else 0:o_+len(it_)-1]=1.0
    return torch.tensor(list(bytes(w_)),dtype=torch.long),mk_
def _inf_xfer(model,cfg,t,ts,vals,olds_,S_slot,S_,allc_,ring_,Lr_,kc_,ka_,R_,filler,fwd,rs_,dev_,nb_,fa_):
    """STORE -> WEIGHTS TRANSFER (cg20 test 2): 'teach' = small steps of KL(pre-sleep weights + store || weights) on conclusion rows, anchors
    KL(pre-sleep weights || weights) on recent windows; 'brake' = the same, rolled back if store-off held-out BPC rises more than inf_xfer_eps"""
    mode_=str(getattr(cfg,'inf_xfer','none')); xs_=int(getattr(cfg,'inf_xfer_steps',0)); live_=[c_ for c_ in allc_ if not c_.get('freed')]
    if mode_ not in ('teach','brake','skill') or xs_<=0 or not live_: return ''
    if mode_=='skill':                                                          # SKILL TRANSFER: worked examples (sources + conclusion) from the store, never a held-out chain
        ho_=_inf_heldout(cfg); live_=[c_ for c_ in live_ if not ({c_['A'],c_['B'],c_['C']}&ho_)]
        if not live_: return ''
    V0_=int(model.__dict__.get('_visit',0)); sinfo_=''
    if int(getattr(cfg,'inf_xfer_salience',0)):                                 # SALIENCE: used (re-encountered) and older conclusions are consolidated first
        live_=sorted(live_,key=lambda c_: -(c_.get('use',0)+(V0_-c_.get('v',V0_))))[:int(getattr(cfg,'inf_xfer_n',64))]
        sinfo_=f", salience-selected {len(live_)} (mean use {np.mean([c_.get('use',0) for c_ in live_]):.1f}, mean age {np.mean([V0_-c_.get('v',V0_) for c_ in live_]):.1f} visits)"
    sp_=int(getattr(cfg,'inf_xfer_space',0)); sched_=None
    if sp_>0:                                                                   # SPACING: every selected conclusion gets exactly inf_xfer_space rows this sleep
        sched_=[c_ for c_ in live_ for _ in range(sp_)]; rs_.shuffle(sched_); xs_=min(xs_,max(1,int(math.ceil(len(sched_)/max(1,kc_)))))
    old_=[d_ for d_ in (S_.get('res') or {}) if d_!=int(t) and S_['res'][d_]]; fo_=float(getattr(cfg,'inf_xfer_old',0.0))
    PA_=[p_ for p_ in model.parameters() if p_.requires_grad]; TP_=[p_.detach().clone() for p_ in PA_]
    def bpc_off():
        _INFM['force']=False
        try: return float(np.mean([bpc(model,vals[t],ts,n=nb_)]+[bpc(model,vals[d_],s_,n=nb_) for d_,s_ in olds_]))
        finally: _INFM.pop('force',None)
    b0_=bpc_off() if mode_ in ('brake','skill') else float('nan')
    def teacher(X_,d_):
        sv_=[p_.data for p_ in PA_]
        for p_,q_ in zip(PA_,TP_): p_.data=q_
        _INFM['force']=True
        try:
            with torch.no_grad(): return F.log_softmax(fwd(X_,d_),-1)
        finally:
            _INFM.pop('force',None)
            for p_,v_ in zip(PA_,sv_): p_.data=v_
    op_=torch.optim.Adam(PA_,lr=float(cfg.lr)*float(getattr(cfg,'inf_xfer_lr',0.05))); was_=model.training; model.train(); ls_=[]
    with torch.enable_grad():
        for st_ in range(xs_):
            grp_={}
            nk_=0; tries_=0; edge_=(mode_=='skill' and str(getattr(cfg,'inf_xfer_sel','random'))=='edge'); cand_=[]
            while (len(cand_) if edge_ else nk_)<(4*kc_ if edge_ else kc_) and tries_<16*kc_:
                if sched_ is not None:
                    if not sched_: break
                    c_=sched_.pop(); tries_+=1
                else: c_=live_[rs_.randint(len(live_))]; tries_+=1
                r0_=_inf_skill_row(c_,S_,Lr_,rs_) if mode_=='skill' else _inf_conc_row(c_,Lr_,filler,rs_,fa_)
                if r0_ is None: continue
                if edge_: cand_.append((c_,r0_)); continue
                grp_.setdefault(int(S_slot(c_['d'])),[]).append((r0_[0],r0_[1],mode_=='skill')); nk_+=1
            if edge_ and cand_:                                                 # EDGE OF COMPETENCE: the model's own in-context answer probability picks the examples
                with torch.no_grad():
                    lpa_=[]
                    for c_,r0_ in cand_:
                        X1_=r0_[0][None].to(dev_); lg1_=F.log_softmax(fwd(X1_,int(S_slot(c_['d']))),-1)[0]; m1_=r0_[1].to(dev_)
                        lpa_.append(float((lg1_.gather(-1,X1_[0,1:,None]).squeeze(-1)*m1_).sum()))
                tg_=math.log(float(getattr(cfg,'inf_xfer_edge_p',0.3))); sc_=[(abs(l_-tg_) if l_<math.log(0.8) else 1e9) for l_ in lpa_]
                for j_ in np.argsort(sc_)[:kc_]:
                    c_,r0_=cand_[int(j_)]; grp_.setdefault(int(S_slot(c_['d'])),[]).append((r0_[0],r0_[1],True)); nk_+=1
                model.__dict__.setdefault('_edge_log',[]).append(float(np.mean([math.exp(lpa_[int(j_)]) for j_ in np.argsort(sc_)[:kc_]])))
            for _ in range(ka_ if ring_ else 0):
                if old_ and rs_.rand()<fo_: dd_=old_[rs_.randint(len(old_))]; w0_=S_['res'][dd_][rs_.randint(len(S_['res'][dd_]))]   # INTERLEAVING: old domains' windows
                else: w0_,dd_=ring_[rs_.randint(len(ring_))]
                o0_=rs_.randint(max(1,len(w0_)-Lr_+1)); xa_=torch.tensor(list(w0_[o0_:o0_+Lr_]),dtype=torch.long)
                if xa_.numel()<Lr_: xa_=torch.cat([xa_,torch.full((Lr_-xa_.numel(),),32,dtype=torch.long)])
                grp_.setdefault(int(S_slot(dd_)),[]).append((xa_,None,False))
            loss_=0.0
            for d_,rows_ in grp_.items():
                X_=torch.stack([r_[0] for r_ in rows_]).to(dev_); T_=teacher(X_,d_); L_=F.log_softmax(fwd(X_,d_),-1); kl_=(T_.exp()*(T_-L_)).sum(-1)
                ce_=-L_.gather(-1,X_[:,1:,None]).squeeze(-1); hard_=torch.tensor([bool(r_[2]) for r_ in rows_],device=dev_)[:,None]
                W_=torch.stack([r_[1] if r_[1] is not None else (X_[j_,1:]>0).float().cpu() for j_,r_ in enumerate(rows_)]).to(dev_)
                loss_=loss_+(torch.where(hard_,ce_,kl_)*W_).sum()/W_.sum().clamp(min=1)*len(rows_)/R_   # skill rows: CE on the answer bytes; others: KL
            op_.zero_grad(set_to_none=True); loss_.backward(); op_.step(); ls_.append(float(loss_.detach())); _ng_zero(model)
            for p_ in PA_: p_.grad=None
    model.train(was_); info_=f" | TRANSFER {mode_}: {xs_} steps at lr x{getattr(cfg,'inf_xfer_lr',0.05)}{sinfo_}{', spacing '+str(sp_)+' rows per conclusion' if sp_ else ''}{', '+str(int(100*fo_))+'% of anchors from '+str(len(old_))+' old domains' if fo_>0 else ''}, KL {ls_[0] if ls_ else float('nan'):.4f} -> {np.mean(ls_[-5:]) if ls_ else float('nan'):.4f}"+(f", edge selection: mean in-context answer probability of the chosen examples {np.mean(model.__dict__['_edge_log'][-xs_:]):.2f}" if model.__dict__.get('_edge_log') and str(getattr(cfg,'inf_xfer_sel','random'))=='edge' else '')
    if mode_ in ('brake','skill'):
        b1_=bpc_off(); eps_=float(getattr(cfg,'inf_xfer_eps',0.01))
        if b1_-b0_>eps_:
            with torch.no_grad():
                for p_,q_ in zip(PA_,TP_): p_.data.copy_(q_)
            info_+=f", BRAKE: store-off BPC {b0_:.3f} -> {b1_:.3f} > +{eps_} ROLLED BACK"
        else: info_+=f", brake ok: store-off BPC {b0_:.3f} -> {b1_:.3f}"
        model.__dict__.setdefault('_xfer_rb',[]).append(int(b1_-b0_>eps_))
    return info_
_SCORE_LEN={'n':0}                                                              # >0: score prompts up to this many bytes (CLS schema / ceiling only)
def _score_norm(cfg,sm_,nb_,dec_=False,raw_=False):
    """cg26w CANDIDATE SCORING: a candidate name's summed byte log-probability favours SHORT names when the model is unsure (cg26v diagnostic:
    summed pick 0.05 = chance against 0.29 per byte; 3 wrong picks in 4 shorter than the right name). READOUTS score by the per-byte MEAN
    (inf_score, 'mean'; 'sum' restores cg26v); DECISIONS (release, restore, freeing, rewriting, refresh order) keep their own scoring
    (inf_score_dec, 'sum'), so training is unchanged. raw_: (summed, bytes scored)"""
    if raw_: return sm_,nb_
    m_=str(getattr(cfg,'inf_score_dec','sum') if dec_ else getattr(cfg,'inf_score','mean'))
    return sm_/np.maximum(nb_,1.0) if m_=='mean' else sm_
@torch.no_grad()
def _inf_score1(model,cfg,prompt_,cands_,slot_,store_,dec_=False):
    if store_: prompt_=_cue_fix(cfg,prompt_)                                     # cg26m CUE REPAIR (question time, store on)
    seqs=[(prompt_+c_+'.').encode('latin-1')[-max(int(cfg.seq_len),_SCORE_LEN['n']):] for c_ in cands_]; L_=max(len(s_) for s_ in seqs); dev_=model.embed.weight.device
    X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
    for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot_; _INFM['force']=bool(store_); was_=model.training; model.eval()
    try: lp_=F.log_softmax(model(X_[:,:-1],slot_).float(),-1)
    finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM.pop('force',None); model.train(was_)
    n0_=[len(s_)-len((c_+'.').encode()) for s_,c_ in zip(seqs,cands_)]
    sm_=np.array([float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum()) for j_ in range(len(seqs))])
    return int(np.argmax(_score_norm(cfg,sm_,np.array([len(seqs[j_])-n0_[j_] for j_ in range(len(seqs))],dtype=np.float64),dec_)))
@torch.no_grad()
def _cai_score(model,cfg):
    """CAI TEST (readout only): per chain, two-hop 'So A is connected to' against 19 fixed distractors, from the weights alone and with the fast store"""
    ch_=inf_probes(cfg); alln_=[n_ for c_ in ch_ for n_ in c_['names']]; out_={}
    son_=_INFM.get('M') is not None and _INFM.get('mu') is not None; sv_=_INFM['on']; sk_=_INF.get('corr',False)
    try:
        for i_,c_ in enumerate(ch_):
            n_=c_['names']
            if len(n_)<3 or (c_.get('upd') or {}).get('type')=='true': continue
            rs_=np.random.RandomState(9100+i_); oth_=[x_ for x_ in alln_ if x_ not in n_]; cs_=[n_[2]]+list(rs_.choice(oth_,19,replace=False))
            sl_=int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0])); pr_=f"So {n_[0]} is connected to "
            _INFM['on']=False; _INF['corr']=False; w_=int(_inf_score1(model,cfg,pr_,cs_,sl_,False)==0)
            if son_: _INFM['on']=True; _INF['corr']=True; s_=int(_inf_score1(model,cfg,pr_,cs_,sl_,True)==0)
            else: s_=w_
            out_[i_]=(w_,s_)
    finally: _INFM['on']=sv_; _INF['corr']=sk_
    return out_
def _cai_report(model,cfg,td_,pre_,post_):
    """CAI TEST: unsolved before this sleep -> solved after, for PRIMED chains (a premise read in the visit just before) and UNPRIMED ones"""
    ch_=inf_probes(cfg); S_=_inf_store(model); vd_=S_.get('_vdom',{}); vis_=int(model.__dict__.get('_visit',0)); C9_=S_.setdefault('_cai',{'last':{},'tot':{},'prev':None,'cyc':{}}); L_=C9_['last']; T_=C9_['tot']
    def grp(i_):
        ds_=[int(d_) for d_,_ in ch_[i_]['facts'][:2]]
        if any(vd_.get(d_,0)<1 for d_ in ds_): return None                 # a premise never read yet: not solvable, not counted
        if td_ not in ds_: return 'unprimed'
        ot_=[d_ for d_ in ds_ if d_!=td_]
        if not ot_: return 'primed near'
        gap_=vis_-max(L_.get(d_,-99) for d_ in ot_)
        return 'primed near' if gap_<=2 else 'primed remote'
    G_={}
    for i_ in pre_:
        g_=grp(i_)
        if g_ is None or i_ not in post_: continue
        for m_,nm_ in ((1,'store'),(0,'weights')):
            a_=G_.setdefault((g_,nm_),[0,0,0,0])                                # unsolved before, of those solved after, solved before, of those lost after
            if pre_[i_][m_]==0: a_[0]+=1; a_[1]+=post_[i_][m_]
            else: a_[2]+=1; a_[3]+=1-post_[i_][m_]
    for k_,a_ in G_.items():
        t_=T_.setdefault(k_,[0,0,0,0])
        for j_ in range(4): t_[j_]+=a_[j_]
    def fm(D_,nm_):
        seg_=[]
        for g_ in ('primed near','primed remote','primed','unprimed'):
            if g_=='primed':
                a_=[sum(D_.get((h_,nm_),[0,0,0,0])[j_] for h_ in ('primed near','primed remote')) for j_ in range(4)]
            else: a_=D_.get((g_,nm_),[0,0,0,0])
            seg_.append(f"{g_} {a_[1]}/{a_[0]}"+(f" ({a_[1]/a_[0]:.2f})" if a_[0] else '')+f", lost {a_[3]}/{a_[2]}")
        return ' | '.join(seg_)
    for nm_ in ('store','weights'):
        print(f"  [CAI TEST | visit {vis_}, domain {td_}, {nm_}] two-hop unsolved before this sleep -> solved after: {fm(G_,nm_)} || all sleeps so far: {fm(T_,nm_)}",flush=True)
    P_=C9_.get('prev')                                                        # ACROSS THE NEXT WAKE: chains primed at the previous sleep, pre-sleep then vs pre-sleep now
    if P_ is not None:
        p0_,pg_=P_; C_={}
        for i_,g_ in pg_.items():
            if i_ not in pre_ or i_ not in p0_ or g_ is None: continue
            for m_,nm_ in ((1,'store'),(0,'weights')):
                if p0_[i_][m_]==0: a_=C_.setdefault((g_.split(' ')[0],nm_),[0,0]); a_[0]+=1; a_[1]+=pre_[i_][m_]
        tc_=C9_.setdefault('cyc',{})
        for k_,a_ in C_.items():
            t_=tc_.setdefault(k_,[0,0]); t_[0]+=a_[0]; t_[1]+=a_[1]
        print("  [CAI TEST | across sleep and the next wake, all so far] unsolved before the previous sleep -> solved before this one: "+' | '.join(f"{g_} {nm_} {tc_.get((g_,nm_),[0,0])[1]}/{tc_.get((g_,nm_),[0,0])[0]}" for g_ in ('primed','unprimed') for nm_ in ('store','weights')),flush=True)
    C9_['prev']=(pre_,{i_:grp(i_) for i_ in pre_})
    L_[td_]=vis_

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_inf_ceiling2': 'platypus.inference.jepa',
    '_inf_heldout': 'platypus.inference.audit',
    '_inf_lp': 'platypus.inference.selftaught',
    '_inf_skill_row': 'platypus.inference.audit',
    'inf_need_report': 'platypus.inference.need',
}
