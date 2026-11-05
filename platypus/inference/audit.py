"""Hypotheses, pruning, abstention and the grounding audit."""
import math
import numpy as np
import re as _re_inf
import torch
import torch.nn.functional as F
from platypus.inference.store import (
    _INFM, _infm_corr, _infm_keys, _infm_write,
)
from platypus.inference.units import (
    _INF_NAME, _ent_ok,
)
from platypus.inference.probes import (
    _INF, inf_probes,
)
from platypus.inference.infer import (
    _inf_truth, _kgx_ho,
)
from platypus.inference.selftaught import (
    _inf_lp,
)

def _inf_heldout(cfg):
    """skill-transfer runs: every second probe chain is held out (its names never appear in a worked example)"""
    if _INF.get('ho') is None: _INF['ho']={n_ for i_,c_ in enumerate(inf_probes(cfg)) if i_%2==1 for n_ in c_['names']}
    return _INF['ho']
def _inf_skill_row(c_,S_,Lr_,rs_):
    """a WORKED EXAMPLE: the conclusion's two verified source sentences, then 'So A is connected to C.'; targets = C and the full stop"""
    A,C=c_['A'],c_['C']; sa_=S_['it'][c_['src'][0]]['s'].decode('latin-1').strip(); sc_=S_['it'][c_['src'][1]]['s'].decode('latin-1').strip()
    if A not in sa_: sa_,sc_=sc_,sa_
    if rs_.rand()<0.5: sa_,sc_=sc_,sa_
    t_=f"{sa_} {sc_} So {A} is connected to {C}."; b_=t_.encode('latin-1','replace')
    if len(b_)>Lr_-1: return None
    w_=b_+b' '*(Lr_-len(b_)); a0_=len(b_)-len((C+'.').encode('latin-1','replace')); mk_=torch.zeros(Lr_-1); mk_[a0_-1:len(b_)-1]=1.0
    return torch.tensor(list(w_),dtype=torch.long),mk_

def _inf_hyp(model,cfg,S_,nid_,st_):
    """TWO-TIER STORE: LOOSE hypothesis generation (rarity threshold x inf_hyp_df_mult, fan-out x inf_hyp_fan_mult) into an unverified tier
    that never touches predictions; a hypothesis is PROMOTED to a conclusion only with independent support: two different bridges"""
    H_=S_.setdefault('hyp',{}); df_=S_['df']; mx_=int(cfg.inf_df)*float(getattr(cfg,'inf_hyp_df_mult',2.0)); fan_=int(cfg.inf_maxfan)*float(getattr(cfg,'inf_hyp_fan_mult',3.0))
    lw_=S_.get('lw') if S_.get('strict') else None; ok_=lambda n_: _ent_ok(n_,df_,mx_) and (lw_ is None or n_.lower() not in lw_)
    done_={frozenset((c_['A'],c_['C'])) for c_ in S_['conc']}; br_=set(n_ for i_ in nid_ if i_<len(S_['it']) for n_ in _INF_NAME.findall(S_['it'][i_]['s']) if ok_(n_)); nh0_=len(H_)
    for B in br_:
        part={}; prel_={}
        for i_ in S_['idx'].get(B,[]):
            it_=S_['it'][i_]
            if it_['dep']>0 or it_.get('k')=='inf' or it_.get('sup'): continue   # hypotheses come from real / verified, current sentences only
            for A in set(_INF_NAME.findall(it_['s'])):
                if A!=B and ok_(A): part.setdefault(A,[]).append(i_)
        if len(part)>fan_: continue
        ks_=sorted(part)
        for x_ in range(len(ks_)):
            for y_ in range(x_+1,len(ks_)):
                A,C=ks_[x_],ks_[y_]; ia_=part[A][0]; ic_=part[C][0]
                if ia_==ic_: continue
                H_.setdefault((A,C),{})[B]=(ia_,ic_)
    cap_=int(getattr(cfg,'inf_hyp_cap',300000))
    while len(H_)>cap_: H_.pop(next(iter(H_)))                                    # oldest hypotheses forgotten first
    out_=[]
    for (A,C),bs_ in list(H_.items()):
        if len(bs_)<int(getattr(cfg,'inf_hyp_support',2)): continue
        key_=frozenset((A.decode('latin-1'),C.decode('latin-1')))
        if key_ in done_ or (set(S_['idx'].get(A,[]))&set(S_['idx'].get(C,[]))): H_.pop((A,C)); continue
        B,(ia_,ic_)=sorted(bs_.items())[0]
        c_={'A':A.decode('latin-1'),'B':B.decode('latin-1'),'C':C.decode('latin-1'),'r':(None,None),'typed':False,'s':f"So {A.decode('latin-1')} is connected to {C.decode('latin-1')}.",
            'src':(ia_,ic_),'d':S_['it'][ia_]['d'],'dep':1,'promoted':len(bs_)}
        if _kgx_ho(cfg,c_): H_.pop((A,C)); continue
        out_.append(c_); done_.add(key_); H_.pop((A,C))
        i2_=len(S_['it']); S_['it'].append({'s':c_['s'].encode('latin-1'),'d':c_['d'],'k':'inf','dep':0})
        for n_ in (A,C): S_['idx'].setdefault(n_,[]).append(i2_)
    S_['conc']+=out_
    print(f"  [inference two-tier store] hypotheses (unverified, never used for prediction): {len(H_)} (+{len(H_)-nh0_+len(out_)} this sleep) | PROMOTED with two independent bridges: {len(out_)}",flush=True)
    return out_
def _inf_use(S_,nid_,vis_):
    """usage: a conclusion is USED when one of its names appears again in newly stored (real or verified) text after it was made"""
    nm_=set(n_.decode('latin-1') for i_ in nid_ if i_<len(S_['it']) for n_ in _INF_NAME.findall(S_['it'][i_]['s']))
    for c_ in S_['conc']:
        if c_.get('v',vis_)<vis_ and (c_['A'] in nm_ or c_['C'] in nm_): c_['use']=c_.get('use',0)+1
@torch.no_grad()
def _inf_prune(model,cfg,S_,allc_,S_slot,Lr_,filler,feats,rs_,dev_,beta_):
    """PRUNING (selective sleep): conclusions never used (no name re-encountered) inf_prune_age visits after they were made have their store
    entries faded and leave the rehearsal pool"""
    if not int(getattr(cfg,'inf_prune',0)) or _INFM['M'] is None: return ''
    vis_=int(model.__dict__.get('_visit',0)); age_=int(getattr(cfg,'inf_prune_age',2))
    cand_=[c_ for c_ in allc_ if not c_.get('freed') and c_.get('use',0)==0 and vis_-c_.get('v',vis_)>=age_][:int(getattr(cfg,'inf_prune_n',200))]
    for c_ in cand_:
        c_['freed']=True; c_['pruned']=True
        for sn_ in [f"So {c_['A']} is connected to {c_['C']}.",f"So {c_['C']} is connected to {c_['A']}."]+([c_['s']] if c_['typed'] else []):
            it_=(sn_+' ').encode('latin-1','replace')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); w_=w_[:Lr_]
            X_=torch.tensor([list(bytes(w_))],dtype=torch.long,device=dev_); lg_,z_=feats(X_,int(S_slot(c_['d']))); K_,id_=_infm_keys(z_,X_[:,:-1],lg_)
            Kf_=K_.reshape(-1,K_.shape[-1]); idf_=id_.reshape(-1); lgf_=lg_.reshape(-1,256); p0_=F.softmax(lgf_,-1); mk_=torch.zeros(Lr_-1,device=dev_); mk_[:len(it_)-1]=1.0
            for _ in range(10): q_=F.softmax(lgf_+_infm_corr(Kf_,idf_),-1); _infm_write(Kf_,idf_,(p0_-q_)*mk_[:,None],beta_)
    t2_,t3_=_inf_truth(cfg); pr_=sum(1 for c_ in cand_ if (c_['A'],c_['C']) in t2_|t3_)
    return f" | PRUNED {len(cand_)} unused conclusions ({pr_} of them true probe chains; total pruned {sum(int(c_.get('pruned',False)) for c_ in allc_)} of {len(allc_)})"

def _inf_abstain_sources(S_,cr_,tag):
    """(1) ANSWER OR ABSTAIN: accuracy on the questions answered vs the share answered, for three policies; bluffs = answers to chains the
    system never found. (2) SHOW SOURCES: answers backed by a stored conclusion cite its two verified source sentences; checked against the truth"""
    cm_={frozenset((c_['A'],c_['C'])):c_ for c_ in S_['conc'] if not c_.get('freed')}
    ok_=np.array([r_[0] for r_ in cr_],bool); cf_=np.array([r_[1] for r_ in cr_]); hit_=np.array([r_[3] for r_ in cr_],bool)
    fnd_=np.array([frozenset((r_[4],r_[6])) in cm_ for r_ in cr_],bool); cite_=np.array([frozenset((r_[4],r_[5])) in cm_ for r_ in cr_],bool)
    pol_={'answer always':np.ones(len(cr_),bool),'answer if confidence >= 0.5':cf_>=0.5,'answer if a stored conclusion backs the answer':cite_,'backed and confidence >= 0.5':cite_&(cf_>=0.5)}
    seg_=[]
    for k_,a_ in pol_.items():
        seg_.append(f"{k_}: answers {a_.mean():.2f}, accuracy when answering {ok_[a_].mean() if a_.any() else float('nan'):.2f}, bluffs on unfound chains {(a_&~fnd_&~ok_).sum()}/{int((~fnd_).sum())}")
    print(f"  [inference answer-or-abstain | {tag}] two-hop with the store, n {len(cr_)} ({int(fnd_.sum())} chains found) || "+" || ".join(seg_),flush=True)
    if _INFM.get('gap_dbg'):                                                    # TWO-HOP GAP DIAGNOSTIC: where each probe question is lost
        ca_={}
        for c_ in S_['conc']: ca_.setdefault(frozenset((c_['A'],c_['C'])),[]).append(c_)
        cat_={}; exs_={}
        for r_ in cr_:
            k2_=frozenset((r_[4],r_[6])); cs_=ca_.get(k2_,[])
            if k2_ in cm_: k_='found, live: '+('right' if r_[0] else 'WRONG')+('' if cm_[k2_].get('typed') else ' (untyped)')
            elif cs_:
                c0_=cs_[-1]; k_='found then freed: '+('retracted' if c0_.get('retracted') else 'pruned' if c0_.get('pruned') else 'uncertain' if c0_.get('uncertain') else str(c0_.get('evicted','other')))
            else:
                k_='never proposed: '+('both ends seen' if (r_[4].encode('latin-1') in S_['idx'] and r_[6].encode('latin-1') in S_['idx']) else 'an end never seen')
            cat_[k_]=cat_.get(k_,0)+1; exs_.setdefault(k_,f"{r_[4]}->{r_[6]} (answered {r_[5]})")
        pm_={c_['names'][0]:c_ for c_ in (_INF.get('probes') or []) if c_.get('style')}
        if pm_:                                                                 # NATURAL PROSE: two-hop accuracy and chains found by the hardest style in the chain
            by_={}
            for r_ in cr_:
                c_=pm_.get(r_[4])
                if c_ is None: continue
                sts_=[x_ for x_ in c_['style'] if x_!='template'] or ['template']
                for x_ in set(sts_):
                    b_=by_.setdefault(x_,[0,0,0]); b_[0]+=1; b_[1]+=int(r_[0]); b_[2]+=int(frozenset((r_[4],r_[6])) in cm_)
            ng_=[c_ for c_ in (_INF.get('probes') or []) if c_.get('neg')]
            na_=sum(1 for c_ in ng_ for c2_ in S_['conc'] if not c2_.get('freed') and c_['neg'] in (c2_['A'],c2_['B'],c2_['C']))
            nf_=sum(1 for c_ in ng_ for f_ in S_.get('facts',[]) if not f_.get('freed') and c_['neg'] in (f_['A'],f_['B']))
            print(f"  [inference NATURAL PROSE | {tag}] two-hop by style (n, right, chain found): "+" | ".join(f"{k_} {v_[0]}: {v_[1]/max(1,v_[0]):.2f}, {v_[2]/max(1,v_[0]):.2f}" for k_,v_ in sorted(by_.items()))
                  +f" || NEGATION: {len(ng_)} negated distractors; recorded as facts {nf_}, used in live conclusions {na_} (both should be 0)",flush=True)
        pr0_={c_['names'][0]:c_ for c_ in (_INF.get('probes') or [])}; dd_=[]       # DISCOVERY DETAIL for chains never proposed / dropped
        for r_ in cr_:
            k2_=frozenset((r_[4],r_[6]))
            if k2_ in cm_ or r_[4] not in pr0_ or len(dd_)>=6: continue
            A_,B_,C_=[n_.encode('latin-1') for n_ in pr0_[r_[4]]['names'][:3]]; ix_=S_['idx']; df0_=S_.get('df',{})
            ab_=len(set(ix_.get(A_,[]))&set(ix_.get(B_,[]))); bc_=len(set(ix_.get(B_,[]))&set(ix_.get(C_,[])))
            fab_=sum(1 for f_ in S_.get('facts',[]) if {f_['A'],f_['B']}=={A_.decode('latin-1'),B_.decode('latin-1')}); fbc_=sum(1 for f_ in S_.get('facts',[]) if {f_['A'],f_['B']}=={B_.decode('latin-1'),C_.decode('latin-1')})
            fan_=len(set(n_ for i_ in ix_.get(B_,[]) for n_ in _INF_NAME.findall(S_['it'][i_]['s']))-{B_})
            dd_.append(f"{A_.decode()}-{B_.decode()}-{C_.decode()}: df {df0_.get(A_,0)}/{df0_.get(B_,0)}/{df0_.get(C_,0)}, bridge rare {int(bool(_INF_NAME.fullmatch(B_)))}, sentences A+B {ab_} B+C {bc_}, fact records A-B {fab_} B-C {fbc_}, bridge partners {fan_}, pair tried {int((min(r_[4],r_[6]),max(r_[4],r_[6])) in S_.get('pairs',set()) or (min(A_,C_),max(A_,C_)) in S_.get('pairs',set()))}")
        if dd_: print(f"  [inference DISCOVERY DETAIL | {tag}] "+" || ".join(dd_),flush=True)
        hp_={c_['names'][0]:bool(c_.get('hot')) for c_ in (_INF.get('probes') or [])}
        if any(hp_.values()):                                                   # FREQUENT ENTITIES: two-hop on chains whose names are frequent vs ordinary
            hb_={True:[0,0,0],False:[0,0,0]}
            for r_ in cr_:
                h0_=hp_.get(r_[4])
                if h0_ is None: continue
                hb_[h0_][0]+=1; hb_[h0_][1]+=int(r_[0]); hb_[h0_][2]+=int(frozenset((r_[4],r_[6])) in cm_)
            print(f"  [inference FREQUENT ENTITIES | {tag}] two-hop (n, right, chain found): frequent names {hb_[True][0]}: {hb_[True][1]/max(1,hb_[True][0]):.2f}, {hb_[True][2]/max(1,hb_[True][0]):.2f} | ordinary names {hb_[False][0]}: {hb_[False][1]/max(1,hb_[False][0]):.2f}, {hb_[False][2]/max(1,hb_[False][0]):.2f}",flush=True)
        print(f"  [inference TWO-HOP GAP | {tag}] "+' | '.join(f"{k_} {v_} (e.g. {exs_[k_]})" for k_,v_ in sorted(cat_.items())),flush=True)
    ex_=[];ver_=0;nc_=0
    for r_ in cr_:
        c_=cm_.get(frozenset((r_[4],r_[5])))
        if c_ is None: continue
        nc_+=1; s1_=S_['it'][c_['src'][0]]['s'].decode('latin-1','replace').strip(); s2_=S_['it'][c_['src'][1]]['s'].decode('latin-1','replace').strip()
        good_=(r_[5]==r_[6]) and (r_[7] in s1_+' '+s2_) and (r_[4] in s1_+' '+s2_) and (r_[6] in s1_+' '+s2_); ver_+=int(good_)
        if len(ex_)<2: ex_.append(f"{r_[4]} -> {r_[5]} because \"{s1_[:80]}\" + \"{s2_[:80]}\" ({'checks out' if good_ else 'WRONG'})")
    print(f"  [inference show-sources | {tag}] {nc_} answers backed by a stored conclusion; cited sources contain the true chain (A, bridge, C) and the answer is right in {ver_}/{nc_}"+(" | e.g. "+" || ".join(ex_) if ex_ else ''),flush=True)

@torch.no_grad()
def _inf_fade_sent(sn_,d_,S_slot,Lr_,filler,feats,dev_,beta_,it_n=10):
    """fade the store's entries for one sentence (writes towards the model's own prediction)"""
    it_=(sn_+' ').encode('latin-1','replace')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(d_)[:Lr_]); w_+=b' '*(Lr_-len(w_)); w_=w_[:Lr_]
    X_=torch.tensor([list(bytes(w_))],dtype=torch.long,device=dev_); lg_,z_=feats(X_,int(S_slot(d_))); K_,id_=_infm_keys(z_,X_[:,:-1],lg_)
    Kf_=K_.reshape(-1,K_.shape[-1]); idf_=id_.reshape(-1); lgf_=lg_.reshape(-1,256); p0_=F.softmax(lgf_,-1); mk_=torch.zeros(Lr_-1,device=dev_); mk_[:len(it_)-1]=1.0
    for _ in range(it_n): q_=F.softmax(lgf_+_infm_corr(Kf_,idf_),-1); _infm_write(Kf_,idf_,(p0_-q_)*mk_[:,None],beta_)
def _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_):
    for sn_ in [f"So {c_['A']} is connected to {c_['C']}.",f"So {c_['C']} is connected to {c_['A']}."]+([c_['s']] if c_['typed'] else []): _inf_fade_sent(sn_,c_['d'],S_slot,Lr_,filler,feats,dev_,beta_)
@torch.no_grad()
def _inf_audit(model,cfg,S_,allc_,S_slot,Lr_,filler,feats,rs_,dev_,beta_):
    """GROUNDING AUDIT (consolidation phase): every (subject, relation) with competing objects is re-grounded on its evidence record.
    Support of an object = sum over its verified / real sentences of exp(-(visits since stored)/tau). For relations LEARNED to be single-valued
    (>= inf_audit_func of their subjects have one object, >= 20 subjects): top >= k x second -> CONFIRMED (the others superseded, their
    conclusions RETRACTED, their store entries faded); otherwise UNCERTAIN (dependent conclusions demoted to the hypothesis tier).
    Then STORE DRIFT: sampled subjects whose store answer is backed by no belief and no fact have that association erased."""
    if not int(getattr(cfg,'inf_audit',0)) or _INFM['M'] is None or _INFM.get('mu') is None: return ''
    R_=S_.get('rel',{}); V_=int(model.__dict__.get('_visit',0)); tau_=float(getattr(cfg,'inf_audit_tau',5.0)); k_=float(getattr(cfg,'inf_audit_k',2.0))
    subj_={}
    for (A,r),ob_ in R_.items(): subj_.setdefault(r,[]).append(sum(1 for o_,ids_ in ob_.items() if any(not S_['it'][i_].get('sup') for i_ in ids_)))
    func_={r:(np.mean([n_==1 for n_ in v_])>=float(getattr(cfg,'inf_audit_func',0.6)) and len(v_)>=int(getattr(cfg,'inf_audit_minsubj',20))) for r,v_ in subj_.items()}
    S_['_func']=func_                                                           # cg26g: REM holds back facts of single-valued relations with a competing object
    w_=lambda i_: math.exp(-(V_-S_['it'][i_].get('v',V_))/tau_)
    sup_new_=set(); unc_=set(); conf_=0
    for (A,r),ob_ in R_.items():
        if not func_.get(r) or len(ob_)<2: continue
        sc_={o_:sum(w_(i_) for i_ in ids_ if not S_['it'][i_].get('sup')) for o_,ids_ in ob_.items()}; rk_=sorted(sc_,key=lambda o_: -sc_[o_])
        if sc_[rk_[1]]<=0: continue
        if str(getattr(cfg,'inf_audit_rule','ratio'))=='recent':          # RECENCY RULE: decayed support keeps its RATIO forever (both decay alike), so a
            cn_={o_:sum(1 for i_ in ids_ if not S_['it'][i_].get('sup')) for o_,ids_ in ob_.items()}   #   near-tie never resolves. Instead: the object stated
            lv_={o_:max([S_['it'][i_].get('v',0) for i_ in ids_ if not S_['it'][i_].get('sup')] or [-1]) for o_,ids_ in ob_.items()}   # most recently wins,
            mc_=max(float(getattr(cfg,'inf_audit_minrep',3)),float(getattr(cfg,'inf_audit_minfrac',0.25))*max(cn_.values()))   # minrep: 3 locally; 2 at GPU exposure (facts seen 1-2 times)   #   if it is REPEATED (>= minfrac of the best count: one noisy line never is)
            el_=[o_ for o_ in rk_ if cn_[o_]>=mc_ and sc_[o_]>0]
            if el_:
                top_=max(el_,key=lambda o_: lv_[o_])
                if all(lv_[top_]>lv_[o_] for o_ in rk_ if o_!=top_ and sc_[o_]>0): rk_=[top_]+[o_ for o_ in rk_ if o_!=top_]; sc_[top_]=float('inf')
        if sc_[rk_[0]]>=k_*sc_[rk_[1]]:
            conf_+=1
            for o_ in rk_[1:]:
                for i_ in ob_[o_]:
                    if not S_['it'][i_].get('sup'): S_['it'][i_]['sup']=True; sup_new_.add(i_)
            for i_ in ob_[rk_[0]]: S_['it'][i_].pop('unc',None)
        else:
            for o_ in rk_:
                for i_ in ob_[o_]: S_['it'][i_]['unc']=True; unc_.add(i_)
    nret_=0; ndem_=0; nrep_=0; t2_,t3_=_inf_truth(cfg,current=False); upd_={(c_['names'][0],c_['names'][2]):c_['upd']['type'] for c_ in inf_probes(cfg) if c_.get('upd')}
    good_=0; falser_=0
    for c_ in allc_:
        sr_=c_.get('src',())
        if any(i_ in sup_new_ for i_ in sr_) and not c_.get('retracted'):
            c_['retracted']=True; c_['freed']=True; nret_+=1; _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_)
            ty_=upd_.get((c_['A'],c_['C'])) or upd_.get((c_['C'],c_['A']))
            if ty_=='true': good_+=1
            elif (c_['A'],c_['C']) in t2_|t3_: falser_+=1
        elif any(S_['it'][i_].get('unc') for i_ in sr_ if i_<len(S_['it'])) and not c_.get('retracted'):
            if not c_.get('uncertain'): c_['uncertain']=True; c_['freed']=True; ndem_+=1; _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_)
        elif c_.get('uncertain') and not any(S_['it'][i_].get('unc') for i_ in sr_ if i_<len(S_['it'])):
            c_['uncertain']=False; c_['freed']=False; nrep_+=1                  # re-confirmed: back into the store (rewritten by the normal writes)
    nff_=0                                                                      # REPLAYED FACT RECORDS follow the audit: a fact whose every sentence is superseded
    for f_ in S_.get('facts',[]):                                               #   is freed (else NREM replay keeps writing the OLD fact back into the store)
        if f_.get('freed'): continue
        ids_=R_.get((f_['A'].encode('latin-1'),f_['r'].encode('latin-1')),{}).get(f_['B'].encode('latin-1'))
        if ids_ and all(S_['it'][i_].get('sup') for i_ in ids_):
            f_['freed']=True; nff_+=1; fp0_=S_.get('fpair',{})
            for k0_ in ((f_['A'],f_['B']),(f_['B'],f_['A'])):
                if fp0_.get(k0_) is f_: fp0_.pop(k0_,None)
            _inf_fade_sent(f_['s'],f_['d'],S_slot,Lr_,filler,feats,dev_,beta_)
    ncas_=0; ncas_t_=0
    if int(getattr(cfg,'inf_audit_cascade',1)):                                 # CASCADING RETRACTION: a retracted conclusion's re-entered sentence is superseded too,
        ik_={}                                                                  #   so every conclusion built ON it (three-hop, REM) is retracted in turn, to a fixed point
        for i_,it_ in enumerate(S_['it']):
            if it_.get('k')=='inf':
                m_=_re_inf.match(rb'So (\S+) is connected to (\S+)\.',it_['s'])
                if m_: ik_.setdefault(frozenset((m_.group(1).decode('latin-1'),m_.group(2).decode('latin-1'))),[]).append(i_)
        fr_=[c_ for c_ in allc_ if c_.get('retracted') and not c_.get('casc_done')]
        while fr_:
            dead_=set()
            for c_ in fr_:
                c_['casc_done']=True
                for i_ in ik_.get(frozenset((c_['A'],c_['C'])),[]):
                    if not S_['it'][i_].get('sup'): S_['it'][i_]['sup']=True; dead_.add(i_)
            fr_=[]
            for c_ in allc_:
                if not c_.get('retracted') and any(i_ in dead_ for i_ in c_.get('src',())):
                    c_['retracted']=True; c_['freed']=True; ncas_+=1; ncas_t_+=int((c_['A'],c_['C']) in t2_|t3_); fr_.append(c_); _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_)
    dg_=''
    if int(getattr(cfg,'inf_audit_debug',0)):                                   # BELIEF-REVISION DIAGNOSTIC: the evidence for each true-update probe subject
        ps_=[]
        for c0_ in inf_probes(cfg):
            u0_=c0_.get('upd')
            if not u0_ or u0_['type']!='true': continue
            ob_=R_.get((c0_['names'][0].encode(),c0_['rels'][0].encode()),{})
            sc0_={o_.decode('latin-1'):(len(ids_),round(sum(w_(i_) for i_ in ids_ if not S_['it'][i_].get('sup')),2),sum(1 for i_ in ids_ if S_['it'][i_].get('sup'))) for o_,ids_ in ob_.items()}
            ps_.append(f"{c0_['names'][0]} {c0_['rels'][0]}: old {c0_['names'][1]} {sc0_.get(c0_['names'][1],'-')} new {u0_['B2']} {sc0_.get(u0_['B2'],'-')} (func {func_.get(c0_['rels'][0].encode())})")
        dg_=" | REVISION EVIDENCE (items, support, superseded): "+"; ".join(ps_[:8])
    for i_ in sup_new_: _inf_fade_sent(S_['it'][i_]['s'].decode('latin-1','replace').strip(),S_['it'][i_]['d'],S_slot,Lr_,filler,feats,dev_,beta_)   # the superseded one-hop fact
    flips_=sum(1 for c_ in inf_probes(cfg) if c_.get('upd',{}).get('type')=='noise' and any(S_['it'][i_].get('sup') for i_ in R_.get((c_['names'][0].encode(),c_['rels'][0].encode()),{}).get(c_['names'][1].encode(),[])))
    nd_=int(getattr(cfg,'inf_audit_drift',20)); er_=0; ck_=0
    live_=[c_ for c_ in allc_ if not c_.get('freed')]; nm_=[n_.decode('latin-1') for n_ in _INFM['names']]
    if live_ and len(nm_)>=25 and nd_>0:                                        # STORE DRIFT: does the store push associations nothing grounds?
        bel_={}
        for c_ in live_: bel_.setdefault(c_['A'],set()).add(c_['C']); bel_.setdefault(c_['C'],set()).add(c_['A'])
        for X in list(dict.fromkeys(c_['A'] for c_ in [live_[i_] for i_ in rs_.choice(len(live_),min(len(live_),3*nd_),replace=False)]))[:nd_]:
            cs_=list(bel_.get(X,set()))+[n_ for n_ in rs_.choice(nm_,24,replace=False) if n_!=X]; sl_=int(S_slot(live_[0]['d']))
            l1_=_inf_lp(model,cfg,f"So {X} is connected to ",cs_,sl_,True); l0_=_inf_lp(model,cfg,f"So {X} is connected to ",cs_,sl_,False); ck_+=1
            Y=cs_[int(np.argmax(np.array(l1_)-np.array(l0_)))]
            gr_=Y in bel_.get(X,set()) or any(Y.encode() in S_['it'][i_]['s'] for i_ in S_['idx'].get(X.encode(),[]) if not S_['it'][i_].get('sup'))
            if not gr_ and max(np.array(l1_)-np.array(l0_))>1.0:
                er_+=1; _inf_fade_sent(f"So {X} is connected to {Y}.",live_[0]['d'],S_slot,Lr_,filler,feats,dev_,beta_)
    nf_=sum(func_.values())
    return (f" | GROUNDING AUDIT: {nf_} relations learned single-valued; {conf_} (subject, relation) conflicts resolved, {len(sup_new_)} sentences superseded, "
            f"{nret_} conclusions RETRACTED ({good_} stale probe beliefs, {falser_} true probe chains wrongly), {ndem_} demoted as uncertain, {nrep_} re-confirmed; "
            f"noise contradictions that flipped a belief {flips_}; CASCADE: {ncas_} dependent conclusions retracted ({ncas_t_} of them true probe chains), {nff_} replayed fact records freed; store drift: {er_} ungrounded associations erased of {ck_} subjects checked"+dg_)
