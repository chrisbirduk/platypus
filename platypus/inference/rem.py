"""Greedy generation, self-inference, episodic fact records, the rule layer and REM recombination."""
import math
import numpy as np
import torch
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
    _INFM,
)
from platypus.inference.units import (
    _INF_NAME, _INF_REL, _INF_TRIP, _ent_ok,
)
from platypus.inference.probes import (
    _inf_phrase, inf_probes,
)
from platypus.inference.infer import (
    _inf_truth, _kgx_ho,
)
from platypus.inference.selftaught import (
    _inf_lp, _inf_words,
)

@torch.no_grad()
def _inf_gen(model,cfg,ctx_,slot_,store_,n_=14):
    """greedy continuation of ctx_ up to '.' (re-reads the whole prefix each step; no carried context)"""
    out_=b''; dev_=model.embed.weight.device; b_=ctx_.encode('latin-1')
    sv4_=_carry_save(); sc_=_XDG['cur']; _XDG['cur']=slot_; _INFM['force']=bool(store_); was_=model.training; model.eval()
    try:
        for _ in range(n_):
            x_=(b_+out_)[-int(cfg.seq_len):]; _CARRY.update(on=False,st={},new={})
            lg_=model(torch.tensor([list(x_)],dtype=torch.long,device=dev_),slot_)[0,-1]; c_=int(lg_.argmax())
            if c_ in (46,10,0): break
            out_+=bytes([c_])
    finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM.pop('force',None); model.train(was_)
    return out_.decode('latin-1').strip()
def inf_self(model,cfg,S_,new_,S_slot,t2_):
    """per sleep, for up to inf_self_n engine conclusions A ~ C (via bridge B, sources sa / sc):
      S2 proposal: the model completes 'sa sc So A is connected to ' (greedy text; and its pick among the context words, no name rule)
      S3 evidence check: gap(w) = lp(w | sa sc Q) - lp(w | sa[B->decoy] sc Q); AUC of gap(C) vs gap(other context words)
      S4 handover (inf_self='hand', from sleep inf_self_hand on): the engine only pairs the sentences; the conclusion is the MODEL'S pick,
         kept only if its gap > inf_self_m nats (else abstain)"""
    mode_=str(cfg.inf_self); vis_=int(model.__dict__.get('_visit',0))+1; hand_=(mode_=='hand' and vis_>=int(getattr(cfg,'inf_self_hand',3)))
    rs_=np.random.RandomState(int(cfg.seed)+31*vis_); m_=float(getattr(cfg,'inf_self_m',0.5)); st_={'n':0,'gen':0,'pick':0,'kept':0,'abst':0,'ktrue':0,'kwrong':0}
    nm_=list(S_['df'].keys()); aucs_=[]; gt_=[];go_=[]; out_=[]
    idx_=list(range(len(new_))); rs_.shuffle(idx_); chk_=set(idx_[:int(getattr(cfg,'inf_self_n',48))])
    for i_,c_ in enumerate(new_):
        if i_ not in chk_:
            if not hand_: out_.append(c_)
            continue
        A,B,C=c_['A'],c_['B'],c_['C']; sa_=S_['it'][c_['src'][0]]['s'].decode('latin-1').strip(); sc_=S_['it'][c_['src'][1]]['s'].decode('latin-1').strip()
        if A not in sa_: sa_,sc_=sc_,sa_
        ctx_=f"{sa_} {sc_} So {A} is connected to "; sl_=int(S_slot(c_['d']))
        dec_=nm_[rs_.randint(len(nm_))].decode('latin-1') if nm_ else 'Someone'; ctxd_=f"{sa_.replace(B,dec_)} {sc_} So {A} is connected to "
        cw_=[w_ for w_ in _inf_words(sa_+' '+sc_) if w_!=A]
        if C not in cw_ or len(cw_)<2:
            if not hand_: out_.append(c_)
            continue
        st_['n']+=1; g_=_inf_gen(model,cfg,ctx_,sl_,True); st_['gen']+=int(g_.startswith(C))
        l1_=_inf_lp(model,cfg,ctx_,cw_,sl_,True); l0_=_inf_lp(model,cfg,ctxd_,cw_,sl_,True); gap_=[a_-b_ for a_,b_ in zip(l1_,l0_)]
        pk_=cw_[int(np.argmax(l1_))]; st_['pick']+=int(pk_==C); ic_=cw_.index(C)
        oth_=[gap_[j_] for j_ in range(len(cw_)) if j_!=ic_]; gt_.append(gap_[ic_]); go_+=oth_
        aucs_.append(float(np.mean([gap_[ic_]>o_ for o_ in oth_]+[0.5*(gap_[ic_]==o_) for o_ in oth_]) if oth_ else 0.5))
        if not hand_: out_.append(c_); continue
        gp_=gap_[cw_.index(pk_)]
        if gp_<=m_ or pk_==A: st_['abst']+=1; continue
        c2_=dict(c_); c2_['C']=pk_; c2_['typed']=False; c2_['s']=f"So {A} is connected to {pk_}."; c2_['self']=True; out_.append(c2_); st_['kept']+=1
        tr_=(A,pk_) in t2_; st_['ktrue']+=int(tr_); st_['kwrong']+=int((not tr_) and ((A,C) in t2_))
    if hand_:                                                                   # the conclusions list in the store follows the model's choices
        S_['conc']=[c_ for c_ in S_['conc'] if c_ not in new_]+out_
    n0_=max(1,st_['n'])
    print(f"  [self-taught inference | {'HANDOVER: model completes + checks, engine only pairs' if hand_ else 'shadow'}] {st_['n']} conclusions tried | S2 proposal: greedy text = C {st_['gen']/n0_:.2f}, "
          f"pick among {np.mean([0]) if False else ''}context words = C {st_['pick']/n0_:.2f} | S3 evidence gap (real vs decoy bridge): true C {np.mean(gt_) if gt_ else float('nan'):+.2f} nats, other words "
          f"{np.mean(go_) if go_ else float('nan'):+.2f}, AUC {np.mean(aucs_) if aucs_ else float('nan'):.2f}"+(f" | S4: kept {st_['kept']} (probe-true {st_['ktrue']}, probe-wrong {st_['kwrong']}), abstained {st_['abst']} (margin {m_} nats)" if hand_ else ''),flush=True)
    return out_

_INF_FUNC={w_.encode() for w_ in 'and or of the a an von van der den de du des del della di da la le les el los las und y e et en in on at to for by from with as is was'.split()}
def _inf_facts(cfg,S_,nid_,vis0_):
    """EPISODIC FACT RECORDS (sleep-order test): this visit's salient one-hop facts 'A r B' (seen >= inf_rem_minrep times, rare names)"""
    F_=S_.setdefault('facts',[]); K_=S_.setdefault('fkey',{}); cnt_={}; first_={}; df_=S_['df']; mx_=int(cfg.inf_df); lw_=S_.get('lw') if S_.get('strict') else None
    ok_=lambda n_: _ent_ok(n_,df_,mx_) and (lw_ is None or n_.lower() not in lw_)
    for i_ in nid_:
        for m_ in _INF_TRIP.finditer(S_['it'][i_]['s']):
            a_,r_,b_=m_.group(1),m_.group(2),m_.group(3)
            if a_==b_ or not ok_(a_) or not ok_(b_): continue
            if int(getattr(cfg,'inf_rem_stop',0)) and set(r_.split())<=_INF_FUNC: continue   # 'Konrad von Wordship', 'Chaloner and Carley': no relation word
            k_=(a_.decode('latin-1'),r_.decode('latin-1'),b_.decode('latin-1')); cnt_[k_]=cnt_.get(k_,0)+1; first_.setdefault(k_,i_)
    mr_=int(getattr(cfg,'inf_rem_minrep',3)); nw_=0; cap_=int(getattr(cfg,'inf_rem_fact_cap',0))
    if int(getattr(cfg,'inf_rem_cumul',0)):                                     # CUMULATIVE: a fact stated once per visit is recorded when it has been read
        pd_=S_.setdefault('fpend',{})                                           #   inf_rem_minrep times in all (sparse exposure: facts seen 1-2 times a visit)
        for k_,c_ in list(cnt_.items()):
            if k_ in K_: continue
            c0_=pd_.get(k_,0)+c_
            if c0_>=mr_: cnt_[k_]=c0_; pd_.pop(k_,None)
            else: pd_[k_]=c0_
        if len(pd_)>400000:                                                     #   (bounded: the pending counts of the oldest half are forgotten)
            for k_ in list(pd_)[:len(pd_)//2]: pd_.pop(k_,None)
    ks_=[k_ for k_,c_ in cnt_.items() if c_>=mr_ and k_ not in K_]
    if cap_>0 and len(ks_)>cap_:                                                # PRIORITY = LENGTH / SPREAD: long, rarely shared names first
        ks_=sorted(ks_,key=lambda k_: -(len(k_[0])+len(k_[2]))/(df_.get(k_[0].encode('latin-1'),1)+df_.get(k_[2].encode('latin-1'),1)))[:cap_]
    ks_=set(ks_)
    for k_,c_ in cnt_.items():
        if c_<mr_: continue
        if k_ not in K_ and k_ not in ks_: continue
        if k_ in K_: K_[k_]['rep']+=c_; continue
        f_={'A':k_[0],'r':k_[1],'B':k_[2],'s':f"{k_[0]} {k_[1]} {k_[2]}.",'d':S_['it'][first_[k_]]['d'],'i':first_[k_],'v':vis0_,'rep':c_,'n':0,'fact':True}
        F_.append(f_); K_[k_]=f_; S_.setdefault('fpair',{}).setdefault((k_[0],k_[2]),f_); nw_+=1
    return nw_
def _inf_rules(cfg,S_,vis0_):
    """RULE LAYER (rules that pay for themselves): from the recorded fact records, propose 'phrase p => phrase q' for every pair of relation
    phrases sharing subjects; score each by the bits it saves on held-out evidence (leave-one-out two-part code); keep a fixed-size pool of
    the best; APPLY only confident rules that reach a fraction of the going rate (median gain of rules that save bits): derived facts
    'A q B' are added to the fact records, so NREM replay writes them into the fast store. Derived facts die with their source."""
    F_=S_.get('facts') or []; K_=S_.setdefault('fkey',{})
    for f_ in F_:                                                               # a derived fact whose source was freed (superseded) is freed too
        if f_.get('derived') and not f_.get('freed') and K_.get(f_['srcf'],{}).get('freed'): f_['freed']=True
    src_=[f_ for f_ in F_ if not f_.get('freed') and not f_.get('derived')]
    by_={}
    for f_ in src_: by_.setdefault(f_['r'],{})[f_['A']]=f_['B']
    objs_=set(f_['B'] for f_ in src_); N_=max(2,len(objs_)); R_=[r_ for r_ in by_ if len(by_[r_])>=2]; cst_=math.log2(max(2,len(R_)*(len(R_)-1)))
    mn_=int(getattr(cfg,'inf_rule_min',3)); cand_=[]
    for p_ in R_:
        for q_ in R_:
            if p_==q_: continue
            sh_=[a_ for a_ in by_[p_] if a_ in by_[q_]]
            if len(sh_)<mn_: continue
            b0_=b1_=0.0; h_=0
            for a_ in sh_:
                oth_=[x_ for x_ in sh_ if x_!=a_]; hh_=sum(by_[p_][x_]==by_[q_][x_] for x_ in oth_); pi_=(hh_+1)/(len(oth_)+2)
                hit_=by_[p_][a_]==by_[q_][a_]; h_+=hit_; b0_+=math.log2(N_); b1_+=-math.log2(pi_) if hit_ else (-math.log2(1-pi_)+math.log2(N_))
            cand_.append({'p':p_,'q':q_,'n':len(sh_),'h':h_,'gain':b0_-b1_-cst_,'conf':(h_+1)/(len(sh_)+2)})
    pos_=[c_['gain'] for c_ in cand_ if c_['gain']>0]; md_=float(np.median(pos_)) if pos_ else 0.0
    cap_=int(getattr(cfg,'inf_rules_cap',24)); pool_=sorted([c_ for c_ in cand_ if c_['gain']>0],key=lambda c_: -c_['gain'])[:cap_]
    use_=[c_ for c_ in pool_ if c_['conf']>=float(getattr(cfg,'inf_rule_conf',0.85)) and c_['gain']>=float(getattr(cfg,'inf_rule_rel',0.5))*md_]
    PP_={_inf_phrase(t_):_INF_PARA[_inf_phrase(t_)] for t_ in _INF_REL}; true_={(k_,v_) for k_,v_ in PP_.items()}|{(v_,k_) for k_,v_ in PP_.items()}
    nd_=0
    for c_ in use_:
        for a_,b_ in by_[c_['p']].items():
            if a_ in by_.get(c_['q'],{}): continue                              # the subject already has a q-fact: never overwrite evidence
            k_=(a_,c_['q'],b_)
            if k_ in K_: continue
            sf_=next((f_ for f_ in src_ if f_['A']==a_ and f_['r']==c_['p'] and f_['B']==b_),None)
            if sf_ is None: continue
            f_={'A':a_,'r':c_['q'],'B':b_,'s':f"{a_} {c_['q']} {b_}.",'d':sf_['d'],'i':sf_['i'],'v':vis0_,'rep':1,'n':0,'fact':True,'derived':True,'srcf':(a_,c_['p'],b_)}
            F_.append(f_); K_[k_]=f_; nd_+=1
    S_['rules']=use_
    print(f"  [inference RULE LAYER] {len(cand_)} candidate rules between {len(R_)} relation phrases; {len(pos_)} save bits (going rate {md_:.1f} bits); pool {len(pool_)}/{cap_}; "
          f"APPLIED {len(use_)} (true paraphrase pairs {sum(1 for c_ in use_ if (c_['p'],c_['q']) in true_)}, other {sum(1 for c_ in use_ if (c_['p'],c_['q']) not in true_)}); {nd_} derived facts added"
          +(" | e.g. "+"; ".join(f"'{c_['p']}' => '{c_['q']}' {c_['gain']:+.0f} bits conf {c_['conf']:.2f}" for c_ in use_[:4]) if use_ else ''),flush=True)
def _inf_nm1(g_):
    w_=(g_.split() or [''])[0].strip('.,;:!?')
    return w_ if _INF_NAME.fullmatch(w_.encode('latin-1','ignore')) else ''
@torch.no_grad()
def _rem_conflict(S_,G_):
    """cg26g: True if any grounding fact is superseded, or its (subject, relation) has a COMPETING object with live evidence - the belief is
    not yet resolved (cg26e GPU pass 2: 30 pct of planted chains change, REM grounded on the old fact before the audit had the repeats to
    supersede it: precision 5/6, 5/8, 3/5, 1/7)"""
    R_=S_.get('rel',{}); it_=S_['it']
    for g_ in G_:
        if g_.get('freed') or g_.get('sup'): return True
        i_=g_.get('i')
        if isinstance(i_,int) and i_<len(it_) and it_[i_].get('sup'): return True
        eb_=lambda x_: x_.encode('latin-1') if isinstance(x_,str) else x_
        ob_=R_.get((g_['A'],g_['r'])) or R_.get((eb_(g_['A']),eb_(g_['r'])))      # (the evidence record is keyed by bytes)
        fn_=S_.get('_func',{}); sv_=fn_.get(g_['r']) or fn_.get(eb_(g_['r']))   # only relations LEARNED to be single-valued (works-with may have many objects)
        if sv_ and ob_ and sum(1 for o_,ids_ in ob_.items() if any(not it_[j_].get('sup') for j_ in ids_ if isinstance(j_,int) and j_<len(it_)))>=2: return True
    return False
def _inf_rem(model,cfg,S_,S_slot,n_,tag_,shadow_=False):
    """REM-LIKE RECOMBINATION (sleep-order test): links proposed from MEMORY (weights + fast store), not from co-occurrence.
      subject: a recorded fact 'A r0 B' (half from this visit, half older); lookup 1: complete 'A r0 ' -> bridge B^;
      lookup 2: for each common relation r1, complete 'B^ r1 ' -> C^; APPRAISAL (knowing it knows): log p(C^ | 'B^ r1 ') - log p(C^ | 'Someone r1 ')
      must exceed inf_rem_m nats (the completion belongs to the bridge, not just to the relation); GROUNDING: kept as a conclusion only if both links are recorded facts, else a hypothesis (not written)"""
    F_=S_.get('facts') or []
    if not F_ or n_<=0: return []
    t2_,_=_inf_truth(cfg); pn_=set(x_ for q_ in inf_probes(cfg) for x_ in q_['names']); vis0_=int(model.__dict__.get('_visit',0))
    rs_=np.random.RandomState(int(cfg.seed)+97*vis0_+len(S_.setdefault('rem_log',[]))); m_=float(getattr(cfg,'inf_rem_m',2.0))
    rc_={}
    for f_ in F_: rc_[f_['r']]=rc_.get(f_['r'],0)+f_['rep']
    rv_=[r_ for r_,_ in sorted(rc_.items(),key=lambda x_:-x_[1])[:int(getattr(cfg,'inf_rem_rels',8))]]
    pool_=sorted(set(x_ for f_ in F_ for x_ in (f_['A'],f_['B'])))
    rec_=[f_ for f_ in F_ if f_['v']==vis0_]; old_=[f_ for f_ in F_ if f_['v']!=vis0_]
    pf_=float(getattr(cfg,'inf_rem_probe_frac',0.0)); sel_=[]
    if pf_>0:                                                                   # MEASUREMENT SAMPLE: a share of subjects from probe facts (scorable), labelled in the report
        pr_=[f_ for f_ in F_ if f_['A'] in pn_]; sel_=[pr_[i_] for i_ in rs_.permutation(len(pr_))[:int(round(n_*pf_))]]
    m0_=n_-len(sel_); h_=m0_//2 if old_ else m0_; sel_+=[rec_[i_] for i_ in rs_.permutation(len(rec_))[:h_] if rec_[i_] not in sel_]; sel_+=[old_[i_] for i_ in rs_.permutation(len(old_))[:n_-len(sel_)] if old_[i_] not in sel_]
    st_={k_:0 for k_ in ('n','bok','nob','prop','abst','known','acc','at','aw','hyp','ht','hw')}; out_=[]
    for f_ in sel_:
        A_,r0_=f_['A'],f_['r']; d_=int(S_slot(f_['d'])); st_['n']+=1
        Bh_=_inf_nm1(_inf_gen(model,cfg,f"{A_} {r0_} ",d_,True)); st_['bok']+=int(Bh_==f_['B'])
        if not Bh_ or Bh_==A_: st_['nob']+=1; continue
        best_=None; ex_=[]; dB_=sorted(set(int(S_slot(g_['d'])) for g_ in F_ if g_['A']==Bh_)) or [d_]   # lookup 2 in the domain(s) where B^ was met (fix: was A's domain)
        for r1_,dd_ in [(r_,x_) for r_ in rv_ for x_ in dB_]:
            Ch_=_inf_nm1(_inf_gen(model,cfg,f"{Bh_} {r1_} ",dd_,True)); ex_.append(f"{r1_}->{Ch_ or '-'}")
            if not Ch_ or Ch_ in (A_,Bh_): continue
            mg_=_inf_lp(model,cfg,f"{Bh_} {r1_} ",[Ch_],dd_,True)[0]-_inf_lp(model,cfg,f"Someone {r1_} ",[Ch_],dd_,True)[0]   # specific to the bridge: not just the relation's usual object
            if int(getattr(cfg,'inf_rem_all',0)):                              # ALL CANDIDATES: the best GROUNDED proposal above the margin wins
                fp0_=S_.get('fpair',{}); gr_=(fp0_.get((Bh_,Ch_)) or fp0_.get((Ch_,Bh_))) is not None and (fp0_.get((A_,Bh_)) or fp0_.get((Bh_,A_))) is not None
                if gr_ and mg_>m_ and (best_ is None or not best_[3] or mg_>best_[0]): best_=(mg_,r1_,Ch_,True); continue
                if best_ is not None and best_[3]: continue
            if best_ is None or mg_>best_[0]: best_=(mg_,r1_,Ch_,False)
        if best_ is None: continue
        best_=best_[:3]
        st_['prop']+=1; mg_,r1_,C_=best_
        if len(st_.setdefault('eg',[]))<4: tc_=[q_['names'][2] for q_ in inf_probes(cfg) if q_['names'][0]==A_]; st_['eg'].append(f"{A_} {r0_} {Bh_} (true {f_['B']}) | {', '.join(ex_)} | chose {r1_}->{C_} margin {mg_:.1f} (true C {tc_[0] if tc_ else '?'})")
        if mg_<=m_: st_['abst']+=1; continue
        key_=(min(A_,C_),max(A_,C_)); tr_=(A_,C_) in t2_; pw_=(A_ in pn_ or C_ in pn_) and not tr_
        if key_ in S_['pairs'] or key_ in S_.setdefault('rem_seen',set()): st_['known']+=1; continue
        fp_=S_.get('fpair',{}); fa_=fp_.get((A_,Bh_)) or fp_.get((Bh_,A_)); fb_=fp_.get((Bh_,C_)) or fp_.get((C_,Bh_))   # grounded: recorded episodes link A-B^ and B^-C^
        if fa_ is not None and fb_ is not None and shadow_:                    # SHADOW: proposals are only counted (nothing written; discovery is unchanged)
            S_['rem_seen'].add(key_); st_['acc']+=1; st_['at']+=int(tr_); st_['aw']+=int(pw_)
        elif fa_ is not None and fb_ is not None and int(getattr(cfg,'inf_rem_conflict',0)) and _rem_conflict(S_,(fa_,fb_)):
            st_['conf']=st_.get('conf',0)+1; st_['cw']=st_.get('cw',0)+int(pw_); st_['hyp']+=1; st_['ht']+=int(tr_); st_['hw']+=int(pw_)   # cg26g: held back as a hypothesis
        elif fa_ is not None and fb_ is not None:
            S_['pairs'].add(key_); r1_=fb_['r'] if fb_['A']==Bh_ else r1_; c_={'A':A_,'B':Bh_,'C':C_,'r':(r0_,r1_),'typed':True,'s':f"{A_} {r0_} someone who {r1_} {C_}.",'src':(fa_['i'],fb_['i']),
                                    'd':f_['d'],'dep':1,'v':vis0_,'rem':True}
            if not _kgx_ho(cfg,c_): S_['conc'].append(c_); out_.append(c_); st_['acc']+=1; st_['at']+=int(tr_); st_['aw']+=int(pw_)
        else: st_['hyp']+=1; st_['ht']+=int(tr_); st_['hw']+=int(pw_)
    eg_=st_.pop('eg',[]); S_['rem_log'].append(st_); n0_=max(1,st_['n'])
    for e_ in eg_[:3]: print(f"    [inference REM e.g.] {e_}",flush=True)
    f2_=len(set(frozenset((c_['A'],c_['C'])) for c_ in S_['conc'] if (c_['A'],c_['C']) in t2_))
    print(f"  [inference REM recombination | {tag_}] {st_['n']} subjects{f' ({pf_:.0%} sampled from probe facts for scoring)' if pf_>0 else ''} ({len(rec_)} facts from this visit, {len(old_)} older; {len(rv_)} relations) | bridge recalled {st_['bok']/n0_:.2f} (no name {st_['nob']}) | "
          f"{st_['prop']} proposals, {st_['abst']} abstained (margin {m_} nats), {st_['known']} already known"+(f", {st_.get('conf',0)} held back (belief not yet resolved; probe-wrong among them {st_.get('cw',0)})" if int(getattr(cfg,'inf_rem_conflict',0)) else "")+f" | ACCEPTED grounded {st_['acc']} (probe-true {st_['at']}, probe-wrong {st_['aw']}) | "
          f"hypotheses (ungrounded, not written) {st_['hyp']} (probe-true {st_['ht']}, probe-wrong {st_['hw']}) | facts recorded {len(F_)} | probe chains found so far: two-hop {f2_}/{len(t2_)//2}",flush=True)
    return out_

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_INF_PARA': 'platypus.inference.qtime',
}
