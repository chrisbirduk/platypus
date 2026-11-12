"""Offline inference: bridges, conclusions, staging and salience."""
import math
import numpy as np
import re as _re_inf
import torch
import torch.nn.functional as F
from platypus.inference.store import (
    _INFM,
)
from platypus.inference.units import (
    _INF_NAME, _INF_TRIP, _RDR, _UNITS, _ent_ok,
)
from platypus.inference.probes import (
    _INF, inf_probes, inf_rule_world,
)
from platypus.inference.sentences import (
    _inf_rel, _inf_store,
)
from platypus.inference.links import (
    _LM, _cue_fix, _lev, _lm_init, _lm_pair, _lm_partners, _lm_sim, _lm_write,
)

def inf_infer(model,cfg):
    """bridge-centred chaining over the store; returns this sleep's NEW conclusions"""
    S_=_inf_store(model); df_=S_['df']; mx_=int(cfg.inf_df); new_=[]; st_={'bridges':0,'hub':0,'prop':0}
    rf_=int(getattr(cfg,'inf_conc_refresh',0)) and _UNITS.get('all'); nrt_=0
    gr_=int(getattr(cfg,'inf_graph',0)) if _UNITS.get('all') else 0; G_=S_.setdefault('graph',{}) if gr_==1 else None; vis_=int(model.__dict__.get('_visit',0)); ngn_=0; ngu_=0
    cmap_={(min(c_['A'],c_['C']).encode('latin-1','replace'),max(c_['A'],c_['C']).encode('latin-1','replace')):c_ for c_ in S_['conc'] if not c_.get('freed') and isinstance(c_.get('A'),str) and isinstance(c_.get('C'),str)} if rf_ else {}
    lw_=S_.get('lw') if S_.get('strict') else None; ok_=lambda n_: _ent_ok(n_,df_,mx_) and (lw_ is None or n_.lower() not in lw_)
    br_=set(n_ for i_ in S_['new'] for n_ in _INF_NAME.findall(S_['it'][i_]['s']) if ok_(n_))
    if str(getattr(cfg,'inf_bridge_order','any'))=='rare': br_=sorted(br_,key=lambda n_: (df_.get(n_,0),n_))
    if int(getattr(cfg,'inf_bridge_facts',0)) and _UNITS.get('all'):           # BRIDGES WITH RECORDED FACTS FIRST: bridges that are arguments of recorded (recurring)
        fa_={}                                                                  #   facts are linked first, most facts first; one-off rare words in ordinary text
        for f_ in S_.get('facts',[]):                                           #   no longer use up the linking budget before them
            if not f_.get('freed'):
                for n_ in (f_['A'],f_['B']): fa_[n_.encode('latin-1')]=fa_.get(n_.encode('latin-1'),0)+1
        br_=sorted(br_,key=lambda n_: (-fa_.get(n_,0),df_.get(n_,0),n_))
    if int(getattr(cfg,'inf_lm_side',0)) and gr_ not in (2,4):                  # DUAL PATHWAY: the overlapping link memory learns beside the exact store
        _lm_init(cfg); _LM['M']*=float(getattr(cfg,'inf_lm_decay',0.97)); nsw_=0
        for i_ in S_.get('lmw',[]):
            it_=S_['it'][i_]
            if it_.get('sup') and not it_.get('lmu'):
                for a_,c_ in it_.get('lml',()): _lm_write(a_,c_,-1.0)
                it_['lmu']=True
        for i_ in S_['new']:
            it_=S_['it'][i_]
            if it_.get('k')=='inf' or it_.get('sup'): continue
            for m_ in _INF_TRIP.finditer(it_['s']):
                a_,c_=m_.group(1),m_.group(3)
                if a_!=c_ and ok_(a_) and ok_(c_):
                    _lm_write(a_,c_,1.0); nsw_+=1; it_.setdefault('lml',[]).append((a_,c_)); S_.setdefault('lmw',[]).append(i_) if len(it_['lml'])==1 else None
        for c_ in S_['conc']:                                                   #   conclusions enter it as strong links (and leave when freed)
            if not c_.get('freed') and not c_.get('lms') and isinstance(c_.get('A'),str): _lm_write(c_['A'].encode(),c_['C'].encode(),9.0); c_['lms']=True
            elif c_.get('freed') and c_.get('lms'): _lm_write(c_['A'].encode(),c_['C'].encode(),-9.0); c_['lms']=False
        S_['lmside']=f"{len(_LM['names'])} entities, {nsw_} links this sleep"
    if gr_==3:                                                                  # ONE-SCAN SALIENCE: links kept as a sparse weighted adjacency W (exact counts, decay,
        import scipy.sparse as _sp                                              #   retraction); every two-step path strength at once as S = W'W', with W' = f(W)
        W_=S_.setdefault('W1',{}); ix0_=S_.setdefault('W1ix',{}); osst_={'w':0,'un':0}; NW_={}   # NW_: links READ this sleep (staging v2: new support)   #   damped by each bridge's fan-out; the most salient links are the top
        dec_=float(getattr(cfg,'inf_lm_decay',0.97))                            #   entries of S, ranked by GAIN x NEED (Mattar & Daw), in one pass - no enumeration
        WV_=S_.setdefault('W1v',{}); vw_=int(model.__dict__.get('_visit',0)); gw_=int(getattr(cfg,'inf_graph_window',0)); ngw_=0
        if gw_>0:                                                                   # a LIVE conclusion keeps its strong link in the window (the window bounds read links;
            for c_ in S_.get('conc',[]):                                            #   a conclusion leaves the graph when it is freed or deleted, as before)
                if c_.get('w9') and not c_.get('freed') and isinstance(c_.get('A'),str): a9_,c9_=c_['A'].encode(),c_['C'].encode(); WV_[(min(a9_,c9_),max(a9_,c9_))]=vw_
        for k0_ in list(W_):
            W_[k0_]*=dec_
            if W_[k0_]<0.05: del W_[k0_]; WV_.pop(k0_,None); continue
            if gw_>0 and vw_-WV_.setdefault(k0_,vw_)>=gw_: del W_[k0_]; WV_.pop(k0_,None); ngw_+=1   # cg26h SLIDING WINDOW: links not read in the last gw_ visits leave the graph
        if gw_>0: osst_['gw']=ngw_
        AL_=S_.setdefault('alias',{}); cn_=lambda n_: AL_.get(n_,n_); er_=int(getattr(cfg,'inf_ent_resolve',0)); erst_={'m':0,'f':0,'mdl':0,'fan':0,'est':0,'ex':[]}
        if er_:
            vt0_={}
            for q_ in inf_probes(cfg):
                if q_.get('var'): vt0_[q_['var'].encode('latin-1')]=q_['names'][1].encode('latin-1')
            fW_={}
            for (a0_,c0_) in W_: fW_[a0_]=fW_.get(a0_,0)+1; fW_[c0_]=fW_.get(c0_,0)+1
            nn_=set(); PW_={}
            for (a0_,c0_) in W_: PW_.setdefault(a0_,set()).add(c0_); PW_.setdefault(c0_,set()).add(a0_)
            for i_ in S_['new']:
                it_=S_['it'][i_]
                if it_.get('k')=='inf' or it_.get('sup'): continue
                for m_ in _INF_TRIP.finditer(it_['s']):
                    a1_,c1_=m_.group(1),m_.group(3)
                    if ok_(a1_) and ok_(c1_) and a1_!=c1_: PW_.setdefault(a1_,set()).add(cn_(c1_)); PW_.setdefault(c1_,set()).add(cn_(a1_))
                    for n_ in (a1_,c1_):
                        if ok_(n_) and n_ not in AL_: nn_.add(n_)
            al0_=set(fW_)|nn_; DX_={}
            for n_ in al0_:
                if n_ in AL_: continue
                l_=n_.lower()
                for k_ in {l_}|{l_[:j_]+l_[j_+1:] for j_ in range(len(l_))}: DX_.setdefault(k_,set()).add(n_)
            nb_=math.log2(max(2,len(al0_))); mf0_=int(cfg.inf_maxfan); df0_=S_['df']
            for V_ in sorted(nn_):
                if V_ in AL_: continue
                l_=V_.lower(); cs_=set()
                for k_ in {l_}|{l_[:j_]+l_[j_+1:] for j_ in range(len(l_))}: cs_|=DX_.get(k_,set())
                cs_={B_ for B_ in cs_ if B_!=V_ and B_ not in AL_ and _lev(l_,B_.lower())<=1}
                if not cs_: continue
                B_=max(cs_,key=lambda b_: (df0_.get(b_,0),fW_.get(b_,0),b_))
                if (df0_.get(B_,0),fW_.get(B_,0),B_)<(df0_.get(V_,0),fW_.get(V_,0),V_): B_,V_=V_,B_   #   the more frequent spelling is canonical
                l_=V_.lower()
                if (min(B_,V_),max(B_,V_)) in S_.setdefault('ersep',set()): continue
                eb_=_lev(l_,B_.lower())*math.log2(27.0*max(len(B_),len(V_)))+(1.0 if B_!=V_ and B_.lower()==l_ else 0.0)
                sp9_=False
                if er_>=2:                                                      # v2: (a) TYPO - per occurrence, coding V as B plus an edit beats V's own
                    sh_=len((PW_.get(V_,set())|{cn_(x_) for x_ in PW_.get(V_,set())})&PW_.get(B_,set()))   #   code only if B is ~2^edit-bits times more frequent;
                    ty_=eb_<math.log2((df0_.get(B_,0)+1.0)/(df0_.get(V_,0)+1.0))                         #   (b) SAME ENTITY - the two share a partner and the
                    if not (ty_ or (sh_>=1 and eb_<nb_)): erst_['mdl']+=1; continue                       #   edit is cheaper than a new name
                    erst_['why']=erst_.get('why',{}); k9_='typo' if ty_ else 'shared partner'; erst_['why'][k9_]=erst_['why'].get(k9_,0)+1; sp9_=not ty_
                elif eb_>=nb_: erst_['mdl']+=1; continue                        #   MDL: an edit must be cheaper than a new name
                if fW_.get(V_,0)>=2 and fW_.get(B_,0)>=2 and not sp9_: erst_['est']+=1; S_['ersep'].add((min(B_,V_),max(B_,V_))); continue   #   both established: distinct
                if fW_.get(V_,0)+fW_.get(B_,0)>mf0_: erst_['fan']+=1; S_['ersep'].add((min(B_,V_),max(B_,V_))); continue      #   would make a hub
                AL_[V_]=cn_(B_); erst_['m']+=1
                for v2_,b2_ in list(AL_.items()):
                    if b2_==V_: AL_[v2_]=AL_[V_]                                   #   (keep aliases pointing at the canonical name)
                tr_=(vt0_.get(V_)==B_ or vt0_.get(B_)==V_); erst_['f']+=int(not tr_)
                if len(erst_['ex'])<4: erst_['ex'].append(f"{V_.decode('latin-1','replace')}->{B_.decode('latin-1','replace')}{'' if tr_ else ' (not planted)'}")
            if erst_['m']:                                                      #   re-key the link table to the canonical names
                for k0_ in [k_ for k_ in W_ if k_[0] in AL_ or k_[1] in AL_]:
                    w0_=W_.pop(k0_); a0_,c0_=cn_(k0_[0]),cn_(k0_[1])
                    if a0_!=c0_: k1_=(min(a0_,c0_),max(a0_,c0_)); W_[k1_]=W_.get(k1_,0.0)+w0_; WV_[k1_]=max(WV_.get(k1_,vw_),WV_.pop(k0_,vw_))
            S_['ergrp']={}
            for v_,b_ in AL_.items(): S_['ergrp'].setdefault(b_,[]).append(v_)
            ST_=S_.setdefault('erstat',[0,0]); ST_[0]+=erst_['m']; ST_[1]+=erst_['f']
            osst_['er']=(f" | ENTITY RESOLUTION: {erst_['m']} merged this sleep ({erst_['f']} not a planted variant; total {ST_[0]}, {ST_[1]} not planted), refused: MDL {erst_['mdl']}, "
                         f"both established {erst_['est']}, would make a hub {erst_['fan']}"+(f" | merged because: {erst_.get('why')}" if erst_.get('why') else "")+(f" | e.g. {', '.join(erst_['ex'])}" if erst_['ex'] else ""))
        for i_ in S_['new']:
            it_=S_['it'][i_]
            if it_.get('k')=='inf' or it_.get('sup'): continue
            for m_ in _INF_TRIP.finditer(it_['s']):
                a_,c_=cn_(m_.group(1)),cn_(m_.group(3))
                if a_!=c_ and ok_(a_) and ok_(c_):
                    k0_=(min(a_,c_),max(a_,c_))
                    rl0_=m_.group(2).lower(); S_.setdefault('relp',{}).setdefault(rl0_,set()).add(k0_); S_.setdefault('W1rel',{}).setdefault(k0_,set()).add(rl0_)   # (relation of each link)
                    W_[k0_]=W_.get(k0_,0.0)+1.0; WV_[k0_]=vw_; NW_[k0_]=NW_.get(k0_,0)+1; osst_['w']+=1; it_.setdefault('w1l',[]).append(k0_); S_.setdefault('w1w',[]).append(i_) if len(it_['w1l'])==1 else None
        dr0_=frozenset(_RDR.get('drop',()) or ()) if _RDR.get('on') else frozenset()
        if dr0_!=S_.get('w1drop',frozenset()):                                   # cg20x: the reader learned new no-fact contexts (negation, hedges):
            S_['w1drop']=dr0_; osst_['rd']=0                                    #   re-read every sentence that wrote links and take back the links
            for i_ in S_.get('w1w',[]):                                         #   it no longer states (W must not count 'A is not B' as an A-B link)
                it_=S_['it'][i_]
                if it_.get('w1u') or not it_.get('w1l'): continue
                now_=set((min(cn_(m_.group(1)),cn_(m_.group(3))),max(cn_(m_.group(1)),cn_(m_.group(3)))) for m_ in _INF_TRIP.finditer(it_['s']))
                kp_=[]
                for k0_ in it_['w1l']:
                    if k0_ in now_: kp_.append(k0_)
                    elif k0_ in W_: W_[k0_]=max(0.0,W_[k0_]-1.0); osst_['rd']+=1
                it_['w1l']=kp_
        for i_ in S_.get('w1w',[]):                                              #   retraction: superseded sentences leave W at once
            it_=S_['it'][i_]
            if it_.get('sup') and not it_.get('w1u'):
                for k0_ in it_.get('w1l',()):
                    if k0_ in W_: W_[k0_]=max(0.0,W_[k0_]-1.0); osst_['un']+=1
                it_['w1u']=True
        for c_ in S_.get('conc',[]):                                           # cg20x: a withdrawn or evicted conclusion gives back its strong link
            if c_.get('freed') and c_.get('w9'):
                a9_,c9_=c_['A'].encode(),c_['C'].encode(); k0_=(min(a9_,c9_),max(a9_,c9_))
                if k0_ in W_: W_[k0_]=max(0.0,W_[k0_]-9.0)
                c_['w9']=False; osst_['un']+=1
        rmp_=int(getattr(cfg,'inf_rel_minpairs',0)); Wx_=W_
        if rmp_>0:                                                              # RELATIONS MUST PAY THEIR WAY: a link counts only if one of its relation phrases
            RP_=S_.get('relp',{}); WR_=S_.get('W1rel',{})                       #   has been seen with at least rmp different argument pairs (a phrase used for one
            Wx_={k0_:w0_ for k0_,w0_ in W_.items() if any(len(RP_.get(r0_,()))>=rmp_ for r0_ in WR_.get(k0_,()))}   #   pair is a fragment, not a relation)
            osst_['relf']=(len(W_)-len(Wx_),len(W_),sum(1 for r0_,v0_ in RP_.items() if len(v0_)>=rmp_))
        sym_=float(getattr(cfg,'inf_sym',0.0))
        if sym_>0:                                                              # SYMMETRY: a link A-B counts only if it is a real share of BOTH words' links -
            dg9_={}                                                             #   min(P(B|A), P(A|B)) = w_AB / max(deg A, deg B) >= inf_sym ('all-confidence', null-invariant):
            for (a9_,c9_),w9_ in Wx_.items(): dg9_[a9_]=dg9_.get(a9_,0.0)+w9_; dg9_[c9_]=dg9_.get(c9_,0.0)+w9_   #   'directly or indirectly', 'suffices to prove'
            mw9_=float(getattr(cfg,'inf_sym_minw',0.0))                         #   inf_sym_minw: AND the pair itself was read at least this often (decayed count) - symmetric
            n9_=len(Wx_); Wx_={k0_:w0_ for k0_,w0_ in Wx_.items() if w0_>=sym_*max(dg9_[k0_[0]],dg9_[k0_[1]]) and w0_>=mw9_}   #   one-offs (two rare words once) are not evidence
            osst_['er']=osst_.get('er','')+f" | SYMMETRY (both ways >= {sym_:.2f}): {n9_-len(Wx_)} of {n9_} links one-way, set aside"
        ents_=sorted(set(x_ for k0_ in Wx_ for x_ in k0_)); eid_={e_:j_ for j_,e_ in enumerate(ents_)}; n0_=len(ents_)
        if n0_>2:
            r0_=[];c0_=[];v0_=[]
            for (a_,c_),w_ in Wx_.items(): ia0_,ic0_=eid_[a_],eid_[c_]; r0_+= [ia0_,ic0_]; c0_+=[ic0_,ia0_]; v0_+=[w_,w_]
            Wm_=_sp.csr_matrix((np.array(v0_,np.float32),(np.array(r0_),np.array(c0_))),shape=(n0_,n0_))
            fan_=np.asarray((Wm_>0.5).sum(1)).ravel().astype(np.float32)
            Wf_=Wm_.copy(); Wf_.data=np.log2(1.0+Wf_.data)                      #   f(w) = bits of a link read w times
            dmp_=_sp.diags((fan_<=int(cfg.inf_maxfan)*4)/np.sqrt(np.maximum(1.0,fan_-1.0)))   #   hub damping: a bridge's paths share its specificity
                                                                                #   (cg20x: hubs past 4x the fan limit leave the product - bounds S at GPU scale)
            Sm_=(Wf_@dmp_@dmp_@Wf_).tocoo()                                     #   S[A,C] = sum over bridges B of f(w_AB) f(w_BC) / (fan_B - 1)
            if int(getattr(cfg,'inf_spectral_probe',0)) and n0_>40:            # SPECTRAL JEPA (Balestriero & LeCun, Prop. 1): with 'linked words' as positive pairs and
                try:                                                            #   whitened codes, the JEPA optimum is the top eigenvectors of the normalised link operator
                    import scipy.sparse.linalg as _sla                          #   T = D^-1/2 W D^-1/2 - closed form, no training. Two-hop scores = Phi diag(mu^2) Phi^T,
                    dg_=np.asarray(Wf_.sum(1)).ravel()+1e-9; Dm_=_sp.diags(1/np.sqrt(dg_)); T_=(Dm_@Wf_@Dm_).tocsr()   #   a low-rank, SMOOTHED T^2: graded scores even
                    k9_=min(int(getattr(cfg,'inf_spectral_k',32)),n0_-2); mu_,Ph_=_sla.eigsh(T_,k=k9_,which='LA')   #   where no exact two-step path exists
                    t2s_,_=_inf_truth(cfg); tp_=[(eid_[a_.encode('latin-1')],eid_[c_.encode('latin-1')]) for a_,c_ in t2s_ if a_<c_ and a_.encode('latin-1') in eid_ and c_.encode('latin-1') in eid_]
                    if tp_:
                        Sc_=Sm_.tocsr(); G_=Ph_*(mu_**2)[None,:]; rk_s=[]; rk_e=[]; nop_=0; nop_hit=0
                        for a9_,c9_ in tp_[:200]:
                            sv_=G_[a9_]@Ph_.T; sv_[a9_]=-1e9; rk_s.append(int((sv_>sv_[c9_]).sum())+1)
                            er_=np.asarray(Sc_.getrow(a9_).todense()).ravel(); er_[a9_]=-1e9; ex9_=er_[c9_]; rk_e.append(int((er_>ex9_).sum())+1 if ex9_>0 else n0_)
                            if ex9_<=0: nop_+=1; nop_hit+=int(rk_s[-1]<=10)
                        mrr_=lambda r_: float(np.mean([1.0/x_ for x_ in r_]))
                        osst_['er']=osst_.get('er','')+(f" | SPECTRAL CODES (k {k9_}, top eigenvalues {', '.join(f'{x_:.2f}' for x_ in sorted(mu_)[::-1][:3])}): true two-hop pairs {len(tp_)}: "
                            f"MRR spectral {mrr_(rk_s):.3f} / exact paths {mrr_(rk_e):.3f}, hit@10 spectral {np.mean([r_<=10 for r_ in rk_s]):.2f} / exact {np.mean([r_<=10 for r_ in rk_e]):.2f}; "
                            f"pairs with NO exact path {nop_}, found by spectral codes in the top 10: {nop_hit}")
                except Exception as e_: osst_['er']=osst_.get('er','')+f" | SPECTRAL CODES failed ({type(e_).__name__}: {e_})"[:200]
            mf_=int(cfg.inf_maxfan); th_=float(getattr(cfg,'inf_graph_theta',1.585)); need_={}
            for i_ in S_['new']:
                for n_ in set(_INF_NAME.findall(S_['it'][i_]['s'])): need_[n_]=need_.get(n_,0)+1
            cand_=[]
            for a0_,c0b_,v1_ in zip(Sm_.row,Sm_.col,Sm_.data):
                if a0_>=c0b_ or v1_<th_: continue                               #   upper triangle; value below the threshold
                A,C=ents_[a0_],ents_[c0b_]
                if (min(A,C),max(A,C)) in W_ or (min(A,C),max(A,C)) in S_['pairs']: continue   #   already directly linked / already concluded
                nd_=1.0+math.log2(1.0+need_.get(A,0)+need_.get(C,0))
                cand_.append((float(v1_)*nd_,float(v1_),a0_,c0b_))
            cand_.sort(reverse=True); osst_['cand']=len(cand_); osst_['pro']=0; osst_['nosrc']=0
            if int(getattr(cfg,'inf_stage',0)):                                 # STAGING BUFFER (over-generate, compete, then an absolute MDL bar): candidates are
                sv2_=int(getattr(cfg,'inf_stage',0))>=2                         # STAGING v2 (cg20z): (a) evidence fades at the rate of the links it rests on
                ST_=S_.setdefault('stage',{}); dk_=float(getattr(cfg,'inf_stage_decay',0.8)) if not sv2_ else float(getattr(cfg,'inf_lm_decay',0.97))   #   (v1: 0.8 per sleep capped evidence at 5 sleeps' worth, about half the bar - nothing ever passed)   #   NOT committed when first proposed; their evidence accumulates
                for k9_ in list(ST_):                                           #   across sleeps (path strength x independent bridges x salience), decays
                    ST_[k9_]['s']*=dk_                                          #   without new support, the weakest are ejected when the buffer is full,
                    if ST_[k9_]['s']<0.5: del ST_[k9_]                          #   and only candidates whose evidence exceeds the cost of stating the link
                sth_=float(np.quantile([float(v_[0]/max(1,v_[1])) for v_ in _UNITS.get('sal',{}).values()],0.5)) if _UNITS.get('sal_on') and _UNITS.get('sal') else None
                for pri_,v1_,a0_,c0b_ in cand_:                                 #   (about 2 log2 N bits for N entities) are committed
                    k9_=(ents_[a0_],ents_[c0b_]); ra9_=set(Wm_.getrow(a0_).indices); rc9_=set(Wm_.getrow(c0b_).indices); co9_=len(ra9_&rc9_)
                    sf9_=1.0
                    if sth_ is not None:
                        for e9_ in k9_:
                            sv9_=_UNITS['sal'].get(e9_)
                            if sv9_ and sv9_[1]>=2 and sv9_[0]/sv9_[1]<sth_: sf9_*=0.25   #   (a predictable endpoint carries less evidence)
                    e9_=ST_.setdefault(k9_,{'s':0.0,'n':0,'co':0,'v0':int(model.__dict__.get('_visit',0))})
                    if sv2_:                                                    #   (b) a sleep adds evidence only if a premise was READ again this sleep (re-encounter,
                        ns9_=sum(NW_.get((min(k9_[0],ents_[j9_]),max(k9_[0],ents_[j9_])),0)+NW_.get((min(k9_[1],ents_[j9_]),max(k9_[1],ents_[j9_])),0) for j9_ in (ra9_&rc9_))
                        e9_['s']+=float(v1_)*min(co9_,3)*sf9_*min(ns9_,3)           #   not mere re-proposal from links already stored), up to 3 reads per sleep
                    else: e9_['s']+=float(v1_)*min(co9_,3)*sf9_
                    e9_['n']+=1; e9_['co']=max(e9_['co'],co9_)
                cap9_=int(getattr(cfg,'inf_stage_cap',20000)); nej_=0
                if len(ST_)>cap9_:
                    for k9_ in sorted(ST_,key=lambda k_: ST_[k_]['s'])[:len(ST_)-cap9_]: del ST_[k9_]; nej_+=1
                thr9_=2.0*math.log2(max(2,n0_))*float(getattr(cfg,'inf_stage_mult',1.0 if not sv2_ else 0.35))   #   (c) bar: 0.35 x the cost of naming the pair (v2 default)
                nb9_=len(cand_); cand_=[c9_ for c9_ in cand_ if ST_.get((ents_[c9_[2]],ents_[c9_[3]]),{}).get('s',0.0)>=thr9_]
                cand_.sort(key=lambda c9_: -ST_[(ents_[c9_[2]],ents_[c9_[3]])]['s'])
                t2s_,t3s_=_inf_truth(cfg); tp9_=sum(1 for k9_ in ST_ if (k9_[0].decode('latin-1','replace'),k9_[1].decode('latin-1','replace')) in t2s_|t3s_)
                osst_['stage']=(f" | STAGING: {nb9_} proposed this sleep, buffer {len(ST_)} (probe-true {tp9_}), ejected {nej_}, bar {thr9_:.1f} bits, {len(cand_)} past the bar")
            for pri_,v1_,a0_,c0b_ in cand_[:int(cfg.inf_budget)]:
                A,C=ents_[a0_],ents_[c0b_]
                ra_=Wm_.getrow(a0_); rc_=Wm_.getrow(c0b_); com_=set(ra_.indices)&set(rc_.indices)
                if not com_: continue
                bb_=max(com_,key=lambda j_: min(ra_[0,j_],rc_[0,j_])/max(1.0,fan_[j_]-1.0)); B=ents_[bb_]   #   the best bridge (for the citation)
                if fan_[bb_]>mf_*4: continue
                gx_=lambda X_: set(j_ for g_ in [X_]+S_.get('ergrp',{}).get(X_,[]) for j_ in S_['idx'].get(g_,[]))   #   (all spellings of a resolved entity)
                ab_=sorted(gx_(A)&gx_(B)); bc_=sorted(gx_(B)&gx_(C))
                pr0_=[(u_,w0_) for u_ in ab_[-8:] for w0_ in bc_[-8:] if u_!=w0_ and not (S_['it'][u_].get('v')==S_['it'][w0_].get('v') and S_['it'][u_]['d']==S_['it'][w0_]['d'])]
                if not pr0_: osst_['nosrc']+=1; continue
                ia_,ic_=pr0_[-1]
                if (S_['it'][ia_]['dep']+S_['it'][ic_]['dep'])>1: continue
                mg_=None
                if int(getattr(cfg,'inf_mutual',0)):                            # MUTUAL COMPRESSION: the two source sentences must help the trunk predict each
                    rsm_=np.random.RandomState(int(ia_)*31+int(ic_))                     #   other, beyond a random sentence in the same place (bits saved on the second
                    js_=[j_ for j_ in rsm_.choice(len(S_['it']),min(12,len(S_['it'])),replace=False) if j_ not in (ia_,ic_) and not S_['it'][j_].get('gone')][:2]   #   sentence)
                    if js_:
                        sa9_,sc9_=S_['it'][ia_]['s'],S_['it'][ic_]['s']; sl9_=int(model.__dict__.get('_t2s',{}).get(S_['it'][ia_]['d'],S_['it'][ia_]['d']))
                        bq_=_inf_cond_bits(model,cfg,[(sa9_,sc9_),(sc9_,sa9_)]+[(S_['it'][j_]['s'],sc9_) for j_ in js_]+[(S_['it'][j_]['s'],sa9_) for j_ in js_],sl9_)
                        nr_=len(js_); g1_=float(np.mean(bq_[2:2+nr_]))-bq_[0]; g2_=float(np.mean(bq_[2+nr_:]))-bq_[1]
                        mg_=min(g1_,g2_) if int(getattr(cfg,'inf_mutual_min',0)) else 0.5*(g1_+g2_)   #   (both directions: mean, or inf_mutual_min - BOTH must help)
                        tr9_=(A.decode('latin-1','replace'),C.decode('latin-1','replace')) in _inf_truth(cfg)[0]
                        osst_.setdefault('mgl',[]).append((mg_,tr9_))
                        if mg_<float(getattr(cfg,'inf_mutual_bits',8.0)): osst_['mgrej']=osst_.get('mgrej',0)+1; continue
                cor_=len(com_); cm9_=int(getattr(cfg,'inf_corrob',0))            # SELECTION (cg20z tests): CORROBORATION = number of independent bridges
                if cm9_==1 and cor_<2: osst_['correj']=osst_.get('correj',0)+1; continue   #   1: two or more bridges required
                if int(getattr(cfg,'inf_plaus',0)) and (cm9_!=2 or cor_<2):     # TRUNK AS JUDGE (2: only single-route links are judged): the conclusion must
                    rsp_=np.random.RandomState(int(a0_)*7919+int(c0b_))                 #   rank near the top among alternatives, by the trunk alone (store off)
                    alt_=[]
                    while len(alt_)<9 and n0_>12:
                        j9_=int(rsp_.randint(n0_))
                        if j9_ not in (a0_,c0b_,bb_) and ents_[j9_][:1].isalpha(): alt_.append(ents_[j9_].decode('latin-1','replace'))
                    cn9_=C.decode('latin-1','replace'); sl9_=int(model.__dict__.get('_t2s',{}).get(S_['it'][ia_]['d'],S_['it'][ia_]['d']))
                    with torch.no_grad(): lq_=_inf_lp(model,cfg,f"So {A.decode('latin-1','replace')} is connected to ",[cn9_]+alt_,sl9_,False)
                    sq_=[v_/max(1,len(x_)+1) for v_,x_ in zip(lq_,[cn9_]+alt_)]     #   (per byte: no length bias)
                    rk_=sum(1 for v_ in sq_[1:] if v_>sq_[0])
                    if rk_>=int(getattr(cfg,'inf_plaus_rank',3)): osst_['plrej']=osst_.get('plrej',0)+1; continue
                    osst_['plok']=osst_.get('plok',0)+1
                osst_.setdefault('corh',{}); osst_['corh'][min(cor_,3)]=osst_['corh'].get(min(cor_,3),0)+1
                sa0_,sc0_=S_['it'][ia_]['s'],S_['it'][ic_]['s']
                sp_=lambda X_,t_: next((g_ for g_ in [X_]+S_.get('ergrp',{}).get(X_,[]) if g_ in t_),X_)   #   the spelling used in that sentence
                r1_=_inf_rel(sa0_,sp_(A,sa0_),sp_(B,sa0_)); r2_=_inf_rel(sc0_,sp_(B,sc0_),sp_(C,sc0_))
                if r1_ is None or r2_ is None:
                    r1b_=_inf_rel(sc0_,sp_(C,sc0_),sp_(B,sc0_)); r2b_=_inf_rel(sa0_,sp_(B,sa0_),sp_(A,sa0_))
                    if r1b_ is not None and r2b_ is not None: A,C,r1_,r2_,ia_,ic_=C,A,r1b_,r2b_,ic_,ia_
                typed_=(r1_ is not None and r2_ is not None)
                sent_=(f"{A.decode()} {r1_} someone who {r2_} {C.decode()}." if typed_ else f"So {A.decode()} is connected to {C.decode()}.")
                new_.append({'A':A.decode(),'B':B.decode(),'C':C.decode(),'r':(r1_,r2_),'typed':typed_,'s':sent_,'src':(ia_,ic_),'d':S_['it'][ia_]['d'],
                             'dep':1+S_['it'][ia_]['dep']+S_['it'][ic_]['dep'],'gain':v1_,'pri':pri_,'fanpen':0.0,'onescan':True,'w9':bool(int(getattr(cfg,'inf_conc_reenter',0))),'cor':len(com_),'mg':(round(mg_,1) if mg_ is not None else None),'stg':round(S_.get('stage',{}).get((ents_[a0_],ents_[c0b_]),{}).get('s',0.0),1)})
                S_['pairs'].add((min(A,C),max(A,C))); osst_['pro']+=1; st_['prop']+=1
                if int(getattr(cfg,'inf_conc_reenter',0)): k0_=(min(A,C),max(A,C)); W_[k0_]=W_.get(k0_,0.0)+9.0   #   a promoted conclusion is a strong link
            t2o_,t3o_=_inf_truth(cfg); wt_=sum(1 for p0_,v1_,a0_,c0b_ in cand_ if (ents_[a0_].decode('latin-1','replace'),ents_[c0b_].decode('latin-1','replace')) in t2o_|t3o_)
            st_['graph']=(f"  [inference ONE-SCAN SALIENCE] W: {n0_} entities, {len(W_)} links"+(f" (sliding window {gw_} visits: {osst_.get('gw',0)} links aged out)" if gw_>0 else "")+f" ({osst_['w']} read this sleep, {osst_['un']} retracted) | S = W'W' gives {Sm_.nnz} two-step paths in one product, "
                          f"{osst_['cand']} new above {th_:.2f} (true probe chains among them {wt_}) | PROMOTED {osst_['pro']} by gain x need, no citable sources {osst_['nosrc']}"
                          +(f" | re-read after new no-fact contexts: {osst_['rd']} links taken back" if 'rd' in osst_ else "")+osst_.get('er','')+osst_.get('stage','')+(lambda l_: f" | MUTUAL COMPRESSION: {len(l_)} checked, rejected {osst_.get('mgrej',0)} (< {float(getattr(cfg,'inf_mutual_bits',8.0)):.0f} bits) | median bits saved: probe-true {np.median([g_ for g_,t_ in l_ if t_]) if any(t_ for _,t_ in l_) else float('nan'):.1f}, others {np.median([g_ for g_,t_ in l_ if not t_]) if any(not t_ for _,t_ in l_) else float('nan'):.1f}" if l_ else "")(osst_.get('mgl',[]))+(f" | SELECTION: corroboration (bridges 1/2/3+) {osst_.get('corh',{})}, rejected single-route {osst_.get('correj',0)}, trunk judge passed {osst_.get('plok',0)} rejected {osst_.get('plrej',0)}" if (int(getattr(cfg,'inf_corrob',0)) or int(getattr(cfg,'inf_plaus',0))) else "")+(f" | RELATION FILTER: {osst_['relf'][0]} of {osst_['relf'][1]} links dropped (no relation phrase seen with >= {rmp_} pairs; {osst_['relf'][2]} phrases qualify)" if 'relf' in osst_ else ""))
        br_=[]
    if gr_==4:                                                                  # FUZZY LINK MEMORY + ONE-SCAN READOUT
        _lm_init(cfg); fst_={'w':0,'q':0,'hub':0,'cand':0,'pro':0,'nosrc':0,'merge':0,'fmerge':0}; th_=float(getattr(cfg,'inf_graph_theta',1.585)); tau_=float(getattr(cfg,'inf_lm_tau',0.6))
        _LM['M']*=float(getattr(cfg,'inf_lm_decay',0.97))
        for i_ in S_.get('lmw',[]):                                             #   retraction (superseded sentences leave the memory)
            it_=S_['it'][i_]
            if it_.get('sup') and not it_.get('lmu'):
                for a_,c_ in it_.get('lml',()): _lm_write(a_,c_,-1.0)
                it_['lmu']=True
        for i_ in S_['new']:
            it_=S_['it'][i_]
            if it_.get('k')=='inf' or it_.get('sup'): continue
            for m_ in _INF_TRIP.finditer(it_['s']):
                a_,c_=m_.group(1),m_.group(3)
                if a_!=c_ and ok_(a_) and ok_(c_):
                    _lm_write(a_,c_,1.0); fst_['w']+=1; it_.setdefault('lml',[]).append((a_,c_)); S_.setdefault('lmw',[]).append(i_) if len(it_['lml'])==1 else None
        nm_=_LM['names']; mf_=int(cfg.inf_maxfan); mg_=float(getattr(cfg,'inf_lm_merge',0.0)); vt_={}
        for c_ in inf_probes(cfg):
            if c_.get('var'): vt_[c_['names'][1].encode('latin-1')]=c_['var'].encode('latin-1')
        if _LM.get('arr') is None or _LM['arr'][0].shape[0]!=len(nm_): _LM['arr']=(np.stack(_LM['idx']),np.stack(_LM['sgn'])) if nm_ else None
        if mg_>0 and nm_:                                                       #   all code overlaps in one sparse product (trigram units only)
            import scipy.sparse as _sp
            I_,S0_=_LM['arr']; Cs_=_sp.csr_matrix((S0_.ravel(),(np.repeat(np.arange(len(nm_)),I_.shape[1]),I_.ravel())),shape=(len(nm_),_LM['D'])); Cs_.sum_duplicates()
        need_={}
        for i_ in S_['new']:
            for n_ in set(_INF_NAME.findall(S_['it'][i_]['s'])): need_[n_]=need_.get(n_,0)+1
        P_={}; via_={}
        for B in list(br_)[:int(getattr(cfg,'inf_lm_q',3000))]:
            if B not in _LM['codes']: continue
            grp_=[B]
            if mg_>0:                                                           #   MERGE: the bridge's near-spellings stand with it
                sm_=np.asarray((Cs_@Cs_[_LM['codes'][B][2]].T).todense()).ravel()/_LM['k']
                for j_ in np.nonzero(sm_>=mg_)[0]:
                    if nm_[j_]==B: continue
                    if int(getattr(cfg,'inf_lm_merge_mdl',0)):
                        V_=nm_[j_]
                        if (min(B,V_),max(B,V_)) in S_.setdefault('lmsep',set()): fst_['sepd']=fst_.get('sepd',0)+1; continue
                        eb_=_lev(B.lower(),V_.lower())*math.log2(27.0*max(len(B),len(V_)))   #   bits to code V as an edit of B
                        if eb_>=math.log2(max(2,len(nm_))): fst_['mdlno']=fst_.get('mdlno',0)+1; continue   #   vs bits to name it as its own entity
                    grp_.append(nm_[j_]); fst_['merge']+=1; fst_['fmerge']+=int(not (vt_.get(B)==nm_[j_] or vt_.get(nm_[j_])==B))
            sc_=sum(_lm_partners(g_) for g_ in grp_); fst_['q']+=1
            for g_ in grp_: sc_[_LM['codes'][g_][2]]=0.0
            pi_=np.nonzero(sc_>=tau_)[0]
            if len(grp_)>1 and len(pi_)>mf_ and int(getattr(cfg,'inf_lm_merge_mdl',0)):
                for g_ in grp_[1:]: S_.setdefault('lmsep',set()).add((min(B,g_),max(B,g_)))   #   COLLISION: separate this pair for good
            if len(grp_)>1 and len(pi_)>mf_ and (int(getattr(cfg,'inf_lm_merge_fan',0)) or int(getattr(cfg,'inf_lm_merge_mdl',0))):   #   a merge that turns the bridge into a hub is undone (it costs
                fst_['unm']=fst_.get('unm',0)+1; fst_['merge']-=len(grp_)-1                #   the specificity it was meant to add)
                fst_['fmerge']-=sum(1 for g_ in grp_[1:] if not (vt_.get(B)==g_ or vt_.get(g_)==B)); grp_=[B]
                sc_=_lm_partners(B).copy(); sc_[_LM['codes'][B][2]]=0.0; pi_=np.nonzero(sc_>=tau_)[0]
            if len(pi_)<2: continue
            if len(pi_)>mf_: fst_['hub']+=1; continue
            dm_=1.0/max(1.0,len(pi_)-1.0)
            for x_ in range(len(pi_)):
                for y_ in range(x_+1,len(pi_)):
                    A,C=nm_[pi_[x_]],nm_[pi_[y_]]
                    if A in grp_ or C in grp_ or _lm_sim(A,C)>=(mg_ if mg_>0 else 0.5): continue   #   (no chain between two spellings of one name)
                    k0_=(min(A,C),max(A,C)); v1_=math.log2(1+float(sc_[pi_[x_]]))*math.log2(1+float(sc_[pi_[y_]]))*dm_
                    P_[k0_]=P_.get(k0_,0.0)+v1_
                    if v1_>=via_.get(k0_,(0,None))[0]: via_[k0_]=(v1_,grp_)
        cand_=[]
        for k0_,v1_ in P_.items():
            if v1_<th_ or k0_ in S_['pairs'] or _lm_pair(k0_[0],k0_[1])>=tau_: continue
            cand_.append((v1_*(1.0+math.log2(1.0+need_.get(k0_[0],0)+need_.get(k0_[1],0))),v1_,k0_))
        cand_.sort(reverse=True); fst_['cand']=len(cand_)
        for pri_,v1_,(A,C) in cand_[:int(cfg.inf_budget)]:
            grp_=via_[(A,C)][1]; B=grp_[0]
            ix0_=lambda X_,Y_: sorted(set(S_['idx'].get(X_,[]))&set(j_ for g_ in Y_ for j_ in S_['idx'].get(g_,[])))
            ab_=ix0_(A,grp_); bc_=ix0_(C,grp_)
            pr0_=[(u_,w0_) for u_ in ab_[-8:] for w0_ in bc_[-8:] if u_!=w0_ and not (S_['it'][u_].get('v')==S_['it'][w0_].get('v') and S_['it'][u_]['d']==S_['it'][w0_]['d'])]
            if not pr0_: fst_['nosrc']+=1; continue
            ia_,ic_=pr0_[-1]
            if (S_['it'][ia_]['dep']+S_['it'][ic_]['dep'])>1: continue
            sa0_,sc0_=S_['it'][ia_]['s'],S_['it'][ic_]['s']
            Ba_=next((g_ for g_ in grp_ if g_ in sa0_),B); Bc_=next((g_ for g_ in grp_ if g_ in sc0_),B)
            r1_=_inf_rel(sa0_,A,Ba_); r2_=_inf_rel(sc0_,Bc_,C)
            if r1_ is None or r2_ is None:
                r1b_=_inf_rel(sc0_,C,Bc_); r2b_=_inf_rel(sa0_,Ba_,A)
                if r1b_ is not None and r2b_ is not None: A,C,r1_,r2_,ia_,ic_=C,A,r1b_,r2b_,ic_,ia_
            typed_=(r1_ is not None and r2_ is not None)
            sent_=(f"{A.decode()} {r1_} someone who {r2_} {C.decode()}." if typed_ else f"So {A.decode()} is connected to {C.decode()}.")
            new_.append({'A':A.decode(),'B':B.decode(),'C':C.decode(),'r':(r1_,r2_),'typed':typed_,'s':sent_,'src':(ia_,ic_),'d':S_['it'][ia_]['d'],
                         'dep':1+S_['it'][ia_]['dep']+S_['it'][ic_]['dep'],'gain':v1_,'pri':pri_,'fanpen':0.0,'lm':True,'fuzzy':len(grp_)>1})
            S_['pairs'].add((min(A,C),max(A,C))); fst_['pro']+=1; st_['prop']+=1
            if int(getattr(cfg,'inf_conc_reenter',0)): _lm_write(A,C,9.0)
        t2m_,t3m_=_inf_truth(cfg); wt_=sum(1 for _,_,(a0_,c0_) in cand_ if (a0_.decode('latin-1','replace'),c0_.decode('latin-1','replace')) in t2m_|t3m_)
        st_['graph']=(f"  [inference FUZZY LINK MEMORY] {len(nm_)} entities, {fst_['w']} links written this sleep, codes: exact"+(f" + spelling (weight {_LM['fz']:.2f})" if _LM.get('fz',0)>0 else "")+
                      f" | queries {fst_['q']}, hubs {fst_['hub']}, bridge merges {fst_['merge']} (not a planted variant: {fst_['fmerge']}; undone as hub-making {fst_.get('unm',0)}; refused by MDL {fst_.get('mdlno',0)}; separated pairs {len(S_.get('lmsep',()))}) | {fst_['cand']} new pairs above {th_:.2f} (true probe chains {wt_}), "
                      f"PROMOTED {fst_['pro']} by gain x need, no citable sources {fst_['nosrc']}")
        br_=[]
    if gr_==2:                                                                  # LINK MEMORY: write this visit's links, then link by querying the memory
        _lm_init(cfg); lmst_={'w':0,'q':0,'hub':0,'cand':0,'pro':0,'nosrc':0,'dir':0}; th_=float(getattr(cfg,'inf_graph_theta',1.585)); tau_=float(getattr(cfg,'inf_lm_tau',0.6))
        _LM['M']*=float(getattr(cfg,'inf_lm_decay',0.97))                      #   links never read again fade
        nun_=0
        if int(getattr(cfg,'inf_lm_retract',0)):                               # RETRACTION: a sentence the grounding audit superseded is subtracted from the
            for i_ in S_.get('lmw',[]):                                         #   memory (the old link stops being recalled at once, not after slow decay)
                it_=S_['it'][i_]
                if it_.get('sup') and not it_.get('lmu'):
                    for a_,c_ in it_.get('lml',()): _lm_write(a_,c_,-1.0); nun_+=1
                    it_['lmu']=True
        lmst_['un']=nun_
        for i_ in S_['new']:
            it_=S_['it'][i_]
            if it_.get('k')=='inf' or it_.get('sup'): continue
            for m_ in _INF_TRIP.finditer(it_['s']):
                a_,c_=m_.group(1),m_.group(3)
                if a_!=c_ and ok_(a_) and ok_(c_):
                    _lm_write(a_,c_,1.0); lmst_['w']+=1
                    if int(getattr(cfg,'inf_lm_retract',0)): it_.setdefault('lml',[]).append((a_,c_)); S_.setdefault('lmw',[]).append(i_) if len(it_['lml'])==1 else None
        nm_=_LM['names']; nq_=int(getattr(cfg,'inf_lm_q',3000)); mf_=int(cfg.inf_maxfan)
        for B in list(br_)[:nq_]:
            if B not in _LM['codes']: continue
            sc_=_lm_partners(B); lmst_['q']+=1; ib0_=_LM['codes'][B][2]; sc_[ib0_]=0.0
            pi_=np.nonzero(sc_>=max(tau_,float(getattr(cfg,'inf_lm_rel',0.35))*float(sc_.max())))[0]   #   (relative cut: a strong link leaks onto entities sharing its units)
            if len(pi_)<2: continue
            if len(pi_)>mf_: lmst_['hub']+=1; continue                          #   a hub: too many partners to link through
            pi_=pi_[np.argsort(-sc_[pi_])]; f0_=len(pi_)
            for x_ in range(len(pi_)):
                for y_ in range(x_+1,len(pi_)):
                    A,C=nm_[pi_[x_]],nm_[pi_[y_]]; key_=(min(A,C),max(A,C)); lmst_['cand']+=1
                    if key_ in S_['pairs']: continue
                    if _lm_pair(A,C)>=tau_: S_['pairs'].add(key_); lmst_['dir']+=1; continue   #   already directly linked
                    val_=math.log2(1+min(float(sc_[pi_[x_]]),float(sc_[pi_[y_]])))-math.log2(max(1,f0_*(f0_-1)/2))
                    if val_<th_ or len(new_)>=int(cfg.inf_budget): continue
                    ab_=sorted(set(S_['idx'].get(A,[]))&set(S_['idx'].get(B,[]))); bc_=sorted(set(S_['idx'].get(B,[]))&set(S_['idx'].get(C,[])))
                    pr_=[(u_,w0_) for u_ in ab_[-8:] for w0_ in bc_[-8:] if u_!=w0_ and not (S_['it'][u_].get('v')==S_['it'][w0_].get('v') and S_['it'][u_]['d']==S_['it'][w0_]['d'])]
                    if not pr_: lmst_['nosrc']+=1; continue                     #   sources to cite, from more than one passage
                    ia_,ic_=pr_[-1]
                    if (S_['it'][ia_]['dep']+S_['it'][ic_]['dep'])>1: continue
                    sa0_,sc0_=S_['it'][ia_]['s'],S_['it'][ic_]['s']; r1_=_inf_rel(sa0_,A,B); r2_=_inf_rel(sc0_,B,C)
                    if r1_ is None or r2_ is None:
                        r1b_=_inf_rel(sc0_,C,B); r2b_=_inf_rel(sa0_,B,A)
                        if r1b_ is not None and r2b_ is not None: A,C,r1_,r2_,ia_,ic_=C,A,r1b_,r2b_,ic_,ia_
                    typed_=(r1_ is not None and r2_ is not None)
                    sent_=(f"{A.decode()} {r1_} someone who {r2_} {C.decode()}." if typed_ else f"So {A.decode()} is connected to {C.decode()}.")
                    new_.append({'A':A.decode(),'B':B.decode(),'C':C.decode(),'r':(r1_,r2_),'typed':typed_,'s':sent_,'src':(ia_,ic_),'d':S_['it'][ia_]['d'],
                                 'dep':1+S_['it'][ia_]['dep']+S_['it'][ic_]['dep'],'gain':val_,'fanpen':math.log2(max(1,f0_*(f0_-1)/2)),'lm':True})
                    S_['pairs'].add(key_); lmst_['pro']+=1; st_['prop']+=1
                    if int(getattr(cfg,'inf_conc_reenter',0)): _lm_write(A,C,9.0)   #   a promoted conclusion is a strong link (three-hop through it)
        t2m_,t3m_=_inf_truth(cfg); rc_=0; nc_=0
        for q_ in inf_probes(cfg):
            n0_=[n_.encode('latin-1') for n_ in q_['names'][:3]]
            if all(n_ in _LM['codes'] for n_ in n0_):
                nc_+=1; rc_+=int(_lm_pair(n0_[0],n0_[1])>=tau_ and _lm_pair(n0_[1],n0_[2])>=tau_)
        rs0_=np.random.RandomState(vis_); nz_=[]
        if len(nm_)>50:
            for _ in range(200): u0_,u1_=rs0_.randint(len(nm_),size=2); nz_.append(abs(_lm_pair(nm_[u0_],nm_[u1_])))
        st_['graph']=(f"  [inference LINK MEMORY] {_LM['D']}x{_LM['D']} memory, {_LM['k']} units per entity, {len(nm_)} entities, {_LM['nw']} links written ({lmst_['w']} this sleep) | "
                      f"retracted {lmst_.get('un',0)} | queries {lmst_['q']}, hubs {lmst_['hub']}, partner pairs {lmst_['cand']}, already linked {lmst_['dir']}, no citable sources {lmst_['nosrc']}, PROMOTED {lmst_['pro']} (threshold {th_:.2f} bits) | "
                      f"probe chains with both links recalled {rc_}/{nc_} | crosstalk (random pairs) mean {np.mean(nz_) if nz_ else float('nan'):.3f}, 99th pct {np.percentile(nz_,99) if nz_ else float('nan'):.3f} (a link read once = 1.0)")
        br_=[]   # cg20i: rarest bridges first (the budget goes to rare links)
    for B in br_:
        part={}; prel_={}
        slots_=_UNITS.get('all') and int(getattr(cfg,'inf_link_slots',0))
        for i_ in S_['idx'].get(B,[]):
            it_=S_['it'][i_]
            if it_.get('sup'): continue                                         # superseded by the grounding audit
            if slots_ and it_.get('k')=='inf':                                  # a conclusion re-entered as 'So A is connected to C.': its ARGUMENTS only, never the template's words
                m6_=_re_inf.match(rb'So (\S+) is connected to (\S+)\.',it_['s']) or (_re_inf.match(rb'(\S+) .{1,60}? someone who .{1,60}? (\S+)\.$',it_['s']) if int(getattr(cfg,'inf_conc_reenter',0)) else None)
                if m6_ and B in (m6_.group(1),m6_.group(2)):
                    A=m6_.group(2) if m6_.group(1)==B else m6_.group(1)
                    if A!=B and ok_(A): part.setdefault(A,[]).append(i_)
                continue
            if slots_ and it_.get('k')!='inf':                                  # word-level layer: only the OTHER ARGUMENT of a fact that has B as an argument
                for m5_ in _INF_TRIP.finditer(it_['s']):
                    a5_,c5_=m5_.group(1),m5_.group(3)
                    if B in (a5_,c5_):
                        A=c5_ if a5_==B else a5_
                        if A!=B and ok_(A): part.setdefault(A,[]).append(i_); prel_.setdefault(A,m5_.group(2))
                continue
            for A in set(_INF_NAME.findall(it_['s'])):
                if A!=B and ok_(A): part.setdefault(A,[]).append(i_)
        st_['bridges']+=1
        hr_=int(getattr(cfg,'inf_hub_rel',0)) and _UNITS.get('all'); rcn_={}
        if hr_:                                                                 # HUBS PER RELATION: 'London' is a fine bridge for 'born in' if that relation's partners stay few
            for A_ in part: rcn_[prel_.get(A_,b'')]=rcn_.get(prel_.get(A_,b''),0)+1
            if all(v_>int(cfg.inf_maxfan) for v_ in rcn_.values()): st_['hub']+=1; continue
        elif len(part)>int(cfg.inf_maxfan): st_['hub']+=1; continue
        ks_=sorted(part)
        if hr_: ks_=[A_ for A_ in ks_ if rcn_.get(prel_.get(A_,b''),0)<=int(cfg.inf_maxfan)]
        for x_ in range(len(ks_)):
            for y_ in range(x_+1,len(ks_)):
                A,C=ks_[x_],ks_[y_]; ia_=part[A][0]; ic_=part[C][0]
                if ia_==ic_ or (S_['it'][ia_]['dep']+S_['it'][ic_]['dep'])>1: continue   # different sentences; never inferred + inferred
                key_=(min(A,C),max(A,C))
                if key_ in S_['pairs']:
                    if rf_:                                                     # RE-ENCOUNTER: the same link reached through another bridge is an independent route
                        c0_=cmap_.get(key_)
                        if c0_ is not None and c0_.get('B')!=B.decode('latin-1') and B.decode('latin-1') not in c0_.setdefault('routes',set()): c0_['routes'].add(B.decode('latin-1')); nrt_+=1
                    continue
                if set(S_['idx'].get(A,[]))&set(S_['idx'].get(C,[])): S_['pairs'].add(key_); continue   # already directly linked: nothing to infer
                if gr_==1:                                                      # LINK GRAPH: every candidate link is kept (no store writes, no cost to predictions);
                    e_=G_.get(key_)                                             #   it is promoted into the store later only if its value clears the threshold
                    if e_ is None: e_=G_[key_]={'A':A,'B':B,'C':C,'ia':ia_,'ic':ic_,'routes':set(),'v0':vis_,'fan':len(part)}; ngn_+=1
                    else: ngu_+=1
                    e_['routes'].add(B); e_['v']=vis_; e_['fan']=min(e_['fan'],len(part))
                    continue
                if _UNITS.get('all') and min(len(part[A])+(9 if int(getattr(cfg,'inf_conc_reenter',0)) and S_['it'][ia_].get('k')=='inf' else 0),
                                             len(part[C])+(9 if int(getattr(cfg,'inf_conc_reenter',0)) and S_['it'][ic_].get('k')=='inf' else 0))<int(getattr(cfg,'inf_conc_minsup',0)): continue   # BOTH source facts must recur (a stored conclusion already passed this test)
                S_['pairs'].add(key_)                                           #   (not marked as tried: the link is reconsidered once its support grows)
                sa_,sc_=S_['it'][ia_]['s'],S_['it'][ic_]['s']
                r1_=_inf_rel(sa_,A,B); r2_=_inf_rel(sc_,B,C); fw_=True
                if r1_ is None or r2_ is None:                                   # try the other direction (C ... B, B ... A)
                    r1b_=_inf_rel(sc_,C,B); r2b_=_inf_rel(sa_,B,A)
                    if r1b_ is not None and r2b_ is not None: A,C,r1_,r2_,ia_,ic_=C,A,r1b_,r2b_,ic_,ia_
                typed_=(r1_ is not None and r2_ is not None)
                sent_=(f"{A.decode()} {r1_} someone who {r2_} {C.decode()}." if typed_ else f"So {A.decode()} is connected to {C.decode()}.")
                f0_=len(part); nw0_=max(2,S_['nwin'])
                if _UNITS.get('all'):                                           # word-level layer: evidence must come from INDEPENDENT contexts
                    it_a_,it_c_=S_['it'][ia_],S_['it'][ic_]
                    if it_a_.get('v')==it_c_.get('v') and it_a_['d']==it_c_['d']: continue   #   (same visit, same domain: a coincidence of one passage, not a link)
                gain_=math.log2(nw0_/max(1,df_.get(B,1)))-math.log2(max(1,f0_*(f0_-1)/2))   # bits: how surprising a bridge this rare is, minus the links it proposes
                if _UNITS.get('all'):
                    re_=int(getattr(cfg,'inf_conc_reenter',0))                  #   (inf_conc_reenter: a stored conclusion counts as well-supported evidence - it already passed)
                    sup_=math.log2(1+min(len(part[A])+(9 if re_ and S_['it'][ia_].get('k')=='inf' else 0),len(part[C])+(9 if re_ and S_['it'][ic_].get('k')=='inf' else 0)))   #   repeated, independent observation of BOTH source facts
                    gain_=(sup_ if str(getattr(cfg,'inf_pool_gain','support'))=='support' else gain_+sup_)-math.log2(max(1,f0_*(f0_-1)/2))   # (rarity is already a filter: not counted twice)
                new_.append({'A':A.decode(),'B':B.decode(),'C':C.decode(),'r':(r1_,r2_),'typed':typed_,'s':sent_,'src':(ia_,ic_),'d':S_['it'][ia_]['d'],
                             'dep':1+S_['it'][ia_]['dep']+S_['it'][ic_]['dep'],'gain':gain_,'fanpen':math.log2(max(1,f0_*(f0_-1)/2))})
                st_['prop']+=1
                if len(new_)>=int(cfg.inf_budget): break
            if len(new_)>=int(cfg.inf_budget): break
        if len(new_)>=int(cfg.inf_budget): break
    if gr_==1:                                                                  # PROMOTION: a graph link enters the store when its value in bits clears the threshold
        ix_=S_['idx']; th_=float(getattr(cfg,'inf_graph_theta',1.585)); ttl_=int(getattr(cfg,'inf_graph_ttl',8)); npr_=0; ndel_=0
        re_=int(getattr(cfg,'inf_conc_reenter',0)); t2g_,t3g_=_inf_truth(cfg); tg_=set()
        for k_,e_ in list(G_.items()):
            if e_.get('promoted'): continue
            if vis_-e_.get('v',vis_)>ttl_: del G_[k_]; ndel_+=1; continue      #   links never met again fade out of the graph
            A,B,C,ia_,ic_=e_['A'],e_['B'],e_['C'],e_['ia'],e_['ic']
            if ia_>=len(S_['it']) or ic_>=len(S_['it']) or S_['it'][ia_].get('gone') or S_['it'][ic_].get('gone'): continue
            sa_=len(set(ix_.get(A,[]))&set(ix_.get(B,[])))+(9 if re_ and S_['it'][ia_].get('k')=='inf' else 0)
            sc_=len(set(ix_.get(B,[]))&set(ix_.get(C,[])))+(9 if re_ and S_['it'][ic_].get('k')=='inf' else 0)
            f0_=e_['fan']; val_=math.log2(1+min(sa_,sc_))+math.log2(len(e_['routes']))-math.log2(max(1,f0_*(f0_-1)/2)); e_['val']=val_
            if (A.decode('latin-1','replace'),C.decode('latin-1','replace')) in t2g_|t3g_: tg_.add(k_)
            if val_<th_ or len(new_)>=int(cfg.inf_budget): continue
            it_a_,it_c_=S_['it'][ia_],S_['it'][ic_]
            if it_a_.get('v')==it_c_.get('v') and it_a_['d']==it_c_['d']: continue   #   evidence from one passage is not promoted
            sa0_,sc0_=it_a_['s'],it_c_['s']
            r1_=_inf_rel(sa0_,A,B); r2_=_inf_rel(sc0_,B,C)
            if r1_ is None or r2_ is None:
                r1b_=_inf_rel(sc0_,C,B); r2b_=_inf_rel(sa0_,B,A)
                if r1b_ is not None and r2b_ is not None: A,C,r1_,r2_,ia_,ic_=C,A,r1b_,r2b_,ic_,ia_
            typed_=(r1_ is not None and r2_ is not None)
            sent_=(f"{A.decode()} {r1_} someone who {r2_} {C.decode()}." if typed_ else f"So {A.decode()} is connected to {C.decode()}.")
            new_.append({'A':A.decode(),'B':B.decode(),'C':C.decode(),'r':(r1_,r2_),'typed':typed_,'s':sent_,'src':(ia_,ic_),'d':S_['it'][ia_]['d'],
                         'dep':1+S_['it'][ia_]['dep']+S_['it'][ic_]['dep'],'gain':val_,'fanpen':math.log2(max(1,f0_*(f0_-1)/2)),'graph':True})
            S_['pairs'].add(k_); e_['promoted']=True; npr_+=1; st_['prop']+=1
        cap0_=int(getattr(cfg,'inf_graph_cap',300000))
        if len(G_)>cap0_:                                                       #   bounded: the oldest, least valuable unpromoted links go first
            for k_ in sorted([k_ for k_,e_ in G_.items() if not e_.get('promoted')],key=lambda k_: (G_[k_].get('v',0),G_[k_].get('val',0.0)))[:len(G_)-cap0_]: del G_[k_]; ndel_+=1
        nlive_=sum(1 for e_ in G_.values() if not e_.get('promoted'))
        st_['graph']=(f"  [inference LINK GRAPH] {len(G_)} links ({nlive_} waiting, {len(G_)-nlive_} promoted) | this sleep: {ngn_} new, {ngu_} met again, {npr_} promoted "
                      f"(threshold {th_:.2f} bits), {ndel_} faded | true probe chains in the graph waiting {sum(1 for k_ in tg_ if not G_.get(k_,{}).get('promoted'))}")
    nst_=0
    if rf_:                                                                     # RE-ENCOUNTER STRENGTHENING: each sleep a live conclusion's support is recounted from
        ix_=S_['idx']; tg_=int(cfg.inf_target)                                  #   the evidence now stored (both source links) plus its independent routes; its value
        for c_ in S_['conc']:                                                   #   in the pool follows, and when the evidence grew it is rehearsed again (spaced)
            if c_.get('freed'): continue
            if not (isinstance(c_.get('A'),str) and isinstance(c_.get('B'),str) and isinstance(c_.get('C'),str)): continue
            A_,B_,C_=(c_['A'].encode('latin-1','replace'),c_['B'].encode('latin-1','replace'),c_['C'].encode('latin-1','replace'))
            sa_=len(set(ix_.get(A_,[]))&set(ix_.get(B_,[]))); sc_=len(set(ix_.get(B_,[]))&set(ix_.get(C_,[])))
            sr_=c_.get('src',(None,None))
            if int(getattr(cfg,'inf_conc_reenter',0)):
                if isinstance(sr_[0],int) and sr_[0]<len(S_['it']) and S_['it'][sr_[0]].get('k')=='inf': sa_+=9
                if isinstance(sr_[1],int) and sr_[1]<len(S_['it']) and S_['it'][sr_[1]].get('k')=='inf': sc_+=9
            su_=math.log2(1+min(sa_,sc_))+math.log2(1+len(c_.get('routes',())))
            c_['gain']=su_-c_.get('fanpen',0.0)
            if su_>c_.get('sup_last',su_)+1e-9: c_['n']=max(0,c_.get('n',0)-tg_//3); c_['refresh']=c_.get('refresh',0)+1; nst_+=1
            c_['sup_last']=su_
    cap_=int(getattr(cfg,'inf_pool',0))
    if cap_>0:                                                                  # THE POOL (rule-free inference layer): links compete on bits saved
        P_=[c_ for c_ in S_['conc'] if not c_.get('freed')]+new_; rel_=float(getattr(cfg,'inf_pool_rel',0.0)); ev_=0; dr_=0
        if rel_>0 and len(P_)>=8 and not (int(getattr(cfg,'inf_pool_room',0)) and len(P_)<=cap_):   # POOL ROOM: the going rate only applies when links compete for room
            md_=float(np.median([c_.get('gain',0.0) for c_ in P_]))
            for c_ in P_:
                if c_.get('gain',0.0)<rel_*md_ and md_>0: c_['freed']=True; c_['evicted']='below going rate'; dr_+=1
        P_=sorted([c_ for c_ in P_ if not c_.get('freed')],key=lambda c_: -c_.get('gain',0.0))
        for c_ in P_[cap_:]: c_['freed']=True; c_['evicted']='pool full'; ev_+=1
        new_=[c_ for c_ in new_ if not c_.get('freed')]
        st_['pool']=f"pool {min(len(P_),cap_)}/{cap_}, {ev_} evicted (worst), {dr_} dropped (below {rel_:.2f} x going rate)"+(f", RE-ENCOUNTER: {nst_} strengthened and rehearsed, {nrt_} new independent routes" if rf_ else "")
    if _kgx_hold(cfg): new_=[c_ for c_ in new_ if not _kgx_ho(cfg,c_)]          # HELD-OUT COMPOSITION: held-out chains get no conclusion
    for c_ in new_:                                                             # conclusions re-enter the store (one more hop allowed)
        i_=len(S_['it']); S_['it'].append({'s':f"So {c_['A']} is connected to {c_['C']}.".encode(),'d':c_['d'],'k':'inf','dep':c_['dep']-1})
        for n_ in (c_['A'].encode(),c_['C'].encode()): S_['idx'].setdefault(n_,[]).append(i_)
    S_['new']=[]; S_['conc']+=new_; return new_,st_
def _inf_truth(cfg,current=True):
    """probe ground truth: set of true (A,C) pairs at 2 and 3 hops. FIX (review): current=True follows CHANGED facts - once the domain of a
    true update has been revisited (its new fact planted), the true pair is (A, C2) and the old (A, C) is stale; current=False keeps the
    original pairs (for readouts that classify stale beliefs themselves)"""
    t2=set(); t3=set(); rv_=_INF.get('revis',set()) if current else set()      # 'connected to' is symmetric: both orders count
    for c_ in inf_probes(cfg):
        n_=c_['names']; u_=c_.get('upd') or {}
        if len(n_)==3 and u_.get('type')=='true' and int(c_['doms'][0]) in rv_: t2|={(n_[0],u_['C2']),(u_['C2'],n_[0])}; continue
        t2|={(n_[0],n_[2]),(n_[2],n_[0])}
        if len(n_)>3: t2|={(n_[1],n_[3]),(n_[3],n_[1])}; t3|={(n_[0],n_[3]),(n_[3],n_[0])}
    return t2,t3
@torch.no_grad()
def inf_idmem(model,cfg,S_,tag,D_=None):
    """IDENTIFIER MEMORY (cg20s): every live recorded fact (A, r, B) is stored as code(A)*code(r) -> code(B) (fixed random +-1 codes from hashes,
    summed outer products; superseded / freed facts are simply left out). Questions are answered by lookups with clean-up to the nearest known
    unit; the returned identifier is the next key (no text round trip). Unknown subject -> low clean-up similarity -> ABSTAIN."""
    import zlib as _zl
    F_=[f_ for f_ in (S_.get('facts') or []) if not f_.get('freed')]
    if not F_: return
    D_=int(D_ or getattr(cfg,'inf_id_dim',1024))                                   # capacity: interference grows as sqrt(facts / D); 4096 for thousands of facts
    def code(s_):
        g_=np.random.RandomState(_zl.crc32(s_.encode('latin-1','replace'))%(2**31)); return torch.tensor(g_.choice([-1.0,1.0],D_),dtype=torch.float32)
    M_=torch.zeros(D_,D_); un_=sorted(set(x_ for f_ in F_ for x_ in (f_['A'],f_['B']))); cc_={}
    gc_=lambda s_: cc_[s_] if s_ in cc_ else cc_.setdefault(s_,code(s_))
    for i0_ in range(0,len(F_),256):
        ch_=F_[i0_:i0_+256]; K_=torch.stack([gc_(f_['A'])*gc_(f_['r']) for f_ in ch_]); V_=torch.stack([gc_(f_['B']) for f_ in ch_]); M_+=K_.T@V_
    C_=torch.stack([gc_(u_) for u_ in un_]); sim_=float(getattr(cfg,'inf_qt_sim',0.5))
    def look(cur_,rels_):
        for r_ in rels_:
            v_=(gc_(cur_)*gc_(r_))@M_; s_=F.cosine_similarity(v_[None,:],C_); j_=int(s_.argmax())
            if float(s_[j_])<sim_: return None
            cur_=un_[j_]
        return cur_
    st_={}; P_=inf_probes(cfg)
    for c_ in P_:
        n_,R_=c_['names'],c_['rels']; u_=c_.get('upd')
        if u_ and u_['type']=='true':                                           # (as in the probe eval: always scored against the NEW answer)
            for k_,(qs_,rr_,tr_,old_) in {'UPDATED one-hop':(n_[0],R_[:1],u_['B2'],n_[1]),'UPDATED two-hop':(n_[0],R_[:2],u_['C2'],n_[2])}.items():
                a_=look(qs_,rr_); s0_=st_.setdefault(k_,[0,0,0,0]); s0_[0]+=1; s0_[1]+=int(a_ is not None); s0_[2]+=int(a_==tr_); s0_[3]+=int(a_==old_)
            continue
        for k_,h_ in (('one-hop',1),('two-hop',2),('three-hop',3)):
            if len(R_)<h_: continue
            a_=look(n_[0],R_[:h_]); s0_=st_.setdefault(k_,[0,0,0,0]); s0_[0]+=1; s0_[1]+=int(a_ is not None); s0_[2]+=int(a_==n_[h_])
    rs_=np.random.RandomState(77); ab_=0; nu_=40
    for _ in range(nu_):
        q_=''.join(rs_.choice(list('abcdefghiklmnoprstuvy'),7)).capitalize(); ab_+=int(look(q_,[P_[0]['rels'][0]]) is None)
    out_=[f"{k_} {v_[2]/max(1,v_[0]):.2f} correct, answers {v_[1]/max(1,v_[0]):.2f}"+(f", stale {v_[3]/max(1,v_[0]):.2f}" if k_.startswith('UPDATED') else '') for k_,v_ in st_.items()]
    print(f"  [inference IDENTIFIER MEMORY | {tag}] {len(F_)} live recorded facts, {len(un_)} units | "+" | ".join(out_)+f" | unknown subjects abstained {ab_/nu_:.2f}",flush=True)
def _cue_readout(model,cfg,S_):
    """cg26m CUE REPAIR readout: two-hop with the store on chains the system concluded, subject intact or damaged, repair off vs on"""
    ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd')]; rs_=np.random.RandomState(778)
    cm_={(c_['A'],c_['C']) for c_ in S_.get('conc',[]) if not c_.get('freed')}
    fd_=[c_ for c_ in ch_ if (c_['names'][0],c_['names'][2]) in cm_ or (c_['names'][2],c_['names'][0]) in cm_][:24]
    if len(fd_)<4: print(f"  [inference CUE REPAIR] only {len(fd_)} concluded chains",flush=True); return
    Cs_=[c_['names'][2] for c_ in ch_]; slot_=lambda c_: int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0]))
    cd_={}
    for c_ in fd_: o_=[x_ for x_ in Cs_ if x_!=c_['names'][2]]; rs_.shuffle(o_); cd_[c_['names'][0]]=[c_['names'][2]]+o_[:9]
    dmg_={'intact':lambda n_: n_,'last letter dropped':lambda n_: n_[:-1],'inner letters swapped':lambda n_: n_[0]+n_[2]+n_[1]+n_[3:] if len(n_)>3 else n_,'lower-cased':lambda n_: n_.lower()}
    seg_=[]
    for k_,f_ in dmg_.items():
        a0_=[];a1_=[];ok_=0;bad_=0;ab_=0
        for c_ in fd_:
            A_=c_['names'][0]; pr_=f"So {f_(A_)} is connected to "; fx_=_cue_fix(cfg,pr_)
            if fx_==pr_: ab_+=int(f_(A_)!=A_)
            elif A_ in fx_.split(' '): ok_+=1
            else: bad_+=1
            _INF['cr_off']=True
            try: a0_.append(_inf_score1(model,cfg,pr_,cd_[A_],slot_(c_),True)==0)
            finally: _INF.pop('cr_off',None)
            a1_.append(_inf_score1(model,cfg,pr_,cd_[A_],slot_(c_),True)==0)
        seg_.append(f"{k_}: off {np.mean(a0_):.2f} -> on {np.mean(a1_):.2f} (repaired right {ok_}, wrong {bad_}, abstained {ab_})")
    ds9_=[]                                                                     # cg26n DESCRIPTION CUES (readout): the subject named only by its first link
    for c_ in fd_:
        r0_=(c_.get('rels') or [''])[0]; B_=c_['names'][1]
        if r0_: ds9_.append(_inf_score1(model,cfg,f"So the one who {r0_} {B_} is connected to ",cd_[c_['names'][0]],slot_(c_),True)==0)
    if ds9_: seg_.append(f"described ('the one who <relation> <B>'): {np.mean(ds9_):.2f} (n {len(ds9_)})")
    print(f"  [inference CUE REPAIR] two-hop with the store on {len(fd_)} concluded chains (10 candidates, chance 0.10), subject name: "+" | ".join(seg_),flush=True)
def _kgx_hold(cfg):
    """HELD-OUT COMPOSITION (inf_kgx_hold N): every Nth probe chain never gets a conclusion (no store, fast store, replay or edit of it): its two facts
    are read as usual, so whether the weights answer its two-hop question measures composition the weights learned, not a stored conclusion"""
    N_=int(getattr(cfg,'inf_kgx_hold',0)); rw_=int(getattr(cfg,'inf_rw_nosleep',0))
    if N_<=0 and not rw_: return frozenset()
    if _INF.get('kgho') is None:
        P_=set()
        for i_,c_ in enumerate(inf_probes(cfg) if N_>0 else ()):
            if i_%N_!=N_-1: continue
            n_=c_['names']; P_|={frozenset((n_[a_],n_[b_])) for a_ in range(len(n_)) for b_ in range(a_+2,len(n_))}
        if rw_:                                                                 # 12 Nov (inf_rw_nosleep): the RULE WORLD's two-hop pairs (grandchild-root,
            W_=inf_rule_world(cfg)                                              #   staff-top) are never concluded by the sleep layer, so a composite fact
            for k_ in ('parent','mgr'):                                         #   reaches the trunk only through the relational module's write (step 4):
                m_=W_.get(k_,{})                                                #   the clean control for consolidation of self-inferred facts
                P_|={frozenset((a_,m_[m_[a_]])) for a_ in m_ if m_.get(a_) in m_}
        _INF['kgho']=frozenset(P_)
    return _INF['kgho']
def _kgx_ho(cfg,c_):
    H_=_kgx_hold(cfg); return bool(H_) and frozenset((c_.get('A'),c_.get('C')))in H_
@torch.no_grad()
def _kgx_readout(model,cfg,S_):
    """GRAPH INTO WEIGHTS (readout only, end of run): (1) what is lost if the store is switched off, by conclusion age; (2) what survives when everything
    the store holds about the subject is erased (a retraction the weights do not see); (3) whether the weights compose two facts they hold with no written conclusion"""
    ch_=inf_probes(cfg); alln_=[n_ for c_ in ch_ for n_ in c_['names']]; son_=_INFM.get('M') is not None and _INFM.get('mu') is not None
    t2_,t3k_=_inf_truth(cfg); vm_={}
    for c0_ in S_['conc']:
        if (c0_['A'],c0_['C']) in t2_: vm_.setdefault(frozenset((c0_['A'],c0_['C'])),int(c0_.get('v',0)))
    sv_=_INFM['on']; sk_=_INF.get('corr',False); NM_=_INFM.get('names') or {}
    def sc(pr_,cs_,sl_,mode_,drop_=None,ix_=False):
        pop_=None
        try:
            if mode_=='w': _INFM['on']=False; _INF['corr']=False; j_=_inf_score1(model,cfg,pr_,cs_,sl_,False); return j_ if ix_ else int(j_==0)
            if drop_ is not None:
                k_=drop_.encode('latin-1'); pop_=(k_,NM_.pop(k_)) if k_ in NM_ else None
            _INFM['on']=True; _INF['corr']=True; j_=_inf_score1(model,cfg,pr_,cs_,sl_,True); return j_ if ix_ else int(j_==0)
        finally:
            if pop_ is not None: NM_[pop_[0]]=pop_[1]
            _INFM['on']=sv_; _INF['corr']=sk_
    R_=[]
    try:
        for i_,c_ in enumerate(ch_):
            n_=c_['names']; r_=c_['rels']
            if len(n_)<3 or (c_.get('upd') or {}).get('type')=='true': continue
            rs_=np.random.RandomState(9300+i_); oth_=[x_ for x_ in alln_ if x_ not in n_]; cs2_=[n_[2]]+list(rs_.choice(oth_,19,replace=False))
            csb_=[n_[1]]+list(rs_.choice(oth_,19,replace=False)); csc_=[n_[2]]+list(rs_.choice(oth_,19,replace=False))
            sl_=int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0])); sl1_=int(model.__dict__.get('_t2s',{}).get(c_['doms'][1],c_['doms'][1]))
            p2_=f"So {n_[0]} is connected to "
            w2_=sc(p2_,cs2_,sl_,'w'); s2_=sc(p2_,cs2_,sl_,'s') if son_ else w2_; d2_=sc(p2_,cs2_,sl_,'s',n_[0]) if son_ else w2_
            ja_=sc(f"{n_[0]} {r_[0]} ",csb_,sl_,'w',ix_=True); wa_=int(ja_==0); wb_=sc(f"{n_[1]} {r_[1]} ",csc_,sl1_,'w')
            t2w_=sc(f"{csb_[ja_]} {r_[1]} ",csc_,sl1_,'w')                     # TWO LOOKUPS FROM THE WEIGHTS: 'A r0 ' -> the weights' bridge, then '<bridge> r1 ' -> C
            rb_=csb_[1+int(rs_.randint(19))]; t2r_=sc(f"{rb_} {r_[1]} ",csc_,sl1_,'w')   # CONTROL: a RANDOM wrong bridge (does step 2 need the bridge at all?)
            if son_: js_=sc(f"{n_[0]} {r_[0]} ",csb_,sl_,'s',ix_=True); t2s_=sc(f"{csb_[js_]} {r_[1]} ",csc_,sl1_,'s')
            else: t2s_=t2w_
            ho_=frozenset((n_[0],n_[2])) in _kgx_hold(cfg)
            R_.append(dict(v=vm_.get(frozenset((n_[0],n_[2]))),w2=w2_,s2=s2_,d2=d2_,wa=wa_,wb=wb_,t2w=t2w_,t2s=t2s_,ho=ho_,t2r=t2r_))
        fm_={'s':[],'w':[]}; pm_={q_['names'][0]:q_ for q_ in ch_}
        wr_=[c_ for c_ in S_['conc'] if c_['A'] in pm_ and len(pm_[c_['A']]['names'])>=3 and (c_['A'],c_['C']) not in t2_ and (c_['A'],c_['C']) not in t3k_ and c_['C'] not in pm_[c_['A']]['names']][:60]   # FIX: true three-hop conclusions (A -> D) are not false memories
        for c_ in wr_:                                                          # false memories: the wrong conclusion preferred over the truth, store on and off
            q_=pm_[c_['A']]; sl_=int(model.__dict__.get('_t2s',{}).get(q_['doms'][0],q_['doms'][0])); pr_=f"So {c_['A']} is connected to "
            fm_['w'].append(sc(pr_,[c_['C'],q_['names'][2]],sl_,'w'))
            if son_: fm_['s'].append(sc(pr_,[c_['C'],q_['names'][2]],sl_,'s'))
    finally: _INFM['on']=sv_; _INF['corr']=sk_
    m=lambda xs_: (f"{np.mean(xs_):.2f} (n {len(xs_)})" if xs_ else "- (n 0)")
    cc_=[x_ for x_ in R_ if x_['v'] is not None]; un_=[x_ for x_ in R_ if x_['v'] is None]
    vs_=sorted(x_['v'] for x_ in cc_); B_=[]
    if vs_:
        q1_,q2_=vs_[len(vs_)//3],vs_[(2*len(vs_))//3]
        B_=[('oldest third',[x_ for x_ in cc_ if x_['v']<q1_]),('middle third',[x_ for x_ in cc_ if q1_<=x_['v']<q2_]),('newest third',[x_ for x_ in cc_ if x_['v']>=q2_])]
    print(f"  [GRAPH INTO WEIGHTS 1 | END OF RUN] two-hop (chance 0.05), store on -> store off (weights alone), concluded chains by when the conclusion was made: "
          +" | ".join(f"{nm_} {m([x_['s2'] for x_ in b_])} -> {m([x_['w2'] for x_ in b_])}" for nm_,b_ in B_)
          +f" | never concluded {m([x_['s2'] for x_ in un_])} -> {m([x_['w2'] for x_ in un_])} || false memories preferred over the truth (chance 0.50): store on {m(fm_['s'])}, store off {m(fm_['w'])}",flush=True)
    ok_=[x_ for x_ in cc_ if x_['s2']]
    print(f"  [GRAPH INTO WEIGHTS 2 | END OF RUN] RETRACTION THE WEIGHTS DO NOT SEE: concluded chains answered with the store on, then everything the store holds about the subject erased: still answered "
          +m([x_['d2'] for x_ in ok_])+" | by age: "+" | ".join(f"{nm_} {m([x_['d2'] for x_ in b_ if x_['s2']])}" for nm_,b_ in B_)
          +" (this much of a retracted conclusion would survive in the weights; chance 0.05)",flush=True)
    kn_=[x_ for x_ in R_ if x_['wa'] and x_['wb']]
    print(f"  [GRAPH INTO WEIGHTS 3 | END OF RUN] COMPOSITION FROM THE WEIGHTS: both one-hop facts recalled from the weights alone in {len(kn_)}/{len(R_)} chains; their two-hop from the weights: "
          +f"no written conclusion {m([x_['w2'] for x_ in kn_ if x_['v'] is None])}, with a conclusion {m([x_['w2'] for x_ in kn_ if x_['v'] is not None])} | chains missing a one-hop fact: {m([x_['w2'] for x_ in R_ if not (x_['wa'] and x_['wb'])])} | chance 0.05",flush=True)
    print(f"  [GRAPH INTO WEIGHTS 4 | END OF RUN] TWO LOOKUPS ('A r0 ' -> the bridge the weights pick, then '<bridge> r1 ' -> C; 20 candidates each): weights alone {m([x_['t2w'] for x_ in R_])} "
          +f"(one pass {m([x_['w2'] for x_ in R_])}) | with the store {m([x_['t2s'] for x_ in R_])} (one pass {m([x_['s2'] for x_ in R_])}) | never concluded: two lookups {m([x_['t2w'] for x_ in R_ if x_['v'] is None])} vs one pass {m([x_['w2'] for x_ in R_ if x_['v'] is None])}",flush=True)
    for nm9_,S9_ in (('all chains',R_),('never concluded',[x_ for x_ in R_ if x_['v'] is None])):
        print(f"  [GRAPH INTO WEIGHTS 4b | END OF RUN] TWO LOOKUPS DECOMPOSED, {nm9_}: step 1 (bridge right) {m([x_['wa'] for x_ in S9_])} | step 2 with the true bridge {m([x_['wb'] for x_ in S9_])} | "
              +f"two lookups when step 1 was right {m([x_['t2w'] for x_ in S9_ if x_['wa']])}, when wrong {m([x_['t2w'] for x_ in S9_ if not x_['wa']])} | CONTROL, a random wrong bridge: {m([x_['t2r'] for x_ in S9_])} "
              +"(high = step 2 answers from the relation alone, not from the bridge)",flush=True)
    if _kgx_hold(cfg):
        H_=[x_ for x_ in R_ if x_['ho']]; T_=[x_ for x_ in R_ if not x_['ho']]; lk_=sum(1 for c0_ in S_['conc'] if _kgx_ho(cfg,c0_))
        print(f"  [GRAPH INTO WEIGHTS 5 | END OF RUN] HELD-OUT COMPOSITION (every {int(getattr(cfg,'inf_kgx_hold',0))}th chain never concluded; leaked conclusions {lk_}): two-hop from the weights, one pass: "
              +f"held out {m([x_['w2'] for x_ in H_])} vs trained {m([x_['w2'] for x_ in T_])} | held out with both facts in the weights {m([x_['w2'] for x_ in H_ if x_['wa'] and x_['wb']])} | "
              +f"two lookups held out {m([x_['t2w'] for x_ in H_])} | with the store, held out {m([x_['s2'] for x_ in H_])} | chance 0.05",flush=True)

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_inf_cond_bits': 'platypus.inference.selftaught',
    '_inf_lp': 'platypus.inference.selftaught',
    '_inf_score1': 'platypus.inference.readouts',
}
