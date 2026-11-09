"""Mixing (splice) dreams: proposal, judging, dreaming."""
from platypus.common import _fail
import numpy as np
import torch
import torch.nn.functional as F
from platypus.layers import (
    _CARRY,
)
from platypus.conduit import (
    _carry_load, _carry_save,
)
from platypus.inference.units import (
    _UWORD,
)
from platypus.inference.probes import (
    _bench, inf_probes,
)
from platypus.inference.selftaught import (
    _blankw, _inf_cond_bytes, _mix_mask, _spec_codes, _spec_pass,
)

@torch.no_grad()
def _inf_mix(model,cfg,S_,ts):
    """MIXING WITHOUT A BRIDGE (cg20z lab): pair new sentences with stored ones (random mixing, and the trunk's own nearest neighbours),
    splice them both ways, and keep a pair only if EACH side makes the other cheaper for the trunk - beyond a random sentence in the same
    place, counting only bytes that are not a shared word (no credit for copying) - by at least inf_mix_bits on the weaker side (symmetry)"""
    it_=S_['it']; rs_=np.random.RandomState(1000+int(model.__dict__.get('_visit',0)))
    ml9_=int(getattr(cfg,'inf_mix_minlen',24))                                   # cg24: short fragments ('Mark; the First Epistle of St.') made easy false matches
    ok9_=lambda i_: it_[i_].get('k')=='real' and not it_[i_].get('gone') and not it_[i_].get('sup') and len(it_[i_]['s'])>=ml9_
    vn_=max((it_[i_].get('v',-1) for i_ in range(max(0,len(it_)-5000),len(it_))),default=-1)   # this visit's sentences (S_['new'] is already cleared here)
    new_=[i_ for i_ in range(len(it_)) if it_[i_].get('v',-2)==vn_ and ok9_(i_)]; old_=[i_ for i_ in range(len(it_)) if it_[i_].get('v',-2)!=vn_ and ok9_(i_)]
    tst9_=int(getattr(cfg,'inf_mix_revisit_test',0)); pl9_=new_ if tst9_ else []; spl9_=set(pl9_)   # (local test only: treat this visit's plain sentences as verified replay, this sleep only)
    if tst9_: new_=[]
    if int(getattr(cfg,'inf_mix_revisit',0)):                                    # earlier visits' verified-replay sentences are candidates too
        okv0_=lambda i_: it_[i_].get('k')=='ver' and not it_[i_].get('gone') and not it_[i_].get('sup') and len(it_[i_]['s'])>=ml9_
        old_=old_+[i_ for i_ in range(len(it_)) if it_[i_].get('v',-2)!=vn_ and okv0_(i_)]
    if len(new_)<16 and int(getattr(cfg,'inf_mix_revisit',0)):                 # cg26e REVISITS: on pass 2+ the conduit writes the domain's text as VERIFIED replay windows
        okv9_=lambda i_: (it_[i_].get('k')=='ver' or i_ in spl9_) and not it_[i_].get('gone') and not it_[i_].get('sup') and len(it_[i_]['s'])>=ml9_   #   ('ver', exact 1.00), so cg26b
        nv9_=[i_ for i_ in range(max(0,len(it_)-400000),len(it_)) if it_[i_].get('v',-2)==vn_ and okv9_(i_)]   #   found <4 plain cues and skipped splicing silently
        import collections as _co9
        kc9_=_co9.Counter(str(it_[i_].get('k')) for i_ in range(max(0,len(it_)-400000),len(it_)) if it_[i_].get('v',-2)==vn_)
        if pl9_: sp9_=set(pl9_); nv9_=sorted(set(nv9_)|sp9_)
        sn9_={it_[i_]['s'] for i_ in new_}; nv9_=[i_ for i_ in nv9_ if it_[i_]['s'] not in sn9_]
        print(f"  [inference MIXING | REVISIT] this visit wrote "+", ".join(f"{k_} {v_}" for k_,v_ in kc9_.items())+f" sentences; plain-text cues {len(new_)}, adding {len(nv9_)} verified-replay sentences as cues",flush=True)
        new_=new_+nv9_
    if len(new_)<4 or len(old_)<16:
        print(f"  [inference MIXING] skipped: {len(new_)} cue sentences from this visit, {len(old_)} older sentences",flush=True); return
    nq_=min(int(getattr(cfg,'inf_mix_q',48)),len(new_)); cr_=float(getattr(cfg,'inf_mix_cuerare',0.0))
    if cr_>0:                                                                   # cg22 CUES WITH A NAME: a share of the cues must hold a rare capitalised word (store frequency
        dfq_=S_.get('df',{}); rn_=[i_ for i_ in new_ if any(w_.group(0)[:1].isupper() and 0<dfq_.get(w_.group(0),0)<=int(cfg.inf_df) for w_ in _UWORD.finditer(it_[i_]['s']))]   #   <= inf_df)
        k1_=min(len(rn_),int(round(nq_*cr_))); Q_=list(rs_.choice(rn_,k1_,replace=False)) if k1_ else []; rest_=[i_ for i_ in new_ if i_ not in set(Q_)]
        Q_+=list(rs_.choice(rest_,min(nq_-k1_,len(rest_)),replace=False)) if nq_-k1_>0 and rest_ else []; model.__dict__['_cueinfo']=(k1_,len(Q_),len(rn_),len(new_))
    else: Q_=list(rs_.choice(new_,nq_,replace=False))
    bq_=float(getattr(cfg,'inf_mix_benchcue',0.0))
    if bq_>0 and str(getattr(cfg,'inf_bench','') or ''):                        # cg26b MEASURABLE CUES: a share of the cues hold >=2 benchmark entity words, so their partners
        try:                                                                    #   can be judged against gold (cg26 GPU: ~520 of 565 tested pairs were 'outside' the benchmark)
            t2e9_=_bench(cfg)['_t2e']
            def ne9_(t_):
                e_=set()
                for w_ in _UWORD.finditer(t_): e_|=t2e9_.get(w_.group(0).decode('utf-8','ignore'),set())
                return len(e_)
            sq9_=set(Q_); bn9_=[i_ for i_ in new_ if i_ not in sq9_ and ne9_(it_[i_]['s'])>=2]
            kb9_=min(len(bn9_),int(round(len(Q_)*bq_)))
            if kb9_>0:
                Q_=Q_[:len(Q_)-kb9_]+list(rs_.choice(bn9_,kb9_,replace=False)); model.__dict__['_benchcue']=(kb9_,len(bn9_))
            else: model.__dict__['_benchcue']=(0,len(bn9_))
        except Exception as e9_: _fail('inference.mixing:66'); print(f"  [inference MIXING] benchmark cues unavailable ({type(e9_).__name__}: {e9_})"[:200],flush=True)
    kr_=int(getattr(cfg,'inf_mix_kr',2)); kn_=int(getattr(cfg,'inf_mix_kn',2))
    P_=[]; CX_=[]
    T_=lambda i_: it_[i_]['s'] if i_>=0 else CX_[-i_-1][0]; D_=lambda i_: it_[i_]['d'] if i_>=0 else -1
    for q_ in Q_: P_+=[(q_,int(o_),'random') for o_ in rs_.choice(old_,kr_,replace=False)]
    if kn_>0:                                                                   # the trunk's own neighbours (mean hidden state), among a sample of stored sentences
        try:
            C_=list(rs_.choice(old_,min(1500,len(old_)),replace=False))
            ext8_=[]
            if int(getattr(cfg,'inf_mix_samevisit',0)):                         # cg26h SAME-VISIT CANDIDATES (for the time weighting): this visit's own sentences, other than the
                sq8_=set(Q_); ext8_=[i_ for i_ in new_ if i_ not in sq8_]       #   cues, may also be partners (before, partners came only from earlier visits); the time
                C_=C_+list(rs_.choice(ext8_,min(500,len(ext8_)),replace=False)) if ext8_ else C_   #   weighting gives nothing within inf_mix_time_near sentences (same document);
                                                                                #   random pairs and the chance bar still use earlier visits only
            if int(getattr(cfg,'inf_mix_sharename',0)) and int(getattr(cfg,'inf_mix_nameidx',0)):   # cg26 NAME-INDEXED POOL: candidates that share a rare name with some cue
                try:                                                            #   come first (random candidates almost never do: cg25 GPU, 1 of 32 cues found one in 1500)
                    dfq9_=S_.get('df',{}); dm9_=int(cfg.inf_df)
                    def nms9_(t_): return {w_.group(0) for w_ in _UWORD.finditer(t_) if len(w_.group(0))>=4 and w_.group(0)[:1].isupper() and w_.group(0)[1:].isalpha() and w_.group(0)[1:].islower() and 0<dfq9_.get(w_.group(0),0)<=dm9_ and dfq9_.get(w_.group(0).lower(),0)<=dm9_}
                    qn9_=set().union(*[nms9_(it_[i_]['s']) for i_ in Q_]) if Q_ else set(); hit9_=[]
                    if qn9_:
                        for i_ in old_+ext8_:
                            if any(w_.group(0) in qn9_ for w_ in _UWORD.finditer(it_[i_]['s'])): hit9_.append(i_)
                    hit9_=hit9_[-1000:]; sh9_=set(hit9_); C_=hit9_+[i_ for i_ in C_ if i_ not in sh9_][:max(0,1500-len(hit9_))]
                    model.__dict__['_nameidx']=(len(qn9_),len(hit9_))
                except Exception as e9_: _fail('inference.mixing:89'); print(f"  [inference MIXING] name index unavailable ({type(e9_).__name__}: {e9_})"[:200],flush=True)
            dev_=model.embed.weight.device; L0_=int(cfg.seq_len)
            def emb_(ids_):
                E_=[]
                for o_ in range(0,len(ids_),256):
                    X_=torch.full((len(ids_[o_:o_+256]),L0_),32,dtype=torch.long,device=dev_)
                    for j_,i_ in enumerate(ids_[o_:o_+256]): b_=it_[i_]['s'][:L0_]; X_[j_,:len(b_)]=torch.tensor(list(b_),device=dev_)
                    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); was_=model.training; model.eval()
                    try: h_=model.features(X_)
                    finally: _carry_load(sv4_); model.train(was_)
                    h_=h_[0] if isinstance(h_,(tuple,list)) else h_; E_.append(F.normalize(h_.float().mean(1),dim=-1))
                return torch.cat(E_)
            SC_=None
            if int(getattr(cfg,'inf_mix_spectral',0)):
                try: SC_=_spec_codes(model,cfg,S_)
                except Exception as e9_: _fail('inference.mixing:104'); print(f"  [inference MIXING] spectral codes failed ({type(e9_).__name__}: {e9_})"[:200],flush=True)
            if SC_ is not None: model.__dict__['_SCcur']=SC_
            if SC_ is not None:                                                 # SPECTRAL PROPOSER (closed-form JEPA on sentence co-occurrence): partners = passages whose
                sim_=_spec_pass(SC_,[it_[i_]['s'] for i_ in Q_])@_spec_pass(SC_,[it_[i_]['s'] for i_ in C_]).T; tag9_='spectral'   #   word codes point the same way
                if int(getattr(cfg,'inf_mix_hybrid',0)):                        # cg22 HYBRID: + TF-IDF cosine (shared rare words), both standardised (offline Re-DocRED: chain
                    tf_=_tfidf_sim(S_,[it_[i_]['s'] for i_ in Q_],[it_[i_]['s'] for i_ in C_],int(cfg.inf_df),bool(int(getattr(cfg,'inf_mix_tfidf_names',0)))).to(sim_.device)   #   premises 405-411 / 6000 vs spectral 376-377,
                    z_=lambda m_: (m_-m_.mean())/(m_.std()+1e-6); sim_=z_(tf_)+float(getattr(cfg,'inf_mix_hybrid_w',0.5))*z_(sim_); tag9_='hybrid'   #   TF-IDF 339-359, random 2-3)
                model.__dict__['_specinfo']=(len(SC_[0]),SC_[1].shape[1],SC_[2],SC_[3])
            elif int(getattr(cfg,'inf_mix_latent',0)) and model.__dict__.get('_jepaLin') is not None:   # LATENT PROPOSER, closed-form linear map (whitened ridge, both ways)
                mu8_,Wh8_,M8_=model.__dict__['_jepaLin']; w8_=lambda z_: (z_-mu8_)@Wh8_
                Zq_=_jepa_emb(model,[it_[i_]['s'] for i_ in Q_]); Zc_=_jepa_emb(model,[it_[i_]['s'] for i_ in C_])
                sim_=F.normalize(w8_(Zq_)@M8_,dim=-1)@F.normalize(w8_(Zc_),dim=-1).T; tag9_='latent-linear'
            elif int(getattr(cfg,'inf_mix_latent',0)) and model.__dict__.get('_jepaP') is not None:   # LATENT PROPOSER (attention + JEPA): partners = nearest (whitened) codes
                Zq_=_jepa_emb(model,[it_[i_]['s'] for i_ in Q_]); Zc_=_jepa_emb(model,[it_[i_]['s'] for i_ in C_]); P9_=model.__dict__['_jepaP']   #   to the PREDICTOR's
                sim_=F.normalize(P9_(Zq_),dim=-1)@F.normalize(Zc_,dim=-1).T; tag9_='latent'   #   output - the passage the model expects to be linked (training metric)
            else: Eq_=emb_(Q_); Ec_=emb_(C_); sim_=Eq_@Ec_.T; tag9_='neighbour'
            if int(getattr(cfg,'inf_mix_sharename',0)) and tag9_ in ('hybrid','spectral'):   # cg25 SHARED NAME: a candidate must share a rare name word with the cue (cg24 GPU: the
                dfq_=S_.get('df',{}); dm_=int(cfg.inf_df)                           #   hybrid paired sentences of the same GENRE - two biographies - not about the same thing)
                def nms_(t_): return {w_.group(0) for w_ in _UWORD.finditer(t_) if len(w_.group(0))>=4 and w_.group(0)[:1].isupper() and w_.group(0)[1:].isalpha() and w_.group(0)[1:].islower() and 0<dfq_.get(w_.group(0),0)<=dm_ and dfq_.get(w_.group(0).lower(),0)<=dm_}
                Nq_=[nms_(it_[i_]['s']) for i_ in Q_]; Nc_=[nms_(it_[i_]['s']) for i_ in C_]
                Mk_=torch.tensor([[bool(a8_&b8_) for b8_ in Nc_] for a8_ in Nq_],dtype=torch.bool,device=sim_.device)
                sim_=sim_.masked_fill(~Mk_,float('-inf')); model.__dict__['_nameinfo']=(int(Mk_.any(1).sum()),len(Q_),float(Mk_.float().sum(1).mean()))
                print(f"  [inference MIXING | SHARED NAME] {int(Mk_.any(1).sum())} of {len(Q_)} cues have a candidate sharing a rare name (mean {float(Mk_.float().sum(1).mean()):.1f} of {len(C_)} candidates per cue)"+(f" | name index: {model.__dict__['_nameidx'][0]} cue names, {model.__dict__['_nameidx'][1]} stored sentences share one" if model.__dict__.get('_nameidx') else ""),flush=True)
            for a_,q_ in enumerate(Q_):
                for j_ in sim_[a_].topk(kn_).indices.tolist():
                    if float(sim_[a_,j_])>float('-inf'): P_.append((q_,int(C_[j_]),tag9_))
            if int(getattr(cfg,'inf_mix_latent_score',0)) and model.__dict__.get('_jepaLin') is not None and tag9_!='latent-linear':   # cg22: the latent (closed-form JEPA)
                try:                                                            #   proposer goes through the same test as a second source (scored; dreamed only if allowed)
                    mu8_,Wh8_,M8_=model.__dict__['_jepaLin']; w8_=lambda z_: (z_-mu8_)@Wh8_
                    with torch.no_grad():
                        Zq_=_jepa_emb(model,[it_[i_]['s'] for i_ in Q_]); Zc_=_jepa_emb(model,[it_[i_]['s'] for i_ in C_])
                        sl8_=F.normalize(w8_(Zq_)@M8_,dim=-1)@F.normalize(w8_(Zc_),dim=-1).T
                    for a_,q_ in enumerate(Q_):
                        for j_ in sl8_[a_].topk(kn_).indices.tolist(): P_.append((q_,int(C_[j_]),'latent-linear'))
                except Exception as e8_: _fail('inference.mixing:138'); print(f"  [inference MIXING] latent proposer unavailable ({type(e8_).__name__}: {e8_})"[:200],flush=True)
            if int(getattr(cfg,'inf_jd',0)):                                    # cg26 JEPA PROPOSER (NREM): partners = stored sentences whose target code the trained JEPA
                try:                                                            #   predictor expects from the cue
                    sj_=_jd_scores(model,cfg,[it_[i_]['s'] for i_ in Q_],[it_[i_]['s'] for i_ in C_])
                    if sj_ is not None:
                        for a_,q_ in enumerate(Q_):
                            for j_ in sj_[a_].topk(kn_).indices.tolist(): P_.append((q_,int(C_[j_]),'jepa'))
                        if int(getattr(cfg,'inf_mix_jdname',0)):                # cg26b JEPA WITHIN THE NAME INDEX: the JEPA ranks only candidates sharing a rare name with the cue
                            dfq7_=S_.get('df',{}); dm7_=int(cfg.inf_df)          #   (name index brings recall, the JEPA judges which partner fits)
                            def nms7_(t_): return {w_.group(0) for w_ in _UWORD.finditer(t_) if len(w_.group(0))>=4 and w_.group(0)[:1].isupper() and w_.group(0)[1:].isalpha() and w_.group(0)[1:].islower() and 0<dfq7_.get(w_.group(0),0)<=dm7_ and dfq7_.get(w_.group(0).lower(),0)<=dm7_}
                            Nq7_=[nms7_(it_[i_]['s']) for i_ in Q_]; Nc7_=[nms7_(it_[i_]['s']) for i_ in C_]
                            Mk7_=torch.tensor([[bool(a7_&b7_) for b7_ in Nc7_] for a7_ in Nq7_],dtype=torch.bool,device=sj_.device)
                            sjn_=sj_.masked_fill(~Mk7_,float('-inf')); nh7_=0
                            bt7_=float(getattr(cfg,'inf_mix_time_w',0.0))
                            if bt7_>0:                                          # cg26h TIME WEIGHTING: score + w x time weight (traces that arrived closer together rank higher)
                                Tw7_=torch.tensor(_time_w(model,cfg,it_,list(Q_),list(C_)),device=sjn_.device,dtype=sjn_.dtype); k7_=min(kn_,len(C_))
                                t0_=sjn_.topk(k7_,dim=1).indices; sjn_=sjn_+bt7_*Tw7_; t1_=sjn_.topk(k7_,dim=1).indices; ok7_=torch.isfinite(sjn_.gather(1,t1_))
                                tw0_=Tw7_.gather(1,t0_)[torch.isfinite(sj_.masked_fill(~Mk7_,float('-inf')).gather(1,t0_))]; tw1_=Tw7_.gather(1,t1_)[ok7_]
                                ch7_=sum(len(set(t1_[a_].tolist())-set(t0_[a_].tolist())) for a_ in range(len(Q_)) if bool(Mk7_[a_].any()))
                                sv7_=lambda i0_: float(np.mean([it_[int(C_[j_])].get('v')==it_[int(Q_[a_])].get('v') for a_ in range(len(Q_)) for j_,o_ in zip(i0_[a_].tolist(),torch.isfinite(sjn_.gather(1,i0_))[a_].tolist()) if o_] or [float('nan')]))
                                print(f"  [inference MIXING | TIME WEIGHTING] jepa-name (NREM) weight {bt7_:g}"+(" - SHUFFLED TIMESTAMPS (control)" if int(getattr(cfg,'inf_mix_time_shuffle',0)) else "")+
                                      f": {ch7_} of {int(ok7_.sum())} chosen partners changed | mean time weight of the chosen {float(tw1_.mean()) if tw1_.numel() else float('nan'):.3f} (without weighting {float(tw0_.mean()) if tw0_.numel() else float('nan'):.3f}), "
                                      f"same visit as the cue {sv7_(t1_):.2f} (without {sv7_(t0_):.2f})",flush=True)
                            for a_,q_ in enumerate(Q_):
                                for j_ in sjn_[a_].topk(min(kn_,len(C_))).indices.tolist():
                                    if float(sjn_[a_,j_])>float('-inf'): P_.append((q_,int(C_[j_]),'jepa-name')); nh7_+=1
                            model.__dict__['_jdname']=(int(Mk7_.any(1).sum()),len(Q_),nh7_)
                            if int(getattr(cfg,'inf_mix_jdwide',0)):            # cg26j JEPA-WIDE: the JEPA chooses among the whole candidate sample (about 1500), a shared rare
                                bw_=float(getattr(cfg,'inf_mix_jdwide_bonus',0.3)); kw_=int(getattr(cfg,'inf_mix_jdwide_k',1))   #   name adds a bonus instead of being required, so
                                sw_=sj_+bw_*Mk7_.to(sj_.dtype); nw_=0; nwn_=0                   #   the JEPA has a real choice (jepa-name had about 2.4 candidates per cue)
                                for a_,q_ in enumerate(Q_):
                                    for j_ in sw_[a_].topk(min(kw_,len(C_))).indices.tolist(): P_.append((q_,int(C_[j_]),'jepa-wide')); nw_+=1; nwn_+=int(bool(Mk7_[a_,j_]))
                                print(f"  [inference MIXING | JEPA-WIDE] {nw_} partners chosen by the JEPA among {len(C_)} candidates per cue (name bonus {bw_:g}); {nwn_} of them share a rare name with the cue",flush=True)
                except Exception as e8_: _fail('inference.mixing:171'); print(f"  [inference MIXING] JEPA proposer unavailable ({type(e8_).__name__}: {e8_})"[:200],flush=True)
            if int(getattr(cfg,'inf_mix_seq',0)):                                # cg26g SEQUENCE REPLAY (first step): the cue with the sentence stored right after it (else right
                okq_=lambda j_: 0<=j_<len(it_) and not it_[j_].get('gone') and not it_[j_].get('sup') and it_[j_].get('k') in ('real','ver') and len(it_[j_]['s'])>=ml9_   #   before) in
                nsq_=0                                                          #   the same visit and domain - episodes in order, judged by the same test
                for q_ in Q_:
                    for j_ in (q_+1,q_-1):
                        if okq_(j_) and it_[j_].get('v')==it_[q_].get('v') and it_[j_]['d']==it_[q_]['d'] and it_[j_]['s'].strip()!=it_[q_]['s'].strip():
                            P_.append((q_,j_,'sequence')); nsq_+=1; break
                model.__dict__['_seqn']=nsq_
        except Exception as e_: _fail('inference.mixing:180'); print(f"  [inference MIXING] neighbour proposer unavailable ({type(e_).__name__}: {e_}) - random mixing only",flush=True)
    kx_=int(getattr(cfg,'inf_mix_cortex',0))
    try: _cx_ok=kx_>0 and isinstance(model.__dict__.get('_bank',{}),dict)
    except Exception: _cx_ok=False
    if _cx_ok:                                                                   # CUED CORTICAL FRAGMENTS (REM-like): a recent store sentence (the hippocampal cue) chooses
      try:
          BK9_=model.__dict__.get('_bank',{})                                     #   fragments the TRUNK generated for OLDER domains (its dream bank = what the cortex can
          cxs_=str(getattr(cfg,'inf_mix_cxsrc','bank'))
          if cxs_ in ('familiar','both'):                                       #   ('familiar': REMOTE stored sentences - from visits at least 2 back - that the trunk now
              rm9_=[i_ for i_ in old_ if it_[i_].get('v',-9)<=vn_-2]               #   predicts WELL, i.e. old material the cortex has absorbed; local dream-bank generations
              rm9_=[rm9_[int(i_)] for i_ in rs_.choice(len(rm9_),min(600,len(rm9_)),replace=False)] if rm9_ else []   #   were incoherent at this model size)
              if rm9_:
                  t2s9_=model.__dict__.get('_t2s',{})
                  def bps9_(ids_):                                              # FIX (review): each remote sentence scored through ITS OWN slot (was the current one,
                      out_=np.full(len(ids_),9.0); g_={}                        #   so 'familiar' favoured the current domain)
                      for k_,i_ in enumerate(ids_): g_.setdefault(int(t2s9_.get(it_[i_]['d'],it_[i_]['d'])),[]).append(k_)
                      for sl_,ks_ in g_.items():
                          bb_=_inf_cond_bytes(model,cfg,[(b'',it_[ids_[k_]]['s']) for k_ in ks_],sl_)
                          for k_,x_ in zip(ks_,bb_): out_[k_]=float(x_[1].mean()) if len(x_[1]) else 9.0
                      return out_
                  bp9_=bps9_(rm9_)
                  for i_ in [rm9_[j_] for j_ in np.argsort(bp9_)[:max(1,len(rm9_)//2)]]: CX_.append((it_[i_]['s'],int(it_[i_]['d']),'fam',int(i_)))
                  if int(getattr(cfg,'inf_mix_cxname',0)):                      # cg26c NAME-INDEXED REM TRACES: remote sentences that share a rare name with some cue, kept
                      try:                                                      #   if the trunk predicts them at least as well as the median familiar sample (familiar =
                          dfq6_=S_.get('df',{}); dm6_=int(cfg.inf_df)           #   absorbed by the cortex); cg26b GPU: REM traces chosen from a random familiar sample were
                          nms6_=lambda t_: {w_.group(0) for w_ in _UWORD.finditer(t_) if len(w_.group(0))>=4 and w_.group(0)[:1].isupper() and w_.group(0)[1:].isalpha() and w_.group(0)[1:].islower() and 0<dfq6_.get(w_.group(0),0)<=dm6_ and dfq6_.get(w_.group(0).lower(),0)<=dm6_}
                          qn6_=set().union(*[nms6_(it_[i_]['s']) for i_ in Q_]) if Q_ else set(); have6_={c_[0] for c_ in CX_}
                          hit6_=[i_ for i_ in old_ if it_[i_].get('v',-9)<=vn_-2 and it_[i_]['s'] not in have6_ and nms6_(it_[i_]['s'])&qn6_][-int(getattr(cfg,'inf_mix_cxname_max',300)):]
                          nk6_=0
                          if hit6_:
                              bp6_=bps9_(hit6_)
                              thr6_=float(np.median(bp9_))
                              for i_,b6_ in zip(hit6_,bp6_):
                                  if b6_<=thr6_: CX_.append((it_[i_]['s'],int(it_[i_]['d']),'fam',int(i_))); nk6_+=1
                          model.__dict__['_cxname']=(len(qn6_),len(hit6_),nk6_)
                      except Exception as e6_: _fail('inference.mixing:215'); print(f"  [inference MIXING] name-indexed REM traces unavailable ({type(e6_).__name__}: {e6_})"[:200],flush=True)
          if cxs_ in ('bank','both'):                                           # (cg25 'both': the trunk's own generations AND the familiar remote sentences, tagged apart)
            for s9_,X9_ in BK9_.items():                                          #   produce about old material) - nearest by spectral word codes - and the pair is spliced
              if int(s9_)==int(ts): continue
              for r9_ in X9_[:int(getattr(cfg,'inf_mix_cxrows',96))]:
                  b9_=bytes([x_ for x_ in r9_.to('cpu',torch.uint8).tolist() if x_]).strip()
                  if len(b9_)>=24: CX_.append((b9_,int(s9_),'bank'))
          if CX_:
              SC9_=model.__dict__.get('_SCcur')
              if SC9_ is not None:
                  Zx9_=_spec_pass(SC9_,[c_[0] for c_ in CX_]); sim9_=_spec_pass(SC9_,[it_[i_]['s'] for i_ in Q_])@Zx9_.T
                  sel9_=str(getattr(cfg,'inf_mix_cxsel','top')); lo9_=float(getattr(cfg,'inf_mix_band_lo',0.80)); hi9_=float(getattr(cfg,'inf_mix_band_hi',0.97)); lam9_=float(getattr(cfg,'inf_mix_mmr',0.7))
                  for a_,q_ in enumerate(Q_):
                      row_=sim9_[a_]; K9_=min(kx_,len(CX_))
                      if sel9_ in ('top','all'):                                # NEAREST traces to the cue (may be near-duplicates: too close)
                          for j_ in row_.topk(K9_).indices.tolist(): P_.append((q_,-(int(j_)+1),'cortex-top'))
                      if sel9_ in ('band','mmr','all'):                         # NOT TOO CLOSE, NOT TOO FAR: traces whose similarity to the cue lies in a middle band
                          ql_,qh_=float(torch.quantile(row_,lo9_)),float(torch.quantile(row_,hi9_))   #   (above the lo quantile, below the hi quantile of
                          cand9_=[int(j_) for j_ in torch.nonzero((row_>=ql_)&(row_<=qh_)).flatten().tolist()]   #   this cue's similarities)
                          if sel9_ in ('band','all'):
                              for j_ in sorted(cand9_,key=lambda j_: -float(row_[j_]))[:K9_]: P_.append((q_,-(j_+1),'cortex-band'))
                          if sel9_ in ('mmr','all') and cand9_:                 # ... and MUTUALLY ORTHOGONAL (maximal marginal relevance): each next trace is relevant to the
                              ch9_=[]                                           #   cue but unlike the traces already chosen - each brings a different factor
                              while len(ch9_)<K9_ and len(ch9_)<len(cand9_):
                                  best9_=max((j_ for j_ in cand9_ if j_ not in ch9_),key=lambda j_: lam9_*float(row_[j_])-(1-lam9_)*max([float(Zx9_[j_]@Zx9_[c_]) for c_ in ch9_],default=0.0))
                                  ch9_.append(best9_)
                              for j_ in ch9_: P_.append((q_,-(j_+1),'cortex-mmr'))
              else:
                  for q_ in Q_:
                      for j_ in rs_.choice(len(CX_),min(kx_,len(CX_)),replace=False): P_.append((q_,-(int(j_)+1),'cortex'))
          if CX_ and int(getattr(cfg,'inf_mix_cxlatent',0)) and model.__dict__.get('_jepaLin') is not None:   # cg25 LATENT CORTEX MATCHER: cue -> trunk code -> closed-form JEPA
              try:                                                              #   predictor -> nearest traces (the trunk's own idea of what goes with the cue); scored
                  mu8_,Wh8_,M8_=model.__dict__['_jepaLin']; w8_=lambda z_: (z_-mu8_)@Wh8_   #   beside the spectral matcher, never dreamed unless allowed
                  with torch.no_grad():
                      Zq8_=_jepa_emb(model,[it_[i_]['s'] for i_ in Q_]); Zx8_=_jepa_emb(model,[c_[0] for c_ in CX_])
                      sl8_=F.normalize(w8_(Zq8_)@M8_,dim=-1)@F.normalize(w8_(Zx8_),dim=-1).T
                  for a_,q_ in enumerate(Q_):
                      for j_ in sl8_[a_].topk(min(kx_,len(CX_))).indices.tolist(): P_.append((q_,-(int(j_)+1),'cortex-latent'))
              except Exception as e8_: _fail('inference.mixing:253'); print(f"  [inference MIXING] latent cortex matcher unavailable ({type(e8_).__name__}: {e8_})"[:200],flush=True)
          if CX_ and int(getattr(cfg,'inf_jd',0)):                              # cg26 JEPA CORTEX MATCHER (REM): the trained JEPA chooses traces for the cue
              try:
                  sj8_=_jd_scores(model,cfg,[it_[i_]['s'] for i_ in Q_],[c_[0] for c_ in CX_])
                  if sj8_ is not None:
                      for a_,q_ in enumerate(Q_):
                          for j_ in sj8_[a_].topk(min(kx_,len(CX_))).indices.tolist(): P_.append((q_,-(int(j_)+1),'cortex-jepa'))
                      if int(getattr(cfg,'inf_mix_cxname',0)):                  # cg26c REM JEPA WITHIN THE NAME INDEX: the JEPA ranks only the traces that share a rare name with the cue
                          dfq5_=S_.get('df',{}); dm5_=int(cfg.inf_df)
                          nms5_=lambda t_: {w_.group(0) for w_ in _UWORD.finditer(t_) if len(w_.group(0))>=4 and w_.group(0)[:1].isupper() and w_.group(0)[1:].isalpha() and w_.group(0)[1:].islower() and 0<dfq5_.get(w_.group(0),0)<=dm5_ and dfq5_.get(w_.group(0).lower(),0)<=dm5_}
                          Nq5_=[nms5_(it_[i_]['s']) for i_ in Q_]; Nx5_=[nms5_(c_[0]) for c_ in CX_]
                          Mk5_=torch.tensor([[bool(a5_&b5_) for b5_ in Nx5_] for a5_ in Nq5_],dtype=torch.bool,device=sj8_.device)
                          sjn5_=sj8_.masked_fill(~Mk5_,float('-inf')); np5_=0
                          bt5_=float(getattr(cfg,'inf_mix_time_w',0.0))
                          if bt5_>0:                                            # cg26h TIME WEIGHTING (REM): familiar store traces by arrival time; dream-bank traces get 0
                              Tw5_=torch.tensor(_time_w(model,cfg,it_,list(Q_),[(c_[3] if len(c_)>3 else None) for c_ in CX_]),device=sjn5_.device,dtype=sjn5_.dtype); k5_=min(kx_,len(CX_))
                              t0_=sjn5_.topk(k5_,dim=1).indices; sjn5_=sjn5_+bt5_*Tw5_; t1_=sjn5_.topk(k5_,dim=1).indices; ok5_=torch.isfinite(sjn5_.gather(1,t1_))
                              ch5_=sum(len(set(t1_[a_].tolist())-set(t0_[a_].tolist())) for a_ in range(len(Q_)) if bool(Mk5_[a_].any()))
                              print(f"  [inference MIXING | TIME WEIGHTING] cortex-jepa-name (REM) weight {bt5_:g}: {ch5_} of {int(ok5_.sum())} chosen traces changed | mean time weight of the chosen "
                                    f"{float(Tw5_.gather(1,t1_)[ok5_].mean()) if bool(ok5_.any()) else float('nan'):.3f}",flush=True)
                          for a_,q_ in enumerate(Q_):
                              for j_ in sjn5_[a_].topk(min(kx_,len(CX_))).indices.tolist():
                                  if float(sjn5_[a_,j_])>float('-inf'): P_.append((q_,-(int(j_)+1),'cortex-jepa-name')); np5_+=1
                          model.__dict__['_cxjn']=(int(Mk5_.any(1).sum()),len(Q_),np5_)
              except Exception as e8_: _fail('inference.mixing:277'); print(f"  [inference MIXING] JEPA cortex matcher unavailable ({type(e8_).__name__}: {e8_})"[:200],flush=True)
          P_=[(a_,b_,(t_+'-fam') if (b_<0 and str(t_).startswith('cortex') and len(CX_[-b_-1])>2 and CX_[-b_-1][2]=='fam') else t_) for a_,b_,t_ in P_]
          if CX_ and int(getattr(cfg,'inf_mix_cxshow',0)):                       # cg25: what the cortical traces look like (coherent memories, or garbled generations?)
              ex8_=[]
              for src8_ in ('bank','fam'):
                  L8_=[c_ for c_ in CX_ if len(c_)>2 and c_[2]==src8_]
                  for c_ in [L8_[int(i_)] for i_ in rs_.choice(len(L8_),min(2,len(L8_)),replace=False)] if L8_ else []: ex8_.append(f"{src8_}: {c_[0][:90]!r}")
              print(f"  [inference MIXING | CORTICAL TRACES] {sum(1 for c_ in CX_ if len(c_)>2 and c_[2]=='bank')} dream-bank generations, {sum(1 for c_ in CX_ if len(c_)>2 and c_[2]=='fam')} familiar remote sentences | e.g. "+" || ".join(ex8_),flush=True)
          model.__dict__['_cxinfo']=(len(CX_),len(BK9_))
      except Exception as e9_: _fail('inference.mixing:286'); print(f"  [inference MIXING] cortex fragments unavailable ({type(e9_).__name__}: {e9_})"[:200],flush=True); P_=[p_ for p_ in P_ if p_[2]!='cortex']
    nd0_=len(P_); P_=[p_ for p_ in P_ if T_(p_[0]).strip()!=T_(p_[1]).strip()]   # cg26e: never pair a sentence with an identical copy (pass-1 text vs pass-2 replay)
    jq_=float(getattr(cfg,'inf_mix_neardup',0.0))
    if jq_>0:                                                                   # cg26g NEAR-DUPLICATES: pairs sharing more than jq_ of their character 4-grams are dropped
        def g4_(t_): t_=t_.lower(); return {t_[k_:k_+4] for k_ in range(max(1,len(t_)-3))}   #   (cg26e GPU, EuroParl revisit: repetitive replays, copying +15.4 bits/pair,
        G4_={}                                                                  #   bar 15.7 bits, 0/16 known links passed, 224 accepted)
        def gg_(i_):
            if i_ not in G4_: G4_[i_]=g4_(T_(i_))
            return G4_[i_]
        nk_=[]; nnd_=0
        for p_ in P_:
            if p_[2]=='control+': nk_.append(p_); continue
            a_,b_=gg_(p_[0]),gg_(p_[1])
            if a_ and b_ and len(a_&b_)/max(1,min(len(a_),len(b_)))>jq_: nnd_+=1; continue
            nk_.append(p_)
        P_=nk_; model.__dict__['_neardup']=nnd_
    if len(P_)<nd0_: model.__dict__['_dupdrop']=nd0_-len(P_)
    pc_=[c_ for c_ in S_.get('conc',[]) if isinstance(c_.get('src'),tuple) and not c_.get('freed')]   # POSITIVE CONTROL: the two premises of committed conclusions
    for c_ in [pc_[int(i_)] for i_ in rs_.choice(len(pc_),min(16,len(pc_)),replace=False)] if pc_ else []:   #   (known linked pairs) go through the SAME test;
        P_.append((int(c_['src'][0]),int(c_['src'][1]),'control+'))             #   never dreamed, only scored - the test must pass these far more often than random
    LPF_=None
    if float(getattr(cfg,'inf_latent_prefilter',0.0))>0:                        # 9 Nov LATENT FIRST PASS: the lowest-SFA-cosine proposals skip the byte judge
        try:
            from platypus.inference.latentjudge import latent_prefilter
            P_,AU9_,LPF_=latent_prefilter(model,cfg,S_,P_,T_); P_=P_+AU9_       #   (audit pairs: judged, never accepted)
        except Exception as e9_: _fail('inference.mixing:prefilter'); print(f"  [LATENT FIRST PASS] unavailable ({type(e9_).__name__}: {e9_})"[:200],flush=True)
    byd_={}
    for i_ in old_: byd_.setdefault(it_[i_]['d'],[]).append(i_)
    nb_=int(getattr(cfg,'inf_mix_nbase',3))
    def rnd_(d_): L9_=byd_.get(d_) or old_; return [int(L9_[rs_.randint(len(L9_))]) for _ in range(nb_)]
    RB_=[(rnd_(D_(a_)),rnd_(D_(b_))) for a_,b_,_ in P_]   # baselines: nb_ random sentences of A's domain (in A's place) and of B's domain (in B's place)
    rq_=[int(x_) for x_ in rs_.choice(old_,len(P_)*2)]
    jobs_=[]
    for k_,(a_,b_,_) in enumerate(P_):
        A_,B_=T_(a_),T_(b_); R1_,R2_=it_[rq_[2*k_]]['s'],it_[rq_[2*k_+1]]['s']
        if int(getattr(cfg,'inf_mix_att',0)): R1_,R2_=_blankw(A_),_blankw(B_)    # ATTRIBUTED BASELINE: the same passage with its words blanked (form kept, content gone)
        jobs_+=[(A_,B_),(R1_,B_),(B_,A_),(R2_,A_)]+[(it_[r_]['s'],B_) for r_ in RB_[k_][0]]+[(it_[r_]['s'],A_) for r_ in RB_[k_][1]]
    tj9_=__import__('time').time(); bb_=_inf_cond_bytes(model,cfg,jobs_,ts); tj9_=__import__('time').time()-tj9_; acc_=[]; thr_=float(getattr(cfg,'inf_mix_bits',2.0)); G_=[]
    st9_=4+2*nb_
    for k_,(a_,b_,src_) in enumerate(P_):
        (tB_,bAB_),(_,bRB_),(tA_,bBA_),(_,bRA_)=bb_[st9_*k_:st9_*k_+4]
        if not int(getattr(cfg,'inf_mix_att',0)):                               # mean over nb_ same-domain random contexts (per byte, aligned: same target bytes)
            bRB_=np.mean([bb_[st9_*k_+4+j_][1] for j_ in range(nb_)],0); bRA_=np.mean([bb_[st9_*k_+4+nb_+j_][1] for j_ in range(nb_)],0)
        A_,B_=T_(a_),T_(b_)
        kp9_=(lambda w_: 0<S_['df'].get(w_,0)<=int(cfg.inf_df)) if int(getattr(cfg,'inf_mix_rarecopy',0)) else None   # rare shared words count (entity identity)
        mB_=_mix_mask(A_,tB_,kp9_); mA_=_mix_mask(B_,tA_,kp9_)                  # (copying excluded: bytes of common words B shares with A, and vice versa)
        gB_=float((bRB_-bAB_)[mB_].sum()); gA_=float((bRA_-bBA_)[mA_].sum()); gr_=float((bRB_-bAB_).sum()+(bRA_-bBA_).sum())
        G_.append((a_,b_,src_,min(gA_,gB_),gr_,gA_,gB_))
    GA9_=[g_ for g_ in G_ if g_[2]=='audit']; G_=[g_ for g_ in G_ if g_[2]!='audit']   # audit pairs leave before anything else sees them
    if int(getattr(cfg,'inf_mix_null',1)):                                      # THRESHOLD SET BY CHANCE: a proposed pair must beat the 95th percentile of what RANDOM pairs
        nr9_=[g_[3] for g_ in G_ if g_[2]=='random']                            #   reach this sleep (both ways), not just a fixed number of bits
        if len(nr9_)>=10: thr_=max(thr_,float(np.quantile(nr9_,float(getattr(cfg,'inf_mix_null_q',0.95)))))   # cg22: quantile settable (GPU cg21c: chance-level junk accepted at 0.95)
    cpn_=[g_[3] for g_ in G_ if g_[2]=='control+']; rdn_=[g_[3] for g_ in G_ if g_[2]=='random']
    model.__dict__['_mixctrl']=(len(cpn_),sum(1 for x_ in cpn_ if x_>=thr_),float(np.median(cpn_)) if cpn_ else float('nan'),len(rdn_),sum(1 for x_ in rdn_ if x_>=thr_),float(np.median(rdn_)) if rdn_ else float('nan'),thr_)
    for a_,b_,src_,mg9_,gr_,gA_,gB_ in G_:
        if src_ not in ('random','control+') and mg9_>=thr_: acc_.append((a_,b_,src_,gA_,gB_))
    if LPF_ is not None:
        try:
            from platypus.inference.latentjudge import latent_prefilter_report
            latent_prefilter_report(model,LPF_,GA9_,thr_,len(acc_),tj9_,len(G_)+len(GA9_))
        except Exception as e9_: _fail('inference.mixing:prefilter_report'); print(f"  [LATENT FIRST PASS] report unavailable ({type(e9_).__name__}: {e9_})"[:200],flush=True)
    if int(getattr(cfg,'inf_jd_judge',0)):                                     # cg26j: the judge's outcome for every tested pair, kept for training the JEPA selector
        JG9_=S_.setdefault('jd_judged',[]); dfj_=S_.get('df',{}); dmj_=int(cfg.inf_df)
        nmj_=lambda t_: {w_.group(0) for w_ in _UWORD.finditer(t_) if len(w_.group(0))>=4 and w_.group(0)[:1].isupper() and w_.group(0)[1:].isalpha() and w_.group(0)[1:].islower() and 0<dfj_.get(w_.group(0),0)<=dmj_}
        for a_,b_,src_,mg9_,gr_,gA_,gB_ in G_:
            if src_=='control+' or a_<0: continue
            JG9_.append((T_(a_),T_(b_),bool(mg9_>=thr_ and src_!='random'),src_,bool(nmj_(T_(a_))&nmj_(T_(b_)))))   # (last: a shared rare name - a sign of a real link)
        del JG9_[:-6000]
    if int(getattr(cfg,'inf_latent_judge',0)):                                 # north-star rung 2 readout: a LATENT judge (slow-feature cosine) on the same pairs
        from platypus.inference.latentjudge import latent_judge_readout
        latent_judge_readout(model,cfg,S_,G_,T_,thr_,tj9_)
    G_=[g_[:5] for g_ in G_]
    S_.setdefault('mix',[]).extend([x_ for x_ in acc_ if x_[1]>=0]); S_.setdefault('mixall',[]).extend(G_)
    ds9_=str(getattr(cfg,'inf_mix_dream_src','') or '')
    if int(getattr(cfg,'inf_mix_dream_conc',0)):                                # HIGH-PURITY DREAMS: the premise pairs of committed conclusions (staging v2: ~85-97 pct real links)
        cp9_=[(a_,b_) for a_,b_,s9_,_,_ in G_ if s9_=='control+']               #   dreamed together - tests whether linked co-occurrence teaches STRUCTURE (two-hop from the weights)
        S_.setdefault('mixtxt',[]).extend((T_(a_),T_(b_),D_(a_),'premise') for a_,b_ in cp9_); del S_['mixtxt'][:-2000]
        model.__dict__['_ndreamsrc']=('premise',len(cp9_))
    elif int(getattr(cfg,'inf_mix_dream_rand',0)):                              # CONTROL: dream RANDOM (untested, unselected) pairs instead - same number as would be accepted
        rp9_=[(a_,b_) for a_,b_,s9_,_,_ in G_ if s9_=='random']; rs_.shuffle(rp9_)
        nr9_=len([1 for g_ in G_ if g_[2]=='control+']) if int(getattr(cfg,'inf_mix_dream_rand_match',0)) else max(1,len(acc_))   # (match the premise arm's count)
        S_.setdefault('mixtxt',[]).extend((T_(a_),T_(b_),D_(a_),'random-dream') for a_,b_ in rp9_[:nr9_]); del S_['mixtxt'][:-2000]
    else:
        nd9_=set(x_ for x_ in str(getattr(cfg,'inf_mix_nodream_src','') or '').split(',') if x_)   # cg22: sources scored but never dreamed (e.g. the latent proposer)
        dd9_=[(a_,b_,src_) for a_,b_,src_,_,_ in acc_ if (not ds9_ or src_ in ds9_.split(',')) and src_ not in nd9_]
        if int(getattr(cfg,'inf_mix_jdfinal',0)) and acc_ and model.__dict__.get('_jdS') is not None:   # cg26j JEPA FINAL SELECTION: every source's pairs that passed the judge (hybrid, sequence,
            try:                                                                #   cortex, latent, jepa ...) compete per cue; the JEPA ranks them and the top k per cue are
                sk9_=set(x_ for x_ in str(getattr(cfg,'inf_mix_jdfinal_skip','jepa,sequence') or '').split(',') if x_)|nd9_|{'random','control+'}   # cg26j: sequence pairs too - their judge gains are huge (neighbouring lines of one document), so the fair ranking would favour them; they keep their own capped path   #   dreamed - the JEPA, not a fixed source list, decides
                cf9_=[(a_,b_,src_) for a_,b_,src_,_,_ in acc_ if src_ not in sk9_]  #   (left out: controls, sources scored but never dreamed, and whole-store 'jepa' - no better
                dfF_=S_.get('df',{}); dmF_=int(cfg.inf_df); bF_=float(getattr(cfg,'inf_mix_jdfinal_bonus',0.3))   #   than random in cg26b; a shared rare name adds a bonus, since
                nmF_=lambda t_: {w_.group(0) for w_ in _UWORD.finditer(t_) if len(w_.group(0))>=4 and w_.group(0)[:1].isupper() and w_.group(0)[1:].isalpha() and w_.group(0)[1:].islower() and 0<dfF_.get(w_.group(0),0)<=dmF_}   #   the JEPA's own proposals score highest by construction)
                qs9_=sorted({a_ for a_,_,_ in cf9_}); bs9_=sorted({b_ for _,b_,_ in cf9_}); qi9_={a_:i_ for i_,a_ in enumerate(qs9_)}; bi9_={b_:j_ for j_,b_ in enumerate(bs9_)}
                Mf9_=_jd_scores(model,cfg,[T_(a_) for a_ in qs9_],[T_(b_) for b_ in bs9_])
                if Mf9_ is not None and cf9_:
                    kf9_=int(getattr(cfg,'inf_mix_jdfinal_k',2)); byq9_={}
                    mgF_={(a_,b_):min(gA_,gB_) for a_,b_,_,gA_,gB_ in acc_}; fair_=int(getattr(cfg,'inf_mix_jdfinal_fair',0))
                    for a_,b_,src_ in cf9_: byq9_.setdefault(a_,{}).setdefault(b_,(float(Mf9_[qi9_[a_],bi9_[b_]])+bF_*float(bool(nmF_(T_(a_))&nmF_(T_(b_)))),src_,float(mgF_.get((a_,b_),0.0))))
                    sel9_=[]
                    for a_,D9_ in byq9_.items():
                        it9_=list(D9_.items())
                        if fair_ and len(it9_)>1:                               # cg26k FAIR RANKING: the JEPA's rank and the judge-gain rank are averaged, so a source the
                            rj_={b_:r_ for r_,(b_,_) in enumerate(sorted(it9_,key=lambda kv_: -kv_[1][0]))}   #   JEPA itself proposed (its own top pick) does not win by
                            rg_={b_:r_ for r_,(b_,_) in enumerate(sorted(it9_,key=lambda kv_: -kv_[1][2]))}   #   construction; ties go to the JEPA score
                            it9_=sorted(it9_,key=lambda kv_: (rj_[kv_[0]]+rg_[kv_[0]],-kv_[1][0]))
                        else: it9_=sorted(it9_,key=lambda kv_: -kv_[1][0])
                        sel9_+=[(a_,b_,v_[1]) for b_,v_ in it9_[:kf9_]]
                    from collections import Counter as _Cn9
                    print(f"  [inference MIXING | JEPA FINAL SELECTION] {len(cf9_)} accepted pairs from {len(set(s_ for _,_,s_ in cf9_))} sources over {len(byq9_)} cues -> {len(sel9_)} dreamed (top {kf9_} per cue by {'JEPA rank and judge-gain rank averaged' if fair_ else 'JEPA score'}) | "
                          f"accepted by source: {dict(_Cn9(s_ for _,_,s_ in cf9_))} | dreamed by source: {dict(_Cn9(s_ for _,_,s_ in sel9_))}",flush=True)
                    dd9_=sel9_+([z_ for z_ in dd9_ if z_[2]=='sequence'] if 'sequence' in sk9_ else [])   # sequence pairs keep their own path (and cap)
            except Exception as e9_: _fail('inference.mixing:381'); print(f"  [inference MIXING | JEPA FINAL SELECTION] unavailable ({type(e9_).__name__}: {e9_})"[:200],flush=True)
        qc9_=float(getattr(cfg,'inf_mix_seq_cap',0.0))
        if qc9_>0:                                                              # cg26h SEQUENCE CAP: neighbouring sentences pass the test easily (cg26h GPU visit 1: 17 of 31
            sq9_=[z_ for z_ in dd9_ if z_[2]=='sequence']; ot9_=[z_ for z_ in dd9_ if z_[2]!='sequence']   #   accepted, 17 of 19 dreams) - at most qc9_ x the other dreams (min 2)
            kq9_=max(2,int(qc9_*len(ot9_)))
            if len(sq9_)>kq9_: rs_.shuffle(sq9_); print(f"  [inference MIXING | SEQUENCE CAP] {len(sq9_)} accepted sequence pairs, {kq9_} dreamed (at most {qc9_:.0%} of the {len(ot9_)} other dreams, min 2)",flush=True); sq9_=sq9_[:kq9_]
            dd9_=ot9_+sq9_
        S_.setdefault('mixtxt',[]).extend((T_(a_),T_(b_),D_(a_),src_) for a_,b_,src_ in dd9_); del S_['mixtxt'][:-2000]   # texts of accepted pairs (cortex fragments too)
        if int(getattr(cfg,'inf_jd_outcome',0)) and dd9_:                       # cg26k OUTCOME RECORD: each dreamed pair's both-ways gain now, with fixed baseline sentences
            try:                                                                #   (same-domain, as the judge picks them); measured again at a later sleep by _jd_train
                vis8_=int(model.__dict__.get('_visit',0)); rc8_=[]
                for a_,b_,src_ in dd9_[:int(getattr(cfg,'inf_jd_out_n',48))]:
                    rr8_=np.random.RandomState(__import__('zlib').crc32(T_(a_)[:80]+b'|'+T_(b_)[:80])%2**31)
                    pk8_=lambda d_: [it_[int(L8_[rr8_.randint(len(L8_))])]['s'] for L8_ in [byd_.get(d_) or old_] for _ in range(3)]
                    rc8_.append({'a':T_(a_),'b':T_(b_),'rb':pk8_(D_(a_)),'ra':pk8_(D_(b_)),'sl':int(ts),'v':vis8_,'src':src_})
                g08_=_jd_pair_gain(model,cfg,S_,rc8_)
                for r8_,g8_ in zip(rc8_,g08_): r8_['g0']=g8_
                S_.setdefault('jd_pend',[]).extend(rc8_); del S_['jd_pend'][:-600]
            except Exception as e8_: _fail('inference.mixing:399'); print(f"  [inference MIXING | OUTCOME RECORD] unavailable ({type(e8_).__name__}: {e8_})"[:200],flush=True)
    # ---- readout: are the accepted pairs related? (benchmark gold and planted chains), against every pair tested
    def rel_(a_,b_):
        sa_,sb_=T_(a_),T_(b_); out_=None
        if str(getattr(cfg,'inf_bench','') or ''):
            G9_=_bench(cfg); t2e_=G9_['_t2e']; ea_=set(); eb_=set()
            for w_ in _UWORD.finditer(sa_): ea_|=t2e_.get(w_.group(0).decode('utf-8','ignore'),set())
            for w_ in _UWORD.finditer(sb_): eb_|=t2e_.get(w_.group(0).decode('utf-8','ignore'),set())
            if ea_ and eb_:
                out_='unrelated'
                for x_ in ea_:
                    for y_ in eb_:
                        if x_==y_: out_='same entity' if out_=='unrelated' else out_; continue
                        k9_=(min(x_,y_),max(x_,y_))
                        if k9_ in G9_['_spec']: return 'TWO-HOP CHAIN'
                        if k9_ in G9_['_rel']: out_='DIRECT FACT'
        ch_={}
        for c_ in inf_probes(cfg):
            for n_ in c_['names']: ch_[n_.encode('latin-1')]=id(c_)
        ca_={ch_[w_.group(0)] for w_ in _UWORD.finditer(sa_) if w_.group(0) in ch_}; cb_={ch_[w_.group(0)] for w_ in _UWORD.finditer(sb_) if w_.group(0) in ch_}
        if ca_ and cb_: return 'PLANTED SAME CHAIN' if ca_&cb_ else 'planted other chain'
        return out_ or 'outside'
    import collections as _co
    na_=_co.Counter(rel_(a_,b_) for a_,b_,_,_,_ in acc_); nt_=_co.Counter(rel_(a_,b_) for a_,b_,_,_,_ in G_)
    ns_=_co.Counter(s9_ for _,_,s9_,_,_ in acc_); nts_=_co.Counter(s9_ for _,_,s9_,_,_ in G_)
    gm_=[g_[3] for g_ in G_]; cp_=float(np.mean([g_[4]-max(0.0,g_[3]) for g_ in G_])) if G_ else 0.0
    fm_=lambda c_: ', '.join(f"{k_} {v_}" for k_,v_ in sorted(c_.items(),key=lambda z_: -z_[1]))
    rs9_={}; ex9_=[]; sm9_=[]; tr9_={}
    for a_,b_,s9_,mg9_,_ in G_:
        l9_=rel_(a_,b_); e9_=rs9_.setdefault(s9_,[0,0,0]); e9_[0]+=1; e9_[1]+=int(l9_ in ('TWO-HOP CHAIN','DIRECT FACT','PLANTED SAME CHAIN'))
        t9_=tr9_.setdefault(s9_,{}); t9_[l9_]=t9_.get(l9_,0)+1
        if l9_ in ('TWO-HOP CHAIN','DIRECT FACT','PLANTED SAME CHAIN'): ex9_.append((s9_,l9_,mg9_,a_,b_))
        elif s9_!='random' and len(sm9_)<2 and rs_.rand()<0.05: sm9_.append((s9_,l9_,mg9_,a_,b_))
    for a_,b_,s9_,_,_ in acc_:
        l9_=rel_(a_,b_); rs9_.setdefault(s9_,[0,0,0])[2]+=int(l9_ in ('TWO-HOP CHAIN','DIRECT FACT','PLANTED SAME CHAIN'))
    si9_=model.__dict__.get('_specinfo')
    mc9_=model.__dict__.get('_mixctrl')
    bc9_=model.__dict__.pop('_benchcue',None); jn9_=model.__dict__.pop('_jdname',None); cn9_=model.__dict__.pop('_cxname',None); cj9_=model.__dict__.pop('_cxjn',None)
    if bc9_ or jn9_: print("  [inference MIXING | CG26B]"+(f" benchmark cues {bc9_[0]} (of {bc9_[1]} eligible sentences with 2+ gold entities)" if bc9_ else "")
                           +(f" | jepa-name: {jn9_[0]} of {jn9_[1]} cues have a name-sharing candidate, {jn9_[2]} pairs proposed" if jn9_ else ""),flush=True)
    sq9_=model.__dict__.pop('_seqn',None)
    if sq9_ is not None: print(f"  [inference MIXING | SEQUENCE] {sq9_} cues paired with the sentence stored next to them (same visit and domain)",flush=True)
    ndp9_=model.__dict__.pop('_neardup',None)
    if ndp9_: print(f"  [inference MIXING | NEAR-DUPLICATES] {ndp9_} proposed pairs dropped (more than {float(getattr(cfg,'inf_mix_neardup',0.0)):.0%} of character 4-grams shared)",flush=True)
    if cn9_ or cj9_: print("  [inference MIXING | REM NAME INDEX]"+(f" {cn9_[0]} cue names, {cn9_[1]} remote sentences share one, {cn9_[2]} familiar enough to join the cortical traces" if cn9_ else "")
                           +(f" | cortex-jepa-name: {cj9_[0]} of {cj9_[1]} cues have a name-sharing trace, {cj9_[2]} pairs proposed" if cj9_ else ""),flush=True)
    if mc9_ and mc9_[0]: print(f"  [inference MIXING | TEST CHECK] bar {mc9_[6]:.1f} bits both ways | KNOWN LINKED PAIRS (premises of committed conclusions): {mc9_[1]}/{mc9_[0]} pass, weaker-side gain median {mc9_[2]:+.1f} bits | random pairs: {mc9_[4]}/{mc9_[3]} pass, median {mc9_[5]:+.1f} bits",flush=True)
    print("  [inference MIXING | TIERS BY SOURCE] "+" | ".join(f"{k_} (n {sum(v_.values())}): "+', '.join(f"{l_} {v_.get(l_,0)}" for l_ in ('TWO-HOP CHAIN','DIRECT FACT','same entity','PLANTED SAME CHAIN','planted other chain','unrelated','outside') if v_.get(l_,0)) for k_,v_ in tr9_.items()),flush=True)
    print(f"  [inference MIXING | RELATED BY SOURCE] "+" | ".join(f"{k_}: {v_[1]}/{v_[0]} proposed pairs related ({v_[1]/max(1,v_[0]):.3f}), {v_[2]} related accepted" for k_,v_ in rs9_.items())
          +(f" | spectral codes: {si9_[0]} words, {si9_[1]} dims, top kept eigenvalue {si9_[2]:.3f}, {si9_[3]} disconnected-piece eigenvectors dropped" if si9_ else "")
          +(f" | cortex fragments (old domains' dream-bank windows): {model.__dict__['_cxinfo'][0]} from {model.__dict__['_cxinfo'][1]} slots" if model.__dict__.get('_cxinfo') else ""),flush=True)
    def nm9_(a_,b_):                                                            # the gold entities behind a related pair (benchmark) or the planted names
        try:
            G9_=_bench(cfg); t2e_=G9_['_t2e']; ea_=set(); eb_=set()
            for w_ in _UWORD.finditer(T_(a_)): ea_|=t2e_.get(w_.group(0).decode('utf-8','ignore'),set())
            for w_ in _UWORD.finditer(T_(b_)): eb_|=t2e_.get(w_.group(0).decode('utf-8','ignore'),set())
            hit_=[(x_,y_) for x_ in ea_ for y_ in eb_ if (min(x_,y_),max(x_,y_)) in G9_['_rel'] or (min(x_,y_),max(x_,y_)) in G9_['_spec']]
            return '; '.join(f"{G9_['ent_name'][str(x_)]} / {G9_['ent_name'][str(y_)]}" for x_,y_ in hit_[:2])
        except Exception: return ''
    for s9_,l9_,mg9_,a_,b_ in ex9_[:4]+sm9_:
        print(f"    MIX PROPOSAL ({s9_}, {l9_}, weaker side {mg9_:+.1f} bits{', gold: '+nm9_(a_,b_) if l9_ in ('TWO-HOP CHAIN','DIRECT FACT') else ''}): {T_(a_)[:80]!r} <> {T_(b_)[:80]!r}",flush=True)
    print(f"  [inference MIXING] {len(G_)} spliced pairs tested ({fm_(nts_)}), ACCEPTED {len(acc_)} ({fm_(ns_)}) at >= {thr_:.0f} bits both ways (copying excluded) | "
          f"weaker-side gain median {float(np.median(gm_)) if gm_ else float('nan'):.1f} bits; copying adds {cp_:.1f} bits per pair on average | accepted: {fm_(na_)} | all tested: {fm_(nt_)}",flush=True)
    for a_,b_,src_,gA_,gB_ in sorted(acc_,key=lambda z_: -min(z_[3],z_[4]))[:3]:
        print(f"    MIX ({src_}, {gA_:.1f}/{gB_:.1f} bits, {rel_(a_,b_)}): {T_(a_)[:70]!r} <> {T_(b_)[:70]!r}",flush=True)

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_jd_pair_gain': 'platypus.inference.jepa',
    '_jd_scores': 'platypus.inference.jepa',
    '_jepa_emb': 'platypus.inference.jepa',
    '_tfidf_sim': 'platypus.inference.jepa',
    '_time_w': 'platypus.inference.jepa',
}
