"""Review fixes for cg26p_jepa, all behind switches (defaults unchanged).
python3 clean_patch.py IN.py OUT.py
- inf_clean 1: HELD-OUT probes (two-hop, typed two-hop, three-hop) in wordings the system never writes, the bridge(s) among the
  candidates, scored with no slot given (a mixture over every slot in use); schema fit (GIST tbits) of a conclusion measured
  with a different prompt from the one it is tested with.
- infer_slots N: the router may open up to N slots (default: the number of domains); heads, private units, n-gram tables and
  FiLM codes are allocated for N slots.
- Main block: PLATYPUS_CLEAN=1 sets inf_clean 1 and turns the oracle options off (inf_rem_probe_frac 0, inf_mix_benchcue 0,
  inf_jd_plantw 0, inf_jd_real 0); PLATYPUS_HINTS=0 turns conduit hints off (cd_hint_frac 0, cd_hint_adapt 0);
  PLATYPUS_SLOTS=N sets infer_slots."""
import sys
src = open(sys.argv[1]).read()
def rep(old, new, count=1):
    global src
    n = src.count(old); assert n == count, (n, count, old[:140]); src = src.replace(old, new)

# ---- constants and the slot count
rep('DEVICE="cuda" if torch.cuda.is_available() else "cpu"\n',
    'DEVICE="cuda" if torch.cuda.is_available() else "cpu"\n'
    '# CLEAN EVALUATION (inf_clean): wordings the system never writes (conclusions are written as "So A is connected to C.", typed\n'
    '#   "A r1 someone who r2 C."); the fit wording measures schema fit with a prompt different from every test prompt\n'
    '_HO_TWO=("{A} is tied to ","The one tied to {A} is ")\n'
    '_HO_TYPED="{A} {r0} a person who {r1} "\n'
    '_FIT_W="{A} keeps company with "\n'
    'def _nslots(cfg):\n'
    '    """slots the router may open (infer_slots; default: one per domain)"""\n'
    '    return max(len(PILE_DOMAINS),int(getattr(cfg,"infer_slots",0) or 0))\n')
rep("    for d in range(len(PILE_DOMAINS)):\n", "    for d in range(_nslots(cfg)):\n")            # ngp_init
rep("            k_=int(cfg.pu_k); ND_=len(PILE_DOMAINS)", "            k_=int(cfg.pu_k); ND_=_nslots(cfg)")
rep("s.film_codes=nn.Parameter(torch.randn(len(PILE_DOMAINS),k_)*0.1)", "s.film_codes=nn.Parameter(torch.randn(_nslots(cfg),k_)*0.1)")
rep("        s.heads=nn.ModuleList([Head(cfg.d_model,cfg.vocab) for _ in range(len(PILE_DOMAINS))])", "        s.heads=nn.ModuleList([Head(cfg.d_model,cfg.vocab) for _ in range(_nslots(cfg))])")
rep("object.__setattr__(model,'_bg',torch.zeros(len(PILE_DOMAINS),256*256,device=dev))", "object.__setattr__(model,'_bg',torch.zeros(_nslots(cfg),256*256,device=dev))")
rep("G=model.__dict__['_borrow_g']=torch.zeros(len(PILE_DOMAINS),device=y.device)", "G=model.__dict__['_borrow_g']=torch.zeros(_nslots(cfg),device=y.device)")
rep("R_=model.__dict__['_borrow_r']=torch.zeros(len(PILE_DOMAINS),device=y.device)", "R_=model.__dict__['_borrow_r']=torch.zeros(_nslots(cfg),device=y.device)")
rep("    if len(S)>=len(PILE_DOMAINS) and float(cfg.infer_gap_eps)>0", "    if len(S)>=_nslots(cfg) and float(cfg.infer_gap_eps)>0")
rep("    elif len(S)>=len(PILE_DOMAINS) and not (rep_ok and loss_ok):", "    elif len(S)>=_nslots(cfg) and not (rep_ok and loss_ok):")
rep("    if gap<=cfg.infer_delta or len(S)>=len(PILE_DOMAINS): s_=best; kind='revisit'", "    if gap<=cfg.infer_delta or len(S)>=_nslots(cfg): s_=best; kind='revisit'")

# ---- held-out probes, bridge distractors, no slot given
rep("""    def score(prompts,cands,slot):
        out=[]
        for p_,cs_ in zip(prompts,cands):
            if _INF.get('corr'): p_=_cue_fix(cfg,p_)                                # cg26m CUE REPAIR (question time, store on)
            seqs=[(p_+c_+'.').encode('latin-1')[-int(cfg.seq_len):] for c_ in cs_]; L_=max(len(s_) for s_ in seqs)
            X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
            for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
            sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot; sv6_=_INFM['on']; _INFM['on']=bool(_INF.get('corr',False))
            try: lp_=F.log_softmax(model(X_[:,:-1],slot).float(),-1)
            finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=sv6_
            n0_=[len(s_)-len((c_+'.').encode()) for s_,c_ in zip(seqs,cs_)]
            sc2_=[float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum()) for j_ in range(len(seqs))]
            out.append(int(np.argmax(sc2_)))""",
"""    def score(prompts,cands,slot):
        out=[]
        for p_,cs_ in zip(prompts,cands):
            if _INF.get('corr'): p_=_cue_fix(cfg,p_)                                # cg26m CUE REPAIR (question time, store on)
            seqs=[(p_+c_+'.').encode('latin-1')[-int(cfg.seq_len):] for c_ in cs_]; L_=max(len(s_) for s_ in seqs)
            X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
            for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
            n0_=[len(s_)-len((c_+'.').encode()) for s_,c_ in zip(seqs,cs_)]
            # slot -1 (inf_clean): NO SLOT GIVEN - the candidates' log-probabilities are mixed over every slot in use (equal weights)
            sls_=[slot] if slot>=0 else (sorted(set(int(v_) for v_ in model.__dict__.get('_t2s',{}).values())) or [0]); tot_=None
            for sl9_ in sls_:
                sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl9_; sv6_=_INFM['on']; _INFM['on']=bool(_INF.get('corr',False))
                try: lp_=F.log_softmax(model(X_[:,:-1],sl9_).float(),-1)
                finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=sv6_
                s9_=np.array([float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum()) for j_ in range(len(seqs))])
                tot_=s9_ if tot_ is None else np.logaddexp(tot_,s9_)
            sc2_=list(tot_-math.log(len(sls_)))
            out.append(int(np.argmax(sc2_)))""")
rep("        if float(getattr(cfg,'inf_para_frac',0.0))>0: KINDS_=KINDS_+('PARAPHRASED one-hop (unseen wording)','SEEN-WORDING one-hop (para probes)')\n",
    "        if float(getattr(cfg,'inf_para_frac',0.0))>0: KINDS_=KINDS_+('PARAPHRASED one-hop (unseen wording)','SEEN-WORDING one-hop (para probes)')\n"
    "        if int(getattr(cfg,'inf_clean',0)): KINDS_=KINDS_+('HELD-OUT two-hop','HELD-OUT typed two-hop','HELD-OUT three-hop')   # unwritten wordings, bridges as candidates, no slot given\n")
rep("                if kind in ('three-hop','IN-CONTEXT three-hop') and len(n_)<4: continue",
    "                if kind in ('three-hop','IN-CONTEXT three-hop','HELD-OUT three-hop') and len(n_)<4: continue")
rep("                tgt={'one-hop':n_[1],'three-hop':n_[-1],'IN-CONTEXT three-hop':n_[-1],",
    "                tgt={'one-hop':n_[1],'three-hop':n_[-1],'IN-CONTEXT three-hop':n_[-1],'HELD-OUT three-hop':n_[-1],")
rep("""                    'UPDATED one-hop':f"{n_[0]} {r_[0]} ",'UPDATED two-hop':f"So {n_[0]} is connected to ",'NOISE-CONTRADICTED one-hop':f"{n_[0]} {r_[0]} "}.get(kind,'')""",
    """                    'UPDATED one-hop':f"{n_[0]} {r_[0]} ",'UPDATED two-hop':f"So {n_[0]} is connected to ",'NOISE-CONTRADICTED one-hop':f"{n_[0]} {r_[0]} ",
                    'HELD-OUT two-hop':_HO_TWO[len(pr_)%len(_HO_TWO)].format(A=n_[0]),'HELD-OUT three-hop':_HO_TWO[len(pr_)%len(_HO_TWO)].format(A=n_[0]),
                    'HELD-OUT typed two-hop':_HO_TYPED.format(A=n_[0],r0=r_[0],r1=(r_[1] if len(r_)>1 else r_[0]))}.get(kind,'')""")
rep("""                elif kind.startswith('CTX2'): ic_=[n_[1],X0_,Y0_]; cs_=[tgt]+ic_+[x_ for x_ in cs_[1:] if x_ not in ic_][:16]   # in-context names compete with the answer
                pr_.append(pp); cd_.append(cs_); bn_.append(n_[1]); sl_.append(int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0])))""",
"""                elif kind.startswith('CTX2'): ic_=[n_[1],X0_,Y0_]; cs_=[tgt]+ic_+[x_ for x_ in cs_[1:] if x_ not in ic_][:16]   # in-context names compete with the answer
                elif kind.startswith('HELD-OUT'): br_=[x_ for x_ in n_[1:] if x_!=tgt]; cs_=[tgt]+br_+cs_[1:20-len(br_)]   # the BRIDGE(S) are candidates too
                pr_.append(pp); cd_.append(cs_); bn_.append(n_[1]); sl_.append(-1 if kind.startswith('HELD-OUT') else int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0])))""")
rep("""                if kind.startswith('CTX2'): _INF.setdefault('copy',{}).setdefault(kind,[]).extend([int(1<=h_<=3) for h_ in hh_])""",
    """                if kind.startswith('CTX2'): _INF.setdefault('copy',{}).setdefault(kind,[]).extend([int(1<=h_<=3) for h_ in hh_])
                if kind.startswith('HELD-OUT'): _INF.setdefault('brg',{}).setdefault(kind,[]).extend([int(1<=h_<=(2 if 'three' in kind else 1)) for h_ in hh_])""")
rep("""            if kind.startswith('CTX2'): cp0_=_INF['copy'].pop(kind,[]); res[kind+f" (picked another in-context name {np.mean(cp0_) if cp0_ else float('nan'):.2f})"]=res.pop(kind)""",
    """            if kind.startswith('CTX2'): cp0_=_INF['copy'].pop(kind,[]); res[kind+f" (picked another in-context name {np.mean(cp0_) if cp0_ else float('nan'):.2f})"]=res.pop(kind)
            if kind.startswith('HELD-OUT'): bg0_=_INF.setdefault('brg',{}).pop(kind,[]); res[kind+f" (picked a bridge {np.mean(bg0_) if bg0_ else float('nan'):.2f})"]=res.pop(kind)""")

# ---- schema fit with a prompt different from the test prompts
rep("""            ans_=c_.get('C') or c_.get('B'); s0_=c_.get('s','')
            if not ans_ or ans_ not in s0_: continue
            ctx_=s0_[:s0_.rfind(ans_)]""",
"""            ans_=c_.get('C') or c_.get('B'); s0_=c_.get('s','')
            if not ans_ or ans_ not in s0_: continue
            ctx_=s0_[:s0_.rfind(ans_)]
            if int(getattr(cfg,'inf_clean',0)) and not c_.get('fact') and c_.get('A') and c_.get('C'): ctx_=_FIT_W.format(A=c_['A'])   # inf_clean: fit prompt != test prompts""")

# ---- main-block switches (after the dream preset, so they win over it)
rep("""    NAME='STAGE C CONDUIT cg20:""",
"""    if os.environ.get('PLATYPUS_CLEAN','0')=='1':                            # CLEAN EVALUATION (review fixes): held-out wordings, bridges as candidates, no
        for k_,v_ in {'inf_clean':1,'inf_rem_probe_frac':0.0,'inf_mix_benchcue':0.0,'inf_jd_plantw':0,'inf_jd_real':0}.items(): setattr(cfg,k_,v_)   #   slot given; oracle options off
        print('  [clean] PLATYPUS_CLEAN=1: held-out probe wordings, bridges as candidates, no slot given; REM probe subjects, benchmark cues and planted-pair weighting off',flush=True)
    if os.environ.get('PLATYPUS_HINTS','1')=='0':
        cfg.cd_hint_frac=0.0; cfg.cd_hint_adapt=0.0; print('  [clean] PLATYPUS_HINTS=0: no true bytes given to the conduit generator in the closed-loop passes',flush=True)
    if os.environ.get('PLATYPUS_SLOTS','').strip().isdigit():
        cfg.infer_slots=int(os.environ['PLATYPUS_SLOTS']); print(f'  [clean] PLATYPUS_SLOTS: the router may open up to {cfg.infer_slots} slots',flush=True)
    NAME='STAGE C CONDUIT cg20:""")
# slot ids beyond the domain list (infer_slots > number of domains) have no domain name: label them by number
rep("def _lab(cfg,d): return PILE_DOMAINS[d][:6]", "def _lab(cfg,d): return PILE_DOMAINS[d][:6] if 0<=int(d)<len(PILE_DOMAINS) else f\"slot{int(d)}\"")
rep("print('  [recency] protection weight per old domain: '+' '.join(f'{PILE_DOMAINS[d][:6]} {w_(d):.2f}'", "print('  [recency] protection weight per old domain: '+' '.join(f'{_lab(None,d)} {w_(d):.2f}'")
open(sys.argv[2], 'w').write(src)
print('written', sys.argv[2])

# ---------------- ROUTING BY BYTE-PAIR PROFILES (drift-free): logged always, rule with infer_bigram 2 ----------------
src = open(sys.argv[2]).read()
rep("        if cfg.head_select_eval or cfg.fwt_zeroshot or cfg.block_attr or cfg.block_auto: head_stats_update(model, x, dom_id, step, cfg)",
    "        if cfg.head_select_eval or cfg.fwt_zeroshot or cfg.block_attr or cfg.block_auto: head_stats_update(model, x, dom_id, step, cfg)\n"
    "        _rbg_add(model,cfg,x,dom_id)                                         # byte-pair profile of the slot being trained (routing evidence, drift-free)")
rep("def infer_slot_rep(model,cfg,get,t_true,vi):\n",
'''def _rbg_add(model,cfg,x,s_):
    B_=model.__dict__.setdefault('_rbg',{})
    if s_ not in B_: B_[s_]=torch.zeros(256*256,device=x.device)
    B_[s_]+=torch.bincount((x[:,:-1].long()*256+x[:,1:].long()).reshape(-1),minlength=256*256).float()
def _rbg_gain(model,xs):
    """bits per byte pair each slot's profile saves over the pooled profile of all slots, on the decision batches (higher = more this slot's text)"""
    B_=model.__dict__.get('_rbg',{})
    if not B_: return {}
    h_=sum(torch.bincount((x_[:,:-1].long()*256+x_[:,1:].long()).reshape(-1),minlength=256*256).float() for x_ in xs); h_=h_/h_.sum()
    al_=0.5; P0_=sum(B_.values()); lp0_=torch.log2((P0_+al_)/(P0_.sum()+al_*P0_.numel())); ce0_=float(-(h_*lp0_).sum())
    return {s_:ce0_-float(-(h_*torch.log2((c_+al_)/(c_.sum()+al_*c_.numel()))).sum()) for s_,c_ in B_.items()}
def infer_slot_rep(model,cfg,get,t_true,vi):
''')
rep("""    cand=min(sc,key=sc.get); rnear=min(ds,key=ds.get)                     # candidate = lowest BPC; the representation must agree""",
"""    G9_=_rbg_gain(model,xs); bg9_=int(getattr(cfg,'infer_bigram',0))
    cand=min(sc,key=sc.get); rnear=min(ds,key=ds.get)                     # candidate = lowest BPC; the representation must agree""")
rep("""    elif rep_ok and gap<=(float(cfg.infer_delta_new) if float(cfg.infer_delta_new)>0 else cfg.infer_delta): s_=cand; kind='revisit'   # cg13c free-slot threshold
    else: s_=len(S); S.append(s_); kind='new'""",
"""    elif bg9_>=2 and G9_:                                                   # BYTE-PAIR RULE (no cap needed): revisit the slot whose profile saves the most bits over
        gb9_=max(G9_,key=G9_.get)                                             #   the pooled profile, if it saves at least infer_bigram_tau bits per pair; else a new slot
        if G9_[gb9_]>=float(getattr(cfg,'infer_bigram_tau',0.3)): s_=gb9_; kind=f'revisit (byte-pair profile, {G9_[gb9_]:.2f} bits)'
        else: s_=len(S); S.append(s_); kind=f'new (best byte-pair gain {G9_[gb9_]:.2f} bits)'
    elif rep_ok and gap<=(float(cfg.infer_delta_new) if float(cfg.infer_delta_new)>0 else cfg.infer_delta): s_=cand; kind='revisit'   # cg13c free-slot threshold
    else: s_=len(S); S.append(s_); kind='new'""")
rep("""          f"(running {A[0]}/{A[1]}) | distances "+' '.join(f"s{k}:{v:.3f}" for k,v in ds.items())+" | BPC "+' '.join(f"s{k}:{v:.2f}" for k,v in sc.items()))""",
    """          f"(running {A[0]}/{A[1]}) | distances "+' '.join(f"s{k}:{v:.3f}" for k,v in ds.items())+" | BPC "+' '.join(f"s{k}:{v:.2f}" for k,v in sc.items())
          +(" | byte-pair gain "+' '.join(f"s{k}:{v:.2f}" for k,v in sorted(G9_.items())) if G9_ else ""))""")
open(sys.argv[2], 'w').write(src)
print('byte-pair routing added')
