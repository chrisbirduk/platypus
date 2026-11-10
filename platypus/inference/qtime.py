"""Question-time chaining with a working memory."""
import numpy as np
import torch
import torch.nn.functional as F
from platypus.private import (
    _HCAP, _XDG,
)
from platypus.layers import (
    _CARRY,
)
from platypus.conduit import (
    _carry_load, _carry_save,
)
from platypus.inference.store import (
    _INFM, _infm_corr, _infm_keys, _infm_sparse,
)
from platypus.inference.units import (
    _INF_NAME_RX, _INF_REL, _INF_TRIP_RX,
)
from platypus.inference.probes import (
    _inf_phrase, inf_probes,
)
from platypus.inference.sentences import (
    _inf_store,
)
from platypus.inference.selftaught import (
    _inf_lp,
)
from platypus.inference.rem import (
    _inf_gen,
)

@torch.no_grad()
def _inf_wm_write(model,cfg,ctx_,sl_,iters_=20,beta_=0.5):
    """WORKING MEMORY: the facts in a prompt ('Name relation Name.') are written into a scratch store with the same name-keyed delta rule as
    sleep writes (to convergence, on top of whatever the long-term store already predicts); returns the names added for the question"""
    import zlib as _zl
    fs_=[m_.group(0).decode('latin-1') for m_ in _INF_TRIP_RX.finditer(ctx_.encode('latin-1'))]   # question-time readout: prompt facts parsed by the template pattern
    if not fs_: return []
    dev_=model.embed.weight.device
    if _INFM.get('Mw') is None or _INFM['Mw'].shape!=_INFM['M'].shape: _INFM['Mw']=torch.zeros_like(_INFM['M'])
    add_=[]
    for f_ in fs_:
        for n_ in _INF_NAME_RX.findall(f_.encode('latin-1')):
            if n_ not in _INFM['names']: _INFM['names'][n_]=int(_zl.crc32(n_)); add_.append(n_)
    rows_=[(' '+f_+'.').encode('latin-1') for f_ in fs_]; Lr_=max(len(r_) for r_ in rows_)+2
    X_=torch.full((len(rows_),Lr_),32,dtype=torch.long,device=dev_); mk_=torch.zeros(len(rows_),Lr_-1,device=dev_)
    for j_,(r_,f_) in enumerate(zip(rows_,fs_)):
        X_[j_,:len(r_)]=torch.tensor(list(r_),device=dev_); a0_=1+len(f_.split(' ')[0]); mk_[j_,a0_:len(r_)-1]=1.0   # targets: everything after the subject
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl_; sh_=_HCAP['on']; _HCAP['on']=True; sv_=_INFM['on']; _INFM['on']=False; fz_=_INFM.pop('force',None); was_=model.training; model.eval()
    try: lg_=model(X_[:,:-1],sl_).float(); z_=_HCAP['h'].float()
    finally:
        _carry_load(sv4_); _XDG['cur']=sc_; _HCAP['on']=sh_; _INFM['on']=sv_; model.train(was_)
        if fz_ is not None: _INFM['force']=fz_
    K_,id_=_infm_keys(z_,X_[:,:-1],lg_); zf_=K_.reshape(-1,K_.shape[-1]); idf_=id_.reshape(-1); lgf_=lg_.reshape(-1,256)
    oh_=F.one_hot(X_[:,1:].reshape(-1),256).float(); mf_=(mk_.reshape(-1)*(idf_>=0).float())[:,None]
    Ms_=_INFM['M']
    for _ in range(int(iters_)):
        q_=F.softmax(lgf_+_infm_corr(zf_,idf_),-1); R_=(oh_-q_)*mf_; sel_=((idf_>=0)&(R_.abs().sum(-1)>0)).nonzero().squeeze(-1)
        for c_ in range(0,sel_.numel(),512):
            s_=sel_[c_:c_+512]; i_,v_=_infm_sparse(zf_[s_],idf_[s_]); _INFM['Mw'].index_add_(0,i_.reshape(-1),(beta_*v_[...,None]*R_[s_][:,None,:]).reshape(-1,256))
    return add_
def _inf_wm_clear(add_):
    if _INFM.get('Mw') is not None: _INFM['Mw'].zero_()
    for n_ in add_: _INFM['names'].pop(n_,None)
@torch.no_grad()
def inf_qtime2(model,cfg,tag=''):
    """QUESTION-TIME CHAINING WITH A WORKING MEMORY. Prompts 'A r0 B. B r1 C. X r1 Y.' (both orders) and the broken-bridge decoy (A r0 D);
    PROBE chains (also in long-term memory) and NEW chains (invented names that appear only in the prompt). Readers:
      model       the weights alone (in-context copying)          long-term   + the sleep-time fast store
      WM          + a working memory written from the prompt only (long-term store off)      WM+LT  both
    Each: DIRECT 'So A is connected to ' and TWO LOOKUPS ('A r0 ' -> B^, then 'B^ r ' for each relation in the prompt). Decoys: answering
    C is a FALSE chain. 16 candidates (chance 0.06)."""
    if _INFM.get('M') is None or _INFM.get('mu') is None: return
    ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd')][:int(getattr(cfg,'inf_qt_n',60))]
    alln_=[n_ for c_ in inf_probes(cfg) for n_ in c_['names']]; t2s_=model.__dict__.get('_t2s',{}); rs_=np.random.RandomState(99)
    used_=set(alln_)|set(n_.decode('latin-1') for n_ in _INFM['names'])
    def fresh():
        while True:
            n_=''.join(rs_.choice(['qa','xo','zu','vi','ke','lom','dar','fen','sul','gor','pim','tas'],rs_.randint(2,4))); n_=n_[0].upper()+n_[1:]
            if 5<=len(n_)<=10 and n_ not in used_: used_.add(n_); return n_
    items_=[]
    for src_ in ('probe','NEW'):
        for i_,c_ in enumerate(ch_):
            r0_,r1_=c_['rels']; sl_=int(t2s_.get(c_['doms'][0],c_['doms'][0]))
            if src_=='probe': A_,B_,C_=c_['names']; oth_=[x_ for x_ in alln_ if x_ not in c_['names']]; X_,Y_,D_=oth_[(3*i_)%len(oth_)],oth_[(3*i_+1)%len(oth_)],oth_[(3*i_+2)%len(oth_)]; ex_=list(rs_.choice([x_ for x_ in oth_ if x_ not in (X_,Y_,D_)],12,replace=False))
            else: A_,B_,C_,X_,Y_,D_=[fresh() for _ in range(6)]; ex_=[fresh() for _ in range(12)]
            f0_,f1_=c_['tp'][0].format(a=A_,b=B_)[:-1],c_['tp'][1].format(a=B_,b=C_)[:-1]; fx_=f"{X_} {r1_} {Y_}"
            for kind_ in ('AB','BA','decoy AB','decoy BA'):
                dec_='decoy' in kind_; g0_=f0_.replace(B_,D_) if dec_ else f0_
                ctx_=(f"{g0_}. {f1_}. {fx_}. " if kind_.endswith('AB') else f"{f1_}. {g0_}. {fx_}. ")
                items_.append(dict(src=src_,kind=kind_,dec=dec_,ctx=ctx_,A=A_,B=(D_ if dec_ else B_),C=C_,r=sorted({r0_,r1_}),r0=r0_,cand=[B_,X_,Y_,C_]+([D_] if dec_ else [])+ex_[:12-int(dec_)],sl=sl_,oth=(B_,X_,Y_,D_)))
    res_={}
    for mode_ in ('model','long-term','WM','WM+LT'):
        for it_ in items_:
            add_=[]; st0_=mode_!='model'
            if mode_=='WM': _INFM['lt_off']=True
            if mode_.startswith('WM'): _INFM['wm_on']=True; add_=_inf_wm_write(model,cfg,it_['ctx'],it_['sl'])
            try:
                cand_=it_['cand']; lp_=_inf_lp(model,cfg,it_['ctx']+f"So {it_['A']} is connected to ",cand_,it_['sl'],st0_); pk_=cand_[int(np.argmax(lp_))]
                lb_=_inf_lp(model,cfg,it_['ctx']+f"{it_['A']} {it_['r0']} ",cand_,it_['sl'],st0_); Bh_=cand_[int(np.argmax(lb_))]; best_=None
                for r_ in it_['r']:
                    cc_=[x_ for x_ in cand_ if x_!=Bh_]; lc_=_inf_lp(model,cfg,it_['ctx']+f"{Bh_} {r_} ",cc_,it_['sl'],st0_); j_=int(np.argmax(lc_))
                    if best_ is None or lc_[j_]>best_[0]: best_=(lc_[j_],cc_[j_])
            finally: _INFM.pop('lt_off',None); _INFM.pop('wm_on',None); _inf_wm_clear(add_)
            k_=(it_['src'],'decoy' if it_['dec'] else 'real',mode_); s_=res_.setdefault(k_,[0,0,0,0,0])
            s_[0]+=1; s_[1]+=int(pk_==it_['C']); s_[2]+=int(Bh_==it_['B']); s_[3]+=int(best_[1]==it_['C']); s_[4]+=int(pk_ in it_['oth'])
    for src_ in ('probe','NEW'):
        out_=[]
        for mode_ in ('model','long-term','WM','WM+LT'):
            r_=res_.get((src_,'real',mode_)); d_=res_.get((src_,'decoy',mode_))
            if not r_: continue
            n_,nd_=max(1,r_[0]),max(1,d_[0])
            out_.append(f"{mode_}: direct {r_[1]/n_:.2f} (copy {r_[4]/n_:.2f}), two lookups bridge {r_[2]/n_:.2f} answer {r_[3]/n_:.2f} | decoy FALSE chain direct {d_[1]/nd_:.2f}, two lookups {d_[3]/nd_:.2f}")
        print(f"  [inference question-time chaining WITH WORKING MEMORY | {tag} | {src_} chains{' (in long-term memory too)' if src_=='probe' else ' (names only in the prompt)'}, {len(ch_)} x 2 orders] "+" || ".join(out_),flush=True)
_INF_PARA={'works with':'is a colleague of','lives near':'resides close to','reports to':'answers to','is a partner of':'partners with','trained under':'was taught by','is linked to':'is connected with'}
@torch.no_grad()
def inf_qtime3(model,cfg,tag='',D_=None):
    """MULTI-HOP QUESTION-TIME CHAINING, depth 2-5, on NEW chains (names only in the prompt): facts N0 r0 N1. N1 r1 N2. ... plus one distractor
    fact per hop (other names, same relation), shuffled; the question supplies the relation path ('N0's r0's r1 ...').
      TEXT  : working memory written from the prompt; each hop asks the store 'cur r ' and picks a name (bytes), then re-reads it as the next key
      ID    : an IDENTIFIER memory: key = code(subject) * code(relation), value = code(object) (fixed random codes from hashes, never trained),
              summed outer products, read + clean-up to the nearest known name code; the returned IDENTIFIER is the next key (no text round trip).
              Background load: every recorded long-term fact is stored in the same memory (interference).
    Decoys: one middle link broken (its object replaced by an outsider) -> a correct reader ABSTAINS. Abstention: TEXT if the best hop margin over the
    runner-up < inf_qt_m nats; ID if the clean-up similarity < inf_qt_sim. Paraphrase: the question uses synonyms of the relations."""
    if _INFM.get('M') is None or _INFM.get('mu') is None: return
    import zlib as _zl
    D_=int(D_ or getattr(cfg,'inf_id_dim',1024))
    rs_=np.random.RandomState(1234); used_=set(n_.decode('latin-1') for n_ in _INFM['names'])|set(x_ for c_ in inf_probes(cfg) for x_ in c_['names'])
    def fresh():
        while True:
            n_=''.join(rs_.choice(['qa','xo','zu','vi','ke','lom','dar','fen','sul','gor','pim','tas','bru','nix'],rs_.randint(2,5))); n_=n_[0].upper()+n_[1:]
            if 5<=len(n_)<=10 and n_ not in used_: used_.add(n_); return n_
    def code(s_):
        g_=np.random.RandomState(_zl.crc32(s_.encode('latin-1'))%(2**31)); return torch.tensor(g_.choice([-1.0,1.0],D_),dtype=torch.float32)
    rels_=[_inf_phrase(t_) for t_ in _INF_REL]; tpl_={_inf_phrase(t_):t_ for t_ in _INF_REL}
    F0_=(_inf_store(model).get('facts') or [])[:3000]; m_=float(getattr(cfg,'inf_qt_m',1.0)); sim_=float(getattr(cfg,'inf_qt_sim',0.5))
    Mbg_=torch.zeros(D_,D_); bg_=[(f_['A'],f_['r'],f_['B']) for f_ in F0_]; nb_=int(getattr(cfg,'inf_qt_bg',2000)); nl_=[n_.decode('latin-1') for n_ in _INFM['names']]
    while len(bg_)<nb_ and len(nl_)>=2: a0_,b0_=rs_.choice(nl_,2,replace=False); bg_.append((a0_,rels_[rs_.randint(len(rels_))],b0_))   # background load: stored-name facts
    for a0_,r0_,b0_ in bg_: Mbg_+=torch.outer(code(a0_)*code(r0_),code(b0_))
    F0_=bg_
    sl_=int(model.__dict__.get('_t2s',{}).get(0,0)); out_=[]; nc_=int(getattr(cfg,'inf_qt_comp',0))
    for depth_ in (2,3,4,5):
        st_={k_:0 for k_ in ('n','t_ok','t_ans','i_ok','i_ans','p_ok','p_ans','ip_ok','ip_ans','dn','t_dab','i_dab','t_dfalse','i_dfalse')}
        for trial_ in range(int(getattr(cfg,'inf_qt_n',40))//2):
            for dec_ in (False,True):
                N_=[fresh() for _ in range(depth_+1)]; R_=[rels_[rs_.randint(len(rels_))] for _ in range(depth_)]
                facts_=[(N_[i_],R_[i_],N_[i_+1]) for i_ in range(depth_)]; brk_=None
                if dec_: brk_=1+rs_.randint(depth_-1) if depth_>1 else 0; facts_[brk_]=(fresh(),R_[brk_],N_[brk_+1])   # the chain breaks: N_brk has no R_brk fact
                facts_+=[(fresh(),R_[i_],fresh()) for i_ in range(depth_)]
                for i_ in range(depth_):   # COMPETING RELATIONS: the chain node also has facts under OTHER relations -> the reader must use the relation
                    oth_=[r_ for r_ in rels_ if r_!=R_[i_]]; rs_.shuffle(oth_)
                    facts_+=[(N_[i_],r_,fresh()) for r_ in oth_[:nc_]]
                rs_.shuffle(facts_)
                ctx_=' '.join(tpl_[r_].format(a=a_,b=b_) for a_,r_,b_ in facts_)+' '
                names_=sorted(set(x_ for a_,_,b_ in facts_ for x_ in (a_,b_))); cand_=names_+[fresh() for _ in range(4)]
                # TEXT: working memory, hop by hop
                _INFM['lt_off']=True; _INFM['wm_on']=True; add_=_inf_wm_write(model,cfg,ctx_,sl_)
                try:
                    res_={}
                    rmap_={c0_['p']:c0_['q'] for c0_ in (_inf_store(model).get('rules') or []) if c0_['q'] in rels_}   # LEARNED RULES rewrite a paraphrased relation to one the memory holds
                    for mode_,RR_ in (('t',R_),('p',[rmap_.get(_INF_PARA[r_],_INF_PARA[r_]) for r_ in R_])):
                        cur_=N_[0]; ok_=True; ab_=False
                        for r_ in RR_:
                            cc_=[x_ for x_ in cand_ if x_!=cur_]; lp_=np.array(_inf_lp(model,cfg,f"{cur_} {r_} ",cc_,sl_,True)); o_=np.argsort(-lp_)
                            if lp_[o_[0]]-lp_[o_[1]]<m_: ab_=True; break
                            cur_=cc_[o_[0]]
                        res_[mode_]=(ab_,cur_)
                finally: _INFM.pop('lt_off',None); _INFM.pop('wm_on',None); _inf_wm_clear(add_)
                # ID: identifier memory (prompt facts + long-term background), identifiers returned and reused
                M_=Mbg_.clone()
                for a_,r_,b_ in facts_: M_+=torch.outer(code(a_)*code(r_),code(b_))
                C_=torch.stack([code(x_) for x_ in cand_]); ires_={}
                for mode_,RR_ in (('i',R_),('ip',[_INF_PARA[r_] for r_ in R_])):   # ip: paraphrased relation identifiers (no canonicalisation)
                    cur_=N_[0]; iab_=False
                    for r_ in RR_:
                        v_=(code(cur_)*code(r_))@M_; s_=F.cosine_similarity(v_[None,:],C_); j_=int(s_.argmax())
                        if float(s_[j_])<sim_: iab_=True; break
                        cur_=cand_[j_]
                    ires_[mode_]=(iab_,cur_)
                iab_,cur_=ires_['i']
                st_['ip_ans']+=int(not dec_ and not ires_['ip'][0]); st_['ip_ok']+=int(not dec_ and (not ires_['ip'][0]) and ires_['ip'][1]==N_[-1])
                if not dec_:
                    st_['n']+=1
                    st_['t_ans']+=int(not res_['t'][0]); st_['t_ok']+=int((not res_['t'][0]) and res_['t'][1]==N_[-1])
                    st_['p_ans']+=int(not res_['p'][0]); st_['p_ok']+=int((not res_['p'][0]) and res_['p'][1]==N_[-1])
                    st_['i_ans']+=int(not iab_); st_['i_ok']+=int((not iab_) and cur_==N_[-1])
                else:
                    st_['dn']+=1; st_['t_dab']+=int(res_['t'][0]); st_['i_dab']+=int(iab_)
                    st_['t_dfalse']+=int((not res_['t'][0]) and res_['t'][1]==N_[-1]); st_['i_dfalse']+=int((not iab_) and cur_==N_[-1])
        n_,d_=max(1,st_['n']),max(1,st_['dn']); acc_=lambda ok,an: f"{ok/n_:.2f} correct, answers {an/n_:.2f}, accuracy when answering {ok/max(1,an):.2f}"
        out_.append(f"DEPTH {depth_}: TEXT {acc_(st_['t_ok'],st_['t_ans'])}; PARAPHRASED question {acc_(st_['p_ok'],st_['p_ans'])}; ID {acc_(st_['i_ok'],st_['i_ans'])}; ID PARAPHRASED {acc_(st_['ip_ok'],st_['ip_ans'])} | "
                    f"decoys: abstained TEXT {st_['t_dab']/d_:.2f} / ID {st_['i_dab']/d_:.2f}, FALSE chain TEXT {st_['t_dfalse']/d_:.2f} / ID {st_['i_dfalse']/d_:.2f}")
    print(f"  [inference multi-hop question-time | {tag} | new chains only in the prompt, one distractor per hop, {nc_} competing-relation facts per chain node, {len(F0_)} long-term facts as background load, abstain margin {m_} nats / similarity {sim_}] "+" || ".join(out_),flush=True)
def _inf_auc(p_,n_):
    p_=np.asarray(p_,float)[:,None]; n_=np.asarray(n_,float)[None,:]
    return float(((p_>n_).sum()+0.5*(p_==n_).sum())/max(1,p_.size*n_.size))
@torch.no_grad()
def inf_splice(model,cfg,tag='',nf_=3):
    """SPLICE-AND-APPRAISE (cg20o test): can the model tell a true link from a false recombination by 'knowing what it knows'?
    For each two-hop chain A-r0-B, B-r1-C the TRUE splice is (A, C); the FALSE splices put the C of other chains (rare names from other
    episodes) into the same common frame. Each candidate is appraised with the store OFF by:
      feels right      log p(X | 'So A is connected to ')                       (raw confidence: fluency of the splice)
      specific to A    the same minus log p(X | 'So someone is connected to ')  (does the link belong to A, or is X just familiar?)
      linked (own)     the weights recall A's bridge (greedy 'A r0 ' -> B^), then log p(X | 'B^ r1 ')   (self-retrieval: two one-hop lookups)
      linked, specific the same minus log p(X | 'Someone r1 ')
      linked (oracle)  log p(X | 'B r1 ') with the true bridge (upper bound if bridge recall were perfect)
      store-backed     'feels right' with the fast store on (memory, not self-appraisal)
    Reported: pooled AUC true vs false (usable as an accept threshold), paired accuracy (true beats each false splice of the same A),
    and the share of true splices accepted when at most 5% of false splices are accepted."""
    ch_=[c_ for c_ in inf_probes(cfg) if len(c_['names'])==3 and not c_.get('upd')][:int(getattr(cfg,'inf_splice_n',150))]
    if len(ch_)<nf_+2: return None
    rs_=np.random.RandomState(4242); t2s_=model.__dict__.get('_t2s',{}); sl_=lambda d_: int(t2s_.get(d_,d_))
    st_=_INFM.get('M') is not None and _INFM.get('mu') is not None
    K_=['feels right','specific to A','linked (own bridge)','linked, specific','linked (oracle bridge)']+(['store-backed','linked (store lookups)'] if st_ else [])
    V_={k_:[] for k_ in K_}; br_=[]; brs_=[]; dg_=[0,0]
    for i_,c_ in enumerate(ch_):
        A_,B_,C_=c_['names']; r0_,r1_=c_['rels']; d0_,d1_=sl_(c_['doms'][0]),sl_(c_['doms'][1])
        cs_=[C_]+[ch_[j_]['names'][2] for j_ in rs_.choice([j_ for j_ in range(len(ch_)) if j_!=i_],nf_,replace=False)]
        f_=np.array(_inf_lp(model,cfg,f"So {A_} is connected to ",cs_,d0_,False)); b_=np.array(_inf_lp(model,cfg,"So someone is connected to ",cs_,d0_,False))
        g_=_inf_gen(model,cfg,f"{A_} {r0_} ",d0_,False); Bh_=(g_.split() or [''])[0].strip('.,;:'); br_.append(int(Bh_==B_))
        l_=np.array(_inf_lp(model,cfg,f"{Bh_ or 'Someone'} {r1_} ",cs_,d1_,False)); lb_=np.array(_inf_lp(model,cfg,f"Someone {r1_} ",cs_,d1_,False))
        o_=np.array(_inf_lp(model,cfg,f"{B_} {r1_} ",cs_,d1_,False))
        for k_,v_ in zip(K_,[f_,f_-b_,l_,l_-lb_,o_]): V_[k_].append(v_)
        if st_:
            _INFM['dbg']=[0,0,0.0]; V_['store-backed'].append(np.array(_inf_lp(model,cfg,f"So {A_} is connected to ",cs_,d0_,True))); dg_[0]+=_INFM['dbg'][1]; dg_[1]+=_INFM['dbg'][0]
            gs_=_inf_gen(model,cfg,f"{A_} {r0_} ",d0_,True); Bs_=(gs_.split() or [''])[0].strip('.,;:'); brs_.append(int(Bs_==B_))
            V_['linked (store lookups)'].append(np.array(_inf_lp(model,cfg,f"{Bs_ or 'Someone'} {r1_} ",cs_,d1_,True)))
    out_=[]; res_={}
    for k_ in K_:
        M_=np.stack(V_[k_]); P_=M_[:,0]; N_=M_[:,1:].ravel(); thr_=np.quantile(N_,0.95)
        res_[k_]=(_inf_auc(P_,N_),float((M_[:,:1]>M_[:,1:]).mean()),float((P_>thr_).mean()))
        out_.append(f"{k_}: AUC {res_[k_][0]:.2f}, paired {res_[k_][1]:.2f}, accepted at 5% false {res_[k_][2]:.2f}")
    print(f"  [inference splice-and-appraise | {tag}] {len(ch_)} two-hop chains, {nf_} false splices each (the C of another chain in the same frame), store OFF unless noted"
          f" | bridge recalled by the weights (A r0 -> B) {np.mean(br_):.2f}"+(f", with the store {np.mean(brs_):.2f} (store-keyed positions {dg_[0]}/{dg_[1]}, names in the store {len(_INFM.get('names') or {})})" if st_ else ", no fast store")+" | "+" | ".join(out_),flush=True)
    return res_
