"""Probe evaluation, CLS measurements and REM examination."""
from platypus.common import _fail
import math
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import (
    _HO_TWO, _HO_TYPED,
)
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
    _INFM, _inf_sync,
)
from platypus.inference.units import (
    _INF_NAME_RX, _INF_REL, _INF_TRIP_RX,
)
from platypus.inference.probes import (
    _INF, _inf_phrase, inf_probes,
)
from platypus.inference.sentences import (
    _inf_store,
)
from platypus.inference.links import (
    _CR, _cue_fix,
)
from platypus.inference.infer import (
    _cue_readout, _inf_truth, _kgx_ho, _kgx_readout, inf_idmem,
)

def inf_eval(model,cfg,S_,tag):
    """probe recall from the weights; with a fast store (cg20h) also WITH the store active"""
    _CR['df']=S_.get('df')                                                      # cg26m CUE REPAIR: word frequencies (a common lower-case word is never a name)
    _inf_sync(cfg,'eval start'); _INF['corr']=False; r_=_inf_eval1(model,cfg,S_,tag); _inf_sync(cfg,'eval from the weights')
    if _INFM['M'] is not None and _INFM['mu'] is not None:
        _INF['corr']=True; _INFM['dbg']=[0,0,0.0]
        try:
            r_=_inf_eval1(model,cfg,S_,tag+" | WITH THE FAST STORE"); d_=_INFM['dbg']; _inf_sync(cfg,'eval with the store')
            print(f"  [inference eval | fast store use] {d_[1]}/{d_[0]} probe positions store-keyed, mean max |logit correction| there {d_[2]/max(1,d_[1]):.3f} | names in the store {len(_INFM.get('names',{}))}",flush=True)
            if int(getattr(cfg,'inf_cue_repair',0)):
                try: _cue_readout(model,cfg,S_)
                except Exception as e9_: _fail('inference.evaluate:48'); print(f"  [inference CUE REPAIR] readout unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
        finally: _INF['corr']=False
    if int(getattr(cfg,'inf_splice',0)) and ('END' in tag or (int(model.__dict__.get('_visit',0))+1)%max(1,int(getattr(cfg,'inf_splice_every',1)))==0):
        try: inf_splice(model,cfg,tag)
        except Exception as e_: _fail('inference.evaluate:52'); print(f"  [inference splice-and-appraise] skipped: {e_!r}"[:300],flush=True)
    if int(getattr(cfg,'inf_eval_qtime',0)) and not _INF.get('corr') and ('END' in tag or (int(model.__dict__.get('_visit',0))+1)%max(1,int(getattr(cfg,'inf_qt_every',getattr(cfg,'inf_splice_every',1))))==0):
        try: inf_qtime2(model,cfg,tag)
        except Exception as e_: _fail('inference.evaluate:55'); print(f"  [inference question-time chaining] skipped: {e_!r}"[:300],flush=True)
        if int(getattr(cfg,'inf_idmem',0)):
            try: inf_idmem(model,cfg,S_,tag)
            except Exception as e_: _fail('inference.evaluate:58'); print(f"  [inference IDENTIFIER MEMORY] skipped: {e_!r}"[:300],flush=True)
        if int(getattr(cfg,'inf_eval_qtime3',0)):
            try:
                for nc0_ in sorted(set([0,int(getattr(cfg,'inf_qt_comp',2))])):
                    cfg.inf_qt_comp=nc0_; inf_qtime3(model,cfg,tag)
            except Exception as e_: _fail('inference.evaluate:63'); print(f"  [inference multi-hop question-time] skipped: {e_!r}"[:300],flush=True)
    if int(getattr(cfg,'inf_kgx',0)) and 'END' in tag:                         # GRAPH INTO WEIGHTS readouts (no state change: names restored, nothing written)
        try: _kgx_readout(model,cfg,S_)
        except Exception as e9_:
            _fail('inference.evaluate:66')
            import traceback as _tb; print(f"  [GRAPH INTO WEIGHTS] unavailable ({type(e9_).__name__}: {e9_}) {_tb.format_exc()[-400:]!r}"[:700],flush=True)
    return r_
@torch.no_grad()
def _inf_eval1(model,cfg,S_,tag):
    """probe recall FROM THE WEIGHTS (no context): one-hop, two-hop ('So A is connected to'), typed two-hop, three-hop; 19 distractor names"""
    ch_=inf_probes(cfg); dev_=model.embed.weight.device; was_=model.training; model.eval(); rs_=np.random.RandomState(77)
    alln_=[n_ for c_ in ch_ for n_ in c_['names']]; res={}
    def score(prompts,cands,slot):
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
            sc2_=list(_score_norm(cfg,tot_-math.log(len(sls_)),np.array([len(seqs[j_])-n0_[j_] for j_ in range(len(seqs))],dtype=np.float64)))   # issue #24: per byte (inf_score)
            out.append(int(np.argmax(sc2_)))
            if _INF.get('cal_on'):
                pp_=np.exp(np.array(sc2_)-max(sc2_)); pp_=pp_/pp_.sum(); A0_=_INF['cal_on'][len(_INF['calrec'])] if len(_INF['calrec'])<len(_INF['cal_on']) else ('','','')
                _INF['calrec'].append((int(np.argmax(sc2_))==0,float(pp_.max()),float(pp_[0]),int(A0_[0].encode('latin-1') in (_INFM.get('names') or {})),A0_[0],cs_[int(np.argmax(sc2_))],A0_[1],A0_[2]))
        return out
    _TRIE_ST={}
    def openh(prompts,targets,slot):
        """OPEN-ENDED (issue #24, inf_eval_open): greedy bytes after the prompt, no candidate list; right if they spell the target name and
        the next byte is not a letter. slot -1: the next-byte distributions of every slot in use, equally mixed (no slot given). Batched:
        rows are right-padded, each row reads its own last position (causal, so padding after it does not matter)."""
        if not prompts: return []
        sls_=[slot] if slot>=0 else (sorted(set(int(v_) for v_ in model.__dict__.get('_t2s',{}).values())) or [0])
        B_=[list((_cue_fix(cfg,p_) if _INF.get('corr') else p_).encode('latin-1'))[-(int(cfg.seq_len)-16):] for p_ in prompts]
        n_=max(len(t_.encode('latin-1')) for t_ in targets)+1; G_=[[] for _ in B_]
        for _ in range(n_):
            R_=[b_+g_ for b_,g_ in zip(B_,G_)]; L_=max(len(r_) for r_ in R_); X_=torch.zeros(len(R_),L_,dtype=torch.long,device=dev_)
            for j_,r_ in enumerate(R_): X_[j_,:len(r_)]=torch.tensor(r_,device=dev_)
            ix_=torch.tensor([len(r_)-1 for r_ in R_],device=dev_); lp9_=None
            for sl9_ in sls_:
                sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl9_; sv6_=_INFM['on']; _INFM['on']=bool(_INF.get('corr',False))
                try: l_=F.log_softmax(model(X_,sl9_).float()[torch.arange(len(R_),device=dev_),ix_],-1)
                finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=sv6_
                lp9_=l_ if lp9_ is None else torch.logaddexp(lp9_,l_)
            for j_,v_ in enumerate(lp9_.argmax(-1).tolist()): G_[j_].append(int(v_))
        hit_=[]
        for g_,t_ in zip(G_,targets):
            o_=bytes(g_).decode('latin-1'); hit_.append(int(o_[:len(t_)]==t_ and (len(o_)==len(t_) or not o_[len(t_)].isalpha())))
        return hit_
    def beamh(prompts,targets,slot,B_=8):
        """BEAM (inf_eval_beam): name-level beam search after the prompt, no candidate list. Beams of letters; every beam of at least one
        letter is also a finished name, scored log p(letters) + log p(any non-letter next) - the NAME SCORE, the same whole-name scoring
        namescore() uses - and the best finished name is the answer. Right if the best is the
        target. slot -1: every slot in use, equally mixed. All prompts' beams in one batch per step."""
        if not prompts: return []
        sls_=[slot] if slot>=0 else (sorted(set(int(v_) for v_ in model.__dict__.get('_t2s',{}).values())) or [0])
        P_=[list((_cue_fix(cfg,p_) if _INF.get('corr') else p_).encode('latin-1'))[-(int(cfg.seq_len)-16):] for p_ in prompts]
        isl_=lambda b_: 65<=b_<=90 or 97<=b_<=122
        LT_=torch.tensor([b_ for b_ in range(256) if isl_(b_)]); NL_=torch.tensor([b_ for b_ in range(256) if not isl_(b_)])
        live_=[[([],0.0)] for _ in P_]; fin_=[[] for _ in P_]; T_=max(len(t_) for t_ in targets)+2
        for st_ in range(T_):
            rows_=[(i_,g_,sc_) for i_,bm_ in enumerate(live_) for g_,sc_ in bm_]
            if not rows_: break
            R_=[P_[i_]+g_ for i_,g_,_ in rows_]; L_=max(len(r_) for r_ in R_); X_=torch.zeros(len(R_),L_,dtype=torch.long,device=dev_)
            for j_,r_ in enumerate(R_): X_[j_,:len(r_)]=torch.tensor(r_,device=dev_)
            ix_=torch.tensor([len(r_)-1 for r_ in R_],device=dev_); lp9_=None
            for sl9_ in sls_:
                sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl9_; sv6_=_INFM['on']; _INFM['on']=bool(_INF.get('corr',False))
                try: l_=F.log_softmax(model(X_,sl9_).float()[torch.arange(len(R_),device=dev_),ix_],-1)
                finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=sv6_
                lp9_=l_ if lp9_ is None else torch.logaddexp(lp9_,l_)
            lp9_=(lp9_-math.log(len(sls_))).cpu()
            cand_=[[] for _ in P_]; Bw_=max(B_,int(getattr(cfg,'inf_eval_beam_first',0))) if st_==0 else B_   # FIRST-STEP WIDTH (inf_eval_beam_first): 26+ keeps every first letter
            for j_,(i_,g_,sc_) in enumerate(rows_):
                if g_: fin_[i_].append((bytes(g_).decode('latin-1'),sc_+float(torch.logsumexp(lp9_[j_][NL_],0))))   # the name ends: any non-letter next
                v_,b9_=lp9_[j_][LT_].topk(min(Bw_,len(LT_)))
                for vv_,bb_ in zip(v_.tolist(),b9_.tolist()): cand_[i_].append((g_+[int(LT_[bb_])],sc_+vv_))
            for i_ in range(len(P_)):
                fin_[i_].sort(key=lambda x_: -x_[1]); best_=fin_[i_][0][1] if len(fin_[i_])>=B_ else -1e18
                live_[i_]=[c_ for c_ in sorted(cand_[i_],key=lambda x_: -x_[1])[:Bw_] if c_[1]>best_]
        return [int(bool(f_) and max(f_,key=lambda x_: x_[1])[0]==t_) for f_,t_ in zip(fin_,targets)]
    def trieh(prompts,targets,slot,K_=32):
        """KNOWN-NAME DECODING (inf_eval_trie): the answer is the best name IN THE FAST STORE'S NAME INVENTORY (_INFM['names'], every name the
        system has stored; probe truth is never added) by the NAME SCORE (log p(name bytes) + log p(any non-letter next), as beamh/namescore).
        Exact: best-first search over a trie of the inventory; a prefix's score bounds every name under it (log-probabilities are <= 0), so the
        search stops when the best finished name scores at least the best open prefix. Up to K_ prefixes per prompt expanded per step, all
        prompts in one batch. slot -1: every slot in use, equally mixed. Returns hits; _TRIE_ST holds inventory size, target-in-inventory,
        steps, and searches cut at the step cap (then the best finished name so far is the answer)"""
        import heapq as _hq
        if not prompts: return []
        inv_=[n_ for n_ in (_INFM.get('names') or {}) if n_ and all(65<=b_<=90 or 97<=b_<=122 for b_ in n_)]
        if not inv_: _TRIE_ST.update(n=0); return [0]*len(prompts)
        nxt_={}; full_=set(inv_)
        for n_ in inv_:
            for i_ in range(len(n_)): nxt_.setdefault(n_[:i_],set()).add(n_[i_])
        sls_=[slot] if slot>=0 else (sorted(set(int(v_) for v_ in model.__dict__.get('_t2s',{}).values())) or [0])
        P_=[list((_cue_fix(cfg,p_) if _INF.get('corr') else p_).encode('latin-1'))[-(int(cfg.seq_len)-24):] for p_ in prompts]
        NL_=torch.tensor([b_ for b_ in range(256) if not (65<=b_<=90 or 97<=b_<=122)])
        fr_=[[(0.0,b'')] for _ in P_]; best_=[(-1e18,b'') for _ in P_]; done_=[False]*len(P_); cap_=int(getattr(cfg,'inf_eval_trie_steps',24)); st9_=0
        while not all(done_) and st9_<cap_:
            st9_+=1; rows_=[]
            for i_ in range(len(P_)):
                if done_[i_]: continue
                for _ in range(min(K_,len(fr_[i_]))):
                    ns_,pf_=_hq.heappop(fr_[i_]); rows_.append((i_,pf_,-ns_))
            R_=[P_[i_]+list(pf_) for i_,pf_,_ in rows_]; L_=max(len(r_) for r_ in R_); lp9_=None
            for a0_ in range(0,len(R_),256):
                RR_=R_[a0_:a0_+256]; X_=torch.zeros(len(RR_),L_,dtype=torch.long,device=dev_)
                for j_,r_ in enumerate(RR_): X_[j_,:len(r_)]=torch.tensor(r_,device=dev_)
                ix_=torch.tensor([len(r_)-1 for r_ in RR_],device=dev_); lc_=None
                for sl9_ in sls_:
                    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl9_; sv6_=_INFM['on']; _INFM['on']=bool(_INF.get('corr',False))
                    try: l_=F.log_softmax(model(X_,sl9_).float()[torch.arange(len(RR_),device=dev_),ix_],-1)
                    finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=sv6_
                    lc_=l_ if lc_ is None else torch.logaddexp(lc_,l_)
                lc_=(lc_-math.log(len(sls_))).cpu(); lp9_=lc_ if lp9_ is None else torch.cat([lp9_,lc_])
            for j_,(i_,pf_,sc_) in enumerate(rows_):
                if pf_ in full_:
                    f9_=sc_+float(torch.logsumexp(lp9_[j_][NL_],0))
                    if f9_>best_[i_][0]: best_[i_]=(f9_,pf_)
                for b_ in nxt_.get(pf_,()): _hq.heappush(fr_[i_],(-(sc_+float(lp9_[j_][b_])),pf_+bytes([b_])))
            for i_ in range(len(P_)):
                if not done_[i_] and (not fr_[i_] or -fr_[i_][0][0]<=best_[i_][0]): done_[i_]=True
        tb_=[t_.encode('latin-1') for t_ in targets]
        _TRIE_ST['n']=len(inv_); _TRIE_ST.setdefault('inv',[]).extend(int(t_ in full_) for t_ in tb_); _TRIE_ST.setdefault('cut',[]).extend(int(not d_) for d_ in done_)
        _TRIE_ST['steps']=max(_TRIE_ST.get('steps',0),st9_)
        return [int(b9_[1]==t_) for b9_,t_ in zip(best_,tb_)]
    def namescore(p_,names_,slot,mode_='sum'):
        """NAME SCORE of each candidate name after prompt p_: log p(name bytes) + log p(any non-letter next) (the beam's scoring), mixed
        over slots; mode 'byte' divides by the name's length + 1"""
        if _INF.get('corr'): p_=_cue_fix(cfg,p_)
        isl_=lambda b_: 65<=b_<=90 or 97<=b_<=122; NL_=torch.tensor([b_ for b_ in range(256) if not isl_(b_)],device=dev_)
        seqs=[(p_+c_).encode('latin-1')[-(int(cfg.seq_len)-1):] for c_ in names_]; L_=max(len(s_) for s_ in seqs)+1
        X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
        for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
        n0_=[len(s_)-len(c_.encode('latin-1')) for s_,c_ in zip(seqs,names_)]
        sls_=[slot] if slot>=0 else (sorted(set(int(v_) for v_ in model.__dict__.get('_t2s',{}).values())) or [0]); tot_=None
        for sl9_ in sls_:
            sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl9_; sv6_=_INFM['on']; _INFM['on']=bool(_INF.get('corr',False))
            try: lp_=F.log_softmax(model(X_,sl9_).float(),-1)
            finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=sv6_
            s9_=np.array([float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum())+float(torch.logsumexp(lp_[j_,len(seqs[j_])-1,NL_],0)) for j_ in range(len(seqs))])
            tot_=s9_ if tot_ is None else np.logaddexp(tot_,s9_)
        tot_=tot_-math.log(len(sls_))
        return tot_/np.array([len(c_.encode('latin-1'))+1 for c_ in names_]) if mode_=='byte' else tot_
    def misslog(kind_,items_,greedy_hits_,beam_hits_):
        """RANK-1 SHARE and PER-MISS LOG (inf_eval_open_diag): for every greedy miss, all 200 probe names scored as whole names (the beam's
        scoring, summed; and per byte); rank-1 share against beam gain; the top wrong name, whether it answers another probe subject with the
        same relation (interference), and its length against the true name (length bias)"""
        ch9_=inf_probes(cfg); obj_={}
        for c9_ in ch9_:                                                              # answers by relation, from the PROBE SET (not the store)
            n9_,r9_=c9_['names'],c9_['rels']
            obj_.setdefault(('one-hop',r9_[0]),set()).add(n9_[1])
            if len(n9_)>=3: obj_.setdefault(('two-hop',r9_[0],r9_[1]),set()).add(n9_[2]); obj_.setdefault(('two-hop-any',),set()).add(n9_[2])
        rr9_=np.random.RandomState(4321); out_={'sum':[], 'byte':[]}; ex_=[]
        for (p_,t_,sl,key_),gh_ in zip(items_,greedy_hits_):
            if gh_: continue
            oth_=[x_ for x_ in rr9_.choice(alln_,min(len(alln_),260),replace=False) if x_!=t_][:199]; nm_=[t_]+oth_
            ss_=namescore(p_,nm_,sl,'sum'); nb9_=np.array([len(c_.encode('latin-1'))+1 for c_ in nm_],dtype=np.float64)
            for md_ in ('sum','byte'):
                sc_=ss_ if md_=='sum' else ss_/nb9_; o_=np.argsort(-sc_); r1_=int(o_[0]==0); tw_=nm_[int(o_[0] if o_[0]!=0 else o_[1])]
                comp_=tw_ in obj_.get(key_,set())-{t_}; comp_any_=(kind_=='two-hop' and tw_ in obj_.get(('two-hop-any',),set())-{t_})
                dl_=len(tw_.encode('latin-1'))-len(t_.encode('latin-1'))
                out_[md_].append((r1_,comp_,comp_any_,dl_))
                if md_=='sum' and len(ex_)<6: ex_.append(f"{t_} -> {tw_} (competitor {comp_}, length delta {dl_:+d}, rank-1 {bool(r1_)})")
        n_=len(out_['sum']); gain_=float(np.mean(beam_hits_))-float(np.mean(greedy_hits_)) if len(beam_hits_) else float('nan')
        if not n_: return f"no greedy misses (n {len(items_)})"
        f_=lambda md_,k_: float(np.mean([x_[k_] for x_ in out_[md_]]))
        return (f"{n_} greedy misses of {len(items_)} | RANK-1 SHARE (true name first of 200 by the name score) summed {f_('sum',0):.2f}, per byte {f_('byte',0):.2f} "
                f"(as a share of all items: {f_('sum',0)*n_/len(items_):.2f}) vs BEAM GAIN (beam - greedy) {gain_:+.2f} | top wrong name: SAME-RELATION COMPETITOR "
                f"summed {f_('sum',1):.2f}, per byte {f_('byte',1):.2f}"+(f", another chain's C (any relation) summed {f_('sum',2):.2f}" if kind_=='two-hop' else '')
                +f" | SHORTER than the true name summed {np.mean([x_[3]<0 for x_ in out_['sum']]):.2f}, per byte {np.mean([x_[3]<0 for x_ in out_['byte']]):.2f} "
                f"(mean length delta summed {np.mean([x_[3] for x_ in out_['sum']]):+.1f}, per byte {np.mean([x_[3] for x_ in out_['byte']]):+.1f}) || e.g. "+" | ".join(ex_))
    def seqlp(p_,cs_,slot):
        """summed log-probability (nats) of each continuation c + '.' after prompt p_, mixed over slots like score(); and its byte count"""
        if _INF.get('corr'): p_=_cue_fix(cfg,p_)
        seqs=[(p_+c_+'.').encode('latin-1')[-int(cfg.seq_len):] for c_ in cs_]; L_=max(len(s_) for s_ in seqs)
        X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
        for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
        n0_=[len(s_)-len((c_+'.').encode('latin-1')) for s_,c_ in zip(seqs,cs_)]
        sls_=[slot] if slot>=0 else (sorted(set(int(v_) for v_ in model.__dict__.get('_t2s',{}).values())) or [0]); tot_=None
        for sl9_ in sls_:
            sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl9_; sv6_=_INFM['on']; _INFM['on']=bool(_INF.get('corr',False))
            try: lp_=F.log_softmax(model(X_[:,:-1],sl9_).float(),-1)
            finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=sv6_
            s9_=np.array([float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum()) for j_ in range(len(seqs))])
            tot_=s9_ if tot_ is None else np.logaddexp(tot_,s9_)
        return tot_-math.log(len(sls_)), np.array([len(seqs[j_])-n0_[j_] for j_ in range(len(seqs))],dtype=np.float64), lp_
    def opendiag(items):
        """OPEN-ENDED DIAGNOSTIC (inf_eval_open_diag): where does the true name sit in the model's own distribution? first-byte rank at the
        answer position; per-byte log2-probability of the true name and of the greedy output; SEARCH ERRORS (the true name is more probable as
        a whole than what greedy wrote: fixable by search); rank of the true name among 200 probe names (an open list, per byte)"""
        import re as _re9
        fr_=[]; tl_=[]; gl_=[]; se_=[]; rk_=[]; rr9_=np.random.RandomState(4321)
        for p_,t_,slot in items:
            sls_=[slot] if slot>=0 else (sorted(set(int(v_) for v_ in model.__dict__.get('_t2s',{}).values())) or [0])
            b_=list((_cue_fix(cfg,p_) if _INF.get('corr') else p_).encode('latin-1'))[-(int(cfg.seq_len)-16):]; G_=[]; lp0_=None
            for st_ in range(14):
                X_=torch.tensor([b_+G_],dtype=torch.long,device=dev_); lp9_=None
                for sl9_ in sls_:
                    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=sl9_; sv6_=_INFM['on']; _INFM['on']=bool(_INF.get('corr',False))
                    try: l_=F.log_softmax(model(X_,sl9_)[0,-1].float(),-1)
                    finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM['on']=sv6_
                    lp9_=l_ if lp9_ is None else torch.logaddexp(lp9_,l_)
                if st_==0: lp0_=lp9_
                G_.append(int(lp9_.argmax()))
            fr_.append(int((lp0_>lp0_[t_.encode('latin-1')[0]]).sum())+1)
            gs_=bytes(G_).decode('latin-1'); m9_=_re9.match(r'[A-Za-z]+',gs_); gn_=m9_.group(0) if m9_ else gs_[:1]
            oth_=[x_ for x_ in rr9_.choice(alln_,min(len(alln_),260),replace=False) if x_!=t_][:199]
            sm_,nb_,_=seqlp(p_,[t_,gn_]+oth_,slot)
            tl_.append(sm_[0]/nb_[0]/math.log(2)); gl_.append(sm_[1]/nb_[1]/math.log(2)); se_.append(int(gn_!=t_ and sm_[0]>sm_[1]))
            pb_=np.delete(sm_/nb_,1); rk_.append(int((pb_>pb_[0]).sum())+1)
        a_=np.array; fr_,rk_=a_(fr_),a_(rk_)
        return (f"first byte rank median {np.median(fr_):.0f} (top-1 {np.mean(fr_<=1):.2f}, top-3 {np.mean(fr_<=3):.2f}) | log2-probability per byte: true name {np.mean(tl_):.2f}, greedy output {np.mean(gl_):.2f} | "
                f"SEARCH ERRORS (true name more probable than the greedy output) {np.mean(se_):.2f} | rank among 200 probe names median {np.median(rk_):.0f} (top-1 {np.mean(rk_<=1):.2f}, top-10 {np.mean(rk_<=10):.2f})")
    try:
        UPD_=float(getattr(cfg,'inf_upd_frac',0.0))>0
        KINDS_=('one-hop','two-hop','typed two-hop','three-hop')+(('UPDATED one-hop','UPDATED two-hop','NOISE-CONTRADICTED one-hop') if UPD_ else ())+(('reworded two-hop','context two-hop','pronoun two-hop') if (int(getattr(cfg,'inf_eval_hard',0)) and ('END' in tag or _INF.get('corr'))) else ())+(('IN-CONTEXT two-hop','IN-CONTEXT decoy bridge','IN-CONTEXT three-hop') if (int(getattr(cfg,'inf_eval_ctx',0)) and not _INF.get('corr')) else ())+(('CTX2 order AB','CTX2 order BA','CTX2 decoy AB','CTX2 decoy BA') if (int(getattr(cfg,'inf_eval_ctx2',0)) and not _INF.get('corr')) else ())
        if float(getattr(cfg,'inf_para_frac',0.0))>0: KINDS_=KINDS_+('PARAPHRASED one-hop (unseen wording)','SEEN-WORDING one-hop (para probes)')
        if int(getattr(cfg,'inf_clean',0)): KINDS_=KINDS_+('HELD-OUT two-hop','HELD-OUT typed two-hop','HELD-OUT three-hop')   # unwritten wordings, bridges as candidates, no slot given
        act_={}; aw9_={}
        vd9_=S_.get('_vdom') if (int(getattr(cfg,'inf_upd_revisited',1)) and S_.get('_vdom_from',-1)==0) else None   # READOUT FIX: changed facts count only after a revisit
        for kind in KINDS_:
            pr_=[];cd_=[];sl_=[];bn_=[];rk_=[]; _INFM['dbg']=[0,0,0.0]
            for c_ in ch_:
                n_=c_['names']; r_=c_['rels']
                if kind in ('three-hop','IN-CONTEXT three-hop','HELD-OUT three-hop') and len(n_)<4: continue
                if kind.startswith(('PARAPHRASED','SEEN-WORDING')) and (not c_.get('wd') or (kind.startswith('PARAPHRASED') and c_['wd'][0]=='both')): continue
                u0_=c_.get('upd')
                if kind.startswith('UPDATED') and not (u0_ and u0_['type']=='true'): continue
                if kind.startswith('NOISE') and not (u0_ and u0_['type']=='noise'): continue
                if UPD_ and u0_ and u0_['type']=='true' and not kind.startswith('UPDATED'): continue   # their old answers are no longer true
                if vd9_ is not None and kind.startswith(('UPDATED','NOISE')) and u0_ and u0_.get('new'):
                    nd9_=[int(dd_) for dd_,_ in u0_['new']][:(2 if kind=='UPDATED two-hop' else 1)]
                    if any(vd9_.get(dd_,0)<2 for dd_ in nd9_): aw9_[kind]=aw9_.get(kind,0)+1; continue   # its new fact not yet read: the old answer is still right
                if kind.startswith(('IN-CONTEXT','CTX2')) and str(getattr(cfg,'inf_xfer','none'))=='skill' and n_[0] not in _inf_heldout(cfg): continue
                if kind.startswith('CTX2') and len(n_)!=3: continue   # skill runs: in-context probes on HELD-OUT chains only
                fs_=[f_ for _,f_ in c_['facts']]; oth0_=[x_ for x_ in alln_ if x_ not in n_]
                tgt={'one-hop':n_[1],'three-hop':n_[-1],'IN-CONTEXT three-hop':n_[-1],'HELD-OUT three-hop':n_[-1],'UPDATED one-hop':(u0_ or {}).get('B2'),'UPDATED two-hop':(u0_ or {}).get('C2'),'NOISE-CONTRADICTED one-hop':n_[1]}.get(kind,n_[2])
                pp={'one-hop':f"{n_[0]} {r_[0]} ",'two-hop':f"So {n_[0]} is connected to ",'typed two-hop':f"{n_[0]} {r_[0]} someone who {r_[1]} ",'three-hop':f"So {n_[0]} is connected to ",
                    'reworded two-hop':f"{n_[0]}, it turns out, is connected to ",'context two-hop':f"The weather was mild that year. So {n_[0]} is connected to ",
                    'pronoun two-hop':f"Consider {n_[0]}. They are connected to ",
                    'IN-CONTEXT two-hop':f"{fs_[0]} {fs_[1]} So {n_[0]} is connected to ",                       # S1: the two source facts in the context
                    'IN-CONTEXT decoy bridge':f"{fs_[0].replace(n_[1],oth0_[len(pr_)%len(oth0_)])} {fs_[1]} So {n_[0]} is connected to ",   # chain broken at the bridge: C still in context
                    'IN-CONTEXT three-hop':(' '.join(fs_[:3])+f" So {n_[0]} is connected to "),
                    'UPDATED one-hop':f"{n_[0]} {r_[0]} ",'UPDATED two-hop':f"So {n_[0]} is connected to ",'NOISE-CONTRADICTED one-hop':f"{n_[0]} {r_[0]} ",
                    'HELD-OUT two-hop':_HO_TWO[len(pr_)%len(_HO_TWO)].format(A=n_[0]),'HELD-OUT three-hop':_HO_TWO[len(pr_)%len(_HO_TWO)].format(A=n_[0]),
                    'HELD-OUT typed two-hop':_HO_TYPED.format(A=n_[0],r0=r_[0],r1=(r_[1] if len(r_)>1 else r_[0]))}.get(kind,'')
                if kind.startswith(('PARAPHRASED','SEEN-WORDING')):
                    pz_=_INF_PARA[r_[0]]; seen_para_=c_['wd'][0]=='para'
                    pp=f"{n_[0]} {(r_[0] if seen_para_ else pz_) if kind.startswith('PARAPHRASED') else (pz_ if seen_para_ else r_[0])} "; tgt=n_[1]
                if kind.startswith('CTX2'):                                     # SHARPER IN-CONTEXT TEST: a distractor fact LAST (the answer is not the last name),
                    X0_,Y0_=oth0_[(3*len(pr_))%len(oth0_)],oth0_[(3*len(pr_)+1)%len(oth0_)]   #   both sentence orders, in-context names among the candidates
                    fx_=f"{X0_} {r_[1]} {Y0_}."; f0_=fs_[0] if 'decoy' not in kind else fs_[0].replace(n_[1],oth0_[(3*len(pr_)+2)%len(oth0_)])
                    pp=(f"{f0_} {fs_[1]} {fx_} " if kind.endswith('AB') else f"{fs_[1]} {f0_} {fx_} ")+f"So {n_[0]} is connected to "
                oth=[x_ for x_ in alln_ if x_ not in n_]; cs_=[tgt]+list(rs_.choice(oth,19,replace=False))
                if kind=='UPDATED one-hop': cs_=[tgt,n_[1]]+cs_[1:19]                 # the OLD answer is a candidate: picking it = a stale belief
                elif kind=='UPDATED two-hop': cs_=[tgt,n_[2]]+cs_[1:19]
                elif kind=='NOISE-CONTRADICTED one-hop': cs_=[tgt,u0_['B3']]+cs_[1:19]  # the single false contradiction is a candidate: picking it = a wrong flip
                elif kind.startswith('CTX2'): ic_=[n_[1],X0_,Y0_]; cs_=[tgt]+ic_+[x_ for x_ in cs_[1:] if x_ not in ic_][:16]   # in-context names compete with the answer
                elif kind.startswith('HELD-OUT'): br_=[x_ for x_ in n_[1:] if x_!=tgt]; cs_=[tgt]+br_+cs_[1:20-len(br_)]   # the BRIDGE(S) are candidates too
                rk_.append(('one-hop',r_[0]) if kind=='one-hop' else (('two-hop',r_[0],r_[1]) if len(r_)>1 else ('two-hop',r_[0])))
                pr_.append(pp); cd_.append(cs_); bn_.append(n_[1]); sl_.append(-1 if kind.startswith('HELD-OUT') else int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0])))
            hits=[]
            for sl in sorted(set(sl_)):
                ix_=[i_ for i_,s_ in enumerate(sl_) if s_==sl]
                if kind=='two-hop' and _INF.get('corr'): _INF['cal_on']=[(pr_[i_].split(' ')[1],cd_[i_][0],bn_[i_]) for i_ in ix_]; _INF['calrec']=[]
                hh_=score([pr_[i_] for i_ in ix_],[cd_[i_] for i_ in ix_],sl); hits+=[int(h_==0) for h_ in hh_]
                if kind.startswith(('UPDATED','NOISE')): _INF.setdefault('stale',{}).setdefault(kind,[]).extend([int(h_==1) for h_ in hh_])
                if kind.startswith('CTX2'): _INF.setdefault('copy',{}).setdefault(kind,[]).extend([int(1<=h_<=3) for h_ in hh_])
                if kind.startswith('HELD-OUT'): _INF.setdefault('brg',{}).setdefault(kind,[]).extend([int(1<=h_<=(2 if 'three' in kind else 1)) for h_ in hh_])
                if _INF.get('cal_on') is not None and kind=='two-hop' and _INF.get('corr'): _INF.setdefault('calrec_all',[]).extend(_INF['calrec']); _INF['cal_on']=None
            res[kind]=(float(np.mean(hits)) if hits else float('nan'),len(hits)); d_=_INFM.get('dbg',[0,0,0.0]); act_[kind]=d_[2]/max(1,d_[0])
            if int(getattr(cfg,'inf_eval_open',0)) and kind in ('one-hop','two-hop','HELD-OUT two-hop','HELD-OUT typed two-hop') and pr_:
                def byitem_(fn_,*a_):
                    h9_=[0]*len(pr_)
                    for sl in sorted(set(sl_)):
                        ix_=[i_ for i_,s_ in enumerate(sl_) if s_==sl]
                        for i_,v_ in zip(ix_,fn_([pr_[i_] for i_ in ix_],[cd_[i_][0] for i_ in ix_],sl,*a_)): h9_[i_]=v_
                    return h9_
                oh_=byitem_(openh); res[f"OPEN {kind} (greedy, no candidates)"]=(float(np.mean(oh_)),len(oh_)); bh_=[]
                if int(getattr(cfg,'inf_eval_beam',0)):
                    bh_=byitem_(beamh,int(getattr(cfg,'inf_eval_beam',8)))
                    res[f"BEAM {kind} (width {int(getattr(cfg,'inf_eval_beam',8))}{', first step '+str(int(getattr(cfg,'inf_eval_beam_first',0))) if int(getattr(cfg,'inf_eval_beam_first',0))>int(getattr(cfg,'inf_eval_beam',8)) else ''}, no candidates)"]=(float(np.mean(bh_)),len(bh_))
                if int(getattr(cfg,'inf_eval_trie',0)) and (int(getattr(cfg,'inf_eval_trie',0))>=2 or 'END' in tag or 'PASS' in tag or int(model.__dict__.get('_visit',0))%int(getattr(cfg,'inf_eval_missevery',10))==0):
                    try:
                        _TRIE_ST.clear(); th_=byitem_(trieh,int(getattr(cfg,'inf_eval_trie_k',32)))
                        res[f"KNOWN-NAME {kind} (decoding restricted to the {_TRIE_ST.get('n',0)} names in the store, exact; target in the store {np.mean(_TRIE_ST.get('inv',[0])):.2f}, search cut {np.mean(_TRIE_ST.get('cut',[0])):.2f})"]=(float(np.mean(th_)),len(th_))
                    except Exception as e9_: _fail('inference.evaluate:trie'); print(f"  [inference KNOWN-NAME DECODING] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
                dgm_=int(getattr(cfg,'inf_eval_open_diag',0))
                if dgm_ and kind in ('one-hop','two-hop') and (dgm_>=2 or 'END' in tag or 'PASS' in tag or int(model.__dict__.get('_visit',0))%int(getattr(cfg,'inf_eval_missevery',10))==0):
                    try:
                        print(f"  [inference MISS LOG | {tag}] {kind} (slot given, from {'the weights + store' if _INF.get('corr') else 'the weights'}): "
                              +misslog(kind,[(pr_[i_],cd_[i_][0],sl_[i_],rk_[i_]) for i_ in range(len(pr_))],oh_,bh_),flush=True)
                    except Exception as e9_: _fail('inference.evaluate:misslog'); print(f"  [inference MISS LOG] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
                dg9_=int(getattr(cfg,'inf_eval_open_diag',0))
                if dg9_ and kind in ('one-hop','HELD-OUT two-hop') and (dg9_>=2 or 'END' in tag):
                    try:
                        print(f"  [inference OPEN-ENDED DIAGNOSTIC | {tag}] {kind} (n {min(40,len(pr_))}"
                              +(", no slot given" if kind.startswith('HELD-OUT') else ", slot given")+"): "+opendiag([(pr_[i_],cd_[i_][0],sl_[i_]) for i_ in range(min(40,len(pr_)))]),flush=True)
                    except Exception as e9_: _fail('inference.evaluate:opendiag'); print(f"  [inference OPEN-ENDED DIAGNOSTIC] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
            if kind.startswith('CTX2'): cp0_=_INF['copy'].pop(kind,[]); res[kind+f" (picked another in-context name {np.mean(cp0_) if cp0_ else float('nan'):.2f})"]=res.pop(kind)
            if kind.startswith('HELD-OUT'): bg0_=_INF.setdefault('brg',{}).pop(kind,[]); res[kind+f" (picked a bridge {np.mean(bg0_) if bg0_ else float('nan'):.2f})"]=res.pop(kind)
            if kind.startswith(('UPDATED','NOISE')): st0_=_INF.setdefault('stale',{}).pop(kind,[]); res[kind+(' (STALE: old answer preferred' if kind.startswith('UPDATED') else ' (WRONG FLIP: false contradiction preferred')+f" {np.mean(st0_) if st0_ else float('nan'):.2f})"]=res.pop(kind)
        if aw9_ and not _INF.get('corr'): print("  [inference eval | changed facts] scored only once the domain of the new fact has been revisited - awaiting a revisit: "+", ".join(f"{k_} {v_}" for k_,v_ in aw9_.items()),flush=True)
        if _INF.get('corr') and int(getattr(cfg,'inf_eval_hard',0)) and 'END' in tag:   # FALSE HITS: invented names never seen anywhere
            rsf_=np.random.RandomState(555); fn_=[]
            while len(fn_)<40:
                n0_=''.join(rsf_.choice(['xa','qe','wu','yo','zi','vex','kru','plo','gne','fyr'],rsf_.randint(2,4))); n0_=n0_[0].upper()+n0_[1:]
                if 5<=len(n0_)<=10 and n0_ not in fn_: fn_.append(n0_)
            _INFM['dbg']=[0,0,0.0]; score([f"So {x_} is connected to " for x_ in fn_],[list(rs_.choice(alln_,20,replace=False)) for _ in fn_],int(model.__dict__.get('_t2s',{}).get(0,0)))
            d_=_INFM['dbg']; act_['FALSE-HIT unknown names']=d_[2]/max(1,d_[0])
        if 'END' in tag and not _INF.get('corr') and S_['conc']:              # TEMPORAL GRADIENT: weights-only recall of found chains, by when their conclusion was made
            vm_={}
            for c0_ in S_['conc']: vm_.setdefault(frozenset((c0_['A'],c0_['C'])),c0_.get('v',0))
            rows_=[]
            for c_ in ch_:
                n_=c_['names']
                if len(n_)<3 or (c_.get('upd') or {}).get('type')=='true': continue
                k0_=frozenset((n_[0],n_[2]))
                if k0_ not in vm_: continue
                oth=[x_ for x_ in alln_ if x_ not in n_]; rows_.append((vm_[k0_],f"So {n_[0]} is connected to ",[n_[2]]+list(rs_.choice(oth,19,replace=False)),int(model.__dict__.get('_t2s',{}).get(c_['doms'][0],c_['doms'][0]))))
            if rows_:
                vs_=sorted(r_[0] for r_ in rows_); cut_=vs_[len(vs_)//2]; res_tg_={}
                for nm0_,sel_ in (('OLDER half',lambda v_: v_<cut_),('RECENT half',lambda v_: v_>=cut_)):
                    rr_=[r_ for r_ in rows_ if sel_(r_[0])]; hh_=[]
                    for sl in sorted(set(r_[3] for r_ in rr_)):
                        q_=[r_ for r_ in rr_ if r_[3]==sl]; hh_+=[int(h_==0) for h_ in score([r_[1] for r_ in q_],[r_[2] for r_ in q_],sl)]
                    res_tg_[nm0_]=(float(np.mean(hh_)) if hh_ else float('nan'),len(hh_))
                print(f"  [inference temporal gradient | {tag}] two-hop recall FROM THE WEIGHTS ALONE (store off) by conclusion age (split at visit {cut_}): "+" | ".join(f"{k_} {v_[0]:.2f} (n {v_[1]})" for k_,v_ in res_tg_.items())+" | CLS predicts OLDER > RECENT once transfer works",flush=True)
        tt_=_inf_truth(cfg,current=False); up_={(c0_['names'][0],c0_['upd']['C2']) for c0_ in inf_probes(cfg) if c0_.get('upd',{}).get('type')=='true'}   # FIX: the NEW chain of an
        wr_=[c_ for c_ in S_['conc'] if (c_['A'],c_['C']) not in tt_[0]|tt_[1]|up_ and any(c_['A']==q_['names'][0] for q_ in ch_)]   #   updated fact is not a false memory
        fm_=float('nan')
        if wr_:
            pm_={q_['names'][0]:q_ for q_ in ch_}; pp_=[];cc_=[];ss_=[]
            for c_ in wr_[:60]:
                q_=pm_[c_['A']]; pp_.append(f"So {c_['A']} is connected to "); cc_.append([c_['C'],q_['names'][2]]); ss_.append(int(model.__dict__.get('_t2s',{}).get(q_['doms'][0],q_['doms'][0])))
            fh_=[]
            for sl in sorted(set(ss_)):
                ix_=[i_ for i_,s_ in enumerate(ss_) if s_==sl]; fh_+=[int(h_==0) for h_ in score([pp_[i_] for i_ in ix_],[cc_[i_] for i_ in ix_],sl)]
            fm_=float(np.mean(fh_))
            if int(getattr(cfg,'inf_fm_debug',0)):                              # FALSE-MEMORY DIAGNOSTIC: which wrong conclusions win, and why they exist
                ord_=[i_ for sl in sorted(set(ss_)) for i_,s_ in enumerate(ss_) if s_==sl]; win_={id(wr_[i_]):h_ for i_,h_ in zip(ord_,fh_)}
                def _cls(c_):
                    sa_=[S_['it'][i_] for i_ in c_.get('src',()) if isinstance(i_,int) and i_<len(S_['it'])]
                    return ('freed' if c_.get('freed') else 'live')+('/stale-source' if any(x_.get('sup') for x_ in sa_) else '')+('/typed' if c_.get('typed') else '/untyped')+(f"/dep{c_.get('dep',1)}")
                agg_={}
                for c_ in wr_[:60]:
                    k_=_cls(c_); a_=agg_.setdefault(k_,[0,0]); a_[0]+=1; a_[1]+=win_.get(id(c_),0)
                ex_=[f"{c_['A']}->{c_['C']} via {c_.get('B')} [{_cls(c_)}] wins={win_.get(id(c_),0)} :: "+' + '.join(repr(S_['it'][i_]['s'][:70]) for i_ in c_.get('src',()) if isinstance(i_,int) and i_<len(S_['it'])) for c_ in wr_[:8]]
                print(f"  [inference FALSE-MEMORY DIAGNOSTIC | {tag}] by kind (n, preferred over truth): "+' | '.join(f"{k_} {v_[0]},{v_[1]}" for k_,v_ in sorted(agg_.items()))+" || "+' || '.join(ex_),flush=True)
    finally: model.train(was_)
    print(f"  [inference eval | {tag}] probe recall {'WEIGHTS + FAST STORE' if _INF.get('corr') else 'FROM THE WEIGHTS'} (chance 0.05): "+" | ".join(f"{k_} {v_[0]:.2f} (n {v_[1]})" for k_,v_ in res.items())
          +(f" | false memory: wrong probe conclusions preferred over the truth in {fm_:.2f} of {min(60,len(wr_))}" if wr_ else " | no wrong probe conclusions"),flush=True)
    cr_ok_=list(_INF.get('calrec_all') or []) if _INF.get('corr') else []
    if _INF.get('corr') and _INF.get('calrec_all'):                               # CALIBRATION: do the 'knows what it knows' signals predict a correct answer?
        cr_=_INF.pop('calrec_all'); ok_=np.array([r_[0] for r_ in cr_]); cf_=np.array([r_[1] for r_ in cr_]); hit_=np.array([r_[3] for r_ in cr_])
        def auc_(sg_,y_):
            P_=sg_[y_==1]; N_=sg_[y_==0]
            return float(np.mean([(p0_>n0_)+0.5*(p0_==n0_) for p0_ in P_ for n0_ in N_])) if len(P_) and len(N_) else float('nan')
        print(f"  [inference eval | calibration, two-hop with store] accuracy {ok_.mean():.2f} | confidence (top probability) AUC {auc_(cf_,ok_):.2f}, mean when right {cf_[ok_==1].mean() if ok_.any() else float('nan'):.2f} / wrong {cf_[ok_==0].mean() if (~ok_.astype(bool)).any() else float('nan'):.2f} | "
              f"store hit (subject in the store) AUC {auc_(hit_.astype(float),ok_):.2f}, accuracy with hit {ok_[hit_==1].mean() if (hit_==1).any() else float('nan'):.2f} (n {int(hit_.sum())}) / without {ok_[hit_==0].mean() if (hit_==0).any() else float('nan'):.2f} (n {int((hit_==0).sum())})",flush=True)
    if _INF.get('corr') and cr_ok_:
        _inf_abstain_sources(S_,cr_ok_,tag)
    if _INF.get('corr') and int(getattr(cfg,'inf_eval_hard',0)): print("  [inference eval | store activity] mean max |logit correction| per probe position: "+" | ".join(f"{k_} {v_:.3f}" for k_,v_ in act_.items()),flush=True)
    return res
def _inf_lps(model,cfg,prompt_,cands_,slot_,store_,dec_=False,raw_=False):
    """cg26g: summed log-probability of each candidate completion (as _inf_score1, which returns only the argmax)"""
    seqs=[(prompt_+c_+'.').encode('latin-1')[-max(int(cfg.seq_len),_SCORE_LEN['n']):] for c_ in cands_]; L_=max(len(s_) for s_ in seqs); dev_=model.embed.weight.device
    X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
    for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot_; f0_=_INFM.get('force','__none__'); _INFM['force']=bool(store_); was_=model.training; model.eval()
    try:
        with torch.no_grad(): lp_=F.log_softmax(model(X_[:,:-1],slot_).float(),-1)
    finally:
        _carry_load(sv4_); _XDG['cur']=sc_; model.train(was_)
        if f0_=='__none__': _INFM.pop('force',None)
        else: _INFM['force']=f0_
    n0_=[len(s_)-len((c_+'.').encode()) for s_,c_ in zip(seqs,cands_)]
    sm_=np.array([float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum()) for j_ in range(len(seqs))])
    return _score_norm(cfg,sm_,np.array([len(seqs[j_])-n0_[j_] for j_ in range(len(seqs))],dtype=np.float64),dec_,raw_)
def _cls_measure(model,cfg,S_slot):
    """cg26g CLS MEASUREMENT, every sleep whatever the dream mode (so the no-replay and random-dream controls report the same numbers):
    TEMPORAL GRADIENT on an unbiased random sample - two-hop with the store OFF vs ON by age of the conclusion (visits since first seen);
    STORE HANDOVER - conclusions tracked across sleeps; one answered with the store off, with a margin, at 3 sleeps running is RELEASED
    (inf_release: erased from the fast store at the next store write, kept in trunk replay); a released one that fails store-off at 2 sleeps
    running is RESTORED (written back) - a small form of reconsolidation"""
    if not int(getattr(cfg,'inf_tgrad',0)): return
    S_=_inf_store(model)
    if int(getattr(cfg,'inf_delete',0)) and S_.get('conc'):                    # cg26h PURGE: a deleted conclusion whose side links (link graph, LM side) have been given back
        n0_=len(S_['conc']); S_['conc']=[c_ for c_ in S_['conc'] if not (c_.get('deleted') is not None and not c_.get('lms') and not c_.get('w9'))]   #   leaves the store
        if n0_>len(S_['conc']): st0_=S_.setdefault('_relstat',{'rel':0,'res':0}); st0_['purg']=st0_.get('purg',0)+n0_-len(S_['conc'])
    allc_=S_.get('conc',[]); lv_=[c_ for c_ in allc_ if not c_.get('freed') and isinstance(c_.get('C'),str)]
    if len(lv_)<5: return
    vis_=int(model.__dict__.get('_visit',0)); rs_=np.random.RandomState(5151+vis_); Cp_=[c_['C'] for c_ in lv_]; mg_=float(getattr(cfg,'inf_release_margin',1.0))
    for c_ in lv_: c_.setdefault('bv',vis_)
    def sc_(c_,on_):
        o_=[x_ for x_ in Cp_ if x_!=c_['C']]; rs_.shuffle(o_)
        if len(o_)<4: return None
        lp_=_inf_lps(model,cfg,f"So {c_['A']} is connected to ",[c_['C']]+o_[:9],int(S_slot(c_['d'])),on_,dec_=True)   # (release / restore decision)
        return bool(int(np.argmax(lp_))==0), float(lp_[0]-np.max(lp_[1:]))
    bk_={'this visit':[],'1 back':[],'2-3 back':[],'4+ back':[]}
    for c_ in lv_:
        a_=vis_-int(c_['bv']); bk_['this visit' if a_<=0 else '1 back' if a_==1 else '2-3 back' if a_<=3 else '4+ back'].append(c_)
    nk_=int(getattr(cfg,'inf_tgrad_n',24))//(2 if model.__dict__.get('_mem_tight') else 1); out_=[]; seen_={}; obs9_={}
    for k_,L_ in bk_.items():
        ch_=[L_[int(i_)] for i_ in rs_.choice(len(L_),min(len(L_),nk_),replace=False)] if L_ else []; off_=[];on_=[]
        for c_ in ch_:
            r0_=sc_(c_,False); r1_=sc_(c_,True)
            if r0_ is None: continue
            off_.append(r0_[0]); on_.append(r1_[0]); seen_[id(c_)]=r0_; obs9_[id(c_)]=(c_,r0_)
            if not int(getattr(cfg,'inf_release_test',0)):                      # FIX (review): every test this sleep enters the conclusion's history
                H0_=c_.setdefault('so',[])
                if not H0_ or H0_[-1][0]!=vis_: H0_.append((vis_,bool(r0_[0] and r0_[1]>=mg_),r0_[1]))
        mo_=np.mean(off_) if off_ else float('nan'); mn_=np.mean(on_) if on_ else float('nan')
        out_.append(f"{k_} (n {len(off_)} of {len(L_)}): off {mo_:.2f} / on {mn_:.2f} (store dependence {mn_-mo_:+.2f})")
    print("  [inference CLS | TEMPORAL GRADIENT] two-hop from the weights (store OFF) / with the store (ON), by age of the conclusion (random sample): "+" | ".join(out_),flush=True)
    TT_=int(getattr(cfg,'inf_tgrad_track',64)); lt_=lambda c_: c_['so'][-1][0] if c_.get('so') else -1   # FIX (review): tracking rotates - released ones by oldest test
    rel_=sorted([c_ for c_ in lv_ if c_.get('released')],key=lt_)[:max(1,TT_//2)]                         #   (so all can be restored), candidates on a passing run
    cand_=[c_ for c_ in lv_ if not c_.get('released') and ((c_.get('so') and c_['so'][-1][1] and c_['so'][-1][0]>=vis_-1) or (int(getattr(cfg,'inf_rem_exam_act',0)) and c_.get('ex') and c_['ex'][-1][0]==vis_ and c_['ex'][-1][1]=='right'))]   # cg26l: + REM exam RIGHT   #   continue every sleep, the rest
    rest_=sorted([c_ for c_ in lv_ if not c_.get('released') and id(c_) not in {id(x_) for x_ in cand_}],key=lt_)   #   by staleness (never-tested
    trk_=rel_+cand_[:max(0,TT_-len(rel_))]; trk_+=rest_[:max(0,TT_-len(trk_))]                                   #   and failed ones are re-tested)
    nrel_=0; nres_=0; rok_=[]; nhl_=0; ndc_=0; nfx9_=0
    th9_=model.__dict__.get('_hl_th'); cal9_=S_.get('_hl_cal',[])[-2:]; df9_=S_.get('df',{}); mtb9_=float(model.__dict__.get('_hl_mtb',2.0)); hz_=float(getattr(cfg,'inf_release_horizon',10.0))
    hlc_=bool(int(getattr(cfg,'inf_release_hl',0)) and th9_ is not None and ((len(cal9_)==2 and all(n0_>=10 and m0_<k0_ for _,m0_,k0_,n0_ in cal9_)) or int(getattr(cfg,'inf_hl_force_cal',0))))   # cg26h: decay model calibrated (cg26l: or forced, local test only)
    pn9_=float(getattr(cfg,'inf_release_pnext',0.0)); npn9_=0
    pq9_=float(getattr(cfg,'inf_release_pnext_q',0.0)); pmn9_=float(getattr(cfg,'inf_release_pnext_min',0.2)); thq9_=None; sdq9_=float('nan'); npq9_=0   # cg26l FIX: RELATIVE predicted recall
    if pq9_>0 and hlc_:                                                         #   (GPU half-lives ~0.3 visits: an absolute 0.8 bar never fires; rank within the pool instead)
        try:
            pr9_=[]; lh9_=[]
            for c7_ in trk_:
                if c7_.get('released'): continue
                h7_=float(2.0**(torch.tensor(_hl_x(c7_,vis_+1,df9_,mtb9_))@th9_)); lh9_.append(float(np.log2(max(h7_,1e-6))))
                pr9_.append(2.0**(-max(1,vis_+1-int(c7_.get('lr',c7_.get('bv',vis_))))/max(h7_,1e-3)))
            npq9_=len(pr9_)
            if npq9_>=10:
                sdq9_=float(np.std(lh9_))
                if sdq9_>=0.1: thq9_=max(float(np.quantile(pr9_,1.0-pq9_)),2.0*float(np.median(pr9_)),pmn9_)
        except Exception as e7_: _fail('inference.evaluate:317'); print(f"  [inference CLS | RELEASE TIMING] relative predicted-recall rule unavailable ({type(e7_).__name__}: {e7_})"[:200],flush=True)
    G9_=int(getattr(cfg,'inf_delete_grace',2)); dl_on_=bool(int(getattr(cfg,'inf_delete',0)) and int(getattr(cfg,'inf_release',0)))
    for c_ in trk_:
        tst_=int(getattr(cfg,'inf_release_test',0))                            # (local test only: judge with the store ON, so the release / restore path runs)
        r0_=sc_(c_,True) if tst_ else (seen_.get(id(c_)) or sc_(c_,False))
        if r0_ is None: continue
        if not tst_: obs9_.setdefault(id(c_),(c_,r0_))
        H_=c_.setdefault('so',[])
        if not H_ or H_[-1][0]!=vis_: H_.append((vis_,bool(r0_[0] and r0_[1]>=mg_),r0_[1]))
        if c_.get('released'):
            rok_.append(H_[-1][1])
            if len(H_)>=2 and not H_[-1][1] and not H_[-2][1] and H_[-2][0]>=c_.get('relv',vis_) and int(getattr(cfg,'inf_release',0)): c_['restore_pending']=True; nres_+=1
            elif dl_on_:                                                        # cg26h DELETE AFTER A GRACE PERIOD: released, then answered store-off with the margin at
                Hp_=[h_ for h_ in H_ if h_[0]>c_.get('relv',vis_)]              #   G consecutive sleeps after the release -> the record is deleted (the weights hold it alone)
                if len(Hp_)>=G9_ and all(h_[1] for h_ in Hp_[-G9_:]) and [h_[0] for h_ in Hp_[-G9_:]]==list(range(vis_-G9_+1,vis_+1)): c_['del_now']=True; ndc_+=1
        else:                                                                   # cg26k EITHER RULE RELEASES: the decay-model timing (once calibrated) OR the fixed rule of
            if (pn9_>0 or thq9_ is not None) and hlc_ and H_[-1][0]==vis_ and H_[-1][1]:   # cg26l RELEASE BY PREDICTED RECALL at the next visit (not by a 10-visit half-life)
                h8_=float(2.0**(torch.tensor(_hl_x(c_,vis_+1,df9_,mtb9_))@th9_)); pnx_=2.0**(-max(1,vis_+1-int(c_.get('lr',c_.get('bv',vis_))))/max(h8_,1e-3))
                hl9_=bool((pn9_>0 and pnx_>=pn9_) or (thq9_ is not None and pnx_>=thq9_)); npn9_+=int(hl9_)
            else: hl9_=bool(hlc_ and H_[-1][0]==vis_ and H_[-1][1] and float(2.0**(torch.tensor(_hl_x(c_,vis_+1,df9_,mtb9_))@th9_))>=hz_)   #   consecutive passing sleeps
            fx9_=bool((lambda k_: len(H_)>=k_ and all(h_[1] for h_ in H_[-k_:]) and [h_[0] for h_ in H_[-k_:]]==list(range(vis_-k_+1,vis_+1)))(2 if (int(getattr(cfg,'inf_schema_fast',0)) and c_.get('tbits',9.0)<float(getattr(cfg,'inf_gist_tau',1.0))) else 3))   #   (GPU cg26i: calibrated half-lives ~0.3 visits never reach the 10-visit horizon, and releases stopped)
            nhl_+=int(hl9_); nfx9_+=int(fx9_ and not hl9_)
            if hl9_ or fx9_:
                c_.setdefault('hflag',vis_)
                if int(getattr(cfg,'inf_release',0)) and not c_.get('rel_pending'): c_['rel_pending']=True; nrel_+=1
    if dl_on_:
        st9_=S_.setdefault('_relstat',{'rel':0,'res':0}); GH_=S_.setdefault('ghost',[])
        for c_ in lv_:                                                          # the deletion: freed (leaves trunk replay, its source sentences become evictable, its side links
            if c_.pop('del_now',False):                                         #   are given back at the next inference pass; the fast-store entry was erased at release),
                c_['freed']=True; c_['faded']=True; c_['deleted']=vis_; c_['evicted']='deleted after release'; st9_['dele']=st9_.get('dele',0)+1   #   and the record itself is purged
                GH_.append({'A':c_['A'],'C':c_['C'],'d':c_['d'],'delv':vis_,'lt':-1,'fl':0,'rec':{k9_:c_[k9_] for k9_ in ('A','B','C','r','typed','s','src','d','dep','v') if k9_ in c_}}) #   once those side links are gone; a compact ghost (A, C, domain) stays for the readout only
        del GH_[:-4000]; gr_={}
        for g_ in sorted(GH_,key=lambda g_: g_['lt'])[:int(getattr(cfg,'inf_ghost_n',32))]:
            if g_['delv']>=vis_: continue                                       # GHOST READOUT: deleted conclusions, store off - still answered from the weights alone?
            o_=[x_ for x_ in Cp_ if x_!=g_['C']]; rs_.shuffle(o_)
            if len(o_)<4: continue
            lp_=_inf_lps(model,cfg,f"So {g_['A']} is connected to ",[g_['C']]+o_[:9],int(S_slot(g_['d'])),False,dec_=True); g_['lt']=vis_; a9_=vis_-g_['delv']
            gr_.setdefault('1' if a9_<=1 else '2-3' if a9_<=3 else '4+',[]).append(float(int(np.argmax(lp_))==0))
            g_['fl']=0 if int(np.argmax(lp_))==0 else g_.get('fl',0)+1
        K9_=int(getattr(cfg,'inf_ghost_reinstate',0)); nri_=0; nsd_=0
        if K9_>0:                                                               # cg26l GHOST REINSTATEMENT: a deleted conclusion the weights have lost (its ghost failed
            it8_=S_['it']; keep8_=[]                                            #   store-off at K tests running) is written back to the fast store and replay, unless one
            for g_ in GH_:                                                      #   of its source sentences has since been superseded
                if g_.get('fl',0)>=K9_ and g_.get('rec'):
                    sr8_=g_['rec'].get('src'); sup8_=isinstance(sr8_,tuple) and any(isinstance(i8_,int) and i8_<len(it8_) and it8_[i8_].get('sup') for i8_ in sr8_)
                    if sup8_: nsd_+=1; continue
                    if _kgx_ho(cfg,g_['rec']): continue
                    c8_=dict(g_['rec']); c8_.update(v=vis_,bv=vis_,so=[],reinst=vis_); S_['conc'].append(c8_); nri_+=1
                    S_.setdefault('pairs',set()).add((min(c8_['A'],c8_['C']),max(c8_['A'],c8_['C'])))
                    continue
                keep8_.append(g_)
            GH_[:]=keep8_; st9_['rein']=st9_.get('rein',0)+nri_
        print(f"  [inference CLS | DELETED] {ndc_} released conclusions deleted this sleep ({st9_.get('dele',0)} so far, {st9_.get('purg',0)} records purged; grace {G9_} passing sleeps after release) | "
              f"ghosts re-tested store-off {sum(len(v_) for v_ in gr_.values())}, still answered by visits since deletion: "+(", ".join(f"{k_} back {np.mean(v_):.2f} (n {len(v_)})" for k_,v_ in sorted(gr_.items())) or "none yet")
              +(f" | REINSTATED {nri_} this sleep ({st9_.get('rein',0)} so far; dropped as superseded {nsd_}) after {K9_} failed tests running" if K9_>0 else ""),flush=True)
    if int(getattr(cfg,'inf_release_hl',0)):
        print("  [inference CLS | RELEASE TIMING] "+(f"decay model calibrated (beat a constant rate out of sample at the last 2 sleeps): {nhl_} conclusions with a predicted half-life of {hz_:.0f}+ visits marked for release"+(f" (by PREDICTED RECALL at the next visit >= {pn9_:g}: {npn9_})" if pn9_>0 else "")+(f" (by PREDICTED RECALL at the next visit in the top {pq9_:.0%} of {npq9_} tracked, bar {thq9_:.3f} = max(quantile, 2 x median, {pmn9_:g}): {npn9_})" if thq9_ is not None else (f" (relative predicted-recall rule idle: {npq9_} tracked, spread of log2 half-lives {sdq9_:.2f}, needs >= 10 and >= 0.1)" if pq9_>0 else ""))+f"; either rule releases - by the fixed rule (consecutive passing sleeps) {nfx9_} more"
              if hlc_ else "decay model not calibrated yet (needs to beat a constant rate out of sample at 2 sleeps running): the fixed rule (consecutive passing sleeps) decides"),flush=True)
    if int(getattr(cfg,'inf_hl',0)):
        try: _hl_model(model,cfg,S_,obs9_,vis_)
        except Exception as e9_: _fail('inference.evaluate:377'); print(f"  [inference DECAY MODEL] unavailable ({type(e9_).__name__}: {e9_})"[:200],flush=True)
    if int(getattr(cfg,'inf_schema_speed',0)):                                  # cg26i SCHEMA SPEED (readout): do conclusions the trunk already predicts well (lower half of trunk
        tb9_=[c_ for c_ in lv_+[x_ for x_ in allc_ if x_.get('released') or x_.get('deleted') is not None] if 'tbits' in c_]   #   bits) pass store-off and reach release
        if len(tb9_)>=6:                                                        #   sooner than misfits (Tse et al. 2007)?
            md9_=float(np.median([c_['tbits'] for c_ in tb9_])); gp9_={'fit':[],'misfit':[]}
            for c_ in tb9_: gp9_['fit' if c_['tbits']<=md9_ else 'misfit'].append(c_)
            def ss9_(L_):
                ps_=[h_[1] for c_ in L_ for h_ in c_.get('so',[])]; rl_=[c_ for c_ in L_ if c_.get('relv') is not None]
                return (f"n {len(L_)}, store-off tests passed {np.mean(ps_) if ps_ else float('nan'):.2f} (of {len(ps_)}), released {len(rl_)}"
                        +(f", visits from first replay to release {np.mean([c_['relv']-c_.get('bv',c_['relv']) for c_ in rl_]):.1f}" if rl_ else ""))
            print(f"  [inference CLS | SCHEMA SPEED] trunk bits median {md9_:.2f} | fit (lower half): {ss9_(gp9_['fit'])} | misfit (upper half): {ss9_(gp9_['misfit'])}",flush=True)
            og9_={'REM':[c_ for c_ in tb9_ if c_.get('rem')],'NREM':[c_ for c_ in tb9_ if not c_.get('rem')]}   # cg26n: by origin - are REM's conclusions more often schema fits?
            print("  [inference CLS | SCHEMA SPEED BY ORIGIN] "+" | ".join(f"{k_} (n {len(L_)}): fits {np.mean([c_['tbits']<=md9_ for c_ in L_]) if L_ else float('nan'):.2f}, {ss9_(L_)}" for k_,L_ in og9_.items()),flush=True)
    st_=S_.setdefault('_relstat',{'rel':0,'res':0})
    print(f"  [inference CLS | STORE HANDOVER] tracked {len(trk_)} conclusions | released so far {st_['rel']} of {len(lv_)} live (re-tested now {len(rok_)}, still answered store-off {np.mean(rok_) if rok_ else float('nan'):.2f}),"
          f" restored so far {st_['res']} | this sleep: {nrel_} to release, {nres_} to restore (applied at the next store write)"+("" if int(getattr(cfg,'inf_release',0)) else " | readout only: release off"),flush=True)
def _rem_rel(s_):
    """readout truth only: the relation phrase of a templated fact sentence, else None"""
    R0_=_INF.setdefault('_relset',set(_inf_phrase(t_) for t_ in _INF_REL)); b_=s_ if isinstance(s_,bytes) else str(s_).encode('latin-1','replace')
    for m_ in _INF_TRIP_RX.finditer(b_):
        r_=m_.group(2).decode('latin-1')
        if r_ in R0_: return r_
    return None
def _jd_embed(model,cfg,txts_,norm_=True):
    """cg26l: (JEPA embedding, standardised trunk code) of each text, both unit length; (None, None) before the selector is trained"""
    SH_=model.__dict__.get('_jdS'); E_=model.__dict__.get('_jdenc')
    if SH_ is None or E_ is None or not txts_: return None,None
    g_,gt_,Pq_,_=SH_; mu_,sd_,V_=E_; Z_=[]; Y0_=[]
    with torch.no_grad():
        for o_ in range(0,len(txts_),128):
            Y_=(_jd_codes(model,cfg,txts_[o_:o_+128])-mu_)/sd_
            if V_ is not None: Y_=(Y_@V_.T) if model.__dict__.get('_jdpca') else Y_-(Y_@V_.T)@V_
            Y0_.append(Y_.float()); Z_.append((g_(Y_) if model.__dict__.get('_jdobj')=='lejepa' else gt_(Y_)).float())
    if not norm_: return torch.cat(Z_),torch.cat(Y0_)
    return F.normalize(torch.cat(Z_),dim=-1),F.normalize(torch.cat(Y0_),dim=-1)
def _eff_rank(L_):
    L_=L_.clamp(min=0).double(); p_=L_/L_.sum().clamp(min=1e-12)+1e-7; return float(torch.exp(-(p_*p_.log()).sum()))
def _lidar(Z_,G_):
    """LiDAR (Thilak et al. 2023): effective rank of S_w^-1/2 S_b S_w^-1/2 over surrogate classes G_ (rows of Z_ with the same label = views of one class)"""
    Z_=Z_.double(); u_,inv_=torch.unique(G_,return_inverse=True); mu_=torch.zeros(len(u_),Z_.shape[1],dtype=Z_.dtype).index_add_(0,inv_,Z_)/torch.bincount(inv_).double()[:,None]
    m_=mu_.mean(0); Sb_=((mu_-m_).T@(mu_-m_))/len(u_); W_=Z_-mu_[inv_]; Sw_=(W_.T@W_)/len(Z_); Sw_=Sw_+1e-3*float(Sw_.diagonal().mean())*torch.eye(Z_.shape[1],dtype=Z_.dtype)
    e_,V_=torch.linalg.eigh(Sw_); Wi_=V_@torch.diag(e_.clamp(min=1e-12)**-0.5)@V_.T
    return _eff_rank(torch.linalg.eigvalsh(Wi_@Sb_@Wi_))
def _jd_lidar(model,cfg,S_):
    """cg26l LiDAR READOUT (label-free; readout only): surrogate classes = linked pairs (two premises of a verified conclusion, or facts that chain
    through a shared entity), the two passages = its two views. LiDAR and RankMe of the JEPA embedding against the trunk codes it reads"""
    P_=_jd_pairs(S_,np.random.RandomState(4343+int(model.__dict__.get('_visit',0))),512)
    P_=[(a_,b_) for a_,b_ in P_ if a_.strip()!=b_.strip()][:int(getattr(cfg,'inf_jd_lidar_n',256))]
    if len(P_)<16: print(f"  [inference JEPA DREAMS | LiDAR] skipped: {len(P_)} linked pairs",flush=True); return
    Ej_,Et_=_jd_embed(model,cfg,[a_ for a_,_ in P_]+[b_ for _,b_ in P_],norm_=False)
    if Ej_ is None: print("  [inference JEPA DREAMS | LiDAR] skipped: the selector is not trained yet",flush=True); return
    G_=torch.arange(len(P_)).repeat(2); out_=[]
    for nm_,Z_ in (('JEPA embedding',Ej_.cpu()),('trunk code (standardised, top directions removed)',Et_.cpu())):
        out_.append(f"{nm_}: LiDAR {_lidar(Z_,G_):.1f}, RankMe {_eff_rank(torch.linalg.svdvals(Z_.double())):.1f} of {Z_.shape[1]}")
    H_=S_.setdefault('_lidar_log',[]); H_.append((int(model.__dict__.get('_visit',0)),out_[0]))
    print(f"  [inference JEPA DREAMS | LiDAR] {len(P_)} linked pairs as surrogate classes (2 views each) | "+" | ".join(out_),flush=True)
def _plat_pairs(S_,cfg):
    """the fixed probe: linked pairs in their own hash bucket (the JEPA uses buckets 0 and 1), frozen once chosen, topped up to inf_plat_n"""
    import zlib as _zl
    P0_=S_.setdefault('_plat',[]); n_=int(getattr(cfg,'inf_plat_n',64))
    if len(P0_)<n_:
        hv_={(a_.strip(),b_.strip()) for a_,b_ in P0_}
        for a_,b_ in _jd_pairs(S_,np.random.RandomState(4545),3000):
            if len(P0_)>=n_: break
            k_=(a_.strip(),b_.strip())
            if k_[0]==k_[1] or k_ in hv_ or (k_[1],k_[0]) in hv_ or _zl.crc32(b'\x00'.join(sorted(k_)))%5!=2: continue
            P0_.append((a_,b_)); hv_.add(k_)
    return P0_
def _plat_codes(model,P_,grad_=False):
    A_=_jepa_emb1(model,[a_ for a_,_ in P_],grad_).float(); B_=_jepa_emb1(model,[b_ for _,b_ in P_],grad_).float()
    mu_=torch.cat([A_,B_]).mean(0,keepdim=True); A_=F.normalize(A_-mu_,dim=-1); B_=F.normalize(B_-mu_,dim=-1)
    M_=A_@B_.T; n_=len(P_); dg_=M_.diagonal(); sep_=dg_.mean()-(M_.sum()-dg_.sum())/max(1,n_*n_-n_)
    return sep_,A_,B_,M_
def _plat_readout(model,cfg,S_):
    """cg26l PLATONIC PROBE (readout): how close the trunk's passage codes are to one shared representation of what they are about"""
    P_=_plat_pairs(S_,cfg)
    if len(P_)<16: return None
    was_=model.training; model.eval()
    try:
        with torch.no_grad(): sep_,A_,B_,M_=_plat_codes(model,P_)
    finally: model.train(was_)
    n_=len(P_); ar_=torch.arange(n_,device=M_.device); k_=min(5,n_-1)
    t1_=0.5*(float((M_.argmax(1)==ar_).float().mean())+float((M_.argmax(0)==ar_).float().mean()))
    SA_=A_@A_.T; SB_=B_@B_.T; SA_.fill_diagonal_(-9.0); SB_.fill_diagonal_(-9.0); na_=SA_.topk(k_,1).indices.tolist(); nb_=SB_.topk(k_,1).indices.tolist()
    mk_=float(np.mean([len(set(na_[i_])&set(nb_[i_]))/k_ for i_ in range(n_)]))
    Z_=torch.cat([A_,B_]); sv_=torch.linalg.svdvals(Z_-Z_.mean(0)); p_=sv_**2/(sv_**2).sum(); er_=float(torch.exp(-(p_*torch.log(p_+1e-12)).sum()))
    return dict(sep=float(sep_),top1=t1_,mknn=mk_,rank=er_,top=float(p_[0]),n=n_)
def _plat_fmt(d_): return f"separation {d_['sep']:+.3f}, partner top-1 {d_['top1']:.2f}, mutual-kNN alignment {d_['mknn']:.3f}, effective rank {d_['rank']:.1f} (top direction {d_['top']:.0%})"
def _plat_select(model,cfg,S_,S_slot):
    """cg26l PLATONIC SELECTION: first-order effect of one training step on each dream candidate (its row through the trunk, store off):
    s1 = -cos(g_pair, grad separation) > 0 -> the step brings linked passages together relative to unlinked ones;
    s2 =  cos(g_pair, grad real-text loss) > 0 -> the step also lowers loss on stored real text of all domains"""
    MX_=S_.get('mixtxt') or []; P_=_plat_pairs(S_,cfg); vis_=int(model.__dict__.get('_visit',0))
    if len(MX_)<8 or len(P_)<16: print(f"  [inference PLATONIC SELECTION] skipped: {len(MX_)} dream pairs, {len(P_)} probe pairs",flush=True); return
    prm_=[p_ for b_ in model.blocks for p_ in b_.parameters() if p_.requires_grad]
    flat_=lambda G_: torch.cat([(g_ if g_ is not None else torch.zeros_like(p_)).reshape(-1).float() for g_,p_ in zip(G_,prm_)])
    rs_=np.random.RandomState(6464+vis_); back_=_ws_eager(model); was_=model.training; sc_={}
    try:
        with torch.enable_grad():
            model.eval(); sep_,_,_,_=_plat_codes(model,P_,grad_=True); vS_=flat_(torch.autograd.grad(sep_,prm_,allow_unused=True))
            rl_=[x_ for x_ in S_['it'] if x_.get('k')=='real' and not x_.get('gone') and len(x_['s'])>=40]; Lr_=0.0; nr_=0
            for i_ in rs_.choice(len(rl_),min(16,len(rl_)),replace=False) if rl_ else []:
                x_=rl_[int(i_)]; X_,M_=_ws_batch(model,[x_['s']]); lg_=_ws_run(model,X_,int(S_slot(x_['d'])),grad_=True)
                Lr_=Lr_+(F.cross_entropy(lg_[0],X_[0,1:],reduction='none')*M_[0,1:]).sum()/M_[0,1:].sum().clamp(min=1); nr_+=1
            vR_=flat_(torch.autograd.grad(Lr_/max(1,nr_),prm_,allow_unused=True))
            cd_=rs_.permutation(len(MX_))[:int(getattr(cfg,'inf_plat_cand',96))]
            for j_ in cd_:
                ta_,tb_,d_,src_=MX_[int(j_)]
                if int(d_)<0: continue
                X_,M_=_ws_batch(model,[ta_.strip()+b' '+tb_.strip()]); lg_=_ws_run(model,X_,int(S_slot(d_)),grad_=True)
                l_=(F.cross_entropy(lg_[0],X_[0,1:],reduction='none')*M_[0,1:]).sum()/M_[0,1:].sum().clamp(min=1)
                gP_=flat_(torch.autograd.grad(l_,prm_,allow_unused=True))
                sc_[int(j_)]=(float(-F.cosine_similarity(gP_,vS_,dim=0)),float(F.cosine_similarity(gP_,vR_,dim=0)),src_)
    except Exception as e9_: _fail('inference.evaluate:490'); print(f"  [inference PLATONIC SELECTION] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True); return
    finally: model.train(was_); back_()
    if not sc_: return
    W_={j_:(4.0 if (a_>0 and b_>-0.05) else 0.25 if a_<0 else 1.0) for j_,(a_,b_,_) in sc_.items()}
    S_['_platw']=(len(MX_),W_)
    by_={}
    for a_,b_,s9_ in sc_.values(): by_.setdefault(s9_,[]).append((a_,b_))
    v_=np.array([[a_,b_] for a_,b_,_ in sc_.values()])
    print(f"  [inference PLATONIC SELECTION] {len(sc_)} of {len(MX_)} dream pairs scored by their first-order effect | towards the shared representation (separation up) {np.mean(v_[:,0]>0):.2f}, "
          f"also lowering real-text loss {np.mean((v_[:,0]>0)&(v_[:,1]>-0.05)):.2f} -> drawn x4; separation down {np.mean(v_[:,0]<0):.2f} -> x0.25 | mean cos: separation {v_[:,0].mean():+.3f}, real-text loss {v_[:,1].mean():+.3f} | by source (separation): "
          +", ".join(f"{k_} {np.mean([a_ for a_,_ in L_]):+.3f} (n {len(L_)})" for k_,L_ in sorted(by_.items())),flush=True)
def _rem_exam(model,cfg,S_,S_slot):
    """cg26l REM SELF-EXAM: after the NREM store writes, the cortex is examined on stored conclusions with the store OFF - an open answer
    (greedy completion of 'So A is connected to ') and a ranking of C against the open answer and 8 other conclusion ends:
      RIGHT   C ranked first by the margin AND given as the open answer -> a release candidate (tracked and tested this sleep)
      WRONG   another name preferred by the margin -> a conflict between cortex and store (logged by kind; replayed x3)
      UNSURE  neither -> NREM priority (replayed x2; dream pairs holding its names drawn x3)
    Half the sample re-examines earlier ones (oldest exam first), so the readout shows what became of each category"""
    allc_=S_.get('conc',[]); lv_=[c_ for c_ in allc_ if not c_.get('freed') and isinstance(c_.get('C'),str) and isinstance(c_.get('A'),str)]
    vis_=int(model.__dict__.get('_visit',0)); S_['_ex_names']=set(); S_['_ex_v']=vis_
    if len(lv_)<5: print(f"  [inference REM SELF-EXAM] skipped: {len(lv_)} live conclusions",flush=True); return
    rs_=np.random.RandomState(7272+vis_); n_=int(getattr(cfg,'inf_rem_exam_n',48)); mg_=float(getattr(cfg,'inf_release_margin',1.0))
    prv_=sorted([c_ for c_ in lv_ if c_.get('ex')],key=lambda c_: c_['ex'][-1][0]); new_=[c_ for c_ in lv_ if not c_.get('ex')]; rs_.shuffle(new_)
    sel_=prv_[:n_//2]; sel_+=new_[:n_-len(sel_)]; sel_+=prv_[n_//2:][:n_-len(sel_)]
    Cp_=sorted(set(c_['C'] for c_ in lv_)); Cset_=set(Cp_); t2_,_=_inf_truth(cfg); t2o_,_=_inf_truth(cfg,current=False); pn_=set(x_ for q_ in inf_probes(cfg) for x_ in q_['names'])
    byA_={}
    for c_ in lv_: byA_.setdefault(c_['A'],set()).add(c_['C'])
    st_={'right':0,'wrong':0,'unsure':0,'on':0,'open':0,'n':0}; wk_={'bridge B':0,'another stored link of A':0,'stale (old fact)':0,'other':0}; wt_=[0,0]; tr_={}; eg_=[]
    f0_=_INFM.get('force','__none__')
    try:
        for c_ in sel_:
            A_,C_=c_['A'],c_['C']; sl_=int(S_slot(c_['d'])); pr_=f"So {A_} is connected to "
            g0_=_inf_nm1(_inf_gen(model,cfg,pr_,sl_,False))
            o_=[x_ for x_ in Cp_ if x_ not in (C_,A_)]; rs_.shuffle(o_)
            cd_=[C_]+o_[:9]                                                     # FIX: ranked against stored conclusion ends only - the cortex's own greedy
            st_['gknown']=st_.get('gknown',0)+int(bool(g0_) and g0_!=C_ and g0_ in Cset_)   #   answer always wins its own ranking (local: 14 of 14 'wrong' by 8-16 nats)
            if len(cd_)<5: continue
            lo_=_inf_lps(model,cfg,pr_,cd_,sl_,False,dec_=True); ln_=_inf_lps(model,cfg,pr_,cd_,sl_,True,dec_=True)   # (already PMI-corrected below; may steer REM)
            k0_=int(np.argmax(lo_)); mo_=float(lo_[0]-np.max(lo_[1:]))
            pm_=lo_-_inf_lps(model,cfg,"So Someone is connected to ",cd_,sl_,False,dec_=True)   # FIX: specific to the cue (pointwise mutual information) - summed log-probabilities
            j9_=1+int(np.argmax(pm_[1:])); top_=cd_[k0_]                         #   of different names mostly reflect their length and frequency (local: 14 of 14 'wrong')
            st_['pmi']=st_.get('pmi',0.0)+float(pm_[0])
            cat_='right' if (k0_==0 and mo_>=mg_ and g0_==C_) else 'wrong' if (float(pm_[j9_]-pm_[0])>=mg_ and float(pm_[j9_])>=mg_) else 'unsure'
            if cat_=='wrong': top_=cd_[j9_]
            H_=c_.setdefault('ex',[]); pc_=H_[-1][1] if (H_ and H_[-1][0]<vis_) else None
            if H_ and H_[-1][0]==vis_: H_[-1]=(vis_,cat_,mo_,top_)
            else: H_.append((vis_,cat_,mo_,top_))
            del H_[:-8]
            if pc_ is not None: tr_[(pc_,cat_)]=tr_.get((pc_,cat_),0)+1
            st_['n']+=1; st_[cat_]+=1; st_['on']+=int(int(np.argmax(ln_))==0); st_['open']+=int(g0_==C_)
            if cat_=='wrong':
                k9_='bridge B' if top_==c_.get('B') else 'another stored link of A' if top_ in byA_.get(A_,()) else 'stale (old fact)' if ((A_,top_) in t2o_ and (A_,top_) not in t2_) else 'other'
                wk_[k9_]+=1
                if A_ in pn_ or C_ in pn_: wt_[0 if (A_,C_) in t2_ else 1]+=1
                if len(eg_)<3: eg_.append(f"{A_} -> store {C_}, cortex {top_} ({k9_}, cue-specific by {float(pm_[j9_]-pm_[0]):.1f} nats over C)")
            if cat_!='right': S_['_ex_names'].update(x_.encode('latin-1','replace') for x_ in (A_,C_))
    finally:
        if f0_=='__none__': _INFM.pop('force',None)
        else: _INFM['force']=f0_
    n0_=max(1,st_['n']); act_=int(getattr(cfg,'inf_rem_exam_act',0))
    trs_=", ".join(f"{a_}->{b_} {v_}" for (a_,b_),v_ in sorted(tr_.items()))
    print(f"  [inference REM SELF-EXAM] {st_['n']} conclusions examined store-off ({len(prv_)} examined before) | RIGHT {st_['right']} ({st_['right']/n0_:.2f}; open answer right {st_['open']/n0_:.2f}) | "
          f"WRONG {st_['wrong']} ({st_['wrong']/n0_:.2f}): "+", ".join(f"{k_} {v_}" for k_,v_ in wk_.items())+f"; probe conclusions among them true {wt_[0]}, false {wt_[1]} | UNSURE {st_['unsure']} ({st_['unsure']/n0_:.2f}) | "
          f"store on answers {st_['on']/n0_:.2f} | cue-specific log-prob of C (vs 'Someone') mean {st_.get('pmi',0.0)/n0_:+.2f} nats | open answer another stored conclusion end {st_.get('gknown',0)} | since the last exam: {trs_ or 'none yet'} | "+("acting: right -> release candidates, wrong x3 and unsure x2 in replay, dream pairs with their names x3" if act_ else "readout only"),flush=True)
    for e_ in eg_: print(f"    [inference REM SELF-EXAM e.g.] {e_}",flush=True)
def _rem_analogy(model,cfg,S_,S_slot):
    """cg26l REM ANALOGY DREAMS: links (two premises of a verified conclusion, or a dreamed pair) are paired with a link that shares NO rare
    name but has the same relation structure, by relation vectors r = e(second) - e(first): partners maximise cos(r_i, r_j). Spaces compared on
    the same candidates: JEPA embedding, trunk code, non-name word overlap, random. Truth (readout only): the relation phrases of templated
    facts - same pattern (r0, r1). The chosen space's pairs are dreamed as 'link_i link_j' with link_j as the target (inf_rem_ana_dream)"""
    it_=S_['it']; df_=S_.get('df',{}); dm_=int(cfg.inf_df); L_=[]; sn_=set()
    for c_ in S_.get('conc',[]):
        if c_.get('freed') or not isinstance(c_.get('src'),tuple) or len(c_['src'])!=2: continue
        i0_,i1_=c_['src']
        if isinstance(i0_,int) and isinstance(i1_,int) and i0_<len(it_) and i1_<len(it_) and int(c_['d'])>=0: L_.append((it_[i0_]['s'],it_[i1_]['s'],int(c_['d']),'premise'))
    for ta_,tb_,d_,src_ in (S_.get('mixtxt') or [])[-600:]:
        if src_ not in ('random-dream','control+','premise') and int(d_)>=0: L_.append((ta_,tb_,int(d_),src_))
    LL_=[]
    for x_ in L_:
        k_=(x_[0],x_[1])
        if k_ not in sn_ and x_[0]!=x_[1]: sn_.add(k_); LL_.append(x_)
    vis_=int(model.__dict__.get('_visit',0)); rs_=np.random.RandomState(6363+vis_); np_=int(getattr(cfg,'inf_rem_ana_pool',400))
    if len(LL_)>np_: LL_=[LL_[int(i_)] for i_ in rs_.choice(len(LL_),np_,replace=False)]
    if len(LL_)<8: print(f"  [inference REM ANALOGY] skipped: {len(LL_)} links",flush=True); return
    tx_=sorted(set(t_ for a_,b_,_,_ in LL_ for t_ in (a_,b_))); ix_={t_:i_ for i_,t_ in enumerate(tx_)}
    Ej_,Et_=_jd_embed(model,cfg,tx_)
    if Ej_ is None: print("  [inference REM ANALOGY] skipped: the JEPA selector is not trained yet",flush=True); return
    ia_=torch.tensor([ix_[a_] for a_,_,_,_ in LL_]); ib_=torch.tensor([ix_[b_] for _,b_,_,_ in LL_])
    def rv_(E_): return F.normalize(E_[ib_]-E_[ia_],dim=-1)
    ents_=[set(w_ for w_ in _INF_NAME_RX.findall(a_+b' '+b_) if 0<df_.get(w_,0)<=dm_) for a_,b_,_,_ in LL_]
    wds_=[set(w_.lower() for w_ in __import__('re').findall(rb'[A-Za-z]+',a_+b' '+b_) if w_ not in e_) for (a_,b_,_,_),e_ in zip(LL_,ents_)]
    n_=len(LL_); ok_=torch.ones(n_,n_,dtype=torch.bool)
    for i_ in range(n_):
        ok_[i_,i_]=False
        for j_ in range(i_+1,n_):
            if ents_[i_]&ents_[j_] or {LL_[i_][0],LL_[i_][1]}&{LL_[j_][0],LL_[j_][1]}: ok_[i_,j_]=ok_[j_,i_]=False
    pat_=[(_rem_rel(a_),_rem_rel(b_)) for a_,b_,_,_ in LL_]; kn_=[p_[0] is not None and p_[1] is not None for p_ in pat_]
    def pick_(Sm_):
        Sm_=Sm_.clone().float(); Sm_[~ok_]=-9.0; out_=[]; used_=set(); K_=int(getattr(cfg,'inf_rem_ana_n',16)); mn_=float(getattr(cfg,'inf_rem_ana_min',0.3))
        fl_=Sm_.triu(1).flatten(); od_=torch.argsort(fl_,descending=True)
        for q_ in od_.tolist():
            v_=float(fl_[q_])
            if v_<mn_ or len(out_)>=K_: break
            i_,j_=divmod(q_,n_)
            if i_ in used_ or j_ in used_: continue
            used_.add(i_); used_.add(j_); out_.append((i_,j_,v_))
        return out_
    def score_(P_):
        k_=[(i_,j_) for i_,j_,_ in P_ if kn_[i_] and kn_[j_]]
        if not k_: return f"{len(P_)} pairs, none scorable"
        sm_=np.mean([pat_[i_]==pat_[j_] for i_,j_ in k_]); s1_=np.mean([bool(set(pat_[i_])&set(pat_[j_])) for i_,j_ in k_])
        return f"{len(P_)} pairs, scorable {len(k_)}: same relation pattern {sm_:.2f}, a shared relation {s1_:.2f}"
    Rj_=rv_(Ej_); Rt_=rv_(Et_); Sj_=Rj_@Rj_.T; St_=Rt_@Rt_.T
    Sw_=torch.tensor([[len(wds_[i_]&wds_[j_])/max(1,len(wds_[i_]|wds_[j_])) for j_ in range(n_)] for i_ in range(n_)])
    Pj_=pick_(Sj_); Pt_=pick_(St_); Pw_=pick_(Sw_*2-1+0.3)                     # (word overlap shifted onto the cosine scale: Jaccard 0.5 -> 0.3 bar)
    ch_=[]
    for i_ in range(n_):                                                        # chance: a random allowed partner (exact expectation)
        if not kn_[i_]: continue
        al_=[j_ for j_ in range(n_) if ok_[i_,j_] and kn_[j_]]
        if al_: ch_.append(np.mean([pat_[i_]==pat_[j_] for j_ in al_]))
    sp_=str(getattr(cfg,'inf_rem_ana_space','jepa')); P_={'jepa':Pj_,'trunk':Pt_,'words':Pw_}.get(sp_,Pj_)
    A_=S_.setdefault('anatxt',[]); rg9_=S_.get('ring'); Lr_=min(int(cfg.inf_rowlen)+1,len(rg9_[0][0]) if rg9_ else int(cfg.inf_rowlen)+1); nd_=0   # FIX: the replay row length (locally 129, not 256)
    if float(getattr(cfg,'inf_rem_ana_dream',0.0))>0:
        MC_=S_.setdefault('mixcmp',{})
        def tx2_(a_,b_,d_):
            raw_=a_.decode('latin-1').strip()+' '+b_.decode('latin-1').strip()
            if len(raw_)<=(Lr_-14)//2: return raw_
            if (a_,b_) not in MC_:
                try: MC_[(a_,b_)]=_compact(model,cfg,a_,b_,int(S_slot(d_)))
                except Exception: MC_[(a_,b_)]=None
            c9_=MC_.get((a_,b_)); return c9_.decode('latin-1') if c9_ and len(c9_)<=(Lr_-14)//2 else None
        for i_,j_,_ in P_:
            for u_,v_ in ((i_,j_),(j_,i_)):                                     # both directions: either link primes the other
                s1_=tx2_(LL_[u_][0],LL_[u_][1],LL_[u_][2]); s2_=tx2_(LL_[v_][0],LL_[v_][1],LL_[v_][2])
                if s1_ and s2_: A_.append((s1_,s2_,LL_[v_][2])); nd_+=1
        del A_[:-400]
    eg_=[f"{LL_[i_][0].decode('latin-1')[:50]} / {LL_[i_][1].decode('latin-1')[:50]}  ~  {LL_[j_][0].decode('latin-1')[:50]} / {LL_[j_][1].decode('latin-1')[:50]} (cos {v_:.2f})" for i_,j_,v_ in Pj_[:2]]
    print(f"  [inference REM ANALOGY] {n_} links ({sum(kn_)} with templated relations, {int(ok_.sum())//2} allowed pairs sharing no rare name) | relation vectors in JEPA space: {score_(Pj_)} | "
          f"trunk space: {score_(Pt_)} | non-name word overlap: {score_(Pw_)} | random partner: same pattern {np.mean(ch_) if ch_ else float('nan'):.2f} | "
          f"dreamed from {sp_}: {nd_} analogy rows added ({len(A_)} held)"+("" if float(getattr(cfg,'inf_rem_ana_dream',0.0))>0 else " | readout only"),flush=True)
    for e_ in eg_: print(f"    [inference REM ANALOGY e.g. JEPA] {e_}",flush=True)

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_INF_PARA': 'platypus.inference.qtime',
    '_SCORE_LEN': 'platypus.inference.readouts',
    '_score_norm': 'platypus.inference.readouts',
    '_compact': 'platypus.inference.selftaught',
    '_hl_model': 'platypus.inference.decay',
    '_hl_x': 'platypus.inference.decay',
    '_inf_abstain_sources': 'platypus.inference.audit',
    '_inf_gen': 'platypus.inference.rem',
    '_inf_heldout': 'platypus.inference.audit',
    '_inf_nm1': 'platypus.inference.rem',
    '_jd_codes': 'platypus.inference.jepa',
    '_jd_pairs': 'platypus.inference.jepa',
    '_jepa_emb1': 'platypus.inference.jepa',
    '_ws_batch': 'platypus.inference.closed_form',
    '_ws_eager': 'platypus.inference.closed_form',
    '_ws_run': 'platypus.inference.closed_form',
    'inf_qtime2': 'platypus.inference.qtime',
    'inf_qtime3': 'platypus.inference.qtime',
    'inf_splice': 'platypus.inference.qtime',
}
