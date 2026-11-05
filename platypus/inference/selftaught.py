"""Self-taught inference: freeing, occlusion, attended-span splices, spectral word codes."""
import math
import numpy as np
import re as _re_inf
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
    _INFM, _infm_corr, _infm_keys, _infm_write,
)
from platypus.inference.units import (
    _UWORD,
)
from platypus.inference.readouts import (
    _inf_score1,
)

@torch.no_grad()
def _inf_free(model,cfg,S_,allc_,S_slot,Lr_,filler,fwd,feats,rs_,dev_,beta_):
    """FREEING (cg20 test 2d): a conclusion the WEIGHTS ALONE now recall (C beats 19 store names after 'So A is connected to') has its store
    entries faded (delta writes towards the weights' own prediction) and is no longer rehearsed"""
    if not int(getattr(cfg,'inf_xfer_free',0)): return ''
    live_=[c_ for c_ in allc_ if not c_.get('freed')]; nm_=[n_.decode('latin-1') for n_ in _INFM['names']]
    if not live_ or len(nm_)<25: return ''
    chk_=[live_[i_] for i_ in rs_.choice(len(live_),min(len(live_),int(getattr(cfg,'inf_xfer_check',64))),replace=False)]; fr_=0
    for c_ in chk_:
        oth_=[n_ for n_ in nm_ if n_ not in (c_['A'],c_['C'])]; cs_=[c_['C']]+list(rs_.choice(oth_,19,replace=False))
        if _inf_score1(model,cfg,f"So {c_['A']} is connected to ",cs_,int(S_slot(c_['d'])),False)!=0: continue
        c_['freed']=True; fr_+=1
        for sn_ in [f"So {c_['A']} is connected to {c_['C']}.",f"So {c_['C']} is connected to {c_['A']}."]+([c_['s']] if c_['typed'] else []):
            it_=(sn_+' ').encode('latin-1')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); w_=w_[:Lr_]
            X_=torch.tensor([list(bytes(w_))],dtype=torch.long,device=dev_); lg_,z_=feats(X_,int(S_slot(c_['d']))); K_,id_=_infm_keys(z_,X_[:,:-1],lg_)
            Kf_=K_.reshape(-1,K_.shape[-1]); idf_=id_.reshape(-1); lgf_=lg_.reshape(-1,256); p0_=F.softmax(lgf_,-1); mk_=torch.zeros(Lr_-1,device=dev_); mk_[:len(it_)-1]=1.0
            for _ in range(10): q_=F.softmax(lgf_+_infm_corr(Kf_,idf_),-1); _infm_write(Kf_,idf_,(p0_-q_)*mk_[:,None],beta_)
    return f" | FREEING: {fr_} of {len(chk_)} checked recalled by the weights alone, entries faded (total freed {sum(int(c_.get('freed',False)) for c_ in allc_)} of {len(allc_)})"

# ---- cg20 SELF-TAUGHT INFERENCE (S2-S4): the model completes and checks the conclusions; the engine only pairs the source sentences --------
def _inf_words(t_):
    """generic word split (no name rule): letter runs of length >= 3"""
    return list(dict.fromkeys(w_ for w_ in _re_inf.findall(r'[A-Za-z]{3,}',t_)))
@torch.no_grad()
def _inf_lp(model,cfg,ctx_,cands_,slot_,store_):
    """log-prob of each candidate (+'.') after ctx_ (no carried context; store on/off)"""
    seqs=[(ctx_+c_+'.').encode('latin-1')[-int(cfg.seq_len):] for c_ in cands_]; L_=max(len(s_) for s_ in seqs); dev_=model.embed.weight.device
    X_=torch.zeros(len(seqs),L_,dtype=torch.long,device=dev_)
    for j_,s_ in enumerate(seqs): X_[j_,:len(s_)]=torch.tensor(list(s_),device=dev_)
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot_; _INFM['force']=bool(store_); was_=model.training; model.eval()
    try: lp_=F.log_softmax(model(X_[:,:-1],slot_).float(),-1)
    finally: _carry_load(sv4_); _XDG['cur']=sc_; _INFM.pop('force',None); model.train(was_)
    n0_=[len(s_)-len((c_+'.').encode()) for s_,c_ in zip(seqs,cands_)]
    return [float(lp_[j_,n0_[j_]-1:len(seqs[j_])-1].gather(-1,X_[j_,n0_[j_]:len(seqs[j_]),None]).sum()) for j_ in range(len(seqs))]
@torch.no_grad()
def _inf_cond_bits(model,cfg,pairs_,slot_):
    """bits the trunk alone (store off) needs for each target given its prefix: pairs_ = [(prefix bytes, target bytes)]"""
    L0_=int(cfg.seq_len); seqs=[]; nt_=[]
    for p_,t_ in pairs_:
        t_=t_[:L0_//2]; q_=((p_+b' ') if p_ else b' ')+t_; q_=q_[-L0_:]; seqs.append(q_); nt_.append(len(t_))
    L_=max(len(q_) for q_ in seqs); dev_=model.embed.weight.device; X_=torch.full((len(seqs),L_),32,dtype=torch.long,device=dev_)
    for j_,q_ in enumerate(seqs): X_[j_,:len(q_)]=torch.tensor(list(q_),device=dev_)
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot_; fz_=_INFM.get('force'); _INFM['force']=False; was_=model.training; model.eval()
    try: lp_=F.log_softmax(model(X_[:,:-1],slot_).float(),-1)
    finally:
        _carry_load(sv4_); _XDG['cur']=sc_; model.train(was_)
        if fz_ is None: _INFM.pop('force',None)
        else: _INFM['force']=fz_
    out_=[]
    for j_,q_ in enumerate(seqs):
        a_=len(q_)-nt_[j_]; out_.append(float(-lp_[j_,a_-1:len(q_)-1].gather(-1,X_[j_,a_:len(q_),None]).sum())/math.log(2))
    return out_
@torch.no_grad()
def _inf_cond_bytes(model,cfg,pairs_,slot_):
    """per-byte bits the trunk alone needs for each target given its prefix (pairs_ = [(prefix, target)]), in chunks of 256"""
    out_=[]; L0_=int(cfg.seq_len); dev_=model.embed.weight.device
    for o_ in range(0,len(pairs_),256):
        seqs=[]; nt_=[]
        for p_,t_ in pairs_[o_:o_+256]:
            t_=t_[:L0_//2]; q_=((p_+b' ') if p_ else b' ')+t_; q_=q_[-L0_:]; seqs.append(q_); nt_.append(len(t_))
        Lq_=max(len(q_) for q_ in seqs); X_=torch.full((len(seqs),Lq_),32,dtype=torch.long,device=dev_)
        for j_,q_ in enumerate(seqs): X_[j_,:len(q_)]=torch.tensor(list(q_),device=dev_)
        sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=slot_; fz_=_INFM.get('force'); _INFM['force']=False; was_=model.training; model.eval()
        try: lp_=F.log_softmax(model(X_[:,:-1],slot_).float(),-1)
        finally:
            _carry_load(sv4_); _XDG['cur']=sc_; model.train(was_)
            if fz_ is None: _INFM.pop('force',None)
            else: _INFM['force']=fz_
        for j_,q_ in enumerate(seqs):
            a_=len(q_)-nt_[j_]; out_.append((q_[a_:],(-lp_[j_,a_-1:len(q_)-1].gather(-1,X_[j_,a_:len(q_),None]).squeeze(-1)/math.log(2)).cpu().numpy()))
    return out_
def _blankw(b_):
    """the passage with every word blanked to spaces (layout and punctuation kept): a baseline with the same form and none of the content"""
    o_=bytearray(b_)
    for w_ in _UWORD.finditer(b_): o_[w_.start():w_.end()]=b' '*(w_.end()-w_.start())
    return bytes(o_)
def _occl(model,cfg,ctx_,tgt_,slot_):
    """OCCLUSION ATTRIBUTION: which words of ctx_ the trunk USES to predict tgt_ - bits added to tgt_ when each word alone is blanked; plus the
    gain of the whole context over its blanked form. Returns (words [(start,end,word)], importance per word, gain in bits)"""
    ws_=[(w_.start(),w_.end(),w_.group(0)) for w_ in _UWORD.finditer(ctx_)][:16]
    jobs_=[(ctx_,tgt_),(_blankw(ctx_),tgt_)]
    for a_,e_,_ in ws_: jobs_.append((ctx_[:a_]+b' '*(e_-a_)+ctx_[e_:],tgt_))
    bb_=_inf_cond_bytes(model,cfg,jobs_,slot_); b0_=float(bb_[0][1].sum())
    return ws_,[float(bb_[2+i_][1].sum())-b0_ for i_ in range(len(ws_))],float(bb_[1][1].sum())-b0_
def _compact(model,cfg,p1_,p2_,slot_):
    """ATTENDED-SPAN SPLICE: keep, from each premise, the words its partner's prediction relies on (occlusion) plus its 2 most surprising words
    (the trunk's own error); drop the rest. Returns compact text 'kept words of p1 . kept words of p2' or None"""
    out_=[]
    sp_={}
    for pa_,pb_ in ((p1_,p2_),(p2_,p1_)):
        ws_,im_,_=_occl(model,cfg,pa_,pb_,slot_)
        if not ws_: return None
        bt_=_inf_cond_bytes(model,cfg,[(b'',pa_)],slot_)[0][1]; L2_=len(bt_)
        su_=[float(bt_[max(0,a_-1):min(L2_,e_-1)].mean()) if min(L2_,e_-1)>max(0,a_-1) else 0.0 for a_,e_,_ in ws_]
        mx_=max(im_) if im_ else 0.0; top_=set(sorted(range(len(ws_)),key=lambda i_: -su_[i_])[:2])
        kp_=[w_ for i_,(_,_,w_) in enumerate(ws_) if (mx_>0.3 and im_[i_]>=0.3*mx_) or i_ in top_]
        out_.append(b' '.join(kp_))
    return out_[0]+b'. '+out_[1]+b'.'
@torch.no_grad()
def _inf_remgen(model,cfg,S_,ts,S_slot):
    """GENERATION FROM ATTENDED SPANS (REM, cg20z lab): for live conclusions A ~ C, the model reads the compact attended splice of the two premises
    and continues 'So A is connected to' by itself; a DECOY run swaps in another chain's first premise (A no longer linked) - composition =
    right answers with the true premise MINUS right answers with the decoy (copying the last name scores in both). Answers the model got right
    with the true premise become dream rows (its OWN inferences), replayed with inf_remgen_dream."""
    it_=S_['it']; rs_=np.random.RandomState(31+int(model.__dict__.get('_visit',0))); L0_=int(cfg.seq_len)
    lv_=[c_ for c_ in S_.get('conc',[]) if not c_.get('freed') and isinstance(c_.get('src'),tuple) and isinstance(c_.get('A'),str)]
    if len(lv_)<4: return
    sel_=[lv_[int(i_)] for i_ in rs_.choice(len(lv_),min(int(getattr(cfg,'inf_remgen_n',32)),len(lv_)),replace=False)]
    nt_=nd_=nok_=0; ex_=[]; rows_=S_.setdefault('rgrows',[])
    for k_,c_ in enumerate(sel_):
        sl_=int(S_slot(c_['d']))
        if 'cmp' not in c_:
            try: c_['cmp']=_compact(model,cfg,it_[c_['src'][0]]['s'],it_[c_['src'][1]]['s'],sl_)
            except Exception: c_['cmp']=None
        cm_=c_.get('cmp')
        if not cm_: continue
        q_=f"{cm_.decode('latin-1')} So {c_['A']} is connected to "
        if len(q_)+len(c_['C'])+1>L0_: continue
        o_=sel_[(k_+1)%len(sel_)]; om_=o_.get('cmp')
        g1_=_inf_gen(model,cfg,q_,sl_,False,14).split(' ')[0].strip('.,;')
        nok_+=1; t9_=(g1_==c_['C']); nt_+=int(t9_)
        if om_:
            p2c_=cm_.split(b'. ')[-1]; d_=f"{om_.split(b'. ')[0].decode('latin-1')}. {p2c_.decode('latin-1')} So {c_['A']} is connected to "
            if len(d_)<=L0_: nd_+=int(_inf_gen(model,cfg,d_,sl_,False,14).split(' ')[0].strip('.,;')==c_['C'])
        if t9_:
            rows_.append((f"{cm_.decode('latin-1')} So {c_['A']} is connected to ",f"{c_['C']}.",int(c_['d']))); del rows_[:-500]
        if len(ex_)<2: ex_.append(f"{q_!r} -> {g1_!r} (true {c_['C']})")
    if nok_: print(f"  [inference REM GENERATION FROM ATTENDED SPANS] {nok_} live conclusions fit as compact splices: model continues to the right name {nt_}/{nok_} = {nt_/nok_:.2f} | with a DECOY first premise {nd_}/{nok_} = {nd_/nok_:.2f} | COMPOSITION (true - decoy) {(nt_-nd_)/nok_:+.2f} | self-confirmed dream rows {len(rows_)} | e.g. {' || '.join(ex_)}",flush=True)
def _spec_codes(model,cfg,S_):
    """SPECTRAL WORD CODES (closed-form JEPA, Balestriero & LeCun Prop. 1): positive pairs = two words read in the same stored sentence (two
    observations of one scene); with whitened codes the JEPA optimum is the top eigenvectors of the normalised co-occurrence operator
    T = D^-1/2 C D^-1/2 (no training). Eigenvectors at eigenvalue ~1 (one per disconnected piece) are dropped. Returns (word->row, codes)."""
    import scipy.sparse as _sp, scipy.sparse.linalg as _sla
    it_=S_['it']; ids_=[i_ for i_ in range(max(0,len(it_)-int(getattr(cfg,'inf_spec_docs',20000))),len(it_)) if it_[i_].get('k')=='real' or (it_[i_].get('k')=='ver' and int(getattr(cfg,'inf_mix_revisit',0)))]   # cg26g: revisits write 'ver' sentences
    if len(ids_)<500: return None
    docs_=[set(w_.group(0) for w_ in _UWORD.finditer(it_[i_]['s']) if len(w_.group(0))>=3) for i_ in ids_]
    df_={}
    for d_ in docs_:
        for w_ in d_: df_[w_]=df_.get(w_,0)+1
    mx_=max(5,int(0.02*len(docs_))); voc_={w_:j_ for j_,w_ in enumerate(w_ for w_,c_ in df_.items() if 2<=c_<=mx_)}
    if len(voc_)<200: return None
    r_=[];c_=[]
    for k_,d_ in enumerate(docs_):
        for w_ in d_:
            if w_ in voc_: r_.append(k_); c_.append(voc_[w_])
    X_=_sp.csr_matrix((np.ones(len(r_),np.float32),(r_,c_)),shape=(len(docs_),len(voc_)))
    C_=(X_.T@X_).tocsr(); C_.setdiag(0); C_.eliminate_zeros()
    dg_=np.asarray(C_.sum(1)).ravel()+1e-6; Dm_=_sp.diags(1/np.sqrt(dg_)); T_=(Dm_@C_@Dm_).tocsr()
    kk_=int(getattr(cfg,'inf_spec_k',48)); mu_,Ph_=_sla.eigsh(T_,k=min(len(voc_)-2,kk_*4),which='LA')
    o_=np.argsort(mu_)[::-1]; mu_,Ph_=mu_[o_],Ph_[:,o_]; keep_=[j_ for j_ in range(len(mu_)) if mu_[j_]<0.999][:kk_]
    if len(keep_)<4: return None
    Phi_=Ph_[:,keep_]*mu_[keep_][None,:]
    idf_=torch.tensor([math.log(len(docs_)/df_[w_]) for w_ in sorted(voc_,key=voc_.get)],dtype=torch.float32)
    return voc_,torch.tensor(Phi_,dtype=torch.float32),float(mu_[keep_[0]]),int(len(mu_)-len([j_ for j_ in range(len(mu_)) if mu_[j_]<0.999])),idf_
def _spec_pass(SC_,txts_):
    voc_,Phi_=SC_[0],SC_[1]; out_=torch.zeros(len(txts_),Phi_.shape[1])
    for j_,t_ in enumerate(txts_):
        ix_=[voc_[w_.group(0)] for w_ in _UWORD.finditer(t_) if w_.group(0) in voc_]
        if ix_: wt_=SC_[4][ix_]; out_[j_]=(Phi_[ix_]*wt_[:,None]).sum(0)/wt_.sum()   # IDF-weighted: rare words carry identity
    return F.normalize(out_,dim=-1)
def _mix_mask(ctx_,tgt_,keep_=None):
    """True for target bytes NOT inside a word (3+ chars) that also occurs in the context - so copying a shared word earns nothing
    (keep_: a test for shared words that DO count - rare words, i.e. a shared entity name, which is the link itself)"""
    sh_={w_.group(0) for w_ in _UWORD.finditer(ctx_) if len(w_.group(0))>=3 and not (keep_ and keep_(w_.group(0)))}; k_=np.ones(len(tgt_),bool)
    for w_ in _UWORD.finditer(tgt_):
        if w_.group(0) in sh_: k_[w_.start():w_.end()]=False
    return k_

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_inf_gen': 'platypus.inference.rem',
}
