"""Rule-free units: the learned lexicon and the MDL reader."""
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
    _INFM,
)

_INF_REL=["{a} works with {b}.","{a} lives near {b}.","{a} reports to {b}.","{a} is a partner of {b}.","{a} trained under {b}.","{a} is linked to {b}."]
_INF_SYL=['ka','to','mi','ra','ve','lo','du','sen','tar','bel','nor','qui','fa','zo','pel','dra','mon','ris','tev','gal','hun','cor','vin','sal','bri','ed','an','ul','or','is']
_INF_NAME=_re_inf.compile(rb'\b[A-Z][a-z]{3,11}\b')
_INF_SENT=_re_inf.compile(rb'(?<=[.!?])\s+')
_INF_TRIP=_re_inf.compile(rb'(?<![A-Za-z])([A-Z][a-z]{3,11}) ((?:[a-z]+ ){0,3}[a-z]+) ([A-Z][a-z]{3,11})(?![A-Za-z])')
_INF_NAME_RX=_INF_NAME; _INF_TRIP_RX=_INF_TRIP                                  # the hand-written patterns (kept for question-time readouts / prompts)

# ---- cg20s RULE-FREE UNITS: a LEARNED lexicon replaces the capitalised-name pattern. A word type becomes a unit the first time the model,
#      reading it in context, cannot COMPLETE it (mean surprisal of its bytes after the first, in the top inf_units_q of recent words).
_UNITS={'lex':set()}
_UWORD=_re_inf.compile(rb"(?<![\x80-\xff])[A-Za-z0-9][A-Za-z0-9'-]*(?![\x80-\xff])")
_UEVT=_re_inf.compile(rb"(?<![\x80-\xff])[A-Za-z0-9][A-Za-z0-9'-]*(?![\x80-\xff])|[.!?\n]")
class _UnitMatch:
    def __init__(s,g,sp): s._g=g; s._sp=sp
    def group(s,i=0): return s._g[i]
    def start(s,i=0): return s._sp[i][0]
    def end(s,i=0): return s._sp[i][1]
def _u_rare(w_):
    """'all' mode (the word-level inference layer): ANY word of 3+ letters whose window frequency so far is at most the rarity limit
    (relational-word demotion: a word seen mostly in the RELATION slot of fact records, rarely as an argument, is not an argument)"""
    if len(w_)<3: return False
    if _UNITS.get('sal_on'):                                                    # SALIENCE: an entity is a word the model finds surprising in context (mean bits
        sv_=_UNITS['sal'].get(w_)                                               #   per byte in the top share), however frequent it is
        if sv_ and sv_[1]>=_UNITS['sal_min']:
            if sv_[0]/sv_[1]<_UNITS['sal_thr']: return False
            if _UNITS.get('rdem'):
                c_=_UNITS['slotc'].get(w_)
                if c_ and c_[1]>=3 and c_[1]>2*c_[0]: return False
            return True
    if _UNITS.get('ent_on'):                                                    # ENTITIES BY COMPRESSION (inf_ent_mdl): judged by the bits they save, not by rarity
        if w_ in _UNITS['ent']: pass
        elif w_ in _UNITS['entno']: return False
        elif _UNITS['df'].get(w_,0)>_UNITS['mx']: return False                 #   (not yet judged: rarity only proposes it)
    elif _UNITS['df'].get(w_,0)>_UNITS['mx']: return False
    if _UNITS.get('rdem'):
        c_=_UNITS['slotc'].get(w_)
        if c_ and c_[1]>=3 and c_[1]>2*c_[0]: return False
    return True
class _UnitName:
    """drop-in for the name regex: matches only word types in the learned lexicon ('all' mode: any statistically rare word)"""
    def finditer(s,b_):
        if _UNITS.get('all'):
            for m_ in _UWORD.finditer(b_):
                if _u_rare(m_.group(0)): yield m_
            return
        L_=_UNITS['lex']
        for m_ in _UWORD.finditer(b_):
            if m_.group(0) in L_: yield m_
    def findall(s,b_): return [m_.group(0) for m_ in s.finditer(b_)]
    def search(s,b_): return next(s.finditer(b_),None)
    def fullmatch(s,b_): return _UWORD.fullmatch(b_) if (_u_rare(b_) if _UNITS.get('all') else b_ in _UNITS['lex']) else None
class _UnitTrip:
    """drop-in for the fact pattern: two consecutive units with 1-4 non-unit words between them (single spaces, no punctuation).
    'all' mode: the clause's two RAREST words (lowest window frequency) are its arguments, the 1-6 words between them its relation"""
    def finditer(s,b_):
        if _UNITS.get('all'):
            if not _RDR.get('on'): yield from s._base(b_); return                 # (reader off: exactly the earlier behaviour)
            bs_={frozenset((m_.group(1),m_.group(3))):m_ for m_ in s._base(b_) if not (_RDR.get('drop') and _rdr_prefix(b_,m_.start(1)) in _RDR['drop'])}
            if _RDR.get('on') and _RDR['acc']:                                  # MDL READER: records from proposal schemes that paid for themselves; a kept
                sh_=_RDR.get('cb',set())                                        #   scheme's record replaces the base reading of the same pair when only it
                for a_,r_,c_,sid_ in _rdr_records(b_):                          #   uses a shared relation (cleft: 'It is B that A r' -> not 'B that A')
                    k_=frozenset((a_,c_)); m0_=bs_.get(k_)
                    if _RDR.get('v2') and (r_ not in sh_ or (m0_ is not None and (m0_.group(2).endswith(b' '+r_) or m0_.group(2)==r_))): continue   # v2: per-record code; no truncation
                    if sid_ in _RDR['acc'] and (m0_ is None or (m0_.group(2) not in sh_ and r_ in sh_ and not getattr(m0_,'_rdr',0))):
                        m1_=_UnitMatch((b_,a_,r_,c_),((0,len(b_)),)*4); m1_._rdr=1; bs_[k_]=m1_
            if _RDR.get('km') and 'seg' in _RDR['acc']:                         # READER v3: core relations with their modifiers stripped
                for a_,core_,c_,mods_ in _rdr_seg(b_,_RDR.get('cn',{}),_RDR['km']):
                    if any(b'mod:'+m_ in _RDR.get('drop',()) for m_ in mods_):
                        bs_.pop(frozenset((a_,c_)),None); continue              #   a modifier that states no current fact: no record at all
                    k_=frozenset((a_,c_)); m0_=bs_.get(k_)
                    if m0_ is None or m0_.group(2) not in _RDR.get('cb',()):
                        m1_=_UnitMatch((b_,a_,core_,c_),((0,len(b_)),)*4); m1_._rdr=1; m1_._mods=mods_; bs_[k_]=m1_
            yield from bs_.values()
            return
        yield from s._lex(b_)
    def _base(s,b_):
            for cl_ in _re_inf.finditer(rb'[^.,;:!?()\"]+',b_):
                ws_=[m_ for m_ in _UWORD.finditer(cl_.group(0)) if _u_rare(m_.group(0))]
                if len(ws_)<2: continue
                if _UNITS.get('sal_on'):
                    sk_=lambda m_: (-(_UNITS['sal'].get(m_.group(0),[0.0,1])[0]/max(1,_UNITS['sal'].get(m_.group(0),[0.0,1])[1])),m_.start())
                    two_=sorted(sorted(ws_,key=sk_)[:2],key=lambda m_: m_.start())
                else: two_=sorted(sorted(ws_,key=lambda m_: (_UNITS['df'].get(m_.group(0),0),m_.start()))[:2],key=lambda m_: m_.start())
                a_,c_=two_; o_=cl_.start(); mid_=cl_.group(0)[a_.end():c_.start()]
                if a_.group(0)==c_.group(0) or not _re_inf.fullmatch(rb" \S+(?: \S+){0,5} ",mid_): continue
                yield _UnitMatch((b_[o_+a_.start():o_+c_.end()],a_.group(0),mid_[1:-1],c_.group(0)),((o_+a_.start(),o_+c_.end()),(o_+a_.start(),o_+a_.end()),(o_+a_.end()+1,o_+c_.start()-1),(o_+c_.start(),o_+c_.end())))
    def _lex(s,b_):
        L_=_UNITS['lex']; ws_=list(_UWORD.finditer(b_)); us_=[k_ for k_,m_ in enumerate(ws_) if m_.group(0) in L_]
        for x_,y_ in zip(us_,us_[1:]):
            if not 1<=y_-x_-1<=4: continue
            a_,c_=ws_[x_],ws_[y_]; mid_=b_[a_.end():c_.start()]
            if not _re_inf.fullmatch(rb" [^\s.,;:!?()\"]+(?: [^\s.,;:!?()\"]+){0,3} ",mid_): continue
            yield _UnitMatch((b_[a_.start():c_.end()],a_.group(0),mid_[1:-1],c_.group(0)),((a_.start(),c_.end()),(a_.start(),a_.end()),(a_.end()+1,c_.start()-1),(c_.start(),c_.end())))
    def search(s,b_): return next(s.finditer(b_),None)
# ---- MDL READER (inf_reader): reading itself is selected by description length. Generic proposal SCHEMES (no grammar, no word lists) turn a
#      stored sentence into candidate records (a, relation, b); a scheme is kept only while its records pay for themselves: a record whose relation
#      is already in the relation codebook (shared by >= 2 argument pairs read by the base scheme) is coded cheaply, a record with a new relation
#      costs bits, and every scheme pays a fixed model cost. Schemes: skip:T (drop the first 1-2 tokens T between two consecutive rare words, punctuation
#      included; one scheme per token sequence T), after (relation = the words after the second rare word: object first), pron:w (a short word w opening a sentence stands for the
#      last rare word of the sentence before; sentence pairs are stored for this). The base scheme (two rarest words of a clause) is always on.
_RDR={'on':False,'acc':set(),'gain':{},'drop':set()}
def _rdr_prefix(b_,st_):
    """the context a record is stated in: up to 3 tokens before its first argument, back to the last punctuation (lower-cased)"""
    h_=_re_inf.split(rb'[.,;:!?()"]',b_[:st_])[-1]; return b' '.join(t_.lower() for t_ in _UWORD.findall(h_)[-3:])
_RTOK=_re_inf.compile(rb"[A-Za-z0-9][A-Za-z0-9'-]*|[,;:()\"]")
def _rdr_word(t_): return bool(_re_inf.fullmatch(rb"[A-Za-z0-9][A-Za-z0-9'-]*",t_))
def _rdr_records(b_):
    """candidate records (a, relation, b, scheme) from a stored sentence (or sentence pair) - every scheme, accepted or not"""
    out_=[]; parts_=[p_ for p_ in _INF_SENT.split(b_) if p_]
    for p_ in parts_:
        tk_=[m_.group(0) for m_ in _RTOK.finditer(p_)]; rr_=[k_ for k_,t_ in enumerate(tk_) if _rdr_word(t_) and _u_rare(t_)]
        for x_,y_ in zip(rr_,rr_[1:]):
            a_,c_=tk_[x_],tk_[y_]
            if a_==c_: continue
            mid_=tk_[x_+1:y_]
            for k_ in (1,2):                                                    # skipK: drop the first K tokens between the two rare words
                m2_=mid_[k_:]
                if 1<=len(m2_)<=6 and all(_rdr_word(t_) for t_ in m2_): out_.append((a_,b' '.join(m2_),c_,'skip:'+b' '.join(mid_[:k_]).decode('latin-1')))   # the skipped tokens name the scheme
            if 1<=len(mid_)<=3:                                                 # after: object first, relation after the second rare word
                af_=[]; e_=len(tk_)
                for q_ in range(y_+1,len(tk_)):
                    t_=tk_[q_]
                    if not _rdr_word(t_) or _u_rare(t_): e_=q_; break
                    af_.append(t_)
                if 1<=len(af_)<=4 and (e_==len(tk_) or not _rdr_word(tk_[e_])): out_.append((c_,b' '.join(af_),a_,'after'))
    if len(parts_)>=2:                                                          # pron:w - a short word opening the second sentence's clause
        tp_=[m_.group(0) for m_ in _RTOK.finditer(parts_[-2])]; lr_=[t_ for t_ in tp_ if _rdr_word(t_) and _u_rare(t_)]
        tq_=[m_.group(0) for m_ in _RTOK.finditer(parts_[-1])]
        if lr_ and tq_:
            j_=2 if (len(tq_)>=3 and tq_[1]==b',') else 0                       # one fronted word and a comma
            w_=tq_[j_]
            if 2<=len(w_)<=4 and w_.isalpha() and w_.islower() and not _u_rare(w_):
                rel_=[]
                for t_ in tq_[j_+1:]:
                    if not _rdr_word(t_): break
                    if _u_rare(t_):
                        if 1<=len(rel_)<=6: out_.append((lr_[-1],b' '.join(rel_),t_,'pron:'+w_.decode('latin-1')))
                        break
                    rel_.append(t_)
    return out_
def _rdr_seg(b_,cn_,km_):
    """READER v3 - CORE + MODIFIERS: between two consecutive rare words, the span is split into prefix | core | suffix, with at most one interior
    gap of 1-2 words; the core must be a codebook relation; only KNOWN modifiers (km_: seen around >= 3 different cores) may be stripped
    (punctuation is free). Best split = most bits for the core minus 0.5 bit per stripped modifier. Returns (a, core, b, modifiers) records;
    with km_ None, returns every candidate split (for learning which modifiers are general)."""
    out_=[]
    for p_ in [q_ for q_ in _INF_SENT.split(b_) if q_]:
        tk_=[m_.group(0) for m_ in _RTOK.finditer(p_)]; rr_=[k_ for k_,t_ in enumerate(tk_) if _rdr_word(t_) and _u_rare(t_)]
        for x_,y_ in zip(rr_,rr_[1:]):
            a_,c_=tk_[x_],tk_[y_]; mid_=tk_[x_+1:y_]; L_=len(mid_)
            if a_==c_ or not 2<=L_<=10: continue
            best_=None; cands_=[]
            for i_ in range(L_):
                for j_ in range(i_+1,L_+1):
                    for g0_,g1_ in [(None,None)]+[(k_,k_+g_) for k_ in range(i_+1,j_-1) for g_ in (1,2) if k_+g_<j_]:
                        cw_=mid_[i_:j_] if g0_ is None else mid_[i_:g0_]+mid_[g1_:j_]
                        if not cw_ or not all(_rdr_word(t_) for t_ in cw_): continue
                        core_=b' '.join(cw_)
                        if core_ not in cn_: continue
                        mods_=[m_ for m_ in (b' '.join(t_ for t_ in mid_[:i_] if _rdr_word(t_)),b' '.join(mid_[g0_:g1_]) if g0_ is not None else b'',b' '.join(t_ for t_ in mid_[j_:] if _rdr_word(t_))) if m_]
                        if not mods_: continue                                  # nothing stripped: the base reading already says this
                        if km_ is None: cands_.append((core_,mods_)); continue
                        if any(m_ not in km_ for m_ in mods_): continue
                        sc_=math.log2(1+cn_[core_])-0.5*len(mods_)
                        if best_ is None or sc_>best_[0]: best_=(sc_,core_,tuple(mods_))
            if km_ is None: out_.append((a_,c_,cands_))
            elif best_ is not None and best_[0]>0: out_.append((a_,best_[1],c_,best_[2]))
    return out_
def _rdr_update(cfg,S_):
    """score every proposal scheme on the stored sentences; keep those that pay for themselves (two-part code, bits)"""
    if not _RDR.get('on'): return ''
    its_=[it_ for it_ in S_['it'] if it_.get('k')!='inf' and not it_.get('sup')][-int(getattr(cfg,'inf_reader_n',20000)):]
    base_=set(); Cb_={}; Ob_={}; Px_={}
    for it_ in its_:
        for m_ in _UnitTrip()._base(it_['s']):
            base_.add(frozenset((m_.group(1),m_.group(3)))); Cb_.setdefault(m_.group(2),set()).add((m_.group(1),m_.group(3)))
            Ob_.setdefault((m_.group(1),m_.group(2)),set()).add(m_.group(3)); Px_.setdefault(_rdr_prefix(it_['s'],m_.start(1)),set()).add((m_.group(1),m_.group(2),m_.group(3)))
    dr_=set(); dseg_=[]                                                         # CONTEXT SCHEMES ('this context does not state a fact'): kept when the records
    ca_=[k_ for k_,v_ in Ob_.items() if len(v_)>=2]; c0_=sum(len(Ob_[k_]) for k_ in ca_)/max(1,sum(len(v_) for v_ in Ob_.values()))   #   stated in it
    for px_,rs0_ in Px_.items():                                                #   contradict the same subject and relation stated elsewhere far more often than chance
        if not px_ or len(rs0_)<int(getattr(cfg,'inf_reader_dmin',4)): continue
        cf_=sum(1 for a_,r_,c_ in rs0_ if len(Ob_.get((a_,r_),()))>=2)/len(rs0_)
        g_=len(rs0_)*(cf_*math.log2(max(cf_,1e-9)/max(c0_,1e-3))) - float(getattr(cfg,'inf_reader_cost',16.0))   # bits: KL of the conflict rate from the base rate
        if cf_>=0.5 and g_>0: dr_.add(px_); dseg_.append(f"'{px_.decode('latin-1')}' {len(rs0_)} records, conflict {cf_:.2f} (base {c0_:.2f}), {g_:+.0f} bits")
    _RDR['drop']=dr_
    if int(getattr(cfg,'inf_reader_recur',0)):                                 # RECURRING CODEBOOK: a relation counts only through argument pairs read from >= 2
        pc_={}                                                                  #   stored sentences (the seen-twice rule, applied to relations: junk readings of
        for it_ in its_:                                                        #   ordinary text rarely link the same pair twice)
            for m_ in _UnitTrip()._base(it_['s']): k0_=(m_.group(1),m_.group(2),m_.group(3)); pc_[k0_]=pc_.get(k0_,0)+1
        Cb_={}
        for (a0_,r0_,c0b_),n0_ in pc_.items():
            if n0_>=2: Cb_.setdefault(r0_,set()).add((a0_,c0b_))
    al_=[x_ for it_ in its_ for x_ in _rdr_records(it_['s'])]; At_={}
    for a_,r_,c_,sid_ in al_: At_.setdefault(r_,set()).add((a_,c_))           # attestations OUTSIDE the base construction
    Cn_={r_:len(v_) for r_,v_ in Cb_.items() if len(v_)>=2 and r_ in At_}       # CODEBOOK: a relation that recurs (>= 2 pairs) AND in more than one construction
    R_={}; bk_=set(frozenset((a_,c_)) for r_,v_ in Cb_.items() if r_ in Cn_ for a_,c_ in v_)
    v2_=bool(_RDR.get('v2')); br0_={}
    if v2_:
        for r_,v_ in Cb_.items():
            for a_,c_ in v_: br0_.setdefault(frozenset((a_,c_)),set()).add(r_)
    for a_,r_,c_,sid_ in al_:
        if frozenset((a_,c_)) in bk_: continue                                   # nothing new: the base scheme already reads this pair with a codebook relation
        if v2_ and any(b0_.endswith(b' '+r_) for b0_ in br0_.get(frozenset((a_,c_)),())): continue   # v2: a truncation of the base reading is not a new reading
        R_.setdefault(sid_,set()).add((a_,r_,c_))
    fam_={}
    for sid_ in R_: fam_[sid_.split(':')[0]]=fam_.get(sid_.split(':')[0],0)+1
    if not v2_: fam_={}
    km_=set(); mseg_=''
    if int(getattr(cfg,'inf_reader',0))>=3:                                     # READER v3: which modifiers are GENERAL (seen around >= 3 different cores)
        Mc_={}
        for it_ in its_:
            for a_,c_,cs_ in _rdr_seg(it_['s'],Cn_,None):
                for core_,mods_ in cs_:
                    for m_ in mods_: Mc_.setdefault(m_,set()).add(core_)
        km_=set(m_ for m_,v_ in Mc_.items() if len(v_)>=int(getattr(cfg,'inf_reader_kmod',3)))
        Ms_={}
        for it_ in its_:
            for a_,core_,c_,mods_ in _rdr_seg(it_['s'],Cn_,km_):
                if frozenset((a_,c_)) in bk_: continue
                R_.setdefault('seg',set()).add((a_,core_,c_))
                for m_ in mods_: Ms_.setdefault(m_,set()).add((a_,core_,c_))
        for m_,rs0_ in Ms_.items():                                             # a MODIFIER whose records contradict current facts far more than chance
            if len(rs0_)<int(getattr(cfg,'inf_reader_dmin',4)): continue        #   does not state a current fact ('used to', 'no longer', 'not')
            cf_=sum(1 for a_,r_,c_ in rs0_ if len(Ob_.get((a_,r_),set())|{c_})>=2)/len(rs0_)
            g_=len(rs0_)*(cf_*math.log2(max(cf_,1e-9)/max(c0_,1e-3)))-float(getattr(cfg,'inf_reader_cost',16.0))
            if cf_>=0.5 and g_>0: dr_.add(b'mod:'+m_); dseg_.append(f"modifier '{m_.decode('latin-1')}' {len(rs0_)} records, conflict {cf_:.2f}, {g_:+.0f} bits")
        mseg_=f" | MODIFIERS: {len(km_)} general (e.g. "+", ".join(repr(m_.decode('latin-1')) for m_ in sorted(km_,key=lambda m_: -len(Mc_[m_]))[:8])+")"
        fam_=fam_ if v2_ else {}; fam_['seg']=1
    mc_=float(getattr(cfg,'inf_reader_cost',16.0)); nw_=float(getattr(cfg,'inf_reader_newrel',4.0)); acc_=set(); seg_=[]
    for sid_,rs_ in R_.items():
        kn_=[r_ for r_ in rs_ if r_[1] in Cn_]
        mc1_=(math.log2(1+fam_[sid_.split(':')[0]])+2.0) if v2_ else mc_          # v2: model cost = bits to name the scheme within its family
        g_=sum(math.log2(1+Cn_[r_[1]]) for r_ in kn_)-(0.0 if v2_ else nw_*(len(rs_)-len(kn_)))-mc1_   # v2: records with a new relation are simply not used
        if g_>0: acc_.add(sid_)
        seg_.append((g_,f"{sid_} {len(rs_)} records, {len(kn_)/max(1,len(rs_)):.2f} known relation, {g_:+.0f} bits{' KEPT' if g_>0 else ''}"))
    if v2_ and dr_:                                                             # v2: RETROACTIVE - records already made in a context now known to state no fact
        it0_=S_['it']; nfr_=0; ncf_=0; bad_=set()
        for f_ in S_.get('facts',[]):
            if f_.get('freed') or f_.get('derived'): continue
            i0_=f_.get('i')
            if isinstance(i0_,int) and i0_<len(it0_):
                b0_=it0_[i0_]['s']; j0_=b0_.find(f_['A'].encode('latin-1'))
                if j0_>=0 and _rdr_prefix(b0_,j0_) in dr_: f_['freed']=True; nfr_+=1; bad_.add((f_['A'],f_['B'])); it0_[i0_]['sup']=True
        for c_ in S_.get('conc',[]):
            if not c_.get('freed') and any(isinstance(i_,int) and i_<len(it0_) and it0_[i_].get('sup') for i_ in c_.get('src',())): c_['freed']=True; ncf_+=1
        if nfr_ or ncf_: dseg_.append(f"withdrawn: {nfr_} fact records, {ncf_} conclusions")
    _RDR['acc']=acc_; _RDR['cb']=set(Cn_); _RDR['cn']=dict(Cn_); _RDR['km']=km_; _RDR['gain']={x_.split(' ')[0]:g_ for g_,x_ in seg_}
    seg_=[x_ for _,x_ in sorted(seg_,key=lambda z_: -z_[0])]
    return (f"  [inference MDL READER v3]{mseg_}" if mseg_ else "")+f"  [inference MDL READER] contexts that state no fact: {len(dr_)} kept"+(" ("+"; ".join(dseg_[:4])+")" if dseg_ else "")+f" | {len(acc_)} of {len(R_)} proposal schemes kept (codebook {len(Cn_)} relations seen in more than one construction, of {len(Cb_)} base readings) | "+" | ".join(seg_[:10])
def _ent_ok(n_,df_,mx_):
    """argument test used by linking and fact records: compression-judged entities when inf_ent_mdl is on, else the rarity limit"""
    return _u_rare(n_) if _UNITS.get('ent_on') else df_.get(n_,0)<=mx_
def _ent_update(cfg,S_):
    """ENTITIES BY COMPRESSION: every content word in a clause is a candidate argument; a candidate X is an entity when treating it as a node
    SAVES BITS on the candidate records (X, words between, Y): leave-one-out code of the partner given X and the relation, against the partner
    given the relation alone, minus the cost of naming X. Function words predict nothing (gain ~0); a one-off word has no held-out evidence
    (gain 0); a name with recurring specific partners gains - at ANY frequency. Re-judged every sleep on the recent stored sentences."""
    its_=[it_ for it_ in S_['it'] if it_.get('k')!='inf' and not it_.get('sup') and not it_.get('gone')][-int(getattr(cfg,'inf_ent_n',20000)):]
    n_={}; nr_={}; nry_={}
    for it_ in its_:
        for cl_ in _re_inf.finditer(rb'[^.,;:!?()"]+',it_['s']):
            ws_=_UWORD.findall(cl_.group(0))
            for i_ in range(len(ws_)):
                if len(ws_[i_])<3 or not ws_[i_][:1].isalpha(): continue
                for j_ in range(i_+2,min(len(ws_),i_+8)):
                    if len(ws_[j_])<3 or not ws_[j_][:1].isalpha() or ws_[j_]==ws_[i_]: continue
                    r_=b' '.join(ws_[i_+1:j_]); k_=(ws_[i_],r_,ws_[j_]); n_[k_]=n_.get(k_,0)+1
    for (x_,r_,y_),c_ in n_.items():
        nr_[r_]=nr_.get(r_,0)+c_; nry_[(r_,y_,1)]=nry_.get((r_,y_,1),0)+c_; nry_[(r_,x_,0)]=nry_.get((r_,x_,0),0)+c_
    nxr_={}; ny_={}; nt_=0
    for (x_,r_,y_),c_ in n_.items():
        nxr_[(x_,r_,0)]=nxr_.get((x_,r_,0),0)+c_; nxr_[(y_,r_,1)]=nxr_.get((y_,r_,1),0)+c_
        ny_[(y_,1)]=ny_.get((y_,1),0)+c_; ny_[(x_,0)]=ny_.get((x_,0),0)+c_; nt_+=c_
    g_={}; a_=1.0                                                               # leave-one-out at BOTH levels: the partner given the relation (baseline) and
    for (x_,r_,y_),c_ in n_.items():                                            #   given the relation AND X; X earns only what it adds beyond the relation
        g_.setdefault(x_,0.0); g_.setdefault(y_,0.0)
        if c_<2: continue                                                       #   no held-out evidence: saves nothing
        for X_,Y_,role_ in ((x_,y_,0),(y_,x_,1)):                                #   X as subject (predicts the object) and as object (predicts the subject)
            pu_=(ny_[(Y_,1-role_)]+0.5)/(nt_+0.5*len(ny_))
            pr_=(nry_[(r_,Y_,1-role_)]-1+a_*pu_)/(nr_[r_]-1+a_)
            pl_=(c_-1+a_*pr_)/(nxr_[(X_,r_,role_)]-1+a_)
            g_[X_]+=c_*math.log2(max(pl_,1e-12)/max(pr_,1e-12))
    cst_=float(getattr(cfg,'inf_ent_cost',16.0))
    bpr_=float(getattr(cfg,'inf_ent_bpr',0.0)); nrec_={}
    if bpr_>0:                                                                  # optional: bits saved PER RECORD (specificity), not only in total
        for (x_,r_,y_),c_ in n_.items(): nrec_[x_]=nrec_.get(x_,0)+c_; nrec_[y_]=nrec_.get(y_,0)+c_
    E_=set(w_ for w_,v_ in g_.items() if v_>=cst_ and (bpr_<=0 or v_/max(1,nrec_.get(w_,1))>=bpr_)); N_=set(g_)-E_
    _UNITS['ent']=E_; _UNITS['entno']=N_; _UNITS['ent_on']=True; _UNITS['entgain']=g_
    hi_=sorted(E_,key=lambda w_: -_UNITS['df'].get(w_,0))[:8]
    return (f"  [inference ENTITIES BY COMPRESSION] {len(g_)} candidates judged on {len(its_)} sentences, {len(E_)} entities (save >= {cst_:.0f} bits), {len(N_)} rejected | "
            f"most frequent entities: "+", ".join(f"{w_.decode('latin-1')} (df {_UNITS['df'].get(w_,0)}, {g_[w_]:.0f} bits)" for w_ in hi_)
            +" | most frequent rejected: "+", ".join(f"{w_.decode('latin-1')} ({g_[w_]:.0f} bits)" for w_ in sorted(N_,key=lambda w_: -_UNITS['df'].get(w_,0))[:6]))
def _units_on(cfg): return int(getattr(cfg,'inf_units',0))>0
@torch.no_grad()
def _units_update(model,cfg,X,S_):
    """read a few rows of the batch with the CURRENT model (first encounter, before it trains on them) and flag word types it cannot complete"""
    S_['un']=S_.get('un',0)+1
    if S_['un']%max(1,int(getattr(cfg,'inf_units_every',4))): return
    R_=min(int(getattr(cfg,'inf_units_rows',8)),X.shape[0]); Xs_=X[:R_].long()
    sv4_=_carry_save(); _CARRY.update(on=False,st={},new={}); sv_=_INFM['on']; _INFM['on']=False; fz_=_INFM.pop('force',None); was_=model.training; model.eval()
    try: lp_=F.log_softmax(model(Xs_[:,:-1],_XDG['cur']).float(),-1)
    finally:
        _carry_load(sv4_); _INFM['on']=sv_; model.train(was_)
        if fz_ is not None: _INFM['force']=fz_
    bits_=(-lp_.gather(-1,Xs_[:,1:,None]).squeeze(-1)/math.log(2)).cpu().numpy(); Xc_=Xs_.to('cpu',torch.uint8).numpy()
    rv_=S_.setdefault('ures',[]); L_=S_.setdefault('lex',set()); _UNITS['lex']=L_; c0_=int(getattr(cfg,'inf_units_ctx',32)); q_=float(getattr(cfg,'inf_units_q',0.2))
    thr_=float(np.quantile(rv_,1-q_)) if len(rv_)>=int(getattr(cfg,'inf_units_warm',2000)) else None; nf_=0
    kf9_=int(getattr(cfg,'inf_sal_first',0)); qr9_=float(np.quantile(rv_[-4000:],1-float(getattr(cfg,'inf_sal_q',0.3)))) if (kf9_>0 and len(rv_)>=300) else None
    for r_ in range(R_):
        w_=bytes(Xc_[r_])
        for m_ in _UWORD.finditer(w_):
            a_,e_=m_.start(),m_.end()
            if e_-a_<3 or a_<c0_ or e_>=len(w_)-1: continue
            c_=float(bits_[r_,a_:e_-1].mean())            # bytes a+1..e-1 are predicted at positions a..e-2
            sg9_=S_.setdefault('sal',{}).setdefault(m_.group(0),[0.0,0])
            if kf9_<=0: sg9_[0]+=c_; sg9_[1]+=1                           # (SALIENCE: running mean surprise per word type)
            elif qr9_ is not None and sg9_[1]<kf9_: sg9_[0]+=c_-qr9_; sg9_[1]+=1   # inf_sal_first K: its FIRST K encounters, each RELATIVE to the top-q surprise of words read
                                                                          #   at that moment (cg20z fix: absolute bits froze every common word as salient, since words
                                                                          #   first met while the model was untrained all look surprising); salient = mean >= 0
            rv_.append(c_)                                # RECENT words only (FIFO): the threshold tracks the model as it is NOW (a long reservoir lagged and missed names)
            w0_=m_.group(0)
            if str(getattr(cfg,'inf_units_mode','rank'))=='mdl':   # MDL CHUNKING: a word type becomes a unit when treating it as ONE symbol would have saved bits:
                g0_=S_.setdefault('ugain',{}).setdefault(w0_,[0,0.0])   #   every repeat after the first saves its byte surprisal minus the cost of a pointer into the lexicon
                if g0_[0]>0: g0_[1]+=c_*(e_-a_-1)-math.log2(len(L_)+1024)
                g0_[0]+=1
                if g0_[1]>=float(getattr(cfg,'inf_units_gain',16.0)) and w0_ not in L_ and w0_ not in S_.get('lexdem',()): L_.add(w0_); nf_+=1
            elif thr_ is not None and c_>=thr_ and w0_ not in L_ and w0_ not in S_.get('lexdem',()): L_.add(w0_); nf_+=1
    del rv_[:-int(getattr(cfg,'inf_units_win',3000))]
    S_['unf']=S_.get('unf',0)+nf_
