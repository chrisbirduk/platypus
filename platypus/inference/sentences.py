"""The sentence store: adding, evicting, relations."""
import numpy as np
import re as _re_inf
import torch
from platypus.common import (
    _set_global,
)
from platypus.inference.store import (
    _inf_sync,
)
from platypus.inference.units import (
    _INF_NAME, _INF_SENT, _INF_TRIP, _RDR, _UNITS, _UWORD, _UnitName, _UnitTrip, _rdr_records, _units_on, _units_update,
    _chk_count, _chk_records,
)

def _inf_store(model):
    return model.__dict__.setdefault('_infS',{'it':[],'df':{},'idx':{},'new':[],'nwin':0,'pairs':set(),'conc':[],'stat':{},'ring':[],'lw':set(),'strict':bool(int(getattr(model.cfg,'inf_name_strict',0)))})
@torch.no_grad()
def inf_store_add(model,cfg,X,dom,mask=None):
    """add windows X (rows of bytes, CPU-converted here) to the sentence store; mask (same shape as X[:,1:]): verified positions only"""
    _inf_sync(cfg,'store add, before ('+('closed loop' if mask is not None else 'real text')+')')
    S_=_inf_store(model)
    if _units_on(cfg):
        if not isinstance(_INF_NAME,_UnitName): _set_global('_INF_NAME',_UnitName()); _set_global('_INF_TRIP',_UnitTrip()); _UNITS['on']=True   # (also after a checkpoint restore)
        _UNITS['lex']=S_.setdefault('lex',set())
        if str(getattr(cfg,'inf_units_mode','rank'))=='all':
            _UNITS.update(all=True,df=S_['df'],mx=int(cfg.inf_df),rdem=bool(int(getattr(cfg,'inf_relword_demote',0))),slotc=S_.setdefault('slotc',{}))
            if int(getattr(cfg,'inf_salience',0)):                              # SALIENCE MODE: entities, clause arguments and store admission by the model's
                if mask is None: _units_update(model,cfg,X,S_)                  #   own surprise in context, not by rarity in the store
                rv0_=S_.get('ures',[])
                if len(rv0_)>=int(getattr(cfg,'inf_units_warm',300)):
                    _UNITS.update(sal_on=True,sal=S_['sal'],sal_thr=(0.0 if int(getattr(cfg,'inf_sal_first',0))>0 else float(np.quantile(rv0_,1-float(getattr(cfg,'inf_sal_q',0.3))))),sal_min=int(getattr(cfg,'inf_sal_min',2)))
        elif mask is None: _units_update(model,cfg,X,S_)                       # real text only (closed-loop bytes are the model's own)
    Xc_=X.detach().to('cpu',torch.uint8).numpy()
    if mask is not None:
        Mc_=(mask.detach()>0).to('cpu').numpy(); Xc_=Xc_.copy(); Xc_[:,1:][~Mc_]=0                # unverified bytes blanked
    cap_=int(cfg.inf_store_cap); chk_=int(getattr(cfg,'inf_reader_chunk',0))
    for r_ in range(Xc_.shape[0]):
        w_=bytes(Xc_[r_]); S_['nwin']+=1
        if chk_ and mask is None: _chk_count(cfg,S_,w_)                       # 11 Nov FAST SEQUENCE CHANNEL: counts over the real text read
        S_['ring'].append((w_,int(dom))); del S_['ring'][:-512]                    # recent windows (real / verified-only bytes): filler + anchors
        if int(getattr(cfg,'inf_sofl',0)):                                       # SLEEP-ONLY FEATURE LEARNING: the visit's windows, buffered (reservoir) for the sleep
            sb_=S_.setdefault('sbuf',[]); S_['sbn']=S_.get('sbn',0)+1; cb_=int(getattr(cfg,'inf_sofl_buf',256))
            if len(sb_)<cb_: sb_.append((w_,int(dom)))
            else:
                j1_=np.random.randint(S_['sbn'])
                if j1_<cb_: sb_[j1_]=(w_,int(dom))
        rv_=S_.setdefault('res',{}).setdefault(int(dom),[]); rn0_=S_.setdefault('resn',{}); rn0_[int(dom)]=rn0_.get(int(dom),0)+1   # PER-DOMAIN RESERVOIR (64 windows, uniform over the domain's history)
        if len(rv_)<64: rv_.append(w_)
        else:
            j0_=np.random.randint(S_['resn'][int(dom)])
            if j0_<64: rv_[j0_]=w_
        for n_ in set((_UWORD.findall(w_) if _UNITS.get('all') else _INF_NAME.findall(w_))): S_['df'][n_]=S_['df'].get(n_,0)+1
        if S_.get('strict'): S_['lw'].update(_re_inf.findall(rb'(?<![A-Za-z])[a-z]{4,12}(?![A-Za-z])',w_))   # lower-case words seen (a capitalised common word is not a name)
        w2_=_re_inf.sub(rb'^[A-Za-z]*',b'',w_[1:] if mask is not None else w_)   # a word cut by the window start is not a name
        w2_=_re_inf.sub(rb'[A-Za-z]*$',b'',w2_)                                       #   nor one cut by the window end
        w2_=_re_inf.sub(rb'[A-Za-z]*\x00+[A-Za-z]*',b'. ',w2_)                       # blanked (unverified) runs: hard sentence breaks, adjacent word fragments dropped
        pv_=None; ss_=_INF_SENT.split(w2_)
        if int(getattr(cfg,'inf_drop_cut',0)) and len(ss_)>1:                   # CUT FRAGMENTS: the window's first and last pieces are sentences cut by the window
            ss_=ss_[1:] if not w2_[:1].isupper() else ss_                       #   ('Anrisul is a partner ' with its object cut off): never stored
            if ss_ and not _re_inf.search(rb'[.!?]["\')]?\s*$',ss_[-1]): ss_=ss_[:-1]
        for s_ in ss_:
            ns_=_INF_NAME.findall(s_)
            if _RDR.get('on') and len(set(ns_))==1 and pv_ is not None and len(pv_)+len(s_)<300:   # MDL READER: a one-name sentence is stored with the
                s0_=s_; s_=pv_+b' '+s_; ns_=_INF_NAME.findall(s_); pv_=s0_           #   sentence before it (a pronoun scheme may read it)
            else: pv_=s_ if ns_ else None
            if len(set(ns_))<2 or len(s_)>300: continue
            if S_.get('nlive',len(S_['it']))>=cap_:
                if not int(getattr(cfg,'inf_store_roll',0)) or not _inf_store_evict(cfg,S_,model): continue   # ROLLING STORE: evict old unreferenced sentences
            S_['nlive']=S_.get('nlive',len(S_['it']))+1
            i_=len(S_['it']); S_['it'].append({'s':s_,'d':int(dom),'k':'real' if mask is None else 'ver','dep':0,'v':int(model.__dict__.get('_visit',0))}); S_['new'].append(i_)
            if _UNITS.get('all') and _UNITS.get('rdem'):                        # slot statistics: how often each word is an ARGUMENT vs inside a RELATION
                for m4_ in _INF_TRIP.finditer(s_):
                    for w4_ in (m4_.group(1),m4_.group(3)): c4_=_UNITS['slotc'].setdefault(w4_,[0,0]); c4_[0]+=1
                    for w4_ in _UWORD.findall(m4_.group(2)): c4_=_UNITS['slotc'].setdefault(w4_,[0,0]); c4_[1]+=1
            if int(getattr(cfg,'inf_audit',0)):                                 # EVIDENCE RECORD: subject-first triples (subject, relation, object)
                m3_=_INF_TRIP.search(s_)
                if m3_: S_.setdefault('rel',{}).setdefault((m3_.group(1),m3_.group(2)),{}).setdefault(m3_.group(3),[]).append(i_)
            for n_ in set(ns_): S_['idx'].setdefault(n_,[]).append(i_)
            if chk_:                                                            # its (filler, frame, filler) records: a SEPARATE evidence record
                RC_=S_.setdefault('relc',{})
                for a9_,r9_,b9_ in _chk_records(cfg,S_,s_): RC_.setdefault((a9_,r9_),{}).setdefault(b9_,[]).append(i_)
                if len(RC_)>int(getattr(cfg,'inf_reader_chunk_cap',1000000)):  #   (bounded: the oldest keys read once go first)
                    for k9_ in [k9_ for k9_,o9_ in RC_.items() if sum(len(v9_) for v9_ in o9_.values())<=1][:len(RC_)//4]: del RC_[k9_]
def _inf_store_evict(cfg,S_,model):
    """ROLLING SENTENCE STORE: when full, the oldest 10% of sentences from earlier visits that no live fact record, conclusion or re-entered
    conclusion depends on are emptied (their index entries removed). Evidence that was never used makes room for new evidence."""
    vis_=int(model.__dict__.get('_visit',0)); keep_=set()
    for f_ in S_.get('facts',[]):
        if not f_.get('freed') and isinstance(f_.get('i'),int): keep_.add(f_['i'])
    for c_ in S_.get('conc',[]):
        if not c_.get('freed'): keep_.update(i_ for i_ in c_.get('src',()) if isinstance(i_,int))
    nt_=max(1,int(0.1*S_.get('nlive',len(S_['it'])))); ev_=[]
    for i_,it_ in enumerate(S_['it']):
        if len(ev_)>=nt_: break
        if it_.get('gone') or it_.get('k')=='inf' or i_ in keep_ or it_.get('v',vis_)>=vis_: continue
        ev_.append(i_)
    if not ev_: return False
    E_=set(ev_); nm_=set()
    if int(getattr(cfg,'inf_evict_clean',0)) and S_.get('rel'):                 # cg26i EVICTION CLEANS UP: evicted sentences leave the evidence record too (they stayed as
        nr9_=0                                                                  #   references to emptied sentences); word counts (df) are stream statistics and stay
        for k9_ in list(S_['rel']):
            ob9_=S_['rel'][k9_]
            for o9_ in list(ob9_):
                l9_=[i_ for i_ in ob9_[o9_] if i_ not in E_]; nr9_+=len(ob9_[o9_])-len(l9_)
                if l9_: ob9_[o9_]=l9_
                else: del ob9_[o9_]
            if not ob9_: del S_['rel'][k9_]
        S_['_evclean']=S_.get('_evclean',0)+nr9_
    for c_ in S_.get('conc',()):                                                # cg26q PROVENANCE: a released conclusion keeps a short copy of its two premises when their
        if c_.get('freed') and 'srct' not in c_ and any(isinstance(i_,int) and i_ in E_ for i_ in c_.get('src',())):   #   sentences leave the store, so it can still cite them
            c_['srct']=tuple(bytes(S_['it'][i_]['s'][:160]) if isinstance(i_,int) and 0<=i_<len(S_['it']) else b'' for i_ in c_['src'])
    for i_ in ev_:
        it_=S_['it'][i_]; nm_.update(_INF_NAME.findall(it_['s'])); it_['s']=b''; it_['gone']=True
    for n_ in nm_:
        l_=S_['idx'].get(n_)
        if l_ is not None:
            l2_=[i_ for i_ in l_ if i_ not in E_]
            if l2_: S_['idx'][n_]=l2_
            else: S_['idx'].pop(n_,None)
    S_['nlive']=S_.get('nlive',len(S_['it']))-len(ev_); S_['nevict']=S_.get('nevict',0)+len(ev_)
    return True
def _inf_rel(s_,a_,b_):
    """relation phrase if the sentence reads '<a> <1-6 lowercase words> <b>' (subject-first), else None"""
    w0_=rb'[^\s.,;:!?()"]+' if _UNITS.get('on') else rb'[a-z]+'
    m_=_re_inf.search(rb'(?<![A-Za-z0-9])'+_re_inf.escape(a_)+rb'((?: '+w0_+rb'){1,6}) '+_re_inf.escape(b_)+rb'(?![A-Za-z0-9])',s_)
    if m_: return m_.group(1).strip().decode('latin-1')
    if _RDR.get('on') and _RDR['acc']:
        for a2_,r2_,c2_,sid_ in _rdr_records(s_):
            if sid_ in _RDR['acc'] and a2_==a_ and c2_==b_: return r2_.decode('latin-1')
    return None
