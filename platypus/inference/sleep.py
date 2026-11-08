"""Sleep: NREM, REM, store writes, replay into the trunk."""
from platypus.common import _fail
import math
import numpy as np
import torch
import torch.nn.functional as F
from platypus.common import (
    _FIT_W,
)
from platypus.protection import (
    _kf_apply,
)
from platypus.private import (
    _HCAP, _XDG,
)
from platypus.layers import (
    _CARRY,
)
from platypus.capacity import (
    _lab,
)
from platypus.conduit import (
    _carry_load, _carry_save, _uh_load, _uh_save,
)
from platypus.inference.store import (
    _INFM, _inf_sync, _infm_corr, _infm_ids, _infm_init, _infm_keys, _infm_latch, _infm_write, _ng_zero, bpc,
)
from platypus.inference.units import (
    _INF_NAME, _INF_SYL, _RDR, _UNITS, _ent_update, _rdr_update, _u_rare, _units_on,
)
from platypus.inference.probes import (
    _INF, _inf_bench_readout, inf_probes,
)
from platypus.inference.sentences import (
    _inf_store,
)
from platypus.inference.infer import (
    _inf_truth, inf_infer,
)
from platypus.inference.evaluate import (
    _cls_measure, _inf_lps, _jd_lidar, _plat_fmt, _plat_readout, _plat_select, _rem_analogy, _rem_exam, inf_eval,
)
from platypus.inference.ledger import (
    _mem_ledger,
)
from platypus.inference.decay import (
    _hl_x,
)
from platypus.inference.closed_form import (
    _CF_VARY, _cf_apply, _gpu_gb, _ws_batch, _ws_closed_form, _ws_direction, _ws_eager,
)
from platypus.inference.supmem import (
    _sup_dreams,
)
from platypus.inference.jindex import (
    _jx_sleep,
)

def inf_sleep(model,cfg,t,ts,vals,olds_,S_slot):
    """timing wrapper: wall time of the whole sleep and of the linking step"""
    t0_=__import__('time').time(); _INF['t_link']=0.0; gd_=_gpu_gb()
    try:                                                                        # READOUT: visits per TRUE domain, so UPDATED probes are scored only after a revisit
        S9v_=_inf_store(model); td9_=int(model.__dict__.get('_s2t',{}).get(int(ts),int(t))); vv9_=int(model.__dict__.get('_visit',0))
        if '_vdom' not in S9v_: S9v_['_vdom']={}; S9v_['_vdom_from']=vv9_
        S9v_['_vdom'][td9_]=S9v_['_vdom'].get(td9_,0)+1
    except Exception: pass
    try: torch.cuda.reset_peak_memory_stats() if gd_ is not None else None
    except Exception: pass
    try:
        ds9_=float(getattr(cfg,'inf_store_downscale',0.0)); _INFM['ds_on']=ds9_>0
        if ds9_>0 and _INFM.get('touch') is not None: _INFM['touch'].zero_()
        cai9_=None
        if int(getattr(cfg,'inf_cai',0)):                                       # CAI TEST (readout only): per-chain two-hop before this sleep
            try: cai9_=_cai_score(model,cfg)
            except Exception as e9_: _fail('inference.sleep:74'); print(f"  [CAI TEST] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
        r9_=_inf_sleep_body(model,cfg,t,ts,vals,olds_,S_slot)
        if ds9_>0 and _INFM.get('M') is not None:                               # cg26i FAST-STORE DOWNSCALING (after the synaptic-homeostasis idea): rows of the fast store that no
            with torch.no_grad():                                               #   write keyed this sleep shrink by inf_store_downscale; rehearsed rows keep their strength
                tc9_=_INFM.get('touch'); M9_=_INFM['M']
                if tc9_ is None or tc9_.shape[0]!=M9_.shape[0]: tc9_=torch.zeros(M9_.shape[0],dtype=torch.bool,device=M9_.device)
                nz9_=(M9_.abs().sum(1)>0); dn9_=nz9_&~tc9_; M9_[dn9_]*=(1.0-ds9_)
                print(f"  [inference FAST-STORE DOWNSCALING] {int(dn9_.sum())} of {int(nz9_.sum())} non-empty store rows not rehearsed this sleep shrunk by {ds9_:.0%} | |M| {float(M9_.norm()):.1f}",flush=True)
        if int(getattr(cfg,'inf_plat_probe',0)):
            try:
                S9_=_inf_store(model); b9_=S9_.pop('_platb',None); a9_=_plat_readout(model,cfg,S9_)
                if a9_ is not None:
                    H9_=S9_.setdefault('_plat_log',[]); H9_.append((int(model.__dict__.get('_visit',0)),b9_,a9_))
                    print(f"  [inference PLATONIC PROBE] {a9_['n']} fixed linked pairs, trunk codes (store off) | "+(f"before this sleep's replay: {_plat_fmt(b9_)} | " if b9_ else "")+f"after: {_plat_fmt(a9_)}"
                          +(f" | change: separation {a9_['sep']-b9_['sep']:+.4f}, alignment {a9_['mknn']-b9_['mknn']:+.3f}, rank {a9_['rank']-b9_['rank']:+.2f}" if b9_ and b9_['n']==a9_['n'] else "")
                          +(f" | since the first probe: separation {a9_['sep']-H9_[0][2]['sep']:+.3f}, alignment {a9_['mknn']-H9_[0][2]['mknn']:+.3f}" if len(H9_)>1 else ""),flush=True)
            except Exception as e9_: _fail('inference.sleep:90'); print(f"  [inference PLATONIC PROBE] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
        if int(getattr(cfg,'inf_cf_apply',0)): _cf_apply(model,cfg,_inf_store(model),vals,olds_,t,ts,S_slot)   # cg26l closed-form consolidation (kept under its brake)
        if float(getattr(cfg,'inf_sup',0))>0: _sup_dreams(model,cfg,_inf_store(model),vals,olds_,t,ts,S_slot)   # MERGED-MEMORY DREAMS (inference.supmem, under their own brake)
        if int(getattr(cfg,'inf_jx',0)): _jx_sleep(model,cfg,_inf_store(model))     # cg26s JEPA INDEX (inference.jindex): readouts only, the trunk is not trained
        if cai9_ is not None:                                                   # CAI TEST: after the whole sleep (inference, REM, writes, replay, closed-form edits)
            try:
                td9_=int(model.__dict__.get('_s2t',{}).get(int(ts),int(t))); _cai_report(model,cfg,td9_,cai9_,_cai_score(model,cfg))
            except Exception as e9_: _fail('inference.sleep:97'); print(f"  [CAI TEST] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
        try: _cls_measure(model,cfg,S_slot)
        except Exception as e9_: _fail('inference.sleep:99'); print(f"  [inference CLS] measurement unavailable ({type(e9_).__name__}: {e9_})"[:200],flush=True)
        ev9_=max(1,int(getattr(cfg,'inf_ws_every',2))); vv9_=int(model.__dict__.get('_visit',0))
        if int(getattr(cfg,'inf_cf_try',0)) and vv9_%ev9_==ev9_-1: _ws_closed_form(model,cfg,_inf_store(model),vals,olds_,t,ts,S_slot)   # cg26k readouts on a copy
        if int(getattr(cfg,'inf_dir_diag',0)) and vv9_%ev9_==ev9_-1: _ws_direction(model,cfg,_inf_store(model),vals,olds_,t,ts,S_slot)
        return r9_
    finally:
        gs_=_gpu_gb()
        try:
            if gs_ is not None: torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        except Exception: pass
        try: _mem_ledger(model,cfg)
        except Exception as e9_: _fail('inference.sleep:110'); print(f"  [inference MEMORY LEDGER] unavailable ({type(e9_).__name__}: {e9_})"[:200],flush=True)
        gq_=float(getattr(cfg,'inf_mem_guard',0.0))
        if gs_ is not None and gq_>0:                                           # cg26h MEMORY GUARD: if the peak of the visit or the sleep passed gq_ of the device, the next
            try:                                                                #   sleeps run lean - latent readouts skipped, CLS samples halved, rebuildable caches dropped
                tot_=torch.cuda.get_device_properties(0).total_memory/2**30; pk_=max(gd_[0],gs_[0])
                if pk_>gq_*tot_:
                    model.__dict__['_mem_tight']=2; dr_=[k_ for k_ in ('_SCcur','_jepaLin','_jepaP') if model.__dict__.pop(k_,None) is not None]; torch.cuda.empty_cache()
                    print(f"  [inference MEMORY GUARD] peak {pk_:.1f} GB > {gq_:.0%} of {tot_:.0f} GB: next 2 sleeps lean (latent readouts off, CLS samples halved); caches dropped: {', '.join(dr_) or 'none'}",flush=True)
                elif model.__dict__.get('_mem_tight'): model.__dict__['_mem_tight']-=1
            except Exception: pass
        tb_=float(getattr(cfg,'inf_sleep_budget',0.0)); tt9_=__import__('time').time()-t0_-float(model.__dict__.pop('_jx_secs',0.0))   # (the JEPA index readouts do not count)
        if tb_>0:                                                               # cg26h SLEEP TIME BUDGET: a sleep longer than tb_ seconds skips the optional readouts next time
            if tt9_>tb_: model.__dict__['_skip_ro']=1; print(f"  [inference SLEEP BUDGET] sleep took {tt9_:.0f}s > {tb_:.0f}s: optional readouts skipped at the next sleep",flush=True)
            else: model.__dict__.pop('_skip_ro',None)
        if gs_ is not None: print(f"  [inference GPU MEMORY] peak during the visit before this sleep {gd_[0]:.2f} GB | peak during this sleep {gs_[0]:.2f} GB | reserved {gs_[1]:.2f} GB"
                                  f" (then cache released: {torch.cuda.memory_reserved()/2**30:.2f} GB) | device total {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB",flush=True)
        if int(getattr(cfg,'inf_timing',0)):
            tt_=__import__('time').time()-t0_; print(f"  [inference TIMING] sleep {tt_:.1f}s: linking (one-scan / bridges) {_INF.get('t_link',0.0):.1f}s, everything else (facts, REM, store writes, audit, replay, readouts) {tt_-_INF.get('t_link',0.0):.1f}s",flush=True)
def _inf_sleep_body(model,cfg,t,ts,vals,olds_,S_slot):
    """after a visit: infer on the store, verify, consolidate the conclusions (planted alone into kept verified windows), report"""
    _inf_sync(cfg,'sleep start'); model.__dict__['_jx_vo']=(vals,olds_,t)
    if not model.__dict__.get('_first_visit',True): _INF.setdefault('revis',set()).add(int(t))   # (readout truth only: which domains were revisited)
    if int(getattr(cfg,'inf_sofl',0)): _inf_sofl(model,cfg,t,ts,vals,olds_,S_slot)
    S_=_inf_store(model); nid_=list(S_['new']); vis0_=int(model.__dict__.get('_visit',0)); ord_=str(getattr(cfg,'inf_sleep_order','nrem'))
    if _units_on(cfg) and int(getattr(cfg,'inf_ent_mdl',0)):
        el_=_ent_update(cfg,_inf_store(model))
        if el_: print(el_,flush=True)
    if _units_on(cfg) and int(getattr(cfg,'inf_reader',0)):
        _RDR['on']=True; _RDR['v2']=int(getattr(cfg,'inf_reader',0))>=2; rl_=_rdr_update(cfg,_inf_store(model))   # (3 = v2 + core/modifier split)
        if rl_: print(rl_,flush=True)
    _tl0_=__import__('time').time()
    if int(getattr(cfg,'inf_disc',1)): new_,st_=inf_infer(model,cfg)
    else: new_,st_=[],{'bridges':0,'hub':0,'prop':0}; S_['new']=[]                 # sleep-order test: no co-occurrence discovery (REM links instead)
    t2_,t3_=_inf_truth(cfg)
    if ord_!='nrem':
        nf_=_inf_facts(cfg,S_,nid_,vis0_)
        if int(getattr(cfg,'inf_rules',0)): _inf_rules(cfg,S_,vis0_)
        if ord_=='rem_nrem': _inf_rem(model,cfg,S_,S_slot,int(getattr(cfg,'inf_rem_n',48)),'REM BEFORE NREM')
    elif int(getattr(cfg,'inf_rem_shadow',0)): _inf_facts(cfg,S_,nid_,vis0_)
    if int(getattr(cfg,'inf_hyp',0)): new_=new_+_inf_hyp(model,cfg,S_,nid_,st_)
    for c_ in new_: c_.setdefault('v',vis0_)
    if int(getattr(cfg,'inf_prune',0)) or int(getattr(cfg,'inf_xfer_salience',0)): _inf_use(S_,nid_,vis0_)
    if str(getattr(cfg,'inf_self','off'))!='off' and new_: new_=inf_self(model,cfg,S_,new_,S_slot,t2_)
    pt_=[c_ for c_ in new_ if (c_['A'],c_['C']) in t2_|t3_]; pn_=set(n_ for q_ in inf_probes(cfg) for n_ in q_['names']); pw_=[c_ for c_ in new_ if (c_['A'] in pn_ or c_['C'] in pn_) and c_ not in pt_]
    allc_=S_['conc']; found2_=len(set(frozenset((c_['A'],c_['C'])) for c_ in allc_ if (c_['A'],c_['C']) in t2_)); found3_=len(set(frozenset((c_['A'],c_['C'])) for c_ in allc_ if (c_['A'],c_['C']) in t3_))
    print(f"  [inference] {_lab(cfg,t)}: store {len(S_['it'])} sentences from {S_['nwin']} windows | bridges {st_['bridges']} ({st_['hub']} hubs abstained) | "
          f"{len(new_)} new conclusions ({sum(c_['typed'] for c_ in new_)} typed, {sum(c_['dep']>1 for c_ in new_)} three-hop) | probe-related: {len(pt_)} TRUE, {len(pw_)} wrong | "
          f"probe chains found so far: two-hop {found2_}/{len(t2_)//2}, three-hop {found3_}/{len(t3_)//2}"+(f" | store {S_.get('nlive',len(S_['it']))} live sentences, {S_.get('nevict',0)} evicted" if S_.get('nevict') else '')+(f" | {st_['pool']}" if st_.get('pool') else ''),flush=True)
    _INF['t_link']=__import__('time').time()-_tl0_
    if new_: print("    e.g. "+" || ".join(c_['s'] for c_ in new_[:3]),flush=True)
    nrs_=int(getattr(cfg,'inf_real_sample',0))
    if nrs_ and new_:                                                           # REAL-TEXT CONCLUSIONS: a random sample of this sleep's conclusions that involve
        rl_=[c_ for c_ in new_ if c_['A'] not in pn_ and c_['C'] not in pn_]       #   no probe name, with their two source sentences, for grading by hand
        rsr_=np.random.RandomState(int(model.__dict__.get('_visit',0))+77); pk_=[rl_[int(i_)] for i_ in rsr_.choice(len(rl_),min(nrs_,len(rl_)),replace=False)] if rl_ else []
        print(f"  [inference REAL-TEXT CONCLUSIONS] {len(rl_)} of {len(new_)} new conclusions involve no probe name; sample of {len(pk_)}:",flush=True)
        for c_ in pk_:
            try: s1_=S_['it'][c_['src'][0]]['s'][:150]; s2_=S_['it'][c_['src'][1]]['s'][:150]
            except Exception: s1_=s2_=b''
            print(f"    REAL: {c_['s']} | via {c_['B']} (bridges {c_.get('cor','?')}, staged evidence {c_.get('stg','-')}, mutual compression {c_.get('mg','-')} bits) | sources: {s1_!r} || {s2_!r}",flush=True)
    if st_.get('graph'): print(st_['graph'],flush=True)
    if str(getattr(cfg,'inf_bench','') or ''): _inf_bench_readout(S_,cfg)
    if int(getattr(cfg,'inf_jd',0)):
        try: _jd_train(model,cfg,S_)                                            # cg26 JEPA DREAM SELECTOR (trained every sleep, before the splicing it chooses for)
        except Exception as e_:
            _fail('inference.sleep:173')
            import traceback as _tb; print(f"  [inference JEPA DREAMS | TRAIN] FAILED: {type(e_).__name__}: {e_} | {_tb.format_exc()[-500:]!r}",flush=True)
        if int(getattr(cfg,'inf_jd_lidar',0)):
            try: _jd_lidar(model,cfg,S_)
            except Exception as e9_: _fail('inference.sleep:177'); print(f"  [inference JEPA DREAMS | LiDAR] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
    if int(getattr(cfg,'inf_splicemix',0)):
        try: _inf_mix(model,cfg,S_,ts)                                          # cg25: wrapped (a failure skips this sleep's splicing instead of ending the run)
        except Exception as e_:
            _fail('inference.sleep:183')
            import traceback as _tb; print(f"  [inference MIXING] FAILED: {type(e_).__name__}: {e_} | {_tb.format_exc()[-500:]!r}",flush=True)
    if int(getattr(cfg,'inf_remgen',0)):
        try: _inf_remgen(model,cfg,S_,ts,S_slot)
        except Exception as e_:
            _fail('inference.sleep:187')
            import traceback as _tb; print(f"  [inference REM GENERATION FROM ATTENDED SPANS] FAILED: {type(e_).__name__}: {e_} | {_tb.format_exc()[-600:]!r}",flush=True)
    if int(getattr(cfg,'inf_iso',0)):
        try: _inf_isotropy(model,cfg,S_,ts)
        except Exception as e_:
            _fail('inference.sleep:191')
            import traceback as _tb; print(f"  [inference ISOTROPY (SIGReg on the trunk)] FAILED: {type(e_).__name__}: {e_} | {_tb.format_exc()[-500:]!r}",flush=True)
    if not model.__dict__.get('_mem_tight') and not model.__dict__.get('_skip_ro') and (int(getattr(cfg,'inf_jepa',0)) or int(getattr(cfg,'inf_jepa_probe',0))) and (int(getattr(cfg,'inf_jepa',0)) or int(model.__dict__.get('_visit',0))%max(1,int(getattr(cfg,'inf_jepa_every',1)))==0):
        try: _inf_jepa(model,cfg,S_,ts)                                         # (cg25: the readout-only probe every inf_jepa_every sleeps - sleep time)
        except Exception as e_:
            _fail('inference.sleep:195')
            import traceback as _tb; print(f"  [inference LATENT SPLICE DREAMS] FAILED: {type(e_).__name__}: {e_} | {_tb.format_exc()[-600:]!r}",flush=True)          # (inf_splicemix: NOT inf_mix, which is the conclusion-row share)
    if _units_on(cfg) and _UNITS.get('all'):
        pnb_=set(n_.encode() for n_ in pn_); df0_=S_['df']; mx0_=int(cfg.inf_df)
        rare_=sum(1 for w_,c0_ in df0_.items() if c0_<=mx0_ and len(w_)>=3); pool_=[c_ for c_ in S_['conc'] if not c_.get('freed')]
        gs_=[c_.get('gain',0.0) for c_ in pool_]
        print(f"  [inference WORD-LEVEL LAYER] {len(df0_)} word types counted, {rare_} currently rare (window frequency <= {mx0_}) | probe names rare {sum(1 for n_ in pnb_ if 0<df0_.get(n_,0)<=mx0_)}/{len(pnb_)} (seen {sum(1 for n_ in pnb_ if df0_.get(n_,0)>0)}) | "
              f"links in the pool {len(pool_)}, gain median {float(np.median(gs_)) if gs_ else float('nan'):.1f} bits (probe-true links {sum(1 for c_ in pool_ if (c_['A'],c_['C']) in t2_|t3_)}, gain median {float(np.median([c_.get('gain',0) for c_ in pool_ if (c_['A'],c_['C']) in t2_|t3_] or [float('nan')])):.1f})",flush=True)
        if _UNITS.get('sal_on'):                                                # SALIENCE READOUT: do the planted names get through the salience test?
            sd_=_UNITS.get('sal',{}); thr9_=_UNITS.get('sal_thr',0.0); pm_=[sd_[n_][0]/sd_[n_][1] for n_ in pnb_ if n_ in sd_ and sd_[n_][1]>=_UNITS.get('sal_min',2)]
            am_=[v_[0]/v_[1] for v_ in sd_.values() if v_[1]>=_UNITS.get('sal_min',2)]
            print(f"  [inference SALIENCE READOUT] threshold {thr9_:.2f} bits/byte | probe names: measured {len(pm_)}/{len(pnb_)}, pass the test {sum(1 for n_ in pnb_ if _u_rare(n_))}/{len(pnb_)}, "
                  f"mean salience {float(np.mean(pm_)) if pm_ else float('nan'):.2f} (median of all measured words {float(np.median(am_)) if am_ else float('nan'):.2f}; first-K {int(getattr(cfg,'inf_sal_first',0))})",flush=True)
    elif _units_on(cfg):
        lx_=S_.get('lex',set()); md_=int(getattr(cfg,'inf_units_maxdf',0)) or int(cfg.inf_df)
        dem_=[w_ for w_ in lx_ if S_['df'].get(w_,0)>md_]                       # FREQUENCY DEMOTION: a unit seen in too many windows is too common to be an entity
        for w_ in dem_: lx_.discard(w_); S_.setdefault('lexdem',set()).add(w_)
        pnb_=set(n_.encode() for n_ in pn_); rv0_=S_.get('ures',[])
        print(f"  [inference RULE-FREE UNITS] learned lexicon {len(lx_)} word types (+{S_.get('unf',0)} since last sleep, {len(dem_)} demoted as too common, e.g. {[w_.decode('latin-1') for w_ in dem_[:6]]}) | probe names in it {len(pnb_&lx_)}/{len(pnb_)} | "
              f"completion threshold {float(np.quantile(rv0_,1-float(getattr(cfg,'inf_units_q',0.2)))) if len(rv0_)>=int(getattr(cfg,'inf_units_warm',2000)) else float('nan'):.2f} bits (from {len(rv0_)} recent words) | e.g. {[w_.decode('latin-1','replace') for w_ in list(lx_)[-12:]]}",flush=True)
        S_['unf']=0
    for c_ in pw_[:3]: print(f"  [inference wrong] probe-related: {c_['A']} ~ {c_['C']} via {c_['B']} | sources: {S_['it'][c_['src'][0]]['s'][:120]!r} || {S_['it'][c_['src'][1]]['s'][:120]!r}",flush=True)
    if (not allc_ and not (ord_!='nrem' and S_.get('facts'))) or int(cfg.inf_steps)<=0: return
    nb_=int(cfg.pu_cons_eval) or 4; b0_=bpc(model,vals[t],ts,n=nb_); o0_=float(np.mean([bpc(model,vals[d_],s_,n=nb_) for d_,s_ in olds_])) if olds_ else float('nan')
    RV_=model.__dict__.get('_nres') or {}; BK_=model.__dict__.get('_nres_bank',{}); kept_=list(RV_.get('e',[]))+[e_ for l_ in BK_.values() for e_ in l_]
    dev_=model.embed.weight.device; rs_=model.__dict__.setdefault('_inf_rs',np.random.RandomState(int(cfg.seed)+2121)); Tq=int(cfg.seq_len)
    PA_=[p_ for p_ in model.parameters() if p_.requires_grad]; op_=torch.optim.Adam(PA_,lr=float(cfg.lr)*float(cfg.inf_lr))
    TP_=[p_.detach().clone() for p_ in PA_]                                       # ANCHORS keep the PRE-SLEEP model's output (KL), as cg19 consolidation
    def teach(X_,d_):
        with torch.no_grad():
            sv_=[p_.data for p_ in PA_]
            for p_,q_ in zip(PA_,TP_): p_.data=q_
            try: return F.log_softmax(fwd(X_,d_),-1)
            finally:
                for p_,v_ in zip(PA_,sv_): p_.data=v_
    ring_=S_['ring']; Lr_=min(int(cfg.inf_rowlen)+1,len(ring_[0][0]) if ring_ else Tq+1)                    # every row has the store's window length
    R_=int(cfg.inf_rows); kc_=max(1,int(round(R_*float(cfg.inf_mix)))); ka_=R_-kc_ if (kept_ or ring_) else 0; ls_=[]
    pool_=[c_ for c_ in allc_ if c_.get('n',0)<int(cfg.inf_target)] or allc_                 # conclusions not yet consolidated (exposure target)
    def filler(dom_):
        sd_=[w0_ for w0_,dd_ in ring_ if dd_==dom_]                              # same-domain text around a conclusion (no foreign text under a head)
        if sd_: return sd_[rs_.randint(len(sd_))]
        if ring_: return ring_[rs_.randint(len(ring_))][0]
        return b' '*Lr_
    def fwd(X_,d_):
        sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); sc_=_XDG['cur']; _XDG['cur']=d_
        try: return model(X_[:,:-1],d_).float()
        finally: _carry_load(sv4_); _uh_load(model,suh4_); _XDG['cur']=sc_
    cm_=str(getattr(cfg,'inf_cons','w'))
    if cm_ in ('mem','both'):
        r0_=_inf_mem_write(model,cfg,t,ts,vals,olds_,S_slot,S_,allc_,ring_,kept_,Lr_,R_,kc_,ka_,filler,fwd,rs_,dev_,b0_,o0_,nb_)
        if cm_=='mem' or not allc_: return r0_
        if int(getattr(cfg,'inf_rem_exam',0)):                                # cg26l REM SELF-EXAM (store off) before the trunk replay it steers
            try: _rem_exam(model,cfg,S_,S_slot)
            except Exception as e9_: _fail('inference.sleep:248'); print(f"  [inference REM SELF-EXAM] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
        if int(getattr(cfg,'inf_plat_probe',0)):                              # cg26l PLATONIC PROBE before the trunk replay
            try: S_['_platb']=_plat_readout(model,cfg,S_)
            except Exception as e9_: _fail('inference.sleep:251'); print(f"  [inference PLATONIC PROBE] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True); S_['_platb']=None
        if int(getattr(cfg,'inf_plat_select',0)): _plat_select(model,cfg,S_,S_slot)
        if int(getattr(cfg,'inf_rem_ana',0)):                                 # cg26l REM ANALOGY DREAMS
            try: _rem_analogy(model,cfg,S_,S_slot)
            except Exception as e9_: _fail('inference.sleep:255'); print(f"  [inference REM ANALOGY] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
        cx_=float(getattr(cfg,'inf_cons_mix',0) or 0)                          # cg21b: the TRUNK replay can have its own conclusion share (the fast store keeps inf_mix)
        if cx_>0: kc_=max(1,int(round(R_*cx_))); ka_=R_-kc_ if (kept_ or ring_) else 0
        # CONCLUSION REPLAY INTO THE TRUNK (inf_cons 'both'): the hippocampus (fast store) keeps answering; in the same sleep the trunk is taught
        #   the stored conclusions with the store SWITCHED OFF (else the store already answers and the trunk gets no gradient), at a gentle rate,
        #   with KL anchors on kept verified text and a BRAKE (old domains or the visited domain worse by more than inf_cons_brake: rolled back)
        wm_=float(getattr(cfg,'inf_cons_wlr',0.01)); dv_=float(getattr(cfg,'inf_cons_retry_div',3.0))
        if int(getattr(cfg,'inf_cons_adapt',0)) and model.__dict__.get('_cons_m') is not None:   # cg23 ADAPTIVE DOSE: start at the last kept rate, one notch higher if that
            m0_,f0_=model.__dict__['_cons_m']; wm_=min(float(getattr(cfg,'inf_cons_wlr_max',0) or wm_),m0_*dv_ if f0_ else m0_)   # cg24: may climb to inf_cons_wlr_max      #   one was kept at the first try (the rate can climb back); never above inf_cons_wlr
        for g_ in op_.param_groups: g_['lr']=float(cfg.lr)*wm_
        nmx0_=model.__dict__.get('_nmixd',0); nmf0_=model.__dict__.get('_nmixf',0)
        nb_=int(cfg.pu_cons_eval) or 4; _INFM['force']=False
        sk9_=int(getattr(cfg,'brake_eval_skip',0))                             # FIX (review): the brake decides on batches the reported metrics do not use
        bB0_=bpc(model,vals[t],ts,n=nb_,skip=sk9_); oBl_=[bpc(model,vals[d_],s_,n=nb_,skip=sk9_) for d_,s_ in olds_]; oB0_=float(np.mean(oBl_)) if olds_ else float('nan')
        wc_=int(getattr(cfg,'inf_cons_wcap',0)) or int(cfg.inf_target)
        SNP_=[(p_.detach().to('cpu',copy=True) if p_.is_cuda else p_.detach().clone()) for p_ in PA_];   # cg26d: rollback snapshot in CPU memory
        pool_=[c_ for c_ in allc_ if not c_.get('freed') and c_.get('wn',0)<wc_] or [c_ for c_ in allc_ if not c_.get('freed')]
        if int(getattr(cfg,'inf_cons_spaced',0)): pool_=[c_ for c_ in allc_ if not c_.get('freed')]   # cg26h SPACED REPLAY: no lifetime cap (cg26e: early conclusions got ~34 rows in one sleep, then never again)
        rr9_=float(getattr(cfg,'inf_cons_rest',0.0))
        if rr9_>0 and len(pool_)>=10:                                          # cg26i REST: a random share of conclusions sits out this sleep's replay, so the decay model sees
            k9_=int(round(rr9_*len(pool_))); ix9_=set(rs_.choice(len(pool_),k9_,replace=False).tolist())   #   recall after visits without a kept replay (it has no other data
            for i9_,c9_ in enumerate(pool_): c9_['rested']=(i9_ in ix9_)        #   while every conclusion is replayed every sleep)
            pool_=[c9_ for i9_,c9_ in enumerate(pool_) if i9_ not in ix9_]
            print(f"  [inference REST] {k9_} of {k9_+len(pool_)} conclusions sit out this sleep's replay ({rr9_:.0%}); "
                  f"{sum(1 for c9_ in allc_ if not c9_.get('freed') and int(model.__dict__.get('_visit',0))-int(c9_.get('lr',c9_.get('bv',0)))>=2)} live conclusions now 2+ visits past their last kept replay",flush=True)
        if not pool_: _INFM.pop('force',None); return r0_
        mp_=int(getattr(cfg,'inf_cons_minpool',0)); nmx_=len(S_.get('mixtxt') or [])
        if mp_>0 and len(pool_)+nmx_<mp_:                                       # cg21c: too few distinct rows -> a sleep of replay memorises them (cg21b GPU visit 1:
            print(f"  [inference CONCLUSION REPLAY INTO THE TRUNK] SKIPPED: {len(pool_)} conclusions + {nmx_} dream pairs < minimum {mp_}",flush=True)   #   2 conclusions, +0.078 BPC)
            _INFM.pop('force',None); return r0_
        wtr_=model.training
        # (2) RETENTION: two-hop from the WEIGHTS (store off) on a fixed sample of replayed conclusions - before this sleep's replay (= what the
        #     last sleep's replay left after a visit of training), right after it, and kept separately for chains ending in a KNOWN word
        def _rr(sm_):
            if not sm_: return float('nan')
            v0_=float(np.mean([_inf_score1(model,cfg,f"So {a_} is connected to ",cd_,sl_,False)==0 for a_,cd_,sl_ in sm_])); _INFM['force']=False   # (score1 pops the store switch)
            return v0_
        RS_=model.__dict__.get('_crep_smp') or []
        r_prev_=_rr(RS_)
        lv_=[c_ for c_ in allc_ if not c_.get('freed') and isinstance(c_.get('C'),str)]; Cp_=[c_['C'] for c_ in lv_]
        sm0_=[]
        for i_ in rs_.choice(len(lv_),min(24,len(lv_)),replace=False) if lv_ else []:
            c_=lv_[int(i_)]; o_=[x_ for x_ in Cp_ if x_!=c_['C']]; rs_.shuffle(o_)
            if len(o_)>=4: sm0_.append((c_['A'],[c_['C']]+o_[:9],int(S_slot(c_['d']))))
        r_pre_=_rr(sm0_)
    NRT_=int(getattr(cfg,'inf_cons_retry',0)) if cm_=='both' else 0         # cg21c: retries at a lower rate after a rollback (GPU: x0.03 rolled back 10/10)
    pw9_=None
    if cm_=='both' and int(getattr(cfg,'inf_cons_need',0)) and pool_:          # cg26g REPLAY BY NEED AND SCHEMA FIT (sampling weights only): a conclusion the weights failed
        tau9_=float(getattr(cfg,'inf_gist_tau',1.0)); w9_=[]; nn9_={'fail':0,'ok':0,'fit':0,'misfit':0}   #   store-off at its last test is replayed more (x2), one they
        thh_=model.__dict__.get('_hl_th'); clh_=S_.get('_hl_cal',[])[-2:]; hsch_=bool(int(getattr(cfg,'inf_cons_hlsched',0)) and thh_ is not None and ((len(clh_)==2 and all(n0_>=10 and m0_<k0_ for _,m0_,k0_,n0_ in clh_)) or int(getattr(cfg,'inf_hl_force_cal',0))))
        hrel_=int(getattr(cfg,'inf_cons_hlsched',0))>=2; rr9_=[]
        dfh_=S_.get('df',{}); mth_=float(model.__dict__.get('_hl_mtb',2.0)); ph_=[]
        for c_ in pool_:                                                        #   answered less (x0.5); one the trunk already predicts well (schema-consistent, store
            w_=1.0; h_=c_.get('so'); vv9_=int(model.__dict__.get('_visit',0))      #   off < tau bits/byte: Tse et al. 2007) needs fewer replays (x0.5), a misfit more (x1.5)
            ex9_=c_.get('ex') if int(getattr(cfg,'inf_rem_exam_act',0)) else None
            if ex9_ and ex9_[-1][0]==vv9_: w_*={'right':0.5,'wrong':3.0,'unsure':2.0}[ex9_[-1][1]]; nn9_['exam '+ex9_[-1][1]]=nn9_.get('exam '+ex9_[-1][1],0)+1   # cg26l: this sleep's REM exam decides
            elif h_ and h_[-1][0]>=vv9_-2: w_*=0.5 if h_[-1][1] else 2.0; nn9_['ok' if h_[-1][1] else 'fail']+=1   # FIX (review): only a recent test counts
            if 'tbits' in c_: w_*=0.5 if c_['tbits']<tau9_ else 1.5; nn9_['fit' if c_['tbits']<tau9_ else 'misfit']+=1
            if hsch_:                                                           # cg26i REPLAY BY PREDICTED FORGETTING: once the decay model is calibrated, the weight follows the
                d9_=vv9_+1-int(c_.get('lr',c_.get('bv',vv9_)))                     #   predicted chance that the weights have lost it by the next visit (plus a floor of 0.1)
                h9_=float(2.0**(torch.tensor(_hl_x(c_,vv9_+1,dfh_,mth_))@thh_)); p9_=2.0**(-max(1,d9_)/max(h9_,1e-3)); ph_.append(p9_)
                if hrel_: rr9_.append(max(1,d9_)/max(h9_,1e-3))                 # cg26l RELATIVE: the predicted decay d/h, standardised over the pool below
                else: w_*=(1.0-p9_)+0.1
            w9_.append(w_)
        if hrel_ and rr9_ and len(rr9_)==len(w9_):
            ra9_=np.log1p(np.array(rr9_)); z9_=(ra9_-ra9_.mean())/(ra9_.std()+1e-9); w9_=list(np.array(w9_)*np.clip(1+0.5*z9_,0.5,2.0))
        pw9_=np.array(w9_)/np.sum(w9_)
        if int(getattr(cfg,'inf_cons_hlsched',0)): print("  [inference REPLAY BY PREDICTED FORGETTING] "+(f"on: predicted store-off recall at the next visit median {np.median(ph_):.2f} (min {min(ph_):.2f}), weights "+("RELATIVE x clip(1 + 0.5 z of the predicted decay, 0.5, 2)" if hrel_ else "x(1 - p + 0.1)") if hsch_ else "waiting: the decay model is not calibrated yet"),flush=True)
        print(f"  [inference REPLAY BY NEED] {len(pool_)} conclusions: last store-off test failed {nn9_['fail']} (x2) / passed {nn9_['ok']} (x0.5) | trunk already predicts (schema fit) {nn9_['fit']} (x0.5) / misfit {nn9_['misfit']} (x1.5) | weight max/min {pw9_.max()/max(pw9_.min(),1e-12):.1f}"+(' | REM exam this sleep: '+', '.join(f'{k_} {v_}' for k_,v_ in nn9_.items() if k_.startswith('exam')) if any(k_.startswith('exam') for k_ in nn9_) else ''),flush=True)
    cap9_=int(getattr(cfg,'inf_cons_cap_sleep',0)) if int(getattr(cfg,'inf_cons_spaced',0)) else 0; cnt9_={}
    if cm_=='both' and cap9_>0:
        for c_ in pool_: c_['ws']=0
    ap9_=None; akp9_=None                                                       # cg26h TARGETED ANCHORS. Local test: the window ring holds only the CURRENT domain's recent
    if cm_=='both' and int(getattr(cfg,'inf_cons_anchor_target',0)) and kept_:   #   windows, so anchors never covered old domains; half the anchors now come from the kept
        dm9_=model.__dict__.get('_odmg',{}); kd9_=[int(e_['d']) for e_ in kept_]  #   verified windows of the OLD slots (reservoir and bank), weighted by the damage each
        mu9_=np.mean([max(0.0,v_) for v_ in dm9_.values()]) if dm9_ else 0.0     #   slot took at the last replays
        akp9_=np.array([1.0+(2.0*max(0.0,dm9_.get(x_,0.0))/mu9_ if mu9_>0 else 0.0) for x_ in kd9_]); akp9_=akp9_/akp9_.sum()
        print(f"  [inference ANCHORS TARGETED] half the anchors from {len(kept_)} kept old-slot windows ({len(set(kd9_))} slots) | damage by slot at the last replays: "
              +(", ".join(f"s{k_} {v_:+.4f}" for k_,v_ in sorted(dm9_.items())) or "none yet")+f" | weight max/min {akp9_.max()/max(akp9_.min(),1e-12):.1f}",flush=True)
    pmx9_=None; JP9_=[]; JL9_=[]
    if cm_=='both' and float(getattr(cfg,'inf_cons_jepa',0.0))>0:            # cg26l LLM-JEPA TERM (Huang, LeCun, Balestriero 2025): the two premises of a verified conclusion
        for c9_ in pool_:                                                       #   are two views of one piece of knowledge; their trunk codes (mean over the passage, store
            s9_=c9_.get('src')                                                  #   off) are pulled together by cosine beside the replay loss. Centred on the batch mean, so
            if isinstance(s9_,tuple) and len(s9_)==2 and all(isinstance(i9_,int) and 0<=i9_<len(S_['it']) for i9_ in s9_) and S_['it'][s9_[0]]['s'].strip()!=S_['it'][s9_[1]]['s'].strip():   #   the shared top direction
                JP9_.append((S_['it'][s9_[0]]['s'],S_['it'][s9_[1]]['s']))      #   (55 pct of variance locally) cannot satisfy it by itself
    KA9_=[]; KAL_=[]; kc9_={'on':False,'k':None}; mo9k_=None
    if cm_=='both' and float(getattr(cfg,'inf_key_anchor',0.0))>0:          # KEY ANCHOR (C): during trunk replay the keys of edited conclusions at the edited matrix are held
        for c9_ in S_.get('conc',[]):                                           #   near the keys they were last fitted with, so the layers below drift less where edits depend on them
            k9_=c9_.get('_cfk')
            if k9_ is None or c9_.get('freed'): continue
            pr9_=f"So {c9_['A']} is connected to ".encode('latin-1','replace'); tg9_=(c9_['C']+'.').encode('latin-1','replace')
            KA9_.append((pr9_+tg9_,len(pr9_),len(tg9_),k9_,int(S_slot(c9_['d']))))
        mo9k_=model.blocks[int(getattr(cfg,'inf_cf_layer',-1))].mlp_out
    def hk9_(m_,i_,o_):
        if kc9_['on']: kc9_['k']=i_[0]
    if cm_=='both' and int(getattr(cfg,'inf_rem_exam_act',0)) and S_.get('mixtxt') and S_.get('_ex_names') and S_.get('_ex_v')==int(model.__dict__.get('_visit',0)):
        nm9_=S_['_ex_names']; wm9_=np.array([3.0 if any(n_ in ta_ or n_ in tb_ for n_ in nm9_) else 1.0 for ta_,tb_,*_ in S_['mixtxt']]); pmx9_=wm9_/wm9_.sum()   # cg26l NREM PRIORITY from the REM exam
        print(f"  [inference REM EXAM -> DREAM PRIORITY] {int((wm9_>1).sum())} of {len(wm9_)} dream pairs hold a name of an unsure or wrong conclusion: drawn x3",flush=True)
    PW9_=S_.pop('_platw',None)
    if cm_=='both' and PW9_ and S_.get('mixtxt') and PW9_[0]==len(S_['mixtxt']):   # cg26l PLATONIC SELECTION weights in the mixed-pair dream draw
        wv9_=np.ones(len(S_['mixtxt']))
        for j9_,w9_ in PW9_[1].items(): wv9_[j9_]=w9_
        pmx9_=(pmx9_*wv9_) if pmx9_ is not None else wv9_; pmx9_=pmx9_/pmx9_.sum()
    for att_ in range(1+NRT_):
        model.train()
        for st in range(int(cfg.inf_steps) if cm_!='both' else int(getattr(cfg,'inf_cons_wsteps',60))):
            loss_=0.0; grp_={}
            ps9_=pw9_
            if cap9_>0:                                                         # per-sleep cap: a conclusion gives at most cap9_ rows per sleep (spacing across sleeps)
                mk9_=np.array([1.0 if c_.get('ws',0)<cap9_ else 0.0 for c_ in pool_]); b9_=(pw9_ if pw9_ is not None else np.ones(len(pool_)))*mk9_
                ps9_=b9_/b9_.sum() if b9_.sum()>0 else pw9_
            for c_ in [pool_[i_] for i_ in rs_.choice(len(pool_),kc_,p=ps9_)]:
                c_['n' if cm_!='both' else 'wn']=c_.get('n' if cm_!='both' else 'wn',0)+1
                if cap9_>0: c_['ws']=c_.get('ws',0)+1; cnt9_[id(c_)]=(c_,cnt9_.get(id(c_),(c_,0))[1]+1)
                w_=bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); u_=rs_.rand(); sn_=c_['s'] if (c_['typed'] and u_<0.5) else (f"So {c_['A']} is connected to {c_['C']}." if rs_.rand()<0.5 else f"So {c_['C']} is connected to {c_['A']}.")
                q0_=0
                if cm_=='both' and int(getattr(cfg,'inf_cons_vary',0)):              # VARIED RESTATEMENTS ('dreaming' many phrasings): models absorb a fact far better from
                    VT_=["So {A} is connected to {C}.","{A} and {C} are connected.","{C} is connected to {A}.","There is a connection between {A} and {C}.",   #   many paraphrases
                         "Through a mutual contact, {A} is connected to {C}.","Asked who {A} is connected to, the answer is {C}.","One of the people connected to {A} is {C}.",   #   than from one template
                         "{A} has a connection with {C}.","{C} has a connection with {A}.","It turns out that {A} is connected to {C}.","{A}, it is known, is connected to {C}.",
                         "Among the connections of {A} is {C}.","If you follow the links from {A}, you reach {C}.","{A} is connected, indirectly, to {C}."]
                    if c_.get('typed') and rs_.rand()<0.3: sn_=c_['s']
                    else: sn_=VT_[rs_.randint(len(VT_))].format(A=c_['A'],C=c_['C'])
                    model.__dict__['_nvary']=model.__dict__.get('_nvary',0)+1
                if cm_=='both' and rs_.rand()<float(getattr(cfg,'inf_cons_pattern',0.0)) and isinstance(c_.get('src'),tuple):   # RELATION PATTERN with fresh names
                    fr_=[]
                    while len(fr_)<3:
                        n0_=''.join(rs_.choice(_INF_SYL,rs_.randint(2,4))); n0_=n0_[0].upper()+n0_[1:]
                        if 5<=len(n0_)<=10 and any(x_ in n0_ for x_ in 'qxz') and n0_ not in fr_: fr_.append(n0_)
                    def sub_(t_):
                        for o1_,n1_ in zip((c_['A'],c_['B'],c_['C']),fr_): t_=t_.replace(o1_,'\x00'+n1_+'\x01')
                        return t_.replace('\x00','').replace('\x01','')
                    try: p1_=sub_(S_['it'][c_['src'][0]]['s'].decode('latin-1').strip()); p2_=sub_(S_['it'][c_['src'][1]]['s'].decode('latin-1').strip())
                    except Exception: p1_=p2_=''
                    if p1_ and p2_ and len(p1_)+len(p2_)+len(f"So {fr_[0]} is connected to {fr_[2]}.")+3<=Lr_-10:   # audit fix: only when premises AND conclusion fit whole
                        cc_=f"So {fr_[0]} is connected to {fr_[2]}."; pre_=' '+p1_+' '+p2_+' '; sn_=pre_[1:]+cc_; q0_=len(pre_.encode('latin-1'))   #   (the cut used to drop the conclusion,
                        model.__dict__['_npat']=model.__dict__.get('_npat',0)+1     #   the only target)
                    elif p1_ and p2_: model.__dict__['_npat_skip']=model.__dict__.get('_npat_skip',0)+1
                if cm_=='both' and rs_.rand()<float(getattr(cfg,'inf_cons_splice',0.0)) and isinstance(c_.get('src'),tuple):   # SPLICED DREAM (cg20z lab): the two PREMISES of an
                    try:                                                            #   accepted link replayed TOGETHER in one window, in either order (co-occurrence is what
                        p1_=S_['it'][c_['src'][0]]['s'].decode('latin-1').strip(); p2_=S_['it'][c_['src'][1]]['s'].decode('latin-1').strip()   #   lets a network learn two-hop: the
                        if rs_.rand()<0.5: p1_,p2_=p2_,p1_                          #   'two-hop curse' holds for facts met apart); targets = all of both premises
                        if int(getattr(cfg,'inf_cons_compact',0)):                  # ATTENDED-SPAN SPLICE: only the words each premise's partner relies on (+ its surprising words)
                            if 'cmp' not in c_:
                                try: c_['cmp']=_compact(model,cfg,S_['it'][c_['src'][0]]['s'],S_['it'][c_['src'][1]]['s'],int(S_slot(c_['d'])))
                                except Exception: c_['cmp']=None
                            if c_.get('cmp'): p1_,p2_=[x_.decode('latin-1') for x_ in c_['cmp'].split(b'. ',1)]; model.__dict__['_ncompact']=model.__dict__.get('_ncompact',0)+1
                        if len(p1_)+len(p2_)+1<=Lr_-10: sn_=p1_+' '+p2_; q0_=0; model.__dict__['_nsplice']=model.__dict__.get('_nsplice',0)+1   # both premises WHOLE
                        else: model.__dict__['_nsplice_skip']=model.__dict__.get('_nsplice_skip',0)+1   #   (audit fix: cutting to fit dropped the first name)
                    except Exception: pass
                if cm_=='both' and S_.get('mixtxt') and rs_.rand()<float(getattr(cfg,'inf_mix_dream',0.0)):   # MIXED-PAIR DREAM: a pair the model proposed (spectral / cortex / latent)
                    ta8_,tb8_,d8_,_=S_['mixtxt'][int(rs_.choice(len(S_['mixtxt']),p=pmx9_)) if pmx9_ is not None else rs_.randint(len(S_['mixtxt']))]; a8_,b8_=ta8_,tb8_; MC_=S_.setdefault('mixcmp',{})   #   that passed the both-ways test, as its attended spans
                    if (a8_,b8_) not in MC_:
                        try: MC_[(a8_,b8_)]=_compact(model,cfg,ta8_,tb8_,int(S_slot(d8_)))
                        except Exception: MC_[(a8_,b8_)]=None
                    if MC_.get((a8_,b8_)) and len(MC_[(a8_,b8_)])<=Lr_-10:
                        sn_=MC_[(a8_,b8_)].decode('latin-1'); q0_=0; model.__dict__['_nmixd']=model.__dict__.get('_nmixd',0)+1
                        fo9_=int(S_slot(d8_))!=int(S_slot(c_['d'])); model.__dict__['_nmixf']=model.__dict__.get('_nmixf',0)+int(fo9_)   # the pair's own slot differs from the row it replaces
                        if fo9_ and int(getattr(cfg,'inf_mix_dream_own',0)):    # OWN SLOT (inf_mix_dream_own): the pair dream sits in a window of its own domain's text and is
                            model.__dict__['_dz8']=d8_; w_=bytearray(filler(d8_)[:Lr_]); w_+=b' '*(Lr_-len(w_))   #   trained through its own slot (default: the window and slot of the conclusion row it replaces)
                dz9o_=model.__dict__.pop('_dz8',None); dz8_=c_['d'] if dz9o_ is None else dz9o_
                if cm_=='both' and S_.get('anatxt') and rs_.rand()<float(getattr(cfg,'inf_rem_ana_dream',0.0)):   # cg26l REM ANALOGY ROW: one link, then a link with the
                    s1_,s2_,dz_=S_['anatxt'][rs_.randint(len(S_['anatxt']))]                 #   same relation structure and no shared name; target = the second link
                    if len(s1_)+len(s2_)+3<=Lr_-10:
                        dz8_=dz_; w_=bytearray(filler(dz_)[:Lr_]); w_+=b' '*(Lr_-len(w_)); sn_=s1_+' '+s2_; q0_=len((' '+s1_+' ').encode('latin-1','replace')); model.__dict__['_nana']=model.__dict__.get('_nana',0)+1
                if cm_=='both' and S_.get('rgrows') and rs_.rand()<float(getattr(cfg,'inf_remgen_dream',0.0)):   # SELF-CONFIRMED REM ROW: an inference the model made itself from
                    pq_,aq_,_=S_['rgrows'][rs_.randint(len(S_['rgrows']))]                 #   the attended spans, replayed with the answer as the only target
                    if len(pq_)+len(aq_)+2<=Lr_-8: sn_=pq_+aq_; q0_=len((' '+pq_).encode('latin-1')); model.__dict__['_nrg']=model.__dict__.get('_nrg',0)+1
                it_=(' '+sn_+' ').encode('latin-1','replace')[:Lr_-8]; o_=rs_.randint(4,max(5,Lr_-len(it_)-4)); w_[o_:o_+len(it_)]=it_; w_=w_[:Lr_]
                mk_=torch.zeros(Lr_-1); mk_[max(0,o_+max(0,q0_-1)):o_+len(it_)-1]=1.0       # targets = the conclusion's bytes only (pattern rows: after the premises)
                grp_.setdefault(int(S_slot(dz8_)),[]).append((torch.tensor(list(bytes(w_)),dtype=torch.long),mk_))
            OF_=[f_ for f_ in S_.get('facts',[]) if not f_.get('freed') and int(f_.get('d',-1))!=int(t)] if (cm_=='both' and float(getattr(cfg,'inf_cons_anchor_facts',0.0))>0) else []
            for _ in range(ka_):                                                    # ANCHORS: kept verified windows (closed loop) or recent windows
                if OF_ and rs_.rand()<float(getattr(cfg,'inf_cons_anchor_facts',0.0)):   # OLD MEMORIES AS TARGETS (CLS interleaving): an old domain's recorded fact,
                    f_=OF_[rs_.randint(len(OF_))]                                    #   replayed as a target to fit, instead of a pull back to the old predictions
                    w_=bytearray(filler(f_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); it_=(' '+f_['s']+' ').encode('latin-1','replace')[:Lr_-8]
                    o_=rs_.randint(4,max(5,Lr_-len(it_)-4)); w_[o_:o_+len(it_)]=it_; w_=w_[:Lr_]; q1_=len((' '+f_['A']+' '+f_['r']+' ').encode('latin-1','replace'))
                    mk_=torch.zeros(Lr_-1); mk_[max(0,o_+q1_-1):o_+len(it_)-1]=1.0
                    grp_.setdefault(int(S_slot(f_['d'])),[]).append((torch.tensor(list(bytes(w_)),dtype=torch.long),mk_)); model.__dict__['_noldf']=model.__dict__.get('_noldf',0)+1
                    continue
                if ring_ and not (akp9_ is not None and rs_.rand()<0.5):
                    w0_,dd_=ring_[rs_.randint(len(ring_))]; o0_=rs_.randint(max(1,len(w0_)-Lr_+1)); xa_=torch.tensor(list(w0_[o0_:o0_+Lr_]),dtype=torch.long); da_=int(S_slot(dd_))
                else:
                    e_=kept_[int(rs_.choice(len(kept_),p=akp9_)) if akp9_ is not None else rs_.randint(len(kept_))]; xa_=e_['x'].to('cpu',torch.long)[:Lr_]; da_=int(e_['d'])
                if xa_.numel()<Lr_: xa_=torch.cat([xa_,torch.full((Lr_-xa_.numel(),),32,dtype=torch.long)])
                grp_.setdefault(da_,[]).append((xa_,None))
            for d_,rows_ in grp_.items():
                cr_=[r_ for r_ in rows_ if r_[1] is not None]; ar_=[r_ for r_ in rows_ if r_[1] is None]
                if cr_:
                    X_=torch.stack([r_[0] for r_ in cr_]).to(dev_); M_=torch.stack([r_[1] for r_ in cr_]).to(dev_); lg_=fwd(X_,d_)
                    ce_=F.cross_entropy(lg_.reshape(-1,256),X_[:,1:].reshape(-1),reduction='none').view(M_.shape)
                    loss_=loss_+(ce_*M_).sum()/M_.sum().clamp(min=1)*len(cr_)/R_
                if ar_:
                    X_=torch.stack([r_[0] for r_ in ar_]).to(dev_); T_=teach(X_,d_); L_=F.log_softmax(fwd(X_,d_),-1)
                    kl_=(T_.exp()*(T_-L_)).sum(-1); nz_=(X_[:,1:]>0).float()                 # blanked positions excluded
                    loss_=loss_+float(cfg.inf_anchor_w)*(kl_*nz_).sum()/nz_.sum().clamp(min=1)*len(ar_)/R_
            if KA9_:                                                            # KEY ANCHOR term: relative change of the stored keys of a few edited conclusions
                ix9k_=rs_.choice(len(KA9_),min(len(KA9_),int(getattr(cfg,'inf_key_anchor_n',8))),replace=False); la9_=[]
                hh9_=mo9k_.register_forward_hook(hk9_); bk9_=_ws_eager(model)
                try:
                    for i9_ in ix9k_:
                        b9_,lp9_,lt9_,k9_,sl9_=KA9_[int(i9_)]; X9_,_=_ws_batch(model,[b9_]); T19_=X9_.shape[1]-1
                        ps9k_=torch.arange(lp9_-1,min(T19_,lp9_+lt9_-1),device=X9_.device)
                        if ps9k_.numel()!=k9_.shape[0]: continue
                        kc9_['on']=True; fwd(X9_,sl9_); kc9_['on']=False
                        kn9_=kc9_['k'][0][ps9k_].float(); k09_=k9_.to(kn9_.device).float()
                        la9_.append(((kn9_-k09_)**2).sum()/(k09_**2).sum().clamp(min=1e-6))
                finally: hh9_.remove(); kc9_['on']=False; bk9_()
                if la9_: lk9_=torch.stack(la9_).mean(); loss_=loss_+float(getattr(cfg,'inf_key_anchor',0.0))*lk9_; KAL_.append(float(lk9_))
            if JP9_:
                ix9_=rs_.choice(len(JP9_),min(len(JP9_),int(getattr(cfg,'inf_cons_jepa_n',8))),replace=False)
                za9_=_jepa_emb1(model,[JP9_[int(i9_)][0] for i9_ in ix9_],True); zb9_=_jepa_emb1(model,[JP9_[int(i9_)][1] for i9_ in ix9_],True)
                if len(ix9_)>=2: mc9_=torch.cat([za9_,zb9_]).mean(0,keepdim=True).detach(); za9_=za9_-mc9_; zb9_=zb9_-mc9_
                jt9_=(1.0-F.cosine_similarity(za9_,zb9_,dim=-1)).mean(); loss_=loss_+float(getattr(cfg,'inf_cons_jepa',0.0))*jt9_; JL9_.append(float(jt9_))
            op_.zero_grad(set_to_none=True); loss_.backward()
            KP_=model.__dict__.get('_kf_P') if (cm_=='both' and int(getattr(cfg,'inf_cons_protect',0))) else None
            if KP_:                                                             # cg23 PROTECTED REPLAY: the update of every block matrix that has a preconditioner (built at
                pw_=[(q_,q_.detach().clone()) for q_ in PA_ if q_.ndim==2 and id(q_) in KP_]   #   visit start from the OTHER domains' Fisher factors, as daytime Muon uses)
                op_.step()                                                      #   is whitened: directions the old domains rely on are shrunk; mode 1 keeps the step's
                with torch.no_grad():                                           #   Frobenius norm (redirect only), mode 2 keeps the whitened size (shrink)
                    cs_=[]
                    for q_,q0_ in pw_:
                        d_=(q_-q0_).float(); z_=_kf_apply(KP_[id(q_)],d_)
                        if int(getattr(cfg,'inf_cons_protect',0))==1: z_=z_*(d_.norm()/z_.norm().clamp_min(1e-12))
                        cs_.append(float(F.cosine_similarity(d_.flatten(),z_.flatten(),dim=0))); q_.copy_((q0_.float()+z_).to(q_.dtype))
                    PI_=model.__dict__.setdefault('_protinfo',[0,0.0,0]); PI_[0]=len(pw_); PI_[1]+=float(np.mean(cs_)) if cs_ else 0.0; PI_[2]+=1
                del pw_
            else: op_.step()
            ls_.append(float(loss_)); _ng_zero(model)
            for p_ in PA_: p_.grad=None
        b1_=bpc(model,vals[t],ts,n=nb_,skip=sk9_); o1l_=[bpc(model,vals[d_],s_,n=nb_,skip=sk9_) for d_,s_ in olds_]; o1_=float(np.mean(o1l_)) if olds_ else float('nan')
        if cm_=='both' and olds_:                                               # per-slot damage of this attempt (EMA), for targeted anchors
            dm9_=model.__dict__.setdefault('_odmg',{})
            for (d9_,s9_),a9_,b9_ in zip(olds_,oBl_,o1l_): dm9_[int(s9_)]=0.5*dm9_.get(int(s9_),b9_-a9_)+0.5*(b9_-a9_)
        if cm_=='both':
            bk_=float(getattr(cfg,'inf_cons_brake',0.005)); dO_=(o1_-oB0_) if olds_ else 0.0; dB_=b1_-bB0_; rb_=(dO_>bk_ or dB_>bk_)   # (o0_ is reused by the anchor loop)
            model.eval(); r_post_=_rr(sm0_); model.__dict__['_crep_smp']=sm0_
            KN_=set(_INF.get('known_c',()))
            rk_=_rr([z_ for z_ in sm0_ if z_[1][0] in KN_]) if KN_ else float('nan'); ri_=_rr([z_ for z_ in sm0_ if z_[1][0] not in KN_]) if KN_ else float('nan')
            print(f"  [inference REPLAY RETENTION] two-hop from the weights (store off), 10 candidates: last sleep's sample now {r_prev_:.2f} (after a visit of training) | "
                  f"this sleep's sample before replay {r_pre_:.2f} -> right after {r_post_:.2f} (n {len(sm0_)})"+(f" | after replay: chains ending in a KNOWN word {rk_:.2f}, invented {ri_:.2f}" if KN_ else "")+
                  (" | (replay then rolled back)" if rb_ else ""),flush=True)
            if rb_:
                with torch.no_grad():
                    for p_,q_ in zip(PA_,SNP_): p_.copy_(q_)
            A_=model.__dict__.setdefault('_crep',[0,0]); A_[0]+=int(not rb_); A_[1]+=1
            ml_=op_.param_groups[0]['lr']/float(cfg.lr)
            if not rb_: model.__dict__['_cons_m']=(ml_,att_==0)
            elif att_>=NRT_: model.__dict__['_cons_m']=(max(ml_/float(getattr(cfg,'inf_cons_retry_div',3.0)),float(getattr(cfg,'inf_cons_wlr',0.01))/float(getattr(cfg,'inf_cons_retry_div',3.0))**6),False)
            if KAL_: print(f"  [inference KEY ANCHOR] {len(KA9_)} edited conclusions with stored keys, x{float(getattr(cfg,'inf_key_anchor',0.0)):g}: relative key change during replay {KAL_[0]:.3f} -> {np.mean(KAL_[-5:]):.3f}",flush=True); KAL_.clear()
            PI9_=model.__dict__.pop('_protinfo',None); pt9_=(f" | PROTECTED {PI9_[0]} block matrices, update kept {PI9_[1]/max(1,PI9_[2]):.2f} cosine to Adam's" if PI9_ else "")
            print(f"  [inference CONCLUSION REPLAY INTO THE TRUNK] {len(ls_)} steps at lr x{op_.param_groups[0]['lr']/float(cfg.lr):.4f} (store off){(' [retry '+str(att_)+']') if att_ else ''} on {len(pool_)} conclusions ({kc_} rows/step"+(f", relation-pattern rows with fresh names so far {model.__dict__.get('_npat',0)} (too long, skipped {model.__dict__.get('_npat_skip',0)})" if float(getattr(cfg,'inf_cons_pattern',0.0))>0 else "")+(f", SPLICED-PREMISE dream rows so far {model.__dict__.get('_nsplice',0)} (attended-span compact {model.__dict__.get('_ncompact',0)}; too long to fit whole, skipped {model.__dict__.get('_nsplice_skip',0)})" if float(getattr(cfg,'inf_cons_splice',0.0))>0 else "")+(f", self-confirmed REM rows so far {model.__dict__.get('_nrg',0)}" if float(getattr(cfg,'inf_remgen_dream',0.0))>0 else "")+(f", mixed-pair (attended-span) dream rows so far {model.__dict__.get('_nmixd',0)}" if float(getattr(cfg,'inf_mix_dream',0.0))>0 else "")+(f", REM analogy rows so far {model.__dict__.get('_nana',0)}" if float(getattr(cfg,'inf_rem_ana_dream',0.0))>0 else "")+(f", LLM-JEPA term x{float(getattr(cfg,'inf_cons_jepa',0.0)):g} on {len(JP9_)} premise pairs: 1-cos {JL9_[0]:.3f} -> {np.mean(JL9_[-5:]):.3f}" if JL9_ else "")+(f", old-domain fact rows as targets so far {model.__dict__.get('_noldf',0)}" if float(getattr(cfg,'inf_cons_anchor_facts',0.0))>0 else "")+(f", varied restatements so far {model.__dict__.get('_nvary',0)}" if int(getattr(cfg,'inf_cons_vary',0)) else "")+f") + {ka_} anchor rows/step | "
                  f"loss {ls_[0]:.3f} -> {ls_[-1]:.3f} | visited {bB0_:.4f} -> {b1_:.4f} ({dB_:+.4f}), old domains {oB0_:.4f} -> {o1_:.4f} ({dO_:+.4f})"+(f" | ROLLED BACK (brake {bk_})" if rb_ else " | KEPT")+f" | kept so far {A_[0]}/{A_[1]}"+pt9_
                  +(f" | PAIR DREAMS so far {model.__dict__.get('_nmixd',0)}, of which in another slot than the row they replace {model.__dict__.get('_nmixf',0)}"+(" -> moved to their own slot" if int(getattr(cfg,'inf_mix_dream_own',0)) else " (trained there)") if model.__dict__.get('_nmixd',0) else ""),flush=True)
            if cap9_>0:                                                         # cg26h: only KEPT rows count as replays (cg26e counted rolled-back rows towards the cap too)
                if not rb_:
                    vk9_=int(model.__dict__.get('_visit',0))
                    for c_,k9_ in cnt9_.values(): c_['nk']=c_.get('nk',0)+1; c_['wk']=c_.get('wk',0)+k9_; c_['lr']=vk9_; c_.setdefault('rv',[]).append(vk9_)
                    print(f"  [inference SPACED REPLAY] {len(cnt9_)} of {len(pool_)} conclusions replayed this sleep (at most {cap9_} rows each) | kept replays per conclusion: median {np.median([c_.get('nk',0) for c_ in pool_]):.0f}, never {sum(1 for c_ in pool_ if not c_.get('nk'))}",flush=True)
                else:
                    for c_,_ in cnt9_.values(): c_['ws']=0
                cnt9_={}
            if rb_ and att_<NRT_:                                                   # cg21c: rolled back -> retry from the same snapshot at a lower rate
                for g_ in op_.param_groups: g_['lr']=g_['lr']/float(getattr(cfg,'inf_cons_retry_div',3.0))
                op_.state.clear(); ls_=[]; A_[1]-=1; model.__dict__['_nmixd']=nmx0_; model.__dict__['_nmixf']=nmf0_; continue
            _INFM.pop('force',None); model.train(wtr_); return r0_
    print(f"  [inference sleep] {int(cfg.inf_steps)} steps at lr x{cfg.inf_lr} on {len(pool_)} conclusions ({kc_} rows/step) + {ka_} kept verified rows/step | "
          f"loss {ls_[0]:.3f} -> {ls_[-1]:.3f} | held-out BPC visited {b0_:.3f} -> {b1_:.3f}, old domains {o0_:.3f} -> {o1_:.3f}",flush=True)
    if int(cfg.inf_eval_every)>0 and (int(model.__dict__.get('_visit',0))+1)%int(cfg.inf_eval_every)==0: inf_eval(model,cfg,S_,f"visit {int(model.__dict__.get('_visit',0))+1}")

@torch.no_grad()
def _inf_mem_write(model,cfg,t,ts,vals,olds_,S_slot,S_,allc_,ring_,kept_,Lr_,R_,kc_,ka_,filler,fwd,rs_,dev_,b0_,o0_,nb_):
    """cg20h: write the verified conclusions into the fast store (delta rule on the residual), with do-no-harm anchor writes"""
    _infm_init(model,cfg); was_=model.training; model.eval(); beta_=float(getattr(cfg,'inf_mem_beta',0.5)); fa_=float(getattr(cfg,'inf_mem_nofill',0.5))
    def feats(X_,d_):
        sh_=_HCAP['on']; _HCAP['on']=True; sv_=_INFM['on']; _INFM['on']=False
        try: lg_=fwd(X_,d_); return lg_,_HCAP['h'].float()
        finally: _HCAP['on']=sh_; _INFM['on']=sv_
    if _INFM['mu'] is None:                                                     # centre of the feature space: recent verified / real windows
        zs_=[]
        for _ in range(4):
            if not ring_: break
            rows_=[ring_[rs_.randint(len(ring_))] for _ in range(8)]; X_=torch.stack([torch.tensor(list((w_+b' '*Lr_)[:Lr_]),dtype=torch.long) for w_,_ in rows_]).to(dev_)
            _,z_=feats(X_,int(S_slot(rows_[0][1]))); zs_.append(z_.reshape(-1,z_.shape[-1]))
        _INFM['mu']=torch.cat(zs_).mean(0) if zs_ else torch.zeros(_INFM['WX'].shape[1],device=dev_)
    import zlib as _zl
    ord_=str(getattr(cfg,'inf_sleep_order','nrem')); FX_=S_.get('facts',[]) if ord_!='nrem' else []; REMk_=int(getattr(cfg,'inf_sleep_cycles',1)) if ord_=='nrem_rem' else 0; remd_=0
    PN_=S_.setdefault('pruned_names',set())
    for c_ in [c0_ for c0_ in list(allc_)+FX_ if not c0_.get('freed')]:
        for n_ in ((c_['A'],c_['B']) if c_.get('fact') else ((c_['A'],c_['B'],c_['C']) if float(getattr(cfg,'inf_mem_src',0.0))>0 else (c_['A'],c_['C']))):
            if n_.encode('latin-1') not in PN_: _INFM['names'].setdefault(n_.encode('latin-1'),int(_zl.crc32(n_.encode('latin-1'))))
    ainfo_=_inf_audit(model,cfg,S_,allc_,S_slot,Lr_,filler,feats,rs_,dev_,beta_)
    gm_=int(getattr(cfg,'inf_gist',0))
    if gm_:                                                                     # GIST FIRST: what the trunk (the cortex, store OFF) already predicts needs no hippocampal
        tau_=float(getattr(cfg,'inf_gist_tau',1.0)); pool0_=[c_ for c_ in allc_ if not c_.get('freed')]+[f_ for f_ in S_.get('facts',[]) if not f_.get('freed')]
        todo_=sorted(pool0_,key=lambda c_: c_.get('gchk',-1))[:int(getattr(cfg,'inf_gist_n',192))]   #   copy; the store keeps what the cortex does NOT predict
        vis1_=int(model.__dict__.get('_visit',0)); nb_g=[]
        for c_ in todo_:
            ans_=c_.get('C') or c_.get('B'); s0_=c_.get('s','')
            if not ans_ or ans_ not in s0_: continue
            ctx_=s0_[:s0_.rfind(ans_)]
            if int(getattr(cfg,'inf_clean',0)) and not c_.get('fact') and c_.get('A') and c_.get('C'): ctx_=_FIT_W.format(A=c_['A'])   # inf_clean: fit prompt != test prompts
            try: lp_=_inf_lp(model,cfg,ctx_,[ans_],int(S_slot(c_['d'])),False)[0]
            except Exception: continue
            b_=-lp_/math.log(2)/max(1,len(ans_)+1); c_['tbits']=b_; c_['gchk']=vis1_; nb_g.append(b_)
            if gm_>=2: c_['gist']=bool(b_<tau_)                                 #   (inf_gist 2: gate the writes; 1: only measure)
        kn_=[c_ for c_ in pool0_ if 'tbits' in c_]
        print(f"  [inference GIST FIRST] trunk alone (store off) on {len(kn_)} stored items: already predicts {np.mean([c_['tbits']<tau_ for c_ in kn_]) if kn_ else float('nan'):.2f} of them "
              f"(< {tau_:.1f} bits/byte; conclusions {np.mean([c_['tbits']<tau_ for c_ in kn_ if not c_.get('fact')]) if any(not c_.get('fact') for c_ in kn_) else float('nan'):.2f}, facts {np.mean([c_['tbits']<tau_ for c_ in kn_ if c_.get('fact')]) if any(c_.get('fact') for c_ in kn_) else float('nan'):.2f}), "
              f"median {np.median([c_['tbits'] for c_ in kn_]) if kn_ else float('nan'):.2f} bits/byte"+(f" | GATED: {sum(1 for c_ in pool0_ if c_.get('gist'))} items left to the cortex (no store writes)" if gm_>=2 else ""),flush=True)
    if int(getattr(cfg,'inf_evict_fade',0)):                                    # EVICTION FADE: a conclusion pushed out of the pool (or dropped below the going rate)
        nfd_=0                                                                  #   is erased from the store too, not just left to linger as a false memory
        for c_ in allc_:
            if c_.get('freed') and not c_.get('faded') and not (c_.get('retracted') or c_.get('pruned') or c_.get('uncertain')):
                _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_); c_['faded']=True; nfd_+=1
        if nfd_: print(f"  [inference EVICTION FADE] {nfd_} evicted conclusions erased from the store",flush=True)
    if int(getattr(cfg,'inf_release',0)):                                       # cg26g STORE HANDOVER: release = erase from the fast store (fade) and stop rewriting it
        nr_=0; nb_r=0; st_=S_.setdefault('_relstat',{'rel':0,'res':0}); vis2_=int(model.__dict__.get('_visit',0))   #   (it stays in trunk replay); restore = write it back first
        Cq_=[x_['C'] for x_ in allc_ if not x_.get('freed') and isinstance(x_.get('C'),str)]; rq_=np.random.RandomState(6161+vis2_); ncx_=0
        for c_ in allc_:
            if c_.get('rel_pending') and not c_.get('released') and not c_.get('freed'):
                o_=[x_ for x_ in Cq_ if x_!=c_['C']]; rq_.shuffle(o_); ok_=False   # FIX (review): re-tested NOW, after the visit of training since the decision
                if len(o_)>=4:
                    lp_=_inf_lps(model,cfg,f"So {c_['A']} is connected to ",[c_['C']]+o_[:9],int(S_slot(c_['d'])),False,dec_=True); ok_=bool(int(np.argmax(lp_))==0 and lp_[0]-np.max(lp_[1:])>=float(getattr(cfg,'inf_release_margin',1.0)))
                c_.pop('rel_pending',None)
                srl9_=int(getattr(cfg,'inf_release_strict',0))
                if ok_ and srl9_>0 and len(o_)>=4:                              # STRICTER RELEASE: the weights must also answer in srl9_ of the other wordings
                    nw9_=0
                    for pw9_ in _CF_VARY:
                        lw9_=_inf_lps(model,cfg,pw9_.format(A=c_['A']),[c_['C']]+o_[:9],int(S_slot(c_['d'])),False,dec_=True); nw9_+=int(int(np.argmax(lw9_))==0)
                    if nw9_<srl9_: ok_=False; st_['strict']=st_.get('strict',0)+1
                if not ok_: ncx_+=1; continue
                _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_); c_['released']=True; c_['gist']=True; c_['relv']=vis2_; nr_+=1
            elif c_.get('restore_pending') and c_.get('released'):
                c_['released']=False; c_['gist']=False; c_['n']=0; c_.pop('restore_pending',None); c_.pop('rel_pending',None); c_['so']=[]; nb_r+=1
                if int(getattr(cfg,'inf_restore_refit',0)): c_['cf_refit']=True       # HANDOVER THAT HOLDS: a failed release is also re-fitted into the weights at the next edit
        st_['rel']+=nr_; st_['res']+=nb_r
        if nr_ or nb_r or ncx_: print(f"  [inference CLS | RELEASE] {nr_} conclusions released (erased from the fast store, kept in trunk replay), {nb_r} restored (written back first), {ncx_} pending releases cancelled (failed the re-test)",flush=True)
    cb9_=int(getattr(cfg,'inf_cap_budget',0)); cr9_=int(getattr(cfg,'inf_cap_read',0))
    if cb9_>0 or cr9_:                                                          # CAPACITY: load of the fast store, and (inf_cap_budget) a budget with eviction by value
        try:
            live9_=[c_ for c_ in allc_ if not c_.get('freed') and not c_.get('released')]; fx9_=[f_ for f_ in S_.get('facts',[]) if not f_.get('freed')]
            ev9_=0; evx_=[]
            if cb9_>0 and len(live9_)>cb9_:
                vis3_=int(model.__dict__.get('_visit',0)); dep9_={}
                for x_ in live9_:
                    if int(x_.get('dep',1))>1: dep9_[(x_.get('A'),x_.get('B'))]=dep9_.get((x_.get('A'),x_.get('B')),0)+1
                def val9_(c_):                                                  # value: dependents x support x recency (recently made or used counts more)
                    nd_=dep9_.get((c_.get('A'),c_.get('C')),0); sp_=min(5.0,float(c_.get('cor',1) if isinstance(c_.get('cor',1),(int,float)) else 1))
                    rc_=1.0/(1.0+max(0,vis3_-int(c_.get('lr',c_.get('v',0))))); return (1+nd_)*(1+sp_)*(0.5+rc_)
                cand9_=[c_ for c_ in live9_ if not (c_.get('uncertain') or c_.get('rel_pending') or c_.get('restore_pending') or c_.get('hold')
                                                    or dep9_.get((c_.get('A'),c_.get('C')),0)>0)]
                for c_ in sorted(cand9_,key=val9_)[:len(live9_)-cb9_]:
                    _inf_conc_fade(c_,S_slot,Lr_,filler,feats,dev_,beta_); c_['freed']=True; c_['faded']=True; c_['evicted']='capacity budget'; ev9_+=1; evx_.append(c_)
                S_.setdefault('_capev',[]).extend((x_['A'],x_['C']) for x_ in evx_)
            M9_=_INFM.get('M'); nz9_=float((M9_.abs().sum(1)>0).float().mean()) if M9_ is not None else float('nan')
            t2c_,_=_inf_truth(cfg); ne9_=len(S_.get('_capev',[])); net_=sum(1 for a_,c9_ in S_.get('_capev',[]) if (a_,c9_) in t2c_)
            print(f"  [inference CAPACITY] fast store: {len(live9_)-ev9_} live conclusions + {len(fx9_)} facts, {len(_INFM.get('names',{}))} names, rows in use {nz9_:.0%}, |M| {float(M9_.norm()) if M9_ is not None else float('nan'):.1f}"
                  +(f" | BUDGET {cb9_}: {ev9_} evicted by value this sleep ({ne9_} in all, {net_} of them true probe conclusions)" if cb9_>0 else ""),flush=True)
        except Exception as e9_: _fail('inference.sleep:614'); print(f"  [inference CAPACITY] unavailable ({type(e9_).__name__}: {e9_})"[:240],flush=True)
    nmt_=int(getattr(cfg,'inf_mem_maint',0))
    if nmt_>0 and _INFM.get('mu') is not None:                                  # STORE MAINTENANCE: the trunk keeps learning, so the store's keys drift;
        old_=[c_ for c_ in allc_ if not c_.get('freed') and not c_.get('released') and c_.get('n',0)>0]   # FIX (review): released ones are erased on purpose       #   stored conclusions it no longer recalls are rewritten first
        nm_=[n_.decode('latin-1') for n_ in _INFM['names']]; fail_=0; ck_=[]
        if old_ and len(nm_)>=25:
            for c_ in [old_[i_] for i_ in rs_.choice(len(old_),min(len(old_),nmt_),replace=False)]:
                oth_=[n_ for n_ in nm_ if n_ not in (c_['A'],c_['C'])]; cs_=[c_['C']]+list(rs_.choice(oth_,19,replace=False))
                ok_=_inf_score1(model,cfg,f"So {c_['A']} is connected to ",cs_,int(S_slot(c_['d'])),True,dec_=True)==0; ck_.append(ok_)
                if not ok_: c_['n']=0; fail_+=1
        ainfo_+=f" | MAINTENANCE: {fail_} of {len(ck_)} stored conclusions no longer recalled (store recall {np.mean(ck_) if ck_ else float('nan'):.2f}) -> rewritten first"
    rc_=[];ra_=[]; wn_=0
    rn_=[i_ for i_,(w0_,_) in enumerate(ring_) if any(n_ in _INFM['names'] for n_ in _INF_NAME.findall(w0_))] if float(getattr(cfg,'inf_mem_anchor_named',0.0))>0 else []
    ad9_=int(getattr(cfg,'inf_sleep_adapt',0)) and ord_=='nrem_rem'; Kx9_=int(getattr(cfg,'inf_sleep_cycles_max',4)); NS9_=int(cfg.inf_steps)
    if ad9_: REMk_=0                                                            # cg26n ADAPTIVE SLEEP: cycles and durations from the write fit and the REM yields
    cy9_={'c0':0,'fit':[],'acc':[],'len':[],'stop':'','end':''}
    for st in range(2*NS9_ if ad9_ else NS9_):
        nrc9_=len(rc_)
        grp_={}
        fxs_=int(getattr(cfg,'inf_mem_fxsep',0))                               # SEPARATE WRITE BUDGETS: conclusions keep their own rows; recorded facts get fxs_ rows of their own
        live_=([c0_ for c0_ in allc_ if not c0_.get('freed') and not c0_.get('gist')] or [c0_ for c0_ in allc_ if not c0_.get('freed') and not c0_.get('released')])+([] if fxs_ else [f0_ for f0_ in FX_ if not f0_.get('freed')])       # sleep-order test: NREM also replays this visit's recorded facts
        fl_=[f0_ for f0_ in FX_ if not f0_.get('freed') and not f0_.get('gist')] if fxs_ else []
        if not live_ and not fl_: break
        if int(getattr(cfg,'inf_mem_prio',0)):                                  # cg20i: conclusions below the exposure target are written first
            lo_=[c0_ for c0_ in live_ if c0_.get('n',0)<int(cfg.inf_target)]; live_=lo_ or live_
            if int(getattr(cfg,'inf_mem_prio',0))>=2:                              # cg20j: RARE LINKS FIRST - the rarest-named conclusions take the write budget
                dfm_=S_['df']; live_=sorted(live_,key=lambda c0_: dfm_.get(c0_['A'].encode('latin-1'),0)+dfm_.get(c0_.get('C',c0_['B']).encode('latin-1'),0))[:max(4*kc_,256)]
        sf_=float(getattr(cfg,'inf_mem_src',0.0)); add_=str(getattr(cfg,'inf_mem_src_mode','split'))=='add'
        if int(getattr(cfg,'inf_replay_gumbel',0)) and live_:                   # GUMBEL TOP-k: replay rows sampled in proportion to priority (gain x need) in one pass
            lp_=np.log(np.array([max(0.05,float(c0_.get('pri',c0_.get('gain',1.0)) or 1.0)) for c0_ in live_]))
            gk_=lp_-np.log(-np.log(rs_.uniform(1e-9,1.0,size=(kc_,len(live_)))))
            sel_=[(live_[int(i_)],'c' if add_ else None) for i_ in gk_.argmax(1)]
        else: sel_=[(live_[i_],'c' if add_ else None) for i_ in rs_.choice(len(live_),kc_)] if live_ else []
        if add_ and sf_>0 and live_: sel_+=[(live_[i_],'s') for i_ in rs_.choice(len(live_),max(1,int(round(kc_*sf_))))]   # one-hop facts IN ADDITION to the conclusion rows
        if fl_:                                                                 #   NREM replay by salience: under-written facts first, derived (rule) facts, then the most repeated
            lo_=[f0_ for f0_ in fl_ if f0_.get('n',0)<int(cfg.inf_target)] or fl_
            lo_=sorted(lo_,key=lambda f0_: (not f0_.get('derived'),-f0_.get('rep',1),f0_.get('n',0)))[:max(4*fxs_,256)]
            sel_+=[(lo_[i_],'c') for i_ in rs_.choice(len(lo_),fxs_)]
        for c_,kd_ in sel_:
            if kd_!='s': c_['n']=c_.get('n',0)+1
            if c_.get('fact'): sn_=c_['s']                                       # a recorded one-hop fact (NREM replay)
            elif kd_=='s' or (kd_ is None and sf_>0 and rs_.rand()<sf_): sn_=S_['it'][c_['src'][rs_.randint(2)]]['s'].decode('latin-1').strip()   # a verified SOURCE fact (one-hop) of this conclusion
            else: sn_=c_['s'] if (c_['typed'] and rs_.rand()<0.5) else (f"So {c_['A']} is connected to {c_['C']}." if rs_.rand()<0.5 else f"So {c_['C']} is connected to {c_['A']}.")
            if rs_.rand()<fa_:                                                  # no context before the conclusion (as the probes are asked)
                it_=(sn_+' ').encode('latin-1')[:Lr_-2]; w_=bytearray(it_)+bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); o_=0
            else:
                w_=bytearray(filler(c_['d'])[:Lr_]); w_+=b' '*(Lr_-len(w_)); it_=(' '+sn_+' ').encode('latin-1')[:Lr_-8]; o_=rs_.randint(4,max(5,Lr_-len(it_)-4)); w_[o_:o_+len(it_)]=it_
            w_=w_[:Lr_]; mk_=torch.zeros(Lr_-1); mk_[max(0,o_-1) if o_>0 else 0:o_+len(it_)-1]=1.0
            grp_.setdefault(int(S_slot(c_['d'])),[]).append((torch.tensor(list(bytes(w_)),dtype=torch.long),mk_))
        for _ in range(ka_):
            if ring_:
                af_=float(getattr(cfg,'inf_mem_anchor_named',0.0))                 # cg20i: do-no-harm anchors aimed at text where stored names occur
                if af_>0 and rn_ and rs_.rand()<af_: w0_,dd_=ring_[rn_[rs_.randint(len(rn_))]]
                else: w0_,dd_=ring_[rs_.randint(len(ring_))]
                o0b_=rs_.randint(max(1,len(w0_)-Lr_+1)); xa_=torch.tensor(list(w0_[o0b_:o0b_+Lr_]),dtype=torch.long); da_=int(S_slot(dd_))
            else:
                e_=kept_[rs_.randint(len(kept_))]; xa_=e_['x'].to('cpu',torch.long)[:Lr_]; da_=int(e_['d'])
            if xa_.numel()<Lr_: xa_=torch.cat([xa_,torch.full((Lr_-xa_.numel(),),32,dtype=torch.long)])
            grp_.setdefault(da_,[]).append((xa_,None))
        for d_,rows_ in grp_.items():
            X_=torch.stack([r_[0] for r_ in rows_]).to(dev_); lg_,z_=feats(X_,d_); Y_=X_[:,1:]
            if _INFM['key']=='latch':                                                    # threshold: 85th percentile of word-start surprise on ordinary (anchor) rows
                ar_=[i_ for i_,r_ in enumerate(rows_) if r_[1] is None]
                if ar_:
                    _,_,ws_=_infm_latch(z_[ar_].float()-_INFM['mu'],X_[ar_,:-1],lg_[ar_])
                    if ws_: q85_=float(np.percentile(ws_,85)); _INFM['lthr']=q85_ if _INFM['lthr'] is None else 0.9*_INFM['lthr']+0.1*q85_
            K_,id2_=_infm_keys(z_,X_[:,:-1],lg_); zf_=K_.reshape(-1,K_.shape[-1]); idf_=id2_.reshape(-1); c0_=_infm_corr(zf_,idf_); lgf_=lg_.reshape(-1,256)
            q_=F.softmax(lgf_+c0_,-1); p0_=F.softmax(lgf_,-1); oh_=F.one_hot(Y_.reshape(-1),256).float()
            isc_=torch.tensor([r_[1] is not None for r_ in rows_],device=dev_)[:,None].expand(-1,Lr_-1).reshape(-1)
            mk_=torch.stack([r_[1] if r_[1] is not None else torch.zeros(Lr_-1) for r_ in rows_]).to(dev_).reshape(-1)
            nz_=(Y_.reshape(-1)>0).float()*(~isc_).float()
            for it_ in range(int(getattr(cfg,'inf_mem_iters',1))):                  # repeated delta writes on the same rows (features fixed): fit to the fixed point
                if it_>0: q_=F.softmax(lgf_+_infm_corr(zf_,idf_),-1)
                R_c_=(oh_-q_)*mk_[:,None]; R_a_=((oh_ if int(getattr(cfg,'inf_mem_replay',0)) and float(model.__dict__.get('_real_w',1.0))==0 else p0_)-q_)*nz_[:,None]   # conclusions: fit the bytes | anchors: undo any effect (or, inf_mem_replay in closed-loop passes: LEARN the verified replay bytes)
                _infm_write(zf_,idf_,R_c_+R_a_,beta_,gate_lr=(float(getattr(cfg,'inf_mem_gate_lr',0.002)) if (_INFM['key']=='slowgate' and it_==0) else 0.0),tmask=((R_c_.abs().sum(-1)>0) if _INFM.get('ds_on') else None))
            q_=F.softmax(lgf_+_infm_corr(zf_,idf_),-1); mk_=mk_*(idf_>=0).float(); _INFM['wpos']=_INFM.get('wpos',0)+int(mk_.sum())     # fit after the writes, on the store-keyed conclusion bytes
            if mk_.sum()>0: rc_.append(float(-(q_.gather(-1,Y_.reshape(-1,1)).squeeze(-1).clamp(min=1e-9).log()*mk_).sum()/mk_.sum()/math.log(2)))
            if nz_.sum()>0: ra_.append(float(((q_*(q_.clamp(min=1e-9).log()-p0_.clamp(min=1e-9).log())).sum(-1)*nz_).sum()/nz_.sum()))
            wn_+=1
        if ad9_:
            if len(rc_)>nrc9_: cy9_['fit'].append(float(np.mean(rc_[nrc9_:])))
            L9_=st+1-cy9_['c0']; f9_=cy9_['fit']
            conv9_=(L9_>=max(4,NS9_//10) and len(f9_)>=5 and f9_[-1]>=0.98*f9_[-5]) or L9_>=max(4,NS9_//2) or not f9_
            if conv9_ and L9_>=max(4,NS9_//10):
                cy9_['len'].append(L9_); cy9_['c0']=st+1; cy9_['fit']=[]
                if not cy9_['stop']:
                    o9_=_inf_rem(model,cfg,S_,S_slot,int(getattr(cfg,'inf_rem_n',48)),f"NREM -> REM, adaptive cycle {len(cy9_['acc'])+1}"); a9_=len(o9_ or []); A9_=cy9_['acc']; A9_.append(a9_)
                    if a9_==0: cy9_['stop']='closure (a cycle accepted nothing new)'
                    elif len(A9_)>=2 and A9_[-2]>0 and a9_*a9_/A9_[-2]<1.0: cy9_['stop']=f'yield falling: next cycle predicted {a9_*a9_/A9_[-2]:.1f} < 1 new conclusion'
                    elif len(A9_)>=Kx9_: cy9_['stop']=f'cycle cap {Kx9_}'
                else: cy9_['end']='writes converged after the last REM'; break
        if REMk_ and remd_<REMk_ and (st+1)%max(1,int(cfg.inf_steps)//(REMk_+1))==0:   # FIX (review): was //REMk_, so one cycle fired at the LAST step   # NREM chunk done -> a REM episode (cycles: its links are written by the next chunk)
            remd_+=1; _inf_rem(model,cfg,S_,S_slot,max(1,int(getattr(cfg,'inf_rem_n',48))//REMk_),f"NREM -> REM, cycle {remd_}/{REMk_}")
    if ad9_: print(f"  [inference SLEEP CYCLES | ADAPTIVE] {len(cy9_['acc'])} REM cycles, accepted per cycle {cy9_['acc']} | NREM chunks (write steps) {cy9_['len']} + {st+1-cy9_['c0']} | "
                   f"{st+1} write steps in all (fixed schedule {NS9_}) | stopped: {cy9_['stop'] or 'step cap'}"+(f"; {cy9_['end']}" if cy9_['end'] else ''),flush=True)
    nh_=int(getattr(cfg,'inf_prune_harm',0))
    if nh_>0 and rn_:                                                           # HARM-BASED PRUNING: each stored name's measured effect on recent REAL text
        H_=S_.setdefault('harm',{}); inv_={v_:k_ for k_,v_ in _INFM['names'].items()}; acc_={}
        pick_=[ring_[rn_[i_]] for i_ in rs_.choice(len(rn_),min(len(rn_),nh_),replace=False)]
        for dd0_ in sorted(set(int(d_) for _,d_ in pick_)):
            rows_=[w_ for w_,d_ in pick_ if int(d_)==dd0_]
            for b0_ in range(0,len(rows_),16):
                X_=torch.stack([torch.tensor(list((w_+b' '*Lr_)[:Lr_]),dtype=torch.long) for w_ in rows_[b0_:b0_+16]]).to(dev_); sl0_=int(S_slot(dd0_)); tg_=X_[:,1:,None]
                _INFM['force']=False; lo_=F.log_softmax(fwd(X_,sl0_),-1).gather(-1,tg_).squeeze(-1)
                _INFM['force']=True; ln_=F.log_softmax(fwd(X_,sl0_),-1).gather(-1,tg_).squeeze(-1); _INFM.pop('force',None)
                dh_=(lo_-ln_).detach().cpu().numpy(); ids_=_infm_ids(X_[:,:-1]).cpu().numpy()     # > 0: the store makes the true byte LESS likely
                for v_ in np.unique(ids_[ids_>=0]):
                    m_=ids_==v_; a0_=acc_.setdefault(int(v_),[0.0,0,[]]); a0_[0]+=float(dh_[m_].sum()); a0_[1]+=int(m_.sum())
                    for j0_ in range(m_.shape[0]):
                        if m_[j0_].any(): a0_[2].append(float(dh_[j0_][m_[j0_]].sum())/math.log(2))   # one SAMPLE per window: bits the store costs (+) or saves (-) there
        thr_=float(getattr(cfg,'inf_prune_harm_thr',0.02)); pn0_=set(x_.encode('latin-1') for q_ in inf_probes(cfg) for x_ in q_['names']); cut_=[]
        if int(getattr(cfg,'inf_prune_mdl',0)):                                # MDL / SEQUENTIAL BIT ACCOUNTING: every stored name is a 'rule' that must pay for itself.
            G_=S_.setdefault('mdl',{}); z_=float(getattr(cfg,'inf_mdl_z',2.0)); nmin_=int(getattr(cfg,'inf_mdl_min',8)); cst_=float(getattr(cfg,'inf_mdl_cost',0.0))
            for v_,(s_,n_,ws_) in acc_.items():                                #   bits it costs or saves accumulate over ALL sleeps (one sample per window it fires in);
                nm_=inv_.get(v_)                                               #   drop only when the whole evidence says it costs bits with confidence (lower bound > 0)
                if nm_ is None: continue
                g_=G_.setdefault(nm_,[0,0.0,0.0])
                for w0_ in ws_: g_[0]+=1; g_[1]+=w0_+cst_; g_[2]+=(w0_+cst_)**2
            nu_=0; nh0_=0; nd_=0; rel_=float(getattr(cfg,'inf_mdl_rel',0.0)); n0_=float(getattr(cfg,'inf_mdl_n0',8.0)); rinfo_=''
            live0_=[(nm_,v_) for nm_,v_ in G_.items() if nm_ in _INFM['names'] and v_[0]>0]
            med_=float(np.median([-v_[1]/v_[0] for _,v_ in live0_])) if live0_ else 0.0      # GOING RATE: median bits SAVED per window across stored names
            nrel_=0
            for nm_,(n_,s_,ss_) in G_.items():
                if nm_ not in _INFM['names'] or n_<nmin_: nd_+=int(nm_ in _INFM['names']); continue
                mu_=s_/n_; se_=math.sqrt(max(ss_/n_-mu_*mu_,1e-12)/n_)
                if mu_-z_*se_>0: cut_.append(nm_)
                elif rel_>0 and med_>0 and (n_*(-mu_)+n0_*med_)/(n_+n0_)<rel_*med_: cut_.append(nm_); nrel_+=1   # EMPIRICAL BAYES: the name's gain shrunk towards the going rate
                elif mu_+z_*se_<0: nu_+=1
                else: nd_+=1
            if rel_>0: rinfo_=f" | going rate {med_:+.3f} bits saved per window; {nrel_} removed as below {rel_:.2f} x the going rate after shrinkage (prior weight {n0_:.0f} windows)"
            ainfo_+=f" | MDL ACCOUNTING: {len(G_)} names with evidence; proven to SAVE bits {nu_}, undecided {nd_}, removed {len(cut_)} (probe names among them {sum(1 for x_ in cut_ if x_ in pn0_)})"+rinfo_
            acc_={}
        for v_,(s_,n_,*_) in acc_.items():
            nm_=inv_.get(v_)
            if nm_ is None or n_<int(getattr(cfg,'inf_prune_harm_min',8)): continue
            h0_=H_.get(nm_); H_[nm_]=s_/n_ if h0_ is None else 0.5*h0_+0.5*s_/n_
            if H_[nm_]>thr_ and h0_ is not None and h0_>thr_: cut_.append(nm_)      # harmful in two sleeps running
        for nm_ in cut_:
            _INFM['names'].pop(nm_,None); PN_.add(nm_); s0_=nm_.decode('latin-1')
            for c_ in list(allc_)+FX_:
                if s0_ in (c_.get('A'),c_.get('B'),c_.get('C')): c_['freed']=True; c_['pruned']=True
        ainfo_+=f" | HARM PRUNING: {len(acc_)} stored names measured on {len(pick_)} recent real windows, {len(cut_)} keys removed as net-harmful (probe names among them {sum(1 for x_ in cut_ if x_ in pn0_)}), total pruned names {len(PN_)}"
    if int(getattr(cfg,'inf_rem_shadow',0)) and ord_=='nrem':                  # SHADOW REM: lookup recombination after the writes, counted only
        try: _inf_rem(model,cfg,S_,S_slot,int(getattr(cfg,'inf_rem_n',48)),'SHADOW after NREM writes (counted, not written; "already known" = found by discovery)',shadow_=True)
        except Exception as e_: _fail('inference.sleep:758'); print(f"  [inference REM recombination] skipped: {e_!r}"[:300],flush=True)
    _inf_sync(cfg,'store writes'); xinfo_=_inf_xfer(model,cfg,t,ts,vals,olds_,S_slot,S_,allc_,ring_,Lr_,kc_,ka_,R_,filler,fwd,rs_,dev_,nb_,fa_)
    xinfo_+=_inf_free(model,cfg,S_,allc_,S_slot,Lr_,filler,fwd,feats,rs_,dev_,beta_)
    xinfo_+=_inf_prune(model,cfg,S_,allc_,S_slot,Lr_,filler,feats,rs_,dev_,beta_)
    model.train(was_); _inf_sync(cfg,'transfer / freeing / pruning')
    b1_=bpc(model,vals[t],ts,n=nb_); _inf_sync(cfg,'held-out BPC with the store'); o1_=float(np.mean([bpc(model,vals[d_],s_,n=nb_) for d_,s_ in olds_])) if olds_ else float('nan')
    print(f"  [inference sleep | FAST STORE] keyed positions written {_INFM.get('wpos',0)} | {int(cfg.inf_steps)} write steps ({kc_} conclusion rows + {ka_} anchor rows) on {len(allc_)} conclusions | "
          f"keyed conclusion bytes after writing {rc_[0] if rc_ else float('nan'):.3f} (first step), {np.mean(rc_[-5:]) if rc_ else float('nan'):.3f} (last steps) bits | anchor KL(store||model) {np.mean(ra_[-5:]) if ra_ else float('nan'):.4f} | "
          f"|M| {float(_INFM['M'].norm()):.1f}{ainfo_} | trunk weight sum {float(sum(p0_.detach().double().abs().sum() for p0_ in model.parameters())):.6e}{' | gates '+str([round(float(g0_),2) for g0_ in _INFM['gates']]) if _INFM['key']=='slowgate' else ''}{xinfo_} | held-out BPC (store on) visited {b0_:.3f} -> {b1_:.3f}, old domains {o0_:.3f} -> {o1_:.3f}",flush=True)
    if int(cfg.inf_eval_every)>0 and (int(model.__dict__.get('_visit',0))+1)%int(cfg.inf_eval_every)==0: inf_eval(model,cfg,S_,f"visit {int(model.__dict__.get('_visit',0))+1}")

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_cai_report': 'platypus.inference.readouts',
    '_cai_score': 'platypus.inference.readouts',
    '_compact': 'platypus.inference.selftaught',
    '_inf_audit': 'platypus.inference.audit',
    '_inf_conc_fade': 'platypus.inference.audit',
    '_inf_facts': 'platypus.inference.rem',
    '_inf_free': 'platypus.inference.selftaught',
    '_inf_hyp': 'platypus.inference.audit',
    '_inf_isotropy': 'platypus.inference.jepa',
    '_inf_jepa': 'platypus.inference.jepa',
    '_inf_lp': 'platypus.inference.selftaught',
    '_inf_mix': 'platypus.inference.mixing',
    '_inf_prune': 'platypus.inference.audit',
    '_inf_rem': 'platypus.inference.rem',
    '_inf_remgen': 'platypus.inference.selftaught',
    '_inf_rules': 'platypus.inference.rem',
    '_inf_score1': 'platypus.inference.readouts',
    '_inf_sofl': 'platypus.inference.need',
    '_inf_use': 'platypus.inference.audit',
    '_inf_xfer': 'platypus.inference.readouts',
    '_jd_train': 'platypus.inference.jepa',
    '_jepa_emb1': 'platypus.inference.jepa',
    'inf_self': 'platypus.inference.rem',
}
