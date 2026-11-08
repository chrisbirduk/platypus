"""run(cfg): the whole continual-learning run."""
from platypus.common import _fail
import numpy as np
import time
import torch
from platypus.common import (
    DEVICE,
)
from platypus.config import (
    BASE_DOMAINS, Config, DOMAIN_DIFFICULTY, PILE_DOMAINS,
)
from platypus.gating import (
    _DSPLIT, _NOISE,
)
from platypus.protection import (
    _DP2, _FILM, _STF, probe_post, probe_pre, proj_setup, smem_reencode,
)
from platypus.private import (
    _XDG, ngp_init,
)
from platypus.slots import (
    _ReplayGet, _StreamGet, _StreamSource, _slot_score, infer_slot, slot_ref,
)
from platypus.layers import (
    _CARRY, setup_kernel,
)
from platypus.capacity import (
    Reserve, Uniq, _lab,
)
from platypus.dream_bank import (
    bank_refresh,
)
from platypus.ngrams import (
    _new_opt, setup_compile,
)
from platypus.model import (
    Model,
)
from platypus.meta import (
    MetaController,
)
from platypus.diagnostics import (
    TopicTracker,
)
from platypus.importance import (
    fold_domain_covariance, install_cov_hooks, rebuild_old_cov, revisit_begin, revisit_end, update_importance,
)
from platypus.data import (
    _RUN_T0, _inject_facts, _using_hf_stream, fact_eval, locality_metrics, make_loader, overlap_check, prefetch_pile, stream_bytes,
)
from platypus.heads import (
    bpc_inferred, bpc_inferred_modes, mix_text_eval,
)
from platypus.train import (
    calibrate_affinity, train_domain,
)
from platypus.orth import (
    orth_summary, print_orth,
)
from platypus.conduit import (
    lora_merge, sem_rescore, set_seed,
)
from platypus.inference.store import (
    bpc,
)
from platypus.inference.probes import (
    inf_plant,
)
from platypus.inference.sentences import (
    _inf_store,
)
from platypus.inference.evaluate import (
    inf_eval,
)
from platypus.inference.sleep import (
    inf_sleep,
)
from platypus.inference.readouts import (
    inf_final_bpc,
)
from platypus.inference.need import (
    pu_consolidate,
)
from platypus.losses import (
    oracle_final, recon_fidelity, replay_diag,
)

def run(cfg):
    if cfg.stream: cfg.infer_ctx=True                  # NO VISIT BOUNDARIES: slots are always inferred
    for k in ('theta_low','theta_high','theta_dist','theta_sigma','theta_reserve','drive_ema','drive_slow','peak_decay','route_width','hebb_rate'): setattr(Config,k,getattr(cfg,k))  # buffers read class attrs
    set_seed(cfg.seed)
    _NOISE['gen']=torch.Generator().manual_seed(cfg.seed+12345)
    for fn in ('_prev_bpc','_plateau_count','_avg_difficulty'):
        if hasattr(train_domain,fn): delattr(train_domain,fn)
    print(f"Run | threshold_mode={cfg.threshold_mode} drive_ref={cfg.drive_ref} use_sens={cfg.use_sens} seed={cfg.seed}")
    PILE_DOMAINS[:]=BASE_DOMAINS+[d for d in cfg.extra_domains if d not in BASE_DOMAINS]   # before the heads are built
    model=Model(cfg).to(DEVICE)
    if getattr(model,'_ng_cur',None) is not None:
        model.__dict__['_ng_cur']=[t_.to(DEVICE) for t_ in model._ng_cur]; model.__dict__['_ng_old']=[t_.to(DEVICE) for t_ in model._ng_old]
    opt=_new_opt(model,cfg)
    for nm_ in ('dream_route','strand_route'):
        r_=getattr(cfg,nm_)
        if r_ is not None:
            if len(r_)!=len(model.blocks)+1: raise ValueError(f"{nm_} needs {len(model.blocks)+1} values (emb, b0..b{len(model.blocks)-1}), got {r_}")
            if any(r_[j]>0 and r_[j+1]<=0 for j in range(len(r_)-1)): print(f"  [route] note: {nm_} {r_}: a part below a zero-weight part gets no dream gradient")
    setup_kernel(model, cfg); setup_compile(model, cfg); _t_run=time.perf_counter()
    _STF['on']=False
    proj_setup(model, cfg)
    _XDG.update(on=cfg.xdg>0,keep=1.0,seed=int(cfg.seed),masks={},cur=None,rows=None,tagon=(cfg.pu_k>0 or cfg.cgate>0 or cfg.film_k>0))
    _FILM['codes']=getattr(model,'film_codes',None) if int(cfg.film_k)>0 else None   # film_k 0: FiLM off
    _DP2['on']=int(cfg.dp_steps)>=2
    if cfg.pu_k>0: print(f"  [private trunk units] {cfg.pu_k} private MLP units per domain in every block")
    for i_,b in enumerate(model.blocks): b.__dict__['_xdg_i']=i_
    from platypus.inference.slowhead import slow_init; slow_init(model, cfg)          # north-star rung 1 (slow_k > 0)
    ngp_init(model, cfg); print(f"  [private n-grams] per-domain hashed tables, orders {tuple(cfg.ngp)}, {cfg.ngp_rows} rows each, before each domain's own head" if cfg.ngp else "  [private n-grams] none (ngp=())")
    try:                                                                         # PARAMETER COUNT (issue #23): shared trunk vs the parts private to each slot
        nps_=lambda ps_: int(sum(p_.numel() for p_ in ps_))
        pu1_=sum(int(b.pu_in[0].numel()+b.pu_b[0].numel()+b.pu_out[0].numel()) for b in model.blocks if hasattr(b,'pu_in'))
        ng1_=len(tuple(cfg.ngp))*int(cfg.ngp_rows)*int(cfg.d_model) if cfg.ngp else 0
        hd1_=0 if cfg.shared_head else nps_(model.heads[0].parameters()); fc1_=int(model.film_codes[0].numel()) if hasattr(model,'film_codes') else 0; ns_=len(model.heads)
        tot_=nps_(model.parameters()); shng_=nps_(model.ngram.parameters()) if hasattr(model,'ngram') else 0
        priv_reg_=ns_*(hd1_+fc1_+pu1_)                                         # registered private parameters (the private n-gram tables live outside .parameters())
        print(f"  [parameters] shared trunk {(tot_-priv_reg_-shng_)/1e6:.1f}M + shared input n-gram tables {shng_/1e6:.1f}M | PRIVATE per slot {(pu1_+ng1_+hd1_+fc1_)/1e6:.1f}M "
              f"(trunk units {pu1_/1e6:.1f}M, n-gram tables {ng1_/1e6:.1f}M, head {hd1_/1e6:.2f}M{' - ONE SHARED HEAD (shared_head), counted in the trunk' if cfg.shared_head else ''}, FiLM code {fc1_} numbers{' - FiLM OFF (film_k 0)' if not fc1_ else ''}), x {ns_} slots = {ns_*(pu1_+ng1_+hd1_+fc1_)/1e6:.1f}M",flush=True)
    except Exception as e_: _fail('runner:123'); print(f"  [parameters] unavailable ({type(e_).__name__}: {e_})")
    for b in model.blocks: b._surp_on=bool(cfg.surprise_mod); b.__dict__['_lean']=bool(cfg.lean_stats)
    _DSPLIT['on']=bool(cfg.dream_mom>0 or cfg.muon_mom_dream>0)
    _init_sd=None
    model._reserve=None
    object.__setattr__(model,'_replay_buffer',[])
    install_cov_hooks(model, cfg)
    kl_log={}; dream_log={}; fd_log={}; fd_snaps={}; battr_log={}
    if cfg.topic_diag: object.__setattr__(model,'_topics',TopicTracker(cfg,DEVICE)); topic_log={}
    cap_s=cfg.d_state*cfg.reserve_mult; cap_m=2*cfg.d_model*cfg.reserve_mult
    def _fx(li,base):                                # 'fresh' extra units held in reserve for later domains
        e=0.0
        return int(round(base*e))
    ns0_=int(round(cfg.d_state*cfg.init_state_mult)); nm0_=int(round(2*cfg.d_model*cfg.init_mlp_mult))
    model._reserve=[Reserve(cfg.d_model, max(0,cap_s-b.d_state)+_fx(li,ns0_), max(0,cap_m-b.mlp_hidden)+_fx(li,nm0_), DEVICE)
                    for li,b in enumerate(model.blocks)]
    print("  [capacity] "+("wider from the start")
                              +": "+" ".join(f"b{li} state {b.d_state}+{R.Wk.shape[0]} MLP {b.mlp_hidden}+{R.min.shape[0]}" for li,(b,R) in enumerate(zip(model.blocks,model._reserve))))
    model._uniq=[Uniq(cfg.d_model, DEVICE) for _ in model.blocks]; model._rgen=torch.Generator().manual_seed(cfg.seed+777)
    model._recruit_log={}
    for b in model.blocks: b._store_uniq=True
    cfg.log_locality=False
    print(f"  [reserve] capacity x{cfg.reserve_mult}: state {cap_s}, MLP {cap_m} per block; initially active "
          f"state {model.blocks[0].d_state}, MLP {model.blocks[0].mlp_hidden}")
    cfg.route_writes=(cfg.route_mode=='write')
    if cfg.unit_budget is None: cfg.unit_budget=cfg.write_budget
    if cfg.channel_budget is None: cfg.channel_budget=cfg.write_budget
    if cfg.channel_gate_mode is None: cfg.channel_gate_mode=cfg.gate_mode
    for b in model.blocks:
        b.route_in_forward=(cfg.route_mode=='forward'); b.write_routed=(cfg.route_mode=='write')
    meta=MetaController().to(DEVICE); meta_opt=torch.optim.Adam(meta.parameters(),lr=cfg.meta_lr)
    npass_=max(1,cfg.n_passes); fresh_=bool(cfg.pass_fresh_data) and npass_>1
    vseed_=(lambda i: 17777+i) if fresh_ else (lambda i: 7777+i); tseed_=(lambda i: 11000+i) if fresh_ else (lambda i: 1000+i)
    shard_=int(cfg.pass_shard_bytes)
    if cfg.no_revisit:                                   # a visit must not run out of fresh windows: steps x batch x (seq_len+1), +1% for the
        shard_=max(shard_,int(cfg.steps_per_task*cfg.batch_size*(cfg.seq_len+1)*1.01))      # document joins
    TRN_=shard_*(npass_ if fresh_ else 1)
    if _using_hf_stream():                           # new domains (fresh-pass data: ALL domains) collected together in one streaming pass
        prefetch_pile([(d,(d,vseed_(i),500_000),(d,tseed_(i),TRN_)) for i,d in enumerate(PILE_DOMAINS)])   # issue #30: every domain document-disjoint (base domains too)
    if fresh_:                                       # UNSEEN TEXT EVERY PASS: disjoint shards, loaders built per pass
        raw_tr_=[stream_bytes(d,tseed_(i),TRN_) for i,d in enumerate(PILE_DOMAINS)]; trp_={}
        print("  [data] fresh text every pass: per domain "+", ".join(f"{d[:10]} {len(r)//npass_/1e6:.1f}MB x{npass_}" for d,r in zip(PILE_DOMAINS,raw_tr_)))
        def train_of(t,p):
            if (t,p) not in trp_:
                for k_ in [k_ for k_ in trp_ if k_[1]<p]: del trp_[k_]               # earlier passes' shards are never used again
                sh_=len(raw_tr_[t])//npass_; trp_[(t,p)]=make_loader(_inject_facts(raw_tr_[t][p*sh_:(p+1)*sh_],cfg,t,p) if cfg.fact_spec else (inf_plant(raw_tr_[t][p*sh_:(p+1)*sh_],cfg,t,p) if cfg.inf_on else raw_tr_[t][p*sh_:(p+1)*sh_]),cfg,no_revisit=cfg.no_revisit,lanes=bool(cfg.ctx_carry or cfg.ctx_lanes),jump=int(cfg.ctx_jump),keep=bool(cfg.ctx_jump_keep))
            return trp_[(t,p)]
        trains=None
    else:
        trains=[make_loader((inf_plant(stream_bytes(d,tseed_(i),TRN_),cfg,i,0) if cfg.inf_on else stream_bytes(d,tseed_(i),TRN_)),cfg,no_revisit=cfg.no_revisit,lanes=bool(cfg.ctx_carry or cfg.ctx_lanes),jump=int(cfg.ctx_jump),keep=bool(cfg.ctx_jump_keep)) for i,d in enumerate(PILE_DOMAINS)]
        train_of=lambda t,p: trains[t]
        if cfg.fact_spec:                                                        # UPDATE-OR-PROTECT TEST (shared text across passes): a fact domain
            fdom_={int(e[0]) for e in cfg.fact_spec}; ftr_={}                    #   gets its own loader per pass, the same text with that pass's facts
            def train_of(t,p):
                if t not in fdom_: return trains[t]
                if (t,p) not in ftr_:
                    raw_=_inject_facts(stream_bytes(PILE_DOMAINS[t],tseed_(t),TRN_),cfg,t,p)
                    ftr_[(t,p)]=make_loader(raw_,cfg,no_revisit=cfg.no_revisit,lanes=bool(cfg.ctx_carry or cfg.ctx_lanes),jump=int(cfg.ctx_jump),keep=bool(cfg.ctx_jump_keep))
                return ftr_[(t,p)]
    vals=[make_loader(stream_bytes(d,vseed_(i),500_000),cfg,fixed_seed=4242+i,span=1+int(cfg.eval_ctx)) for i,d in enumerate(PILE_DOMAINS)]
    if int(getattr(cfg,'data_overlap_check',1)):                                 # issue #30: validation text must not appear in the training text
        try:
            ov_=[(d,)+overlap_check(stream_bytes(d,vseed_(i),500_000),(raw_tr_[i] if fresh_ else stream_bytes(d,tseed_(i),TRN_))) for i,d in enumerate(PILE_DOMAINS)]
            print("  [data] train/validation overlap (128-byte training chunks also found in the validation text): "
                  +", ".join(f"{d[:10]} {h_}/{n_}" for d,h_,n_ in ov_)+f" | total {sum(x_[1] for x_ in ov_)} of {sum(x_[2] for x_ in ov_)} ({'NONE' if not sum(x_[1] for x_ in ov_) else ('boilerplate level' if sum(x_[1] for x_ in ov_)<=0.001*max(1,sum(x_[2] for x_ in ov_)) else 'LEAK: validation text is in the training text')})",flush=True)
        except Exception as e_: _fail('runner:overlap'); print(f"  [data] train/validation overlap check unavailable ({type(e_).__name__}: {e_})")
    NDOM=len(PILE_DOMAINS); schedule=[d for _ in range(max(1,cfg.n_passes)) for d in range(NDOM)]     # REPEATED PASSES over all domains
    if cfg.pass_shuffle:
        _rp=np.random.RandomState(cfg.seed*7919+13); schedule=list(range(NDOM))+[int(d) for _ in range(max(1,cfg.n_passes)-1) for d in _rp.permutation(NDOM)]
        print(f"  [pass shuffle] visit order: "+' '.join(str(d) for d in schedule))
    _t2s=model.__dict__.setdefault('_t2s',{})                      # LABEL-FREE: true domain -> the slot its text was routed to (evaluation only)
    _S=(lambda q: _t2s.get(q,q)) if cfg.infer_ctx else (lambda q: q)
    bpc_m=[[None]*NDOM for _ in range(len(schedule))]; post=[None]*NDOM; prior_buffer=[]; fz_tok={}
    last_visit={}; post_hist={}; pre_visit={d:{} for d in range(NDOM)}; pass_log=[]
    locality_log=[]; prior_imp=[np.zeros(b.mlp_hidden) for b in model.blocks]
    model._orth=None; orth_log={}
    alt_m={}
    pre_lock=post_lock=None
    zeroshot=[None]*len(PILE_DOMAINS)
    _src=_StreamSource(schedule,train_of,cfg,NDOM) if cfg.stream else None; _pend=None; _seen_true=set()   # NO VISIT BOUNDARIES
    for vi,t in enumerate(schedule):                     # v = visit (position in the stream), t = domain (head) being trained
        pass_i=vi//NDOM; seen_d=sorted(set(schedule[:vi+1]))
        if cfg.stream:                                                         # NO VISIT BOUNDARIES: the slot was decided by the detector
            _eo=_src.done
            if _eo: print(f"  [stream] the stream has ended ({vi} detected visits for a schedule of {len(schedule)}): evaluation only"); ts=shist_[-1] if shist_ else 0; _b0=[]
            elif _pend is None: ts,_b0=infer_slot(model,cfg,_src,_src.true_dom,vi)
            else: ts,_b0=_pend
            if not _eo: t=_src.true_dom; _seen_true.add(t); seen_d=sorted(_seen_true|set(schedule[:vi+1]))   # bookkeeping (evaluation only) follows the true text
            _gv=_StreamGet(_src,model,cfg,ts,vi,_b0)
        else:
            _eo=False
            ts,_b0=infer_slot(model,cfg,train_of(t,pass_i),t,vi) if cfg.infer_ctx else (t,[])      # LABEL-FREE: the slot this visit trains
            _gv=_ReplayGet(train_of(t,pass_i),_b0,cfg.infer_batches) if cfg.infer_ctx else train_of(t,pass_i)
        _tsp_=_t2s.get(t) if cfg.infer_ctx else None                        # FIX (review): the slot before this visit's routing (for 'before relearning')
        if cfg.infer_ctx and not _eo: _t2s[t]=ts
        shist_=model.__dict__.setdefault('_slot_hist',[]) if cfg.infer_ctx else None
        olds=[]
        for d_ in reversed(shist_ if cfg.infer_ctx else schedule[:vi]):
            if d_!=ts and d_ not in olds: olds.append(d_)
        olds=olds[::-1]                                  # least -> most recently visited (single pass: range(t))
        if cfg.dream_lesion: olds=[d_ for d_ in olds if d_ not in model.__dict__.get('_lesion',set())]   # LESION TEST: no dreams for lesioned slots
        model.__dict__['_self_slot']=None
        if cfg.self_prot_w>0 and ts in (shist_ if cfg.infer_ctx else schedule[:vi]) and ts in model.__dict__.get('_bank',{}) and ts not in olds:
            olds.append(ts); model.__dict__['_self_slot']=ts               # cg15b SELF-PROTECTION: a revisited slot keeps (part of) its own dreams
            print(f"  [self protection] revisited slot {ts}: its own dream bank stays protected at share x{cfg.self_prot_w:g}")
        if (cfg.self_dream_nreal and cfg.real_w_pass and float(cfg.real_w_pass[min(pass_i,len(cfg.real_w_pass)-1)])==0.0
                and ts in (shist_ if cfg.infer_ctx else schedule[:vi]) and ts not in olds): olds.append(ts)   # cg26i ORDER-SAFE: slots seen, not domain ids   # NO-REAL-TEXT PASS: the visited domain is protected by its OWN dreams too
        object.__setattr__(model,'_olds',olds); object.__setattr__(model,'_visit',vi); model.__dict__['_cur_dom']=int(t)
        _sinf_=bool(int(getattr(cfg,'inf_sleep_inferred',0))) and bool(cfg.infer_ctx)
        if _sinf_: model.__dict__['_cur_dom']=int(ts); model.__dict__.setdefault('_s2t',{})[int(ts)]=int(t)   # cg26i SLEEP KEYED BY THE INFERRED DOMAIN: the store records the routed slot
        if cfg.n_passes>1 and cfg.data_free and not cfg.topic_mode and not _eo: rebuild_old_cov(model, olds)   # protect OTHER domains only
        if vi>0 and t in pre_visit:                                                       # before relearning (repeat visits)
            pre_visit[t][pass_i]=bpc_m[vi-1][t] if (bpc_m[vi-1][t] is not None or t not in schedule[:vi]) else bpc(model,vals[t],(_tsp_ if _tsp_ is not None else _S(t)))
        _pf9=__import__('sys').modules.get('platypus.profiling')
        if _pf9 is not None and vi>0: _pf9._ft_report(f"visit {vi-1} and its sleep")
        print(f"\n=== Phase {vi+1}: {PILE_DOMAINS[t]}"+(f" (pass {pass_i+1}/{cfg.n_passes})" if cfg.n_passes>1 else "")+f" === [elapsed {(time.time()-_RUN_T0)/60:.1f} min]")
        if cfg.fwt_zeroshot and vi>0 and t not in schedule[:vi] and cfg.scratch_domain is None:              # before ANY training on domain t
            zeroshot[t]=bpc_inferred(model,vals[t],t,t,'bytes',cfg)[0]
            print(f"  [forward transfer] {PILE_DOMAINS[t]} before training (earlier heads, inferred): {zeroshot[t]:.3f} BPC")
        model._curve_val=None
        # PROACTIVE difficulty-based widening: before training a HARD domain, open up
        # the shared manifold proportional to how much harder it is than the easiest
        # domain (measured difficulty). ArXiv (2.66, hardest) gets the most room; easy
        # domains get none. New units unfrozen (for this domain), old units protected.
        easiest=min(DOMAIN_DIFFICULTY.values())
        # every domain gets a minimum fresh block (so later domains write to NEW units,
        # not earlier domains') + extra proportional to difficulty (hard domains more)
        extra=cfg.min_fresh + int((DOMAIN_DIFFICULTY.get(PILE_DOMAINS[t],easiest)-easiest)*cfg.widen_per_difficulty)
        added=0
        while added<extra and model.blocks[0].d_state < cfg.d_state_max and not cfg.reserve_mult:
            model.widen_all(cfg.widen_chunk); added+=cfg.widen_chunk
        if added>0:
            opt=_new_opt(model,cfg)
            print(f"  [proactive widen] {PILE_DOMAINS[t]} difficulty={DOMAIN_DIFFICULTY[PILE_DOMAINS[t]]:.2f} -> +{added} units (d_state={model.blocks[0].d_state})")
        train_domain._prev_bpc=None; train_domain._plateau_count=0   # reset per domain
        if cfg.log_locality:
            W0=[(b.mlp_in.weight.detach().cpu().numpy().copy(), b.mlp_out.weight.detach().cpu().numpy().copy()) for b in model.blocks]
        model.__dict__['_first_visit']=(model._slot_log[-1][3]=='new') if cfg.infer_ctx else (t not in schedule[:vi])
        if (cfg.imp_revisit or cfg.imp_combine=='max') and not _eo and revisit_begin(model,cfg,ts) and cfg.imp_revisit:
            print(f"  [revisit gates] {PILE_DOMAINS[t]}'s own earlier importance taken out for this visit")
        if not cfg.joint_train and not _eo:
            model.__dict__['_s0_train']=None
            _pb=None
            try:
                model.__dict__['_pass_lr']=float(cfg.pass_lr[min(pass_i,len(cfg.pass_lr)-1)]) if cfg.pass_lr else 1.0
                model.__dict__['_dwp']=float(cfg.dream_w_pass[min(pass_i,len(cfg.dream_w_pass)-1)]) if cfg.dream_w_pass else 1.0   # cg14
                if not olds and float(cfg.pass_lr_noold)>0: model.__dict__['_pass_lr']=float(cfg.pass_lr_noold)   # cg13: visits with no old domain
                model.__dict__['_real_w']=float(cfg.real_w_pass[min(pass_i,len(cfg.real_w_pass)-1)]) if cfg.real_w_pass else 1.0
                model.__dict__['_pass_i']=int(pass_i)
                _prb=probe_pre(model,cfg,olds,t,vals) if (vi in tuple(cfg.probe_visits) and olds and cfg.dream_bank and cfg.bank_targets) else None
                if cfg.cd_teacher=='domain' and float(model.__dict__.get('_real_w',1.0))==0.0 and ts in model.__dict__.get('_dom_P',{}):
                    dv_=model.embed.weight.device; model.__dict__['_cd_P']={k_:v_.to(dv_) for k_,v_ in model.__dict__['_dom_P'][ts].items()}
                    print(f"  [teacher] slot {ts}: the conduit teacher is the model saved at the end of this slot's last visit")
                _les_=bool(cfg.trace_lesion) and pass_i>=int(cfg.trace_lesion_pass) and int(t) in tuple(int(z_) for z_ in cfg.trace_lesion)
                if _les_: print(f"  [trace lesion] {_lab(cfg,t)}: visit SKIPPED (no traces, no replay; its old dreams stay)")
                opt=(opt if _les_ else train_domain(model, None, meta_opt, opt, _gv, ts, prior_buffer, cfg))
                if int(cfg.pu_cons_steps)>0 and not _les_ and float(model.__dict__.get('_real_w',1.0))==0.0:   # cg19 CONSOLIDATION PHASE
                    _sd_=set(int(z_) for z_ in schedule[:vi]); pu_consolidate(model,cfg,t,ts,vals,[(d_,_S(d_)) for d_ in range(NDOM) if d_!=t and d_ in _sd_],_S)
                if cfg.inf_on and not _les_ and _sinf_:                         # cg26i SLEEP KEYED BY THE INFERRED DOMAIN: sleep sees slots only - the current slot, the other slots
                    s2t_=model.__dict__['_s2t']; sl_=[int(z_) for z_ in dict.fromkeys(shist_) if int(z_)!=int(ts) and int(z_) in s2t_]   #   visited, identity slot map; held-out
                    vs_={s0_:vals[s2t_[s0_]] for s0_ in list(sl_)+[int(ts)]}                    #   text for the brakes is that of the text last routed to each slot
                    print(f"  [inference SLEEP KEYED BY SLOT] slot {int(ts)} (routed text: {PILE_DOMAINS[t][:10]}), other slots {sl_}",flush=True)
                    inf_sleep(model,cfg,int(ts),int(ts),vs_,[(s0_,s0_) for s0_ in sl_],lambda q_: q_)
                elif cfg.inf_on and not _les_: inf_sleep(model,cfg,t,ts,vals,[(d_,_S(d_)) for d_ in range(NDOM) if d_!=t and d_ in set(int(z_) for z_ in schedule[:vi])],_S)   # cg20
                if cfg.cd_teacher=='domain':                                    # PER-DOMAIN TEACHER: save the model as it was at this visit's end
                    model.__dict__.setdefault('_dom_P',{})[ts]={k_:(p_.detach().to('cpu',copy=True) if p_.is_cuda else p_.detach().clone()) for k_,p_ in model.named_parameters()}
                if _prb is not None: probe_post(model,cfg,_prb,vi,vals)
            finally:
                if _pb is not None: cfg.batch_size,cfg.steps_per_task,cfg.dream_batch,cfg.lr,cfg.dream_early_steps=_pb
            if cfg.infer_ctx: _les_ or slot_ref(model,cfg,(_gv.ref_batches or _gv.last) if cfg.stream else _gv.last,ts); shist_.append(ts)
            if cfg.fact_spec:                                                                                # UPDATE-OR-PROTECT TEST: every fact domain's slot
                fd_=sorted({int(e[0]) for e in cfg.fact_spec}); fact_eval(model,cfg,vi,[(_lab(cfg,d_),_t2s[d_]) for d_ in fd_ if d_ in _t2s])
            if cfg.dream_lesion and pass_i>=int(cfg.dream_lesion_pass) and t in tuple(cfg.dream_lesion):   # LESION TEST: after this visit the domain's
                model.__dict__.setdefault('_lesion',set()).add(ts)                                          #   bank is never dreamed again
                print(f"  [lesion] {_lab(cfg,t)} (slot {ts}): its dream bank is switched off for the rest of the run")
            if cfg.stream:
                _pend=_gv.pending
                nb_=[b_ for b_ in _src.bounds if _gv.log and abs(b_-_gv.log[-1][0])<=4*max(1,int(cfg.infer_batches))]
                tr_=model.__dict__.pop('_strace',None)
                if tr_:
                    for b_ in [b_ for b_ in _src.bounds if tr_[0][0]<=b_<=tr_[-1][0]+1]:
                        print(f"  [stream trace] true boundary at batch {b_}: "+' '.join(f"{p_}:{l_:.2f}/{m_:.2f}/{sd_:.2f}/{z_:.1f}/{c_:.1f}" for p_,l_,m_,sd_,z_,c_ in tr_ if b_-4<=p_<=b_+8))
                if cfg.stream_rep: print(f"  [stream rep] visit {vi}: alarm z {_gv.rz:.1f} | largest in-visit z without alarm {_gv.rzmax:.1f} (threshold {cfg.stream_rep_z})")
                print(f"  [stream] visit {vi}: {len(_gv.log)} alarm(s) at stream batches "+' '.join(f'{p_} ({l_:.2f} > {th_:.2f} bits -> slot {sl_})' for p_,l_,th_,sl_ in _gv.log)
                      +(f" | switch {_gv.log[-1][0]-nb_[0]:+d} batches from the true boundary" if (_pend is not None and nb_) else (' | no switch' if _pend is None else ' | switch NOT at a true boundary')))
            if cfg.slot_mem and model.__dict__.get('_smem',{}).get(ts):
                print("  [slot memory] slot "+str(ts)+": mean |S_d| per head, per block: "+' '.join(f"{float(v_.norm(dim=(1,2)).mean()):.2f}" for v_ in model._smem[ts].values())+f" | slots with memory {len(model._smem)}")
            if cfg.ctx_carry and _CARRY['st']:
                print("  [context carry] carried memory at the visit end, mean |S| per row-head: "+' '.join(f"{float(v_.norm(dim=(1,2)).mean()):.2f}" for k_,v_ in _CARRY['st'].items() if not isinstance(k_,tuple)))
            model.__dict__['_s0_train']=None
            if int(cfg.lm_rank)>=0 and float(model.__dict__.get('_real_w',1.0))==0.0: lora_merge(model, cfg, olds, ts)   # REM-like DREAM MERGE
            if cfg.dream_bank and cfg.dream and vi<len(schedule)-1:
                bank_refresh(model, cfg, ts)
                if int(getattr(cfg,'infer_drift_fix',0)) and ts in model.__dict__.get('_bank',{}):   # cg20x: drift reference for routing
                    Xr_=model.__dict__['_bank'][ts][:16].long().to(model.embed.weight.device)
                    model.__dict__.setdefault('_bank_ref',{})[ts]=_slot_score(model,[Xr_[:,:-1]],[Xr_[:,1:]],ts)[0]   # the domain's dreams (v9h: not after the final visit - never read)
            if float(cfg.sem_share)>0 and cfg.sem_tgt=='student': sem_rescore(model, cfg, ts)
            if cfg.slot_mem and cfg.slot_mem_reenc>0: smem_reencode(model, cfg, ts)   # SLOT MEMORY RE-ENCODING
        object.__setattr__(model,'_live_dom',None)
        # add this domain's examples to the prior buffer (for future forgetting signal) - none when data_free
        for _ in range(0):
            bx,by=train_of(t,pass_i)(); prior_buffer.append((bx,by,t))
        if cfg.data_free and not cfg.topic_mode and not _eo: fold_domain_covariance(model, (ts) if cfg.n_passes>1 else None)                 # running input covariance -> 'old domains'
        for _ in range(cfg.replay_buffer_batches):          # separate, larger REPLAY store (bytes: stored compactly)
            bx,by=train_of(t,pass_i)(); model._replay_buffer.append((bx.to(torch.uint8),by.to(torch.uint8),t))
        # snapshot this domain's usage as "home" (polarization feature = deviation from it)
        with torch.no_grad():
            for b in model.blocks:
                b.unit_home.copy_(b.unit_usage); b.mlp_home.copy_(b.mlp_usage)
        if cfg.log_locality:
            dom_use=[b.mlp_usage.detach().cpu().numpy().copy() for b in model.blocks]
            if t>0:
                ms=[locality_metrics(W0[i][0],b.mlp_in.weight.detach().cpu().numpy(),W0[i][1],
                                     b.mlp_out.weight.detach().cpu().numpy(),dom_use[i],prior_imp[i],cfg.locality_k)
                    for i,b in enumerate(model.blocks)]
                avg={key:float(np.mean([m[key] for m in ms])) for key in ms[0]}
                locality_log.append(avg)
                print(f"  [locality] cluster={avg['cluster']:+.3f} damage_locality={avg['damage_locality']:+.3f} "
                      f"neigh_R2_gain={avg['neigh_r2_gain']:+.3f} own_neigh_corr={avg['own_neigh_corr']:+.2f}")
            for i in range(len(model.blocks)):
                u=dom_use[i]; prior_imp[i]=prior_imp[i]+u/(u.mean()+1e-8)
        if cfg.use_write_threshold and not cfg.topic_mode and not _eo: update_importance(model, cfg)
        if (cfg.imp_revisit or cfg.imp_combine=='max') and not _eo: revisit_end(model,cfg,ts)
        if t==cfg.lock_after_domains-1 and cfg.route_mode=='none':
            model.lock_distinctness()   # no routing: lock is a no-op for computation; skip switch-cost evals
            pre_lock=post_lock=[bpc_m_q for bpc_m_q in [None]*(t+1)]
        elif t==cfg.lock_after_domains-1:
            # ROUTING-SWITCH COST: the lock switches routing from uniform to distinctness-based.
            # Score every trained domain just before and just after it. Eval passes nudge the
            # (not-yet-locked) distinctness scale, so snapshot/restore it: training is unchanged.
            fs=model.feat_scale.clone()
            pre_lock=[bpc(model,vals[q],q) for q in range(t+1)]
            model.feat_scale.copy_(fs)
            model.lock_distinctness()   # freeze reference (= prose cluster over first domains)
            if cfg.route_writes: calibrate_affinity(model, cfg, t+1)
            post_lock=[bpc(model,vals[q],q) for q in range(t+1)]
            for q in range(t+1):
                print(f"  [routing switch] {PILE_DOMAINS[q][:12]:12} pre-lock {pre_lock[q]:.3f} -> post-lock "
                      f"{post_lock[q]:.3f} (switch cost {post_lock[q]-pre_lock[q]:+.3f})")
            print(f"  [distinctness reference LOCKED after {cfg.lock_after_domains} domains - routing now stable, reference=prose cluster]")
        with torch.no_grad():
            for b in model.blocks:
                b.unit_usage.zero_(); b.mlp_usage.zero_()
        end_pass=(vi%NDOM==NDOM-1)
        for prev in (seen_d if (end_pass or not cfg.lean or cfg.fz_diag or cfg.sim_diag) else [t]): bpc_m[vi][prev]=bpc(model,vals[prev],_S(prev))   # lean: all domains at pass ends
        if int(getattr(cfg,'grad_eval',0))>=2 and not (cfg.n_passes>1 and end_pass):   # cg27a GRADUATION SERIES: reliance on the private parts per domain after every visit
            from platypus.inference.residual import grad_readout                  #   (all on vs all off; the full breakdown at pass ends)
            grad_readout(model,cfg,vals,sorted(set(seen_d)),_S,f"visit {vi+1}",lite=True)
        if cfg.head_select_eval:                         # label-free: every seen domain at pass ends (single pass: every visit, or with
            for prev in (seen_d if ((cfg.n_passes==1 and not cfg.diag_light) or end_pass) else [t]):     # diag_light current + end only)
                for mode,r_ in bpc_inferred_modes(model,vals[prev],_S(prev),len(model._slots) if cfg.infer_ctx else len(seen_d),tuple(cfg.head_select_eval),cfg).items():
                    alt_m.setdefault(mode,{})[(vi,prev)]=r_
        if int(getattr(cfg,'mix_eval',0)) and end_pass:   # MIXED-TEXT READOUT: several slots active at once (readout only)
            try: mix_text_eval(model,vals,seen_d,_S,cfg,tag=f' | visit {vi+1}')
            except Exception as e_: _fail('runner:368'); print(f"  [mixed text] skipped: {repr(e_)[:160]}")
        post[t]=bpc_m[vi][t]; last_visit[t]=vi; post_hist.setdefault(pass_i,{})[t]=post[t]
        print(f"  post-train BPC: {post[t]:.3f}  (width {model.blocks[0].d_state})"
              +(f" | before this visit {pre_visit[t][pass_i]:.3f}, after previous visit {post_hist[pass_i-1][t]:.3f}" if (pass_i>0 and pass_i in pre_visit.get(t,{}) and t in post_hist.get(pass_i-1,{})) else ""))
        cl=(getattr(model,'_cross_log',None) or {}).get(t)
        if cl: print('  [cross] mean |gain| on each earlier domain\'s specific shunt: '+' '.join(f'{PILE_DOMAINS[j][:6]} {g:.3f}' for j,g in zip(sorted(model._spec)[:len(cl)],cl)))
        if getattr(model,'_spec',None) and t in model._spec:
            nz=sum(int((v!=0).sum()) for v in model._spec[t].values()); print(f'  [specific shunt] {nz/1e6:.2f}M weights kept for this domain')
        dsn=model.__dict__.pop('_dreamstat',None)
        if dsn: dsn={d:[(float(a) if torch.is_tensor(a) else a) for a in v] for d,v in dsn.items()}; dsn={d:v[:4]+[int(round(v[4]))] for d,v in dsn.items()}
        if dsn:
            dream_log[vi]={d:w_[0]/max(w_[1],1) for d,w_ in dsn.items()}
            rbl=[r for r in (getattr(model,'_rb_log',None) or []) if r[0]==t]
            print('  [dream outliers] batches with dream KL > 1 / robust mean KL: '+' '.join(f'{_lab(cfg,d)} {v[4]}/{v[1]} {v[3]/max(v[1],1):.4f}' for d,v in sorted(dsn.items())))
            for d,(kw,txt,lmin,pr) in sorted((model.__dict__.pop('_dworst',None) or {}).items()):
                if kw>1.0:
                    print(f'  [worst dream] {_lab(cfg,d)} batch KL {kw:.1f}, live min log-prob {lmin:.0f}: '+repr(txt[:100]))
                    if pr: print('      max |.| per block (input, gate pre-act, SSM pred, gated, MLP act): '
                                 +' | '.join(f'{a:.0f} {g:.0f} {q:.0f} {z:.0f} {m_:.0f}' for a,g,q,z,m_ in pr[:-1])
                                 +f' || final features {pr[-1][0]:.0f}, logit range {pr[-1][1]:.0f}')
            print('  [dream KL] mean KL to main weights on each old domain\'s DREAMS (weight %.1f): '%cfg.dream_weight
                  +' '.join(f'{_lab(cfg,d)} {w_:.4f}' for d,w_ in sorted(dream_log[vi].items())))
            if cfg.share_mode=='kl':                                         # cg10: next visits' dream shares follow the smoothed dream KL
                SG_=model.__dict__.setdefault('_share_sig',{})
                for d,w_ in dream_log[vi].items():
                    if d==model.__dict__.get('_self_slot'): continue           # cg15b: self-KL includes this visit's learning, not drift
                    SG_[d]=float(w_) if d not in SG_ else 0.5*SG_[d]+0.5*float(w_)
        ks=model.__dict__.pop('_klstat',None)
        if ks: ks={d:[(float(a) if torch.is_tensor(a) else a) for a in v] for d,v in ks.items()}
        if ks:
            kl_log[vi]={d:(w_[0]/max(w_[2],1),w_[1]/max(w_[2],1)) for d,w_ in ks.items()}
            print('  [shunt KL] mean KL to main weights / mean weight per old domain: '+' '.join(f'{_lab(cfg,d)} {a:.4f}/{w:.2f}' for d,(a,w) in kl_log[vi].items()))
        if getattr(model,'_reserve',None):
            ev=model._recruit_log.get(t,[])
            print(f"  [reserve] {PILE_DOMAINS[t]}: activated state {sum(e[1] for e in ev)} / MLP {sum(e[2] for e in ev)} units in {len(ev)} "
                  f"block-events | active per block state "+",".join(str(b.d_state) for b in model.blocks)
                  +" MLP "+",".join(str(b.mlp_hidden) for b in model.blocks)
                  +f" | reserve left state {model._reserve[0].Wk.shape[0]} MLP {model._reserve[0].min.shape[0]} (block 0)")
        # update running difficulty average (for the difficulty-relative widening trigger)
        seen=[post[k] for k in dict.fromkeys(int(z_) for z_ in schedule[:vi+1]) if post[k] is not None]   # cg26i ORDER-SAFE: the domains seen so far, not 0..t
        train_domain._avg_difficulty=float(np.mean(seen))
        if model._orth is not None and t>0:
            orth_log[t]=orth_summary(model._orth.log, t, cfg.orth_energies)
            if orth_log[t] is not None: print_orth(orth_log[t], PILE_DOMAINS[t], cfg.orth_energies)
        if cfg.topic_diag:
            tr=model._topics.report(vals,t); topic_log[t]=tr
            top=[" ".join(f"t{k}:{int(100*v/max(sum(row),1))}%" for k,v in sorted(enumerate(row),key=lambda z:-z[1])[:3] if v>0) for row in tr['M']]
            print(f"  [topics] {tr['used']} in use | purity {tr['purity']:.2f} NMI {tr['nmi']:.2f} | confidence "
                  +" ".join(f"{c:.2f}" for c in tr['conf'])+" | kept own topic since trained: "
                  +(" ".join(f"{PILE_DOMAINS[d][:6]} {v:.2f}" for d,v in tr['stab'].items()) or "-")
                  +" | head-coverage "+" ".join(f"{v:.2f}" for v in tr['cover'].values()))
            for d,tt in enumerate(top): print(f"     {PILE_DOMAINS[d][:10]:10} -> {tt}")
        rcs=model.__dict__.pop('_recon_stat',None)
        if rcs: model.__dict__.setdefault('_recon_log',{})[t]={li:v[0]/max(v[1],1e-12) for li,v in rcs.items()}
        rsh=model.__dict__.pop('_recon_share',None)
        if rcs: print('  [reconsolidation] mean squared output error on dreams before each correction, relative to the output energy: '
                      +' '.join(f'b{li} {v[0]/max(v[1],1e-12):.4f}' for li,v in sorted(rcs.items()))
                      +(f' | dream tokens corrected (one-sided) {rsh[0]/max(rsh[1],1):.2f}' if rsh else ''))
        ba=model.__dict__.pop('_battr_acc',None)
        if ba:
            battr_log[t]={g:(v[0]/v[2],v[1]/v[2],v[3]/v[2]) for g,v in ba.items()}
            order=[g for g in ['emb']+[f'b{i}' for i in range(len(model.blocks))]+['all'] if g in battr_log[t]]
            print('  [block attribution] this domain\'s learning, each part alone on the main weights (data-free): benefit (current-loss '
                  'drop) / change (old heads\' dream KL) / SIGNED harm (old heads\' loss change on samples from their byte-pair statistics): '
                  +' | '.join(f"{g} {battr_log[t][g][0]:+.4f}/{battr_log[t][g][1]:.4f}/{battr_log[t][g][2]:+.4f}" for g in order)
                  +(''))
        if cfg.n_passes>1 and end_pass:                 # PASS SUMMARY
            fin_=[bpc_m[vi][d] for d in range(NDOM)]; po_=[post_hist.get(pass_i,{}).get(d,post[d] if post[d] is not None else fin_[d]) for d in range(NDOM)]
            fg_=[fin_[d]-po_[d] for d in range(NDOM) if d!=t]
            al_={m:float(np.mean([alt_m[m][(vi,d)][0] for d in range(NDOM)])) for m in cfg.head_select_eval}
            pass_log.append({'pass':pass_i,'final':fin_,'post':po_,'bpc':float(np.mean(fin_)),'forget':float(np.mean(fg_)),'alt':al_,
                             'pre':[pre_visit.get(d,{}).get(pass_i) for d in range(NDOM)]})
            print(f"\n  ===== PASS {pass_i+1} SUMMARY: mean BPC {np.mean(fin_):.3f} | mean forgetting since each domain's visit {np.mean(fg_):+.3f} | label-free "
                  +" ".join(f"{m} {x_:.3f}" for m,x_ in al_.items())+" =====")
            if 'heads' in al_:                                                     # issue #23: inferred - given, with its sign, and the spread over domains
                dl_=[alt_m['heads'][(vi,d)][0]-fin_[d] for d in range(NDOM)]
                print(f"    [label-free] INFERRED - GIVEN: mean {np.mean(dl_):+.4f} (range {min(dl_):+.4f} to {max(dl_):+.4f}) | per domain "
                      +" ".join(f"{PILE_DOMAINS[d][:6]} {dl_[d]:+.4f}" for d in range(NDOM)),flush=True)
            if int(getattr(cfg,'grad_eval',0)):                                  # cg27a GRADUATION at each pass end
                from platypus.inference.residual import grad_readout
                grad_readout(model,cfg,vals,sorted(set(seen_d)),_S,f"PASS {pass_i+1}")
            if int(getattr(cfg,'inf_resid',0)) and cfg.inf_on:
                from platypus.inference.residual import resid_readout
                resid_readout(model,cfg,_inf_store(model),[stream_bytes(d,vseed_(i),500_000) for i,d in enumerate(PILE_DOMAINS)],sorted(set(seen_d)),_S,f"PASS {pass_i+1}")
            CLp_=[r_ for r_ in model.__dict__.get('_pucons_log',[])]
            if CLp_:
                print(f"    [consolidation] so far: {len(CLp_)} phases | median gap recovered {100*np.median([r_[5] for r_ in CLp_]):.0f}% | held-out with private units "
                      f"{np.mean([r_[2]-r_[1] for r_ in CLp_]):+.4f} per phase | old domains {np.nanmean([r_[7]-r_[6] for r_ in CLp_]):+.4f} per phase | "
                      f"retired {sum(r_[8] for r_ in CLp_)}, evicted {sum(r_[9] for r_ in CLp_)}")
            if cfg.trace_lesion and pass_i>=int(cfg.trace_lesion_pass):
                LS_=[int(z_) for z_ in cfg.trace_lesion]; pv_=pass_i-1
                ch_=[(d,fin_[d]-post_hist.get(pv_,{}).get(d,float('nan'))) for d in LS_]
                wt_=[pre_visit.get(d,{}).get(pass_i)-post_hist.get(pv_,{}).get(d,float('nan')) for d in range(NDOM) if d not in LS_ and pre_visit.get(d,{}).get(pass_i) is not None and d in post_hist.get(pv_,{})]
                print(f"    [trace lesion] pass {pass_i+1}, no visit: "+" | ".join(f"{_lab(cfg,d)} {post_hist.get(pv_,{}).get(d,float('nan')):.3f} -> {fin_[d]:.3f} ({c_:+.3f})" for d,c_ in ch_)
                      +f" || the other domains while WAITING for their visit: mean {np.mean(wt_) if wt_ else float('nan'):+.3f} (max {max(wt_) if wt_ else float('nan'):+.3f})")
            for d in range(NDOM):
                pv=pre_visit.get(d,{}).get(pass_i)
                print(f"    {PILE_DOMAINS[d][:18]:18} end of pass {fin_[d]:.3f} | post-train {po_[d]:.3f}"
                      +(f" (pass {pass_i}: {post_hist[pass_i-1][d]:.3f}, {po_[d]-post_hist[pass_i-1][d]:+.3f})" if (pass_i>0 and d in post_hist.get(pass_i-1,{})) else "")
                      +(f" | before relearning {pv:.3f} (forgotten since last visit {pv-post_hist[pass_i-1][d]:+.3f})" if (pv is not None and d in post_hist.get(pass_i-1,{})) else "")
                      +f" | forgetting {fin_[d]-po_[d]:+.3f}")
    alt={}
    for mode,M in alt_m.items():
        L_=len(PILE_DOMAINS)-1; VL=len(schedule)-1; lastd=schedule[-1]
        fin=[M[(VL,q)][0] for q in range(L_+1)]; pst=[M.get((last_visit.get(q,VL),q),M[(VL,q)])[0] for q in range(L_+1)]
        acc=[M[(VL,q)][1] for q in range(L_+1)]
        alt[mode]={'final':fin,'post':pst,'bpc':float(np.mean(fin)),'forget':float(np.mean([fin[q]-pst[q] for q in range(L_+1) if q!=lastd])),'acc':acc}
        print(f"  [inferred heads: {mode}] final BPC {alt[mode]['bpc']:.3f} forget {alt[mode]['forget']:+.3f} | "
              f"domain inferred correctly (token share, final): "+" ".join(f"{PILE_DOMAINS[q][:6]} {acc[q]:.2f}" for q in range(L_+1)))
    print("\n=== REAL FORGETTING ===")
    fg=[]
    for p in range(len(PILE_DOMAINS)):
        final=bpc_m[-1][p]; post[p]=final if post[p] is None else post[p]; f=final-post[p]
        if p!=schedule[-1]: fg.append(f)
        print(f"  {PILE_DOMAINS[p][:12]:12} | post {post[p]:.3f} | final {final:.3f} | forget {f:+.3f}")
    mf=float(np.mean(fg)); mb=float(np.mean([bpc_m[-1][p] for p in range(len(PILE_DOMAINS))]))
    if cfg.inf_on: inf_eval(model,cfg,_inf_store(model),"END OF RUN"); inf_final_bpc(model,cfg,vals,_S)
    if int(getattr(cfg,'gist_test',0)):                                          # GIST TEST: does a learned gist code carry information the trunk lacks?
        from platypus.inference.gisttest import gist_test
        gist_test(model,cfg,[stream_bytes(d,vseed_(i),500_000) for i,d in enumerate(PILE_DOMAINS)],sorted(set(schedule)),_S)
    if int(getattr(cfg,'inf_resid',0)) and cfg.inf_on:                          # cg27a MEMORY RESIDUAL (no gist): what the trunk still cannot predict of what the store holds
        from platypus.inference.residual import resid_readout
        resid_readout(model,cfg,_inf_store(model),[stream_bytes(d,vseed_(i),500_000) for i,d in enumerate(PILE_DOMAINS)],sorted(set(schedule)),_S,"END OF RUN")
    if int(getattr(cfg,'inf_selcmp',0)) and cfg.inf_on:                          # SELECTOR COMPARISON: trained JEPA selector against CCA / SFA / TF-IDF
        from platypus.inference.selcmp import selector_compare
        selector_compare(model,cfg,_inf_store(model),_S,"END OF RUN")
    if int(getattr(cfg,'inf_rdr_judge',0)):                                      # cg27a RELATION JUDGE ON PROSE (Re-DocRED pairs, trunk alone)
        from platypus.inference.residual import pair_judge_readout
        pair_judge_readout(model,cfg,_S,"END OF RUN")
    if int(getattr(cfg,'grad_eval',0)):                                          # cg27a GRADUATION: private parts off, per slot
        from platypus.inference.residual import grad_readout
        grad_readout(model,cfg,vals,sorted(set(schedule)),_S,"END OF RUN")
    _pf9=__import__('sys').modules.get('platypus.profiling')
    if _pf9 is not None: _pf9._ft_total()
    print(f"\nMEAN FORGETTING: {mf:+.3f} | MEAN FINAL BPC: {mb:.3f}")
    _F9=__import__('platypus.common',fromlist=['_FAILS'])._FAILS                  # issue #28: every print-and-continue handler that fired during the run
    if _F9:
        cnt9_={}
        for w_,t_,m_ in _F9: cnt9_.setdefault((w_,t_),[0,m_])[0]+=1
        print(f"FAILURES: {len(_F9)} ({len(cnt9_)} distinct) - "+" | ".join(f"{w_} {t_} x{c_[0]}: {c_[1][:80]}" for (w_,t_),c_ in sorted(cnt9_.items(),key=lambda x_: -x_[1][0])[:20]),flush=True)
    else: print("FAILURES: none",flush=True)
    if 'heads' in alt:                                                            # HEADLINE (issue #23): the slot inferred from the text itself
        dl9_=[alt['heads']['final'][q]-bpc_m[-1][q] for q in range(len(PILE_DOMAINS))]
        print(f"  [label-free] INFERRED - GIVEN at the end: mean {np.mean(dl9_):+.4f} (range {min(dl9_):+.4f} to {max(dl9_):+.4f}) | per domain "
              +" ".join(f"{PILE_DOMAINS[q][:6]} {dl9_[q]:+.4f}" for q in range(len(PILE_DOMAINS))),flush=True)
        print(f"HEADLINE (label-free: each test window's slot inferred from the text, heads posterior): FORGETTING {alt['heads']['forget']:+.3f} | FINAL BPC {alt['heads']['bpc']:.3f} "
              f"|| with the slot given (task-incremental): {mf:+.3f} / {mb:.3f}",flush=True)
    if cfg.recon_diag: (replay_diag if cfg.recon_mark else recon_fidelity)(model,cfg,vals,NDOM,_S)
    orc_=None
    if cfg.oracle_mem:                                                                # ORACLE LINEAR MEMORY
        trs_=[make_loader(stream_bytes(d,tseed_(i),max(400_000,int(cfg.oracle_batches*cfg.batch_size*(cfg.seq_len+1)*1.2))),cfg) for i,d in enumerate(PILE_DOMAINS)]
        orc_=oracle_final(model,cfg,vals,trs_,NDOM,bpc_m[-1],_S)
    interf=None; hc_=getattr(model,'_hconf_last',None)
    wall_run=time.perf_counter()-_t_run; tl_=getattr(model,'_timing_log',{})
    if tl_: print(f"RUN WALL-CLOCK: {wall_run/60:.1f} min in total (training {sum(v['wall'] for v in tl_.values())/60:.1f} min, of which dream sampling "
                  f"{sum(v['dream'] for v in tl_.values())/60:.1f} min; evaluation and diagnostics the rest) | {np.mean([v['ms'] for v in tl_.values()]):.0f} ms per regular step")
    vl_=getattr(model,'_vlog',None)
    if vl_: print("  [visit length] steps per visit: "+" ".join(f"{PILE_DOMAINS[d][:6]} {L_}" for d,(g_,L_,b_) in sorted(vl_.items()))
                  +f" | total {sum(L_ for g_,L_,b_ in vl_.values())+cfg.steps_per_task*(NDOM-len(vl_))} vs {cfg.steps_per_task*NDOM} budget")
    return {'_model':model,'_slot':_S,'_vals':vals,'_raw_vals':(lambda: [stream_bytes(d,vseed_(i),500_000) for i,d in enumerate(PILE_DOMAINS)]),'_schedule':list(schedule),'oracle':orc_,'vlen':vl_,'cov':getattr(model,'_cov_log',{}),'fsplit':getattr(model,'_fsplit_log',{}),'interf':interf,'hconf':(None if hc_ is None else (hc_[0],hc_[1].tolist())),'fdrift':getattr(model,'_fdrift_log',{}),
            'klpos':getattr(model,'_klpos_log',{}),'wall_run':wall_run,'forget':mf,'bpc':mb,'per_domain':[bpc_m[-1][p]-post[p] for p in range(len(PILE_DOMAINS)) if p!=schedule[-1] and post[p] is not None],'passes':pass_log,'domains':list(PILE_DOMAINS),
            'gut_after_wiki':(bpc_m[1][0]-post[0]) if (len(bpc_m)>1 and bpc_m[1][0] is not None) else None,'locality':locality_log,'orth':orth_log,
            'post':list(post),'alt':alt,'topics':(topic_log if cfg.topic_diag else {}),'kl':kl_log,'dream':dream_log,'fdiag':fd_log,'stale':getattr(model,'_stale_log',{}),'beta':getattr(model,'_beta_log',{}),'timing':getattr(model,'_timing_log',{}),'recon':getattr(model,'_recon_log',{}),'sleep':getattr(model,'_sleep_log',{}),'battr':battr_log,'bw_dyn':getattr(model,'_bw_dyn',None),'tbirths':getattr(model,'_tspawn_log',[]),'tcombine':getattr(model,'_tcombine_log',[]),'tmerges':getattr(model,'_tmerge_log',[]),'treuse':getattr(model,'_tswitch_log',[]),'rollbacks':getattr(model,'_rb_log',[]),'cross':{int(k):v for k,v in (getattr(model,'_cross_log',None) or {}).items()},'zeroshot':zeroshot,'curve':getattr(model,'_curve',{}),'width':model.blocks[0].d_state,'mlp_width':model.blocks[0].mlp_hidden,
            'recruit':getattr(model,'_recruit_log',{}),'merges':getattr(model,'_merge_log',[]),
            'final':[bpc_m[-1][q] for q in range(len(PILE_DOMAINS))],
            # decomposition (lock after domain L): for domains <=L, measure interference from the
            # post-lock score, so the one-off routing switch isn't counted as forgetting
            **({'switch_gut':post_lock[0]-pre_lock[0], 'wiki_on_gut':pre_lock[0]-post[0],
                'wiki_true_post':pre_lock[1] if len(pre_lock)>1 else float('nan'),
                'interf_forget':float(np.mean([bpc_m[-1][q]-(post_lock[q] if q<len(post_lock) else post[q])
                                              for q in range(len(PILE_DOMAINS)-1)]))}
               if (pre_lock is not None and pre_lock[0] is not None) else
               {'switch_gut':0.0,'wiki_on_gut':float('nan'),'wiki_true_post':float('nan'),'interf_forget':mf})}
