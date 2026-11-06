"""One visit of training (train_domain) and its step-level helpers."""
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
from platypus.common import (
    LOG2, _nslots,
)
from platypus.config import (
    PILE_DOMAINS,
)
from platypus.gating import (
    _RAMP,
)
from platypus.protection import (
    _KF, _STF, _kl_topk, aff_weights, guard_sense, head_center_, kf_build, kf_live_warmup, kf_main_on, kf_store, muon_apply, muon_prep, noise_diag,
    noise_report, proj_post, stiff_post, varpro_old_refit, varpro_refit,
)
from platypus.private import (
    _SLP, _XDG, priv,
)
from platypus.slots import (
    _ema_params, _rbg_add,
)
from platypus.layers import (
    _BPTT, _CARRY, _carry_off, _carry_on, _pair_on,
)
from platypus.capacity import (
    _lab, _pmap, recruit,
)
from platypus.sampler import (
    state_spectrum_diag, teacher_role,
)
from platypus.dream_bank import (
    _BYTE_CLS, _BYTE_NAMES, _pool_add, build_teachers, dream_cap, dream_pick, dream_refresh, dream_shares, fast_dream_visit, heads_logp, stack_heads,
    stream_batch, stream_generate,
)
from platypus.ngrams import (
    _dream_route_rows, adam_capture, adam_rebuild, ngram_lazy_step,
)
from platypus.meta import (
    _functional_buffer_loss, _virtual_gated_step, apply_update_gate, snapshot_threshold,
)
from platypus.importance import (
    _imp_bufs, apply_meta_gate, online_importance,
)
from platypus.data import (
    make_loader, stream_bytes,
)
from platypus.heads import (
    _tmark, head_stats_update,
)


_NOV={'m':None}

def borrow_align(model, cfg, olds, dom_id, x, y, n_rows=8):
    """BORROW ONLY IF IT HELPS: per old domain d, cos(grad of the borrowing KL on d's queued dreams, grad of the current domain's loss on the
    current batch) over the blocks and the current head; stores max(0, cos) as d's borrowing weight."""
    ps=[p_ for p_ in model.blocks.parameters() if p_.requires_grad]+list(model.heads[dom_id].parameters())
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    flat=lambda g: torch.cat([(gi if gi is not None else torch.zeros_like(p_)).flatten().float() for gi,p_ in zip(g,ps)])
    gc=flat(torch.autograd.grad(F.cross_entropy(model.heads[dom_id](model.features(x)).reshape(-1,256),y.reshape(-1)),ps,allow_unused=True))
    G=model.__dict__.get('_borrow_g')
    if G is None: G=model.__dict__['_borrow_g']=torch.zeros(_nslots(cfg),device=y.device)
    Pp=min(cfg.dream_stream_prompt,min(cfg.dream_len,cfg.seq_len)//2); Q=model._dsq; out={}
    for d in olds:
        q=Q.get(d) or []
        if not q: G[d]=0.0; continue
        xd=torch.stack([a for a,_ in q[:n_rows]]); tg=torch.stack([b for _,b in q[:n_rows]])
        lc=F.log_softmax(model.heads[dom_id](model.features(xd)).float()/cfg.shunt_T,-1)
        kb=(tg.exp()*(tg-lc)).sum(-1)[:,Pp:].mean()
        gb=flat(torch.autograd.grad(kb,ps,allow_unused=True))
        c=(gb@gc)/(gb.norm()*gc.norm()+1e-30); G[d]=c.clamp(min=0.0); out[d]=c
    model.feat_scale.copy_(fs); model.train(was)
    A=model.__dict__.setdefault('_borrow_acc',{})
    for d,c in out.items(): a_=A.setdefault(d,[0.0,0]); a_[0]+=c; a_[1]+=1

def dctl_state(model):
    return model.__dict__.setdefault('_dctl',{'v':0,'ref':{},'cal':{},'D':{},'debt':{},'u':1.0,'acc':0.0,'s0':None,'s':None,'rff':1.0,
                                               'c':{},'rema':1.0,'nd':0,'calib':False,'last':None,'ns':0})
def dctl_rate(model,cfg,step,olds):
    """Visit bookkeeping at step 0; returns the dream interval for this step (1 = dream now, step+2 = skip) - weight stays x1."""
    S=dctl_state(model)
    if S.get('vtok')!=model.__dict__.get('_dctl_vtok'):                  # first dream decision of a new visit
        S['vtok']=model.__dict__.get('_dctl_vtok')
        if S['last'] is not None:                                      # close the previous visit: calibration means, debts
            for d,(sm,n) in S['cal'].items():
                if n>0 and d not in S['ref']: S['ref'][d]=sm/n
            if not S['calib']:
                for d,Dv in S['D'].items():
                    if d in S['ref']: S['debt'][d]=S['debt'].get(d,0.0)+Dv-cfg.dctl_rel*S['ref'][d]
        S['v']+=1; S['cal']={}; S['D']={}; S['s0']=None; S['rff']=1.0; S['nd']=0; S['ns']=0; S['acc']=0.0
        S['calib']=any(d not in S['ref'] for d in olds) and not cfg.dctl_calwin>0 and cfg.dctl!='rate'; S['last']=list(olds); S['H']={}; S['vref']={}
    S['ns']+=1
    if S['calib']: S['nd']+=1; return 1                                 # calibration visit: dream every step
    r=min(1.0,max(cfg.dctl_rmin,(1.0)*S['u'])); S['rema']=0.98*S['rema']+0.02*r
    S['acc']+=r
    if S['acc']>=1.0 or S['ns']==1: S['acc']=max(0.0,S['acc']-1.0); S['nd']+=1; return 1
    return step+2
def guard_step(model,cfg,olds,dom_id,step):
    """EARLY DRIFT GUARD: sense (held-out bank KL) early in the visit, flag fast-rising old domains, set their row boost / Kalman weight."""
    Lv=int(cfg.steps_per_task); ev_=max(1,int(round(cfg.guard_every*Lv))); span_=int(round(cfg.guard_span*Lv)); dur_=max(1,int(round(cfg.guard_dur*Lv)))
    G=model.__dict__.setdefault('_guard',{})
    if step==0: G.clear(); G.update(base={},until={},kfw={},log=[],n=0,last={},atr={},na={})
    atr_=cfg.guard_mode=='atr'
    if step<=span_ and step%ev_==0:
        kl_=guard_sense(model,cfg,olds); G['n']+=1
        if step==0: G['base']=dict(kl_); G['last']=dict(kl_)
        else:
            for d,v in kl_.items():
                b_=G['base'].get(d,v)
                if atr_:                                                                    # AVERAGE TRUE RANGE of the held-out KL
                    tr_=abs(v-G['last'].get(d,v)); G['last'][d]=v; n_=G['na'].get(d,0)+1; G['na'][d]=n_
                    a_=2.0/(int(cfg.guard_atr_n)+1); E_=G.setdefault('ema',{}); e0_=E_.get(d,b_)
                    hit_=n_>=3 and v>e0_+cfg.guard_atr_k*G['atr'].get(d,tr_) and (v-b_)>0.5*cfg.guard_min   # breakout against the channel so far
                    G['atr'][d]=tr_ if n_==1 else (1-a_)*G['atr'][d]+a_*tr_; E_[d]=(1-a_)*e0_+a_*v
                else: hit_=v>max(cfg.guard_ratio*b_,b_+cfg.guard_min)
                if hit_:
                    if G['until'].get(d,-1)<=step: G['log'].append((step,d,round(v,4),round(b_,4)))
                    G['until'][d]=step+dur_
    act={d for d,u in G['until'].items() if u>step}
    bst_={d:float(cfg.guard_boost) for d in act} if (act and cfg.guard_boost!=1.0) else {}
    if atr_ and cfg.guard_atr_share and len(G['atr'])==len(olds) and len(olds)>1:              # shares follow volatility
        m_=float(np.mean(list(G['atr'].values())))
        if m_>0: bst_={d:max(bst_.get(d,1.0),float(np.clip(G['atr'][d]/m_,0.5,cfg.guard_boost))) for d in olds}
    model.__dict__['_guard_boost']=bst_ or None
    if atr_ and cfg.guard_atr_rows and G.get('atr'):                                          # ATR-CONTROLLED DREAM ROWS
        A_=float(np.mean(list(G['atr'].values()))); GA_=model.__dict__.setdefault('_atr_glob',[0.0,0])
        if step<=span_ and step%ev_==0 and G['na'] and min(G['na'].values())>=2: GA_[0]+=A_; GA_[1]+=1
        if GA_[1]>=3 and GA_[0]>0:
            R0_=int(cfg.dream_batch); R_=int(np.clip(round(R0_*A_/(GA_[0]/GA_[1])),max(1,round(R0_*cfg.guard_rows_min)),max(1,round(R0_*cfg.guard_rows_max))))
            model.__dict__['_sldb']=R0_; cfg.dream_batch=R_
            st_=G.setdefault('rows',[0,0]); st_[0]+=R_; st_[1]+=1
    if cfg.guard_kf!=1.0 and cfg.pmuon:
        new={d:float(cfg.guard_kf) for d in act}
        if new!=G['kfw']:
            G['kfw']=new; model.__dict__['_kf_wt']=new or None; kf_build(model,cfg,dom_id,quiet=True)
    if not G.get('rep') and (step==int(cfg.steps_per_task)-1 or step>span_+dur_):
        G['rep']=True
        print(f"    [drift guard] {G['n']} checks, {len(G['log'])} flags: "+(' '.join(f"s{s_} {_lab(cfg,d)} {v:.3f}>{b:.3f}" for s_,d,v,b in G['log']) or 'none')
              +(" | ATR "+' '.join(f"{_lab(cfg,d)} {a:.4f}" for d,a in sorted(G['atr'].items())) if G.get('atr') else "")
              +(f" | ATR rows: mean {G['rows'][0]/max(1,G['rows'][1]):.1f} per step so far" if G.get('rows') else ""))
def _dstep_obj(model,cfg,dom_id,olds,seed):
    """DIRECT STEP held-out objective J: CE of the new domain's head on held-out text rows + KL(stored top-k targets || old heads) on
    held-out bank rows (rows < dstep_hold of each old domain's bank, never drawn for training), prompt positions excluded."""
    xh,yh=model.__dict__['_dsH']; dev=xh.device
    rs_=np.random.RandomState(int(seed)%(2**31-1))
    it_=torch.as_tensor(rs_.choice(xh.shape[0],size=min(int(cfg.dstep_rows),xh.shape[0]),replace=False),device=dev)
    xt,yt=xh[it_],yh[it_]; nt=xt.shape[0]
    BK=model.__dict__.get('_bank',{}); TV=model.__dict__.get('_bank_tv',{}); ol=[d for d in olds if d in BK and d in TV]
    dm=sorted(int(ol[i]) for i in rs_.choice(len(ol),size=int(cfg.dstep_drows))) if (ol and int(cfg.dstep_drows)>0) else []
    T_=xt.shape[1]; parts=[xt]; V=I=None
    if dm:
        ri=[int(rs_.randint(0,max(1,min(int(cfg.dstep_hold),BK[d].shape[0])))) for d in dm]
        Xd=torch.stack([BK[d][i][:T_] for d,i in zip(dm,ri)]).long().to(dev); parts.append(Xd)
        V=torch.stack([TV[d][0][i][:T_] for d,i in zip(dm,ri)]).to(dev).float(); I=torch.stack([TV[d][1][i][:T_] for d,i in zip(dm,ri)]).to(dev).long()
    was=model.training; fs=model.feat_scale.clone(); sv_=model.__dict__.get('_s0_dom')
    try:
        model.eval(); model.__dict__['_s0_dom']=[dom_id]*nt+dm
        rt_=_dream_route_rows(cfg,len(dm),0,dev,len(model.blocks)) if dm else None
        if rt_ is not None: rt_=torch.cat([torch.ones(nt,rt_.shape[1],device=dev),rt_])
        H=model.features(torch.cat(parts),route=rt_)
        J=F.cross_entropy(model.heads[dom_id](priv(model,xt,H[:nt],dom_id)).float().reshape(-1,256),yt.reshape(-1))
        if dm:
            Pp=min(int(cfg.dream_stream_prompt),T_//2); spans=[]; i0=0
            for j in range(1,len(dm)+1):
                if j==len(dm) or dm[j]!=dm[i0]: spans.append((dm[i0],i0,j)); i0=j
            ld=torch.cat([F.log_softmax(model.heads[d](priv(model,Xd[a:b],H[nt+a:nt+b],d)).float()/cfg.shunt_T,-1) for d,a,b in spans])
            J=J+_kl_topk(ld,V,I)[:,Pp:].mean()
    finally:
        model.__dict__['_s0_dom']=sv_; model.feat_scale.copy_(fs); model.train(was)
    return J

def _dstep_grad(model,cfg,dom_id,olds,seed,P_):
    with torch.enable_grad():
        J=_dstep_obj(model,cfg,dom_id,olds,seed)
        gs=torch.autograd.grad(J,P_,allow_unused=True)
    return [torch.zeros_like(p_) if g_ is None else g_ for p_,g_ in zip(P_,gs)], J.detach()

@torch.no_grad()
def _srep_note(model,h):
    """cg15 stream_rep: this step's representation jump (before the running mean takes the batch in), read by _StreamGet."""
    with torch.no_grad():
        m=h.detach().float().mean((0,1)); e=model.__dict__.get('_srep')
        if e is None: model.__dict__['_srep']=m.clone(); return
        model.__dict__['_rj']=float(1.0-F.cosine_similarity(m,e,dim=0)); model.__dict__['_srep']=0.9*e+0.1*m
@torch.no_grad()
def _cdet_step(model,cfg,x,y,h,dom_id,step):
    """CHANGE-DETECTION LOG (no training effect): representation jump 1 - cos(batch-mean top-block output, running mean) and loss jump
    (real-text CE before the head refit minus its running mean), kept on the device; reported at the visit end."""
    D=model.__dict__.setdefault('_cdet',{'ema':None,'el':None,'cur':[],'hist':[]})
    m=h.detach().float().mean((0,1))
    ce=F.cross_entropy(model.heads[dom_id](priv(model,x,h.detach(),dom_id)).float().reshape(-1,256),y.reshape(-1))
    vt_=model.__dict__.get('_dctl_vtok',0)
    if D.get('vt')!=vt_: D['vt']=vt_; D['cur']=[]; D['sw']=D['ema'] is not None      # first measured step of a visit (step 0 may take the
                                                                                    #   plain path); first measured visit: no switch to score
    if D['ema'] is not None:
        jr=1.0-F.cosine_similarity(m,D['ema'],dim=0); jl=ce-D['el']; D['cur'].append(torch.stack([jr,jl]))
        D['ema']=0.9*D['ema']+0.1*m; D['el']=0.9*D['el']+0.1*ce
    else: D['ema']=m.clone(); D['el']=ce.clone()

def _cdet_report(model,cfg):
    D=model.__dict__.get('_cdet')
    if not D or len(D['cur'])<20 or not D.get('sw'):
        if D: D['cur']=[]
        return
    A=torch.stack(D['cur']).cpu().numpy(); D['cur']=[]
    sw=A[0]; w=A[5:]; mu=w.mean(0); sd=w.std(0)+1e-12; z=(sw-mu)/sd; sh=(w>=sw[None]).mean(0)
    D['hist'].append((float(z[0]),float(z[1]),float(sh[0]),float(sh[1])))
    H_=np.array(D['hist'])
    print(f"  [change detection] switch into this visit: representation jump {sw[0]:.4f} (z {z[0]:.1f}, {sh[0]:.1%} of the visit's steps at or above it) | "
          f"loss jump {sw[1]/LOG2:+.3f} bits (z {z[1]:.1f}, {sh[1]:.1%}) || {len(H_)} switches so far, flagged with no false alarm in the visit "
          f"(no step at or above): representation {int((H_[:,2]==0).sum())}, loss {int((H_[:,3]==0).sum())} | median z: representation {np.median(H_[:,0]):.1f}, loss {np.median(H_[:,1]):.1f}")

def train_domain(model, meta, meta_opt, opt, get, dom_id, prior_buffer, cfg):
    _t_dom=time.perf_counter(); model.__dict__['_tm']={'last':None,'acc':{},'n':0,'dc0':model.__dict__.get('_dcount',0)}; model.__dict__['_tm_dream']=0.0
    # OLD DOMAINS = every domain seen so far except the current one, ordered from least to most recently visited (single pass:
    # range(dom_id), exactly as before; repeated passes: revisited domains are 'current', all others 'old')
    olds=list(getattr(model,'_olds',None) if getattr(model,'_olds',None) is not None else range(dom_id)); n_old=len(olds)
    model.train()
    _RAMP['s']=1.0   # weak base impedance early
    imp_base=None
    _L=cfg.steps_per_task; _ce_hist=[]                       # this visit's length (VISIT LENGTH BY PROGRESS may change it at visit_decide)
    if cfg.stream: _L=int(cfg.steps_per_task*cfg.stream_cap)                # NO VISIT BOUNDARIES: visits end by detection
    _lr_replan=None
    _nov_j=None
    model.__dict__['_dctl_vtok']=model.__dict__.get('_dctl_vtok',0)+1            # visit token (drift-budget controller bookkeeping)
    if cfg.ctx_carry and int(cfg.ctx_bptt)>1:                                       # WRITE-PATH TRAINING: segments per row this visit (curriculum)
        nv_=model.__dict__['_dctl_vtok']; f_=int(cfg.ctx_bptt_from)
        _BPTT['K']=1 if nv_<=f_ else min(int(cfg.ctx_bptt),2**(nv_-f_))
        print(f"    [write path] visit {nv_}: {_BPTT['K']} segment(s) per training row ({max(1,cfg.batch_size//_BPTT['K'])} rows of {_BPTT['K']*cfg.seq_len} bytes)")
    else: _BPTT['K']=1
    _prof_=None
    _nsl_=int(round(float(cfg.sleep_phase)*cfg.steps_per_task)) if (float(cfg.sleep_phase)>0 and not cfg.stream and float(model.__dict__.get('_real_w',1.0))==0.0) else 0
    if _nsl_>0: _L=cfg.steps_per_task+_nsl_
    _SLP['n']=[0,0]; model.__dict__['_ctrl_bytes']=False
    for step in range(max(_L if (cfg.stream or _nsl_>0) else cfg.steps_per_task,0)):
        if _nsl_>0:                                                                  # OFFLINE SLEEP PHASE (after the awake steps)
            j_=step-cfg.steps_per_task
            ph_=None if (j_<0 or cfg.sleep_ctrl) else ('nrem' if j_<int(round(_nsl_*float(cfg.sleep_nrem))) else 'rem')
            model.__dict__['_sleep_phase']=ph_
            model.__dict__['_ctrl_bytes']=bool(cfg.sleep_ctrl and cfg.sleep_ctrl_bytes and j_>=0)
            if ph_: _SLP['n'][0 if ph_=='nrem' else 1]+=1
        if cfg.profile_visit and model.__dict__.get('_dctl_vtok')==cfg.profile_visit and getattr(cfg,'profile_cpu',False):   # v9.8: CPU COPY-SOURCE PROFILE
            if step==20:
                _prof_=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_stack=True); _prof_.__enter__()
            elif step==24 and _prof_ is not None:
                _prof_.__exit__(None,None,None); cnt_={}
                for ev_ in _prof_.events():
                    if ev_.name in ('aten::copy_','aten::_to_copy','aten::clone','aten::contiguous'):
                        fr_=[f_ for f_ in (ev_.stack or []) if 'threshold_lab' in f_ or '_defs' in f_]
                        k_=(ev_.name,fr_[0] if fr_ else '?'); cnt_[k_]=cnt_.get(k_,0)+1
                print("  [copy sources] 4 steps, copies by code line:\n"+"\n".join(f"    {c_:6d}  {n_:16} {l_[:150]}" for (n_,l_),c_ in sorted(cnt_.items(),key=lambda z:-z[1])[:30]),flush=True)
                _prof_=None
        elif cfg.profile_visit and model.__dict__.get('_dctl_vtok')==cfg.profile_visit and torch.cuda.is_available():   # ONE-OFF GPU PROFILE
            if step==100:
                _prof_=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]); _prof_.__enter__(); _PRF['on']=True
            elif step==110 and _prof_ is not None:
                torch.cuda.synchronize(); _prof_.__exit__(None,None,None); ka_=_prof_.key_averages()
                print("  [profile] 10 steps of visit "+str(cfg.profile_visit)+", top operations by SELF GPU time:\n"+ka_.table(sort_by="self_cuda_time_total",row_limit=30))
                print("  [profile] top operations by TOTAL GPU time (incl. children):\n"+ka_.table(sort_by="cuda_time_total",row_limit=25))
                try:
                    rg_=[e for e in ka_ if e.key.startswith('R:')]; rg_.sort(key=lambda e:-e.device_time_total if hasattr(e,'device_time_total') else -e.cuda_time_total)
                    print("  [profile] NAMED REGIONS (GPU time incl. everything inside, per 10 steps):\n"+"\n".join(f"    {e.key[2:]:20} {((e.device_time_total if hasattr(e,'device_time_total') else e.cuda_time_total)/1e3):9.1f} ms  ({e.count} calls)" for e in rg_))
                except Exception as ex: print(f"  [profile] region table unavailable ({repr(ex)[:100]})")
                _PRF['on']=False; _prof_=None
        if step>=_L: break
        if model.__dict__.get('_sldb') is not None: cfg.dream_batch=model.__dict__.pop('_sldb')   # (safety: never carried past a step)
        if model.__dict__.get('_slmul') is not None:
            mu_=model.__dict__.pop('_slmul')
            for g_ in opt.param_groups: g_['lr']/=mu_
        gs=getattr(model,'_gstep',0); object.__setattr__(model,'_gstep',gs+1)
        _tmark(model,cfg,step,'start')
        if imp_base is not None and step>0 and step%cfg.imp_online_every==0:
            online_importance(model, cfg, imp_base, step/_L)
        if step==0:
            _XDG['cur']=dom_id
            model.__dict__['_dmom']={}
            P_=_ema_params(model); model.__dict__['_ema']=(P_,[p_.detach().clone() for p_ in P_],1.0-1.0/max(2.0,cfg.ema_tail*(cfg.steps_per_task if cfg.stream else _L)))
            model.__dict__['_selh']=[]; model.__dict__['_selk']=0
            model.__dict__.pop('_gcache',None); model.__dict__['_vstart']=None; model.__dict__.pop('_lwfG',None)   # fresh per visit
            object.__setattr__(model,'_live_dom',dom_id)
            object.__setattr__(model,'_xcur',None)
        if step==0:                                     # PERFORMANCE: frozen teachers + stacked old heads for this visit
            object.__setattr__(model,'_teach',None); object.__setattr__(model,'_hstk',None); model.__dict__.pop('_klacc',None)
            if n_old>0 and not cfg.topic_mode and cfg.shunt and (cfg.fast_teacher or cfg.fast_dreams):
                if cfg.grow_blocks_at or (cfg.block_extra and cfg.block_extra_mode=='fresh') or getattr(model,'_spec',None) or cfg.dream_filter:
                    print("    [fast] frozen teachers off for this config (growing / fresh units / specific shunts / dream filter)")
                else: build_teachers(model, cfg)
            if n_old>0 and not cfg.topic_mode and cfg.fast_kl: stack_heads(model, olds)
        fdv=bool(cfg.fast_dreams and getattr(model,'_teach',None) is not None)
        if cfg.dream and n_old>0 and not cfg.topic_mode and fdv and cfg.dream_stream:   # CONTINUOUS DREAM STREAMS
            if step==0: model._dreams={}; model.__dict__['_dsq']={}; model.__dict__['_dstream_last']={}
            if step%cfg.dream_stream_block==0 and (cfg.dream_parallel or cfg.dream_bank):
                model._dsq.setdefault('__par__',[1]); fl_=None                         # parallel dreams: generated per step in stream_batch
            elif step%cfg.dream_stream_block==0:
                fl_=stream_generate(model, cfg, olds, min(cfg.dream_stream_block, _L-step))
                if step==0 and fl_: print("    [dream fidelity] streams, share of each old domain's chunks attributed to that domain: "+" ".join(f"{_lab(cfg,d)} {v:.2f}" for d,v in fl_.items()))
        elif cfg.dream and n_old>0 and not cfg.topic_mode and fdv:      # RECURRENT SAMPLER: the whole visit's dreams at its start
            if step==0: fast_dream_visit(model, cfg, dom_id, olds); model.__dict__['_onpol']={}
            elif step%cfg.dream_refresh==0:
                q_=model.__dict__.get('_dqueue',{}).pop(step,None)
                if q_ is not None: _pool_add(model, cfg, q_[0], q_[1], q_[2])
        elif cfg.dream and n_old>0 and not cfg.topic_mode:
            if step==0:
                model._dreams={}
                sh=dream_shares(cfg, n_old)                     # shares by POSITION in olds (last = most recently visited)
                for i_,d in enumerate(olds):
                    if sh is None: dream_refresh(model, cfg, dom_id, d, cfg.dream_n)
                    elif sh[i_]>0: dream_refresh(model, cfg, dom_id, d, dream_cap(cfg, n_old, i_))   # graduated: no pool
                if sh is not None:
                    print("    [dream budget] share per old domain (recency-weighted"+(f", graduated at age {cfg.dream_graduate}")+"): "
                          +" ".join(f"{_lab(cfg,d)} {sh[i_]:.2f} ({dream_cap(cfg,n_old,i_)})" for i_,d in enumerate(olds)))
            elif step%cfg.dream_refresh==0:
                sh=dream_shares(cfg, n_old)
                dr=olds[(step//cfg.dream_refresh)%n_old] if sh is None else olds[dream_pick(model, cfg, n_old, sh, 1)]
                dream_refresh(model, cfg, dom_id, dr, cfg.dream_batch)
        if step==0:                                                                     # v9.8 cg8 DIRECT LEARNED STEP: fresh multiplier, held-out rows
            _ds_on=bool(cfg.dstep and n_old>0 and cfg.dream_bank and (not cfg.dstep_pass1 or float(model.__dict__.get('_real_w',1.0))>0))
            model.__dict__['_dsm']=1.0; _ds_t=0.0
            if _ds_on: xh_,yh_=get(); model.__dict__['_dsH']=(xh_,yh_); model.__dict__['_dsS']=[]; model.__dict__['_dsM']=[]; model.__dict__['_dsJ']=[]
        x,y=get()
        if cfg.inf_on and float(model.__dict__.get('_real_w',1.0))>0: inf_store_add(model,cfg,x,int(model.__dict__.get('_cur_dom',dom_id)))   # cg20 store
        if cfg.nrem_real and float(model.__dict__.get('_real_w',1.0))>0:            # nrem_real DIAGNOSTIC: keep random pass-1 rows per domain
            with torch.no_grad():
                P1_=model.__dict__.setdefault('_p1win',{}); td1_=int(model.__dict__.get('_cur_dom',dom_id)); E1_=P1_.setdefault(td1_,{'r':[],'seen':0})
                rs1_=model.__dict__.setdefault('_p1_rs',np.random.RandomState(int(cfg.seed)+515))
                for r_ in range(x.shape[0]):
                    E1_['seen']+=1; j1_=len(E1_['r']) if len(E1_['r'])<int(cfg.nrem_real_cap) else rs1_.randint(E1_['seen'])
                    if j1_<int(cfg.nrem_real_cap):
                        rw_=torch.cat([x[r_],y[r_,-1:]]).detach().clone()
                        if j1_<len(E1_['r']): E1_['r'][j1_]=rw_
                        else: E1_['r'].append(rw_)
        if step==0 or (cfg.ctx_carry and getattr(get,'fresh',False)): _CARRY['st']={}          # CONTINUOUS CONTEXT: empty memory at a visit start / lane restart
        if cfg.slot_mem and step==0: _CARRY['slot']=dom_id; _CARRY['sm']=model.__dict__.setdefault('_smem',{})   # SLOT MEMORY of this visit's slot
        model.__dict__.pop('_pairA',None)
        if cfg.stream and model.__dict__.pop('_vend',False): _L=step+1                    # NO VISIT BOUNDARIES: the detector ended this visit (this is its last step)
        if cfg.guard and n_old>0: guard_step(model,cfg,olds,dom_id,step)                   # EARLY DRIFT GUARD
        if step==0: model.__dict__.pop('_film_aff',None)
        if step==0 and n_old>0 and (cfg.borrow_aff>0 or (cfg.film_k>0 and cfg.film_aff and model.__dict__.get('_first_visit',True))):
            w_,b_=aff_weights(model,cfg,x,y,olds,cfg.film_tau)
            if cfg.film_k>0 and cfg.film_aff and model.__dict__.get('_first_visit',True):
                wf_=w_
                with torch.no_grad():
                    C_=model.film_codes; C_[dom_id]=sum(wf_[d_]/len(olds)*C_[d_] for d_ in olds)+0.02*torch.randn_like(C_[dom_id])
            print("    [affinity] old heads' bits on the first batch: "+" ".join(f"{_lab(cfg,d_)} {b_[d_]:.2f}" for d_ in olds)
                  +" | weights: "+" ".join(f"{_lab(cfg,d_)} {w_[d_]:.2f}" for d_ in olds))
        if step==0 and n_old>0 and cfg.pu_k_base>0 and cfg.pu_k>cfg.pu_k_base and model.__dict__.get('_first_visit',True):   # SURPRISE-TRIGGERED GROWTH
            bg_=aff_weights(model,cfg,x,y,olds,1.0)[1]; mb_=min(bg_.values()); hg_=model.__dict__.setdefault('_grow_hist',[])
            med_=float(np.median(hg_)) if hg_ else None; grow_=med_ is not None and mb_>cfg.pu_grow_rel*med_
            if grow_:
                for b_ in model.blocks:
                    if getattr(b_,'pu_mask',None) is not None: b_.pu_mask[dom_id]=1.0
            hg_.append(mb_)
            print(f"    [growth] {_lab(cfg,dom_id)}: best old head {mb_:.2f} bits on the first batch (median of earlier new domains "
                  +(f"{med_:.2f}" if med_ is not None else "-")+f") -> {'ALL '+str(cfg.pu_k) if grow_ else str(cfg.pu_k_base)} private units per block")
        if step==0 and cfg.film_k>0: model.__dict__['_film_frozen']=model.film_codes.detach().clone()
        xf_,yf_=x,y                                                                         # full real batch (counts see every row)
        if cfg.sleep_keep>0 and not (cfg.sleep_revisit and model.__dict__.get('_first_visit',True)):   # WAKE / SLEEP: queue the unexplained rows, train them merged
            SQ_=model.__dict__.setdefault('_slq',[])
            if step==0: SQ_.clear(); model.__dict__['_slst']=[0,0,0]                    # training steps, rows trained, rows left to the fast parts
            else:
                B_=x.shape[0]; k_=max(1,int(round(B_*cfg.sleep_keep)))
                if k_<B_:
                    hs0_=None
                    if cfg.sleep_random:
                        rs_=model.__dict__.setdefault('_slrng',np.random.RandomState(cfg.seed+505))
                        ixk_=torch.as_tensor(np.sort(rs_.choice(B_,k_,replace=False)),device=x.device)
                    else:
                        with torch.no_grad():
                            was_=model.training; model.eval(); fs_=model.feat_scale.clone()
                            hs0_=model.features(x)
                            zs_=model.heads[dom_id](priv(model,x,hs0_,dom_id))
                            lr_row=F.cross_entropy(zs_.reshape(-1,256).float(),y.reshape(-1),reduction='none').view(B_,-1).mean(1)
                            if cfg.sleep_score=='mem':                          # MEMORY GAIN on the second half (context vs none)
                                Th_=x.shape[1]//2; x2_=x[:,Th_:]; h2_=model.features(x2_)
                                z2_=model.heads[dom_id](priv(model,x2_,h2_,dom_id))
                                c0_=F.cross_entropy(z2_.reshape(-1,256).float(),y[:,Th_:].reshape(-1),reduction='none').view(B_,-1).mean(1)
                                c1_=F.cross_entropy(zs_[:,Th_:].reshape(-1,256).float(),y[:,Th_:].reshape(-1),reduction='none').view(B_,-1).mean(1)
                                lr_row=c0_-c1_
                            model.feat_scale.copy_(fs_); model.train(was_)
                        ixk_=torch.sort(torch.topk(lr_row,k_).indices).values
                    msk_=torch.ones(B_,dtype=torch.bool,device=x.device); msk_[ixk_]=False
                    if cfg.sleep_wake and (cfg.ngp or cfg.fcp_tables):                  # rows left to the fast parts: private rows learn now
                        xw_,yw_=x[msk_],y[msk_]
                        if hs0_ is None:
                            with torch.no_grad():
                                was_=model.training; model.eval(); fs_=model.feat_scale.clone()
                                hw_=model.features(xw_); model.feat_scale.copy_(fs_); model.train(was_)
                        else: hw_=hs0_[msk_]
                        opt.zero_grad()
                        with torch.enable_grad():
                            lk_=F.cross_entropy(model.heads[dom_id].detached(priv(model,xw_,hw_.detach(),dom_id)).reshape(-1,256),yw_.reshape(-1)); lk_.backward()
                        ngp_step(model,cfg,opt.param_groups[0]['lr'])
                    model._slst[2]+=int(msk_.sum())
                    x,y=x[ixk_],y[ixk_]
                SQ_.append((x,y))
                nq_=sum(a_.shape[0] for a_,_ in SQ_); need_=int(cfg.sleep_merge*cfg.batch_size)
                if nq_<need_ and step<_L-1: continue                                    # still awake: no trunk step
                xs_=torch.cat([a_ for a_,_ in SQ_]); ys_=torch.cat([b_ for _,b_ in SQ_]); SQ_.clear()
                if step<_L-1 and xs_.shape[0]>need_: SQ_.append((xs_[need_:],ys_[need_:])); xs_,ys_=xs_[:need_],ys_[:need_]
                x,y=xs_,ys_; G_=x.shape[0]/cfg.batch_size
                mu_=cfg.sleep_lr if cfg.sleep_lr>0 else math.sqrt(G_)
                if abs(mu_-1.0)>1e-9: model.__dict__['_slmul']=mu_
                if cfg.sleep_dream and abs(G_-1.0)>1e-9:
                    model.__dict__['_sldb']=cfg.dream_batch; cfg.dream_batch=max(1,int(round(cfg.dream_batch*G_)))
                model._slst[0]+=1; model._slst[1]+=x.shape[0]
            if step==_L-1: print(f"    [sleep] {model._slst[0]} trunk steps on {model._slst[1]} rows (merged {cfg.sleep_merge} x {cfg.batch_size}); "
                                 f"{model._slst[2]} rows left to the fast parts")
        if cfg.pmuon and step==0:
            model.__dict__['_kf_dacc']={}
            if float(cfg.kf_dream)>=1.0 and getattr(cfg,'kf_drop_a',True) and n_old>0:   # v9.8 (g3): live factors from the first step
                with torch.no_grad(): lv0_=kf_live_warmup(model,cfg,olds)
                kf_build(model,cfg,dom_id,live=lv0_); model.__dict__['_kf_dacc']={}
            else: kf_build(model,cfg,dom_id)
        if cfg.pmuon and cfg.kf_dream>0 and n_old>0 and step>0 and step%max(1,int(cfg.kf_dream_every))==0 and model.__dict__.get('_kf_dacc'):
            with torch.no_grad():                                                     # LIVE KEYS: whitening rebuilt from the dream rows' current inputs
                lv_={k_:e_['A']/max(1,e_['n']) for k_,e_ in model._kf_dacc.items() if e_['A'] is not None}
                kf_build(model,cfg,dom_id,live=lv_,quiet=True)
                model.__dict__['_kf_drb']=model.__dict__.get('_kf_drb',0)+1
        _kfm_=bool(cfg.pmuon and cfg.kf_source=='main' and step%cfg.kf_every==cfg.kf_every//2)
        if cfg.noise_diag>0 and n_old>0 and step%cfg.noise_diag==cfg.noise_diag//2: noise_diag(model,cfg,x,y,dom_id,olds)
        if cfg.head_select_eval or cfg.fwt_zeroshot or cfg.block_attr or cfg.block_auto: head_stats_update(model, x, dom_id, step, cfg)
        _rbg_add(model,cfg,x,dom_id)                                         # byte-pair profile of the slot being trained (routing evidence, drift-free)
        lv=model.__dict__.setdefault('_live',[]); lv.append((x,y)); del lv[:-cfg.buffer_batches]   # transient, current domain only
        model._cov_on=(step%cfg.cov_every==0) and not (cfg.cov_skip and not (cfg.orth_gate or cfg.pg_mode or cfg.og_common_always or cfg.polar_diag))
        if cfg.topic_diag: model._topics.update(x, dom_id)
        rb=model.__dict__.setdefault('_recent',[]); rb.append((x,y)); del rb[:-cfg.shunt_batches]
        orth=getattr(model,'_orth',None)
        model._keep_h=bool(cfg.head_select_eval or cfg.fwt_zeroshot)
        if orth is not None and step==0: orth.begin(model, prior_buffer, (x,y), dom_id)
        prog=step/_L
        f_=min(1.0,(step+1)/max(1,cfg.lr_warmup)) if (n_old==0 and model.__dict__.get('_first_visit',True)) else 1.0
        for g_ in opt.param_groups: g_['lr']=cfg.lr*f_*model.__dict__.get('_pass_lr',1.0)*model.__dict__.get('_dsm',1.0)
        if model.__dict__.get('_slmul') is not None:                               # WAKE / SLEEP: merged step's lr (undone after the step)
            for g_ in opt.param_groups: g_['lr']*=model._slmul
        _snrm_=1.0
        if n_old>0 and meta is not None and step % cfg.meta_every == 0 and len(prior_buffer)>0:
            # PROPER UNROLLED META-LEARNING (MAML-style): take a VIRTUAL gated step
            # that stays in the computation graph (differentiable w.r.t. the meta
            # params), measure buffer-loss (forgetting) AFTER the step, and backprop
            # THROUGH the step to the meta-controller. This gives a CLEAN gradient
            # about the gate's effect on forgetting (unlike the noisy FD / invalid
            # proxy that failed). meta learns: "gate so that a step here doesn't
            # raise loss on prior tasks, while keeping current loss low."
            meta_opt.zero_grad()
            # 1. compute current-task grads w.r.t. model params (graph-connected)
            cur_loss=F.cross_entropy(model(x,dom_id).reshape(-1,256),y.reshape(-1))
            block_params=[p for b in model.blocks for p in b.parameters() if p.requires_grad]
            grads=torch.autograd.grad(cur_loss, block_params, create_graph=True, allow_unused=True)
            # 2. build per-unit gates from meta-controller (differentiable), scale grads,
            #    form VIRTUAL updated params: p' = p - lr * gate * grad
            gated=_virtual_gated_step(model, meta, block_params, grads, prog, cfg.lr)
            # 3. measure forgetting: buffer-loss under the VIRTUAL params (functional fwd)
            bx,by,bd=prior_buffer[np.random.randint(len(prior_buffer))]
            buf_loss=_functional_buffer_loss(model, gated, bx, by, bd)
            # 4. plasticity: current-task loss under virtual params (should stay low)
            plast_loss=_functional_buffer_loss(model, gated, x, y, dom_id)
            # 5. meta-objective: low forgetting (buf) + low current loss (plasticity)
            meta_loss = buf_loss + cfg.meta_lambda*plast_loss
            meta_loss.backward()
            meta_opt.step()
        # REAL step: recompute grads, gate with the (updated) meta, step
        if (cfg.borrow_w>0 and cfg.borrow_mode=='grad' and n_old>0 and step%cfg.borrow_check==0 and getattr(model,'_dsq',None)
                and getattr(model,'_teach',None) is not None and (not cfg.borrow_first or model.__dict__.get('_first_visit',True))):
            if float(model.__dict__.get('_real_w',1.0))>0:                    # FIX (review): borrow_align uses the REAL batch's gradient - only in real-text passes;
                borrow_align(model,cfg,olds,dom_id,x,y); model.__dict__.setdefault('_borrow_by',{})[dom_id]=model.__dict__['_borrow_g'].clone()
            elif dom_id in model.__dict__.get('_borrow_by',{}): model.__dict__['_borrow_g']=model.__dict__['_borrow_by'][dom_id].clone()   #   closed loop: weights of the slot's last real-text visit
            elif model.__dict__.get('_borrow_g') is not None: model.__dict__['_borrow_g'].zero_()
        opt.zero_grad()
        model.__dict__.pop('_ng_touch',None)                          # lazy n-gram optimiser: rows touched by THIS step only
        _tmark(model,cfg,step,'prep')
        if cfg.shunt and cfg.shunt_consol=='continuous' and n_old>0 and (getattr(model,'_shunt',None) or (cfg.shunt_lazy and model.__dict__.get('_vstart'))):
            FT=getattr(model,'_teach',None)
            if FT is not None:      # FROZEN TEACHERS: same targets as below without weight swaps; both teachers in ONE stacked pass
                with torch.no_grad():
                    rl_=[teacher_role(model,cfg,d,olds,FT) for d in olds]; grp={}
                    for i_,r_ in enumerate(rl_): grp.setdefault(r_,[]).append(i_)
                    roles=list(grp)
                    lwf_on=(step%max(1,cfg.lwf_every)==0) and cfg.shunt_old_weight>0   # B2: every lwf_every steps; weight 0: never (no teacher pass)
                    hh=dict(zip(roles,FT.features_multi(x,roles))) if lwf_on else {}
                    if not lwf_on or cfg.lwf_metric: tgt=None                   # B1: no per-head targets (Fisher metric instead)
                    elif getattr(model,'_hstk',None) is not None:               # FUSED: all old heads in one matmul per teacher
                        tgt=torch.empty(n_old,*x.shape,cfg.vocab,device=x.device)
                        for r_,ix in grp.items(): tgt[ix]=heads_logp(model,hh[r_],ix,cfg.shunt_T)
                    else: tgt=[F.log_softmax(model.heads[d](hh[rl_[i_]])/cfg.shunt_T,-1) for i_,d in enumerate(olds)]
                    if cfg.dream and getattr(model,'_dreams',None):               # a dream batch from one old domain
                        sh_=dream_shares(cfg, n_old)
                        dd_=olds[step%n_old] if sh_ is None else olds[dream_pick(model, cfg, n_old, sh_, 0)]
                        pool=model._dreams[dd_]
                        if not hasattr(model,'_dream_rng'): object.__setattr__(model,'_dream_rng',np.random.RandomState(cfg.seed+7))
                        ix_=model._dream_rng.randint(len(pool),size=cfg.dream_batch)
                        xd=torch.stack([pool[i] for i in ix_])
                        tp_=model.__dict__.get('_dream_tgt',{}).get(dd_)
                        tgt_d=torch.stack([tp_[i] for i in ix_]) if (tp_ and len(tp_)==len(pool)) else None   # CACHED dream targets
                        if tgt_d is None: tgt_d=F.log_softmax(model.heads[dd_](FT.features(xd,teacher_role(model,cfg,dd_,olds,FT)))/cfg.shunt_T,-1)
                        op_=None
                        model.__dict__['_op_n']=0
                        if op_ is not None and len(op_[0]):                    # ON-POLICY share of the batch (live samples, teacher targets)
                            n_on=max(1,min(cfg.dream_batch-1,int(round(cfg.dream_onpolicy*cfg.dream_batch))))
                            if not hasattr(model,'_onpol_pick'):
                                g_=torch.Generator(device=x.device); g_.manual_seed(cfg.seed+31); object.__setattr__(model,'_onpol_pick',g_)
                            io_=torch.randint(len(op_[0]),(n_on,),generator=model._onpol_pick,device=x.device)
                            xd=torch.cat([xd[:cfg.dream_batch-n_on],op_[0][io_]]); tgt_d=torch.cat([tgt_d[:cfg.dream_batch-n_on],op_[1][io_]])
                            model.__dict__['_op_n']=n_on; model.__dict__.pop('_prio_ix',None)
            else:
                # CONTINUOUS CONSOLIDATION: old memories read live from the MAIN weights (weights minus shunt);
                # the full network's old-domain outputs on this batch are kept matching them
                Pm={id(p_):p_ for p_ in model.parameters()}; Sd={k:v for k,v in model._shunt.items() if k in Pm and Pm[k].shape==v.shape}
                with torch.no_grad():
                    wasm=model.training; model.eval(); fsm=model.feat_scale.clone()
                    _Pl=[Pm[k] for k in Sd]; _Sl=list(Sd.values())
                    if _Pl: torch._foreach_sub_(_Pl,_Sl)                                  # main weights (fused)
                    hm=model.features(x); tgt=[F.log_softmax(model.heads[d](hm)/cfg.shunt_T,-1) for d in olds]
                    if cfg.dream and getattr(model,'_dreams',None):               # a dream batch from one old domain
                        sh_=dream_shares(cfg, n_old)
                        dd_=olds[step%n_old] if sh_ is None else olds[dream_pick(model, cfg, n_old, sh_, 0)]
                        pool=model._dreams[dd_]
                        if not hasattr(model,'_dream_rng'): object.__setattr__(model,'_dream_rng',np.random.RandomState(cfg.seed+7))
                        xd=torch.stack([pool[i] for i in model._dream_rng.randint(len(pool),size=cfg.dream_batch)])
                        tgt_d=F.log_softmax(model.heads[dd_](model.features(xd))/cfg.shunt_T,-1)
                    if _Pl: torch._foreach_add_(_Pl,_Sl)
                    model.feat_scale.copy_(fsm); model.train(wasm)
            pre_=None
            _dk0_=None
            _dk0_=int(model.__dict__.get('_dream_k',max(1,cfg.dream_every)))
            if model.__dict__.get('_sleep_phase')=='rem': _dk0_=1                         # REM: dreams on every step
            if step==0 and (cfg.sleep_eps>0 or cfg.sleep_rel>0): model.__dict__['_dream_k']=1; _dk0_=1   #   (they used to ride along in the
            if (cfg.merge_dream_fwd and cfg.dream_stream and getattr(model,'_dsq',None) and not cfg.share_blocks and cfg.strand_anchor<=0    #   merged pass
                    and step%max(1,cfg.dream_every)==0 and not cfg.dream_reuse and not cfg.proj_nodream          #   and its backward with no loss)
                    and (_dk0_ is None or step%_dk0_==0)):
                sbp_=stream_batch(model,cfg,olds)                                     # A6: real + dream rows in ONE forward pass
                if sbp_ is not None and sbp_[0].shape[1]==x.shape[1]:
                    ns_p=sbp_[3]['n'] if sbp_[3] is not None else 0
                    rtp_=_dream_route_rows(cfg,sbp_[0].shape[0]-ns_p,ns_p,x.device,len(model.blocks))
                    if rtp_ is not None: rtp_=torch.cat([torch.ones(x.shape[0],rtp_.shape[1],device=x.device),rtp_])
                    model.__dict__['_s0_dom']=[dom_id]*x.shape[0]+list(sbp_[2])+[None]*(sbp_[0].shape[0]-len(sbp_[2]))
                    kfd_=bool(cfg.kf_dream>0 and cfg.pmuon and model.__dict__.get('_dcount',0)%max(1,int(cfg.kf_dream_stride))==0)
                    if _kfm_ or kfd_:
                        kf_main_on(model,cfg,x.shape[0]); _KF['dream']=kfd_; _KF['real_off']=not _kfm_; _KF['dacc']=model.__dict__.setdefault('_kf_dacc',{})
                    _carry_on(cfg,x.shape[0])
                    if cfg.dream_pair>0: _pair_on(model,cfg,x.shape[0])                         # PAIRED-DREAM EPISODES
                    try: Hall=model.features(torch.cat([x,sbp_[0]]),route=rtp_,nstat=x.shape[0])
                    finally: _carry_off(cfg)
                    _KF['on']=False; _KF['dream']=False; _KF['real_off']=False
                    model.__dict__['_s0_dom']=None
                    h_=Hall[:x.shape[0]]; pre_=(sbp_,Hall[x.shape[0]:])
                elif sbp_ is not None: pre_=(sbp_,None)
            if pre_ is None or pre_[1] is None:
                model.__dict__['_s0_dom']=dom_id
                if _kfm_: kf_main_on(model,cfg,x.shape[0])
                _carry_on(cfg,x.shape[0])
                _nr_=bool(cfg.cd_skip_real and cfg.cd_w>0 and float(model.__dict__.get('_real_w',1.0))==0.0 and not _kfm_)
                try:
                    with _rf('real_forward'):
                        if _nr_:
                            with torch.no_grad(): h_=model.features(x)                # (its loss is weighted 0: no trunk backward)
                        else: h_=model.features(x)
                finally: _carry_off(cfg)
                model.__dict__['_s0_dom']=None; _KF['on']=False
            if cfg.stream:                                                                  # change detection: this batch's loss BEFORE the head refit
                with torch.no_grad(): model.__dict__['_rl']=F.cross_entropy(model.heads[dom_id](priv(model,x,h_.detach(),dom_id)).float().reshape(-1,256),y.reshape(-1))
                if cfg.stream_rep: _srep_note(model,h_)
            if cfg.cdet_log: _cdet_step(model,cfg,x,y,h_,dom_id,step)                       # v9.8 cg8 change-detection log (no training effect)
            if cfg.vp_nreal or float(model.__dict__.get('_real_w',1.0))>0: varpro_refit(model,cfg,opt,x,y,h_,dom_id)   # (vp_nreal False: no label leak)
            zc_=model.heads[dom_id](priv(model,x,h_,dom_id))
            loss=F.cross_entropy(zc_.reshape(-1,256),y.reshape(-1))
            _rw_=float(model.__dict__.get('_real_w',1.0))
            if _rw_!=1.0: loss=loss*_rw_                                                     # REAL-TEXT LOSS WEIGHT (real_w_pass)
            klm=0.0
            fk_=bool(cfg.fast_kl and getattr(model,'_hstk',None) is not None and cfg.shunt_kl_mode=='fixed')
            skip_lwf_=False
            if FT is not None and (not lwf_on or cfg.lwf_metric):
                skip_lwf_=True; fk_=False
                if lwf_on:                                                           # B1: FISHER-METRIC KL (all old heads in one metric)
                    Gd_,Gr_=model._lwfG
                    for r_,ix in grp.items():
                        dl_=(h_-hh[r_]).float(); klm=klm+0.5*((dl_@Gr_[r_])*dl_).sum(-1).mean()
                    if step%cfg.stat_every==0:                                       # per-domain values for the log (sampled)
                        with torch.no_grad():
                            kd_=torch.stack([0.5*(((h_-hh[rl_[i_]]).float()@Gd_[i_])*(h_-hh[rl_[i_]]).float()).sum(-1).mean() for i_ in range(n_old)])
                            ka_=model.__dict__.get('_klacc'); model.__dict__['_klacc']=(kd_ if ka_ is None else ka_[0]+kd_, 1 if ka_ is None else ka_[1]+1)
            pw_=None
            if cfg.lwf_posterior and n_old>0 and not skip_lwf_:                       # POSTERIOR WEIGHTS for the current-batch KL
                with torch.no_grad():
                    tgs_=tgt if torch.is_tensor(tgt) else torch.stack(list(tgt))
                    llo_=tgs_.float().gather(-1,y.unsqueeze(0).unsqueeze(-1).expand(tgs_.shape[0],-1,-1,1)).squeeze(-1)   # (n_old,B,T)
                    lln_=F.log_softmax(model.heads[dom_id].detached(h_.detach()).float(),-1).gather(-1,y.unsqueeze(-1)).squeeze(-1).unsqueeze(0)
                    LL_=torch.cat([llo_,lln_]); po_=F.softmax((torch.cumsum(LL_,-1)-LL_)*cfg.hs_temp,0)              # causal: bytes before t
                    pw_=cfg.lwf_post_floor+(1-cfg.lwf_post_floor)*(n_old+1)*po_[:n_old]
                    wa_=model.__dict__.get('_lwfwacc'); m_=pw_.mean((1,2)); pn_=po_[n_old].mean()
                    model.__dict__['_lwfwacc']=(m_,pn_,1) if wa_ is None else (wa_[0]+m_,wa_[1]+pn_,wa_[2]+1)
                    rel_=(po_[:n_old].mean((1,2))*(n_old+1)).clamp(max=1.0)
                    R_=model.__dict__.get('_borrow_r'); oi_=torch.tensor(olds,device=y.device)
                    if R_ is None or step<=1:
                        R_=model.__dict__['_borrow_r']=torch.zeros(_nslots(cfg),device=y.device); R_[oi_]=rel_
                    else: R_[oi_]=0.9*R_[oi_]+0.1*rel_
            if fk_:                                                                  # FUSED KL over every old head at once
                tg_=tgt if torch.is_tensor(tgt) else torch.stack(tgt)
                lt=heads_logp(model,h_,tuple(range(n_old)),cfg.shunt_T)
                kt_=(tg_.exp()*(tg_-lt)).sum(-1)                                     # (n_old,B,T)
                if pw_ is not None: kt_=kt_*pw_
                kd_=kt_.mean((1,2)); klm=kd_.sum()
                ka_=model.__dict__.get('_klacc'); model.__dict__['_klacc']=(kd_.detach() if ka_ is None else ka_[0]+kd_.detach(), 1 if ka_ is None else ka_[1]+1)
            for i_,d in (enumerate(olds) if not (fk_ or skip_lwf_) else []):
                hd=model.heads[d]; lt=F.log_softmax(hd.detached(h_)/cfg.shunt_T,-1)
                kt=(tgt[i_].exp()*(tgt[i_]-lt)).sum(-1)                                # per-token KL
                if pw_ is not None: kt=kt*pw_[i_]
                kd=kt.mean()
                kst=model.__dict__.setdefault('_klstat',{}).setdefault(d,[0.0,0.0,0])
                kst[1]+=cfg.shunt_old_weight
                kst[0]+=kt.mean().detach(); kst[2]+=1                    # tensor: no GPU sync per step
                klm=klm+kd
            loss=loss+cfg.shunt_old_weight*klm*(cfg.shunt_T**2)/n_old*(max(1,cfg.lwf_every) if FT is not None else 1)
            eb_=1.0+(cfg.dream_early_mult-1.0)*max(0.0,1.0-step/max(1,cfg.dream_early_steps))   # EARLY-VISIT dream boost
            _dterm=0.0
            if _dk0_ is not None: _dk_=_dk0_                                         # (decided before the merged forward)
            else:
                _dk_=int(model.__dict__.get('_dream_k',max(1,cfg.dream_every)))
                if step==0 and (cfg.sleep_eps>0 or cfg.sleep_rel>0): model.__dict__['_dream_k']=1; _dk_=1
                if model.__dict__.get('_sleep_phase')=='rem': _dk_=1
            if cfg.dream_stream and getattr(model,'_dsq',None) and step%_dk_==0 and not cfg.proj_nodream:   # CONTINUOUS STREAMS (every dream_every steps)
                sb_=pre_[0] if pre_ is not None else stream_batch(model,cfg,olds)
                if sb_ is not None:
                    xd,tgt_d,dm_,ex_=sb_
                    Bd=xd.shape[0]; ns_=ex_['n'] if ex_ is not None else 0; n1_=Bd-ns_; Pp=min(cfg.dream_stream_prompt,xd.shape[1]//2)
                    rt_=_dream_route_rows(cfg,n1_,ns_,xd.device,len(model.blocks))
                    shb_=None
                    if shb_:                                                                  # DREAM ONLY THE SIMILARITIES in these blocks
                        for li_,(U_,_,_) in shb_.items(): model.blocks[li_]._gproj=(U_,float(cfg.share_spec),float(cfg.share_keep))
                    anc_=bool(ns_ and cfg.strand_anchor>0 and getattr(model,'_sabasis',None))
                    if anc_:
                        for li_ in cfg.strand_anchor_blocks: model.blocks[li_]._cap=True
                    if pre_ is not None and pre_[1] is not None: hfd=pre_[1]              # A6: already computed with the real batch
                    else:
                        wasd=model.training; model.eval(); fsd=model.feat_scale.clone()
                        model.__dict__['_s0_dom']=list(dm_)+[None]*(xd.shape[0]-len(dm_))
                        _STF['dream_call']=True
                        with _rf('dream_forward'): hfd=model.features(xd,route=rt_)
                        _STF['dream_call']=False; model.__dict__['_s0_dom']=None
                        model.feat_scale.copy_(fsd); model.train(wasd)
                    if shb_:
                        for li_ in shb_: model.blocks[li_]._gproj=None
                    spans=[]; i0=0
                    for j in range(1,n1_+1):
                        if j==n1_ or dm_[j]!=dm_[i0]: spans.append((dm_[i0],i0,j)); i0=j
                    if ex_ is None:
                        hh_=hfd                 # OLD HEADS learn from their own dreams (option)
                        def _dz(d,a,b):
                            hq_=priv(model,xd[a:b],hh_[a:b],d)
                            z_=model.heads[d](hq_)
                            return z_
                        vv_=model.__dict__.get('_bank_vi'); vv_=vv_ if (vv_ is not None and spans and vv_[0].shape[0]==spans[-1][2]) else None
                        model.__dict__['_vpo_k']=int(_dk_)
                        varpro_old_refit(model,cfg,xd,hfd,spans,Pp,vv_,tgt_d)
                        ld=torch.cat([F.log_softmax(_dz(d,a,b)/cfg.shunt_T,-1) for d,a,b in spans])
                        R_b=model.__dict__.get('_borrow_g')
                        if cfg.borrow_w>0 and R_b is not None and (not cfg.borrow_first or model.__dict__.get('_first_visit',True)):   # FORWARD BORROWING
                            rr_=R_b[torch.tensor(dm_,device=xd.device)]
                            lc_=F.log_softmax(model.heads[dom_id](hfd).float()/cfg.shunt_T,-1)
                            kb_=(tgt_d.exp()*(tgt_d-lc_)).sum(-1)[:,Pp:].mean(-1)
                            loss=loss+cfg.borrow_w*_dk_*(kb_*rr_).mean()
                        vi_=model.__dict__.pop('_bank_vi',None)
                        if vi_ is not None and vi_[0].shape[0]==ld.shape[0]:                  # STORED TOP-k TARGETS: top-k bytes + one rest bin
                            ktk=_kl_topk(ld,vi_[0],vi_[1])[:,Pp:]
                        else: ktk=(tgt_d.exp()*(tgt_d-ld)).sum(-1)[:,Pp:]                     # no KL on the prompt
                        bw_=model.__dict__.pop('_bank_w',None)                               # importance weights of the bank rows (bank_select='is')
                        if bw_ is not None and bw_.shape[0]==ktk.shape[0]: ktk=ktk*bw_[:,None]
                        kr_=ktk.detach().mean(-1)
                        bri_=model.__dict__.pop('_bank_ri',None)
                        if bri_ is not None and len(bri_[1])==kr_.shape[0] and float(cfg.dream_prio)>0:   # cg16: each row's KL (no sync) for its priority
                            RK_=model.__dict__.setdefault('_rowkl',{}); BKp_=model.__dict__.get('_bank',{})
                            for d_p,a_p,b_p in spans:
                                if d_p not in BKp_: continue
                                if d_p not in RK_: RK_[d_p]=torch.full((BKp_[d_p].shape[0],),float('nan'),device=kr_.device)
                                RK_[d_p][torch.as_tensor(bri_[1][a_p:b_p],device=kr_.device)]=kr_[a_p:b_p].float()
                            pc_=model.__dict__.get('_rowkl_n',0)+1; model.__dict__['_rowkl_n']=pc_
                            if pc_%max(1,int(cfg.dream_prio_sync))==0:
                                model.__dict__['_rowkl_h']={k_p:v_p.cpu().numpy() for k_p,v_p in RK_.items()}
                        ktk=cfg.dream_cap*torch.log1p(ktk/cfg.dream_cap)
                        tab_=None
                        if tab_:                                                             # BYTE-TYPE IMPORTANCE WEIGHTS per dream token
                            wt_=torch.ones_like(ktk); cn_=_BYTE_CLS.to(xd.device)[xd[:,Pp+1:]]   # KL at t predicts byte t+1
                            for d,a,b in spans:
                                if d in tab_: wt_[a:b,:-1]=tab_[d][cn_[a:b]]
                            wt_=wt_/wt_.mean(); ktk=ktk*wt_
                        kdr=ktk.mean()*float(cfg.sleep_fixed)                            #   (no host sync, deterministic schedule)
                        km_=int(cfg.sleep_kmax)
                        if cfg.sleep_kmax_pass: km_=int(cfg.sleep_kmax_pass[min(int(model.__dict__.get('_pass_i',0)),len(cfg.sleep_kmax_pass)-1)])
                        model.__dict__['_dream_k']=max(1,km_)
                        sc_=model.__dict__.setdefault('_sleepC',[0,0]); sc_[0]+=1
                    else:                                                                    # STRANDS in the batch: per-position heads and masks
                        T_=xd.shape[1]
                        hd_=torch.cat([torch.tensor(dm_,device=xd.device).view(-1,1).expand(-1,T_),ex_['hd']])
                        w1_=torch.ones(n1_,T_,device=xd.device); w1_[:,:Pp]=0.0; wm_=torch.cat([w1_,ex_['w']])
                        ktk=torch.zeros(Bd,T_,device=xd.device)
                        for d in sorted(set(hd_.unique().tolist())-{-1}):
                            m_=(hd_==d); rw_=m_.any(1).nonzero().squeeze(1)
                            ld_=F.log_softmax(model.heads[d].detached(hfd[rw_])/cfg.shunt_T,-1)
                            ktk=ktk.index_add(0,rw_,(tgt_d[rw_].exp()*(tgt_d[rw_]-ld_)).sum(-1)*m_[rw_])
                        for j_,pr_ in enumerate(ex_['pr']):                                  # AGREEMENT rows: KL(teacher product || live product)
                            if pr_ is None: continue
                            i_=n1_+j_; hh_=hfd[i_:i_+1]
                            lp_=F.log_softmax(0.5*F.log_softmax(model.heads[pr_[0]].detached(hh_)/cfg.shunt_T,-1)+0.5*F.log_softmax(model.heads[pr_[1]].detached(hh_)/cfg.shunt_T,-1),-1)
                            ka_=(tgt_d[i_:i_+1].exp()*(tgt_d[i_:i_+1]-lp_)).sum(-1)
                            ktk=ktk.index_add(0,torch.tensor([i_],device=xd.device),ka_)
                        kraw_=ktk.detach()
                        if any(k_==3 for k_ in ex_['kind']):
                            st0_=model.__dict__.setdefault('_strandstat',{'a':0.0,'b':0.0,'n':0,'anc':{}})
                            ag_=kraw_[n1_:]; wa_=ex_['w']; st0_['ag']=st0_.get('ag',0.0)+(ag_*wa_).sum()/wa_.sum().clamp(min=1); st0_['agn']=st0_.get('agn',0)+1
                        kr_=(kraw_*wm_).sum(-1)/wm_.sum(-1).clamp(min=1.0)
                        ktk=cfg.dream_cap*torch.log1p(ktk/cfg.dream_cap)
                        kdr=(ktk*wm_).sum()/wm_.sum().clamp(min=1.0)
                        st_=model.__dict__.setdefault('_strandstat',{'a':0.0,'b':0.0,'n':0,'anc':{}})
                        sg_=ex_['seg']
                        if bool((sg_==1).any()):
                            ks_=kraw_[n1_:]
                            st_['a']+=(ks_*(sg_==1)).sum()/(sg_==1).sum().clamp(min=1); st_['b']+=(ks_*(sg_==2)).sum()/(sg_==2).sum().clamp(min=1); st_['n']+=1
                        if anc_:                                                             # OLD-SUBSPACE ANCHORING on the strand rows
                            hts_=model._teach.features_taps(xd[n1_:],'main',tuple(cfg.strand_anchor_blocks)); aw_=ex_['aw']
                            for li_ in cfg.strand_anchor_blocks:
                                bk_=model.blocks[li_]; bk_._cap=False; hs_=bk_.__dict__.pop('_cap_out')[n1_:].float(); U_=model._sabasis[li_]
                                ds_=(hs_-hts_[li_])@U_; dt_=hts_[li_]@U_
                                fe_=((ds_**2).sum(-1)*aw_).sum()/((dt_**2).sum(-1)*aw_).sum().clamp(min=1e-8)
                                loss=loss+cfg.strand_anchor*fe_
                                a_=st_['anc'].setdefault(li_,[0.0,0,U_.shape[1]]); a_[0]+=fe_.detach(); a_[1]+=1; a_[2]=U_.shape[1]
                    for li_ in (cfg.strand_anchor_blocks if anc_ else ()):
                        model.blocks[li_]._cap=False; model.blocks[li_].__dict__.pop('_cap_out',None)
                    wdr=(cfg.dream_weight*model.__dict__.get('_dwp',1.0))*(1.0-cfg.dream_decay*step/_L)*eb_*(_dk_)
                    model.__dict__['_dk_last']=_dk_
                    model.__dict__['_dstep_k']=int(_dk_)            # this step carries dreams standing for k steps
                    loss=loss+wdr*kdr*(cfg.shunt_T**2)
                    for d,a,b in spans:
                        kd_=kr_[a:b].mean(); ds=model.__dict__.setdefault('_dreamstat',{}).setdefault(d,[0.0,0,0.0,0.0,0])
                        ds[0]+=kd_; ds[1]+=1; ds[2]+=wdr; ds[3]+=kd_; ds[4]+=(kd_>1.0).float()
                        dm=model.__dict__.setdefault('_dmax',{}); dm[d]=torch.maximum(dm[d],kd_) if d in dm else kd_
            if cfg.dream and getattr(model,'_dreams',None):
                wasd=model.training; model.eval(); fsd=model.feat_scale.clone()      # eval mode: dreams leave no statistics
                fa_=cfg.feat_anchor>0 and getattr(model,'_teach',None) is not None
                fab_=(cfg.feat_anchor_block,)
                if fa_:
                    for li_ in fab_: model.blocks[li_]._cap=True
                hdd=model.heads[dd_]; ld=F.log_softmax(hdd.detached(model.features(xd))/cfg.shunt_T,-1)
                if fa_:
                    hts_=model._teach.features_taps(xd,teacher_role(model,cfg,dd_,olds,model._teach),fab_)
                    bi_=min(cfg.dream_burnin,xd.shape[1]//2) if cfg.dream_burnin>0 else 0
                    fa_s=model.__dict__.setdefault('_fastat',{})
                    for li_ in fab_:
                        bk_=model.blocks[li_]; bk_._cap=False; hs_=bk_.__dict__.pop('_cap_out').float()[:,bi_:]; ht_=hts_[li_][:,bi_:]
                        fe_=((hs_-ht_)**2).mean()/(ht_.pow(2).mean()+1e-8)             # relative error of the block's output on dreams
                        loss=loss+cfg.feat_anchor*fe_
                        a_=fa_s.setdefault(li_,[0.0,0]); a_[0]+=fe_.detach(); a_[1]+=1
                model.feat_scale.copy_(fsd); model.train(wasd)
                ktk=(tgt_d.exp()*(tgt_d-ld)).sum(-1)                              # per-token KL (B,T)
                if cfg.dream_burnin>0: ktk=ktk[:,min(cfg.dream_burnin,ktk.shape[1]//2):]   # BURN-IN: no KL on the context-free first positions
                kraw_t=ktk.detach().mean()
                on_=model.__dict__.get('_op_n',0)
                if on_:                                                      # on-policy vs teacher-sampled dream KL (no sync)
                    os_=model.__dict__.setdefault('_opstat',[0.0,0.0,0]); os_[0]+=ktk.detach()[-on_:].mean(); os_[1]+=ktk.detach()[:-on_].mean(); os_[2]+=1
                pix_=model.__dict__.pop('_prio_ix',None)
                if pix_ is not None and pix_[0]==dd_: model._dscore[dd_][pix_[1]]=ktk.detach().mean(-1)   # new priorities (GPU)
                ktk=cfg.dream_cap*torch.log1p(ktk/cfg.dream_cap)
                kdr=ktk.mean()
                _sync=(step%cfg.stat_every==0) or cfg.dream_adapt or cfg.dream_rollback   # host values only when needed
                kraw=float(kraw_t) if _sync else 0.0; kv=0.0
                wd_=model.__dict__.setdefault('_dworst',{})                         # worst dream batch this domain (diagnostic)
                if _sync and kraw>wd_.get(dd_,(0.0,))[0]:
                    with torch.no_grad():
                        sq=ktk.detach().mean(-1); i_=int(sq.argmax())
                        pr=[]
                        if kraw>1.0:
                            wasp=model.training; model.eval(); fsp=model.feat_scale.clone()
                            for b in model.blocks: b._probe=pr
                            hp=model.features(xd[i_:i_+1]); lg=model.heads[dd_](hp)
                            for b in model.blocks: b._probe=None
                            model.feat_scale.copy_(fsp); model.train(wasp)
                            pr.append((float(hp.abs().max()),float(lg.max()-lg.min())))
                    wd_[dd_]=(kraw,bytes(xd[i_].tolist()).decode('latin-1'),float(ld[i_].detach().min()),pr)
                ds=model.__dict__.setdefault('_dreamstat',{}).setdefault(dd_,[0.0,0,0.0,0.0,0]); ds[0]+=kraw_t; ds[1]+=1
                ds[3]+=kdr.detach(); ds[4]+=(kraw_t>1.0).float()
                dm=model.__dict__.setdefault('_dmax',{}); dm[dd_]=torch.maximum(dm[dd_],kraw_t) if dd_ in dm else kraw_t
                wdr=(cfg.dream_weight*model.__dict__.get('_dwp',1.0))*(1.0-cfg.dream_decay*step/_L)*eb_
                ds[2]+=wdr
                loss=loss+wdr*kdr*(cfg.shunt_T**2)
        else:
            if _kfm_: kf_main_on(model,cfg,x.shape[0])                         # Fisher factors of the first domain too (observation only)
            _carry_on(cfg,x.shape[0])                                          # CONTINUOUS CONTEXT on the first domain too
            try: loss=F.cross_entropy(model(x,dom_id).reshape(-1,256),y.reshape(-1))
            finally: _carry_off(cfg)
            _KF['on']=False
            if cfg.stream: model.__dict__['_rl']=loss.detach()                  # change detection (first domain: no head refit before the loss)
            if cfg.stream and cfg.stream_rep:                                  # cg15: the first domain's representation (one no-grad trunk pass)
                model.__dict__['_s0_dom']=dom_id
                try:
                    with torch.no_grad(): _srep_note(model,model.features(x))
                finally: model.__dict__['_s0_dom']=None
        if cfg.recon_w>0 and float(model.__dict__.get('_real_w',1.0))>0:             # REPLAY TRAINING (only with real-text gradient on)
            loss=loss+cfg.recon_w*(recon_loss_mark(model,cfg,x,dom_id) if cfg.recon_mark else recon_loss(model,cfg,x,y,dom_id))
            if step==_L-1 and model.__dict__.get('_rc_stat'):
                st_=model.__dict__.pop('_rc_stat'); nb_=model.__dict__.pop('_rc_bad',0)
                print(f"    [replay training] mean loss of a window regenerated from its own memory {st_[0]/max(1,st_[1])/LOG2:.4f} BPC"+(f" | {nb_} non-finite steps skipped" if nb_ else "")+_rs_msg())
        if cfg.cd_w>0 and float(model.__dict__.get('_real_w',1.0))==0.0:            # CONDUIT REPLAY (Stage C)
            if cfg.cd_teacher=='live': model.__dict__.pop('_cd_P',None)              # LIVE TEACHER: no copy - the current weights read the traces
            elif (step==0 and not cfg.cd_fixed) or '_cd_P' not in model.__dict__: model.__dict__['_cd_P']={k_:p_.detach().clone() for k_,p_ in model.named_parameters()}
            _cdf_=_cd_lrf(cfg,x); model.__dict__['_cdf']=_cdf_                          # (cd_lr: conduit update scale)
            loss=loss+cfg.cd_w*(_cdf_ if model.__dict__.get('_dstep_k') is not None else 1.0)*cd_loss(model,cfg,x,dom_id)
            if step==_L-1:
                st_=model.__dict__.pop('_cd_stat',None)
                if not cfg.cd_fixed: model.__dict__.pop('_cd_P',None)
                model.__dict__.pop('_cd_cache',None)
                if _CDR[1] and (cfg.cd_reuse>1 or cfg.cd_lr_teach): print(f"    [v9] replays reused on {_CDR[0]} of {_CDR[1]} conduit steps"+(f" | mean step scale (teaching share) {_CDR[2]/max(1,_CDR[3]):.3f}" if _CDR[3] else ""))
                _CDR[:]=[0,0,0.0,0]
                if _CDT['n']:
                    a_=_CDT['acc']; print("    [conduit timing] ms per conduit step (sampled, synchronised): "+" | ".join(f"{k_} {1000*v_/_CDT['n']:.0f}" for k_,v_ in a_.items() if k_!='_start')+f" | total {1000*sum(v_ for k_,v_ in a_.items() if k_!='_start')/_CDT['n']:.0f}"+_cdg_msg())
                    _CDT.update(acc={},n=0)
                RVp_=model.__dict__.get('_nres')
                if RVp_ and RVp_.get('seen') and int(cfg.nrem_res)>0:
                    print(f"    [NREM v1] reservoir {len(RVp_['e'])}/{cfg.nrem_res} windows of {RVp_['seen']} seen | sleep: {_SLP['n'][0]} NREM steps, {RVp_['rep']} re-teachings, {RVp_['ret']} retired as consolidated | "
                          f"mean remaining signal {np.mean([e_['v'] for e_ in RVp_['e']]) if RVp_['e'] else 0:.3f} (at storage {np.mean([e_['v0'] for e_ in RVp_['e']]) if RVp_['e'] else 0:.3f})"
                          +(f" || cross-domain: {RVp_.get('rep_o',0)} re-teachings of other domains' windows, {RVp_.get('ret_o',0)} retired; kept samples "+", ".join(f"domain {d_}: {len(l_)}" for d_,l_ in sorted(model.__dict__.get('_nres_bank',{}).items())) if float(cfg.nrem_mix)>0 else ""))
                SBp_=model.__dict__.get('_sbuf')
                if SBp_ and SBp_['st'].get('seen'):
                    S_=SBp_['st']; E_=SBp_['e']; bd_={}
                    for e_ in E_: bd_[e_['dom']]=bd_.get(e_['dom'],0)+1
                    print(f"    [long-term store] size {len(E_)}/{cfg.cd_store_cap} | admitted {S_.get('adm',0)} by signal + {S_.get('admr',0)} at random of {S_.get('seen',0)} windows | "
                          f"re-checks {S_.get('chk',0)} ({S_.get('chk_old',0)} of other domains' windows) | retired as consolidated {S_.get('ret',0)}"
                          f" (after {S_.get('ret_n',0)/max(1,S_.get('ret',0)):.1f} re-checks) | evicted {S_.get('evict',0)} | mean signal of stored {np.mean([e_['v'] for e_ in E_]) if E_ else 0:.3f}"
                          f" | by domain "+" ".join(f"{_lab(cfg,d_)}:{c_}" for d_,c_ in sorted(bd_.items()))
                          +(f" || sleep: NREM {_SLP['n'][0]} steps ({S_.get('nrem',0)} re-replays, {S_.get('nrem_ret',0)} retired), REM {_SLP['n'][1]} steps (dreams only)" if sum(_SLP['n']) else ""))
                    SMs_=model.__dict__.pop('_semstat',None); SMb_=model.__dict__.get('_sem',{})
                    if float(cfg.sem_share)>0: print(f"    [semantic rows] written this visit {S_.get('sem',0)} | per domain "+" ".join(f"{_lab(cfg,d_)}:{len(v_['x'])}" for d_,v_ in sorted(SMb_.items()))
                                                     +(f" | dream rows that were semantic {SMs_[0]}/{SMs_[1]}" if SMs_ else " | none used yet"))
                    SBp_['st']={}
                LBp_=model.__dict__.get('_lbuf')
                if LBp_ and LBp_.get('st') and LBp_['st'][0]>0:
                    S_=LBp_['st']; print(f"    [maintenance loop] {S_[0]} re-checks of stored windows | easy (interval x2) {S_[1]}, degrading (re-taught, interval halved) {S_[2]}, "
                                         f"dropped after two failures {S_[3]}, dropped as consolidated {S_[4]} | mean degradation {S_[5]/S_[0]:.4f} | stored now {len(LBp_['e'])}")
                    LBp_['st']=[0,0,0,0,0,0.0]
                RUp_=model.__dict__.pop('_rustat',None)
                if RUp_: print(f"    [replay reuse] {RUp_[0]} windows reused ({cfg.cd_reuse_mode}, lag {cfg.cd_reuse_lag}) | fragile-piece share: reused {float(RUp_[1])/max(1,RUp_[0]):.3f}, eligible pool {float(RUp_[2])/max(1,RUp_[0]):.3f}")
                RSp_=model.__dict__.pop('_rsig',None)
                if RSp_:
                    A_=torch.cat(RSp_,1).float(); kq_,cq_,nq_=A_[0],A_[1],A_[2]; sig_=kq_*nq_; tot_=float(sig_.sum()); N_=kq_.numel()
                    qs_=[float(v_) for v_ in torch.quantile(kq_[:min(N_,200000)],torch.tensor([0.1,0.5,0.9],device=kq_.device))]
                    srt_=torch.sort(sig_).values; cum_=srt_.cumsum(0)
                    bot50_=float(cum_[N_//2-1])/max(tot_,1e-12) if N_>1 else 0.0; top20_=1.0-float(cum_[int(N_*0.8)-1])/max(tot_,1e-12) if N_>4 else 1.0
                    und_=' '.join(f"<{t_}:{float((kq_<t_).float().mean()):.2f}" for t_ in (0.02,0.05,0.1,0.2))
                    gs_=[]
                    for f_ in (0.5,0.25):                                       # keep only the most surprising f_ of pieces: share of the signal kept
                        k_=max(1,int(N_*f_)); ix_=torch.topk(cq_,k_).indices; gs_.append(float(sig_[ix_].sum())/max(tot_,1e-12))
                    km_=kq_-kq_.mean(); cm_=cq_-cq_.mean(); r_=float((km_*cm_).sum()/((km_.norm()*cm_.norm()).clamp_min(1e-12)))
                    print(f"    [replay signal] {N_} pieces sampled | KL per piece p10 {qs_[0]:.3f} p50 {qs_[1]:.3f} p90 {qs_[2]:.3f} | share of pieces {und_} | "
                          f"signal share: bottom 50% of pieces {bot50_:.3f}, top 20% {top20_:.3f} | keep the most surprising 50% / 25% of pieces -> {gs_[0]:.2f} / {gs_[1]:.2f} of the signal | "
                          f"corr(signal, surprise at experience) {r_:+.2f}")
                GSp_=model.__dict__.pop('_gsel',None)
                GKp_=model.__dict__.pop('_gkl',None)
                GPp_=model.__dict__.pop('_gprot',None)
                GRp_=model.__dict__.pop('_grho',None)
                SMp_=model.__dict__.pop('_simstat',None)
                if SMp_ and SMp_[0]>0: print(f"    [similar dreams] {cfg.dream_sim}: {SMp_[0]} rows chosen among {cfg.dream_sim_k} candidates | share of candidates the current step would harm {SMp_[1]/SMp_[0]:.2f} | chosen row changed {SMp_[3]/SMp_[0]:.2f}")
                if GRp_ and GRp_[0]>0: print(f"    [reducible gate] {GRp_[0]} calls | overlap with the surprise selection {float(GRp_[1])/GRp_[0]:.2f} | share of pieces the trace predicts WORSE than the weights (stale teacher) {float(GRp_[2])/GRp_[0]:.2f}")
                if GPp_ and GPp_[0]>0: print(f"    [protected gate] {GPp_[0]} calls | overlap with the KL selection {GPp_[1]/GPp_[0]:.2f}, with the surprise selection {GPp_[2]/GPp_[0]:.2f} | context novelty of the kept pieces x{GPp_[3]/GPp_[0]:.2f} the mean")
                if GKp_ and GKp_[0]>0: print(f"    [KL gate] {GKp_[0]} calls | overlap with the surprise selection {GKp_[1]/GKp_[0]:.2f} | share of the KL signal the surprise selection would keep {GKp_[2]/GKp_[0]:.2f}")
                if GSp_ and GSp_[0]>0: print(f"    [diversity gate] {GSp_[0]} calls | overlap with the surprise selection {GSp_[1]/GSp_[0]:.2f} | gradient energy kept: diversity {GSp_[2]/GSp_[0]:.2f}, surprise would keep {GSp_[3]/GSp_[0]:.2f}")
                EVp_=model.__dict__.pop('_evid',None)
                if EVp_ is not None: print(f"    [evidence targets] teacher's top byte differed from the verified replayed byte at {float(EVp_[0])/max(1.0,float(EVp_[1])):.4f} of teaching positions (blend w {cfg.cd_evid_w}{', all positions' if cfg.cd_evid_all else ''})"+(' - INACTIVE: with cd_bytes_all the conduit trains next-byte targets on the verified bytes, the blend is not used' if (getattr(cfg,'cd_bytes_all',False) and float(getattr(cfg,'cd_bytes_w',1.0))>=1.0) else ''))   # FIX (review): readout said it trains
                if st_: k_=max(1,st_[1]); print(f"    [conduit replay] KL {st_[0]/k_:.4f} | exact segments {st_[2]/k_:.2f} | tries/segment {st_[3]/k_:.2f} | true byte fidelity {st_[4]/k_:.2f} (diagnostic) | exactness test agrees with truth {st_[6]/k_:.3f} | eps {st_[5]/k_:.1e} over {st_[1]} steps"
                               +(f" | replay training on exact replays {st_[7]/max(1,st_[8])/LOG2:.4f} BPC ({int(st_[8])} steps)" if len(st_)>8 and st_[8] else "")+_rs_msg())
                if _HINT['calls']:                                                 # cg12b hints
                    tb_=max(1,_HINT['tot']); wr_=float(_HINT['wrong'])
                    print(f"    [hints] {float(_HINT['bytes'])/tb_:.3f} of replayed bytes hinted ({float(_HINT['pieces'])/(tb_/max(1,int(cfg.cd_seg))):.2f} of pieces) | "
                          f"teacher-forced wrong bytes {wr_/tb_:.4f}, of them hinted {float(_HINT['wcov'])/max(1.0,wr_):.2f} | {_HINT['calls']} calls"
                          +(" | REPLAY FIDELITY ALARM: over 5% of bytes wrong with the trace (cg14c Gutenberg 16.9%)" if wr_/tb_>0.05 else ""))
                    _HINT.update(calls=0,bytes=0,tot=0,wrong=0,wcov=0,pieces=0)
                if _HEMB['n'] or _HEMB['skip']:                                    # LAB embedded hints
                    tE_=max(1,_HEMB.get('tot',0))
                    print(f"    [embedded hints] {_HEMB['n']} calls ({_HEMB['skip']} skipped: trace not factored) | teacher-forced wrong bytes with the trace: written {_HEMB['w0']/tE_:.4f} -> "
                          f"strengthened {_HEMB['w1']/tE_:.4f} -> after {int(cfg.cd_hint_embed_rounds)} round(s) {_HEMB['wf']/tE_:.4f} | hinted wrong bytes made right by the first strengthening "
                          f"{_HEMB['fix']/max(1.0,_HEMB['hit']):.2f} | x{1+float(cfg.cd_hint_embed):g} at offsets {cfg.cd_hint_embed_off}; no hint bytes forced",flush=True)
                    _HEMB.update(n=0,w0=0,w1=0,wf=0,hit=0,fix=0,skip=0,tot=0,diag=None)
                if FOKS['skip']:
                    fz_=_fok_dom(dom_id)
                    es_=(_fsum(FOKS['estn'])/max(1.0,_fsum(FOKS['estd']))) if FOKS.get('estd') else (_fm(FOKS['est']) if FOKS['est'] else 0.0)
                    tr_=(_fsum(FOKS['trn'])/max(1.0,_fsum(FOKS['trd']))) if FOKS.get('trd') else (_fm(FOKS['true']) if FOKS['true'] else 0.0)
                    print(f"    [confidence gate] checks skipped {_fm(FOKS['skip']):.2f} | wrongly accepted: estimated {es_:.4f} "
                          f"(random checks), true {tr_:.4f} (diagnostic) | tries per piece {_fm(FOKS['k']):.2f} | threshold {float(fz_['thr']):.2f} | running exact share {float(fz_['exr'] if fz_.get('exrT') is None else fz_['exrT']):.2f}")
                if FOKS['runs']: print(f"    [exact runs] share of positions that teach {float(torch.stack([torch.as_tensor(v_,dtype=torch.float32).cpu() for v_ in FOKS['runs']]).mean()):.3f} (exact share {st_[2]/k_:.2f})")
                if cfg.cd_repair and _V97['calls']:                                 # v9e report
                    pr_=_fsum(_V97['pn'])/max(1.0,_fsum(_V97['pd'])); ag_=_fsum(_V97['ag'])/max(1.0,_fsum(_V97['agd'])); nchk_=max(1,sum(_V97['chk']))
                    fz2_=_fok_dom(dom_id)
                    print(f"    [v9.7 recall] positions that teach (verified) {_fm(_V97['vp']) if _V97['vp'] else 0:.3f} | repaired {_V97['rep']/nchk_:.2f} of checked pieces, "
                          f"{_fsum(_V97['imp'])/max(1,_V97['rep']):.2f} of them improved | verified positions truly correct {pr_:.4f} | verified prefix = true prefix {ag_:.3f} (checked pieces)"
                          +(f" | Gram/key disagreements {int(_fsum(_V97['dis']))}" if _V97['dis'] else "")+f" | key tolerance {float(model.__dict__.get('_cd_ktol',cfg.cd_keytol)):.3f}"
                          +(f" | floor weight mean {np.mean(_V97['fw']):.2f}, skipped calls {_V97['skip']} of {_V97['calls']} (running verified share {float(fz2_['vps']) if fz2_.get('vps') is not None else float('nan'):.2f})" if float(cfg.cd_floor_hi)>float(cfg.cd_floor_lo) else ""))
                    for k3_ in _V97: _V97[k3_]=[] if isinstance(_V97[k3_],list) else 0
                if float(cfg.cd_buf)>0 and _CBUF['calls']:
                    print(f"    [v9.7 buffer] {_CBUF['mix']/_CBUF['calls']:.1f} buffered windows per conduit step | from: "+(" ".join(f"{dn_(d_) if callable(globals().get('dn_')) else d_}:{c_}" for d_,c_ in sorted(_CBUF['src'].items())) or "-")
                          +" | buffer now: "+" ".join(f"{d_}:{len(e_['x'])}" for d_,e_ in sorted(_CBUF['d'].items())))
                    _CBUF.update(mix=0,calls=0,src={})
                for k2_ in list(FOKS): FOKS[k2_]=[]
        if cfg.mk_dream>0 and (cfg.mk_always or float(model.__dict__.get('_real_w',1.0))==0.0):   # MEMORY-KEYED DREAMS
            if step%max(1,int(cfg.mk_every))==0: mk_refresh(model,cfg,x,dom_id)
            loss=loss+cfg.mk_w*mk_loss(model,cfg,dom_id)
            if step==_L-1 and model.__dict__.get('_mk_stat'):
                st_=model.__dict__.pop('_mk_stat'); print(f"    [memory-keyed dreams] mean KL(teacher || student) {st_[0]/max(1,st_[1]):.4f} over {st_[1]} steps, pool {len(model.__dict__.get('_mk_pool',[]))}")
            if step==_L-1: model.__dict__['_mk_pool']=[]
        if hasattr(model,'_klw_opt'): model._klw_opt.zero_grad()
        _dt=locals().get('_dterm',0.0)
        _pt=locals().get('_pterm',None); _pterm=None; _pg=None
        with _rf('backward'): loss.backward()
        if _pg is not None:
            with torch.no_grad():
                R_=model.__dict__.setdefault('_decR',{}); dl_=[]
                for p_,g_ in _pg:
                    if g_ is None: continue
                    if cfg.dec_mode=='move' and p_.grad is not None: p_.grad.sub_(g_)
                    r_=float(g_.float().pow(2).mean()); R_[id(p_)]=r_ if id(p_) not in R_ else 0.99*R_[id(p_)]+0.01*r_
                    dl_.append((p_,g_.float()/math.sqrt(max(R_[id(p_)],1e-30))))
                model.__dict__['_decstep']=dl_
        _tmark(model,cfg,step,'forward+backward')
        if hasattr(model,'_klw_opt') and model._klw.grad is not None: model._klw_opt.step()
        xg=None
        if getattr(model,'_xcur',None):                  # d loss / d gain = <grad of weight tensor, earlier domain's delta>
            P_=_pmap(model); X=model._xcur; xg=torch.zeros_like(X['A'])
            with torch.no_grad():
                for jj,j in enumerate(X['olds']):
                    for kk,k in enumerate(X['keys']):
                        v_=model._spec[j].get(k)
                        if v_ is not None and P_[k].grad is not None: xg[jj,kk]=float((P_[k].grad*v_).sum())
        _dh_ids=set()
        _dsk_=model.__dict__.pop('_dstep_k',None)                                       # dream step (the k steps it stands for) or None
        _cth_=1.0
        if _dsk_ is not None and cfg.dream_clip_pass:                                   # cg14b: DREAM-STEP CLIP THRESHOLD per pass (GPU: dream steps sat at
            _cth_=float(cfg.dream_clip_pass[min(int(model.__dict__.get('_pass_i',0)),len(cfg.dream_clip_pass)-1)])   #   norm 1.7-6.2, clipped 100%)
        with _rf('grad_clip'): _gn_=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and id(p) not in _dh_ids],_cth_)
        if n_old>0:                                                                     # CLIP STATISTICS, dream vs other steps (on device: no sync)
            cs_=model.__dict__.setdefault('_clipst',{}); e_=cs_.setdefault('d' if _dsk_ is not None else 'r',[0.0,0.0,0])
            e_[0]=e_[0]+_gn_.detach().float(); e_[1]=e_[1]+(_gn_.detach().float()>_cth_).float(); e_[2]+=1
        if _dh_ids and hasattr(model,'draft_heads'): torch.nn.utils.clip_grad_norm_(list(model.draft_heads.parameters()),1.0)   # draft heads clipped on their own
        mg=None
        if n_old>0 and meta is not None:
            with torch.no_grad(): apply_meta_gate(model, meta, prog)
        _gopen_=(cfg.write_gate_open and mg is None and not cfg.route_writes and not cfg.shared_head and not cfg.block_write and not cfg.block_auto
                 and not cfg.surprise_mod and not cfg.dual_rate and not cfg.dream_rollback and not (cfg.shunt and cfg.shunt_mode=='subspace')
                 and not (cfg.shunt and cfg.shunt_mode=='all' and not cfg.shunt_lazy))
        _gneed_=not _gopen_ or (cfg.shunt and cfg.shunt_mode=='all' and cfg.shunt_lazy and model.__dict__.get('_vstart') is None)
        if (cfg.use_write_threshold or mg is not None) and n_old>=cfg.gate_start_domain and _gneed_:   # gates all open: the per-step snapshot /
                                                                                                        #   blend is a no-op - taken only when the lazy fast store needs its segment start
            with _rf('gate_snapshot'): snaps,elig=snapshot_threshold(model, cfg, mg)
        else:
            snaps,elig=[],1.0
        ngp_=None
        stv_=None
        pjv_=None
        mu_out_=muon_prep(model,cfg)
        tl_snap_=None
        hk_=None
        if cfg.head_dream_k and _dsk_ is not None and _dsk_>1:                           # OLD HEADS: Adam step x k on a dream step standing for k
            hk_=[(p_,p_.detach().clone()) for d_ in olds if d_!=dom_id for p_ in model.heads[d_].parameters() if p_.grad is not None]
        _cdf_=model.__dict__.pop('_cdf',None); cds_=None
        if _cdf_ is not None and _cdf_<1.0 and _dsk_ is None:                             # CONDUIT-ONLY step: the whole update scaled by cd_lr
            cds_=[(p_,p_.detach().clone()) for p_ in model.parameters() if p_.requires_grad]
        _dsc_=bool(_ds_on and step%max(1,int(cfg.dstep_every))==int(cfg.dstep_every)-1)
        if _dsc_:                                                                       # DIRECT STEP: held-out gradient BEFORE the update (same rows after)
            _t_ds=time.perf_counter(); _dsP=[p_ for p_ in model.parameters() if p_.requires_grad]
            _dsseed=1000003*step+17+97*int(dom_id); _dsg0,_dsJ0=_dstep_grad(model,cfg,dom_id,olds,_dsseed,_dsP); _dsth=[p_.detach().clone() for p_ in _dsP]
            _ds_t+=time.perf_counter()-_t_ds
        with _rf('adam_step'): opt.step()
        if hk_:
            with torch.no_grad():
                for p_,o_ in hk_: p_.add_(p_-o_,alpha=float(_dsk_)-1.0)
        if mu_out_: muon_apply(mu_out_,opt.param_groups[0]['lr']*cfg.muon_lr)
        if model.__dict__.get('_decstep'):                                             # DECOUPLED PROTECTION step (linear restoring force)
            with torch.no_grad():
                a_=opt.param_groups[0]['lr']*cfg.muon_lr*cfg.dec_prot*0.2
                for p_,d_ in model.__dict__.pop('_decstep'): p_.add_(d_.to(p_.dtype),alpha=-a_)
        if cfg.pmuon and step==_L-1: kf_store(model,dom_id)
        head_center_(model)
        if cfg.film_k>0 and model.__dict__.get('_film_frozen') is not None:             # only the current domain's code learns
            with torch.no_grad():
                fz_=model._film_frozen; keep_=model.film_codes[dom_id].clone(); model.film_codes.copy_(fz_); model.film_codes[dom_id]=keep_
        if cfg.varpro_old and step==_L-1 and n_old>0: print(f"    [varpro old] {model.__dict__.pop('_vpo_n',0)} old-head refits this visit")
        if (cfg.sleep_eps>0 or cfg.sleep_rel>0) and step==_L-1 and n_old>0:
            sc_=model.__dict__.pop('_sleepC',[0,0]); P_=model.__dict__.get('_sleepP',{}); SE_=model.__dict__.get('_sleepE',{})
            dr_=model.__dict__.pop('_drows',[0,0])
            print(f"    [sleep pressure] dreams on {sc_[0]}/{_L} steps, {dr_[0]/max(1,dr_[1]):.1f} rows per dream step | pressure / smoothed KL per old domain: "+(f"fixed {cfg.sleep_fixed} (open loop)"))
        if cfg.noise_diag>0 and step==_L-1 and n_old>0: noise_report(model,cfg,dom_id)
        if step==_L-1 and n_old>0 and model.__dict__.get('_clipst'):
            cs_=model.__dict__.pop('_clipst'); f_=lambda k_: (lambda e_: f"{float(e_[0])/max(1,e_[2]):.3f} mean norm, clipped {float(e_[1])/max(1,e_[2]):.0%} of {e_[2]}" if e_ else "none")(cs_.get(k_))
            print(f"    [grad clip] dream steps: {f_('d')} | other steps: {f_('r')} (threshold 1.0"+(f", dream steps {float(cfg.dream_clip_pass[min(int(model.__dict__.get('_pass_i',0)),len(cfg.dream_clip_pass)-1)]):.1f}" if cfg.dream_clip_pass else '')+")")
        if cfg.kf_dream>0 and step==_L-1 and n_old>0:
            bt_=model.__dict__.pop('_kfbt',[0.0,0]); dc_=model.__dict__.get('_kf_dacc',{}); print(f"    [live keys] {bt_[1]} whitening builds took {bt_[0]:.1f}s ({'eigh'}) | rebuilt {model.__dict__.pop('_kf_drb',0)} times from the dream rows' current input factors ({len(dc_)} weights, {max([e_['n'] for e_ in dc_.values()] or [0])} dream positions each, mix {cfg.kf_dream})")
        if tl_snap_:
            with torch.no_grad():
                for p_,o_ in tl_snap_: p_.add_(p_-o_,alpha=cfg.trunk_lr_mult-1.0)
        if stv_ is not None: stiff_post(stv_)
        if pjv_: proj_post(pjv_,model,step)
        ngp_step(model,cfg,opt.param_groups[0]['lr'])
        if cds_:
            with torch.no_grad():
                for p_,o_ in cds_: p_.copy_(o_+_cdf_*(p_-o_))
            cds_=None
        if _dsc_:                                                                       # 1-D quadratic of J along the update u just taken (s = 1)
            _t_ds=time.perf_counter()
            _dsg1,_dsJ1=_dstep_grad(model,cfg,dom_id,olds,_dsseed,_dsP)
            with torch.no_grad():
                u_=[p_.detach()-q_ for p_,q_ in zip(_dsP,_dsth)]
                d0_=float(sum((a_.float()*b_.float()).sum() for a_,b_ in zip(_dsg0,u_))); d1_=float(sum((a_.float()*b_.float()).sum() for a_,b_ in zip(_dsg1,u_))); c_=d1_-d0_
            if d0_>=0: sst_=0.5                                                         #   not downhill on the held-out score: shrink
            elif c_<=1e-12: sst_=2.0                                                    #   no measurable curvature: still descending at s = 1 -> grow
            else: sst_=min(2.0,max(0.5,-d0_/c_))
            model.__dict__['_dsm']=float(min(4.0,max(0.25,model.__dict__['_dsm']*sst_**float(cfg.dstep_k))))
            model._dsS.append(sst_); model._dsM.append(model._dsm); model._dsJ.append(float(_dsJ1-_dsJ0))
            del _dsg0,_dsg1,_dsth,u_; _ds_t+=time.perf_counter()-_t_ds
        if cfg.cdet_log and step==_L-1: _cdet_report(model,cfg)
        if _ds_on and step==_L-1 and model.__dict__.get('_dsS'):
            S_=np.array(model._dsS); M_=np.array(model._dsM); J_=np.array(model._dsJ)
            print(f"    [direct step] {len(S_)} checks | best step length median {np.median(S_):.2f}x the step taken (at the 2x cap {np.mean(S_>=2.0):.0%}, at the 0.5x cap {np.mean(S_<=0.5):.0%}) | "
                  f"lr multiplier at the end {M_[-1]:.2f} (mean over the visit {M_.mean():.2f}) | held-out objective change per checked step {J_.mean()/LOG2:+.5f} bits | {_ds_t:.0f}s")
        _ek_=max(1,int(cfg.ema_every))
        if cfg.ema_tail>0 and model.__dict__.get('_ema') is not None and ((step+1)%_ek_==0 or step==_L-1):
            with torch.no_grad():
                P_,E_,d_=model._ema
                pw_=((1.0-d_)/cfg.ema_pull_rel) if cfg.ema_pull_rel>0 else 0.0     # v9i: <= 0 -> tail averaging without the pull
                if _ek_>1: d_=d_**_ek_; pw_=1.0-(1.0-pw_)**_ek_                 # k steps at once: same decay horizon
                try: torch._foreach_lerp_(E_,[p_.detach() for p_ in P_],1.0-d_)      # one fused update (GPU: few kernels)
                except Exception:
                    for p_,e_ in zip(P_,E_): e_.mul_(d_).add_(p_.detach(),alpha=1.0-d_)
                if pw_>0:                                                          # BIDIRECTIONAL: weights pulled towards their running average
                    try: torch._foreach_lerp_([p_.data for p_ in P_],E_,pw_)      # one fused update (GPU)
                    except Exception:
                        for p_,e_ in zip(P_,E_):
                            if p_.shape==e_.shape: p_.data.lerp_(e_,pw_)
            if step==_L-1 and model.__dict__.get('_ngp') is not None: print("    [private n-grams] table norms per domain: "+' '.join(f"{_lab(cfg,d_)} {sum(float(E.weight.detach().norm()) for E in model._ngp[d_]):.2f}" for d_ in sorted(set(olds)|{dom_id})))
            if step==_L-1 and model.__dict__.get('_ngpi') is not None: print("    [private input rows] table norms per domain: "+' '.join(f"{_lab(cfg,d_)} {sum(float(E.weight.detach().norm()) for E in model._ngpi[d_]):.2f}" for d_ in sorted(set(olds)|{dom_id})))
        if ngp_ is not None:
            with torch.no_grad():
                for W_,W0_,sc_ in ngp_:
                    if sc_ is not None: W_.copy_(W0_+(W_-W0_)*sc_[:,None])
        if getattr(model,'ngram',None) is not None and cfg.ngram_sparse: ngram_lazy_step(model,cfg,opt.param_groups[0]['lr'])
        if model.__dict__.get('_slmul') is not None:
            mu_=model.__dict__.pop('_slmul')
            for g_ in opt.param_groups: g_['lr']/=mu_
        if _snrm_!=1.0:
            for g_ in opt.param_groups: g_['lr']/=_snrm_
        if model.__dict__.get('_sldb') is not None: cfg.dream_batch=model.__dict__.pop('_sldb')
        apply_update_gate(snaps, None, cfg.og_in, cfg.og_orth,
                          None, cfg.og_orth_boost,
                          getattr(model,'_og_common',None), cfg.og_common,
                          getattr(model,'_og_out',None),
                          (None))   # threshold x meta write strength, applied to the UPDATE
        _tmark(model,cfg,step,'optimiser+gates')
        if cfg.shunt and cfg.shunt_mode=='all' and (n_old>0):     # FAST STORE: all new learning tracked in the shunt
            Sd=model.__dict__.setdefault('_shunt',{})
            if model.__dict__.get('_vstart') is None: model.__dict__['_vstart']={id(p_):(p_,old_.clone()) for p_,old_,_,_ in snaps}
        if xg is not None:                                 # Adam step on the cross gains; the change enters the live weights
            X=model._xcur; X['t']+=1; X['m'].mul_(0.9).add_(0.1*xg); X['v'].mul_(0.999).add_(0.001*xg*xg)
            mh=X['m']/(1-0.9**X['t']); vh=X['v']/(1-0.999**X['t'])
            An=(X['A']-cfg.cross_lr*mh/(vh.sqrt()+1e-8)).clamp(cfg.cross_min,cfg.cross_max); dA=An-X['A']; X['A']=An
            P_=_pmap(model)
            with torch.no_grad():
                for jj,j in enumerate(X['olds']):
                    for kk,k in enumerate(X['keys']):
                        if dA[jj,kk]!=0 and k in model._spec[j]: P_[k].add_(float(dA[jj,kk])*model._spec[j][k])
        if cfg.shunt_lazy and cfg.shunt and n_old>0 and model.__dict__.get('_vstart') and ((step+1)%cfg.shunt_every==0 or step+1==_L):
            with torch.no_grad(): model._shunt={k:(p_.data-o_) for k,(p_,o_) in model._vstart.items()}   # A5: shunt = weights - segment start
            model.__dict__['_vstart']=None
        if cfg.shunt and cfg.shunt_consol=='continuous' and n_old>0 and getattr(model,'_shunt',None) and not cfg.topic_mode:
            do=False
            do=(step+1)%cfg.shunt_every==0 or step+1==_L
            if do:
                model._shunt={}                                   # MERGE: shunt absorbed into the main weights
                model.__dict__.setdefault('_merge_log',[]).append((dom_id,step+1))
        if getattr(model,'_reserve',None) and not (cfg.block_extra and cfg.block_extra_mode=='fresh'):
            for b,U in zip(model.blocks,model._uniq): U.update(b._uh,b._um,cfg)
            if (step+1)%cfg.recruit_every==0:
                before=adam_capture(opt,model); ev=recruit(model,cfg)
                if ev:
                    opt=adam_rebuild(model,cfg.lr,before)
                    if getattr(model,'_teach',None) is not None: build_teachers(model,cfg)
                    L_=model._recruit_log.setdefault(dom_id,[]); L_.extend(ev)
        orth_edge=orth is not None and (step+1)%cfg.log_every==0
        if orth_edge: orth.end(model, (x,y), dom_id)     # before any widening, so the window's width is fixed
        if (step+1)%cfg.log_every==0:
            if getattr(model,'_curve_val',None) is not None:
                model.__dict__.setdefault('_curve',{}).setdefault(dom_id,[]).append(bpc(model,model._curve_val,dom_id,n=5))
            mgm=float(np.mean([g[0].mean().item() for g in mg])) if mg is not None else float('nan')
            ee=getattr(model,'_emb_elig',None); rr=getattr(model,'_route_mean',None); ce=getattr(model,'_ch_elig',None)
            msg=f"    step {step+1}: "+("")+f"(eligible={elig:.2f}"+(f", ch_eligible={ce:.2f}" if ce is not None else "")+(f", emb_eligible={ee:.2f}" if ee is not None else "")+(f", route={rr:.2f}" if rr is not None else "")+(f", distinct={float(model.blocks[0].cur_distinct):.2f}" if model.blocks[0].cur_distinct is not None else "")+")"
            lim=(cfg.min_eligible)
            if getattr(model,'_dmax',None):
                msg+=" [dream KL max: "+" ".join(f"{_lab(cfg,d)} {float(v):.3f}" for d,v in sorted(model._dmax.items()))+"]"; model._dmax={}
            msg+=(f" [uniqueness ratio "+" ".join(f"{U.ratio():.2f}" for U in model._uniq)
                  +f" | active state {model.blocks[0].d_state} MLP {model.blocks[0].mlp_hidden}]")
            print(msg)
        if orth_edge and step+1<_L: orth.begin(model, prior_buffer, (x,y), dom_id)
        _tmark(model,cfg,step,'rest')
    if cfg.ema_tail>0 and model.__dict__.get('_ema') is not None:     # TAIL AVERAGING: the visit ends on the averaged weights
        with torch.no_grad():
            P_,E_,_=model.__dict__.pop('_ema')
            for p_,e_ in zip(P_,E_):
                if p_.shape==e_.shape: p_.copy_(e_)
    if getattr(model,'_ng_cur',None) is not None:                     # ROW PROTECTION: this domain's usage shares join the old usage
        for c_,o_ in zip(model._ng_cur,model._ng_old):
            if float(c_.sum())>0: o_+=c_/c_.sum()
            c_.zero_()
        model.__dict__.pop('_ng_sc',None)                             # protection scales change only here
    sa_=model.__dict__.pop('_surpacc',None)
    if sa_:
        f_=lambda i,ph,j: (float(sa_[(i,ph)][j])/sa_[(i,ph)][2]) if (i,ph) in sa_ else float('nan')
        print("  [surprise] per block, first 100 / last 100 steps: write scale "+" ".join(f"b{i} {f_(i,0,0):.2f}/{f_(i,1,0):.2f}" for i in range(len(model.blocks)))
              +" | memory prediction-error ratio "+" ".join(f"b{i} {f_(i,0,1):.3f}/{f_(i,1,1):.3f}" for i in range(len(model.blocks))))
    rl_=model.__dict__.pop('_relaxlog',None)
    if rl_ is not None: print(f"  [residual gates] dream coverage of the top block's units (activity on dreams / long-run activity, capped at 1): state {rl_[0]:.2f}, MLP {rl_[1]:.2f}")
    C_=model.__dict__.pop('_conf_acc',None)
    if C_: print("  [protection conflict] mean cos(old domain's protection gradient, current-loss gradient): "+" ".join(f"{_lab(cfg,d)} {float(v[0])/v[1]:+.2f}" for d,v in C_.items()))
    S_=model.__dict__.get('_share_last')
    if cfg.share_mode and S_: print(f"  [dream shares] ({cfg.share_mode}) at the visit end: "+" ".join(f"{_lab(cfg,d)} {v:.2f}" for d,v in S_.items()))
    A_=model.__dict__.pop('_borrow_acc',None)
    if A_: print("  [borrow alignment] mean cos(borrowing gradient, own-loss gradient) per old domain: "+" ".join(f"{_lab(cfg,d)} {float(v[0])/v[1]:+.2f}" for d,v in A_.items()))
    R_=model.__dict__.get('_borrow_r')
    if R_ is not None and (cfg.borrow_w>0 or cfg.em_w>0) and olds:
        print("  [relevance] old domains' relevance to this domain at the visit end (EMA of (n_old+1) x posterior, cap 1): "
              +" ".join(f"{_lab(cfg,d)} {float(R_[d]):.2f}" for d in olds))
    wa_=model.__dict__.pop('_lwfwacc',None)
    if wa_ is not None:
        print(f"  [posterior KL] mean weight per old head on this domain's text (1 = unweighted): "
              +" ".join(f"{_lab(cfg,d)} {float(wa_[0][i_])/wa_[2]:.2f}" for i_,d in enumerate(olds))+f" | posterior of the current domain {float(wa_[1])/wa_[2]:.2f}")
    ca_=model.__dict__.pop('_ctxacc',None)
    if ca_ is not None:
        print("  [context] responsibility of each old memory for this domain's text (mean causal posterior): "
              +" ".join(f"{_lab(cfg,d)} {float(ca_[0][i_])/ca_[1]:.2f}" for i_,d in enumerate(olds)))
    fa_s=model.__dict__.pop('_fastat',None)
    if fa_s:
        print("  [feature anchor] block output on dreams vs frozen teacher, mean relative error: "+" ".join(f"b{li_} {float(a_[0])/max(a_[1],1):.5f}" for li_,a_ in sorted(fa_s.items()))+f" (weight {cfg.feat_anchor} per block)")
    os_=model.__dict__.pop('_opstat',None)
    if os_ is not None and os_[2]:
        print(f"  [on-policy dreams] mean per-token KL to the teacher on LIVE-model samples {float(os_[0])/os_[2]:.4f} vs teacher samples {float(os_[1])/os_[2]:.4f}")
    pe_=model.__dict__.pop('_prio_ess',None)
    if pe_ is not None: print(f"  [dream priority] effective share of each pool drawn from (1 = uniform): {float(pe_[0])/pe_[1]:.2f}")
    ka_=model.__dict__.pop('_klacc',None)
    if ka_ is not None:                                  # fused KL statistics -> per old domain (as the loop records them)
        for i_,d in enumerate(olds):
            kst=model.__dict__.setdefault('_klstat',{}).setdefault(d,[0.0,0.0,0]); kst[0]+=ka_[0][i_]; kst[1]+=cfg.shunt_old_weight*ka_[1]; kst[2]+=ka_[1]
    if cfg.timing and model.__dict__.get('_tm',{}).get('n'):
        T_=model._tm; n_=T_['n']; ph=['prep','forward+backward','optimiser+gates','rest']; tot=sum(T_['acc'].get(k,0.0) for k in ph)/n_
        wall=time.perf_counter()-_t_dom
        model.__dict__.setdefault('_timing_log',{})[dom_id]={'ms':1000*tot,'wall':wall,'dream':model.__dict__.get('_tm_dream',0.0),
                                                            **{k:1000*T_['acc'].get(k,0.0)/n_ for k in ph}}
        print("  [timing] ms per step (sampled, synchronised): "+" | ".join(f"{k} {1000*T_['acc'].get(k,0.0)/n_:.0f}" for k in ph)
              +f" | total {1000*tot:.0f} || domain wall {wall:.0f}s ({_L/max(wall,1e-9):.1f} steps/s): regular steps {_L*tot:.0f}s,"
              +f" periodic work {max(0.0,wall-_L*tot):.0f}s (dream sampling {model.__dict__.get('_tm_dream',0.0):.0f}s, evaluation, gate refreshes)"
              +(" [bf16]" if cfg.amp=='bf16' else ""))
        sd_,sn_=T_.get('sd',[0.0,0]),T_.get('sn',[0.0,0]); nd_=model.__dict__.get('_dcount',0)-T_.get('dc0',0)
        if sd_[1] and sn_[1]:
            md_,mn_=1000*sd_[0]/sd_[1],1000*sn_[0]/sn_[1]
            print(f"  [timing] whole step, sampled: WITHOUT dreams {mn_:.0f} ms ({sn_[1]} steps) | WITH dreams {md_:.0f} ms ({sd_[1]} steps, +{md_-mn_:.0f} ms) || this visit: {nd_} dream steps -> training steps ~{(nd_*md_+(_L-nd_)*mn_)/1000:.0f}s, other work ~{max(0.0,wall-(nd_*md_+(_L-nd_)*mn_)/1000):.0f}s")
        model.__dict__['_tm']={'last':None,'acc':{},'n':0,'dc0':model.__dict__.get('_dcount',0)}
    if dom_id in tuple(cfg.state_spec_diag or ()) and n_old>0 and not cfg.topic_mode:
        was_=model.training; model.eval()
        try: state_spectrum_diag(model, cfg, olds)
        except Exception as e_: print(f"  [state spectrum] diagnostic skipped ({type(e_).__name__}: {e_})")   # never stops a run
        model.train(was_)
    sst_=model.__dict__.pop('_strandstat',None); scn_=model.__dict__.pop('_strandcnt',None)
    if scn_:
        tot_=sum(scn_.values()); top_=sorted(scn_.items(),key=lambda z:-z[1])[:6]
        print(f"  [strands] {cfg.strand_kind}, {tot_} generated; most drawn pairs ({cfg.strand_pair_mode}): "+" ".join(f"{_lab(cfg,a)}>{_lab(cfg,b)} {c/tot_:.0%}" for (a,b),c in top_)
              +(f" | KL before / after the switch {float(sst_['a'])/sst_['n']:.4f} / {float(sst_['b'])/sst_['n']:.4f}" if sst_ and sst_['n'] else "")
              +(f" | agreement-dream KL {float(sst_['ag'])/sst_['agn']:.4f}" if sst_ and sst_.get('agn') else "")
              +(" | old-subspace anchor rel. error "+" ".join(f"b{li} {float(v[0])/max(v[1],1):.4f} ({v[2]} dims)" for li,v in sorted(sst_['anc'].items())) if sst_ and sst_['anc'] else ""))
    tb_=None
    if tb_: print("  [byte-type weights] real share / dream share per byte type (clipped), per old domain: "
                  +" | ".join(f"{_lab(cfg,d)} "+" ".join(f"{_BYTE_NAMES[c][:3]} {float(v[c]):.2f}" for c in range(len(_BYTE_NAMES)-1)) for d,v in sorted(tb_.items())))
    dsa_=model.__dict__.pop('_dsteer_acc',None)
    if dsa_: print("  [drift-steered dreams] mean dream weight per old domain (fresh-chunk KL / mean, ^%.1f, clipped): "%cfg.dream_steer
                   +" ".join(f"{_lab(cfg,d)} {float(v[0])/max(v[1],1):.2f}" for d,v in sorted(dsa_.items())))
    shs_=model.__dict__.pop('_shstat',None)
    if shs_: print("  [similarities only] shared subspace (used by >= %.0f%% of old domains) per protected block, mean dims: "%(100*cfg.share_thr)
                   +" ".join(f"b{li} {a_[0]/max(a_[1],1):.1f}/{model.blocks[li].ln.weight.shape[0]}" for li,a_ in sorted(shs_.items()))
                   +f" | specific component x{cfg.share_spec}")
    bs=[getattr(b,'_bstat',None) for b in model.blocks]
    if any(bs):
        row=[(float(x[0])/x[2], float(x[1])/x[2]) if x else (float('nan'),float('nan')) for x in bs]
        model.__dict__.setdefault('_beta_log',{})[getattr(model,'_visit',dom_id)]=row
        print("  [reflections] share of writes with beta > 1 / mean beta, per block: "+" | ".join(f"b{i} {a:.2f}/{m:.2f}" for i,(a,m) in enumerate(row)))
        for b in model.blocks: b._bstat=None
    if imp_base is not None:     # restore pre-domain importance; run() then adds the full increment
        with torch.no_grad():
            for (o_,nm),b0 in zip(_imp_bufs(model),imp_base):
                cur=getattr(o_,nm); setattr(o_,nm,torch.cat([b0,torch.zeros(cur.numel()-b0.numel(),device=cur.device)]))
    return opt

@torch.no_grad()
def calibrate_affinity(model, cfg, n_domains, n_batches=6):
    """Right after the lock: run held-out-free calibration batches from each trained domain
    (fixed generator, no global RNG use) and set every unit's affinity to the drive-weighted
    distinctness of that content ('the position that activated it most')."""
    was=model.training; model.eval()
    for b in model.blocks:
        dev=b.unit_aff.device
        b.calib={'s_w':torch.zeros(b.d_state,device=dev),'s_wd':torch.zeros(b.d_state,device=dev),
                 'm_w':torch.zeros(b.mlp_hidden,device=dev),'m_wd':torch.zeros(b.mlp_hidden,device=dev)}
    for q in range(n_domains):
        get=make_loader(stream_bytes(PILE_DOMAINS[q],1000+q,4_000_000),cfg,fixed_seed=900+q)
        for _ in range(n_batches):
            x,_=get(); model(x,q)
    for b in model.blocks:
        c=b.calib
        for w,wd,aff in ((c['s_w'],c['s_wd'],b.unit_aff),(c['m_w'],c['m_wd'],b.mlp_aff)):
            ok=w>0.05*w.max()
            aff[ok]=(wd[ok]/w[ok]).clamp(0,1)
        b.calib=None
    a=model.blocks[0].mlp_aff; asg=a[a>=0]
    print(f"  [affinity calibrated] assigned {float((a>=0).float().mean()):.2f} of units | "
          f"affinity mean {float(asg.mean()) if len(asg) else float('nan'):.2f} range "
          f"[{float(asg.min()) if len(asg) else 0:.2f},{float(asg.max()) if len(asg) else 0:.2f}]")
    if was: model.train()

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    'FOKS': 'platypus.conduit',
    '_CBUF': 'platypus.conduit',
    '_CDR': 'platypus.losses',
    '_CDT': 'platypus.conduit',
    '_HINT': 'platypus.conduit',
    '_HEMB': 'platypus.conduit',
    '_PRF': 'platypus.profiling',
    '_V97': 'platypus.conduit',
    '_cd_lrf': 'platypus.losses',
    '_cdg_msg': 'platypus.conduit',
    '_fm': 'platypus.conduit',
    '_fok_dom': 'platypus.conduit',
    '_fsum': 'platypus.conduit',
    '_rf': 'platypus.profiling',
    '_rs_msg': 'platypus.conduit',
    'bpc': 'platypus.inference.store',
    'cd_loss': 'platypus.losses',
    'inf_store_add': 'platypus.inference.sentences',
    'mk_loss': 'platypus.conduit',
    'mk_refresh': 'platypus.conduit',
    'ngp_step': 'platypus.inference.need',
    'recon_loss': 'platypus.conduit',
    'recon_loss_mark': 'platypus.conduit',
}
