"""Conduit losses, replay diagnostics, reconstruction fidelity and oracle readouts."""
import math
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from platypus.common import (
    DEVICE, LOG2,
)
from platypus.config import (
    PILE_DOMAINS,
)
from platypus.private import (
    _BUFDET, _HCAP, _NGPD, _XDG,
)
from platypus.layers import (
    _ADPT, _CARRY, _ORACLE, _PMEM, _RC_GRAD,
)
from platypus.ngrams import (
    _SEG,
)
from platypus.conduit import (
    FOKS, _CBUF, _CDCONF, _CDT, _INC, _MIS32, _RESET, _V97, _carry_load, _carry_save, _cbuf_take, _cd_cat, _cd_encode, _cd_encode_chk, _cd_fb, _cd_g,
    _cd_gen, _cd_hints, _cd_kdist, _cd_mis, _cd_rows, _cd_slices, _cd_vcheck, _cd_weights, _cdt, _fok_dom, _fok_thr_gpu, _nrem_loss, _prot_minv,
    _recon_loss_mark, _rp_tags, _rs_pick, _sig_update, _store_score, _uh_load, _uh_save,
)
from platypus.inference.store import (
    bpc,
)
from platypus.inference.sentences import (
    inf_store_add,
)

def _nrem_loss_v1(model,cfg,dom):
    """NREM v1: re-teach the visit's reservoir windows (stored bytes + stored teacher targets), highest remaining signal first; retire consolidated ones.
    nrem_mix>0: part of each batch comes from other domains' kept samples (cross-domain NREM), each taught under its own domain."""
    RV_=model.__dict__.get('_nres'); z0_=next(model.parameters()).sum()*0.0
    td_=int(model.__dict__.get('_cur_dom',dom))
    if not RV_ or RV_.get('td')!=td_: return z0_
    k_=int(cfg.nrem_k); BK_=model.__dict__.get('_nres_bank',{}) if float(cfg.nrem_mix)>0 else {}
    oth_=[e_ for d_,l_ in BK_.items() if d_!=td_ for e_ in l_]
    ko_=min(len(oth_),int(round(k_*float(cfg.nrem_mix)))) if oth_ else 0
    if cfg.nrem_mix_add: k_=k_+ko_
    if cfg.nrem_own_sel=='rand' and RV_['e']:
        rs3_=model.__dict__.setdefault('_nres_rs3',np.random.RandomState(int(cfg.seed)+919)); own_=[RV_['e'][i_] for i_ in rs3_.choice(len(RV_['e']),min(len(RV_['e']),k_-ko_),replace=False)]
    else: own_=sorted(RV_['e'],key=lambda e_:-e_['v'])[:k_-ko_]
    if len(own_)<k_-ko_: ko_=min(len(oth_),k_-len(own_))
    if ko_>0:
        if cfg.nrem_mix_sel=='rand':
            rs_=model.__dict__.setdefault('_nres_rs2',np.random.RandomState(int(cfg.seed)+717)); osel_=[oth_[i_] for i_ in rs_.choice(len(oth_),ko_,replace=False)]
        else: osel_=sorted(oth_,key=lambda e_:-e_['v'])[:ko_]
    else: osel_=[]
    sel_=own_+osel_; OWN_=set(id(e_) for e_ in own_)
    if not sel_: return z0_
    L=int(cfg.cd_seg); kb_=int(cfg.bank_targets); C=sel_[0]['x'].numel()//L; tot_=sum(e_['w'].numel() for e_ in (own_ if (cfg.nrem_mix_add and own_) else sel_)); loss_=z0_; drop_=set()
    for d_ in sorted(set(e_['d'] for e_ in sel_)):
        g_=[e_ for e_ in sel_ if e_['d']==d_]
        with torch.no_grad():
            xb_=torch.stack([e_['x'] for e_ in g_]); wl_=torch.stack([e_['w'] for e_ in g_]).float(); vml_=(wl_>0).float()
            V_=torch.stack([e_['tv'] for e_ in g_]).float(); I_=torch.stack([e_['ti'] for e_ in g_]).long()
            rest_=(1-V_.exp().sum(-1,keepdim=True)).clamp(min=1e-8)/(256-kb_); TG_=rest_.log().expand(*V_.shape[:-1],256).clone(); TG_.scatter_(-1,I_,V_)
            qy_=torch.stack([e_['q'] for e_ in g_]).float()
        sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
        try: lp_=F.log_softmax(model(xb_[:,:-1],d_).float(),-1)
        finally: _carry_load(sv4_); _uh_load(model,suh4_); _NGPD['on']=False
        if cfg.nrem_tgt=='bytes': loss_=loss_-(lp_.gather(-1,xb_[:,1:,None]).squeeze(-1)*wl_).sum()/max(1,tot_)
        else: loss_=loss_+((TG_.exp()*(TG_-lp_)).sum(-1)*wl_).sum()/max(1,tot_)
        with torch.no_grad():
            rn_=_store_score(qy_,lp_.gather(-1,xb_[:,1:,None]).squeeze(-1),vml_,C,L).tolist()
            for e_,r_ in zip(g_,rn_):
                o_=id(e_) not in OWN_; e_['n']+=1; RV_['rep_o' if o_ else 'rep']=RV_.get('rep_o' if o_ else 'rep',0)+1
                if float(cfg.nrem_retire)>=0 and r_<=max(1e-4,float(cfg.nrem_retire)*e_['v0']): drop_.add(id(e_)); RV_['ret_o' if o_ else 'ret']=RV_.get('ret_o' if o_ else 'ret',0)+1
                else: e_['v']=r_
    if drop_:
        RV_['e']=[e_ for e_ in RV_['e'] if id(e_) not in drop_]
        for d_ in list(BK_): BK_[d_]=[e_ for e_ in BK_[d_] if id(e_) not in drop_]
    return loss_
def _cd_lrf(cfg,x):
    if float(cfg.cd_lr)>=0: return float(cfg.cd_lr)
    C=int(cfg.cd_chain) if int(cfg.cd_chain)>0 else max(1,x.shape[1]//int(cfg.cd_seg))
    return min(1.0,int(cfg.cd_rows)*min(int(cfg.cd_seg)*C,x.shape[1])/float(x.numel()))
_CDR=[0,0,0.0,0]                                                           # v9: reused conduit steps, conduit steps, teaching-share sum / count
_CDP={'n':0,'todo':None}
def _cd_will_reuse(model,cfg,dom):
    rc_=model.__dict__.get('_cd_cache')
    return bool(int(cfg.cd_reuse)>1 and rc_ is not None and rc_['dom']==dom and rc_['k']<int(cfg.cd_reuse)-1)
def cd_loss(model,cfg,x,dom):
    _CDP['n']+=1
    if int(getattr(cfg,'cd_sync_find',-1))==_CDP['n'] and torch.cuda.is_available():   # v9.8 SYNC FINDER: every CPU wait in one conduit call, by code line
        import warnings as _wn
        prev_=torch.cuda.get_sync_debug_mode(); torch.cuda.set_sync_debug_mode(1)
        try:
            with _wn.catch_warnings(record=True) as wl_:
                _wn.simplefilter('always')
                with _rp_tags(model): r_=_cd_loss(model,cfg,x,dom)
        finally: torch.cuda.set_sync_debug_mode(prev_)
        cnt_={}
        for w_ in wl_:
            if 'synchroniz' in str(w_.message).lower(): kk_=(os.path.basename(w_.filename),w_.lineno); cnt_[kk_]=cnt_.get(kk_,0)+1
        print(f"  [sync finder] conduit call {_CDP['n']}: {sum(cnt_.values())} CPU waits for the GPU, by code line:"+''.join(f"\n    {f_}:{l_}  x{c_}" for (f_,l_),c_ in sorted(cnt_.items(),key=lambda z:-z[1])[:25]),flush=True)
        return r_
    if cfg.cd_prof:
        if _CDP['todo'] is None: _CDP['todo']=sorted(int(c_) for c_ in cfg.cd_prof_calls)
        if _CDP['todo'] and _CDP['n']>=_CDP['todo'][0] and not _cd_will_reuse(model,cfg,dom):   # v9b: a GENERATION step
            _CDP['todo'].pop(0); cu_=torch.cuda.is_available()
            try:
                acts_=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if cu_ else [])
                if cu_: torch.cuda.synchronize()
                t0_=time.time(); _PRF['on']=True
                try:
                    with torch.profiler.profile(activities=acts_) as pr_:
                        with _rp_tags(model): r_=_cd_loss(model,cfg,x,dom)
                        if cu_: torch.cuda.synchronize()
                finally: _PRF['on']=False
                wt_=time.time()-t0_; ka_=pr_.key_averages()
                gt_=lambda e: (e.device_time_total if hasattr(e,'device_time_total') else e.cuda_time_total)
                nl_=sum(e_.count for e_ in ka_ if e_.key in ('cudaLaunchKernel','cuLaunchKernel'))
                gpu_=sum((e_.self_device_time_total if hasattr(e_,'self_device_time_total') else e_.self_cuda_time_total) for e_ in ka_)/1e3 if cu_ else 0.0
                print(f"  [conduit profile] GENERATION step (conduit call {_CDP['n']}, domain {dom}): wall {1000*wt_:.0f} ms (profiled) | GPU busy {gpu_:.0f} ms | {nl_} kernel launches")
                rg_=[e for e in ka_ if e.key.startswith('R:')]
                if rg_:
                    rg_.sort(key=lambda e:-(gt_(e) if cu_ else e.cpu_time_total))
                    print("  [conduit profile] NAMED REGIONS (GPU time | CPU time, incl. everything inside):\n"+"\n".join(f"    {e.key[2:]:20} GPU {gt_(e)/1e3:8.1f} ms | CPU {e.cpu_time_total/1e3:8.1f} ms  ({e.count} calls)" for e in rg_[:16]))
                print("  [conduit profile] top operations by "+("GPU" if cu_ else "CPU")+" time:\n"+ka_.table(sort_by=('self_cuda_time_total' if cu_ else 'self_cpu_time_total'),row_limit=14))
                print("  [conduit profile] top operations by CPU time (launch / Python overhead):\n"+ka_.table(sort_by='self_cpu_time_total',row_limit=10))
                return r_
            except Exception as e_: print(f"  [conduit profile] failed ({repr(e_)[:160]})")
    with _rp_tags(model): return _cd_loss(model,cfg,x,dom)
def _cd_loss(model,cfg,x,dom):
    _MIS32['on']=bool(cfg.cd_mis32)
    return _cd_loss_(model,cfg,x,dom)
def _cd_loss_(model,cfg,x,dom):
    """CONDUIT REPLAY (see cd_w): trace -> exactness-checked free replay (frozen teacher) -> KL into the student reading ordinary text."""
    ph_=model.__dict__.get('_sleep_phase')
    if ph_=='rem': return next(model.parameters()).sum()*0.0                  # SLEEP REM: no hippocampal replay (dreams only)
    if ph_=='nrem': return _nrem_loss_v1(model,cfg,dom) if int(cfg.nrem_res)>0 else _nrem_loss(model,cfg,dom)   # SLEEP NREM (v1: reservoir, no traces)
    n0=int(cfg.cd_rows); Ls_=int(cfg.cd_seg); C=int(cfg.cd_chain) if int(cfg.cd_chain)>0 else max(1,x.shape[1]//Ls_)   # CHAINED SUB-SEGMENTS: n0 windows of
    SELB_=_cbuf_take(model,cfg,dom,n0,x.device); n0=n0-len(SELB_); _CBUF['calls']+=1; _CBUF['mix']+=len(SELB_)   # v9f: buffered windows replace fresh ones
    if 0<float(cfg.cd_win_keep)<1.0 and n0>1:                                 # WINDOW-LEVEL SURPRISE GATING: only the most surprising share of the rows is replayed
        k_w=max(1,int(round(n0*float(cfg.cd_win_keep))))
        with torch.no_grad():
            if cfg.cd_win_rand: iw_=torch.randperm(x.shape[0],generator=_cd_g(model,x.device,cfg),device=x.device)[:k_w]
            else:
                was_w=model.training; model.eval(); fs_w=model.feat_scale.clone(); sv_w=model.__dict__.get('_s0_dom'); model.__dict__['_s0_dom']=dom
                try: lw_=F.cross_entropy(model(x[:,:-1].long(),dom).float().reshape(-1,256),x[:,1:].long().reshape(-1),reduction='none').view(x.shape[0],-1).mean(1)
                finally: model.__dict__['_s0_dom']=sv_w; model.feat_scale.copy_(fs_w); model.train(was_w)
                iw_=torch.topk(lw_,min(k_w,x.shape[0])).indices
        x=x[iw_]; n0=min(k_w,x.shape[0])
    Xw=_cd_slices(model,cfg,x,n0,Ls_*C)                                       #   sub-segment its own trace; all replayed together; the student reads
    if Xw.shape[1]!=Ls_*C: C=1                                                #   the JOINED windows (full context), positions weighted by exactness
    n=n0*C; X=Xw.reshape(n,-1); L=X.shape[1]; dev=X.device; M=int(cfg.cd_mark); g_=_cd_g(model,dev,cfg)
    fdf_=_fok_dom(dom); fw_=1.0; _V97['calls']+=1
    if float(cfg.cd_floor_hi)>float(cfg.cd_floor_lo) and fdf_.get('vps') is not None:      # v9e LOW-EXACT FLOOR
        fw_=min(1.0,max(0.0,(float(fdf_['vps'])-float(cfg.cd_floor_lo))/(float(cfg.cd_floor_hi)-float(cfg.cd_floor_lo))))
        fdf_['fcall']=fdf_.get('fcall',0)+1
        if fw_<=0.0 and fdf_['fcall']%max(1,int(cfg.cd_floor_every))!=0:
            _V97['skip']+=1; _V97['fw'].append(0.0); return next(model.parameters()).sum()*0.0
    rc_=model.__dict__.get('_cd_cache')                                       # v9 ROUTE 3: learn again from the last verified batch
    ru_=bool(int(cfg.cd_reuse)>1 and rc_ is not None and rc_['dom']==dom and rc_['k']<int(cfg.cd_reuse)-1 and rc_['best'].shape[0]==n)
    if ru_: rc_['k']+=1; _CDR[0]+=1
    _CDR[1]+=1
    sv=_carry_save(); was=model.training; model.eval(); fs=model.feat_scale.clone(); kh=getattr(model,'_keep_h',False); model._keep_h=False
    _CARRY['fac']=bool(cfg.cd_fac)                                             # FACTORED TRACES (cd_fac)
    _CDT['on']=(_CDT['calls']%20==0); _CDT['calls']+=1
    if _CDT['on']: _CDT['n']+=1; _cdt('_start')
    try:
        with torch.no_grad(), _cd_weights(model):
            if ru_:
                best,tg,ex_,chkd_,X,tries,eps=rc_['best'],rc_['tg'],rc_['ex'],rc_['chk'],rc_['X'],rc_['tries'],rc_['eps']
                pm_=rc_.get('pm',ex_[:,None].expand(n,L))
            else:
                fdq_=_fok_dom(dom); one_=bool(float(cfg.cd_fok_1try)>0 and fdq_['exr']>=float(cfg.cd_fok_1try))
                Gc_=max(1,int(cfg.cd_gen_chunk1) if ((one_ or cfg.cd_repair) and int(cfg.cd_gen_chunk1)>0) else int(cfg.cd_gen_chunk)); chs_=[(a_,min(n,a_+Gc_)) for a_ in range(0,n,Gc_)]   # v9c
                snap=_cd_cat([_cd_encode(model,X[a_:b_]) for a_,b_ in chs_]) if len(chs_)>1 else _cd_encode(model,X)
                _cdt('encode')
                ec_=model.__dict__.get('_cd_epsc')
                if cfg.cd_repair: _cd_fb(model,snap)
                if float(cfg.cd_eps)>0: eps=float(cfg.cd_eps)
                elif ec_ is not None and ec_[0]%max(1,int(cfg.cd_eps_every))!=0: eps=ec_[1]; model.__dict__['_cd_epsc']=(ec_[0]+1,eps)
                elif cfg.cd_repair and cfg.cd_first_block:                         # v9e: exact iff no key row differs; tolerance = key-row noise
                    a_,b_=chs_[0]; d0_=_cd_kdist(model,_cd_encode_chk(model,cfg,X[a_:b_]),_cd_rows(snap,torch.arange(a_,b_,device=dev),n),b_-a_,L)
                    model.__dict__['_cd_ktol']=max(float(cfg.cd_keytol),10.0*float(d0_.max())); eps=0.5; model.__dict__['_cd_epsc']=((ec_[0] if ec_ else 0)+1,eps)
                else:
                    a_,b_=chs_[0]; e1_=_cd_encode(model,X[a_:b_]); rf_=_cd_rows(snap,torch.arange(a_,b_,device=dev),n); e0_=_cd_mis(e1_,rf_,b_-a_)
                    eps=max(float(cfg.cd_eps_floor),10.0*float(e0_.max())); model.__dict__['_cd_epsc']=((ec_[0] if ec_ else 0)+1,eps)
                    if cfg.cd_repair: model.__dict__['_cd_ktol']=max(float(cfg.cd_keytol),10.0*float(_cd_kdist(model,e1_,rf_,b_-a_,L).max()))
                _cdt('tolerance')
                best=None; beste=torch.full((n,),float('inf'),device=dev); tries=0
                tgb=None; chkd_=torch.ones(n,dtype=torch.bool,device=dev)            # v7: which pieces were verified by the re-encode check
                JV_=torch.full((n,),L,dtype=torch.long,device=dev)                 # v9e: verified prefix length per piece
                if cfg.cd_inc or cfg.cd_greedy:                                    # v5: generation logits = teacher targets; greedy first try
                    best=torch.zeros(n,L,dtype=torch.long,device=dev); tgb=torch.empty(n,L,256,device=dev); K_=max(1,int(cfg.cd_tries))
                    for a_,b_ in chs_:
                        m_=b_-a_; st_=_cd_rows(snap,torch.arange(a_,b_,device=dev),n)
                        if cfg.cd_greedy:
                            Ys,lp_=_cd_gen(model,cfg,st_,m_,L,dom,M,0.0,g_); tries+=m_
                            best[a_:b_]=Ys; tgb[a_:b_]=lp_; beste[a_:b_]=_cd_mis(_cd_encode(model,Ys),st_,m_)
                            for _ in range(K_-1):                                      # sampled retries: still-inexact segments only
                                fl_=(beste[a_:b_]>eps).nonzero().flatten()
                                if len(fl_)==0: break
                                sf_=_cd_rows(st_,fl_,m_); Ys,lp_=_cd_gen(model,cfg,sf_,len(fl_),L,dom,M,float(cfg.cd_temp),g_); tries+=len(fl_)
                                e_=_cd_mis(_cd_encode(model,Ys),sf_,len(fl_)); bt_=e_<beste[a_+fl_]; ix_=a_+fl_[bt_]
                                best[ix_]=Ys[bt_]; tgb[ix_]=lp_[bt_]; beste[ix_]=e_[bt_]
                        elif cfg.cd_repair:                                            # v9e RECALL FIX: greedy try -> key-row check -> repair
                            fd_=_fok_dom(dom)
                            hn_=_cd_hints(model,cfg,st_,m_,L,dom,M,X[a_:b_]) if (float(cfg.cd_hint_frac)>0 or float(cfg.cd_hint_adapt)>0) else None   # cg12b
                            Ys,lp_=_cd_gen(model,cfg,st_,m_,L,dom,M,0.0,g_,hint=hn_); tries+=m_; cf_=_CDCONF['c'].clone()
                            if cfg.cd_fok:                                             # the GATE on the greedy try (as v7)
                                cnf_=cf_>=fd_['thr']; cal_=cnf_&(torch.rand(m_,generator=g_,device=dev)<float(cfg.cd_fok_chk)); chk_=(~cnf_)|cal_
                            else: cnf_=torch.zeros(m_,dtype=torch.bool,device=dev); cal_=cnf_; chk_=~cnf_
                            best[a_:b_]=Ys; tgb[a_:b_]=lp_; beste[a_:b_]=0.0; chkd_[a_:b_]=chk_
                            pc_=chk_.nonzero().flatten()
                            if len(pc_)>0:
                                sP_=_cd_rows(st_,pc_,m_); e_,j_=_cd_vcheck(model,cfg,Ys[pc_].contiguous(),sP_,len(pc_),L,eps)
                                beste[a_+pc_]=e_; JV_[a_+pc_]=j_; _V97['chk'].append(len(pc_))
                                if cfg.cd_fok:
                                    fd_['c'].append(cf_[pc_].flatten()); fd_['e'].append((e_<=eps).flatten()); fd_['nc']+=len(pc_)
                                    cl_=cal_[pc_]; FOKS.setdefault('estn',[]).append((cl_&(e_>eps)).sum()); FOKS.setdefault('estd',[]).append(cl_.sum())
                                act_=torch.ones(len(pc_),dtype=torch.bool,device=dev)
                                for _ in range(max(0,int(cfg.cd_repair_n))):           # REPAIR from the first wrong byte (excluded), greedy after it
                                    nd_=((j_<L)&act_).nonzero().flatten()
                                    if len(nd_)==0: break
                                    rr_=pc_[nd_]; fl_=j_[nd_]; fo_=best[a_+rr_]; bn_=fo_.gather(1,fl_[:,None]).squeeze(1)
                                    sR_=_cd_rows(st_,rr_,m_); Yr,lpr=_cd_gen(model,cfg,sR_,len(rr_),L,dom,M,0.0,g_,forced=fo_,flen=fl_,ban=bn_,hint=((hn_[0][rr_],hn_[1][rr_]) if hn_ is not None else None))
                                    tries+=len(rr_); er_,jr_=_cd_vcheck(model,cfg,Yr,sR_,len(rr_),L,eps); bt_=jr_>fl_; ix_=a_+rr_   # v9.8: selects on the GPU
                                    best[ix_]=torch.where(bt_[:,None],Yr,best[ix_]); tgb[ix_]=torch.where(bt_[:,None,None],lpr,tgb[ix_])   #   (no boolean-mask
                                    beste[ix_]=torch.where(bt_,er_,beste[ix_]); JV_[ix_]=torch.where(bt_,jr_,JV_[ix_])                   #   indexing: no CPU wait)
                                    j_=j_.clone(); j_[nd_]=torch.where(bt_,jr_,j_[nd_]); act_=torch.zeros_like(act_); act_[nd_]=bt_
                                    _V97['rep']+=len(rr_); _V97['imp'].append(bt_.sum()); del sR_
                                del sP_
                            if cfg.cd_fok:
                                nch_=~chk_; FOKS['skip'].append(nch_.float().mean()); FOKS['k'].append(1)
                                FOKS.setdefault('trn',[]).append(((best[a_:b_]!=X[a_:b_]).any(1)&nch_).sum()); FOKS.setdefault('trd',[]).append(nch_.sum())
                                fd_['n']+=1
                                if fd_['nc']>=int(cfg.cd_fok_min) and fd_['n']%5==1: _fok_thr_gpu(fd_,cfg)
                        else:                                                          # all tries sampled, in one batch
                            fd_=_fok_dom(dom); Kq_=1 if (float(cfg.cd_fok_1try)>0 and fd_['exr']>=float(cfg.cd_fok_1try)) else K_   # v7: adaptive tries
                            stK=_cd_rows(st_,torch.arange(m_,device=dev).repeat(Kq_),m_)
                            Ys,lp_=_cd_gen(model,cfg,stK,m_*Kq_,L,dom,M,float(cfg.cd_temp),g_); tries+=m_*Kq_
                            cf_=_CDCONF['c'].reshape(Kq_,m_); ar_=torch.arange(m_,device=dev); Ysr=Ys.reshape(Kq_,m_,L); lpr=lp_.reshape(Kq_,m_,L,-1)
                            cmx_,ci_=cf_.max(0)
                            if cfg.cd_fok:                                             # v7 GATE: confident -> accepted unchecked (+ random calibration checks)
                                cnf_=cmx_>=fd_['thr']; cal_=cnf_&(torch.rand(m_,generator=g_,device=dev)<float(cfg.cd_fok_chk)); chk_=(~cnf_)|cal_
                            else: cnf_=torch.zeros(m_,dtype=torch.bool,device=dev); cal_=cnf_; chk_=~cnf_
                            best[a_:b_]=Ysr[ci_,ar_]; tgb[a_:b_]=lpr[ci_,ar_]; beste[a_:b_]=0.0; chkd_[a_:b_]=chk_
                            pc_=chk_.nonzero().flatten()
                            if len(pc_)>0:
                                rows_=(torch.arange(Kq_,device=dev)[:,None]*m_+pc_[None]).flatten(); sK_=_cd_rows(stK,rows_,m_*Kq_)
                                e_=_cd_mis(_cd_encode(model,Ys[rows_]),sK_,len(rows_)).reshape(Kq_,len(pc_)); be_,bi_=e_.min(0)
                                best[a_+pc_]=Ysr[bi_,pc_]; tgb[a_+pc_]=lpr[bi_,pc_]; beste[a_+pc_]=be_; del sK_
                                if cfg.cd_fok and cfg.cd_nosync:                       # v9d: calibration data and statistics stay on the GPU
                                    fd_['c'].append(cf_[:,pc_].flatten()); fd_['e'].append((e_<=eps).flatten()); fd_['nc']+=Kq_*len(pc_)
                                    cl_=cal_[pc_]; okc_=(e_[ci_[pc_],torch.arange(len(pc_),device=dev)]<=eps)
                                    FOKS.setdefault('estn',[]).append((cl_&~okc_).sum()); FOKS.setdefault('estd',[]).append(cl_.sum())
                                elif cfg.cd_fok:
                                    fd_['c']+=cf_[:,pc_].flatten().tolist(); fd_['e']+=(e_<=eps).float().flatten().tolist()
                                    fd_['c']=fd_['c'][-40000:]; fd_['e']=fd_['e'][-40000:]
                                    cl_=cal_[pc_]
                                    if bool(cl_.any()):                                # estimated wrong-accept: calibration pieces whose most confident try failed
                                        ok_=(e_[ci_[pc_],torch.arange(len(pc_),device=dev)]<=eps)[cl_]; FOKS['est'].append(float(1.0-ok_.float().mean()))
                            if cfg.cd_fok and cfg.cd_nosync:                           # v9d: skipped share / true wrong accepts as tensors; threshold on the GPU
                                nch_=~chk_; FOKS['skip'].append(nch_.float().mean()); FOKS['k'].append(Kq_)
                                FOKS.setdefault('trn',[]).append(((best[a_:b_]!=X[a_:b_]).any(1)&nch_).sum()); FOKS.setdefault('trd',[]).append(nch_.sum())
                                fd_['n']+=1
                                if fd_['nc']>=int(cfg.cd_fok_min) and fd_['n']%5==1: _fok_thr_gpu(fd_,cfg)
                            elif cfg.cd_fok:
                                ac_=(~chk_).nonzero().flatten(); FOKS['skip'].append(float((~chk_).float().mean())); FOKS['k'].append(Kq_)
                                if len(ac_)>0: FOKS['true'].append(float((best[a_+ac_]!=X[a_+ac_]).any(1).float().mean()))   # TRUE bytes: diagnostic
                                fd_['n']+=1
                                if len(fd_['c'])>=int(cfg.cd_fok_min) and fd_['n']%5==1:
                                    c_=np.array(fd_['c']); e2_=np.array(fd_['e']); o_=np.argsort(-c_); bad_=np.cumsum(1.0-e2_[o_])/np.arange(1,len(o_)+1)
                                    okk_=np.where((bad_<=float(cfg.cd_fok_far))&(np.arange(1,len(o_)+1)>=200))[0]
                                    fd_['thr']=float(c_[o_][okk_.max()]) if len(okk_) else float('inf')
                            del stK
                elif cfg.cd_par:                                                   # PARALLEL TRIES: all rows x tries generated as ONE batch (one call per byte)
                    K_=max(1,int(cfg.cd_tries)); best=torch.zeros(n,L,dtype=torch.long,device=dev)
                    for a_,b_ in chs_:                                             #   (in chunks of cd_gen_chunk segments)
                        m_=b_-a_; stK=_cd_rows(snap,torch.arange(a_,b_,device=dev).repeat(K_),n); Y=torch.full((m_*K_,1),M,dtype=torch.long,device=dev)
                        for _ in range(L):
                            _CARRY.update(on=True,nr=m_*K_,st=stK,new={},rows=None)
                            lg=model(Y,dom)[:,-1].float()/float(cfg.cd_temp); Y=torch.cat([Y,torch.multinomial(F.softmax(lg,-1),1,generator=g_)],1)
                        Ys=Y[:,1:]; e_=_cd_mis(_cd_encode(model,Ys),stK,m_*K_).reshape(K_,m_); be_,bi_=e_.min(0)
                        best[a_:b_]=Ys.reshape(K_,m_,L)[bi_,torch.arange(m_,device=dev)]; beste[a_:b_]=be_; tries+=m_*K_; del stK
                for k_ in (range(0) if (cfg.cd_par or cfg.cd_inc or cfg.cd_greedy) else range(max(1,int(cfg.cd_tries)))):
                    rows=torch.arange(n,device=dev) if k_==0 else (beste>eps).nonzero().flatten()
                    if len(rows)==0: break
                    st_=_cd_rows(snap,rows,n); m=len(rows); Y=torch.full((m,1),M,dtype=torch.long,device=dev)
                    for _ in range(L):
                        _CARRY.update(on=True,nr=m,st=st_,new={},rows=None)
                        lg=model(Y,dom)[:,-1].float()/float(cfg.cd_temp); Y=torch.cat([Y,torch.multinomial(F.softmax(lg,-1),1,generator=g_)],1)
                    Ys=Y[:,1:]; tries+=m; e_=_cd_mis(_cd_encode(model,Ys),st_,m)
                    if best is None: best=Ys.clone()
                    bt_=e_<beste[rows]; best[rows[bt_]]=Ys[bt_]; beste[rows[bt_]]=e_[bt_]
                _cdt('generate + check')
                ex_=beste<=eps
                if cfg.cd_repair:                                                  # v9e: the positions that teach
                    pm_=(torch.arange(L,device=dev)[None]<JV_[:,None]) if cfg.cd_partial else ex_[:,None].expand(n,L)
                    tcp_=torch.cumprod((best==X).long(),1).bool()                      #   (diagnostic) the TRULY correct prefix
                    _V97['vp'].append(pm_.float().mean()); _V97['pn'].append((pm_&tcp_).sum()); _V97['pd'].append(pm_.sum())
                    _V97['ag'].append(((JV_==tcp_.sum(1))&chkd_).sum()); _V97['agd'].append(chkd_.sum())
                    vm_=pm_.float().mean(); fdv_=_fok_dom(dom); fdv_['vpsT']=vm_ if fdv_.get('vpsT') is None else 0.9*fdv_['vpsT']+0.1*vm_
                    fdv_['vcall']=fdv_.get('vcall',0)+1
                    if fdv_['vcall']%10==1: fdv_['vps']=float(fdv_['vpsT'])
                fdd_=_fok_dom(dom)
                if cfg.cd_nosync:                                                  # v9d: EMA on the device, read back every 10 conduit calls
                    exm_=ex_.float().mean(); fdd_['exrT']=exm_ if fdd_['exrT'] is None else 0.9*fdd_['exrT']+0.1*exm_; fdd_['ncall']+=1
                    if fdd_['ncall']%10==1: fdd_['exr']=float(fdd_['exrT'])
                else: fdd_['exr']=float(ex_.float().mean()) if fdd_['exr']==0.0 else 0.9*fdd_['exr']+0.1*float(ex_.float().mean())
                Yin=torch.cat([torch.full((n,1),M,dtype=torch.long,device=dev),best[:,:-1]],1); tg=[]
                for a_,b_ in (chs_ if tgb is None else ()):
                    _CARRY.update(on=True,nr=b_-a_,st=_cd_rows(snap,torch.arange(a_,b_,device=dev),n) if len(chs_)>1 else snap,new={},rows=None)
                    tg.append(F.log_softmax(model(Yin[a_:b_],dom).float(),-1))       # teacher WITH the trace (replay mode): tg[:,j] predicts best[:,j]
                tg=torch.cat(tg) if tgb is None else tgb
                if (cfg.cd_reuse_mode=='loop' and int(cfg.cd_reuse_k)>0) or int(cfg.cd_store_cap)>0: model.__dict__['_snap_last']=snap   # kept for the maintenance loop
                del snap
                _cdt('teacher targets')
    finally:
        _INC['on']=False; _CARRY.pop('fac',None); _carry_load(sv); model.feat_scale.copy_(fs); model.train(was); model._keep_h=kh
    if int(cfg.cd_reuse)>1 and not ru_: model.__dict__['_cd_cache']={'dom':dom,'k':0,'best':best,'tg':tg,'ex':ex_,'chk':chkd_,'X':X,'tries':tries,'eps':eps,'pm':(pm_ if cfg.cd_repair else None)}
    bj_=best.reshape(n0,-1); tgj_=tg.reshape(n0,-1,tg.shape[-1])
    if int(cfg.lm_rank)>=0: model.__dict__['_last_rep']=(bj_.detach(),int(dom))     # (dream merge: the visit's verified replays, for its drift check)               # joined windows (C = 1: unchanged)
    if float(cfg.cd_buf)>0 and dom in _CBUF['d']:                               # v9f: this step's verified windows into the buffer
        eb_=_CBUF['d'][dom]; pmb_=(pm_ if cfg.cd_repair else ex_[:,None].expand(n,L))&chkd_[:,None]
        for r_ in range(min(int(cfg.cd_buf_per_step),n0)):
            if len(eb_['x'])>=int(cfg.cd_buf_cap): break
            eb_['x'].append(bj_[r_].clone()); eb_['t'].append(tgj_[r_].half()); eb_['m'].append(pmb_.reshape(n0,-1)[r_].clone())
    rz_=None
    if cfg.cd_runs and C>1:                                                    # v7 EXACT RUNS: piece j starts a run after an inexact piece j-1
        exw_=ex_.view(n0,C); st0_=torch.zeros(n0,C,dtype=torch.bool,device=dev); st0_[:,1:]=(~exw_[:,:-1]) if cfg.cd_runs_any else (exw_[:,1:]&~exw_[:,:-1])
        rb_=st0_.repeat_interleave(L,1)&(torch.arange(C*L,device=dev)%L==0)[None]   # (n0, C*L): first byte p of such a run
        if cfg.cd_runs_mask:                                                   # v9 FIX 1: the run is its own clean segment from byte p
            _RESET['m']=rb_[:,:-1].contiguous()                                #   memory reset AT input byte p
            ps_=torch.arange(C*L-1,device=dev)[None].expand(n0,-1)
            _SEG['s']=torch.cummax(torch.where(rb_[:,:-1],ps_,torch.zeros_like(ps_)),1).values   #   local filter + n-grams stop at p
            sk_=int(cfg.cd_run_excl)
        else:
            sk_=int(cfg.cd_run_skip); rs_=torch.zeros_like(rb_); rs_[:,sk_:]=rb_[:,:C*L-sk_]
            _RESET['m']=rs_[:,:-1].contiguous()                                #   memory reset at input byte p + skip
        rz_=torch.zeros(n0,C*L,dtype=torch.bool,device=dev)
        for o_ in range(sk_+1): rz_[:,o_:]|=rb_[:,:C*L-o_]                     #   target bytes p .. p + skip do not teach (fix 1: p .. p + excl)
    sv2=_carry_save(); suh=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
    if cfg.cd_ngp_mask and not cfg.cd_ngp_frozen:                              # v8b: n-gram rows learn only at positions that teach (input t -> target t+1)
        wm_=torch.where(ex_,torch.ones(n,device=dev),torch.full((n,),float(cfg.cd_inexact_w),device=dev)).view(n0,C).repeat_interleave(L,1)[:,1:]
        if rz_ is not None: wm_=wm_*(~rz_[:,1:]).float()
        _NGPD['m']=wm_>0
    KLB_=0.0; NPB_=0
    try:
        _HCAP['on']=((float(cfg.cd_gate_keep)>0 and str(cfg.cd_gate_mode)=='prot') or bool(cfg.dream_sim)); _HCAP['h']=None
        try: lp=F.log_softmax(model(bj_[:,:-1],dom).float(),-1)
        finally: _HCAP['on']=False
        hcap_=_HCAP['h']; _HCAP['h']=None                                       # (protected gate: the student's head input)
        if cfg.dream_sim and hcap_ is not None:                                 # SWIL: the current learning's rank-1 signature (u = mean z, v = mean dL/dz)
            with torch.no_grad():
                _sig_update(model,dom,hcap_,lp.detach(),tgj_[:,1:])
        if False: lp=F.log_softmax(model(bj_[:,:-1],dom).float(),-1)            # student: the replay as ORDINARY text, no memory, no marker
        if SELB_:                                                               # v9f: buffered windows, each with ITS OWN domain
            cur_=_XDG.get('cur')
            try:
                for d_ in sorted({d_ for d_,_ in SELB_}):
                    ii_=[i_ for e_,i_ in SELB_ if e_==d_]; eb_=_CBUF['d'][d_]
                    xb_=torch.stack([eb_['x'][i_] for i_ in ii_]); tb_=torch.stack([eb_['t'][i_] for i_ in ii_]).float()[:,1:]; mb_=torch.stack([eb_['m'][i_] for i_ in ii_]).float()[:,1:]
                    _XDG['cur']=d_; _BUFDET['on']=bool(cfg.cd_buf_detach)
                    try: lpb_=F.log_softmax(model(xb_[:,:-1],d_).float(),-1)
                    finally: _BUFDET['on']=False
                    KLB_=KLB_+((tb_.exp()*(tb_-lpb_)).sum(-1)*mb_).sum(); NPB_+=mb_.numel()
                    _CBUF['src'][d_]=_CBUF['src'].get(d_,0)+len(ii_)
            finally: _XDG['cur']=cur_
    finally: _carry_load(sv2); _uh_load(model,suh); _NGPD['on']=False; _NGPD['m']=None; _RESET['m']=None; _SEG['s']=None
    _cdt('student forward'); _CDT['on']=False
    iw_=float(cfg.cd_inexact_w); w_=torch.where(ex_,torch.ones(n,device=dev),torch.full((n,),iw_,device=dev)); tg2=tgj_[:,1:]
    wp_=w_.view(n0,C).repeat_interleave(L,1)[:,1:]                              # per-position weight = its sub-segment's
    if cfg.cd_repair: wp_=pm_.float().reshape(n0,C*L)[:,1:]                         # v9e: per-position (verified prefix) weights
    if rz_ is not None: wp_=wp_*(~rz_[:,1:]).float(); FOKS['runs'].append(wp_.mean().detach())   # v9.8: no CPU wait (read at the visit end)   # v7: targets whose context crosses a run start
    if float(cfg.cd_gate_keep)>0 and C>1:                                       # SURPRISE-GATED REPLAY (test): low-surprise pieces do not teach
        with torch.no_grad():
            T1g_=lp.shape[1]; pg_=((torch.arange(T1g_,device=lp.device)+1)//max(1,L)).clamp(max=C-1)
            ceg_=-lp.gather(-1,bj_[:,1:,None]).squeeze(-1)
            sg_=torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,ceg_.float()); cg_=torch.zeros_like(sg_).index_add_(1,pg_,torch.ones_like(ceg_,dtype=torch.float))
            sur_=(sg_/cg_.clamp_min(1)).flatten(); k_=max(1,int(sur_.numel()*float(cfg.cd_gate_keep)))
            ksur_=torch.zeros_like(sur_,dtype=torch.bool); ksur_[torch.topk(sur_,k_).indices]=True
            gm_=str(cfg.cd_gate_mode); gg_=_cd_g(model,lp.device,cfg)
            if gm_ in ('rho','rhopos'):                                         # REDUCIBLE SIGNAL ('rhopos': every piece with a POSITIVE signal, no fixed share) (data-averaged trace benefit, RHO-LOSS form):
                yr_=bj_[:,1:,None]; vm_=(wp_>0).float()                         #   log q(y) - log p(y) at the VERIFIED byte = student surprise
                rr_=(tg2.gather(-1,yr_)-lp.gather(-1,yr_)).squeeze(-1)*vm_      #   minus teacher surprise (teacher = with the trace, pre-blend)
                srh_=(torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,rr_.float())/
                      torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,vm_).clamp_min(1)).flatten()
                keep_=torch.zeros_like(sur_,dtype=torch.bool); keep_[torch.topk(srh_,k_).indices]=True
                if gm_=='rhopos': keep_=srh_>0
                GR_=model.__dict__.setdefault('_grho',[0,0.0,0.0]); GR_[0]+=1; GR_[1]=GR_[1]+(keep_&ksur_).sum().float()/max(1,k_)   # (device tensors:
                GR_[2]=GR_[2]+(srh_<0).float().mean()                                                                           #  no CPU wait)
            elif gm_ in ('kl','prot','kl0'):                                      # THE OBJECTIVE ITSELF: per-piece KL to the target actually taught
                qg_=tg2.exp()                                                   #   ('kl0': to the teacher's OWN distribution, no evidence blend)
                if float(cfg.cd_evid_w)>0 and gm_!='kl0':                                      #   (evidence blend applied, as the loss will see it)
                    yg_=bj_[:,1:]; dg_=(tg2.argmax(-1)!=yg_).to(qg_.dtype).unsqueeze(-1)*float(cfg.cd_evid_w)
                    qg_=(1-dg_)*qg_+dg_*F.one_hot(yg_,qg_.shape[-1]).to(qg_.dtype)
                klg_=(qg_*(torch.log(qg_.clamp_min(1e-12))-lp)).sum(-1)*(wp_>0).float()
                skl_=torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,klg_.float()).flatten()
                kkl_=torch.zeros_like(sur_,dtype=torch.bool); kkl_[torch.topk(skl_,k_).indices]=True; keep_=kkl_
                if gm_=='prot':                                                 # PROTECTED GATE: KL x context novelty to the old domains,
                    M_=_prot_minv(model,cfg,lp.device) if hcap_ is not None else None   #   h^T (A_old + lam)^-1 h  (the K-FAC factorisation)
                    if M_ is not None:
                        hz_=hcap_.float(); nov_=torch.einsum('btd,de,bte->bt',hz_,M_,hz_)/hz_.shape[-1]
                        spr_=torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,(klg_*nov_).float()).flatten()
                        keep_=torch.zeros_like(sur_,dtype=torch.bool); keep_[torch.topk(spr_,k_).indices]=True
                        GP_=model.__dict__.setdefault('_gprot',[0,0.0,0.0,0.0]); GP_[0]+=1; GP_[1]+=float((keep_&kkl_).sum())/max(1,k_)
                        GP_[2]+=float((keep_&ksur_).sum())/max(1,k_)
                        nvp_=torch.zeros(lp.shape[0],C,device=lp.device).index_add_(1,pg_,nov_.float()*(wp_>0).float()).flatten()
                        GP_[3]+=float(nvp_[keep_].mean()/nvp_.mean().clamp_min(1e-12))
                if gm_ in ('kl','kl0'): GK_=model.__dict__.setdefault('_gkl',[0,0.0,0.0]); GK_[0]+=1; GK_[1]+=float((keep_&ksur_).sum())/max(1,k_)
                if gm_ in ('kl','kl0'): GK_[2]+=float(skl_[ksur_].sum()/skl_.sum().clamp_min(1e-12))
            elif gm_=='rand':
                keep_=torch.zeros_like(sur_,dtype=torch.bool); keep_[torch.randperm(sur_.numel(),generator=gg_,device=lp.device)[:k_]]=True
            elif gm_=='kpp':                                                    # ORTHOGONALITY / DIVERSITY GATE (BADGE-style k-means++)
                gp_=(lp.exp()-tg2.exp())*wp_.unsqueeze(-1)                    #   teaching gradient w.r.t. the logits, per position
                E_=torch.zeros(lp.shape[0],C,gp_.shape[-1],device=lp.device).index_add_(1,pg_,gp_.float()).reshape(-1,gp_.shape[-1])
                D2_=(E_*E_).sum(1); keep_=torch.zeros_like(sur_,dtype=torch.bool)
                for _ in range(k_):
                    tot_=float(D2_.sum())
                    if not tot_>0: break
                    j_=int(torch.multinomial(D2_/tot_,1,generator=gg_)); keep_[j_]=True
                    D2_=torch.minimum(D2_,((E_-E_[j_])**2).sum(1)); D2_[j_]=0.0
                nk_=int(keep_.sum())
                if nk_<k_:                                                      #   (all signal used up: fill at random)
                    rest_=(~keep_).nonzero().flatten(); keep_[rest_[torch.randperm(rest_.numel(),generator=gg_,device=lp.device)[:k_-nk_]]]=True
                GS_=model.__dict__.setdefault('_gsel',[0,0.0,0.0,0.0])
                sigp_=D2_.new_zeros(1); en_=(E_*E_).sum(1)
                GS_[0]+=1; GS_[1]+=float((keep_&ksur_).sum())/max(1,k_); GS_[2]+=float(en_[keep_].sum()/en_.sum().clamp_min(1e-12)); GS_[3]+=float(en_[ksur_].sum()/en_.sum().clamp_min(1e-12))
            else: keep_=ksur_
            wp_=wp_*keep_.view(lp.shape[0],C)[:,pg_].float()
    if int(cfg.cd_store_cap)>0 or int(cfg.nrem_res)>0: model.__dict__['_tqy']=tg2.gather(-1,bj_[:,1:,None]).squeeze(-1).detach()   # (store: pre-blend teacher log q(y))
    if float(cfg.cd_evid_w)>0:                                                  # EVIDENCE TARGETS: blend the verified replayed byte into the teacher target
        yb_=bj_[:,1:]; ew_=float(cfg.cd_evid_w)
        model.__dict__['_dis_last']=(tg2.argmax(-1)!=yb_)                     # (pre-blend disagreement: the reuse fragility measure)
        dis_=torch.ones_like(yb_,dtype=torch.bool) if cfg.cd_evid_all else (tg2.argmax(-1)!=yb_)
        pt_=tg2.exp(); oh_=F.one_hot(yb_,pt_.shape[-1]).to(pt_.dtype); lam_=(ew_*dis_.to(pt_.dtype)).unsqueeze(-1)
        if float(cfg.cd_evid_adapt)>0:                                          # ADAPTIVE: the piece's share of disagreeing positions raises the blend
            T1e_=yb_.shape[1]; pe_=((torch.arange(T1e_,device=yb_.device)+1)//max(1,L)).clamp(max=C-1)
            dsum_=torch.zeros(yb_.shape[0],C,device=yb_.device).index_add_(1,pe_,dis_.float()); dcnt_=torch.zeros_like(dsum_).index_add_(1,pe_,torch.ones_like(dis_,dtype=torch.float))
            dshare_=(dsum_/dcnt_.clamp_min(1))[:,pe_]
            lam_=((ew_+(float(cfg.cd_evid_adapt)-ew_)*dshare_)*dis_.float()).clamp(0,1).unsqueeze(-1).to(pt_.dtype)
        tg2=torch.log(((1-lam_)*pt_+lam_*oh_).clamp_min(1e-12))
        EV_=model.__dict__.setdefault('_evid',[0.0,0.0]); EV_[0]+=(dis_.float()*wp_).sum().detach(); EV_[1]+=wp_.sum().detach()
    klp_=(tg2.exp()*(tg2-lp)).sum(-1)
    KLR_=0.0; NPR_=0
    if int(cfg.cd_reuse_k)>0 and cfg.cd_reuse_mode!='loop':                     # SPACED REPLAY REUSE
        RB_=model.__dict__.setdefault('_rbuf',{}); st_r=model.__dict__.get('_rstep',0)+1; model.__dict__['_rstep']=st_r
        if RB_.get('dom')!=dom: RB_.clear(); RB_['dom']=dom; RB_['e']=[]
        with torch.no_grad():
            T1r_=tg2.shape[1]; prr_=((torch.arange(T1r_,device=tg2.device)+1)//max(1,L)).clamp(max=C-1)
            dis0_=model.__dict__.pop('_dis_last',None)                          # fragility from the PRE-evidence teacher (the blend always agrees)
            dz_=(dis0_ if dis0_ is not None else (tg2.argmax(-1)!=bj_[:,1:])).float()*(wp_>0).float()
            fp_=(torch.zeros(dz_.shape[0],C,device=dz_.device).index_add_(1,prr_,dz_)>0); frag_=fp_.float().mean(1)
            for r_ in range(bj_.shape[0]):
                RB_['e'].append({'s':st_r,'x':bj_[r_].clone(),'t':tg2[r_].half(),'w':wp_[r_].clone(),'fp':fp_[r_][prr_].clone(),'f':(float(frag_[r_]) if cfg.cd_reuse_mode=='mix' else frag_[r_])})
            RB_['e']=[e_ for e_ in RB_['e'] if st_r-e_['s']<=3*int(cfg.cd_reuse_lag)]
        el_=[e_ for e_ in RB_['e'] if st_r-e_['s']>=int(cfg.cd_reuse_lag)]
        if el_:
            k_r=min(int(cfg.cd_reuse_k),len(el_)); gr_=_cd_g(model,dev,cfg)
            if cfg.cd_reuse_mode=='mix':
                kp_=k_r//2; srt_=sorted(range(len(el_)),key=lambda i_:-el_[i_]['f']); pick_=srt_[:kp_]
                rest_=[i_ for i_ in range(len(el_)) if i_ not in set(pick_)]; ru2_=torch.randperm(len(rest_),generator=gr_,device=dev)[:k_r-kp_].tolist()
                pick_=pick_+[rest_[i_] for i_ in ru2_]
            else: pick_=model.__dict__.setdefault('_ru_rs',np.random.RandomState(int(cfg.seed)+97)).permutation(len(el_))[:k_r].tolist()   # (CPU draw: no GPU wait)
            sel_=[el_[i_] for i_ in pick_]; idset_={id(e_) for e_ in sel_}; RB_['e']=[e_ for e_ in RB_['e'] if id(e_) not in idset_]
            xr_=torch.stack([e_['x'] for e_ in sel_]); trr_=torch.stack([e_['t'] for e_ in sel_]).float(); wr_=torch.stack([e_['w'] for e_ in sel_]).float()
            if cfg.cd_reuse_mode=='upw': wr_=wr_*(1.0+(float(cfg.cd_reuse_upw)-1.0)*torch.stack([e_['fp'] for e_ in sel_]).float())
            sv3_=_carry_save(); suh3_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
            try: lpr_=F.log_softmax(model(xr_[:,:-1],dom).float(),-1)
            finally: _carry_load(sv3_); _uh_load(model,suh3_); _NGPD['on']=False
            KLR_=((trr_.exp()*(trr_-lpr_)).sum(-1)*wr_).sum(); NPR_=wr_.numel()
            RU_=model.__dict__.setdefault('_rustat',[0,0.0,0.0]); RU_[0]+=k_r; _fs_=lambda L_: (torch.stack(L_).sum() if torch.is_tensor(L_[0]) else float(sum(L_))); RU_[1]=RU_[1]+_fs_([e_['f'] for e_ in sel_]); RU_[2]=RU_[2]+_fs_([e_['f'] for e_ in el_])/max(1,len(el_))*k_r   # (device sums)
    if int(cfg.cd_reuse_k)>0 and cfg.cd_reuse_mode=='loop':                     # DEGRADATION-DRIVEN MAINTENANCE LOOP
        LB_=model.__dict__.setdefault('_lbuf',{}); st_l=model.__dict__.get('_lstep',0)+1; model.__dict__['_lstep']=st_l
        if LB_.get('dom')!=dom: LB_.clear(); LB_['dom']=dom; LB_['e']=[]; LB_['st']=[0,0,0,0,0,0.0]
        snp_=model.__dict__.pop('_snap_last',None)
        if snp_ is not None:
            with torch.no_grad():
                for r_ in range(bj_.shape[0]):
                    LB_['e'].append({'due':st_l+int(cfg.cd_reuse_lag),'iv':int(cfg.cd_reuse_lag),'n':0,'second':False,'x':bj_[r_].clone(),'w':wp_[r_].clone(),
                                     'tr':_cd_rows(snp_,torch.arange(r_*C,(r_+1)*C,device=dev),n)})
        due_=[e_ for e_ in LB_['e'] if e_['due']<=st_l]
        if due_:
            due_.sort(key=lambda e_:(not e_['second'],e_['due'])); sel_=due_[:int(cfg.cd_reuse_k)]; k_l=len(sel_)
            with torch.no_grad():
                trs_=_cd_cat([e_['tr'] for e_ in sel_]) if k_l>1 else sel_[0]['tr']
                xb_=torch.stack([e_['x'] for e_ in sel_]); pcs_=xb_.reshape(k_l*C,L)
                Yl_=torch.cat([torch.full((k_l*C,1),M,dtype=torch.long,device=dev),pcs_[:,:-1]],1)
                svl_=_carry_save(); wasl_=model.training; model.eval(); fsl_=model.feat_scale.clone(); _CARRY['fac']=bool(cfg.cd_fac)
                try:
                    with _cd_weights(model):
                        _CARRY.update(on=True,nr=k_l*C,st=trs_,new={},rows=None); tgl_=F.log_softmax(model(Yl_,dom).float(),-1)
                finally: _CARRY.pop('fac',None); _carry_load(svl_); model.feat_scale.copy_(fsl_); model.train(wasl_)
                tgl_=tgl_.reshape(k_l,C*L,-1)[:,1:]; ybl_=xb_[:,1:]; wl_=torch.stack([e_['w'] for e_ in sel_]).float()
                dgp_=(tgl_.argmax(-1)!=ybl_).float()*(wl_>0).float(); dg_=dgp_.sum(1)/(wl_>0).float().sum(1).clamp_min(1)
                if float(cfg.cd_evid_w)>0:
                    ptl_=tgl_.exp(); lam2_=(float(cfg.cd_evid_w)*dgp_).unsqueeze(-1); tgl_=torch.log(((1-lam2_)*ptl_+lam2_*F.one_hot(ybl_,ptl_.shape[-1]).to(ptl_.dtype)).clamp_min(1e-12))
                T1l_=ybl_.shape[1]; pl_=((torch.arange(T1l_,device=dev)+1)//max(1,L)).clamp(max=C-1)
                fpl_=(torch.zeros(k_l,C,device=dev).index_add_(1,pl_,dgp_)>0).float()[:,pl_]; wl2_=wl_*(1.0+(float(cfg.cd_reuse_upw)-1.0)*fpl_)
            sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
            try: lpl_=F.log_softmax(model(xb_[:,:-1],dom).float(),-1)
            finally: _carry_load(sv4_); _uh_load(model,suh4_); _NGPD['on']=False
            kll_=(tgl_.exp()*(tgl_-lpl_)).sum(-1); KLR_=(kll_*wl2_).sum(); NPR_=wl2_.numel()
            with torch.no_grad():
                klw_=((kll_*wl_).sum(1)/wl_.sum(1).clamp_min(1)).tolist(); dgl_=dg_.tolist(); S_=LB_['st']; drop_=set()
                for e_,d_,kw_ in zip(sel_,dgl_,klw_):
                    S_[0]+=1; S_[5]+=d_; e_['n']+=1
                    if kw_<float(cfg.cd_loop_kldrop): S_[4]+=1; drop_.add(id(e_)); continue
                    if d_>float(cfg.cd_loop_fail):
                        if e_['second']: S_[3]+=1; drop_.add(id(e_)); continue
                        e_['second']=True; e_['due']=st_l+1; S_[2]+=1; continue
                    e_['second']=False
                    if d_<=float(cfg.cd_loop_easy): S_[1]+=1; e_['iv']=e_['iv']*2
                    else: S_[2]+=1; e_['iv']=max(2,e_['iv']//2)
                    if e_['n']>=int(cfg.cd_loop_maxuse): drop_.add(id(e_)); continue
                    e_['due']=st_l+e_['iv']
                LB_['e']=[e_ for e_ in LB_['e'] if id(e_) not in drop_]
    if int(cfg.cd_sig_every)>0:                                                 # REPLAY-SIGNAL DIAGNOSTIC (no training effect): per piece, the learning signal
        sc_=model.__dict__.get('_rsig_n',0); model.__dict__['_rsig_n']=sc_+1    #   (KL to the trace-reading teacher over its teaching positions) and its
        if sc_%int(cfg.cd_sig_every)==0:                                        #   SURPRISE at experience (the plain model's loss on the bytes)
            with torch.no_grad():
                T1_=klp_.shape[1]; pid_=((torch.arange(T1_,device=klp_.device)+1)//max(1,L)).clamp(max=C-1)
                ce_=-lp.gather(-1,bj_[:,1:,None]).squeeze(-1)/LOG2; w_s=wp_.float()
                ks_=torch.zeros(klp_.shape[0],C,device=klp_.device).index_add_(1,pid_,(klp_*w_s).float())
                cs_=torch.zeros_like(ks_).index_add_(1,pid_,(ce_*w_s).float()); ns_=torch.zeros_like(ks_).index_add_(1,pid_,w_s)
                ok_=ns_>0; RS_=model.__dict__.setdefault('_rsig',[])
                RS_.append(torch.stack([(ks_/ns_.clamp_min(1))[ok_],(cs_/ns_.clamp_min(1))[ok_],ns_[ok_]]).half())
    if cfg.cd_kl_norm: kl=(klp_*wp_).sum()/wp_.sum().clamp_min(1.0)               # v9 FIX 2a: averaged over the positions that teach
    else: kl=(klp_*wp_).mean() if iw_==0 else (klp_*wp_).sum()/wp_.sum().clamp_min(1e-6)   # (iw_ = 0: EXACT replays only; fewer exact positions teach less)
    if fw_<1.0: kl=kl*fw_                                                       # v9e LOW-EXACT FLOOR
    if model.__dict__.get('_ctrl_bytes') or cfg.cd_bytes_all:
        ceb_=(-(lp.gather(-1,bj_[:,1:,None]).squeeze(-1))*wp_).mean(); bw_=1.0 if model.__dict__.get('_ctrl_bytes') else float(cfg.cd_bytes_w)
        kl=ceb_ if bw_>=1.0 else bw_*ceb_+(1.0-bw_)*kl   # sleep_ctrl_bytes: next-byte CE on the fresh checked replays only
    if NPB_>0: kl=(kl*wp_.numel()+KLB_)/(wp_.numel()+NPB_)                      # v9f: averaged over generated + buffered windows
    if int(cfg.cd_store_cap)>0:                                                 # BOUNDED LONG-TERM STORE (admission / spaced re-checks / retirement / cap)
        SB_=model.__dict__.setdefault('_sbuf',{'e':[],'st':{}}); st_s=model.__dict__.get('_sstep',0)+1; model.__dict__['_sstep']=st_s
        ST_=SB_['st']; snp_=model.__dict__.pop('_snap_last',None); tqy_=model.__dict__.pop('_tqy',None)
        if snp_ is not None and tqy_ is not None:
            with torch.no_grad():
                rl_=_store_score(tqy_,lp.gather(-1,bj_[:,1:,None]).squeeze(-1),(wp_>0).float(),C,L).tolist()
            rs_=model.__dict__.setdefault('_s_rs',np.random.RandomState(int(cfg.seed)+131)); ka_=int(round(len(rl_)*float(cfg.cd_store_adm)))
            if cfg.cd_store_admit=='rand': top_=set(rs_.permutation(len(rl_))[:ka_].tolist())
            else: top_=set(i_ for i_ in sorted(range(len(rl_)),key=lambda i_:-rl_[i_])[:ka_] if rl_[i_]>0)
            rnd_=set(i_ for i_ in range(len(rl_)) if rs_.rand()<float(cfg.cd_store_rand))-top_
            if float(cfg.sem_share)>0 and (top_|rnd_):                          # SEMANTIC COPIES (trace-free): bytes + evidence-blended teacher targets
                with torch.no_grad():
                    SM_=model.__dict__.setdefault('_sem',{}).setdefault(int(dom),{'x':[],'v':[],'i':[],'p':0}); kb_=int(cfg.bank_targets)
                    for r_ in sorted(top_|rnd_):
                        if float((wp_[r_]>0).float().mean())<float(cfg.sem_minver): continue
                        lt_=torch.cat([tg2[r_],tg2[r_][-1:]],0)                    #   (T positions; the last one repeats)
                        v_,i_=torch.topk(lt_.float(),kb_,dim=-1)
                        it_=(bj_[r_].to(torch.uint8).cpu(),v_.to(torch.float16).cpu(),i_.to(torch.uint8).cpu())
                        if len(SM_['x'])<int(cfg.sem_cap): SM_['x'].append(it_[0]); SM_['v'].append(it_[1]); SM_['i'].append(it_[2])
                        else: j_=SM_['p']%int(cfg.sem_cap); SM_['x'][j_],SM_['v'][j_],SM_['i'][j_]=it_; SM_['p']+=1
                        ST_['sem']=ST_.get('sem',0)+1
            for r_ in sorted(top_|rnd_):
                SB_['e'].append({'dom':int(dom),'due':st_s+int(cfg.cd_reuse_lag),'iv':int(cfg.cd_reuse_lag),'n':0,'x':bj_[r_].clone(),'w':wp_[r_].clone(),
                                 'tr':_cd_rows(snp_,torch.arange(r_*C,(r_+1)*C,device=dev),n),'v':float(rl_[r_]),'v0':float(rl_[r_])})
            ST_['seen']=ST_.get('seen',0)+len(rl_); ST_['adm']=ST_.get('adm',0)+len(top_); ST_['admr']=ST_.get('admr',0)+len(rnd_)
            while len(SB_['e'])>int(cfg.cd_store_cap):                           #   full: evict the MOST CONSOLIDATED (lowest current signal)
                j_=min(range(len(SB_['e'])),key=lambda i_:SB_['e'][i_]['v']); SB_['e'].pop(j_); ST_['evict']=ST_.get('evict',0)+1
        due_=[e_ for e_ in SB_['e'] if e_['due']<=st_s]
        if due_:
            due_.sort(key=lambda e_:e_['due']); sel_=due_[:int(cfg.cd_store_k)]; drop_=set()
            for d_ in sorted({e_['dom'] for e_ in sel_}):
                sd_=[e_ for e_ in sel_ if e_['dom']==d_]; k_l=len(sd_)
                with torch.no_grad():
                    trs_=_cd_cat([e_['tr'] for e_ in sd_]) if k_l>1 else sd_[0]['tr']
                    xb_=torch.stack([e_['x'] for e_ in sd_]); pcs_=xb_.reshape(k_l*C,L)
                    Yl_=torch.cat([torch.full((k_l*C,1),M,dtype=torch.long,device=dev),pcs_[:,:-1]],1)
                    svl_=_carry_save(); wasl_=model.training; model.eval(); fsl_=model.feat_scale.clone(); _CARRY['fac']=bool(cfg.cd_fac)
                    try:
                        with _cd_weights(model):                                    #   the frozen teacher reads the STORED traces (its own domain)
                            _CARRY.update(on=True,nr=k_l*C,st=trs_,new={},rows=None); tgl_=F.log_softmax(model(Yl_,d_).float(),-1)
                    finally: _CARRY.pop('fac',None); _carry_load(svl_); model.feat_scale.copy_(fsl_); model.train(wasl_)
                    tgl_=tgl_.reshape(k_l,C*L,-1)[:,1:]; ybl_=xb_[:,1:]; wl_=torch.stack([e_['w'] for e_ in sd_]).float(); vml_=(wl_>0).float()
                    tqyl_=tgl_.gather(-1,ybl_[...,None]).squeeze(-1)
                    if float(cfg.cd_evid_w)>0:
                        dgp_=(tgl_.argmax(-1)!=ybl_).float()*vml_; ptl_=tgl_.exp(); lam2_=(float(cfg.cd_evid_w)*dgp_).unsqueeze(-1)
                        tgl_=torch.log(((1-lam2_)*ptl_+lam2_*F.one_hot(ybl_,ptl_.shape[-1]).to(ptl_.dtype)).clamp_min(1e-12))
                sv4_=_carry_save(); suh4_=_uh_save(model); _CARRY.update(on=False,st={},new={}); _NGPD['on']=bool(cfg.cd_ngp_frozen)
                try: lpl_=F.log_softmax(model(xb_[:,:-1],d_).float(),-1)            #   the student, its own domain's head (old domains too)
                finally: _carry_load(sv4_); _uh_load(model,suh4_); _NGPD['on']=False
                kll_=(tgl_.exp()*(tgl_-lpl_)).sum(-1); KLR_=KLR_+(kll_*wl_).sum(); NPR_+=wl_.numel()
                with torch.no_grad(): rn_=_store_score(tqyl_,lpl_.gather(-1,ybl_[...,None]).squeeze(-1),vml_,C,L).tolist()
                for e_,r_ in zip(sd_,rn_):
                    e_['n']+=1; ST_['chk']=ST_.get('chk',0)+1; ST_['chk_old']=ST_.get('chk_old',0)+(1 if e_['dom']!=int(dom) else 0)
                    if r_<=max(1e-4,float(cfg.cd_store_retire)*e_['v0']): drop_.add(id(e_)); ST_['ret']=ST_.get('ret',0)+1; ST_['ret_n']=ST_.get('ret_n',0)+e_['n']; continue
                    e_['iv']=e_['iv']*2 if r_<=e_['v'] else max(2,e_['iv']//2)      #   consolidating: wait longer; being forgotten: sooner
                    e_['v']=r_; e_['due']=st_s+e_['iv']
            SB_['e']=[e_ for e_ in SB_['e'] if id(e_) not in drop_]
    if cfg.inf_on and model.__dict__.get('_tqy') is not None: inf_store_add(model,cfg,bj_,int(model.__dict__.get('_cur_dom',dom)),wp_)   # cg20 store (verified)
    if int(cfg.nrem_res)>0 and model.__dict__.get('_tqy') is not None:            # NREM v1 RESERVOIR (uniform over the visit's verified windows)
        with torch.no_grad():
            RV_=model.__dict__.setdefault('_nres',{'dom':None,'e':[],'seen':0,'ret':0,'rep':0})
            td_=int(model.__dict__.get('_cur_dom',dom))                       # the TRUE domain (slots can be shared); windows are taught under their slot 'd'
            if RV_.get('td')!=td_:
                if float(cfg.nrem_mix)>0 and RV_.get('td') is not None and RV_['e']: model.__dict__.setdefault('_nres_bank',{})[RV_['td']]=RV_['e']   # keep it for cross-domain NREM
                model.__dict__.setdefault('_nres_bank',{}).pop(td_,None)
                RV_.update(dom=int(dom),td=td_,e=[],seen=0,ret=0,rep=0,rep_o=0,ret_o=0)
            tq_=model.__dict__['_tqy']; vm_=(wp_>0).float()
            sc_=_store_score(tq_,lp.gather(-1,bj_[:,1:,None]).squeeze(-1),vm_,C,L).tolist()
            rsr_=model.__dict__.setdefault('_nres_rs',np.random.RandomState(int(cfg.seed)+313)); kb_=int(cfg.bank_targets)
            for r_ in range(bj_.shape[0]):
                RV_['seen']+=1; j_=len(RV_['e']) if len(RV_['e'])<int(cfg.nrem_res) else rsr_.randint(RV_['seen'])
                if j_>=int(cfg.nrem_res): continue
                v_,i_=torch.topk(tg2[r_].float(),kb_,dim=-1)
                xr_=bj_[r_].clone()
                if cfg.nrem_real:
                    PR_=model.__dict__.get('_p1win',{}).get(td_)
                    if PR_ and PR_['r']:
                        rw_=PR_['r'][rsr_.randint(len(PR_['r']))]; Wn_=xr_.numel()
                        if rw_.numel()>=Wn_: o_=rsr_.randint(rw_.numel()-Wn_+1); xr_=rw_[o_:o_+Wn_].to(xr_.device,xr_.dtype).clone()
                en_={'d':int(dom),'x':xr_,'w':wp_[r_].clone(),'tv':v_.half(),'ti':i_.to(torch.uint8),'q':tq_[r_].half(),'v0':float(sc_[r_]),'v':float(sc_[r_]),'n':0}
                if j_<len(RV_['e']): RV_['e'][j_]=en_
                else: RV_['e'].append(en_)
    if NPR_>0: kl=(kl*wp_.numel()+KLR_)/(wp_.numel()+NPR_)                      # reuse / maintenance: averaged over fresh + reused windows
    _V97['fw'].append(fw_)
    if cfg.cd_lr_teach and model.__dict__.get('_cdf') is not None:             # v9 FIX 2b: step scaled by the teaching share
        tsh_=max(1e-3,float(wp_.float().mean())); model.__dict__['_cdf']=float(model.__dict__['_cdf'])*tsh_; _CDR[2]+=tsh_; _CDR[3]+=1
    st=model.__dict__.setdefault('_cd_stat',[0.0,0,0.0,0.0,0.0,0.0,0.0])
    if cfg.cd_nosync:                                                          # v9d: accumulated on the device, read at the visit end
        st[0]=st[0]+kl.detach(); st[1]+=1; st[2]=st[2]+ex_.float().mean(); st[3]+=tries/n; st[4]=st[4]+(best==X).float().mean(); st[5]+=eps
        st[6]=st[6]+(ex_==(best==X).all(1)).float().mean()
    else:
        st[0]+=float(kl.detach()); st[1]+=1; st[2]+=float(ex_.float().mean()); st[3]+=tries/n; st[4]+=float((best==X).float().mean()); st[5]+=eps
        st[6]+=float((ex_==(best==X).all(1)).float().mean())                     # exactness test agrees with the TRUE bytes (diagnostic)
    if float(cfg.cd_recon_w)>0 and not model.__dict__.get('_ctrl_bytes'):     # v6: REPLAY TRAINING ON EXACT REPLAYS (live model, no real bytes)
        ix_=(ex_&chkd_).nonzero().flatten()                                    # v7: CHECKED exact replays only (never gate-accepted)
        if len(ix_)>0:
            kr_=int(cfg.cd_recon_rows)
            if cfg.rs_sel and len(ix_)>kr_:                                    # v8: borderline + random among the checked exact replays
                pl_=ix_[torch.randperm(len(ix_),generator=g_,device=dev)[:max(kr_,int(cfg.rs_pool)*kr_)]]
                Xr_=_rs_pick(model,cfg,best[pl_].contiguous(),kr_,dom,g_)
            else: Xr_=best[ix_[torch.randperm(len(ix_),generator=g_,device=dev)[:kr_]]].contiguous()
            rl_=_recon_loss_mark(model,cfg,None,dom,X=Xr_)
            kl=kl+float(cfg.cd_recon_w)*rl_
            if len(st)<9: st.extend([0.0]*(9-len(st)))
            st[7]=st[7]+(rl_.detach() if cfg.cd_nosync else float(rl_.detach())); st[8]+=1
    return kl
_RDIAG={'rec':False,'v':[]}                                          # replay diagnostics: memory decay over a segment (recorded in the carry path)
def replay_diag(model,cfg,vals,NDOM,_S,n=16):
    with _rp_tags(model): return _replay_diag(model,cfg,vals,NDOM,_S,n)
def _replay_diag(model,cfg,vals,NDOM,_S,n=16):
    """ENCODER/DECODER DIAGNOSTICS (recon_mark runs): per segment length L, teacher-forced replay BPC of [mark]+segment[:-1] -> segment read from
    the segment's OWN end-of-segment memory vs from an empty memory (information the trace carries), greedy and sampled (T=0.7) replay byte
    accuracy, and the memory decay across the segment; plus which blocks carry a memory and whether replay training reaches the encoder."""
    print("\n=== ENCODER/DECODER DIAGNOSTICS (replay of a segment from its own end-of-segment memory) ===")
    was=model.training; model.eval(); fs=model.feat_scale.clone(); kh=getattr(model,'_keep_h',False); model._keep_h=False
    sv=_carry_save(); M=int(cfg.cd_mark); dev=DEVICE; g=torch.Generator(device=dev); g.manual_seed(99); res={}; cur0_=_XDG.get('cur')
    print(f"  replay-training gradient into the encoder: {'ON' if _RC_GRAD['on'] else 'OFF (detached snapshot)'}")
    try:
        data={}
        for d in range(NDOM):
            g_=vals[d]
            if hasattr(g_,'reset'): g_.reset()
            data[d]=torch.cat([g_()[0] for _ in range(2)],0).to(dev)
        X0=data[0]; T0=X0.shape[1]; Ls=[L_ for L_ in (8,12,16,24,32,48,64,96,128) if L_<=T0]
        with torch.no_grad():
            Xs=X0[:n,:min(32,T0)].long(); snap=_cd_encode(model,Xs)
            print("  blocks: "+" | ".join(f"b{i} "+("local filter" if getattr(b,'loc',False) else "attention" if getattr(b,'lattn',False) else "delta memory")
                                           +f" {getattr(b,'mh',1)} heads x dim {b.d_state//getattr(b,'mh',1)}" for i,b in enumerate(model.blocks)))
            print(f"  trace size: {sum(v_.numel() for v_ in snap.values())//n} numbers per segment, in {len(snap)} blocks")
        with torch.enable_grad():
            sl=_S(0); _CARRY.update(on=True,nr=n,st={},new={},rows=None,grad=True); model.features(Xs); sn=dict(_CARRY['new'])
            _CARRY.update(on=True,nr=n,st=sn,new={},rows=None,grad=False)
            Xin=torch.cat([torch.full((n,1),M,dtype=torch.long,device=dev),Xs[:,:-1]],1)
            l_=F.cross_entropy(model(Xin,sl).reshape(-1,256).float(),Xs.reshape(-1)); _CARRY.pop('grad',None)
            tr=[v_ for k_,v_ in sn.items() if not isinstance(k_,tuple)]; rg=all(t_.requires_grad for t_ in tr)
            gn=sum(float(gr.norm()**2) for gr in torch.autograd.grad(l_,tr,allow_unused=True) if gr is not None)**0.5 if rg else 0.0
            print(f"  gradient check: snapshot differentiable {rg} | grad norm at the snapshot {gn:.3e}"); model.zero_grad(set_to_none=True)
        print(f"  {'L':>4} | {'empty BPC':>9} {'trace BPC':>9} {'carries':>8} | {'greedy acc':>10} {'sampled':>8} | {'decay over L':>12}")
        for Lx in Ls:
            e_=t_=a_=s_=0.0; dec=[]
            for d in range(NDOM):
                x=data[d]; B,T=x.shape
                r_=torch.randint(0,B,(n,),generator=g,device=dev); o_=torch.randint(0,T-Lx+1,(n,),generator=g,device=dev)
                X=torch.stack([x[r_[i],o_[i]:o_[i]+Lx] for i in range(n)]).long(); sl=_S(d)
                Xin=torch.cat([torch.full((n,1),M,dtype=torch.long,device=dev),X[:,:-1]],1)
                if isinstance(sl,int): _XDG['cur']=sl                     # this domain's private units / FiLM code (not the last-trained domain's)
                with torch.no_grad():
                    _RDIAG.update(rec=True,v=[]); snap=_cd_encode(model,X); _RDIAG['rec']=False
                    _CARRY.update(on=False,st={},new={})
                    e_+=float(F.cross_entropy(model(Xin,sl).reshape(-1,256).float(),X.reshape(-1)))/LOG2
                    _RDIAG.update(rec=True,v=[]); _CARRY.update(on=True,nr=n,st=snap,new={},rows=None)
                    t_+=float(F.cross_entropy(model(Xin,sl).reshape(-1,256).float(),X.reshape(-1)))/LOG2
                    _RDIAG['rec']=False; dec+=_RDIAG['v']
                    for mode in ('greedy','sampled'):
                        Y=torch.full((n,1),M,dtype=torch.long,device=dev)
                        for _ in range(Lx):
                            _CARRY.update(on=True,nr=n,st=snap,new={},rows=None); lg=model(Y,sl)[:,-1].float()
                            nx=lg.argmax(-1,keepdim=True) if mode=='greedy' else torch.multinomial(F.softmax(lg/0.7,-1),1,generator=g)
                            Y=torch.cat([Y,nx],1)
                        acc=float((Y[:,1:]==X).float().mean())
                        if mode=='greedy': a_+=acc
                        else: s_+=acc
            k_=NDOM; res[Lx]=(e_/k_,t_/k_,a_/k_,s_/k_)
            print(f"  {Lx:4d} | {e_/k_:9.3f} {t_/k_:9.3f} {(e_-t_)/k_:+8.3f} | {a_/k_:10.2f} {s_/k_:8.2f} | {sum(dec)/max(1,len(dec)):12.3f}")
        print("  [replay diag] "+" ".join(f"L{k}:carries {v[0]-v[1]:+.3f} greedy {v[2]:.2f}" for k,v in res.items()))
    finally:
        _carry_load(sv); model.feat_scale.copy_(fs); model.train(was); model._keep_h=kh; _XDG['cur']=cur0_
    return res
@torch.no_grad()
def recon_fidelity(model,cfg,vals,NDOM,_S,n=8):
    print("\n=== REPLAY FIDELITY: held-out BPC of a window read from its OWN end-of-window memory (snapshot) vs from an empty memory ===")
    out={}; was=model.training; model.eval()
    for d in range(NDOM):
        sl_=_S(d); g_=vals[d]
        if hasattr(g_,'reset'): g_.reset()
        a_=b_=0.0
        for _ in range(n):
            x,y=g_(); Ts_=int(cfg.seq_len); x,y=x[:,-Ts_:],y[:,-Ts_:]; sv=_carry_save()
            try:
                a_+=float(F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)))/n
                _CARRY.update(on=True,nr=x.shape[0],st={},new={},rows=None); model.features(x); snap=dict(_CARRY['new'])
                _CARRY.update(on=True,nr=x.shape[0],st=snap,new={},rows=None)
                b_+=float(F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)))/n
            finally: _carry_load(sv)
        out[d]=(a_/LOG2,b_/LOG2); print(f"  {PILE_DOMAINS[d][:12]:12} empty memory {a_/LOG2:.4f} | own snapshot {b_/LOG2:.4f} | replay gain {(b_-a_)/LOG2:+.4f} BPC")
    print(f"  [replay fidelity] mean gain from the window's own snapshot: {np.mean([b-a for a,b in out.values()]):+.4f} BPC")
    model.train(was); return out
def oracle_final(model,cfg,vals,trs,NDOM,fin,_S):
    """ORACLE LINEAR MEMORY: per domain, the exact pooled least-squares delta-rule memory (and optionally a BPC-optimised one) as the slot's
    initial state, weights frozen; held-out BPC change vs no memory."""
    print("\n=== ORACLE LINEAR MEMORY (final weights frozen; slot initial state = pooled least-squares S* per head) ===")
    sm0_=cfg.slot_mem; cfg.slot_mem=True; old_=dict(model.__dict__.get('_smem',{})); model.__dict__['_smem']={}
    res={}
    try:
        for d in range(NDOM):
            sl_=_S(d); _ORACLE['st']={}; _ORACLE['on']=True; model.eval()
            try:
                with torch.no_grad():
                    for _ in range(int(cfg.oracle_batches)): x,_y=trs[d](); model(x,sl_)
            finally: _ORACLE['on']=False
            model.train(); st_=_ORACLE['st']; _ORACLE['st']={}
            base_=bpc(model,vals[d],sl_); row={'none':base_}
            best_=(base_,None,None)
            for lam in cfg.oracle_lams:
                Sm={}
                for bid,(Ckk,Cvk) in st_.items():
                    dk_=Ckk.shape[-1]; r_=lam*Ckk.diagonal(dim1=-2,dim2=-1).sum(-1)/dk_
                    Sm[bid]=torch.linalg.solve((Ckk+r_[:,None,None]*torch.eye(dk_,device=Ckk.device)).transpose(-1,-2),Cvk.transpose(-1,-2)).transpose(-1,-2)
                for sc_ in cfg.oracle_scales:
                    Sc={k_:v_*float(sc_) for k_,v_ in Sm.items()}
                    model.__dict__['_smem']={sl_:Sc}; b_=bpc(model,vals[d],sl_); row[f'ls{lam:g}x{sc_:g}']=b_
                    if b_<best_[0]: best_=(b_,f'{lam:g} x{sc_:g}',Sc)
            if int(cfg.oracle_gd)>0:                                            # BPC-OPTIMISED initial memory (ceiling of any state)
                Sm={k_:(v_.clone() if best_[2] is not None else torch.zeros_like(v_)) for k_,v_ in (best_[2] or {bid:st_[bid][1] for bid in st_}).items()}
                P_=[torch.nn.Parameter(v_) for v_ in Sm.values()]; ks_=list(Sm.keys())
                rq_=[(p_,p_.requires_grad) for p_ in model.parameters()]
                for p_,_ in rq_: p_.requires_grad_(False)
                og_=torch.optim.Adam(P_,lr=float(cfg.oracle_gd_lr))
                try:
                    for it_ in range(int(cfg.oracle_gd)):
                        model.__dict__['_smem']={sl_:dict(zip(ks_,P_))}; x,y=trs[d]()
                        l_=F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)); og_.zero_grad(); l_.backward(); og_.step()
                finally:
                    for p_,r0_ in rq_: p_.requires_grad_(r0_)
                model.__dict__['_smem']={sl_:{k_:p_.detach() for k_,p_ in zip(ks_,P_)}}; row['gd']=bpc(model,vals[d],sl_)
            if cfg.oracle_persist:                                              # PERSISTENT-READ ORACLE
                model.__dict__['_smem']={}; bestp_=(base_,None,None)
                try:
                    for lam in cfg.oracle_lams:
                        Sm={}
                        for bid,(Ckk,Cvk) in st_.items():
                            dk_=Ckk.shape[-1]; r_=lam*Ckk.diagonal(dim1=-2,dim2=-1).sum(-1)/dk_
                            Sm[bid]=torch.linalg.solve((Ckk+r_[:,None,None]*torch.eye(dk_,device=Ckk.device)).transpose(-1,-2),Cvk.transpose(-1,-2)).transpose(-1,-2)
                        for sc_ in cfg.oracle_pscales:
                            _PMEM['S']={k_:v_*float(sc_) for k_,v_ in Sm.items()}; _PMEM['on']=True
                            b_=bpc(model,vals[d],sl_); row[f'p{lam:g}x{sc_:g}']=b_
                            if b_<bestp_[0]: bestp_=(b_,f'{lam:g} x{sc_:g}',dict(_PMEM['S']))
                    if int(cfg.oracle_gd)>0:                                    # BPC-optimised persistent memory
                        Sm={k_:v_.clone() for k_,v_ in (bestp_[2] or {bid:torch.zeros_like(st_[bid][1]) for bid in st_}).items()}
                        P_=[torch.nn.Parameter(v_) for v_ in Sm.values()]; ks_=list(Sm.keys())
                        rq_=[(p_,p_.requires_grad) for p_ in model.parameters()]
                        for p_,_ in rq_: p_.requires_grad_(False)
                        og_=torch.optim.Adam(P_,lr=float(cfg.oracle_gd_lr))
                        try:
                            for it_ in range(int(cfg.oracle_gd)):
                                _PMEM['S']=dict(zip(ks_,P_)); _PMEM['on']=True; x,y=trs[d]()
                                l_=F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)); og_.zero_grad(); l_.backward(); og_.step()
                        finally:
                            for p_,r0_ in rq_: p_.requires_grad_(r0_)
                        _PMEM['S']={k_:p_.detach() for k_,p_ in zip(ks_,P_)}; _PMEM['on']=True; row['gdp']=bpc(model,vals[d],sl_)
                finally: _PMEM['on']=False; _PMEM['S']={}
                print(f"  {'':12} persistent read: best least-squares {bestp_[0]-base_:+.4f}"+(f" ({bestp_[1]})" if bestp_[1] else "")
                      +(f" | BPC-optimised {row['gdp']-base_:+.4f}" if 'gdp' in row else ""))
            for rk_ in tuple(cfg.oracle_adapter):                               # LOW-RANK MLP ADAPTER ORACLE
                model.__dict__['_smem']={}; gA_=torch.Generator(device='cpu'); gA_.manual_seed(cfg.seed+31+d)
                P_={}
                for b_ in model.blocks:
                    dm_=b_.d_model
                    P_[id(b_)]=(torch.nn.Parameter((torch.randn(int(rk_),dm_,generator=gA_)/math.sqrt(dm_)).to(model.embed.weight.device)),
                                torch.nn.Parameter(torch.zeros(dm_,int(rk_),device=model.embed.weight.device)))
                rq_=[(p_,p_.requires_grad) for p_ in model.parameters()]
                for p_,_ in rq_: p_.requires_grad_(False)
                og_=torch.optim.Adam([t_ for ab_ in P_.values() for t_ in ab_],lr=float(cfg.oracle_ad_lr))
                _ADPT['P']=P_; _ADPT['on']=True
                try:
                    for it_ in range(int(cfg.oracle_ad_steps)):
                        x,y=trs[d](); l_=F.cross_entropy(model(x,sl_).reshape(-1,256),y.reshape(-1)); og_.zero_grad(); l_.backward(); og_.step()
                    row[f'ad{int(rk_)}']=bpc(model,vals[d],sl_)
                finally:
                    _ADPT['on']=False; _ADPT['P']={}
                    for p_,r0_ in rq_: p_.requires_grad_(r0_)
                print(f"  {'':12} low-rank MLP adapter rank {int(rk_)}: {row[f'ad{int(rk_)}']-base_:+.4f} ({cfg.oracle_ad_steps} steps)")
            res[d]=row
            print(f"  {PILE_DOMAINS[d][:12]:12} no memory {base_:.4f} | "+" | ".join(f"{k_} {v_-base_:+.4f}" for k_,v_ in row.items() if k_!='none')
                  +(f" | best {best_[1]}" if best_[1] is not None else " | none helps"))
        bl_=lambda r_: min(v_ for k_,v_ in r_.items() if k_.startswith('ls'))
        print(f"  [oracle] mean BPC change, best least-squares memory per domain: {np.mean([bl_(r_)-r_['none'] for r_ in res.values()]):+.4f}"
              +(f" | BPC-optimised memory: {np.mean([r_['gd']-r_['none'] for r_ in res.values()]):+.4f}" if int(cfg.oracle_gd)>0 else ""))
        for rk_ in tuple(cfg.oracle_adapter):
            print(f"  [oracle, low-rank MLP adapter rank {int(rk_)}] mean BPC change: {np.mean([r_[f'ad{int(rk_)}']-r_['none'] for r_ in res.values()]):+.4f}")
        if cfg.oracle_persist:
            bp_=lambda r_: min([v_ for k_,v_ in r_.items() if k_.startswith('p')]+[r_['none']])
            print(f"  [oracle, persistent read] mean BPC change, best least-squares: {np.mean([bp_(r_)-r_['none'] for r_ in res.values()]):+.4f}"
                  +(f" | BPC-optimised: {np.mean([r_['gdp']-r_['none'] for r_ in res.values()]):+.4f}" if int(cfg.oracle_gd)>0 else ""))
    finally:
        cfg.slot_mem=sm0_; model.__dict__['_smem']=old_
    return res

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_PRF': 'platypus.profiling',
}
