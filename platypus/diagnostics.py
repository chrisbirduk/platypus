"""Topic identification, reconstruction and forgetting diagnostics."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from platypus.config import (
    PILE_DOMAINS,
)
from platypus.gating import (
    _PG_KEY, _f32,
)
from platypus.capacity import (
    _lab, _pmap,
)


# ---------------- TOPIC IDENTIFICATION FROM BYTE CONTEXT (diagnostic: no effect on training) ----------------

_RECON_LAYERS=('Wk','Wv','gate','Wout','mlp_in','mlp_out')

@torch.no_grad()
def _recon2_once(model, cfg, d, xd):
    """One v2 reconsolidation step on old domain d's dreams xd. Returns per-layer (weighted error ratio) and the share of
    dream tokens corrected."""
    mods=[(li,nm,getattr(model.blocks[li],nm)) for li in cfg.recon_blocks if li<len(model.blocks) for nm in cfg.recon_layers]
    cc=getattr(model,'_cur_cov',None) or {}; uc=model.__dict__.setdefault('_recon_unew',{})
    for li,nm,m in mods:
        if id(m.weight) not in uc:
            c_=cc.get((li,_PG_KEY[nm]))
            uc[id(m.weight)]=None if (c_ is None or c_[1]==0) else _subspaces(c_[0]/c_[1],[cfg.recon_new_energy])[cfg.recon_new_energy][0]
    if not mods: return None
    rec={}
    def mk(tag,li,nm):
        def hk(m,inp,out): rec[(tag,li,nm)]=(inp[0].reshape(-1,inp[0].shape[-1]).float(),out.reshape(-1,out.shape[-1]).float())
        return _f32(hk)
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    P=_pmap(model); S=getattr(model,'_shunt',None) or {}; Sd={k:v for k,v in S.items() if k in P and P[k].shape==v.shape}
    for k,v in Sd.items(): P[k].sub_(v)                                        # main weights
    hooks=[m.register_forward_hook(mk('main',li,nm)) for li,nm,m in mods]; lm=F.log_softmax(model.heads[d](model.features(xd)).float(),-1)
    for h in hooks: h.remove()
    for k,v in Sd.items(): P[k].add_(v)                                        # live weights
    hooks=[m.register_forward_hook(mk('live',li,nm)) for li,nm,m in mods]; ll=F.log_softmax(model.heads[d](model.features(xd)).float(),-1)
    for h in hooks: h.remove()
    model.feat_scale.copy_(fs); model.train(was)
    B,T=xd.shape; w=torch.ones(B,T,device=xd.device)
    kl=(lm.exp()*(lm-ll)).sum(-1)                                              # per-token KL main -> live
    w=w/(1.0+kl/cfg.recon_cap)
    tgt=xd[:,1:]
    dn=-(ll[:,:-1].gather(-1,tgt[...,None])[...,0]-lm[:,:-1].gather(-1,tgt[...,None])[...,0])   # nll_live - nll_main
    worse=torch.zeros(B,T,device=xd.device); worse[:,:-1]=(dn>0).float(); w=w*worse
    wf=w.reshape(-1); W_=float(wf.sum())
    if W_<1e-6: return {'share':0.0,'err':{}}
    out={}
    for li,nm,m in mods:
        _,v_old=rec[('main',li,nm)]; k_,y=rec[('live',li,nm)]; E=(v_old-y)*wf[:,None]
        U=model._recon_unew.get(id(m.weight))
        if U is not None: U=U.to(k_.dtype); k_=k_-(k_@U)@U.T
        nrm=float(((k_*k_).sum(1)*wf).sum()/W_)+1e-6
        dW=cfg.recon_lr*(E.T@k_)/(W_*nrm); m.weight.add_(dW.to(m.weight.dtype))
        if id(m.weight) in S: S[id(m.weight)].add_(dW.to(m.weight.dtype))
        out[(li,nm)]=(float(((v_old-y)**2*wf[:,None]).sum()/(W_*v_old.shape[1])),float((v_old*v_old).mean()))
    return {'share':float((wf>0).float().mean()),'err':out}

def _recon2_stat(model, r):
    if not r: return
    st=model.__dict__.setdefault('_recon_stat',{})
    for (li,nm),(e,v) in r['err'].items():
        a=st.setdefault(li,[0.0,0.0,0]); a[0]+=e; a[1]+=v; a[2]+=1
    sh=model.__dict__.setdefault('_recon_share',[0.0,0]); sh[0]+=r['share']; sh[1]+=1

@torch.no_grad()
def _dream_kl_now(model, cfg, d, xd):
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    P=_pmap(model); S=getattr(model,'_shunt',None) or {}; Sd={k:v for k,v in S.items() if k in P and P[k].shape==v.shape}
    for k,v in Sd.items(): P[k].sub_(v)
    lm=F.log_softmax(model.heads[d](model.features(xd)).float(),-1)
    for k,v in Sd.items(): P[k].add_(v)
    ll=F.log_softmax(model.heads[d](model.features(xd)).float(),-1)
    model.feat_scale.copy_(fs); model.train(was)
    return float((lm.exp()*(lm-ll)).sum(-1).mean())

@torch.no_grad()
def recon_sleep(model, cfg, x, y, dom_id):
    """v2 SLEEP phase at domain end, before the shunt merge: recon_sleep correction steps over the old domains' dreams
    (cycled, random batches from each pool), nothing else learned. Reports dream KL (old heads, live vs main weights) and the
    current domain's loss on this batch before and after."""
    pools=getattr(model,'_dreams',None)
    if not pools: return
    keys=[d for d in sorted(pools) if pools[d]]
    if not keys: return
    rng=np.random.RandomState(cfg.seed*1000+dom_id+77)
    model.__dict__.pop('_recon_unew',None)
    probe={d:torch.stack(pools[d][:cfg.dream_batch]) for d in keys}
    def cur():
        was=model.training; model.eval(); fs=model.feat_scale.clone()
        v=float(F.cross_entropy(model.heads[dom_id](model.features(x)).reshape(-1,256).float(),y.reshape(-1)))/math.log(2)
        model.feat_scale.copy_(fs); model.train(was); return v
    kb={d:_dream_kl_now(model,cfg,d,probe[d]) for d in keys}; cb=cur()
    for i in range(cfg.recon_sleep):
        d=keys[i%len(keys)]; pl=pools[d]
        xd=torch.stack([pl[j] for j in rng.randint(len(pl),size=cfg.dream_batch)])
        _recon2_stat(model,_recon2_once(model,cfg,d,xd))
    ka={d:_dream_kl_now(model,cfg,d,probe[d]) for d in keys}; ca=cur()
    model.__dict__.setdefault('_sleep_log',{})[dom_id]={'kl_before':kb,'kl_after':ka,'cur_before':cb,'cur_after':ca}
    print(f"  [sleep] {cfg.recon_sleep} reconsolidation steps before the merge | old heads' dream KL to main weights before -> after: "
          +" ".join(f"{_lab(cfg,d)} {kb[d]:.4f}->{ka[d]:.4f}" for d in keys)
          +f" | current-domain BPC on this batch {cb:.3f} -> {ca:.3f} ({ca-cb:+.3f})")

@torch.no_grad()
def stale_diag(model, cfg, t):
    """Do the protected subspaces match where old domains' inputs REALLY are now? Overlap (fraction of the real subspace's
    energy directions captured) of the stored and of the dream-refreshed covariances with real validation text."""
    vals=getattr(model,'_vals',None)
    if vals is None: return
    bt=[]
    for d in range(t):
        g=vals[d]
        if hasattr(g,'reset'): g.reset()
        x,y=g(); bt.append((x,y,d))
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    real=_input_cov(model,bt); model.feat_scale.copy_(fs); model.train(was)
    old=getattr(model,'_old_cov',None) or {}; stored={k:v[0]/v[1] for k,v in old.items()}; dream=_dream_cov(model,cfg)
    e=cfg.og_energy; res={'stored':{},'dream':{}}
    for k,C in real.items():
        Ur,kr=_subspaces(C,[e])[e]
        for tag,src in (('stored',stored),('dream',dream)):
            if k in src:
                Ue,_=_subspaces(src[k],[e])[e]; ov=float((Ur.T@Ue).pow(2).sum())/max(kr,1)
                res[tag].setdefault(k[0],[]).append(ov)
    nb_=len(model.blocks)
    row={tag:[float(np.mean(res[tag].get(i,[float('nan')]))) for i in range(nb_)] for tag in res}
    model.__dict__.setdefault('_stale_log',{})[t]=row
    print('    [staleness] protected old-input subspaces vs the REAL ones (old domains\' validation text through the current '
          'network), overlap per block: stored '+' '.join(f'b{i} {v:.2f}' for i,v in enumerate(row['stored']))
          +' | dream-refreshed '+' '.join(f'b{i} {v:.2f}' for i,v in enumerate(row['dream'])))

# ---------------- FORGETTING DIAGNOSTICS: which part of the model lost an earlier domain ----------------
_ROUTE_KEYS=('mu','used','n','gm','W')
def _fd_group(name):
    if name.startswith('heads.') or name.startswith('theads.'): return 'heads'
    if name.startswith('blocks.'): return 'block'+name.split('.')[1]
    return 'emb'                                       # embeddings, positions, model-level buffers

@torch.no_grad()
def _fd_swap(model, snap, groups, routing):
    """Put the named groups (and optionally the routing) back to the snapshot; returns what is needed to undo it."""
    cur={}; named=dict(list(model.named_parameters())+list(model.named_buffers()))
    existed=None                                       # topic heads that did not exist yet at that time keep their current state
    for n,t in named.items():
        if existed is not None and n.startswith('theads.') and int(n.split('.')[1]) not in existed: continue
        if _fd_group(n.replace('module.','')) in groups or ('backbone' in groups and _fd_group(n)!='heads'):
            o=snap['p'].get(n)
            if o is not None and o.dim()==t.dim() and o.shape!=t.shape and all(a<=b_ for a,b_ in zip(o.shape,t.shape)):
                z=torch.zeros_like(t); z[tuple(slice(0,a) for a in o.shape)]=o.to(t.device,t.dtype); o=z   # units added later: absent then
            if o is not None and o.shape==t.shape: cur[n]=t.detach().clone(); t.copy_(o.to(t.device))
            elif o is None and n.startswith('blocks.') and n.split('.')[-2] in ('Wout','mlp_out'):   # block added later: absent then
                cur[n]=t.detach().clone(); t.zero_()
    rs=None
    if routing:
        tr=getattr(model,'_topics',None)
        if tr is not None and 'tr' in snap:
            rs=('tr',{k:(getattr(tr,k).clone() if torch.is_tensor(getattr(tr,k)) else getattr(tr,k)) for k in _ROUTE_KEYS})
            for k,v in snap['tr'].items(): setattr(tr,k,v.clone() if torch.is_tensor(v) else v)
        nb=getattr(model,'_nb',None)
        if nb is not None and 'nb' in snap:
            rs=('nb',{'C':nb['C'].clone(),'used':nb['used'].clone()}); nb['C']=snap['nb']['C'].clone(); nb['used']=snap['nb']['used'].clone(); nb['lp']=None
            R=getattr(model,'_routers',None)
            if R is not None and 'routers' in snap:
                rs[1]['routers']={k:v.clone() for k,v in R.state_dict().items()}; R.load_state_dict(snap['routers']); object.__setattr__(model,'_rver',model._rver+1)
    return cur,rs

@torch.no_grad()
def _fd_undo(model, cur, rs):
    named=dict(list(model.named_parameters())+list(model.named_buffers()))
    for n,t in cur.items(): named[n].copy_(t)
    if rs is not None:
        if rs[0]=='tr':
            for k,v in rs[1].items(): setattr(model._topics,k,v)
        else:
            model._nb['C']=rs[1]['C']; model._nb['used']=rs[1]['used']; model._nb['lp']=None
            if 'routers' in rs[1]: model._routers.load_state_dict(rs[1]['routers']); object.__setattr__(model,'_rver',model._rver+1)

def _cka(X, Y):
    """Linear CKA between two (N,d) representations of the same tokens (1 = identical up to rotation / isotropic scale)."""
    X=X-X.mean(0); Y=Y-Y.mean(0)
    return float((Y.T@X).pow(2).sum()/((X.T@X).norm()*(Y.T@Y).norm()).clamp(min=1e-12))

def forget_diag(model, cfg, vals, t, snaps, post, doms=None):
    """For every earlier domain d: BPC now, and the BPC change when one part is restored to its state at the end of d.
    Negative 'undo X' = restoring X recovers that much = X's later changes caused that much forgetting."""
    n=cfg.fd_batches; out={}; nb_=len(model.blocks)
    for d in (range(t) if doms is None else doms):
        sd=snaps[d]; now=bpc(model,vals[d],d,n=n); row={'now':now,'drift':now-sd.get('post',post[d])}   # same batches both times
        tests=[('backbone',{'backbone'},False),('heads',{'heads'},False)]
        tests+=[('emb',{'emb'},False)]+[(f'b{i}',{f'block{i}'},False) for i in range(nb_)]
        for lab,g,rt in tests:
            cur,rs=_fd_swap(model,sd,g,rt); row[lab]=bpc(model,vals[d],d,n=n)-now; _fd_undo(model,cur,rs)
        out[d]=row
    print(f"  [forgetting diagnostics] BPC change when one part is put back to its state at the end of that domain (n={n} batches; negative = that part's later change caused forgetting):")
    for d,r in out.items():
        line=(f"     {PILE_DOMAINS[d][:8]:8} drift {r['drift']:+.3f} | undo backbone {r['backbone']:+.3f} heads {r['heads']:+.3f}"
              +(f" routing {r['routing']:+.3f}" if 'routing' in r else '')
              +" | per part: emb "+f"{r['emb']:+.3f} "+' '.join(f"b{i} {r[f'b{i}']:+.3f}" for i in range(nb_)))
        xk_=[k for k in r if isinstance(k,str) and k.startswith('X ')]
        if xk_: line+=" | WITH PRIVATE ROWS: "+" ".join(f"{k[2:]} {r[k]:+.3f}" for k in xk_)
        if 'overlap' in r:
            line+=f" | routing overlap {r['overlap']:.2f} | its topics (mass then->now, head change, protection): "+' '.join(f"t{k} {a:.2f}->{b:.2f} {rel:.0%} {pi:.1f}" for k,a,b,rel,pi in r['topics'])
        print(line)
    if any('fdrift' in r for r in out.values()):
        print("  [feature drift since each domain was learned] its real text, each block's output now vs at the end of that domain: relative squared change / linear CKA (1 = same up to rotation and scale):")
        for d,r in out.items():
            if 'fdrift' in r: print(f"     {PILE_DOMAINS[d][:8]:8} "+" ".join(f"b{i} {a:.3f}/{c:.3f}" for i,(a,c) in enumerate(r['fdrift']))
                                    +(" | shared/specific rel. change (shared share): "+" ".join(f"b{i} {a:.3f}/{b_:.3f} ({c:.0%})" for i,(a,b_,c) in enumerate(r['fsplit'])) if 'fsplit' in r else ""))
    return out


# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_dream_cov': 'platypus.orth',
    '_input_cov': 'platypus.orth',
    '_subspaces': 'platypus.orth',
    'bpc': 'platypus.inference.store',
}
