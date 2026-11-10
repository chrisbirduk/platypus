"""Text data: Pile streaming with a disk cache, fact injection, loaders."""
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from platypus.common import (
    DEVICE,
)
from platypus.layers import (
    _BPTT,
)


# ---------------- data ----------------
def _rank(v): return np.argsort(np.argsort(v)).astype(float)
def _spearman(a,b):
    ra,rb=_rank(a),_rank(b); ra-=ra.mean(); rb-=rb.mean()
    return float((ra*rb).sum()/(np.sqrt((ra*ra).sum()*(rb*rb).sum())+1e-12))

def locality_metrics(W0_in,W1_in,W0_out,W1_out,dom_usage,prior_imp,k):
    """Orientation = each MLP unit's input weight direction (row of mlp_in).
    (1) clustering: are this domain's top-10% units closer in orientation than random units?
    (2) damage locality: among units important for EARLIER domains, does orientation proximity
        to this domain's most active units predict how much they were rewritten?
    (3) neighbour effect: does neighbours' prior importance predict damage beyond own importance?
    Uses its own RNG so measurement never perturbs training."""
    rng=np.random.RandomState(0)
    O=W1_in/(np.linalg.norm(W1_in,axis=1,keepdims=True)+1e-8)
    C=O@O.T; n=len(O); np.fill_diagonal(C,np.nan)
    top=np.argsort(-dom_usage)[:max(4,n//10)]
    rnd=rng.choice(n,len(top),replace=False)
    clust=float(np.nanmean(C[np.ix_(top,top)])-np.nanmean(C[np.ix_(rnd,rnd)]))
    damage=np.linalg.norm(W1_in-W0_in,axis=1)+np.linalg.norm(W1_out-W0_out,axis=0)
    prox=np.nanmax(C[:,top],axis=1)                          # closeness to this domain's active units
    old=np.argsort(-prior_imp)[:max(4,n//4)]                  # units important for earlier domains
    loc=_spearman(prox[old],damage[old])
    Cn=np.nan_to_num(C,nan=-2.0); nn_idx=np.argsort(-Cn,axis=1)[:,:k]
    neigh=prior_imp[nn_idx].mean(1)
    z=lambda v:(v-v.mean())/(v.std()+1e-8)
    X1=np.stack([np.ones(n),z(prior_imp)],1); X2=np.stack([np.ones(n),z(prior_imp),z(neigh)],1); y=z(damage)
    b1=np.linalg.lstsq(X1,y,rcond=None)[0]; b2=np.linalg.lstsq(X2,y,rcond=None)[0]
    r2=lambda X,b:1-((y-X@b)**2).mean()/(y.var()+1e-12)
    own_neigh=float(np.corrcoef(prior_imp,neigh)[0,1]) if prior_imp.std()>0 else float('nan')
    # neigh_r2_gain is the reliable statistic; if own_neigh_corr is near 1 the test can't
    # separate a unit's own importance from its neighbours' (report it so this is visible)
    return {'cluster':clust,'damage_locality':loc,'neigh_r2_gain':float(r2(X2,b2)-r2(X1,b1)),'own_neigh_corr':own_neigh}

_BYTES_CACHE={}
_RUN_T0=time.time()                                                     # v9h: wall-clock origin for the visit headers
PILE_CACHE_DIR=os.environ.get('PILE_CACHE','pile_cache')      # streamed domain text is also cached on disk (re-runs, seeds)
def _cache_path(key):
    if _stream_bytes is not _hf_stream_bytes: return None       # local stand-in data: no disk cache
    d,seed,n=key; return os.path.join(PILE_CACHE_DIR,f"{''.join(c if c.isalnum() else '_' for c in d)}_{seed}_{n}.bin")
def _cache_put(key, data):
    fp=_cache_path(key)
    if fp:
        os.makedirs(PILE_CACHE_DIR,exist_ok=True)
        with open(fp,'wb') as f: f.write(data)
def stream_bytes(domain,seed,n):
    key=(domain,seed,n)
    if key not in _BYTES_CACHE:
        fp=_cache_path(key)
        if fp and os.path.exists(fp): _BYTES_CACHE[key]=open(fp,'rb').read()
        else: _BYTES_CACHE[key]=_stream_bytes(domain,seed,n); _cache_put(key,_BYTES_CACHE[key])
    return _BYTES_CACHE[key]
def overlap_check(val, train, k=128):
    """issue #30: training text that also occurs in the validation text. Every k-byte string of the validation text is hashed; the training
    text is scanned in k-byte steps; low-variety chunks (fewer than 16 distinct bytes: runs of spaces, digits, rules) are skipped. Document
    leakage shows as many hits; shared boilerplate as a few."""
    if not val or not train: return 0, 0
    V_={hash(val[i:i+k]) for i in range(0, len(val)-k+1)}; n_=h_=0
    for i in range(0, len(train)-k+1, k):
        c_=train[i:i+k]
        if len(set(c_))<16: continue
        n_+=1; h_+=int(hash(c_) in V_)
    return h_, n_
def prefetch_pile(reqs, seed=2026, max_docs=8_000_000):
    """ONE streaming pass collecting several (new) domains at once - a separate scan per domain and split would read the Pile many
    times over for the rarer subsets. reqs: [(domain, val_key, train_key)]; per domain, whole documents go to validation until it is
    full, then to training (document-level disjoint)."""
    need={}
    for d,vk,tk in reqs:
        if all(k in _BYTES_CACHE or (_cache_path(k) and os.path.exists(_cache_path(k))) for k in (vk,tk)): continue
        need[d]={'val':(vk,bytearray()),'train':(tk,bytearray())}
    if not need: return
    from datasets import load_dataset
    print(f"  [data] one streaming pass for {len(need)} new domains: "+", ".join(need))
    ds=load_dataset("monology/pile-uncopyrighted",split="train",streaming=True).shuffle(seed=seed,buffer_size=10000)
    seen_names={}; n_docs=0; t0=time.time()
    for ex in ds:
        n_docs+=1; nm=(ex.get("meta") or {}).get("pile_set_name"); seen_names[nm]=seen_names.get(nm,0)+1
        if nm in need and ex.get("text"):
            slot=need[nm]; tb=ex["text"].encode("utf-8","ignore")+b"\n"
            (vk,vb),(tk,tb_)=slot['val'],slot['train']
            if len(vb)<vk[2]: vb.extend(tb)
            elif len(tb_)<tk[2]: tb_.extend(tb)
            if len(vb)>=vk[2] and len(tb_)>=tk[2]:
                for k,b in ((vk,vb),(tk,tb_)): _BYTES_CACHE[k]=bytes(b[:k[2]]); _cache_put(k,_BYTES_CACHE[k])
                del need[nm]; print(f"    [data] {nm} complete after {n_docs} documents ({time.time()-t0:.0f}s)")
                if not need: return
        if n_docs==1_000_000:                                          # a requested name never seen: fail early with the names that exist
            miss_=[d for d in need if seen_names.get(d,0)==0]
            if miss_: raise RuntimeError(f"no documents named {miss_} in the first 1M; pile_set_names seen: {sorted(k for k in seen_names if k)}")
        if n_docs%500000==0: print(f"    [data] {n_docs} documents scanned ({time.time()-t0:.0f}s); still filling: "
                                   +", ".join(f"{d} {len(v['val'][1])+len(v['train'][1])}" for d,v in need.items()))
        if n_docs>=max_docs:
            raise RuntimeError(f"domains not filled after {n_docs} documents: {list(need)}; pile_set_names seen: {seen_names}")
def _stream_bytes(domain,seed,n):
    from datasets import load_dataset
    ds=load_dataset("monology/pile-uncopyrighted",split="train",streaming=True).shuffle(seed=seed,buffer_size=10000)
    buf=bytearray()
    for ex in ds:
        if (ex.get("meta") or {}).get("pile_set_name")==domain and ex.get("text"):
            buf.extend(ex["text"].encode("utf-8","ignore")+b"\n")
            if len(buf)>=n: break
    return bytes(buf[:n])
_hf_stream_bytes=_stream_bytes
def _using_hf_stream(): return _stream_bytes is _hf_stream_bytes
def set_stream_source(fn):
    """replace the text source (local stand-in data for tests): fn(domain, seed, n) -> bytes"""
    global _stream_bytes
    _stream_bytes=fn
def make_loader(raw,cfg,fixed_seed=None,no_revisit=False,lanes=False,span=1,jump=0,keep=False):
    """fixed_seed: eval loader that resets to the SAME batches on every bpc() call,
    so evaluation noise is identical across configs and seeds. no_revisit: non-overlapping windows in random order, each at most once."""
    src=torch.frombuffer(bytearray(raw),dtype=torch.uint8).long().to(DEVICE)
    if lanes:                                                          # CONTINUOUS CONTEXT: lanes, each reading its own part in order
        Tq=int(cfg.seq_len); st={'j':0,'ep':0,'K':None}
        def _lay(K):                                                   # K segments per row (ctx_bptt): batch_size // K lanes of K x seq_len bytes
            Bn=max(1,int(cfg.batch_size)//K); Ln=(len(src)-1)//Bn
            st.update(K=K,Bn=Bn,nst=max(1,(Ln-1)//(Tq*K)),base=(torch.arange(Bn,device=DEVICE)*Ln)[:,None],tx=torch.arange(Tq*K,device=DEVICE)[None],j=0,k=0,b=None)
        if int(jump)>0:                                                # EPISODES: every `jump` segments, fresh random places
            gj_=torch.Generator(device='cpu'); gj_.manual_seed(len(src)%100003+7); J=int(jump)
            def get():
                K=int(_BPTT['K'])
                if st['K']!=K: _lay(K)
                Jf=max(1,J//K)                                         # fetches per episode
                if st['k']%Jf==0: st['b']=torch.randint(len(src)-Jf*K*Tq-2,(st['Bn'],1),generator=gj_).to(DEVICE)
                get.fresh=(st['k']==0) if keep else (st['k']%Jf==0); ix=st['b']+(st['k']%Jf)*K*Tq+st['tx']; st['k']+=1
                return src[ix], src[ix+1]
            get.reset=lambda: None; get.fresh=True; get.windows=max(1,(len(src)-1)//Tq)
            return get
        def get():
            K=int(_BPTT['K'])
            if st['K']!=K: _lay(K)
            if st['j']>=st['nst']:
                st['j']=0; st['ep']+=1
                if st['ep']==1: print(f"    [data] lanes: shard read through ({st['nst']} rows of {K} segments per lane) - lanes start again (memory reset)")
            get.fresh=(st['j']==0); ix=st['base']+st['j']*K*Tq+st['tx']; st['j']+=1
            return src[ix], src[ix+1]
        get.reset=lambda: None; get.fresh=True; get.windows=max(1,(len(src)-1)//Tq)
        return get
    if no_revisit:
        W=cfg.seq_len+1; nw=max(1,(len(src)-1)//W); st={'perm':torch.randperm(nw).to(DEVICE)*W,'pos':0,'ep':0}
        tx=torch.arange(cfg.seq_len,device=DEVICE)
        def get():
            if st['pos']+cfg.batch_size>nw:                          # shard exhausted: reshuffle (reported - should not happen when sized right)
                st['perm']=torch.randperm(nw).to(DEVICE)*W; st['pos']=0; st['ep']+=1
                print(f"    [data] WARNING: training shard exhausted ({nw} windows) - text now repeats (epoch {st['ep']+1})")
            ix=st['perm'][st['pos']:st['pos']+cfg.batch_size]; st['pos']+=cfg.batch_size
            return src[ix[:,None]+tx[None]], src[ix[:,None]+tx[None]+1]
        get.reset=lambda: None; get.windows=nw
        return get
    Tsp_=int(cfg.seq_len)*max(1,int(span))                            # span > 1: longer evaluation windows (eval_ctx)
    N=max(1,len(src)-Tsp_-1); tx=torch.arange(Tsp_,device=DEVICE)
    gen=torch.Generator(device='cpu')
    def reset():
        if fixed_seed is not None: gen.manual_seed(fixed_seed)
    def get():
        if fixed_seed is not None: ix=torch.randint(N,(cfg.batch_size,),generator=gen).to(DEVICE)
        else: ix=torch.randint(N,(cfg.batch_size,),device=DEVICE)
        return src[ix[:,None]+tx[None]], src[ix[:,None]+tx[None]+1]
    get.reset=reset; reset()
    return get
