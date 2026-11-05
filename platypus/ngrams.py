"""Hashed n-gram embeddings and count features, the main optimiser and compilation."""
import torch
import torch.nn.functional as F


def adam_capture(opt, model): return {n:dict(opt.state.get(p,{})) for n,p in model.named_parameters()}
def adam_rebuild(model, lr, before):
    """New Adam that KEEPS moment estimates of every existing weight (zero for newly activated rows/cols)."""
    opt=torch.optim.Adam(_opt_params(model),lr=lr)
    for n,p in model.named_parameters():
        st=before.get(n)
        if not st or not p.requires_grad: continue
        ns={}
        for k,v in st.items():
            if torch.is_tensor(v) and v.dim()>0 and v.shape!=p.shape:
                z=torch.zeros_like(p); z[tuple(slice(0,m) for m in v.shape)]=v; ns[k]=z
            else: ns[k]=v.clone() if torch.is_tensor(v) else v
        opt.state[p]=ns
    return opt

class _GScale(torch.autograd.Function):
    """Identity forward; the backward pass multiplies the gradient by s (per row: s of shape (B,1,1))."""
    @staticmethod
    def forward(ctx, x, s): ctx.save_for_backward(s); return x.view_as(x)
    @staticmethod
    def backward(ctx, g): (s,)=ctx.saved_tensors; return g*s.to(g.dtype), None

class _GProj(torch.autograd.Function):
    """Identity forward; backward: a x the gradient's component in span(U) + c x the rest: g -> a g U U^T + c (g - g U U^T)."""
    @staticmethod
    def forward(ctx, x, U, c, a=1.0): ctx.save_for_backward(U); ctx.c=c; ctx.a=a; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        (U,)=ctx.saved_tensors; gf=g.float(); gs=(gf@U)@U.T
        return (ctx.a*gs+ctx.c*(gf-gs)).to(g.dtype), None, None, None

_NG_P=2147483647                                        # Mersenne prime 2^31 - 1
_NG_A=[1103515245,  987654323, 1789773571,  612481199]   # per-hash multipliers (< P) of the universal ((A h + C) mod P) mod size map
_NG_C=[  12345,     40503,    7919,     104729]
def _ngram_idx(cols, n, size, salt=0):
    """Hash of n bytes (cols: list of (B,T) or (B,1) long tensors, newest first; -1 = before the start); salt = which hash. A polynomial
    fingerprint mod P (base depends on the hash) followed by a universal map ((A h + C) mod P) mod size, so that different hashes
    collide INDEPENDENTLY (a multiplicative map mod a power of two would make hash 2 a function of hash 1)."""
    base=1000003+7777*salt; h=torch.full_like(cols[0],n)
    for i in range(n): h=(h*base+cols[i]+1)%_NG_P
    return ((h*_NG_A[salt%4]+_NG_C[salt%4]+salt//4)%_NG_P)%size

def ngram_embed(tables, x, touched=None, counts=None, ncount=None):
    """C1: sum over n (and over the hashes) of the hashed n-gram embedding ending at each position (x (B,T) bytes). touched: rows used
    (lazy optimiser); counts: per-row usage of the first ncount rows (real text) for the row protection."""
    out=0.0; nmax=max(tables.ns); nh=getattr(tables,'nh',1)
    cols=_seg_cols([x]+[F.pad(x,(i,0),value=-1)[:,:x.shape[1]] for i in range(1,nmax)])
    for i_,(n,E) in enumerate(zip(tables.ns,tables)):
        for j in range(nh):
            idx=_ngram_idx(cols,n,E.num_embeddings,j); out=out+E(idx)
            if touched is not None: touched.setdefault(i_,[]).append(idx.flatten())
            if counts is not None: ic_=idx[:ncount].flatten(); counts[i_].index_add_(0,ic_,torch.ones(ic_.shape,device=ic_.device,dtype=counts[i_].dtype))   # (no CPU wait)
    return out

def ngram_embed_tok(tables, x, hist):
    """One position: x (B,1) current byte, hist (B,nmax-1) previous bytes, newest LAST (-1 = before the start)."""
    nmax=max(tables.ns); nh=getattr(tables,'nh',1); cols=[x]+[hist[:,-i].unsqueeze(1) for i in range(1,nmax)]
    out=0.0
    for n,E in zip(tables.ns,tables):
        for j in range(nh): out=out+E(_ngram_idx(cols,n,E.num_embeddings,j))
    return out

def _new_opt(model, cfg, lr=None):
    """Main optimiser: Adam over _opt_params; fused CUDA kernel when adam_fused and on GPU."""
    kw={'fused':True} if (getattr(cfg,'adam_fused',False) and torch.cuda.is_available()) else {}
    return torch.optim.Adam(_opt_params(model),lr=cfg.lr if lr is None else lr,**kw)

_COMPILED={}
def setup_compile(model, cfg):
    """torch.compile of Model._features_eager (registered by id: frozen copies and teachers stay eager)."""
    return
    try:
        import torch._dynamo as _dyn; _dyn.config.cache_size_limit=max(64,_dyn.config.cache_size_limit)
        _COMPILED[id(model)]=torch.compile(model._features_eager, dynamic=True); print("  [compile] model forward compiled (torch.compile, dynamic shapes)")
    except Exception as e: print(f"  [compile] not available ({repr(e)[:120]}); eager")

def _opt_params(model):
    """Parameters of the main optimiser (the n-gram tables have their own lazy optimiser when ngram_sparse)."""
    ex={id(p) for p in model.ngram.parameters()} if (getattr(model,'ngram',None) is not None and model.cfg.ngram_sparse) else set()
    return [p for p in model.parameters() if p.requires_grad and id(p) not in ex]

def _ng_prot_scale(model, cfg, i_):
    """Row-protection step scale 1/(1 + gamma u/mean u) of table i_ (None when off or before any old usage)."""
    old=model.__dict__.get('_ng_old')
    if cfg.ngram_protect<=0 or old is None: return None
    C=model.__dict__.setdefault('_ng_sc',{})                        # cached per visit (no GPU sync per step)
    if i_ not in C:
        u=old[i_]; C[i_]=None if float(u.sum())<=0 else 1.0/(1.0+cfg.ngram_protect*(u/u[u>0].mean()))
    return C[i_]

_SEG={'s':None}                                                     # v9 RUN MASKING: (B,T) start of the clean segment of each position
def _seg_cols(cols):
    s_=_SEG['s']
    if s_ is None or tuple(s_.shape)!=tuple(cols[0].shape): return cols
    t_=torch.arange(cols[0].shape[1],device=cols[0].device)[None]
    return [cols[0]]+[c_.masked_fill((t_-i)<s_,-1) for i,c_ in enumerate(cols[1:],1)]
def _cols(x, nmax):
    """Newest-first byte columns of x (B,T): cols[i] = the byte i positions back (-1 before the start)."""
    return _seg_cols([x]+[F.pad(x,(i,0),value=-1)[:,:x.shape[1]] for i in range(1,nmax)])

def _cnt_idx(cfg, L, cols, j):
    """Row of the order-L count sketch j for the context ending at each position (L = 1: the byte itself)."""
    if L==1: return cols[0].clamp(min=0)
    return _ngram_idx(cols,L,cfg.ngram_count_rows,j+2)

def _cnt_tables(cfg):
    """(order, sketch, rows) of every count sketch."""
    out=[]
    for L in cfg.ngram_count:
        nh=1 if L<=2 else max(1,int(cfg.ngram_count_hashes))
        for j in range(nh): out.append((int(L),j,256 if L==1 else int(cfg.ngram_count_rows)))
    return out

@torch.no_grad()
def count_logratio(cfg, C, cols):
    """Interpolated absolute-discount backoff over the orders, from the count sketches C (list aligned with _cnt_tables); returns
    log(p * 256) (B,T,256): zero for an uninformative context."""
    tabs=_cnt_tables(cfg); D=cfg.ngram_count_D; sh=cols[0].shape
    p=torch.full((*sh,256),1.0/256,device=cols[0].device)
    for L in sorted(set(cfg.ngram_count)):
        c=None
        for (L_,j,_),T_ in zip(tabs,C):
            if L_!=L: continue
            ix_=_cnt_idx(cfg,L,cols,j)
            g=(T_[0][ix_]+T_[1][ix_]) if isinstance(T_,tuple) else T_[ix_]
            c=g if c is None else torch.minimum(c,g)
        tot=c.sum(-1,keepdim=True); nz=(c>0).float().sum(-1,keepdim=True)
        pn=((c-D).clamp(min=0)+D*nz*p)/tot.clamp(min=1e-9)
        p=torch.where(tot>0,pn,p)
    return torch.log(p.clamp(min=1e-7)*256.0)

def count_feature(model, cfg, cols, C=None, proj=None):
    """Count-predictor feature for the residual stream (live counts = old + current unless C is given)."""
    if C is None: C=list(zip(model._cnt_old,model._cnt_cur))            # live = old + current (summed after the gather)
    lr_=count_logratio(cfg,C,cols)
    return (proj or model.cproj)(lr_.to((proj or model.cproj).weight.dtype))

@torch.no_grad()
def ngram_lazy_step(model, cfg, lr, b1=0.9, b2=0.999, eps=1e-8):
    """LAZY ADAM for the n-gram tables: only the rows used this step are updated (per-row step counts and moments); optional frequency-
    based protection of rows the earlier domains used heavily."""
    T_=model.__dict__.pop('_ng_touch',{}) ; ST=model.__dict__.setdefault('_ng_state',{})
    for i_,E in enumerate(model.ngram):
        W=E.weight; G=W.grad
        if G is None: continue
        if i_ in T_: r=torch.unique(torch.cat(T_[i_]))
        else: r=torch.nonzero(G.abs().sum(-1)>0).squeeze(1)
        cnt_=cfg.ngram_opt=='count'
        if i_ not in ST: ST[i_]=({} if cnt_ else {'m':torch.zeros_like(W),'v':torch.zeros_like(W)})|{'t':torch.zeros(W.shape[0],device=W.device)}
        st=ST[i_]; g=G[r]
        if cnt_:                                                    # COUNTING RULE: no moments - the row's gradient direction (RMS-
            n_=st['t'][r][:,None]                                   # normalised over the row) x (n0 / (n0 + n))^p, n = the row's lifetime uses
            up=g/(g.pow(2).mean(-1,keepdim=True).sqrt()+1e-12)*(cfg.ngram_count_n0/(cfg.ngram_count_n0+n_))**cfg.ngram_count_p
            st['t'][r]+=1
        else:
            st['t'][r]+=1; tr=st['t'][r][:,None]
            st['m'][r]=b1*st['m'][r]+(1-b1)*g; st['v'][r]=b2*st['v'][r]+(1-b2)*g*g
            up=(st['m'][r]/(1-b1**tr))/((st['v'][r]/(1-b2**tr)).sqrt()+eps)
        sc=_ng_prot_scale(model,cfg,i_)
        if sc is not None: up=up*sc[r][:,None]
        W[r]-=lr*cfg.ngram_lr_mult*cfg.ngram_in_mult*up
        W.grad=None

def _hist_push(TT, hist, x):
    """Sampler byte history for the n-gram embeddings (None when the model has none)."""
    ng=getattr(TT.model,'ngram',None)
    if ng is None and not TT.model.cfg.ngram_count: return None
    if hist is None: hist=torch.full((x.shape[0],_hist_len(TT.model)),-1,dtype=torch.long,device=x.device)
    return torch.cat([hist[:,1:],x],1) if hist.shape[1]>0 else hist

def _hist_len(model):
    ng=getattr(model,'ngram',None); a=max(ng.ns) if ng is not None else 1
    return max(a,1)-1

def _dream_route_rows(cfg, n1, ns, dev, nb):
    """Per-row part multipliers for a dream batch (single-domain rows then strand rows), or None when nothing is routed."""
    return None
    r1=torch.tensor([1.0]*(nb+1),device=dev,dtype=torch.float32)
    r2=r1
    return torch.cat([r1.expand(n1,-1),r2.expand(ns,-1)])

def route_scales(M):
    """Per-row part multipliers M (B, n_blocks+1) = (emb, b0, b1, ...) -> gradient scales at each block's input and after the last block,
    such that the gradient reaching part j's parameters is M[:,j] x the unrouted gradient (a part below a zero-weight part gets 0)."""
    L=M.shape[1]-1; sc=[]
    for j in range(L):
        lo,up=M[:,j],M[:,j+1]
        sc.append(torch.where(up>0,lo/up.clamp(min=1e-12),torch.zeros_like(lo)).view(-1,1,1))
    return sc, M[:,L].reshape(-1,1,1)
