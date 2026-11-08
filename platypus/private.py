"""Per-domain private parts: masks, private trunk units, private n-gram tables."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from platypus.common import (
    _nslots,
)
from platypus.protection import (
    _STF,
)

def ngp_init(model,cfg):
    if not tuple(cfg.ngp or ()):                                                  # ngp=(): no private n-gram tables (issue #23 ablation)
        model.__dict__['_ngp']=None; model.__dict__['_ngp_opt']=None; return
    dev=model.embed.weight.device; T={}
    for d in range(_nslots(cfg)):
        T[d]=[nn.Embedding(int(cfg.ngp_rows),cfg.d_model,sparse=True).to(dev) for _ in cfg.ngp]
        for E in T[d]: E.weight.data.zero_()
    model.__dict__['_ngp']=T; model.__dict__['_ngp_opt']=torch.optim.SparseAdam([E.weight for d in T for E in T[d]],lr=cfg.lr*cfg.ngram_lr_mult)
_XDG={'on':False,'cur':None,'rows':None,'masks':{},'keep':1.0,'seed':0}
def xdg_mask(i,H,dev):
    """(rows,H) or (H,) gate for block i from the current row tags (int, list per row, or the current domain)"""
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if tag is None: return None
    def one(d):
        k=(i,int(d))
        if k not in _XDG['masks']:                                   # one long mask per (block, domain): widening appends units, old units keep their gate
            g=torch.Generator().manual_seed(_XDG['seed']*100003+i*1009+int(d)*17)
            _XDG['masks'][k]=(torch.rand(1<<15,generator=g)<_XDG['keep']).float()
        return _XDG['masks'][k][:H].to(dev)
    if isinstance(tag,(list,tuple)):
        cur=_XDG['cur']; return torch.stack([one(d if d is not None else (cur if cur is not None else 0)) for d in tag])
    return one(tag)
_SLP={'n':[0,0]}
_HCAP={'on':False,'h':None}                                              # cg-gate: capture of the head input (protected gate)
_BUFDET={'on':False}                                                    # v9g: per-domain parameters detached (buffered windows)
_PUOFF={'d':None}                                                        # cg19: domain whose private units are switched OFF
def pu_term(b,hn,ns_=None):
    r_=_pu_term0(b,hn,ns_); d_=_PUOFF['d']
    if d_ is None or not torch.is_tensor(r_): return r_
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if isinstance(tag,(list,tuple)):
        cur=_XDG['cur'] if _XDG['cur'] is not None else 0
        m_=torch.tensor([0.0 if (t_ if t_ is not None else cur)==d_ else 1.0 for t_ in tag],device=r_.device,dtype=r_.dtype)
        return r_*m_.view(-1,*([1]*(r_.dim()-1))) if m_.shape[0]==r_.shape[0] else r_
    return r_*0.0 if tag==d_ else r_
def _pu_term0(b,hn,ns_=None):
    """private trunk units of each row's domain. LEAK FIX (pu_dream_grad False): DREAM rows use their domain's units with the parameters
    DETACHED - an old domain's private units are then trained only by its own real text (its visits). Otherwise every loss on the dream rows
    reaches them, including BORROWING (the current domain's head fitting the old domain's dreams), which pulls the old domain's private units
    towards the current domain's objective."""
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if tag is None: return 0.0
    W_in,W_b,W_out=b.pu_in,b.pu_b,b.pu_out
    if _BUFDET['on']: W_in,W_b,W_out=W_in.detach(),W_b.detach(),W_out.detach()
    if isinstance(tag,(list,tuple)):
        mk_=getattr(b,'pu_mask',None); G_=_XDG.get('groups')
        if G_ is not None and G_[0]==hn.shape[0] and not torch.is_grad_enabled():     # SAMPLER: contiguous domain groups, one matmul per domain
            out_=[]
            for d_,sl_ in G_[1]:
                z=F.gelu(hn[sl_]@W_in[d_]+W_b[d_])
                if mk_ is not None: z=z*mk_[d_].to(z.dtype)
                out_.append(z@W_out[d_])
            return torch.cat(out_,0)
        cur=_XDG['cur'] if _XDG['cur'] is not None else 0
        ix=torch.tensor([d if d is not None else cur for d in tag],device=hn.device)
        if ix.shape[0]!=hn.shape[0]: return 0.0
        def f(Wi,Wb,Wo,h_,ix_):
            z=F.gelu(torch.einsum('btd,bdk->btk',h_,Wi[ix_])+Wb[ix_][:,None])
            if mk_ is not None: z=z*mk_[ix_][:,None,:].to(z.dtype)
            return torch.einsum('btk,bkd->btd',z,Wo[ix_])
        tr=torch.is_grad_enabled() and W_out.requires_grad and not getattr(b,'_pu_dream_grad',True)
        if tr and _STF.get('dream_call'): return f(W_in.detach(),W_b.detach(),W_out.detach(),hn,ix)        # dream-only forward
        if tr and ns_ is not None and 0<ns_<hn.shape[0]:                                                    # merged: real rows live, dream rows detached
            return torch.cat([f(W_in,W_b,W_out,hn[:ns_],ix[:ns_]),f(W_in.detach(),W_b.detach(),W_out.detach(),hn[ns_:],ix[ns_:])],0)
        return f(W_in,W_b,W_out,hn,ix)
    z=F.gelu(hn@W_in[tag]+W_b[tag])
    if getattr(b,'pu_mask',None) is not None: z=z*b.pu_mask[tag].to(z.dtype)
    return z@W_out[tag]
def xdg_apply(m,i):
    if not _XDG['on']: return m
    g=xdg_mask(i,m.shape[-1],m.device)
    if g is None: return m
    if g.dim()==2:
        if g.shape[0]!=m.shape[-3] if m.dim()>=3 else True: return m
        return m*g.view(*([1]*(m.dim()-3)),g.shape[0],1,g.shape[1]).to(m.dtype)
    return m*g.to(m.dtype)
def side_out(model,taps,dom,nstat):
    """ladder side network over the trunk's (detached) taps; per-row domain projection; dream rows detached when side_dream=False"""
    z=model.side_in(taps[0].float())
    for i,(t_,P_,M_) in enumerate(zip(taps[1:],model.side_tap,model.side_mlp)):
        z=z+M_(z+P_(t_.float()))
    B=z.shape[0]
    if isinstance(dom,int): o=z@model.side_out[dom]
    else:
        o=torch.zeros(*z.shape[:-1],model.side_out.shape[-1],device=z.device,dtype=z.dtype)
        for d in sorted({e for e in dom if e is not None}):
            ix=torch.tensor([j for j,e in enumerate(dom) if e==d],device=z.device); o=o.index_copy(0,ix,z[ix]@model.side_out[d])
    return o
def priv(model,x,h,d):
    """features + the domain's private tables (n-gram rows and feature-coded rows)"""
    cfg=model.cfg
    ng_=ngp_vec(model,x,d)
    if torch.is_tensor(ng_) and ng_.dim()==3 and ng_.shape[1]!=h.shape[1]: ng_=ng_[:,-h.shape[1]:]   # (incremental generation: newest positions)
    h=h+ng_
    return h
def cg_apply(b,m,ns_):
    """conflict-driven context-keyed gate (Config.cgate): gate rows by their domain's gate vector; returns (m, backward hook or None)"""
    G=b._cg_g; H_=m.shape[-1]
    if G.device!=m.device: b.__dict__['_cg_g']=G=G.to(m.device); b.__dict__['_cg_c']=b._cg_c.to(m.device)
    if G.shape[1]!=H_: return m,None
    tag=_XDG['rows'] if _XDG['rows'] is not None else _XDG['cur']
    if tag is None: return m,None
    cur=_XDG['cur'] if _XDG['cur'] is not None else 0
    if isinstance(tag,(list,tuple)):
        if len(tag)!=m.shape[0]: return m,None
        ix=[int(d) if d is not None else int(cur) for d in tag]; g=G[torch.tensor(ix,device=G.device)][:,None,:]
    else: ix=None; g=G[int(tag)]
    m=m*g.to(m.dtype)
    hook=None
    if b.training and torch.is_grad_enabled() and ix is not None and ns_ is not None and 0<ns_<m.shape[0]:
        b._cg_n=getattr(b,'_cg_n',0)+1
        if b._cg_n%b._cg_every==0:
            md=m.detach(); olds=sorted(set(d for d in ix[ns_:] if d!=cur))
            if olds:
                def hook(gr,md=md,ix=ix,ns_=ns_,olds=olds,cur=int(cur)):
                    with torch.no_grad():
                        gr=gr.float(); mf=md.float(); Gr=torch.einsum('btd,bth->dh',gr[:ns_],mf[:ns_]); nr=Gr.norm(dim=0)
                        rows=torch.tensor(ix,device=gr.device)
                        for d in olds:
                            sel=(rows==d).nonzero().flatten(); sel=sel[sel>=ns_]
                            if sel.numel()==0: continue
                            Gd=torch.einsum('btd,bth->dh',gr[sel],mf[sel]); c=-(Gr*Gd).sum(0)/(nr*Gd.norm(dim=0)+1e-12)
                            b._cg_c[d]=(1-b._cg_rho)*b._cg_c[d]+b._cg_rho*c
                        cm=b._cg_c.max(0).values; cl=((cm-b._cg_thr)/(b._cg_sc if b._cg_sc>0 else 1-b._cg_thr)).clamp(0,1)
                        b._cg_g[cur]=1-b._cg_k*cl
                    return None
    return m,hook
def inh_apply(b,m,hn,ns_):
    """learned inhibitory gate on the MLP hidden units m (see Config.inh_rank)"""
    A,B_,c=b.inh_A,b.inh_B,b.inh_c; H_=m.shape[-1]
    if B_.shape[1]!=H_: return m                                          # widened block: gate not resized (not used with widening)
    hd=hn.to(A.dtype); tr=b.training and torch.is_grad_enabled()
    split=tr and b._inh_mode=='dream' and ns_ is not None and 0<ns_<m.shape[0]
    if tr and b._inh_mode=='dream' and not split: z=(hd@A.detach())@B_.detach()+c.detach()        # no dream rows: the gate does not learn
    elif split: z=torch.cat([(hd[:ns_]@A.detach())@B_.detach()+c.detach(),(hd[ns_:]@A)@B_+c],0)
    else: z=(hd@A)@B_+c
    i_=torch.sigmoid(z.float())
    if tr and b._inh_open>0:                                              # openness penalty (gate parameters only)
        ip_=torch.sigmoid(((hd.detach()@A)@B_+c).float()) if b._inh_mode=='dream' else i_
        b.__dict__['_inh_pen']=b._inh_open*ip_.mean()
    if b.training and ns_ is not None and 0<ns_<m.shape[0]:
        with torch.no_grad():
            ir=i_[:ns_].mean((0,1)); idr=i_[ns_:].mean((0,1)); o_=b.__dict__.get('_inh_stat')
            st=torch.stack([ir,idr]); b.__dict__['_inh_stat']=st if o_ is None else 0.98*o_+0.02*st
    return m*(1-i_).to(m.dtype)
_NGPD={'on':False,'m':None}                                           # conduit student pass: private n-gram rows read but not trained
def _ngp_look(E,ix,m_):
    """v8b: rows looked up where m_ (B,T) is False are read DETACHED; only teaching positions' rows get a (sparse) gradient"""
    if m_ is None or tuple(m_.shape)!=tuple(ix.shape[:2]): return E(ix)
    v=F.embedding(ix,E.weight.detach()); sel=m_.nonzero(as_tuple=True)
    if len(sel[0])==0: return v
    v=v.clone(); v[sel]=E(ix[sel]); return v
def ngp_vec(model,x,d):
    T=model.__dict__.get('_ngp')
    if T is None: return 0.0
    cols=_cols(x,max(model.cfg.ngp)); out=0.0; m_=_NGPD.get('m')
    for n,E in zip(model.cfg.ngp,T[d]): out=out+_ngp_look(E,_ngram_idx(cols,n,E.num_embeddings,5),m_)
    O=model.__dict__.get('_ngo')                                      # SHARED OUTPUT-SIDE n-gram rows (every domain, before its head)
    if O is not None:
        for (n,j),E in O: out=out+_ngp_look(E,_ngram_idx(cols,n,E.num_embeddings,7+j),m_)
    return out.detach() if (_NGPD['on'] and torch.is_tensor(out)) else out

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_cols': 'platypus.ngrams',
    '_ngram_idx': 'platypus.ngrams',
}
