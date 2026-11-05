"""Frozen teachers, stacked teacher pass and the recurrent dream sampler."""
import contextlib
import torch
import torch.nn.functional as F
from platypus.protection import (
    _DP2, _FILM, film_mod,
)
from platypus.private import (
    _XDG, ngp_vec, pu_term, xdg_apply,
)
from platypus.layers import (
    _PREC, _SAMP, _delta_core, _kw_rec,
)


# ---------------- PERFORMANCE: FROZEN TEACHERS, STACKED TEACHER PASS, RECURRENT DREAM SAMPLER ----------------
_TEACH_DROP=('_pg','_pg_maha','_bstat','_uh','_um','_probe')

@torch.no_grad()
def _frozen_blocks(model, subs):
    """Frozen copy of model.blocks with every {id(param): delta} dict in subs subtracted (shape-checked, as in the weight-swapping
    code). No hooks, no gradients, eval mode."""
    import copy
    saved=[{a:b.__dict__.pop(a) for a in _TEACH_DROP if a in b.__dict__} for b in model.blocks]
    try: cp=copy.deepcopy(model.blocks)
    finally:
        for b,sv in zip(model.blocks,saved): b.__dict__.update(sv)
    for m in cp.modules():
        for nm in ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_forward_hooks_with_kwargs','_forward_pre_hooks_with_kwargs','_forward_hooks_always_called'):
            if isinstance(getattr(m,nm,None),dict): getattr(m,nm).clear()
    for lp,cp_ in zip(model.blocks.parameters(),cp.parameters()):
        cp_.requires_grad_(False)
        for D in subs:
            d_=D.get(id(lp)) if D else None
            if d_ is not None and d_.shape==cp_.shape: cp_.sub_(d_)
    for b in cp: b.cur_distinct=None; b.calib=None; b._s0_idx=None
    cp.eval(); return cp

def _stk_ln(x, w, b): return F.layer_norm(x,(x.shape[-1],))*w[:,None,None,:]+b[:,None,None,:]
def _stk_lin(z, W, b=None):
    """Stacked linear for M models: z (M,B,T,in), W (M,in,out) [pre-transposed], b (M,out) -> (M,B,T,out)."""
    M,B,T,_=z.shape; zf=z.reshape(M,B*T,-1)
    y=torch.baddbmm(b.unsqueeze(1),zf,W) if b is not None else torch.bmm(zf,W)
    return y.reshape(M,B,T,-1)
def _stk_conv(z, w, prev=False):
    """Causal depthwise convolution per model: z (M,B,T,C), w (M*C,1,K). prev=True: previous K values only (local filter blocks)."""
    M,B,T,C=z.shape; K=w.shape[-1]
    zt=F.pad(z.permute(1,0,3,2).reshape(B,M*C,T),(K if prev else K-1,0))
    o=F.conv1d(zt,w.to(zt.dtype),groups=M*C)
    if prev: o=o[...,:T]
    return o.reshape(B,M,C,T).permute(1,0,3,2)

def _cons_levels(model, cfg):
    """Mid-store levels from most recent to slowest: three timescales = [mid]; timescale chain = [L1, L2, ...] (non-empty prefix)."""
    L=[]
    out=[]
    for l_ in L:
        if not l_: break
        out.append(l_)
    return out

def teacher_role(model, cfg, d, olds, TT):
    """Teacher for old domain d: 'main' (recent), or 'lvl<k>' = main without the k most recent mid-store levels. Three timescales: age
    >= cons3_age. Timescale chain: the DEEPEST teacher that still contains >= cons_chain_keep of the domain's own learning (so older
    domains are held to slower memories without being pulled back from what they learned themselves)."""
    if d not in olds: return 'main'
    age=len(olds)-1-olds.index(d)
    k=min(1 if age>=cfg.cons3_age else 0, TT.depth)
    return 'main' if k==0 else f'lvl{k}'

class Teachers:
    """FROZEN TEACHERS for one visit. role 'main' = live weights minus the shunt (= the weights at the start of the visit: everything the
    visit writes into the blocks goes through the shunt); roles 'lvl1', 'lvl2', ... = main minus the 1, 2, ... most recent mid-store levels
    (three timescales: lvl1 = the slow weights). The byte and position embeddings are read LIVE unless they are in the shunt (emb_shunt),
    then each teacher has its own frozen copy."""
    def __init__(s, model, cfg, roles):
        s.model=model; s.cfg=cfg; s.blocks={r:_frozen_blocks(model,subs) for r,subs in roles.items()}; s._stk={}
        s.emb=None
        s.depth=len(roles)-1
        s.stackable=all(not getattr(b,'mq',False) and not b.route_in_forward and not getattr(b,'lattn',False) and not getattr(b,'nov_g',None) for b in model.blocks)
        if getattr(model,'ngram',None) is not None:                 # C1: frozen n-gram tables for this visit's teachers
            import copy
            s.ngram=copy.deepcopy(model.ngram)
            for p_ in s.ngram.parameters(): p_.requires_grad_(False)
    def _ng(s):                                                   # n-gram tables: the visit's frozen copy (live for ad-hoc samplers)
        return getattr(s,'ngram',None) if hasattr(s,'ngram') else getattr(s.model,'ngram',None)
    def _emb(s, x, role):
        T=x.size(1); ng=s._ng()
        if s.emb is None: e=s.model.embed(x)+s.model.pos(torch.arange(T,device=x.device))
        else: E,P_=s.emb[role]; e=F.embedding(x,E)+P_[:T]
        if getattr(s,'cnt',None) is not None: e=e+count_feature(s.model,s.cfg,_cols(x,max(s.cfg.ngram_count)),C=s.cnt,proj=s.cproj)
        return e+ngram_embed(ng,x) if ng is not None else e
    def _emb_tok(s, x, t, role, hist=None):                       # one position (recurrent sampler)
        ng=s._ng()
        if s.emb is None: e=s.model.embed(x)+s.model.pos.weight[t]
        else: E,P_=s.emb[role]; e=F.embedding(x,E)+P_[t]
        if ng is not None or getattr(s,'cnt',None) is not None:
            if hist is None: hist=torch.full((x.shape[0],_hist_len(s.model)),-1,dtype=torch.long,device=x.device)
        if ng is not None: e=e+ngram_embed_tok(ng,x,hist[:,hist.shape[1]-(max(ng.ns)-1):] if max(ng.ns)>1 else hist[:,:0])
        if getattr(s,'cnt',None) is not None:
            cols=[x]+[hist[:,-i].unsqueeze(1) for i in range(1,max(s.cfg.ngram_count))]
            e=e+count_feature(s.model,s.cfg,cols,C=s.cnt,proj=s.cproj)
        return e
    def _ac(s):
        return torch.autocast(device_type=s.model.embed.weight.device.type, dtype=torch.bfloat16) if s.cfg.amp=='bf16' else contextlib.nullcontext()
    def _out(s, h):
        h=h.float()
        return h
    @torch.no_grad()
    def features(s, x, role):
        h=s._emb(x,role)
        with s._ac():
            for b in s.blocks[role]: h=b(h)
        return s._out(h)
    @torch.no_grad()
    def features_taps(s, x, role, lis):
        """Residual stream after each block in lis (one pass)."""
        h=s._emb(x,role); out={}
        with s._ac():
            for i,b in enumerate(s.blocks[role][:max(lis)+1]):
                h=b(h)
                if i in lis: out[i]=h.float()
        return out
    @torch.no_grad()
    def features_upto(s, x, role, li):
        """Residual stream after block li of teacher `role` (feature anchoring)."""
        h=s._emb(x,role)
        with s._ac():
            for b in s.blocks[role][:li+1]: h=b(h)
        return h.float()
    @torch.no_grad()
    def _stack(s, roles):
        key=tuple(roles)
        if key in s._stk: return s._stk[key]
        lp=torch.bfloat16 if s.cfg.amp=='bf16' else None             # matmul weights pre-cast exactly as autocast would cast them
        cv=lambda t_: t_.to(lp).contiguous() if lp is not None else t_.contiguous()
        out=[]
        for li in range(len(s.model.blocks)):
            bs=[s.blocks[r][li] for r in roles]; b0=bs[0]
            st=lambda f: torch.stack([f(b) for b in bs])
            loc=bool(getattr(b0,'loc',False))
            P=dict(ln_w=st(lambda b:b.ln.weight),ln_b=st(lambda b:b.ln.bias),norm_w=st(lambda b:b.norm.weight),norm_b=st(lambda b:b.norm.bias),
                   Wv=cv(st(lambda b:b.Wv.weight.T)),gate_w=cv(st(lambda b:b.gate.weight.T)),gate_b=cv(st(lambda b:b.gate.bias)),
                   Wout=cv(st(lambda b:b.Wout.weight.T)),mi_w=cv(st(lambda b:b.mlp_in.weight.T)),mi_b=cv(st(lambda b:b.mlp_in.bias)),
                   mo_w=cv(st(lambda b:b.mlp_out.weight.T)),mo_b=cv(st(lambda b:b.mlp_out.bias)))
            mh=getattr(b0,'mh',1); mk=getattr(b0,'mk',0); mb=getattr(b0,'mb',False); md=getattr(b0,'md',False)
            if loc: P['cl']=cv(torch.cat([b.conv_loc for b in bs]))
            else:
                P['Wk']=cv(st(lambda b:b.Wk.weight.T))
                if mb: P['Wb_w']=cv(st(lambda b:b.Wb.weight.T)); P['Wb_b']=cv(st(lambda b:b.Wb.bias))
                else: P['beta']=st(lambda b:b.beta)
                if md: P['Wa_w']=cv(st(lambda b:b.Wa.weight.T)); P['Wa_b']=cv(st(lambda b:b.Wa.bias))
                if mk>0: P['ck']=cv(torch.cat([b.conv_k for b in bs])); P['cvv']=cv(torch.cat([b.conv_v for b in bs]))
            P['meta']=(mh,mk,mb,md,getattr(b0,'bmax',1.0),bool(getattr(b0,'gate_ln',False)),getattr(b0,'_kern','torch'),b0.d_state,loc)
            P['par']=bool(getattr(b0,'parallel',False))
            out.append(P)
        s._stk[key]=out; return out
    @torch.no_grad()
    def features_multi(s, x, roles):
        """Features of several teachers on the same input: ONE stacked forward pass (batched matmuls over the teachers)."""
        if len(roles)<=1 or not s.stackable: return [s.features(x,r) for r in roles]
        Ps=s._stack(roles); M=len(roles); B,T=x.shape
        if s.emb is None: h=s._emb(x,roles[0]).unsqueeze(0).expand(M,-1,-1,-1).contiguous()
        else: h=torch.stack([s._emb(x,r) for r in roles])
        with s._ac():
            for li_,P in enumerate(Ps):
                mh,mk,mb,md,bmax,gln,kern,C,loc=P['meta']
                hl=_stk_ln(h,P['ln_w'],P['ln_b'])
                v=_stk_lin(hl,P['Wv'])
                if loc: pred=_stk_conv(v,P['cl'],prev=True).float()
                else:
                    k=_stk_lin(hl,P['Wk'])
                    if mk>0: k=_stk_conv(k,P['ck']); v=_stk_conv(v,P['cvv'])
                    bt=bmax*torch.sigmoid(_stk_lin(hl,P['Wb_w'],P['Wb_b'])) if mb else torch.sigmoid(P['beta'].float())[:,None,None,None].expand(M,B,T,mh)
                    la=-F.softplus(_stk_lin(hl,P['Wa_w'],P['Wa_b'])) if md else None
                    pred=_delta_core(k.reshape(M*B,T,C),v.reshape(M*B,T,C),bt.reshape(M*B,T,mh),la.reshape(M*B,T,mh) if md else None,
                                     mh,None,M*B,T,kern).reshape(M,B,T,C)
                g_=_stk_lin(hl if gln else h,P['gate_w'],P['gate_b'])
                h0=h; h=h+_stk_lin(pred*F.silu(g_),P['Wout'])
                hn=_stk_ln(h0 if P.get('par') else h,P['norm_w'],P['norm_b'])
                mm_=F.gelu(_stk_lin(hn,P['mi_w'],P['mi_b']))
                if _XDG['on']: mm_=xdg_apply(mm_,li_)
                h=h+_stk_lin(mm_,P['mo_w'],P['mo_b'])
        return [s._out(h[m]) for m in range(M)]

@torch.no_grad()
def _rec_step(b, x, st):
    """One token through block b in RECURRENT form (eval): x (B,1,d) residual; st = this block's state (memory S per head, the last
    conv-width keys/values). Same arithmetic as the parallel WY solve: S <- a S; out = S k; S <- S + beta (v - out) k^T."""
    hl=b.ln(x); Bn=x.shape[0]; C=b.d_state; x0=x
    if getattr(b,'lattn',False):                                  # C2: sliding-window attention over a ring of the last W keys / values
        mh=getattr(b,'mh',1); mk=getattr(b,'mk',0); dh=C//mh; Wn=b.la_W; R=Bn*mh
        k=b.Wk(hl); v=b.Wv(hl)
        if mk>0:
            if 'bk' not in st: st['bk']=torch.zeros(Bn,mk,C,dtype=k.dtype,device=k.device); st['bv']=torch.zeros(Bn,mk,C,dtype=v.dtype,device=v.device)
            st['bk']=torch.cat([st['bk'][:,1:],k],1); st['bv']=torch.cat([st['bv'][:,1:],v],1)
            k=F.conv1d(st['bk'].transpose(1,2),b.conv_k.to(k.dtype),groups=C).transpose(1,2)
            v=F.conv1d(st['bv'].transpose(1,2),b.conv_v.to(v.dtype),groups=C).transpose(1,2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            kr=F.normalize(k.float().reshape(R,dh),dim=-1); vr=v.float().reshape(R,dh)
            if 'K' not in st: st['K']=torch.zeros(R,Wn,dh,device=x.device); st['V']=torch.zeros(R,Wn,dh,device=x.device); st['cnt']=0
            sc=torch.bmm(st['K'],kr.unsqueeze(-1)).squeeze(-1)*torch.exp(b.la_scale.float()).repeat(Bn)[:,None]+b.la_rel.float().flip(-1).repeat(Bn,1)
            valid=torch.arange(Wn,device=x.device)>=Wn-st['cnt']
            sc=sc.masked_fill(~valid[None],float('-inf'))
            Pm=F.softmax(torch.cat([sc,torch.zeros_like(sc[:,:1])],-1),-1)[:,:Wn]
            out=torch.bmm(Pm.unsqueeze(1),st['V']).squeeze(1).reshape(Bn,1,C)
            st['K']=torch.cat([st['K'][:,1:],kr.unsqueeze(1)],1); st['V']=torch.cat([st['V'][:,1:],vr.unsqueeze(1)],1); st['cnt']+=1
    elif getattr(b,'loc',False):                                  # LOCAL FILTER: the previous K values
        v=b.Wv(hl); K=b.conv_loc.shape[-1]
        if 'bl' not in st: st['bl']=torch.zeros(Bn,K,C,dtype=v.dtype,device=v.device)
        out=F.conv1d(st['bl'].transpose(1,2),b.conv_loc.to(v.dtype),groups=C).transpose(1,2).float()
        st['bl']=torch.cat([st['bl'][:,1:],v],1)
    else:
        mh=getattr(b,'mh',1); mk=getattr(b,'mk',0); mb=getattr(b,'mb',False); md=getattr(b,'md',False)
        k=b.Wk(hl); v=b.Wv(hl); dh=C//mh
        if mk>0:
            if 'bk' not in st: st['bk']=torch.zeros(Bn,mk,C,dtype=k.dtype,device=k.device); st['bv']=torch.zeros(Bn,mk,C,dtype=v.dtype,device=v.device)
            st['bk']=torch.cat([st['bk'][:,1:],k],1); st['bv']=torch.cat([st['bv'][:,1:],v],1)
            k=F.conv1d(st['bk'].transpose(1,2),b.conv_k.to(k.dtype),groups=C).transpose(1,2)
            v=F.conv1d(st['bv'].transpose(1,2),b.conv_v.to(v.dtype),groups=C).transpose(1,2)
        bt=b.bmax*torch.sigmoid(b.Wb(hl)) if mb else None
        la=-F.softplus(b.Wa(hl)) if md else None
        dp2_=bool(getattr(b,'dp2',False))
        if dp2_: k2_=b.Wk2(hl); v2_=b.Wv2(hl); bt2_=b.bmax*torch.sigmoid(b.Wb2(hl))
        with torch.autocast(device_type=x.device.type, enabled=False):
            kf=F.normalize(k.float().reshape(Bn,mh,dh),dim=-1); vf=v.float().reshape(Bn,mh,dh)
            if dp2_: k2f=F.normalize(k2_.float().reshape(Bn*mh,dh),dim=-1); v2f=v2_.float().reshape(Bn*mh,dh); be2=bt2_.float().reshape(Bn*mh,1)
            be=bt.float().reshape(Bn,mh) if mb else torch.sigmoid(b.beta.float()).expand(Bn,mh)
            R=Bn*mh
            if _SAMP['W']>0:                                               # WINDOWED SAMPLER: S = sum_j exp(L_now - L_j) u_j k_j^T, last W
                Wn=_SAMP['W']; dt_=torch.bfloat16 if _SAMP['bf16'] else torch.float32
                if 'U' not in st:
                    st['U']=torch.zeros(R,Wn,dh,dtype=dt_,device=x.device); st['K']=torch.zeros(R,Wn,dh,dtype=dt_,device=x.device)
                    st['Lw']=torch.full((R,Wn),1e9,device=x.device); st['Lc']=torch.zeros(R,device=x.device); st['p']=0
                if md: st['Lc']=st['Lc']+la.float().reshape(R)
                kr=kf.reshape(R,dh); w_=torch.exp((st['Lc'][:,None]-st['Lw']).clamp(max=0.0))              # decay since each write
                if _SAMP.get('mm') and dt_!=torch.float32:                                                # window products in bf16 (no fp32
                    sc=torch.bmm(st['K'],kr.to(dt_).unsqueeze(-1)).squeeze(-1).float()*w_                 #   copies of the whole window per step)
                    pred=torch.bmm(sc.to(dt_).unsqueeze(1),st['U']).squeeze(1).float()
                else:
                    sc=torch.bmm(st['K'].float(),kr.unsqueeze(-1)).squeeze(-1)*w_                        # (R, W)
                    pred=torch.bmm(sc.unsqueeze(1),st['U'].float()).squeeze(1)                           # a_t S_{t-1} k_t
                sl=st['p']%Wn; st['U'][:,sl]=((vf.reshape(R,dh)-pred)*be.reshape(R,1)).to(dt_); st['K'][:,sl]=_kw_rec(st,kr).to(dt_)
                st['Lw'][:,sl]=st['Lc']; st['p']+=1
                if dp2_:                                                       # DELTAPRODUCT second write (no decay in between)
                    w2_=torch.exp((st['Lc'][:,None]-st['Lw']).clamp(max=0.0))
                    p2_=torch.bmm((torch.bmm(st['K'].float(),k2f.unsqueeze(-1)).squeeze(-1)*w2_).unsqueeze(1),st['U'].float()).squeeze(1)
                    sl=st['p']%Wn; st['U'][:,sl]=((v2f-p2_)*be2).to(dt_); st['K'][:,sl]=k2f.to(dt_); st['Lw'][:,sl]=st['Lc']; st['p']+=1
            else:
                S=st.get('S')
                if S is None: S=st['S']=torch.zeros(R,dh,dh,device=x.device)       # (rows x heads, value dim, key dim), updated in place
                if md: S.mul_(torch.exp(la.float().reshape(R))[:,None,None])
                kr=kf.reshape(R,dh)
                pred=torch.bmm(S,kr.unsqueeze(-1)).squeeze(-1)                     # a S_{t-1} k_t
                S.baddbmm_(((vf.reshape(R,dh)-pred)*be.reshape(R,1)).unsqueeze(-1),_kw_rec(st,kr).unsqueeze(1))   # + beta (v - S k) k_w^T
                if dp2_:                                                       # DELTAPRODUCT second write
                    p2_=torch.bmm(S,k2f.unsqueeze(-1)).squeeze(-1); S.baddbmm_(((v2f-p2_)*be2).unsqueeze(-1),k2f.unsqueeze(1))
            out=pred.reshape(Bn,1,C)
    g_=b.gate(hl if getattr(b,'gate_ln',False) else x)
    x=x+b.Wout(out*F.silu(g_)); hn=b.norm(x0 if getattr(b,'parallel',False) else x)   # C3: parallel block reads the block input
    x=x+b.mlp_out(F.gelu(b.mlp_in(film_mod(b,hn) if _FILM.get('codes') is not None else hn)))
    if getattr(b,'pu_in',None) is not None and _XDG.get('tagon') and isinstance(_XDG.get('rows'),(list,tuple)) and len(_XDG['rows'])==x.shape[0]:
        x=x+pu_term(b,hn).to(x.dtype)                                 # PRIVATE TRUNK UNITS in the recurrent sampler (rows tagged by their domain)
    return x+b.nov_term(hn,rec=True) if getattr(b,'nov_g',None) else x

def _lowrank_(S, r):
    """In place: each (dv x dk) state matrix replaced by its best rank-r approximation."""
    if r>=min(S.shape[-2:]): return
    with torch.autocast(device_type=S.device.type, enabled=False):              # fp32 (the sampler runs inside bf16 autocast; QR has no bf16)
        U,sv,V=torch.svd_lowrank(S.float(),q=min(r+6,min(S.shape[-2:])),niter=2)   # randomised (the states are close to low rank)
        S.copy_(((U[...,:r]*sv[...,None,:r])@V[...,:r].transpose(-1,-2)).to(S.dtype))

@torch.no_grad()
def state_spectrum_diag(model, cfg, olds, n_rows=None):
    """STATE SPECTRUM while dreaming (diagnostic; own random stream). 1) Exact recurrent sampler from the visit-start teacher, rows spread
    over the old domains: at the end, per block, the singular-value spectrum of every head's state matrix (rank holding 90% / 99% of the
    energy, participation ratio), mean decay (memory timescale 1/(1-a)) and mean beta. 2) The same dreams re-run teacher-forced with a
    CHEAPER state - truncated to rank r after every update (all blocks, or one block at a time at the smallest rank), or stored in bf16 - and
    the KL of its next-byte predictions to the exact ones (and top-byte agreement)."""
    TT=getattr(model,'_teach',None)
    if TT is None or not olds: return
    n_rows=n_rows or cfg.state_spec_rows
    dev=model.embed.weight.device; T=min(cfg.dream_len,cfg.seq_len); blocks=TT.blocks['main']
    doms=sorted([olds[i%len(olds)] for i in range(n_rows)])
    g_=torch.Generator(device=dev); g_.manual_seed(cfg.seed+53)
    stat=[{'a':0.0,'b':0.0,'n':0} for _ in blocks]
    def run(x_forced, comp):
        sts=[{} for _ in blocks]; out=[]; x=x_forced[:,:1] if x_forced is not None else torch.full((n_rows,1),10,dtype=torch.long,device=dev); seq=[x]
        groups=[]; i0=0
        for i in range(1,n_rows+1):
            if i==n_rows or doms[i]!=doms[i0]: groups.append((doms[i0],slice(i0,i))); i0=i
        hist=None
        for t in range(T):
            h=TT._emb_tok(x,t,'main',hist); hist=_hist_push(TT,hist,x)
            with TT._ac():
                for li,(b,st) in enumerate(zip(blocks,sts)):
                    if comp is None and x_forced is None and getattr(b,'md',False):      # exact pass: decay and beta statistics
                        hl=b.ln(h); a_=torch.exp(-F.softplus(b.Wa(hl).float())); stat[li]['a']+=float(a_.mean())
                        if getattr(b,'mb',False): stat[li]['b']+=float((b.bmax*torch.sigmoid(b.Wb(hl).float())).mean())
                        stat[li]['n']+=1
                    h=_rec_step(b,h,st)
                    if comp is not None and 'S' in st:
                        kind,r,which=comp
                        if which is None or which==li:
                            if kind=='rank': _lowrank_(st['S'],r)
                            elif kind=='bf16': st['S'].copy_(st['S'].bfloat16().float())
            hh=TT._out(h)[:,0]
            lg=torch.cat([model.heads[d](hh[sl]) for d,sl in groups]).float(); out.append(F.log_softmax(lg,-1))
            if t<T-1:
                x=x_forced[:,t+1:t+2] if x_forced is not None else torch.multinomial(F.softmax(lg/cfg.dream_temp,-1),1,generator=g_)
                seq.append(x)
        return torch.cat(seq,1), torch.stack(out,1), sts
    xs,lp0,sts=run(None,None)
    lines=[]
    for li,(b,st) in enumerate(zip(blocks,sts)):
        if 'S' not in st: lines.append(f"b{li} local filter"); continue
        sv=torch.linalg.svdvals(st['S']).pow(2); sv=sv/sv.sum(-1,keepdim=True).clamp(min=1e-20)   # (rows*heads, d)
        cs=torch.cumsum(sv,-1); r90=float(((cs<0.90).sum(-1)+1).float().mean()); r99=float(((cs<0.99).sum(-1)+1).float().mean())
        pr=float((1.0/sv.pow(2).sum(-1).clamp(min=1e-20)).mean()); dh=sv.shape[-1]
        a_=stat[li]['a']/max(stat[li]['n'],1); b_=stat[li]['b']/max(stat[li]['n'],1)
        lines.append(f"b{li} rank90 {r90:.1f} rank99 {r99:.1f} PR {pr:.1f} of {dh} | decay {a_:.3f} (memory ~{1/max(1-a_,1e-6):.0f} bytes) beta {b_:.2f}")
    print("  [state spectrum] while dreaming, per block, each head's state matrix at the end of a dream: "+" || ".join(lines))
    burn=min(cfg.dream_burnin,T//2)
    def kl(lp):
        k=(lp0.exp()*(lp0-lp)).sum(-1)[:,burn:]; ag=(lp0.argmax(-1)==lp.argmax(-1)).float()[:,burn:]
        return float(k.mean()), float(ag.mean())
    res=[]
    for r in cfg.state_spec_ranks: _,lp,_=run(xs,('rank',r,None)); res.append((f"rank {r}",)+kl(lp))
    _,lp,_=run(xs,('bf16',0,None)); res.append(("bf16",)+kl(lp))
    r0=min(cfg.state_spec_ranks); per=[]
    for li in range(len(blocks)): _,lp,_=run(xs,('rank',r0,li)); per.append((li,)+kl(lp))
    print("  [state spectrum] cheaper sampler vs exact on the same dreams, KL of next-byte predictions / top-byte agreement: "
          +" | ".join(f"{n} {k:.4f}/{a:.3f}" for n,k,a in res)
          +f" || rank {r0} in ONE block only: "+" ".join(f"b{li} {k:.4f}/{a:.3f}" for li,k,a in per))
    model.__dict__.setdefault('_spec_log',{})[getattr(model,'_visit',0)]={'lines':lines,'res':res,'per':per}

@torch.no_grad()
def rec_sample(TT, role, doms, T, temp, gen, tgt_T=None, start=10, prompt=None):
    sv_=(_XDG.get('rows'),_XDG.get('groups'),_XDG.get('rows_ix'))
    if _XDG.get('tagon'):
        _XDG['rows']=list(doms)                                          # per-row domain tags for private trunk units (pu_k)
        gr_=[]; i0=0; n_=len(doms)
        for i in range(1,n_+1):
            if i==n_ or doms[i]!=doms[i0]: gr_.append((int(doms[i0]),slice(i0,i))); i0=i
        _XDG['groups']=(n_,gr_); _XDG['rows_ix']=torch.tensor([int(d) for d in doms],device=TT.model.embed.weight.device)
    try: return _rec_sample(TT, role, doms, T, temp, gen, tgt_T, start, prompt)
    finally: _XDG['rows'],_XDG['groups'],_XDG['rows_ix']=sv_
def _rec_sample(TT, role, doms, T, temp, gen, tgt_T=None, start=10, prompt=None):
    """RECURRENT DREAM SAMPLER: rows = old domains (one head per row, rows grouped by domain), all generated together from teacher
    `role`, one token at a time with the recurrent mixer state (O(T) instead of O(T^2)). Same sampling as dream_sample (one multinomial
    draw per step over all rows). tgt_T: also return the teacher's log-probabilities log_softmax(logits / tgt_T) at every position."""
    model=TT.model; blocks=TT.blocks[role]; n=len(doms); dev=model.embed.weight.device
    groups=[]; i0=0
    for i in range(1,n+1):
        if i==n or doms[i]!=doms[i0]: groups.append((doms[i0],slice(i0,i))); i0=i
    P=0 if prompt is None else prompt.shape[1]                      # PROMPTED: the first P bytes are given (teacher-forced), the rest sampled
    x=prompt[:,:1] if P else torch.full((n,1),start,dtype=torch.long,device=dev); seq=[x]; sts=[{} for _ in blocks]
    tg=torch.empty(n,T,model.cfg.vocab,device=dev) if tgt_T else None
    hist=None; xw_=None; t=0
    while t<T:
        if _SAMP.get('graph') and P==0 and tg is None and t==int(_SAMP.get('gwarm',64)) and T-t>4:   # v9.8 cg8: the remaining bytes as graph replays
            x,t,hist,xw_=_rsg_run(TT,role,blocks,sts,groups,x,t,hist,xw_,T,temp,gen,seq)
            if t>=T: break
        lg,hist,xw_=_rs_body(TT,role,blocks,sts,groups,x,t,hist,xw_)
        if tg is not None: tg[:,t]=F.log_softmax(lg/tgt_T,-1)
        if t<T-1:
            x=prompt[:,t+1:t+2] if t+1<P else torch.multinomial(F.softmax(lg/temp,-1),1,generator=gen)
            seq.append(x)
        t+=1
    xs=torch.cat(seq,1)
    if tg is not None and _SAMP['W']>0 and _SAMP['rescore']:          # EXACT targets: parallel teacher pass over the generated text
        tg=_rescore_targets(TT,role,groups,xs,tgt_T,tg)
    return xs, tg

def _rsg_msg():
    if not _SAMP.get('graph'): return ""
    g_=_SAMP.get('gstat') or {}
    if not g_.get('calls'): return " | sampler graph: not used"
    return (f" | sampler graph: {g_['ok']}/{g_['calls']} sampling calls replayed ({g_['bytes']} bytes), self-check bit-identical" if g_['ok']==g_['calls']
            else f" | sampler graph: {g_['ok']}/{g_['calls']} calls replayed, eager otherwise ({g_.get('why')})")

def _rs_body(TT,role,blocks,sts,groups,x,t,hist,xw_):
    """One byte of the recurrent sampler (eager): logits for every row, state advanced in place / rebound in sts, hist, xw_."""
    model=TT.model
    h=TT._emb_tok(x,t,role,hist); hist=_hist_push(TT,hist,x)
    with TT._ac():
        for b,st in zip(blocks,sts): h=_rec_step(b,h,st)
    h=TT._out(h)[:,0]
    if model.cfg.ngp and model.__dict__.get('_ngp') is not None:   # PRIVATE N-GRAM ROWS in the recurrent sampler (last max(ngp) bytes)
        xw_=x if xw_ is None else torch.cat([xw_,x],1)[:,-max(model.cfg.ngp):]   # rolling window (was: whole history re-joined per byte)
        lg=torch.cat([model.heads[d](h[sl]+ngp_vec(model,xw_[sl],d)[:,-1]) for d,sl in groups]).float()
    else: lg=torch.cat([model.heads[d](h[sl]) for d,sl in groups]).float()
    return lg,hist,xw_

def _rec_tail(b,x,x0,hl,out):
    """Block output after the mixer (same arithmetic as _rec_step)."""
    g_=b.gate(hl if getattr(b,'gate_ln',False) else x)
    x=x+b.Wout(out*F.silu(g_)); hn=b.norm(x0 if getattr(b,'parallel',False) else x)
    x=x+b.mlp_out(F.gelu(b.mlp_in(film_mod(b,hn) if _FILM.get('codes') is not None else hn)))
    if getattr(b,'pu_in',None) is not None and _XDG.get('tagon') and isinstance(_XDG.get('rows'),(list,tuple)) and len(_XDG['rows'])==x.shape[0]:
        x=x+pu_term(b,hn).to(x.dtype)
    return x+b.nov_term(hn,rec=True) if getattr(b,'nov_g',None) else x

def _rec_step_ip(b,x,g,tb):
    """_rec_step with every state write IN PLACE at a device-side position tb (graph-capturable): local-filter blocks and windowed
    delta-rule blocks (decay, per-token beta, block-level write-key preconditioner whose D is refreshed on the host between replays)."""
    hl=b.ln(x); Bn=x.shape[0]; C=b.d_state; x0=x
    if getattr(b,'loc',False):
        v=b.Wv(hl)
        out=F.conv1d(g['bl'].transpose(1,2),b.conv_loc.to(v.dtype),groups=C).transpose(1,2).float()
        g['bl'].copy_(torch.cat([g['bl'][:,1:],v],1))
    else:
        mh=getattr(b,'mh',1); mb=getattr(b,'mb',False); md=getattr(b,'md',False)
        k=b.Wk(hl); v=b.Wv(hl); dh=C//mh
        bt=b.bmax*torch.sigmoid(b.Wb(hl)) if mb else None
        la=-F.softplus(b.Wa(hl)) if md else None
        with torch.autocast(device_type=x.device.type, enabled=False):
            kf=F.normalize(k.float().reshape(Bn,mh,dh),dim=-1); vf=v.float().reshape(Bn,mh,dh)
            be=bt.float().reshape(Bn,mh) if mb else torch.sigmoid(b.beta.float()).expand(Bn,mh)
            R=Bn*mh; Wn=_SAMP['W']; dt_=torch.bfloat16 if _SAMP['bf16'] else torch.float32
            if md: g['Lc'].add_(la.float().reshape(R))
            kr=kf.reshape(R,dh); w_=torch.exp((g['Lc'][:,None]-g['Lw']).clamp(max=0.0))
            if _SAMP.get('mm') and dt_!=torch.float32:
                sc=torch.bmm(g['K'],kr.to(dt_).unsqueeze(-1)).squeeze(-1).float()*w_
                pred=torch.bmm(sc.to(dt_).unsqueeze(1),g['U']).squeeze(1).float()
            else:
                sc=torch.bmm(g['K'].float(),kr.unsqueeze(-1)).squeeze(-1)*w_
                pred=torch.bmm(sc.unsqueeze(1),g['U'].float()).squeeze(1)
            sl=torch.remainder(tb,Wn)
            g['U'].index_copy_(1,sl,((vf.reshape(R,dh)-pred)*be.reshape(R,1)).to(dt_).unsqueeze(1))
            if _PREC['p']:
                kw=g['kD']*kr; g['kc'].add_(kr*kr); kw=kw/(kw*kr).sum(-1,keepdim=True).clamp(min=1e-6)
            else: kw=kr
            g['K'].index_copy_(1,sl,kw.to(dt_).unsqueeze(1))
            g['Lw'].index_copy_(1,sl,g['Lc'].unsqueeze(1))
        out=pred.reshape(Bn,1,C)
    return _rec_tail(b,x,x0,hl,out)

def _rsg_supported(TT,blocks,sts,t):
    if getattr(TT,'cnt',None) is not None or _SAMP['W']<=0: return 'count features or no window'
    if _PREC['p'] and _PREC['C']<=0: return 'per-token preconditioner'
    for b,st in zip(blocks,sts):
        if getattr(b,'lattn',False) or getattr(b,'mk',0) or getattr(b,'dp2',False) or getattr(b,'mq',False): return 'block type'
        if getattr(b,'loc',False):
            if 'bl' not in st: return 'local state missing'
        else:
            if 'U' not in st: return 'window state missing'
            if _PREC['p'] and ('kc' not in st or ('kD' not in st and t%int(_PREC['C'])!=0)): return 'preconditioner state'
    return None

def _rsg_kd(gst,t):
    """Host-side refresh of the block-level write-key preconditioner (as _kw_rec at n0 = t writes)."""
    if not (_PREC['p'] and _PREC['C']>0 and t>0 and t%int(_PREC['C'])==0): return
    for g in gst:
        if 'kc' in g:
            dh=g['kc'].shape[-1]
            with torch.autocast(device_type=g['kc'].device.type, enabled=False): g['kD'].copy_((g['kc']/t*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0))

@torch.no_grad()
def _rsg_run(TT,role,blocks,sts,groups,x,t,hist,xw_,T,temp,gen,seq):
    """CUDA-GRAPH SAMPLER from byte t on. The graph runs on CLONES of the eager state; the first two bytes are run both ways (eager advances
    the real state and draws the bytes) and must agree bit for bit (logits and states), then the graph takes over. Returns (x, t, hist, xw_):
    t = T when the graph finished, else the eager loop continues from t with the real state."""
    model=TT.model; dev=x.device; real_=dev.type=='cuda' and not _SAMP.get('gfake'); st_=_SAMP.setdefault('gstat',{'calls':0,'ok':0,'bytes':0,'why':None,'diff':0.0})
    st_['calls']+=1
    why=_rsg_supported(TT,blocks,sts,t)
    if why is not None: st_['why']=why; return x,t,hist,xw_
    ngw_=bool(model.cfg.ngp and model.__dict__.get('_ngp') is not None)
    if ngw_ and (xw_ is None or xw_.shape[1]!=max(model.cfg.ngp)): st_['why']='n-gram window'; return x,t,hist,xw_
    gst=[]
    for b,st in zip(blocks,sts):
        if getattr(b,'loc',False): gst.append({'bl':st['bl'].clone()})
        else:
            g={k_:st[k_].clone() for k_ in ('U','K','Lw','Lc')}
            if _PREC['p']: g['kc']=st['kc'].clone(); g['kD']=st['kD'].clone() if 'kD' in st else torch.zeros_like(st['kc'])
            gst.append(g)
    xb=x.clone(); hb=hist.clone() if hist is not None else None; xwb=xw_.clone() if ngw_ else None
    tb=torch.full((1,),int(t),dtype=torch.long,device=dev); ng=TT._ng(); lgb=torch.empty(x.shape[0],model.cfg.vocab,device=dev)
    amp_=(TT.cfg.amp=='bf16')
    def gstep():
        if TT.emb is None: e=model.embed(xb)+model.pos.weight.index_select(0,tb)
        else: E,P_=TT.emb[role]; e=F.embedding(xb,E)+P_.index_select(0,tb)
        if ng is not None:
            if hb is None: raise RuntimeError('no history buffer')
            e=e+ngram_embed_tok(ng,xb,hb[:,hb.shape[1]-(max(ng.ns)-1):] if max(ng.ns)>1 else hb[:,:0])
        if hb is not None and hb.shape[1]>0: hb.copy_(torch.cat([hb[:,1:],xb],1))
        h=e
        with (torch.autocast(device_type=dev.type,dtype=torch.bfloat16,cache_enabled=False) if amp_ else contextlib.nullcontext()):
            for b,g in zip(blocks,gst): h=_rec_step_ip(b,h,g,tb)
        h=h.float()[:,0]
        if ngw_:
            xwb.copy_(torch.cat([xwb[:,1:],xb],1))
            lg=torch.cat([model.heads[d](h[sl]+ngp_vec(model,xwb[sl],d)[:,-1]) for d,sl in groups]).float()
        else: lg=torch.cat([model.heads[d](h[sl]) for d,sl in groups]).float()
        lgb.copy_(lg); tb.add_(1)
    G=None; took_=False
    try:
        if real_:
            bufs=[xb,tb]+([hb] if hb is not None else [])+([xwb] if xwb is not None else [])+[v_ for g in gst for v_ in g.values()]
            snap=[v_.clone() for v_ in bufs]
            s_=torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for _ in range(2): gstep()
            torch.cuda.current_stream().wait_stream(s_)
            for v_,o_ in zip(bufs,snap): v_.copy_(o_)
            del snap
            G=torch.cuda.CUDAGraph()
            with torch.cuda.graph(G): gstep()
        rep=(G.replay if G is not None else gstep)
        ok=True; md_=0.0
        for c_ in range(2):                                           # SELF-CHECK: two bytes both ways
            _rsg_kd(gst,t); rep()
            lg,hist,xw_=_rs_body(TT,role,blocks,sts,groups,x,t,hist,xw_)
            if not torch.equal(lgb,lg): ok=False; md_=max(md_,float((lgb-lg).abs().max()))
            if t<T-1:
                x=torch.multinomial(F.softmax(lg/temp,-1),1,generator=gen); seq.append(x); xb.copy_(x)
            t+=1
        if ok:
            for b,st,g in zip(blocks,sts,gst):
                for k_,v_ in g.items():
                    if k_ in st and torch.is_tensor(st[k_]) and not torch.equal(st[k_],v_): ok=False
        if not ok:
            st_['why']=f'self-check mismatch (max logit diff {md_:.1e})'; st_['diff']=max(st_['diff'],md_); return x,t,hist,xw_
        st_['ok']+=1; took_=True
        while t<T:
            _rsg_kd(gst,t); rep()
            if t<T-1:
                x=torch.multinomial(F.softmax(lgb/temp,-1),1,generator=gen); seq.append(x); xb.copy_(x)
            t+=1; st_['bytes']+=1
        return x,t,hist,xw_
    except Exception as ex:
        st_['why']=f'error {repr(ex)[:160]}'
        if took_: raise                                               # the graph owned the state: no eager continuation possible
        return x,t,hist,xw_                                           # before the hand-over the real state is intact: eager continues
    finally:
        del G

def _stack_ok(TT):
    """The stacked sampler covers the configurations in use: stackable teachers, shared embeddings, no key/value convolution."""
    m=TT.model
    return bool(TT.stackable and TT.emb is None and all(getattr(b,'mk',0)==0 and not getattr(b,'mq',False) for b in m.blocks))

@torch.no_grad()
def rec_sample_multi(TT, specs, T, temp, gen, tgt_T, start=10):
    """ONE recurrent sampling loop for rows from several teachers and domains. specs: [(role, domain, prompt or None)]. Teachers are stacked
    (batched matmuls over roles; rows padded per role with copies of that role's first row), every row's logits come from one matmul over all
    the old heads (each row then takes its own head), warm rows are teacher-forced through their prompt, cold rows start from `start`.
    Returns xs (n,T) and targets (n,T,V) in spec order: re-scored by the exact parallel teacher when the sampler is windowed, else the sampler's."""
    model=TT.model; dev=model.embed.weight.device; V=model.cfg.vocab
    roles=sorted(set(r for r,_,_ in specs)); M=len(roles)
    by_role={r:[i for i,sp in enumerate(specs) if sp[0]==r] for r in roles}
    N=max(len(v) for v in by_role.values())
    order=[]                                                                       # flat row -> spec index (padding: the role's first row)
    for r in roles: ix=by_role[r]; order+=ix+[ix[0]]*(N-len(ix))
    order_t=torch.tensor(order,device=dev); R0=M*N
    doms=sorted(set(sp[1] for sp in specs)); dpos={d:i for i,d in enumerate(doms)}
    dix=torch.tensor([dpos[specs[i][1]] for i in order],device=dev)
    Wh=torch.cat([model.heads[d].weight.detach().float() for d in doms]); bh=torch.cat([model.heads[d].bias.detach().float() for d in doms])
    cap=model.heads[doms[0]].cap; ar=torch.arange(R0,device=dev)
    P=max([sp[2].shape[0] for sp in specs if sp[2] is not None] or [0])
    forced=torch.tensor([specs[i][2] is not None for i in order],device=dev)
    pr=torch.full((R0,max(P,1)),start,dtype=torch.long,device=dev)
    for j,i in enumerate(order):
        if specs[i][2] is not None: pr[j,:P]=specs[i][2][:P]
    x=pr[:,:1].clone(); seq=[x]; Ps=TT._stack(roles); sts=[{} for _ in Ps]; hist=None
    resc=_SAMP['W']>0 and _SAMP['rescore']
    tg=None if (resc or not tgt_T) else torch.empty(R0,T,V,device=dev)
    for t in range(T):
        e=TT._emb_tok(x,t,roles[0],hist); hist=_hist_push(TT,hist,x)
        h=e.reshape(M,N,1,-1)
        with TT._ac():
            for P_,st in zip(Ps,sts): h=_rec_step_stk(P_,h,st,M,N)
        hf=TT._out(h).reshape(R0,-1)
        z=(hf@Wh.T+bh).view(R0,len(doms),V)[ar,dix]
        if cap>0: z=cap*torch.tanh(z/cap)
        if tg is not None: tg[:,t]=F.log_softmax(z/tgt_T,-1)
        if t<T-1:
            smp=torch.multinomial(F.softmax(z/temp,-1),1,generator=gen)
            x=torch.where(forced[:,None]&(t+1<P),pr[:,min(t+1,max(P,1)-1):min(t+1,max(P,1)-1)+1],smp)
            seq.append(x)
    xs_all=torch.cat(seq,1); n=len(specs); first={}
    for j,i in enumerate(order): first.setdefault(i,j)
    pick=torch.tensor([first[i] for i in range(n)],device=dev); xs=xs_all[pick]
    if tg is not None: return xs, tg[pick]
    if not tgt_T: return xs, None
    out=torch.empty(n,T,V,device=dev); ch=max(4,int(128*(256/max(T,1))**2))
    for r in roles:                                                                # exact targets, per teacher, rows grouped by domain
        ix=sorted(by_role[r],key=lambda i:specs[i][1]); xr=xs[ix]; tr=torch.empty(len(ix),T,V,device=dev)
        gr=[]; i0=0
        for i in range(1,len(ix)+1):
            if i==len(ix) or specs[ix[i]][1]!=specs[ix[i0]][1]: gr.append((specs[ix[i0]][1],slice(i0,i))); i0=i
        tr=_rescore_targets(TT,r,gr,xr,tgt_T,tr,chunk=ch); out[torch.tensor(ix,device=dev)]=tr
    return xs, out

@torch.no_grad()
def _rec_step_stk(P, h, st, M, N):
    assert not _DP2['on'], 'dp_steps 2 is not implemented in the stacked sampler (use dream_bank / dream_stack False)'
    """One token through one block for M stacked teachers: h (M,N,1,d); same arithmetic as _rec_step (windowed or full-state memory)."""
    mh,mk,mb,md,bmax,gln,kern,C,loc=P['meta']; dev=h.device
    hl=_stk_ln(h,P['ln_w'],P['ln_b']); v=_stk_lin(hl,P['Wv'])
    if loc:                                                                        # local filter: the previous K values
        K=P['cl'].shape[-1]; cl=P['cl'].reshape(M,C,K)
        if 'bl' not in st: st['bl']=torch.zeros(M,N,K,C,dtype=v.dtype,device=dev)
        out=torch.einsum('mnkc,mck->mnc',st['bl'].float(),cl.float()).unsqueeze(2)
        st['bl']=torch.cat([st['bl'][:,:,1:],v.to(st['bl'].dtype)],2)
    else:
        k=_stk_lin(hl,P['Wk'])
        bt=bmax*torch.sigmoid(_stk_lin(hl,P['Wb_w'],P['Wb_b'])) if mb else None
        la=-F.softplus(_stk_lin(hl,P['Wa_w'],P['Wa_b'])) if md else None
        with torch.autocast(device_type=dev.type, enabled=False):
            Bn=M*N; dh=C//mh; R=Bn*mh
            kf=F.normalize(k.float().reshape(Bn,mh,dh),dim=-1); vf=v.float().reshape(Bn,mh,dh)
            be=bt.float().reshape(Bn,mh) if mb else torch.sigmoid(P['beta'].float())[:,None,None].expand(M,N,mh).reshape(Bn,mh)
            kr=kf.reshape(R,dh)
            if _SAMP['W']>0:
                Wn=_SAMP['W']; dt_=torch.bfloat16 if _SAMP['bf16'] else torch.float32
                if 'U' not in st:
                    st['U']=torch.zeros(R,Wn,dh,dtype=dt_,device=dev); st['K']=torch.zeros(R,Wn,dh,dtype=dt_,device=dev)
                    st['Lw']=torch.full((R,Wn),1e9,device=dev); st['Lc']=torch.zeros(R,device=dev); st['p']=0
                if md: st['Lc']=st['Lc']+la.float().reshape(R)
                w_=torch.exp((st['Lc'][:,None]-st['Lw']).clamp(max=0.0))
                if _SAMP.get('mm') and dt_!=torch.float32:
                    sc=torch.bmm(st['K'],kr.to(dt_).unsqueeze(-1)).squeeze(-1).float()*w_
                    pred=torch.bmm(sc.to(dt_).unsqueeze(1),st['U']).squeeze(1).float()
                else:
                    sc=torch.bmm(st['K'].float(),kr.unsqueeze(-1)).squeeze(-1)*w_
                    pred=torch.bmm(sc.unsqueeze(1),st['U'].float()).squeeze(1)
                sl=st['p']%Wn; st['U'][:,sl]=((vf.reshape(R,dh)-pred)*be.reshape(R,1)).to(dt_); st['K'][:,sl]=_kw_rec(st,kr).to(dt_)
                st['Lw'][:,sl]=st['Lc']; st['p']+=1
            else:
                S=st.get('S')
                if S is None: S=st['S']=torch.zeros(R,dh,dh,device=dev)
                if md: S.mul_(torch.exp(la.float().reshape(R))[:,None,None])
                pred=torch.bmm(S,kr.unsqueeze(-1)).squeeze(-1)
                S.baddbmm_(((vf.reshape(R,dh)-pred)*be.reshape(R,1)).unsqueeze(-1),_kw_rec(st,kr).unsqueeze(1))
            out=pred.reshape(M,N,1,C)
    g_=_stk_lin(hl if gln else h,P['gate_w'],P['gate_b'])
    h0=h; h=h+_stk_lin(out*F.silu(g_),P['Wout'])
    hn=_stk_ln(h0 if P['par'] else h,P['norm_w'],P['norm_b'])
    return h+_stk_lin(F.gelu(_stk_lin(hn,P['mi_w'],P['mi_b'])),P['mo_w'],P['mo_b'])

@torch.no_grad()
def _rescore_targets(TT, role, groups, xs, tgt_T, tg, chunk=128):
    """Teacher log-probabilities log_softmax(head_d(teacher(x)) / tgt_T) for every row, from the exact (parallel) teacher."""
    model=TT.model
    for c0 in range(0,xs.shape[0],chunk):
        c1=min(xs.shape[0],c0+chunk); h=TT.features(xs[c0:c1],role)
        for d,sl in groups:
            a,b=max(sl.start,c0),min(sl.stop,c1)
            if a<b: tg[a:b]=F.log_softmax(model.heads[d](h[a-c0:b-c0]).float()/tgt_T,-1)
    return tg

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_cols': 'platypus.ngrams',
    '_hist_len': 'platypus.ngrams',
    '_hist_push': 'platypus.ngrams',
    'count_feature': 'platypus.ngrams',
    'ngram_embed': 'platypus.ngrams',
    'ngram_embed_tok': 'platypus.ngrams',
    'run': 'platypus.runner',
}
