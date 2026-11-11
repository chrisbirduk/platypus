"""Output heads and the delta-rule core: WY solve, chunked and blocked kernels, DeltaBlock."""
from platypus.common import _fail
import contextlib
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from platypus.config import (
    Config, sample_theta,
)
from platypus.gating import (
    _track, _wl,
)
from platypus.protection import (
    _FILM, film_mod,
)
from platypus.private import (
    _XDG, cg_apply, inh_apply, pu_term, xdg_apply,
)

class Head(nn.Linear):
    """Output head; optional logit SOFT-CAP (cap*tanh(z/cap)) bounds how extreme any prediction can get."""
    cap=0.0
    def _c(s,z): return s.cap*torch.tanh(z/s.cap) if s.cap>0 else z
    def forward(s,x): return s._c(super().forward(x))
    def detached(s,x): return s._c(F.linear(x,s.weight.detach(),s.bias.detach()))   # same head, no gradient to its weights

# ---------------- DELTA-RULE CORE: torch WY solve, or the Triton chunked kernel (flash-linear-attention) ----------------
_FLA={'fn':None,'tried':False,'err':None,'dtype':torch.float32,'checked':{}}
def _fla_get():
    """chunk_gated_delta_rule from flash-linear-attention (pip install flash-linear-attention), or None."""
    if not _FLA['tried']:
        _FLA['tried']=True
        for attempt in (0,1):
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f_; _FLA['fn']=f_; _FLA['err']=None; break
            except Exception as e:
                _fail('layers:38')
                _FLA['err']=repr(e)[:300]
                if attempt==0 and isinstance(e,ModuleNotFoundError) and torch.cuda.is_available() and os.environ.get('FLA_AUTO_INSTALL','1')!='0':
                    import subprocess, importlib, sys as _sys    # kernels only (fla-core, no dependencies: never touches torch/triton)
                    print("  [kernel] installing the Triton kernels: pip install --no-deps fla-core einops ...", flush=True)
                    subprocess.run([_sys.executable,'-m','pip','install','-q','--no-deps','fla-core','einops'],check=False)
                    importlib.invalidate_caches()
                else: break
    return _FLA['fn']

_WIN={'W':0}                   # windowed WY solve (mix_window); set by setup_kernel
_SAMP={'W':0,'bf16':False,'rescore':True}     # windowed recurrent sampler (samp_window, samp_bf16, samp_rescore)
_ADPT={'on':False,'P':{}}                                                     # LOW-RANK MLP ADAPTER (oracle): x += B (A hn) beside each block's MLP
_PMEM={'on':False,'S':{}}                                                     # PERSISTENT MEMORY READ (oracle): out += S_b k~ at every position
_ORACLE={'on':False,'st':{}}                                                  # ORACLE MEMORY statistics collection
_BPTT={'K':1}                                                                 # segments per training row (ctx_bptt; set per visit)
_RC_GRAD={'on':True}   # replay training: the end-of-segment memory stays differentiable (False = the old detached behaviour)
class _SlotF:
    """v9.8 cg5 SLOT MEMORY of one delta block for nb = rows x heads: S = sum_s exp(Gt - Gs[s]) U[s]^T K[s] (factor slots; unused slots U = 0).
    Writes go to slot i (in place); Gt is the running log decay, Gs the log decay at each slot's write (so old slots are never rescaled)."""
    __slots__=('U','K','Gs','Gt','r0','i')
    def __init__(s,U,K,Gs,Gt,r0,i=0): s.U=U; s.K=K; s.Gs=Gs; s.Gt=Gt; s.r0=r0; s.i=i
_CHN={'on':False,'k':{}}   # merged-memory chaining loss: keys captured per delta block (inference.supmem)
_CARRY={'on':False,'nr':0,'st':{},'new':{},'sm':None,'slot':None,'rows':None}   # CONTINUOUS CONTEXT (ctx_carry): carried delta-rule memories of the real rows, per block
_PREC={'p':0.0,'C':0,'det':False}   # preconditioned delta-rule writes (mix_precond, mix_precond_chunk, mix_precond_detach); set by setup_kernel
def _smem_seed(slot,nr):
    """slot memory -> per-row initial states (each block's S_d repeated over the rows)"""
    SM=(_CARRY['sm'] or {}).get(slot) or {}
    return {k_:v_.unsqueeze(0).expand(int(nr),*v_.shape).reshape(int(nr)*v_.shape[0],*v_.shape[1:]) for k_,v_ in SM.items()}
def _carry_on(cfg,nr):
    if cfg.slot_mem:
        _CARRY['on']=True; _CARRY['nr']=int(nr); _CARRY['new']={}
        if cfg.slot_mem and not cfg.ctx_carry: _CARRY['st']=_smem_seed(_CARRY['slot'],nr)
def _carry_off(cfg):
    if _CARRY['on']:
        _CARRY['on']=False
        if _CARRY.get('rows') is not None:                                # paired-dream rows rode along: keep only the real rows' memories
            ns_=int(_CARRY['rows'].numel()); nr_=int(_CARRY['nr'])
            _CARRY['new']={k_:v_[:(v_.shape[0]//ns_)*nr_] for k_,v_ in _CARRY['new'].items()} if nr_>0 else {}
            _CARRY['rows']=None
        if cfg.slot_mem and _CARRY['sm'] is not None and _CARRY['slot'] is not None:      # S_d <- g S_d + (1 - g) mean of the segment-end states
            SM=_CARRY['sm'].setdefault(_CARRY['slot'],{}); g_=float(cfg.slot_mem_gamma)
            with torch.no_grad():
                for k_,v_ in _CARRY['new'].items():
                    if isinstance(k_,tuple): continue
                    nr_=int(_CARRY['nr']); m_=v_.reshape(nr_,v_.shape[0]//nr_,*v_.shape[1:]).mean(0)
                    o_=SM.get(k_); SM[k_]=m_ if (o_ is None or o_.shape!=m_.shape) else g_*o_+(1-g_)*m_
            _CARRY['st']=({})
        else: _CARRY['st']=_CARRY['new']
        _CARRY['new']={}
def _kw_par(k):
    if _PREC['C']>0: return _kw_par_blk(k)
    """Write keys of the preconditioned delta rule for normalised keys k (BH,T,dh): causal running mean of k^2 per feature -> diagonal
    preconditioner D = (c dh + 0.1)^-p in [0.25, 4]; k_w = D k / (k . D k) (so k_w . k = 1)."""
    dh=k.shape[-1]; c=torch.cumsum(k*k,1)/torch.arange(1,k.shape[1]+1,device=k.device,dtype=k.dtype)[None,:,None]
    D=(c*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0); kw=D*k
    return kw/(kw*k).sum(-1,keepdim=True).clamp(min=1e-6)
def _kw_par_blk(k):
    """Block-level preconditioner (mix_precond_chunk = C): D of block c from the mean k^2 over all positions of blocks < c (block 0: D = 1);
    normaliser n_t = sum_i D_i k_ti^2 per token. With mix_precond_detach, D and n carry no gradient (k_w = k * D / n is linear in k)."""
    BH,T,dh=k.shape; C=int(_PREC['C'])
    if T<C: C=T                                                                                        # v9c: one (partial) block - no padding (exact)
    nc=-(-T//C); pad=nc*C-T
    k2=k*k; k2p=F.pad(k2,(0,0,0,pad)) if pad else k2
    with torch.set_grad_enabled(torch.is_grad_enabled() and not _PREC['det']):
        k2b=k2p.reshape(BH,nc,C,dh); bs=k2b.sum(2)                                                     # block sums (BH,nc,dh)
        pre=torch.cumsum(bs,1)-bs                                                                      # sums over earlier blocks
        cnt=(torch.arange(nc,device=k.device,dtype=k.dtype)*C)[None,:,None].expand(BH,nc,1)
        pr_=_PREC.get('prior')                                                                         # CONTINUOUS CONTEXT: statistics of the
        if pr_ is not None:                                                                            #   earlier segments (first nb rows)
            P0_,n0_,nb_=pr_
            if torch.is_tensor(nb_): pre=pre.clone(); pre[nb_]+=P0_[:,None,:]; cnt=cnt.clone(); cnt[nb_]+=n0_      # selected rows (paired dreams)
            else: pre=torch.cat([pre[:nb_]+P0_[:,None,:],pre[nb_:]],0); cnt=torch.cat([cnt[:nb_]+n0_,cnt[nb_:]],0)
        D=(pre/cnt.clamp(min=1)*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0); D=torch.where(cnt>0,D,torch.ones_like(D))   # no history: D = 1
        n=torch.bmm(k2b.reshape(BH*nc,C,dh),D.reshape(BH*nc,dh,1)).reshape(BH,nc*C)[:,:T].clamp(min=1e-6)   # k . D k per token
        if _PREC['det']: D=D.detach(); n=n.detach()
    kb=F.pad(k,(0,0,0,pad)) if pad else k
    return (kb.reshape(BH,nc,C,dh)*D[:,:,None,:]).reshape(BH,nc*C,dh)[:,:T]/n[...,None]
def _kw_rec(st,kr):
    """The same write key in the recurrent samplers (running sums kept in the sampler state)."""
    if not _PREC['p']: return kr
    if _PREC['C']>0:                                                    # block-level statistics (as _kw_par_blk)
        C=int(_PREC['C']); dh=kr.shape[-1]; n0=st.get('kn',0)
        if n0>0 and n0%C==0: st['kD']=(st['kc']/n0*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0)
        D=st.get('kD'); kw=kr if D is None else D*kr
        st['kc']=kr*kr if 'kc' not in st else st['kc']+kr*kr; st['kn']=n0+1
        return kw if D is None else kw/(kw*kr).sum(-1,keepdim=True).clamp(min=1e-6)
    dh=kr.shape[-1]; st['kc']=kr*kr if 'kc' not in st else st['kc']+kr*kr; st['kn']=st.get('kn',0)+1
    D=(st['kc']/st['kn']*dh+0.1).pow(-_PREC['p']).clamp(0.25,4.0); kw=D*kr
    return kw/(kw*kr).sum(-1,keepdim=True).clamp(min=1e-6)

def _local_attn(k, v, mh, B, T, la_scale, la_rel, W):
    """Chunked sliding-window attention (fp32): position t reads positions t-W..t-1 (and a null slot)."""
    with torch.autocast(device_type=k.device.type, enabled=False):
        C=k.shape[-1]; dh=C//mh; BH=B*mh
        sp=lambda z: z.float().reshape(B,T,mh,dh).transpose(1,2).reshape(BH,T,dh)
        kn=F.normalize(sp(k),dim=-1); vv=sp(v); n_=-(-T//W); Tp=n_*W
        kp=F.pad(kn,(0,0,W,Tp-T)); vp=F.pad(vv,(0,0,W,Tp-T))
        Kc=kp.unfold(1,2*W,W).permute(0,1,3,2); Vc=vp.unfold(1,2*W,W).permute(0,1,3,2)          # (BH,n,2W,dh)
        Qc=F.pad(kn,(0,0,0,Tp-T)).reshape(BH,n_,W,dh)
        sc=torch.exp(la_scale.float()).repeat(B)                                                  # (BH,)
        S=torch.einsum('bnqd,bnkd->bnqk',Qc,Kc)*sc[:,None,None,None]
        a_=torch.arange(W,device=k.device)[:,None]; b_=torch.arange(2*W,device=k.device)[None,:]; off=W+a_-b_
        ok=(off>=1)&(off<=W)
        kabs=torch.arange(n_,device=k.device)[:,None]*W-W+torch.arange(2*W,device=k.device)[None,:]       # (n,2W) absolute key position
        mask=ok[None,:,:]&(kabs>=0)[:,None,:]                                                     # (n,W,2W)
        bias=la_rel.float().repeat(B,1)[:,(off-1).clamp(0,W-1)]                                    # (BH,W,2W)
        S=(S+bias[:,None]).masked_fill(~mask[None],float('-inf'))
        Pm=F.softmax(torch.cat([S,torch.zeros_like(S[...,:1])],-1),-1)[...,:2*W]
        out=torch.einsum('bnqk,bnkd->bnqd',Pm,Vc).reshape(BH,Tp,dh)[:,:T]
        return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)

_CHUNK={'C':0,'lp':None,'Cng':0,'B':None,'Bng':None}      # Cng: chunk for no-grad passes (bank targets: large batches, forward only)
# ---- BLOCKED FULL-WY SOLVE (tensor-core GEMMs, no d x d states, no T x T tensors) ----
_BLK={'prec':'tf32','emu':False}      # blocked solve: GEMM precision ('fp32' | 'tf32' | 'bf16'); emu = TF32 rounding emulated (CPU tests)
def _tf32_round(x):
    i=x.contiguous().view(torch.int32); return (i & ~0x1FFF).view(torch.float32)
def _mm(a,b):
    """bmm in the solve's GEMM precision: fp32 | tf32 (tensor cores, fp32 in/out) | bf16 (tensor cores, fp32 out via upcast)"""
    p=_BLK['prec']
    if p=='bf16': return torch.bmm(a.bfloat16(),b.bfloat16()).float()
    if p=='tf32' and _BLK['emu']: return torch.bmm(_tf32_round(a.float()),_tf32_round(b.float()))
    if p=='tf32' and a.is_cuda:                                        # TF32 for THIS product only (the small diagonal-block products stay fp32)
        pv_=torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32=True
        try: return torch.bmm(a,b)
        finally: torch.backends.cuda.matmul.allow_tf32=pv_
    return torch.bmm(a,b)
class _DeltaBlocked(torch.autograd.Function):
    """Delta rule with decay, read before write, as the FULL WY system (I+M) r = v, out = v - r, M_ts = beta_s exp(g_t-g_s) k_t.k_s (s<t),
    solved by BLOCKED substitution: diagonal blocks inverted in fp32 (small), every off-diagonal coupling a GEMM on the fly (decays folded into
    per-block scale vectors, all <= 1). No d x d states, no T x T tensors, no explicit inverse applied to large values (the precision-sensitive
    step of the chunk form). k (BH,T,dh) normalised, v (BH,T,dh), be (BH,T), g (BH,T) cumulative log decay; T a multiple of b."""
    @staticmethod
    def forward(ctx, k, v, be, g, b, kw=None):
        BH,T,dh=k.shape; n=T//b; dev=k.device; ctx.sep=kw is not None
        kw=k if kw is None else kw                                                                   # WRITE keys (preconditioned delta rule): columns
        Kb=k.reshape(BH,n,b,dh); KWb=kw.reshape(BH,n,b,dh); gb=g.reshape(BH,n,b); bb=be.reshape(BH,n,b)
        dd=gb[...,:,None]-gb[...,None,:]; low=torch.tril(torch.ones(b,b,dtype=torch.bool,device=dev),-1)
        Md=torch.einsum('bntd,bnsd->bnts',Kb,KWb)*torch.exp(torch.where(low,dd,torch.full_like(dd,-float('inf'))))*bb[...,None,:]   # diagonal blocks (fp32)
        I=torch.eye(b,device=dev,dtype=k.dtype)
        Linv=torch.linalg.solve_triangular(I+Md.reshape(BH*n,b,b),I.expand(BH*n,b,b),upper=False).reshape(BH,n,b,b)
        r=torch.empty_like(v)
        for i in range(n):
            s0,s1=i*b,(i+1)*b; rhs=v[:,s0:s1]
            if i:
                P=_mm(k[:,s0:s1],kw[:,:s0].transpose(1,2))                                      # K_i KW_<i^T         (b x ib)
                e=torch.exp(g[:,s0:s1]-g[:,s0:s0+1]); w=torch.exp(g[:,s0:s0+1]-g[:,:s0])*be[:,:s0]   # <= 1, <= beta
                rhs=rhs-_mm(P.mul_(e[...,None]).mul_(w[:,None,:]),r[:,:s0])                     # - M_{i,<i} r_<i (scaled in place)
            r[:,s0:s1]=torch.bmm(Linv[:,i],rhs)                                                 # fp32: small b x b
        ctx.save_for_backward(k,be,g,r,Linv,kw); ctx.b=b
        return v-r
    @staticmethod
    def backward(ctx, dout):
        """lam = L^-T (-dout); dv = dout + lam; dM = -lam r^T (strictly lower) -> dA, dbeta, dg, dk. Products are formed as +lam r^T and the
        sign folded into the accumulations; decays and beta scale the (b x .) tiles in place; E = G*beta never formed (row / column sums)."""
        k,be,g,r,Linv,kw=ctx.saved_tensors; b=ctx.b; BH,T,dh=k.shape; n=T//b; dev=k.device
        dout=dout.contiguous(); lam=torch.empty_like(r)
        for i in range(n-1,-1,-1):                                                                  # backward substitution
            s0,s1=i*b,(i+1)*b; rhs=-dout[:,s0:s1]
            if i<n-1:
                gL=g[:,s1-1:s1]
                Q=_mm(kw[:,s0:s1],k[:,s1:].transpose(1,2))                                          # KW_i K_>i^T         (b x rest)
                Q.mul_((torch.exp(gL-g[:,s0:s1])*be[:,s0:s1])[...,None]).mul_(torch.exp(g[:,s1:]-gL)[:,None,:])   # = M_{>i,i}^T (decays <= 1)
                rhs=rhs-_mm(Q,lam[:,s1:])
            lam[:,s0:s1]=torch.bmm(Linv[:,i].transpose(1,2),rhs)
        dv=dout+lam
        # all diagonal blocks at once
        lb=lam.view(BH,n,b,dh); rb=r.view(BH,n,b,dh); Kb=k.view(BH,n,b,dh); KWb=kw.view(BH,n,b,dh); gb=g.view(BH,n,b); bb=be.view(BH,n,b)
        low=torch.tril(torch.ones(b,b,dtype=torch.bool,device=dev),-1); dd=gb[...,:,None]-gb[...,None,:]
        Rd=torch.einsum('bntd,bnsd->bnts',lb,rb).mul_(torch.exp(torch.where(low,dd,torch.full_like(dd,-float('inf')))))   # -dM o gamma
        Gd=torch.einsum('bntd,bnsd->bnts',Kb,KWb).mul_(Rd)                                          # -G
        Ad=Rd.mul_(bb[...,None,:])                                                                  # -dA (in place: Rd no longer needed)
        csd=Gd.sum(-2)
        dbe=(-csd).reshape(BH,T).contiguous()
        dg=((bb*csd)-(Gd@bb[...,None]).squeeze(-1)).reshape(BH,T).contiguous()
        if ctx.sep:                                                                                 # A_ts = k_t . kw_s: rows -> dk, columns -> dkw
            dk=(-(Ad@KWb)).reshape(BH,T,dh).contiguous(); dkw=(-(Ad.transpose(-1,-2)@Kb)).reshape(BH,T,dh).contiguous()
        else: dk=(-(Ad@Kb)-(Ad.transpose(-1,-2)@Kb)).reshape(BH,T,dh).contiguous(); dkw=dk
        del Rd,Gd,Ad,csd
        for i in range(1,n):                                                                        # off-diagonal row blocks (s < block start)
            s0,s1=i*b,(i+1)*b; ki=k[:,s0:s1]
            R=_mm(lam[:,s0:s1],r[:,:s0].transpose(1,2))                                             # -dM                 (b x ib)
            R.mul_(torch.exp(g[:,s0:s1]-g[:,s0:s0+1])[...,None]).mul_(torch.exp(g[:,s0:s0+1]-g[:,:s0])[:,None,:])
            P=_mm(ki,kw[:,:s0].transpose(1,2)).mul_(R)                                              # -G
            A_=R.mul_(be[:,None,:s0])                                                               # -dA
            cs=P.sum(1)
            dbe[:,:s0]-=cs; dg[:,:s0]+=be[:,:s0]*cs; dg[:,s0:s1]-=torch.bmm(P,be[:,:s0,None]).squeeze(-1)
            dk[:,s0:s1]-=_mm(A_,kw[:,:s0]); dkw[:,:s0]-=_mm(A_.transpose(1,2),ki)
        return dk,dv,dbe,dg,None,(dkw if ctx.sep else None)
_DC1={'on':True,'n':0}
def delta_blocked(k,v,be,la,b=64,kw=None):
    """k,v (BH,T,dh) (k normalised), be (BH,T), la (BH,T) log decays or None -> (BH,T,dh) = a_t S_{t-1} k_t"""
    BH,T,dh=k.shape
    if T<b: b=T                                                      # short sequences (replay segments): one block, no padding
    n=-(-T//b); pad=n*b-T
    g=torch.cumsum(la.float() if la is not None else torch.zeros(BH,T,device=k.device),-1)
    if pad: k=F.pad(k,(0,0,0,pad)); v=F.pad(v,(0,0,0,pad)); be=F.pad(be,(0,pad)); g=torch.cat([g,g[:,-1:].expand(BH,pad)],1)
    # padded positions: k=0 -> no coupling; g REPEATS the last value (zero padding made exp(g_t - g_s) overflow to inf once the decay over
    #   the block passed ~88 nats, and 0 x inf = NaN reached the real positions in the backward pass)
    if kw is not None and pad: kw=F.pad(kw,(0,0,0,pad))
    if kw is not None: return _DeltaBlocked.apply(k.float().contiguous(),v.float().contiguous(),be.float().contiguous(),g.contiguous(),b,kw.float().contiguous())[:,:T]
    return _DeltaBlocked.apply(k.float().contiguous(),v.float().contiguous(),be.float().contiguous(),g.contiguous(),b)[:,:T]

def _delta_chunked(k,v,beta,mb,la_,S0_,Cc,kw=None):
    """CHUNKED delta rule (same recurrence as the full T x T solve below, never builds T x T tensors): k, v (BH,T,dh) with k normalised;
    beta (BH,1,T) per token or a scalar tensor; la_ (BH,T) log decays or None; S0_ (BH,dh,dh) initial state (value x key) or None.
    Per chunk of Cc positions: r = (I+M_c)^-1 (v_c - decayed state readout), output v - r, state passed to the next chunk.
    Returns (BH,T,dh) = a_t S_{t-1} k_t."""
    BH,T,dh=k.shape; n=-(-T//Cc); Tp=n*Cc; pad=Tp-T
    be=(beta.reshape(BH,T) if mb else beta.reshape(1,1).expand(BH,T)).float()
    lg=(la_ if la_ is not None else torch.zeros(BH,T,device=k.device,dtype=k.dtype)).float()
    if pad:
        k=F.pad(k,(0,0,0,pad)); v=F.pad(v,(0,0,0,pad)); be=F.pad(be,(0,pad)); lg=F.pad(lg,(0,pad))
        if kw is not None: kw=F.pad(kw,(0,0,0,pad))
    Kc=k.reshape(BH,n,Cc,dh); Vc=v.reshape(BH,n,Cc,dh); bc=be.reshape(BH,n,Cc); g=torch.cumsum(lg.reshape(BH,n,Cc),-1)   # local cumulative log decay
    KWc=Kc if kw is None else kw.reshape(BH,n,Cc,dh)                     # write keys (preconditioned delta rule) or the keys
    A=torch.einsum('bncd,bnsd->bncs',Kc,KWc)
    dd=g[...,:,None]-g[...,None,:]; A=A*torch.exp(torch.where(dd<=0,dd,torch.zeros_like(dd)))
    M=torch.tril(A,-1)*bc[...,None,:]                                    # M_ts = beta_s gamma_t/gamma_s k_t.k_s (s<t, same chunk)
    I=torch.eye(Cc,device=k.device,dtype=k.dtype).expand(BH*n,Cc,Cc)
    Tinv=torch.linalg.solve_triangular(I+M.reshape(BH*n,Cc,Cc),I,upper=False).reshape(BH,n,Cc,Cc)   # all chunks in one call
    S=S0_.float() if S0_ is not None else torch.zeros(BH,dh,dh,device=k.device,dtype=k.dtype)   # (value, key)
    gl=g[...,-1:]; eg=torch.exp(g); wdec=(bc*torch.exp(gl-g))[...,None]; egl=torch.exp(gl)[...,None]
    KL=Kc.unbind(1); KWL=KWc.unbind(1); VL=Vc.unbind(1); TL=Tinv.unbind(1); EL=eg.unbind(1); WL=wdec.unbind(1); GL=egl.unbind(1)   # unbind: ONE stack in backward
    outs=[]                                                                                                      #   (per-chunk slicing made it quadratic)
    lp_=_CHUNK.get('lp')                                                # LOW-PRECISION STATE PRODUCTS (tensor cores): the two d x d-sized products
    for c in range(n):                                                  #   per chunk in bf16, the state itself and the small solves stay fp32
        Kj=KL[c]; Vj=VL[c]
        if lp_ is not None:
            Kl=Kj.to(lp_)
            rhs=Vj-EL[c][...,None]*torch.bmm(Kl,S.to(lp_).transpose(1,2)).float()
            r=torch.bmm(TL[c],rhs); outs.append(Vj-r)
            S=GL[c]*S+torch.bmm((WL[c]*r).to(lp_).transpose(1,2),KWL[c].to(lp_)).float()
        else:
            rhs=Vj-EL[c][...,None]*torch.bmm(Kj,S.transpose(1,2))          # v_t - gamma_t/gamma_0 S k_t
            r=torch.bmm(TL[c],rhs); outs.append(Vj-r)
            S=GL[c]*S+torch.bmm((WL[c]*r).transpose(1,2),KWL[c])          # writes decayed to the chunk end
    return torch.stack(outs,1).reshape(BH,Tp,dh)[:,:T]
def _delta_core(k, v, bt, la, mh, beta_p, B, T, kern='torch', q=None, s0=None, carry=None):
    """k, v (B,T,C) projected (+ convolved) keys / values; bt (B,T,H) per-token write strengths or None (then sigmoid(beta_p));
    la (B,T,H) log decay or None; q (B,T,C) separate query or None. Returns the mixer output (B,T,C) in fp32:
    S_t = a_t S_{t-1} + beta_t (v_t - a_t S_{t-1} k_t) k_t^T, output a_t S_{t-1} k_t (read before write), or S_t q_t with a query."""
    with torch.autocast(device_type=k.device.type, enabled=False):
        C=k.shape[-1]; dh=C//mh
        if kern=='fla' and q is None and not _FLA.get('broken') and _FLA['dtype']==torch.bfloat16 and s0 is None and not _PREC['p'] and carry is None:
            fn=_fla_get(); dt=torch.bfloat16                                    # LEAN bf16 PATH: no fp32 copies; keys L2-normalised in the kernel
            kd=k.reshape(B,T,mh,dh).to(dt); vd=v.reshape(B,T,mh,dh).to(dt)
            qs=torch.cat([kd[:,1:],kd[:,-1:]],1)                                # SHIFTED QUERIES q_t = k_{t+1} (last one unused)
            be=(bt if bt is not None else torch.sigmoid(beta_p.float()).expand(B,T,mh)).to(dt)
            g=la.float() if la is not None else torch.zeros(B,T,mh,device=k.device)
            try: o,_=fn(qs,kd,vd,g,be,scale=1.0,use_qk_l2norm_in_kernel=True)
            except Exception as e:
                _fail('layers:308')
                _FLA['broken']=True; print(f"  [kernel] Triton kernel failed at shape {tuple(k.shape)} ({repr(e)[:150]}); torch solve from now on")
                return _delta_core(k,v,bt,la,mh,beta_p,B,T,'torch',q,s0)
            o=o[:,:-1]
            if la is not None: o=o*torch.exp(g[:,1:,:,None]).to(dt)
            return torch.cat([torch.zeros_like(o[:,:1]),o],1).reshape(B,T,C)  # output_t = a_t S_{t-1} k_t (bf16)
        if kern=='fla' and q is None and not _FLA.get('broken') and not _PREC['p'] and carry is None:
            fn=_fla_get(); dt=_FLA['dtype']
            kf=F.normalize(k.float().reshape(B,T,mh,dh),dim=-1); vf=v.float().reshape(B,T,mh,dh)
            be=bt.float() if bt is not None else torch.sigmoid(beta_p.float()).expand(B,T,mh)
            g=la.float() if la is not None else torch.zeros(B,T,mh,device=k.device)
            qs=torch.cat([kf[:,1:],torch.zeros_like(kf[:,:1])],1)               # SHIFTED QUERIES: q_t = k_{t+1}
            h0_=None if s0 is None else s0.float().transpose(-1,-2).contiguous().to(dt)   # FLA state convention (K, V)
            try: o,_=(fn(qs.to(dt),kf.to(dt),vf.to(dt),g.to(dt),be.to(dt),scale=1.0) if h0_ is None else
                      fn(qs.to(dt),kf.to(dt),vf.to(dt),g.to(dt),be.to(dt),scale=1.0,initial_state=h0_))   # o_t = S_t k_{t+1}
            except Exception as e:                                              # any later failure: torch solve from now on
                _fail('layers:323')
                _FLA['broken']=True; print(f"  [kernel] Triton kernel failed at shape {tuple(k.shape)} ({repr(e)[:150]}); torch solve from now on")
                return _delta_core(k,v,bt,la,mh,beta_p,B,T,'torch',q,s0)
            o=o.float()[:,:-1]                                                  # S_{t-1} k_t for t = 1..T-1
            if la is not None: o=o*torch.exp(g[:,1:,:,None])                    # x a_t
            o0=torch.zeros_like(vf[:,:1]) if s0 is None else (torch.einsum('bhvk,bhk->bhv',s0.float(),kf[:,0])*torch.exp(g[:,0])[...,None]).unsqueeze(1)
            out=torch.cat([o0,o],1)                                             # output_t = a_t S_{t-1} k_t, output_0 = a_0 S0 k_0 (0 without S0)
            return out.reshape(B,T,C)
        sp=lambda z: z.reshape(B,T,mh,dh).transpose(1,2).to(torch.float32,memory_format=torch.contiguous_format).reshape(B*mh,T,dh)   # one fused cast + transpose
        if carry is not None and q is None and s0 is None:            # CONTINUOUS CONTEXT: the real rows start from the carried memory S0 (detached).
            rows_=carry.get('rows'); nrw_=int(carry['nr']) if rows_ is None else int(rows_.numel())   # rows: which batch rows carry a memory
            nb_=nrw_*mh; Sin=carry.get('S'); fac_=bool(carry.get('fac'))  #   Its readout c_t = gamma_t S0 k_t enters as v - c (same triangular system in
            Fin=Sin if isinstance(Sin,tuple) else None                   #   FACTORED memory (U, K): S0 = U^T K
            Slt=Sin if isinstance(Sin,_SlotF) else None                  #   v9.8 SLOT memory (fixed size, in place)
            if Fin is not None or Slt is not None: Sin=None
            ib_=None if rows_ is None else (rows_.to(k.device)[:,None]*mh+torch.arange(mh,device=k.device)[None]).reshape(-1)
            SEL=(lambda z: z[:nb_]) if ib_ is None else (lambda z: z[ib_])
            kn_=F.normalize(SEL(sp(k)),dim=-1)                          #   every solve form); out = solve(v - c) + c; the final memory
            pk_=carry.get('P') if (_PREC['p'] and _PREC['C']>0) else None   # write-key preconditioner statistics carried too (block form)
            if pk_ is not None and tuple(pk_[0].shape)!=(nb_,dh): pk_=None
            _PREC['prior']=(pk_[0],pk_[1],nb_ if ib_ is None else ib_) if pk_ is not None else None
            g_=torch.cumsum(SEL(la.float().transpose(1,2).reshape(B*mh,T)),-1) if la is not None else torch.zeros(nb_,T,device=k.device)
            if Sin is None and pk_ is not None and not fac_ and Fin is None: Sin=torch.zeros(nb_,dh,dh,device=k.device)
            if Fin is not None and Fin[0].shape[0]!=nb_: Fin=None
            if Slt is not None and (Slt.U.shape[0]!=nb_ or ib_ is not None): raise RuntimeError('slot memory: row layout mismatch')
            if Slt is not None:                                          #   S0 k = sum_s exp(Gt - Gs) U_s (K_s . k): never rescaled, never concatenated
                c_=torch.bmm(torch.bmm(kn_,Slt.K.transpose(1,2))*torch.exp(Slt.Gt[:,None]-Slt.Gs)[:,None,:],Slt.U)*torch.exp(g_)[...,None]
                c_btc=torch.zeros(B,T,C,device=k.device,dtype=torch.float32)
                c_btc[:nrw_]=c_.reshape(nrw_,mh,T,dh).transpose(1,2).reshape(nrw_,T,C)
                out=_delta_core(k,v.float()-c_btc,bt,la,mh,beta_p,B,T,'torch',None,None)+c_btc
            elif Fin is not None or (Sin is not None and tuple(Sin.shape)==(nb_,dh,dh)):   #   S_T = gamma_T S0 + sum_s gamma_T/gamma_s beta_s r_s kw_s^T is handed back
                if Fin is not None: c_=torch.bmm(torch.bmm(kn_,Fin[1].float().transpose(1,2)),Fin[0].float())*torch.exp(g_)[...,None]   # S0 k = U^T (K k)
                else: c_=torch.bmm(kn_,Sin.transpose(1,2))*torch.exp(g_)[...,None]                       # (nb,T,dh)
                c_btc=torch.zeros(B,T,C,device=k.device,dtype=torch.float32)
                cr_=c_.reshape(nrw_,mh,T,dh).transpose(1,2).reshape(nrw_,T,C)
                if rows_ is None: c_btc[:nrw_]=cr_
                else: c_btc[rows_.to(k.device)]=cr_
                try: out=_delta_core(k,v.float()-c_btc,bt,la,mh,beta_p,B,T,'torch',None,None)+c_btc
                finally: pass
            else: Sin=None; out=_delta_core(k,v,bt,la,mh,beta_p,B,T,'torch',None,None)
            _PREC['prior']=None
            if _RDIAG['rec'] and la is not None: _RDIAG['v'].append(float(torch.exp(la.float().sum(1)).mean()))
            _gr_=bool(carry.get('grad'))                                                       # RECONSTRUCTION TRAINING: differentiable end state
            _dt_=(lambda t_: t_) if _gr_ else (lambda t_: t_.detach())
            with (contextlib.nullcontext() if _gr_ else torch.no_grad()):
                r_=(SEL(sp(v.float()))-SEL(sp(_dt_(out))))                                              # r = v - out (carried rows)
                be_=(SEL(bt.float().transpose(1,2).reshape(B*mh,T)) if bt is not None else torch.sigmoid(beta_p.float()).reshape(1,1).expand(nb_,T))
                if _PREC['p'] and pk_ is not None: _PREC['prior']=(pk_[0],pk_[1],nb_)
                kw_=_kw_par(_dt_(kn_)) if _PREC['p'] else _dt_(kn_)
                _PREC['prior']=None
                if _PREC['p'] and _PREC['C']>0:
                    k2s_=(kn_.detach()**2).sum(1); carry['Pout']=(k2s_ if pk_ is None else pk_[0]+k2s_, T if pk_ is None else pk_[1]+T)
                gd_=_dt_(g_); w_=be_*torch.exp(gd_[:,-1:]-gd_)
                if Slt is not None:                                                                  # v9.8 SLOT WRITE in place: slots i..i+T-1
                    i_=int(Slt.i); T_=r_.shape[1]
                    Slt.U[:,i_:i_+T_]=r_*be_[...,None]; Slt.K[:,i_:i_+T_]=kw_; Slt.Gs[:,i_:i_+T_]=Slt.Gt[:,None]+gd_; Slt.Gt.add_(gd_[:,-1])
                    carry['out']=Slt
                elif fac_ or Fin is not None:                                                          # FACTORED end state: the write terms unsummed
                    U_=r_*w_[...,None]; K_=kw_
                    if Fin is not None: U_=torch.cat([Fin[0].float()*torch.exp(gd_[:,-1])[:,None,None],U_],1); K_=torch.cat([Fin[1].float(),K_],1)
                    carry['out']=(U_,K_)
                else:
                    ST=torch.bmm((r_*w_[...,None]).transpose(1,2),kw_)                                  # (value, key)
                    if Sin is not None: ST=ST+torch.exp(gd_[:,-1])[:,None,None]*Sin
                    carry['out']=ST
            return out
        if T==1 and q is None and s0 is None and _DC1['on']:               # v9.8 cg8 SINGLE-BYTE PATH: read before write from an empty memory = 0 exactly
            _DC1['n']+=1; return torch.zeros(B,1,C,device=k.device,dtype=torch.float32)   #   (the solve gave v - 1*v = 0 in fp32; generation calls the inner solve with T = 1)
        k=F.normalize(sp(k),dim=-1); v=sp(v); kw=_kw_par(k) if _PREC['p'] else k      # write keys (preconditioned delta rule)
        mb=bt is not None; md=la is not None
        W_=_WIN['W']
        if W_ and q is None and T>2*W_:                                    # WINDOWED WY: overlapping chunks of 2W, keep the second half
            BH=k.shape[0]; n_=-(-T//W_); Tp=n_*W_
            be_=(bt.float().transpose(1,2).reshape(BH,T) if mb else torch.sigmoid(beta_p.float()).expand(BH,T))
            la_=(la.float().transpose(1,2).reshape(BH,T) if md else torch.zeros(BH,T,device=k.device))
            pz=lambda z,dims: F.pad(z,dims)
            kp=pz(k,(0,0,W_,Tp-T)); vp=pz(v,(0,0,W_,Tp-T)); bp=pz(be_,(W_,Tp-T)); lp=pz(la_,(W_,Tp-T))
            kc=kp.unfold(1,2*W_,W_).permute(0,1,3,2).reshape(BH*n_,2*W_,dh); vc=vp.unfold(1,2*W_,W_).permute(0,1,3,2).reshape(BH*n_,2*W_,dh)
            kwc=pz(kw,(0,0,W_,Tp-T)).unfold(1,2*W_,W_).permute(0,1,3,2).reshape(BH*n_,2*W_,dh) if _PREC['p'] else kc
            bc=bp.unfold(1,2*W_,W_).reshape(BH*n_,1,2*W_); lc=lp.unfold(1,2*W_,W_).reshape(BH*n_,2*W_)
            A=torch.bmm(kc,kwc.transpose(1,2))
            if md:
                lg=torch.cumsum(lc,-1); dd=lg[:,:,None]-lg[:,None,:]; A=A*torch.exp(torch.where(dd<=0,dd,torch.zeros_like(dd)))
            M=torch.tril(A,-1)*bc; I=torch.eye(2*W_,device=k.device).unsqueeze(0)
            r=torch.linalg.solve_triangular(I+M,vc,upper=False)
            out=(vc-r)[:,W_:].reshape(BH,n_*W_,dh)[:,:T]
            return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)
        if mb: beta=bt.float().transpose(1,2).reshape(B*mh,1,T)           # column s scaled by beta_s
        else: beta=torch.sigmoid(beta_p.float())
        SEL_=(_CHUNK.get('by') or {}).get(dh) or _CHUNK                   # solve form chosen for this head dim (start-up timing)
        ng_=(not torch.is_grad_enabled()) and bool(SEL_.get('Bng') or SEL_.get('Cng',0))   # no-grad passes have their own choice
        Bm_=SEL_.get('Bng') if ng_ else SEL_.get('B')
        if Bm_ and q is None and s0 is None:                               # BLOCKED full-WY solve (tensor-core GEMMs)
            be_=(bt.float().transpose(1,2).reshape(B*mh,T) if mb else torch.sigmoid(beta_p.float()).reshape(1,1).expand(B*mh,T))
            la_=la.float().transpose(1,2).reshape(B*mh,T) if md else None
            _BLK['prec']=Bm_[1]; out=delta_blocked(k,v,be_,la_,b=Bm_[0],kw=kw if _PREC['p'] else None)
            return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)
        Cc_=SEL_.get('Cng',0) if ng_ else SEL_.get('C',0)
        if Cc_>0 and q is None and T>Cc_:                                  # CHUNKED solve (no T x T tensors)
            la_=la.float().transpose(1,2).reshape(B*mh,T) if md else None
            S0c=s0.float().reshape(B*mh,dh,dh) if s0 is not None else None
            out=_delta_chunked(k,v,beta,mb,la_,S0c,Cc_,kw=kw if _PREC['p'] else None)
            return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)
        A=torch.bmm(k,kw.transpose(1,2))                                   # read key (row) . write key (column)
        if md:                                                             # A_ts * gamma_t/gamma_s (s<t), gamma = cumulative decay
            lg=torch.cumsum(la.float().transpose(1,2).reshape(B*mh,T),-1)
            dd=lg[:,:,None]-lg[:,None,:]; D=torch.exp(torch.where(dd<=0,dd,torch.zeros_like(dd)))
            A=A*D
        M=torch.tril(A,-1)*beta
        I=torch.eye(T,device=k.device).unsqueeze(0)
        if s0 is not None and q is None:                                  # INITIAL STATE S0: readout of S0 through the decays,
            S0_=s0.float().reshape(B*mh,dh,dh)                             #   c_t = gamma_t S0 k_t, enters the same triangular system
            c_=torch.bmm(k,S0_.transpose(1,2))
            if md: c_=c_*torch.exp(lg)[:,:,None]
            r=torch.linalg.solve_triangular(I+M,v-c_,upper=False)
        else: r=torch.linalg.solve_triangular(I+M,v,upper=False)
        if q is not None:
            q=F.normalize(sp(q),dim=-1); u=r*(beta.transpose(1,2) if mb else beta)
            QK=torch.bmm(q,kw.transpose(1,2))
            if md: QK=QK*D
            out=torch.bmm(torch.tril(QK),u)                                  # S_t q_t (includes the current write)
        else: out=v-r
        return out.reshape(B,mh,T,dh).transpose(1,2).reshape(B,T,C)

def _kernel_check(dh, mh, T, bmax, dev, decay=True):
    """Triton kernel vs torch solve on random inputs of one block's shape: relative error of outputs and of all input gradients."""
    g_=torch.Generator(device='cpu').manual_seed(0); B=2; C=dh*mh
    mk=lambda *s_: torch.randn(*s_,generator=g_).to(dev)
    base=[mk(B,T,C),mk(B,T,C),bmax*torch.sigmoid(mk(B,T,mh)),-F.softplus(mk(B,T,mh)*2-4) if decay else None]
    w=mk(B,T,C); res={}
    for kern in ('torch','fla'):
        xs=[z.clone().requires_grad_(True) if z is not None else None for z in base]
        o=_delta_core(xs[0],xs[1],xs[2],xs[3],mh,None,B,T,kern); (o*w).sum().backward()
        res[kern]=[o.detach()]+[z.grad for z in xs if z is not None]
    rel=lambda a,b: float((a-b).norm()/(b.norm()+1e-12))
    return max(rel(a,b) for a,b in zip(res['fla'],res['torch']))

def setup_kernel(model, cfg):
    _PREC['p']=float(getattr(cfg,'mix_precond',0.0) or 0.0)                 # before the solve-form timing below (it times the real path)
    _PREC['C']=int(getattr(cfg,'mix_precond_chunk',0) or 0); _PREC['det']=bool(getattr(cfg,'mix_precond_detach',False))
    mc_=getattr(cfg,'mix_chunk',0) or 0
    _CHUNK['C']=0 if (mc_=='auto' or (isinstance(mc_,str) and mc_.startswith('b'))) else int(mc_); _CHUNK['lp']=torch.bfloat16 if (getattr(cfg,'mix_lowp',False) and torch.cuda.is_available()) else None
    _CHUNK['by']={}; _CHUNK['B']=None; _CHUNK['Bng']=None
    if isinstance(mc_,str) and mc_.startswith('b'):                    # forced blocked solve: 'b64' (TF32 GEMMs) / 'b64h' (bf16 GEMMs)
        _CHUNK['C']=0; _CHUNK['B']=(int(mc_[1:].rstrip('h')),'bf16' if mc_.endswith('h') else ('tf32' if torch.cuda.is_available() else 'fp32'))
    if mc_=='auto' and torch.cuda.is_available():                      # time the delta-rule solve forms on every distinct block shape, keep the fastest
        lp0_=_CHUNK['lp']; B=int(cfg.batch_size)+int(cfg.dream_batch); T=int(cfg.seq_len); dev=model.embed.weight.device
        Bn_=int(getattr(cfg,'bank_tgt_chunk_big',0) or 0) or int(cfg.bank_tgt_chunk)
        nm_=lambda f: 'full' if f[0]=='full' else (f"chunk {f[1]}" if f[0]=='chunk' else f"blocked {f[1]} {f[2]}")
        def setf(f,ng=False):
            k1,k2=('Cng','Bng') if ng else ('C','B')
            _CHUNK[k1]=f[1] if f[0]=='chunk' else 0; _CHUNK[k2]=(f[1],f[2]) if f[0]=='blk' else None
        cands=[('full',)]+[('chunk',c) for c in (64,128,256,512) if c<T]+[('blk',b,p) for b in (64,128,256) for p in ('tf32','bf16') if b<=T]
        shapes=sorted({(getattr(b_,'mh',1),b_.d_state) for b_ in model.blocks if not getattr(b_,'loc',False) and not getattr(b_,'mq',False)}) or [(getattr(model.blocks[0],'mh',1),model.blocks[0].d_state)]
        by_={}
        for mh,C in shapes:
            dh=C//mh; sel={'C':0,'B':None,'Cng':0,'Bng':None}
            try:
                g_=torch.Generator(device='cpu').manual_seed(0); tm={}; err_={}; outs_={}
                mk=lambda *s_: torch.randn(*s_,generator=g_).to(dev)
                kk,vv,ww=mk(B,T,C),mk(B,T,C),mk(B,T,C); bb=1.2*torch.sigmoid(mk(B,T,mh)-1.0); ll=-F.softplus(mk(B,T,mh)*1.5-3)
                _CHUNK['by']={}
                for f in cands:
                    try:
                        setf(f); _CHUNK['Cng']=0; _CHUNK['Bng']=None; xs=[z.clone().requires_grad_(True) for z in (kk,vv,bb,ll)]
                        for it in range(3):
                            if it==1: torch.cuda.synchronize(); t0=time.perf_counter()
                            o_=_delta_core(xs[0],xs[1],xs[2],xs[3],mh,None,B,T,'torch'); (o_*ww).sum().backward()
                        torch.cuda.synchronize(); tm[f]=(time.perf_counter()-t0)/2*1e3; outs_[f]=o_.detach()
                    except Exception as ex: _fail('layers:492'); print(f"  [delta solve] {nm_(f)} failed at {mh}x{dh} ({repr(ex)[:100]})"); torch.cuda.empty_cache()
                ref_=outs_.get(('full',))
                err_={f:(0.0 if f==('full',) or ref_ is None else float((outs_[f]-ref_).norm()/(ref_.norm()+1e-30))) for f in outs_}
                ok_=[f for f in tm if err_[f]<=5e-3]                           # every form must match the fp32 full solve
                best=min(ok_,key=tm.get); setf(best); sel['C'],sel['B']=_CHUNK['C'],_CHUNK['B']
                del kk,vv,ww,xs,outs_; torch.cuda.empty_cache()
                print(f"  [delta solve] forward+backward per block ({B}x{T}, {mh} heads x {dh}"+(", chunks with bf16 state products" if lp0_ is not None else "")+"): "
                      +" | ".join(f"{nm_(f)} {t:.0f} ms"+(f" (err {err_[f]:.0e})" if f!=('full',) else "") for f,t in tm.items())+f" -> {nm_(best)}")
                try:                                                             # NO-GRAD passes (bank targets): their own form, timed at their batch
                    kk,vv=mk(Bn_,T,C),mk(Bn_,T,C); bb=1.2*torch.sigmoid(mk(Bn_,T,mh)-1.0); ll=-F.softplus(mk(Bn_,T,mh)*1.5-3); tng={}
                    with torch.no_grad():
                        for f in [f for f in ok_ if f[0]!='full']:
                            try:
                                setf(f)
                                for it in range(3):
                                    if it==1: torch.cuda.synchronize(); t0=time.perf_counter()
                                    _delta_core(kk,vv,bb,ll,mh,None,Bn_,T,'torch')
                                torch.cuda.synchronize(); tng[f]=(time.perf_counter()-t0)/2*1e3
                            except Exception: torch.cuda.empty_cache()
                    del kk,vv; torch.cuda.empty_cache()
                    if tng:
                        bng=min(tng,key=tng.get); sel['Cng']=bng[1] if bng[0]=='chunk' else 0; sel['Bng']=(bng[1],bng[2]) if bng[0]=='blk' else None
                        print(f"  [delta solve] no-grad forward per block ({Bn_}x{T}, {mh} heads x {dh}, bank targets): "+" | ".join(f"{nm_(f)} {t:.0f} ms" for f,t in tng.items())+f" -> {nm_(bng)}")
                except Exception as ex: _fail('layers:515'); print(f"  [delta solve] no-grad timing failed ({repr(ex)[:120]})"); torch.cuda.empty_cache()
            except Exception as ex: _fail('layers:516'); print(f"  [delta solve] timing failed at {mh}x{dh} ({repr(ex)[:120]}): full solve"); torch.cuda.empty_cache()
            by_[dh]=sel
        _CHUNK['by']=by_
        S0_=_CHUNK['by'].get(model.blocks[0].d_state//getattr(model.blocks[0],'mh',1)) or next(iter(_CHUNK['by'].values()),{'C':0,'B':None,'Cng':0,'Bng':None})
        _CHUNK['C'],_CHUNK['B'],_CHUNK['Cng'],_CHUNK['Bng']=S0_['C'],S0_['B'],S0_['Cng'],S0_['Bng']    # default for any other shape
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    _PREC['p']=float(getattr(cfg,'mix_precond',0.0) or 0.0)
    _WIN['W']=int(getattr(cfg,'mix_window',0) or 0); _SAMP['W']=int(getattr(cfg,'samp_window',0) or 0); _SAMP['bf16']=bool(getattr(cfg,'samp_bf16',False)); _SAMP['mm']=bool(getattr(cfg,'samp_lowp_mm',False))
    _SAMP['graph']=bool(getattr(cfg,'samp_graph',False)); _SAMP['gwarm']=int(getattr(cfg,'samp_graph_warm',64))
    _SAMP['rescore']=bool(getattr(cfg,'samp_rescore',True))
    """Choose the delta-rule kernel per block; the Triton kernel only after it matched the torch solve (values + gradients)."""
    for b in model.blocks: b._kern='torch'
    return
    fn=_fla_get()
    if fn is None:
        print(f"  [kernel] flash-linear-attention not available ({_FLA['err']}); torch solve everywhere. Install: pip install flash-linear-attention"); return
    dev=model.embed.weight.device; msg=[]
    for li,b in enumerate(model.blocks):
        mh=getattr(b,'mh',1); dh=b.d_state//mh
        if getattr(b,'loc',False): msg.append(f"b{li} local filter (no solve)"); continue
        if getattr(b,'mq',False): msg.append(f"b{li} torch (separate query)"); continue
        if dh>256: msg.append(f"b{li} torch (head dim {dh} > 256)"); continue
        key=(dh,mh)
        if key not in _FLA['checked']:
            err=None
            dts_=(torch.float32,torch.bfloat16)
            for dt in (dts_ if dev.type=='cuda' else (torch.float32,)):
                _FLA['dtype']=dt
                try: e_=max(_kernel_check(dh,mh,T_,getattr(b,'bmax',1.0),dev,getattr(b,'md',False)) for T_ in sorted({cfg.seq_len,min(cfg.dream_len,cfg.seq_len)}))
                except Exception as ex: err=repr(ex)[:200]; continue
                tol=2e-2 if dt==torch.float32 else 5e-2
                if e_<tol: _FLA['checked'][key]=(True,dt,e_); break
                err=f"rel. error {e_:.1e} ({str(dt)[6:]})"
            else: _FLA['checked'][key]=(False,None,err)
        ok,dt,e_=_FLA['checked'][key]
        if ok: _FLA['dtype']=dt; b._kern='fla'; msg.append(f"b{li} Triton ({str(dt)[6:]}, max rel. error vs torch {e_:.1e})")
        else: msg.append(f"b{li} torch (kernel check failed: {e_})")
    print("  [kernel] delta-rule solve per block: "+" | ".join(msg))
    if dev.type=='cuda' and any(b._kern=='fla' for b in model.blocks):      # one-off micro-benchmark on the real shape
        seen_=set()
        for b in [b_ for b_ in model.blocks if b_._kern=='fla']:
            mh=b.mh; C=b.d_state; B,T=int(cfg.batch_size)+int(cfg.dream_batch),cfg.seq_len
            if (mh,C) in seen_: continue
            seen_.add((mh,C))
            z=[torch.randn(B,T,C,device=dev,requires_grad=True) for _ in range(2)]; bt=torch.rand(B,T,mh,device=dev)
            la=-torch.rand(B,T,mh,device=dev)*0.01
            tm={}
            for kern in ('torch','fla'):
                for it in range(8):
                    if it==3: torch.cuda.synchronize(); t0=time.perf_counter()
                    _delta_core(z[0],z[1],bt,la,mh,None,B,T,kern).float().sum().backward()
                torch.cuda.synchronize(); tm[kern]=(time.perf_counter()-t0)/5*1000
            del z; torch.cuda.empty_cache()
            print(f"  [kernel] forward+backward of one mixer solve ({B}x{T}, {mh} heads x {C//mh}): torch {tm['torch']:.1f} ms, Triton {tm['fla']:.1f} ms")

class DeltaBlock(nn.Module):
    def __init__(s, d_model, d_state, mlp_hidden=None):
        super().__init__(); s.d_model,s.d_state=d_model,d_state; s.mlp_hidden=mlp_hidden or d_model*2
        s.Wk=nn.Linear(d_model,d_state,bias=False); s.Wv=nn.Linear(d_model,d_state,bias=False)
        s.Wout=nn.Linear(d_state,d_model,bias=False); s.gate=nn.Linear(d_model,d_state)
        s.beta=nn.Parameter(torch.tensor(0.5)); s.ln=nn.LayerNorm(d_model); s.norm=nn.LayerNorm(d_model)
        s.mlp_in=nn.Linear(d_model,s.mlp_hidden); s.mlp_out=nn.Linear(s.mlp_hidden,d_model)
        s.register_buffer('unit_usage',torch.zeros(d_state))
        s.register_buffer('mlp_usage',torch.zeros(s.mlp_hidden))
        # home usage (frozen after each domain) - polarization feature = |current - home|
        s.register_buffer('unit_home',torch.zeros(d_state))
        s.register_buffer('mlp_home',torch.zeros(s.mlp_hidden))
        # HARD importance-freeze masks (the +0.32 floor) - meta-gate modulates ON TOP
        s.register_buffer('frozen_units',torch.zeros(d_state,dtype=torch.bool))
        s.register_buffer('mlp_frozen',torch.zeros(s.mlp_hidden,dtype=torch.bool))
        # DISTINCTNESS ROUTING: each unit has a learned distinctness-AFFINITY (which
        # part of the manifold it serves - low=prose/shared region, high=distinct region).
        # Content routes to units whose affinity matches its distinctness.
        s.unit_affinity=nn.Parameter(torch.rand(d_state)*0.1)      # learned per-unit region
        s.mlp_affinity=nn.Parameter(torch.rand(s.mlp_hidden)*0.1)
        s.cur_distinct=None   # current batch's distinctness level (set in forward)
        s.route_in_forward=True   # False when cfg.route_writes (set in run)
        # HEBBIAN AFFINITY: each unit's affinity = distinctness of the content that drives it most.
        # -1 = unassigned (accepts writes from any content until strongly driven).
        s.register_buffer('unit_aff', -torch.ones(d_state))
        s.register_buffer('mlp_aff', -torch.ones(s.mlp_hidden))
        s.calib=None              # dict of accumulators during affinity calibration
        s.register_buffer('distinct_locked', torch.zeros(1))   # per-block copy of the Model's lock flag
        # random base write thresholds + cumulative importance (persist across domains)
        s.register_buffer('unit_theta0', sample_theta(d_state))
        s.register_buffer('mlp_theta0', sample_theta(s.mlp_hidden))
        s.register_buffer('unit_imp', torch.zeros(d_state))
        s.register_buffer('mlp_imp', torch.zeros(s.mlp_hidden))
        s.register_buffer('ssm_drive', torch.zeros(d_state))    # last batch's per-unit drive
        s.register_buffer('mlp_drive', torch.zeros(s.mlp_hidden))
        s.register_buffer('ssm_drive_ema', torch.zeros(d_state))  # each unit's typical drive
        # CHANNEL NEURONS: this block's d_model residual channels, each with its own random
        # threshold, importance and drive (used when channel_thresholds=True)
        s.register_buffer('ch_theta0', sample_theta(d_model))
        s.register_buffer('ch_imp', torch.zeros(d_model))
        s.register_buffer('ch_usage', torch.zeros(d_model))
        s.register_buffer('ch_drive', torch.zeros(d_model))
        s.register_buffer('ch_drive_ema', torch.zeros(d_model))
        s.register_buffer('mlp_drive_ema', torch.zeros(s.mlp_hidden))
        # SIGNAL 1: long-memory drive references; SIGNAL 2: loss sensitivity on earlier domains
        for nm,n in (('ssm',d_state),('mlp',s.mlp_hidden),('ch',d_model)):
            s.register_buffer(nm+'_drive_slow', torch.zeros(n)); s.register_buffer(nm+'_drive_peak', torch.zeros(n))
        s.register_buffer('unit_sens', torch.zeros(d_state)); s.register_buffer('mlp_sens', torch.zeros(s.mlp_hidden))
        s.register_buffer('ch_sens', torch.zeros(d_model))
    def setup_local(s, cfg):
        """LOCAL FILTER BLOCK: causal depthwise filter over the previous mix_local_width values instead of the delta-rule memory
        (starts as their average)."""
        K=int(cfg.mix_local_width); s.loc=True
        s.conv_loc=nn.Parameter(torch.full((s.d_state,1,K),1.0/K,device=s.Wv.weight.device))
    def setup_lattn(s, cfg):
        """C2 SLIDING-WINDOW ATTENTION mixer: softmax over the previous lattn_window positions of scale_h x cos(k_t, k_j) + bias_h[t-j],
        plus a null slot (reads nothing when nothing matches); keys / values from the same projections and short convolution."""
        s.lattn=True; s.la_W=int(cfg.lattn_window); mh=getattr(s,'mh',1); dev=s.Wk.weight.device
        s.la_scale=nn.Parameter(torch.full((mh,),math.log(8.0),device=dev)); s.la_rel=nn.Parameter(torch.zeros(mh,s.la_W,device=dev))
    def setup_nov(s, cfg):
        """NOVELTY RESERVE: nov_pool closed MLP slices (zero-init output: opening one does not change the function)."""
        k=max(1,int(round(2*cfg.d_model*cfg.init_mlp_mult*cfg.nov_frac))); dev=s.Wv.weight.device
        s.nov_in=nn.ModuleList([nn.Linear(s.d_model,k).to(dev) for _ in range(int(cfg.nov_pool))])
        s.nov_out=nn.ModuleList([nn.Linear(k,s.d_model,bias=False).to(dev) for _ in range(int(cfg.nov_pool))])
        for o in s.nov_out: o.weight.data.zero_()
        s.nov_g=[0.0]*int(cfg.nov_pool)                              # python-side gates (no GPU sync; copied into the teachers)
        s.nov_tok=bool(cfg.nov_token)
    def nov_term(s, hn, rec=False):
        out=0.0; tok=getattr(s,'nov_tok',False); M=_NOV['m'] if tok else None
        for j,g in enumerate(getattr(s,'nov_g',())):
            if g<=0: continue
            if tok:                                                    # per-token gate (the sampler leaves token-gated slices out)
                if rec or M is None or j>=len(M) or M[j] is None or M[j].shape[:2]!=hn.shape[:2]: continue
                out=out+g*M[j].to(hn.dtype)*s.nov_out[j](F.gelu(s.nov_in[j](hn)))
            else: out=out+g*s.nov_out[j](F.gelu(s.nov_in[j](hn)))
        return out
    def setup_mixer(s, cfg):
        """Optional mixer upgrades; each starts out computing exactly what the plain mixer computes."""
        s.mh=max(1,int(cfg.mix_heads)); s.mk=int(cfg.mix_conv); s.mq=bool(cfg.mix_query); s.mb=bool(cfg.mix_beta_tok)
        hd_=int(getattr(cfg,'mix_head_dim',0) or 0)
        if hd_>0:                                                              # HEAD DIM CAP: per-block head count
            ds_=[d_ for d_ in range(1,min(hd_,s.d_state)+1) if s.d_state%d_==0]
            for q_ in (64,32,16,8,1):
                c_=[d_ for d_ in ds_ if d_%q_==0]
                if c_: s.mh=s.d_state//max(c_); break
        assert s.d_state%s.mh==0, 'd_state must divide into mix_heads'
        dev=s.Wk.weight.device
        if s.mq: s.Wq=nn.Linear(s.d_model,s.d_state,bias=False).to(dev); s.Wq.weight.data.copy_(s.Wk.weight.data)
        if s.mb:
            s.Wb=nn.Linear(s.d_model,s.mh).to(dev); s.Wb.weight.data.zero_()
            s.bmax=float(getattr(cfg,'mix_beta_max',1.0)); p0=float(torch.sigmoid(s.beta.data))/s.bmax      # same starting beta
            s.Wb.bias.data.fill_(math.log(p0/(1-p0)))
        s.dp2=bool(int(getattr(cfg,'dp_steps',1))>=2 and s.mb and not s.mq)
        if s.dp2:                                                              # DELTAPRODUCT: second write per token (starts ~off)
            s.Wk2=nn.Linear(s.d_model,s.d_state,bias=False).to(dev); s.Wv2=nn.Linear(s.d_model,s.d_state,bias=False).to(dev)
            s.Wb2=nn.Linear(s.d_model,s.mh).to(dev); s.Wb2.weight.data.zero_(); p2_=0.02/s.bmax; s.Wb2.bias.data.fill_(math.log(p2_/(1-p2_)))
        s.md=bool(getattr(cfg,'mix_decay',False))
        if s.md:
            s.Wa=nn.Linear(s.d_model,s.mh).to(dev); s.Wa.weight.data.zero_(); s.Wa.bias.data.fill_(float(cfg.mix_decay_bias))
        if s.mk>0:
            def idk():
                w=torch.zeros(s.d_state,1,s.mk,device=dev); w[:,0,-1]=1.0; return nn.Parameter(w)
            s.conv_k=idk(); s.conv_v=idk()
            if s.mq: s.conv_q=idk()
    def _conv(s, z, w):
        """Causal depthwise convolution along time: z (B,T,C), w (C,1,K)."""
        K=w.shape[-1]; zt=F.pad(z.transpose(1,2),(K-1,0))
        return F.conv1d(zt,w.to(zt.dtype),groups=w.shape[0]).transpose(1,2)
    def mix(s, h, x, B, T):
        """Delta-rule sequence mixer: memory S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T. Plain: output S_{t-1} k_t (the key
        reads memory before writing). With a query: output S_t q_t. WY form: one triangular solve per sequence and head
        (or the Triton chunked kernel, mix_kernel='fla'). Runs in fp32 even under autocast."""
        if getattr(s,'loc',False):                                              # LOCAL FILTER: previous values only (read before write)
            v=_wl(s,'Wv',h); K=s.conv_loc.shape[-1]
            sg_=_SEG['s']
            if sg_ is not None and tuple(sg_.shape)==(B,T):                    # v9 RUN MASKING: values before the segment start excluded
                w_=s.conv_loc.to(v.dtype)[:,0,:]; t_=torch.arange(T,device=v.device)[None]; o_=0.0
                for i in range(1,K+1): o_=o_+F.pad(v,(0,0,i,0))[:,:T]*((t_-i)>=sg_).to(v.dtype)[...,None]*w_[:,K-i]
                return o_.float()
            zt=F.pad(v.transpose(1,2),(K,0))
            return F.conv1d(zt,s.conv_loc.to(zt.dtype),groups=s.d_state)[...,:T].transpose(1,2).float()
        mh=getattr(s,'mh',1); mk=getattr(s,'mk',0); mq=getattr(s,'mq',False); mb=getattr(s,'mb',False)
        if getattr(s,'lattn',False):                                            # C2: sliding-window attention
            k=_wl(s,'Wk',h); v=_wl(s,'Wv',h)
            if mk>0: k=s._conv(k,s.conv_k); v=s._conv(v,s.conv_v)
            return _local_attn(k,v,mh,B,T,s.la_scale,s.la_rel,s.la_W)
        k=_wl(s,'Wk',h); v=_wl(s,'Wv',h); q=_wl(s,'Wq',h) if mq else None
        if mk>0:
            k=s._conv(k,s.conv_k); v=s._conv(v,s.conv_v)
            if mq: q=s._conv(q,s.conv_q)
        bt=getattr(s,'bmax',1.0)*torch.sigmoid(_wl(s,'Wb',h)) if mb else None  # (B,T,H)
        if mb and s.training and getattr(s,'bmax',1.0)>1.0:                     # usage of the reflection range (no GPU sync)
            bd=bt.detach()[:getattr(s,'_nstat',None)].float(); st=getattr(s,'_bstat',None)
            s._bstat=(bd.gt(1.0).float().mean(), bd.mean(), 1) if st is None else (st[0]+bd.gt(1.0).float().mean(), st[1]+bd.mean(), st[2]+1)
        md=getattr(s,'md',False); la=-F.softplus(_wl(s,'Wa',h)) if md else None  # log decay (B,T,H), <= 0
        if la is not None and _RESET['m'] is not None and tuple(_RESET['m'].shape)==tuple(la.shape[:2]):
            la=la.masked_fill(_RESET['m'][...,None],-30.0)                     # v7 EXACT RUNS: memory wiped at the start of a run (exp(-30) ~ 1e-13)
        s0_=None; S0p=getattr(s,'S0',None); ix_=getattr(s,'_s0_idx',None); tr_=getattr(s,'_s0_train',None)
        if S0p is not None and ix_ is not None and q is None:                    # PER-DOMAIN INITIAL STATE (only the trained domain's learns)
            g_=lambda i: S0p[i] if i==tr_ else S0p[i].detach()
            if isinstance(ix_,int): s0_=g_(ix_).unsqueeze(0).expand(B,-1,-1,-1)
            elif isinstance(ix_,(list,tuple)) and len(ix_)==B:
                z_=torch.zeros_like(S0p[0]); s0_=torch.stack([(g_(i) if i is not None and i>=0 else z_) for i in ix_])
        if getattr(s,'dp2',False) and q is None:                                 # DELTAPRODUCT: two writes per token as interleaved sub-steps
            k2=s.Wk2(h); v2=s.Wv2(h); bt2=s.bmax*torch.sigmoid(s.Wb2(h))
            K_=torch.stack([k,k2.to(k.dtype)],2).reshape(B,2*T,k.shape[-1]); V_=torch.stack([v,v2.to(v.dtype)],2).reshape(B,2*T,v.shape[-1])
            BT_=torch.stack([bt,bt2.to(bt.dtype)],2).reshape(B,2*T,bt.shape[-1])
            LA_=torch.stack([la,torch.zeros_like(la)],2).reshape(B,2*T,la.shape[-1]) if la is not None else None
            out=_delta_core(K_, V_, BT_, LA_, mh, None, B, 2*T, getattr(s,'_kern','torch'), None, s0_).reshape(B,T,2,-1)[:,:,0]
        else:
            if _ORACLE['on'] and q is None:                                     # ORACLE MEMORY: pooled least-squares statistics per head
                with torch.no_grad():
                    kn_o=F.normalize(k.float().reshape(B,T,mh,-1),dim=-1); vv_o=v.float().reshape(B,T,mh,-1)
                    be_o=(bt.float() if bt is not None else torch.sigmoid(s.beta.float()).expand(B,T,mh))[...,None]
                    A_o=_ORACLE['st'].setdefault(id(s),[0.0,0.0])
                    A_o[0]=A_o[0]+torch.einsum('bthi,bthj->hij',kn_o*be_o,kn_o); A_o[1]=A_o[1]+torch.einsum('bthi,bthj->hij',vv_o*be_o,kn_o)
            if _CHN['on'] and q is None: _CHN['k'][id(s)]=k
            cr_={'S':_CARRY['st'].get(id(s)),'P':_CARRY['st'].get((id(s),'P')),'nr':_CARRY['nr'],'rows':_CARRY.get('rows'),'grad':bool(_CARRY.get('grad')) and _RC_GRAD['on'],'fac':bool(_CARRY.get('fac'))} if (_CARRY['on'] and q is None and s0_ is None) else None
            out=_delta_core(k, v, bt, la, mh, None if mb else s.beta, B, T, getattr(s,'_kern','torch'), q, s0_, carry=cr_)   # CONTINUOUS CONTEXT
            if cr_ is not None and 'out' in cr_:
                _CARRY['new'][id(s)]=cr_['out']
                if 'Pout' in cr_: _CARRY['new'][(id(s),'P')]=cr_['Pout']
            if _PMEM['on'] and q is None and id(s) in _PMEM['S']:                # PERSISTENT READ: a domain memory read at every position, no decay
                kn_p=F.normalize(k.float().reshape(B,T,mh,-1),dim=-1)
                out=out+torch.einsum('hvk,bthk->bthv',_PMEM['S'][id(s)],kn_p).reshape(B,T,-1).to(out.dtype)
        if s.training and getattr(s,'_surp_on',False):                          # SURPRISE: memory prediction-error ratio (no GPU sync)
            with torch.no_grad():
                n_=getattr(s,'_nstat',None); vf=v[:n_].float(); s._surp_now=((vf-out[:n_])**2).mean()/(vf.pow(2).mean()+1e-12)
        return out
    def write_route(s, aff):
        """Write-routing gate: a unit accepts writes from content whose distinctness matches its
        affinity. Unassigned units accept everything. Uniform before the lock."""
        if s.cur_distinct is None or s.distinct_locked.item()==0: return torch.ones_like(aff)
        g=torch.exp(-((s.cur_distinct-aff)**2)/Config.route_width)
        return torch.where(aff<0, torch.ones_like(g), g)
    @torch.no_grad()
    def _hebb(s, drive, aff):
        d=float(s.cur_distinct); w=drive/(drive.max()+1e-8)
        un=aff<0
        assign=un&(w>0.5); aff[assign]=d                       # strongly driven unassigned unit adopts this content's distinctness
        up=~un; aff[up]+=Config.hebb_rate*w[up]*(d-aff[up])   # assigned units drift toward content that drives them
    def route_gate(s, affinity):
        # soft routing gate: unit active for content whose distinctness ~ unit affinity.
        # BEFORE lock: route uniformly (gate=1) - accumulate reference without unstable
        # routing. AFTER lock: route by distinctness (stable reference).
        if s.cur_distinct is None or s.distinct_locked.item()==0: return torch.ones_like(affinity)
        return torch.exp(-((s.cur_distinct-torch.sigmoid(affinity))**2)/0.1)
    def forward(s,x):
        ns_=getattr(s,'_nstat',None)                                   # MERGED FORWARD: statistics from the first ns_ (real) rows only
        _lean=s.__dict__.get('_lean',False)                            # LEAN: drive statistics unused (gates open, no growth)
        if s.training and not _lean:
            cd=x.detach()[:ns_].abs().mean((0,1))
            s.ch_usage+=cd; s.ch_drive=cd
            _track(s,'ch',cd)
        h=s.ln(x); B,T,_=h.shape
        pred=s.mix(h,x,B,T)
        # DISTINCTNESS ROUTING: gate SSM units by affinity-match to content distinctness.
        # Distinct content activates its own region's units; prose the shared region.
        if s.route_in_forward: pred=pred*s.route_gate(s.unit_affinity)   # (d_state,) soft routing (read-gating)
        if s.calib is not None:
            dv=pred.detach().abs().mean((0,1)); dd=float(s.cur_distinct)
            s.calib['s_w']+=dv; s.calib['s_wd']+=dv*dd
        if s.training and getattr(s,'_ddrive_on',False) and ns_ is not None and ns_<pred.shape[0]:   # RESIDUAL GATES: activity on dreams
            dd_=pred.detach()[ns_:].abs().float().mean((0,1)); o_=getattr(s,'_dd_ssm',None)
            s._dd_ssm=dd_ if (o_ is None or o_.shape!=dd_.shape) else 0.99*o_+0.01*dd_
        if s.training and not _lean:
            d_=pred.detach()[:ns_].abs().mean((0,1))                        # computed once (usage and drive are the same statistic)
            s.unit_usage+=d_; s.ssm_drive=d_
            _track(s,'ssm',s.ssm_drive)
        gpre=_wl(s,'gate',h if getattr(s,'gate_ln',False) else x)   # gate reads the raw residual (original) or its LN
        h2=pred*F.silu(gpre); x_in=x
        _dc_=getattr(s,'_dcontent',False) and ns_ is not None and 0<ns_<h2.shape[0] and torch.is_grad_enabled()
        if _dc_: h2=torch.cat([h2[:ns_],h2[ns_:].detach()])                   # CONTENT-ONLY DREAMS: dream rows train Wout only
        o1_=_wl(s,'Wout',h2)
        x_par=x
        gp_=getattr(s,'_gproj',None)                                 # DREAM ONLY THE SIMILARITIES: projected gradient on this block's increments
        if gp_ is not None: o1_=_GProj.apply(o1_,*gp_)
        x=x+o1_
        hn=s.norm(x_par if getattr(s,'parallel',False) else x)                 # C3: parallel block
        if s.training and getattr(s,'_store_uniq',False): s._uh=h.detach()[:ns_]; s._um=hn.detach()[:ns_]   # for the uniqueness detector
        m=F.gelu(_wl(s,'mlp_in',film_mod(s,hn,ns_) if _FILM.get('codes') is not None else hn))
        if _XDG['on']: m=xdg_apply(m,getattr(s,'_xdg_i',0))
        if getattr(s,'inh_A',None) is not None: m=inh_apply(s,m,hn,ns_)
        cgm_=None
        if s.__dict__.get('_cg_g') is not None: m,cgm_=cg_apply(s,m,ns_)
        if getattr(s,'_probe',None) is not None:     # BLOW-UP PROBE: largest magnitudes inside this block
            s._probe.append((float(x_in.abs().max()),float(gpre.abs().max()),float(pred.abs().max()),float(h2.abs().max()),float(m.abs().max())))
        if s.route_in_forward: m=m*s.route_gate(s.mlp_affinity)          # route MLP units too
        if s.calib is not None:
            dv=m.detach().abs().mean((0,1)); dd=float(s.cur_distinct)
            s.calib['m_w']+=dv; s.calib['m_wd']+=dv*dd
        if (s.training and getattr(s,'write_routed',False) and s.distinct_locked.item()==1
                and s.cur_distinct is not None):
            s._hebb(s.ssm_drive, s.unit_aff)
        if s.training and getattr(s,'_ddrive_on',False) and ns_ is not None and ns_<m.shape[0]:
            dm_=m.detach()[ns_:].abs().float().mean((0,1)); o_=getattr(s,'_dd_mlp',None)
            s._dd_mlp=dm_ if (o_ is None or o_.shape!=dm_.shape) else 0.99*o_+0.01*dm_
        if s.training and not _lean:
            d_=m.detach()[:ns_].abs().mean((0,1))
            s.mlp_usage+=d_; s.mlp_drive=d_
            _track(s,'mlp',s.mlp_drive)
            if getattr(s,'write_routed',False) and s.distinct_locked.item()==1 and s.cur_distinct is not None:
                s._hebb(s.mlp_drive, s.mlp_aff)
        if _dc_: m=torch.cat([m[:ns_],m[ns_:].detach()])                        # CONTENT-ONLY DREAMS: dream rows train mlp_out only
        o2_=_wl(s,'mlp_out',m)
        if _ADPT['on'] and id(s) in _ADPT['P']:                                 # LOW-RANK ADAPTER beside the MLP (oracle)
            A_a,B_a=_ADPT['P'][id(s)]; o2_=o2_+F.linear(F.linear(hn.float(),A_a),B_a).to(o2_.dtype)
        if cgm_ is not None and o2_.requires_grad: o2_.register_hook(cgm_)
        if gp_ is not None: o2_=_GProj.apply(o2_,*gp_)
        x=x+o2_
        if getattr(s,'pu_in',None) is not None and _XDG.get('tagon'): x=x+pu_term(s,hn,ns_)
        if getattr(s,'nov_g',None): x=x+s.nov_term(hn)
        if getattr(s,'_cap',False): s._cap_out=x                   # FEATURE ANCHORING: this block's output (student, with gradient)
        return x
    @torch.no_grad()
    def widen(s, chunk, dev, rows=None):
        """Open up: add `chunk` new SSM units (zero-init Wout cols -> no disruption).
        New units are UNFROZEN (available for new content); existing frozen units stay
        frozen (protected). Widening = capacity; freeze machinery = protection.
        rows: optional {'Wk','Wv','gate','gate_b'} for the new units (activated reserve units)."""
        if getattr(s,'mq',False) or getattr(s,'mk',0) or getattr(s,'mb',False) or getattr(s,'mh',1)>1 or getattr(s,'md',False):
            raise NotImplementedError('widening is not implemented for the mixer upgrades (use block_extra_mode="start")')
        def grow_in(lin, add, new=None, newb=None):   # (out=d_state, in): add small-random (or given) rows
            w=torch.cat([lin.weight.data, new if new is not None else torch.randn(add, lin.weight.shape[1], device=dev)*0.02],0)
            nl=nn.Linear(lin.weight.shape[1], w.shape[0], bias=lin.bias is not None).to(dev)
            nl.weight.data=w
            if lin.bias is not None: nl.bias.data=torch.cat([lin.bias.data, newb if newb is not None else torch.zeros(add,device=dev)])
            return nl
        def grow_out(lin, add):  # (out, in=d_state): add ZERO cols (no disruption)
            w=torch.cat([lin.weight.data, torch.zeros(lin.weight.shape[0], add, device=dev)],1)
            nl=nn.Linear(w.shape[1], w.shape[0], bias=lin.bias is not None).to(dev)
            nl.weight.data=w
            if lin.bias is not None: nl.bias.data=lin.bias.data
            return nl
        r=rows or {}
        s.Wk=grow_in(s.Wk,chunk,r.get('Wk')); s.Wv=grow_in(s.Wv,chunk,r.get('Wv')); s.gate=grow_in(s.gate,chunk,r.get('gate'),r.get('gate_b'))
        s.Wout=grow_out(s.Wout,chunk); s.d_state+=chunk
        # extend all per-unit buffers (new units: zero usage/home, UNFROZEN)
        s.unit_usage=torch.cat([s.unit_usage, torch.zeros(chunk,device=dev)])
        s.unit_home=torch.cat([s.unit_home, torch.zeros(chunk,device=dev)])
        s.frozen_units=torch.cat([s.frozen_units, torch.zeros(chunk,dtype=torch.bool,device=dev)])
        # new units get HIGH affinity (they open for DISTINCT content, which triggered widening)
        new_aff=nn.Parameter(torch.cat([s.unit_affinity.data, torch.ones(chunk,device=dev)*2.0]))
        s.unit_affinity=new_aff
        s.unit_aff=torch.cat([s.unit_aff, -torch.ones(chunk,device=dev)])
        s.unit_theta0=torch.cat([s.unit_theta0, sample_theta(chunk,dev)])
        s.unit_imp=torch.cat([s.unit_imp, torch.zeros(chunk,device=dev)])
        s.ssm_drive=torch.cat([s.ssm_drive, torch.zeros(chunk,device=dev)])
        s.ssm_drive_ema=torch.cat([s.ssm_drive_ema, torch.zeros(chunk,device=dev)])
        s.ssm_drive_slow=torch.cat([s.ssm_drive_slow, torch.zeros(chunk,device=dev)])
        s.ssm_drive_peak=torch.cat([s.ssm_drive_peak, torch.zeros(chunk,device=dev)])
        s.unit_sens=torch.cat([s.unit_sens, torch.zeros(chunk,device=dev)])
    @torch.no_grad()
    def widen_mlp(s, n, dev, w_in, b_in):
        """Activate n MLP units: given input rows/bias, ZERO output columns (no disruption), fresh buffers."""
        nl=nn.Linear(s.d_model, s.mlp_hidden+n).to(dev)
        nl.weight.data=torch.cat([s.mlp_in.weight.data, w_in],0); nl.bias.data=torch.cat([s.mlp_in.bias.data, b_in]); s.mlp_in=nl
        no=nn.Linear(s.mlp_hidden+n, s.d_model).to(dev)
        no.weight.data=torch.cat([s.mlp_out.weight.data, torch.zeros(s.d_model,n,device=dev)],1); no.bias.data=s.mlp_out.bias.data; s.mlp_out=no
        s.mlp_hidden+=n; z=lambda: torch.zeros(n,device=dev)
        for nm in ('mlp_usage','mlp_home','mlp_imp','mlp_drive','mlp_drive_ema','mlp_drive_slow','mlp_drive_peak','mlp_sens'):
            setattr(s,nm,torch.cat([getattr(s,nm),z()]))
        s.mlp_frozen=torch.cat([s.mlp_frozen, torch.zeros(n,dtype=torch.bool,device=dev)])
        s.mlp_aff=torch.cat([s.mlp_aff, -torch.ones(n,device=dev)])
        s.mlp_theta0=torch.cat([s.mlp_theta0, sample_theta(n,dev)])
        s.mlp_affinity=nn.Parameter(torch.cat([s.mlp_affinity.data, torch.ones(n,device=dev)*2.0]))

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_GProj': 'platypus.ngrams',
    '_NOV': 'platypus.train',
    '_RDIAG': 'platypus.losses',
    '_RESET': 'platypus.conduit',
    '_SEG': 'platypus.ngrams',
}
