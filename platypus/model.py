"""The Model."""
from platypus.common import _fail
import contextlib
import math
import torch
import torch.nn as nn
from platypus.common import (
    _nslots,
)
from platypus.config import (
    sample_theta,
)
from platypus.gating import (
    _track,
)
from platypus.protection import (
    _PJ, _STF,
)
from platypus.private import (
    _BUFDET, _HCAP, _XDG, priv, side_out,
)
from platypus.layers import (
    DeltaBlock, Head, _CARRY, _smem_seed,
)
from platypus.ngrams import (
    _COMPILED, _GScale, ngram_embed, route_scales,
)


class CaseEmbed(nn.Module):
    """CASE-FACTORED INPUT EMBEDDING (case_factor 1): a byte's embedding is its own vector PLUS, for letters, a vector shared by both cases
    of the letter and a case vector (lower / upper) - so 'c' and 'C' share what is learned about the letter. Bytes stay exact (the own
    vector is kept); the shared and case vectors start at zero (the model starts as the plain embedding) and use no random numbers."""
    def __init__(s, E):
        super().__init__(); s.weight = E.weight; d = E.weight.shape[1]
        s.letter = nn.Parameter(torch.zeros(27, d)); s.case = nn.Parameter(torch.zeros(3, d))
        f = torch.zeros(256, dtype=torch.long); c = torch.zeros(256, dtype=torch.long)
        for b in range(65, 91): f[b] = b - 64; c[b] = 2
        for b in range(97, 123): f[b] = b - 96; c[b] = 1
        s.register_buffer('fold', f); s.register_buffer('cas', c)
    @property
    def num_embeddings(s): return s.weight.shape[0]
    @property
    def embedding_dim(s): return s.weight.shape[1]
    def forward(s, x):
        return nn.functional.embedding(x, s.weight) + s.letter[s.fold[x]] + s.case[s.cas[x]]
class Model(nn.Module):
    def __init__(s,cfg):
        super().__init__(); s.cfg=cfg
        s.embed=nn.Embedding(cfg.vocab,cfg.d_model); s.pos=nn.Embedding(cfg.seq_len,cfg.d_model)
        if int(getattr(cfg,'case_factor',0)) and int(cfg.vocab)==256: s.embed=CaseEmbed(s.embed)   # case-factored input (off by default)
        s.pos.weight.data.zero_(); s.pos.weight.requires_grad_(False)
        ns0=int(round(cfg.d_state*cfg.init_state_mult)); nm0=int(round(2*cfg.d_model*cfg.init_mlp_mult))
        ex=lambda li: (cfg.block_extra[li] if (cfg.block_extra and li<len(cfg.block_extra) and cfg.block_extra_mode=='start') else 0.0)
        s.blocks=nn.ModuleList([DeltaBlock(cfg.d_model,int(round(ns0*(1+ex(li)))),int(round(nm0*(1+ex(li))))) for li in range(cfg.n_layers)])
        for li_,b in enumerate(s.blocks):
            b.gate_ln=cfg.gate_ln; b.setup_mixer(cfg)
            if li_ in tuple(cfg.mix_local_blocks or ()): b.setup_local(cfg)
            if li_ in tuple(cfg.lattn_blocks or ()): b.setup_lattn(cfg)
            b.parallel=bool(cfg.parallel_block); b._ddrive_on=cfg.gate_dream_relax>0
        sz=cfg.ngram_size if isinstance(cfg.ngram_size,(tuple,list)) else [cfg.ngram_size]*len(cfg.ngram_emb)
        s.ngram=nn.ModuleList([nn.Embedding(int(z),cfg.d_model) for z in sz]); s.ngram.ns=tuple(int(n) for n in cfg.ngram_emb)
        s.ngram.nh=max(1,int(cfg.ngram_hashes))
        for E in s.ngram: E.weight.data.zero_()
        s.__dict__['_ng_cur']=[torch.zeros(int(z)) for z in sz]; s.__dict__['_ng_old']=[torch.zeros(int(z)) for z in sz]
        if cfg.pu_k>0:                                                  # PRIVATE TRUNK UNITS per domain
            k_=int(cfg.pu_k); ND_=_nslots(cfg)
            for b in s.blocks:
                b.pu_in=nn.Parameter(torch.randn(ND_,cfg.d_model,k_)*(1.0/math.sqrt(cfg.d_model))); b.pu_b=nn.Parameter(torch.zeros(ND_,k_))
                b.pu_out=nn.Parameter(torch.zeros(ND_,k_,cfg.d_model)); b._pu_dream_grad=bool(cfg.pu_dream_grad)
                if 0<cfg.pu_k_base<k_:
                    mk_=torch.zeros(ND_,k_); mk_[:,:int(cfg.pu_k_base)]=1.0; b.register_buffer('pu_mask',mk_)
        k_=int(cfg.film_k); s.film_codes=nn.Parameter(torch.randn(_nslots(cfg),k_)*0.1)
        for b in s.blocks:
            b.film_g=nn.Parameter(torch.zeros(cfg.d_model,k_)); b.film_b=nn.Parameter(torch.zeros(cfg.d_model,k_))
        s.heads=nn.ModuleList([Head(cfg.d_model,cfg.vocab) for _ in range(_nslots(cfg))])
        if cfg.shared_head: s.heads=nn.ModuleList([s.heads[0]]*_nslots(cfg))   # cg27a SHARED HEAD: every slot index points at ONE head (tied; counted once)
        for h_ in s.heads: h_.cap=cfg.logit_cap
        # EMBEDDING CHANNEL NEURONS: shared byte/position entry point, gated like any layer
        s.register_buffer('emb_theta0', sample_theta(cfg.d_model))
        s.register_buffer('emb_imp', torch.zeros(cfg.d_model))
        s.register_buffer('emb_usage', torch.zeros(cfg.d_model))
        s.register_buffer('emb_drive', torch.zeros(cfg.d_model))
        s.register_buffer('emb_drive_ema', torch.zeros(cfg.d_model))
        s.register_buffer('emb_drive_slow', torch.zeros(cfg.d_model)); s.register_buffer('emb_drive_peak', torch.zeros(cfg.d_model))
        s.register_buffer('emb_sens', torch.zeros(cfg.d_model))
        # DISTINCTNESS: running feature-centroid of content seen so far (the "typical"
        # content, dominated by the majority=prose). distinctness = distance from it.
        s.register_buffer('feat_centroid', torch.zeros(cfg.d_model))
        s.register_buffer('feat_scale', torch.ones(1))         # running distance scale (for [0,1] normalization)
        s.register_buffer('centroid_init', torch.zeros(1))
        s.register_buffer('distinct_locked', torch.zeros(1))   # once locked, centroid/scale frozen
    def lock_distinctness(s):
        s.distinct_locked.fill_(1)   # freeze the reference -> stable routing for all content
        for b in s.blocks: b.distinct_locked.fill_(1)   # blocks read their own copy in route_gate
    def _distinctness(s, h):
        # h: (B,T,d_model). distinctness = normalized distance from the running centroid.
        feat=h.detach().mean((0,1))
        if s.centroid_init.item()==0:
            s.feat_centroid.copy_(feat); s.centroid_init.fill_(1); return torch.tensor(0.0,device=h.device)
        dist=(feat-s.feat_centroid).norm()
        if s.distinct_locked.item()==0:               # only update reference BEFORE lock
            s.feat_scale.mul_(0.99).add_(0.01*dist)
        d=torch.clamp(dist/(s.feat_scale*2+1e-6),0,1)
        # centroid drifts ONLY before lock; after lock it's frozen -> STABLE routing
        # (a given content always gets the same distinctness -> reads its own region)
        if s.training and s.distinct_locked.item()==0:
            s.feat_centroid.mul_(0.999).add_(0.001*feat)
        return d
    def forward(s,x,domain=0):
        if s.cfg.slot_mem and not _CARRY['on'] and int(domain) in s.__dict__.get('_smem',{}):     # SLOT MEMORY: scoring slot d starts from S_d
            sv_=(_CARRY['st'],_CARRY['slot'],_CARRY['sm']); _CARRY['sm']=s.__dict__['_smem']; _CARRY['slot']=int(domain)
            _CARRY['on']=True; _CARRY['nr']=x.shape[0]; _CARRY['new']={}; _CARRY['st']=_smem_seed(int(domain),x.shape[0])
            try: return s._forward0(x,domain)
            finally: _CARRY['on']=False; _CARRY['new']={}; _CARRY['st'],_CARRY['slot'],_CARRY['sm']=sv_
        return s._forward0(x,domain)
    def _forward0(s,x,domain=0):
        if (s.cfg.ngp and s.__dict__.get('_ngp') is not None) or (s.cfg.fcp_tables and s.__dict__.get('_fcp') is not None):   # + private tables
            sv_=s.__dict__.get('_s0_dom')
            s.__dict__['_s0_dom']=int(domain)
            try: h_=s.features(x)
            finally: s.__dict__['_s0_dom']=sv_
            hd_=s.heads[domain]; z_=priv(s,x,h_,int(domain))
            if _HCAP['on']: _HCAP['h']=z_.detach()
            return hd_.detached(z_) if _BUFDET['on'] else hd_(z_)
        sv_=s.__dict__.get('_s0_dom'); s.__dict__['_s0_dom']=int(domain)
        try:
            z_=s.features(x)
            if _HCAP['on']: _HCAP['h']=z_.detach()
            return s.heads[domain].detached(z_) if _BUFDET['on'] else s.heads[domain](z_)
        finally: s.__dict__['_s0_dom']=sv_
        return s.heads[domain](s.features(x))
    def features(s,x,route=None,nstat=None):
        f_=_COMPILED.get(id(s))
        if f_ is not None and not _COMPILED.get('failed'):
            try: return f_(x,route=route,nstat=nstat)
            except Exception as e:
                _fail('model:117')
                _COMPILED['failed']=True; print(f"  [compile] compiled forward failed ({repr(e)[:160]}); eager from now on")
        return s._features_eager(x,route,nstat)
    def _features_eager(s,x,route=None,nstat=None):
        T=x.size(1); h=s.embed(x)+s.pos(torch.arange(T,device=x.device)%s.pos.num_embeddings)
        if _STF['on']:                                                                  # STIFFNESS: which rows are dreams in this pass
            if torch.is_grad_enabled() and (s.training or _STF['dream_call']):
                ns0_=0 if _STF['dream_call'] else nstat; _STF['B']=x.shape[0]; _STF['ns']=ns0_ if (ns0_ is not None and ns0_<x.shape[0]) else None
            else: _STF['ns']=None
        if _XDG['on'] or _XDG.get('tagon'): _XDG['rows']=s.__dict__.get('_s0_dom')
        if _PJ['on']:                                                                   # PROJECTION: which rows are dreams in this pass
            _PJ['B']=x.shape[0]; _PJ['ns']=0 if _PJ['dream_call'] else (nstat if (s.training and nstat is not None and nstat<x.shape[0]) else None)
        if getattr(s,'ngram',None) is not None:                                         # C1: hashed byte n-grams
            rec_=s.training and torch.is_grad_enabled()
            h=h+ngram_embed(s.ngram,x,touched=s.__dict__.setdefault('_ng_touch',{}) if (rec_ and s.cfg.ngram_sparse) else None,
                            counts=s.__dict__.get('_ng_cur') if (rec_ and s.cfg.ngram_protect>0) else None,ncount=nstat)
        s._nstat=nstat
        for b in s.blocks: b._nstat=nstat
        if s.training:
            cd=h.detach()[:nstat].abs().mean((0,1))
            s.emb_usage+=cd; s.emb_drive=cd
            _track(s,'emb',cd)
        if s.cfg.route_mode=='none' and all(b.calib is None for b in s.blocks):   # unused: skip (it forced 2 GPU syncs per pass)
            distinct=None
        else: distinct=s._distinctness(h)                      # online distinctness of this content
        for b in s.blocks: b.cur_distinct=distinct
        ac=(torch.autocast(device_type=h.device.type, dtype=torch.bfloat16) if s.cfg.amp=='bf16' else contextlib.nullcontext())
        rs_=route_scales(route) if route is not None else None      # PER-LAYER DREAMS: gradient multipliers at the block boundaries
        with ac:
            side_=s.cfg.side_w>0 and s.__dict__.get('_side_on') and s.__dict__.get('_s0_dom') is not None
            taps_=[h.detach()] if side_ else None
            for i_,b in enumerate(s.blocks):
                if _INC['on'] and i_==_INC['k']: h=h[:,-1:]                       # INCREMENTAL GENERATION: newest byte only from here on
                b._dcontent=bool(s.cfg.dream_content>0 and i_>=len(s.blocks)-int(s.cfg.dream_content))
                if rs_ is not None: h=_GScale.apply(h,rs_[0][i_])
                h=b(h)
                if side_: taps_.append(h.detach())
                if _ENC_STOP['k'] is not None and i_>=_ENC_STOP['k']: break          # v9e FIRST-BLOCK CHECK: the re-encode stops here
            if rs_ is not None: h=_GScale.apply(h,rs_[1])
        h=h.float()
        if side_: h=h+side_out(s,taps_,s.__dict__.get('_s0_dom'),nstat)
        if s.training and getattr(s,'_keep_h',False): s._last_h=h.detach()[:nstat]
        if nstat is not None:
            s._nstat=None
            for b in s.blocks: b._nstat=None
        _STF['ns']=None; _PJ['ns']=None; _XDG['rows']=None
        return h
    def widen_all(s, chunk):
        for b in s.blocks:
            if b.d_state < s.cfg.d_state_max: b.widen(chunk, s.embed.weight.device)

# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded
_LATE = {
    '_ENC_STOP': 'platypus.conduit',
    '_INC': 'platypus.conduit',
}
