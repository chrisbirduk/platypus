"""Orthogonality diagnostics."""
import numpy as np
import torch
from platypus.config import (
    Config,
)
from platypus.gating import (
    _f32,
)


# ---------------- ORTHOGONALITY DIAGNOSTICS ----------------
# Question: would making writes ORTHOGONAL to what earlier domains use help, and is that
# orthogonal room already used up (so writes must be shunted elsewhere)?
# Every orth_every steps (a "window") of domains 2..5:
#  begin: from buffered earlier-domain batches, each projectable Linear's INPUT covariance ->
#         the old-domain input subspace U (top eigenvectors holding 90% / 99% of the energy).
#  end:   the window's actual weight change dW is split into its part inside U (dW U U^T, which
#         moves old-domain outputs) and the orthogonal part (which cannot, to first order).
#         Counterfactual losses, all from the same start weights:
#           actual        : what training did
#           weights_only  : only the projectable matrices changed (others kept at start)
#           orth@90/@99   : projectable matrices get only their orthogonal part
#         -> % of the window's forgetting orthogonal writes would remove, and % of the new
#            domain's learning they would keep. Eval-only: training is unchanged.
ORTH_KEYS=('Wk','gate','mlp_in','Wout','mlp_out')            # Wv shares Wk's input
ORTH_PARAM_KEY={'Wk':'Wk','Wv':'Wk','gate':'gate','mlp_in':'mlp_in','Wout':'Wout','mlp_out':'mlp_out'}

@torch.no_grad()
def _dream_cov(model, cfg):
    """Layer-input covariances of the old domains' dreams through the current network (equal weight per old domain)."""
    pools=getattr(model,'_dreams',None) or {}
    bt=[(torch.stack(pl[:cfg.og_dream_n]),None,d) for d,pl in sorted(pools.items()) if len(pl)>0]
    if not bt: return {}
    was=model.training; model.eval(); fs=model.feat_scale.clone()
    dc=_input_cov(model,[(x,x,d) for x,_,d in bt])
    model.feat_scale.copy_(fs); model.train(was); return dc

@torch.no_grad()
def _input_cov(model, batches):
    acc={}; hooks=[]
    for li,b in enumerate(model.blocks):
        for key in ORTH_KEYS:
            def hk(m,inp,out,k=(li,key)):
                x=inp[0].reshape(-1,inp[0].shape[-1]).float()
                c,n=acc.get(k,(0.0,0)); acc[k]=(c+x.T@x, n+x.shape[0])
            hooks.append(getattr(b,key).register_forward_hook(_f32(hk)))
    for x,_,d in batches: model(x,d)
    for h in hooks: h.remove()
    return {k:c/max(n,1) for k,(c,n) in acc.items()}

def _subspaces(C, energies):
    ev,U=torch.linalg.eigh(C.float()); ev=ev.flip(0).clamp(min=0); U=U.flip(1)
    cum=torch.cumsum(ev,0)/ev.sum().clamp(min=1e-30); out={}
    for e in energies:
        k=min(int((cum<e).sum().item())+1, C.shape[0]); out[e]=(U[:,:k].float(),k)
    return out

def orth_summary(log, dom, energies):
    """Aggregate a domain's windows: sums of loss changes (BPC) over windows."""
    W=[w for w in log if w['dom']==dom]
    if not W: return None
    S=lambda mode,e,i,ref=('start',None): sum(w['L'][(mode,e)][i]-w['L'][ref][i] for w in W)
    base=('others_only',None)
    # forgetting / learning caused by the projectable matrices, given all other changes (heads, norms, biases)
    f_mat=S('actual',None,0,base); l_mat=-S('actual',None,1,base)
    r={'windows':len(W),'forget_actual':S('actual',None,0),'forget_weights_only':S('weights_only',None,0),
       'forget_matrices':f_mat,'learn_matrices':l_mat}
    for e in energies:
        r[f'removed@{e}']=(S('actual',None,0,base)-S('orth',e,0,base))/f_mat if f_mat>1e-6 else float('nan')
        r[f'kept@{e}']=(-S('orth',e,1,base))/l_mat if l_mat>1e-6 else float('nan')
    avg=lambda key,e=None: {k:float(np.mean([np.mean([v for (li,kk),v in (w[key][e] if e is not None else w[key]).items() if kk==k]) for w in W])) for k in ORTH_KEYS}
    for e in energies: r[f'rank@{e}']=avg('rank',e)
    WU=[w for w in W if ('unit_own',None) in w['L']]
    if WU:
        g=lambda mode,i: sum(w['L'][(mode,None)][i]-w['L'][('actual',None)][i] for w in WU)
        r['unit_own_old']=g('unit_own',0); r['unit_own_new']=g('unit_own',1)       # BPC change vs actual (old<0 = less forgetting)
        r['unit_layer_old']=g('unit_layer',0); r['unit_layer_new']=g('unit_layer',1)
        r['unit_rank']=float(np.mean([np.mean([v['rank'] for v in w['unit_stats'].values()]) for w in WU if w['unit_stats']]))
        r['unit_new_in']=float(np.mean([np.mean([v['new_in'] for v in w['unit_stats'].values()]) for w in WU if w['unit_stats']]))
    r['new_in_old']=avg('ovl')
    r['upd_in_old']={k:float(np.mean([w['upd'][k] for w in W if k in w['upd']])) for k in ORTH_KEYS if any(k in w['upd'] for w in W)}
    r['imp_split']={t:float(np.nanmean([w['imp_split'].get(t,float('nan')) for w in W])) for t in ('top','rest')}
    return r

def print_orth(r, name, energies):
    e_lo,e_hi=min(energies),max(energies); f=lambda d:" ".join(f"{k} {v:.2f}" for k,v in d.items())
    print(f"  [orthogonality] {name}: old-domain loss rise over {r['windows']} windows {r['forget_actual']:+.3f} BPC, "
          f"of which the weight matrices cause {r['forget_matrices']:+.3f}; matrices' new-domain gain {r['learn_matrices']:+.3f}")
    print("     LAYER-level orthogonal writes (subspace energy -> % of matrix forgetting removed / % of matrix learning kept): "
          +", ".join(f"{e:.0%}: {100*r[f'removed@{e}']:.0f}/{100*r[f'kept@{e}']:.0f}" for e in energies))
    if 'unit_own_old' in r:
        print(f"     PER-UNIT (protected MLP units, own subspace @{Config.orth_unit_energy:.0%}): each fills {r['unit_rank']:.2f} of its input dims; "
              f"new-domain input inside it {r['unit_new_in']:.2f}")
        print(f"        making only those units' writes orthogonal: own subspace old {r['unit_own_old']:+.4f} / new {r['unit_own_new']:+.4f} BPC | "
              f"layer subspace old {r['unit_layer_old']:+.4f} / new {r['unit_layer_new']:+.4f} BPC  (old<0 = less forgetting, new>0 = lost learning)")
    print(f"     old subspace fills (fraction of input dims @{e_hi:.0%}): {f(r[f'rank@{e_hi}'])}")
    print(f"     new-domain input energy already inside old subspace:  {f(r['new_in_old'])}")
    print(f"     actual update energy inside old subspace:             {f(r['upd_in_old'])}")
    print(f"     inside-old share of updates, top-20%-importance units {r['imp_split']['top']:.2f} vs others {r['imp_split']['rest']:.2f}")
