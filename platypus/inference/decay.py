"""Decay model of store-off recall."""
import math
import numpy as np
import torch

_HL_FEAT=('bias','log kept replays','log mean spacing','trunk bits (information)','previous margin','log name frequency')
def _hl_x(c_,vis_,df_,mtb_):
    """cg26h: decay-model features of a conclusion as seen at visit vis_ (the fit in _hl_model and release timing share them)"""
    rv_=c_.get('rv',[]); sp_=float(np.mean(np.diff(rv_))) if len(rv_)>=2 else 0.0
    pm_=[h_[2] for h_ in c_.get('so',[]) if h_[0]<vis_]; pm_=pm_[-1] if pm_ else 0.0
    an_=c_['A'].encode('latin-1','replace') if isinstance(c_['A'],str) else c_['A']
    return [1.0,math.log2(1+c_.get('nk',0)),math.log2(1+sp_),float(c_.get('tbits',mtb_)),float(np.clip(pm_,-5,5))/5,math.log2(1+df_.get(an_,0))/10]
def _hl_model(model,cfg,S_,obs9_,vis_):
    """cg26h DECAY MODEL (log only - nothing acts on it yet): store-off recall of a conclusion d visits after its last KEPT replay is modelled as
    p = 2^(-d / h), half-life h = 2^(theta . x), x = (1, log kept replays, log mean spacing of replays, trunk bits per byte (its information for
    the trunk), margin at the previous test, log frequency of its subject name). Fitted online, pooled over conclusions (half-life regression).
    Each sleep the previous fit predicts this sleep's tests first (out of sample), against a constant-rate baseline."""
    df_=S_.get('df',{}); rows_=[]
    tb0_=[c_['tbits'] for c_,_ in obs9_.values() if 'tbits' in c_]; mtb_=float(np.median(tb0_)) if tb0_ else 2.0
    for c_,r_ in obs9_.values():
        d_=vis_-int(c_.get('lr',c_.get('bv',vis_)))
        if d_<=0: continue
        rows_.append((_hl_x(c_,vis_,df_,mtb_),float(d_),float(bool(r_[0]))))
    D_=S_.setdefault('_hl_obs',[]); th_=model.__dict__.get('_hl_th'); oos_=''
    def pr_(th9_,R_):
        X_=torch.tensor([x_ for x_,_,_ in R_]); dd_=torch.tensor([d_ for _,d_,_ in R_])
        return torch.exp(-math.log(2)*dd_*torch.pow(2.0,-(X_@th9_))).clamp(1e-4,1-1e-4)
    bce_=lambda p_,y_: float(-(y_*torch.log(p_)+(1-y_)*torch.log(1-p_)).mean())
    if th_ is not None and rows_ and D_:
        y_=torch.tensor([y9_ for _,_,y9_ in rows_]); p1_=pr_(th_,rows_); b0_=float(np.clip(np.mean([y9_ for _,_,y9_ in D_]),1e-3,1-1e-3))
        lm9_,lc9_=bce_(p1_,y_),bce_(torch.full_like(y_,b0_),y_); S_.setdefault('_hl_cal',[]).append((vis_,lm9_,lc9_,len(rows_))); del S_['_hl_cal'][:-50]   # calibration record for release timing
        oos_=(f" | OUT OF SAMPLE on this sleep (n {len(rows_)}): log-loss model {lm9_:.3f} vs constant {lc9_:.3f}, "
              f"predicted recall {float(p1_.mean()):.2f} vs observed {float(y_.mean()):.2f}")
    D_.extend(rows_); del D_[:-5000]
    if len(D_)<30: print(f"  [inference DECAY MODEL] {len(D_)} observations so far (need 30) | this sleep {len(rows_)}"+oos_,flush=True); return
    th9_=(th_.clone() if th_ is not None else torch.tensor([2.0,0,0,0,0,0])).requires_grad_(True); op9_=torch.optim.Adam([th9_],lr=0.05); y_=torch.tensor([y9_ for _,_,y9_ in D_])
    with torch.enable_grad():
        for _ in range(300):
            p_=pr_(th9_,D_); l_=-(y_*torch.log(p_)+(1-y_)*torch.log(1-p_)).mean()+1e-3*(th9_[1:]**2).sum(); op9_.zero_grad(); l_.backward(); op9_.step()
    th9_=th9_.detach(); model.__dict__['_hl_th']=th9_; model.__dict__['_hl_mtb']=mtb_
    Xm_=torch.tensor([x_ for x_,_,_ in D_]).mean(0); hm_=float(torch.pow(2.0,Xm_@th9_))
    print(f"  [inference DECAY MODEL] (log only) {len(D_)} observations (this sleep {len(rows_)}) | half-life at the mean conclusion {hm_:.1f} visits | theta: "
          +", ".join(f"{n_} {float(v_):+.2f}" for n_,v_ in zip(_HL_FEAT,th9_))+" (positive = longer half-life)"+oos_,flush=True)
