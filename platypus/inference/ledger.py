"""Memory ledger."""
import torch
from platypus.common import (
    _pkg_globals,
)
from platypus.inference.store import (
    _INFM,
)
from platypus.inference.sentences import (
    _inf_store,
)

def _tbytes(o_,seen_,acc_,dep_=0):
    """cg26h: bytes of the tensors reachable from o_ (dicts, lists, tuples, sets, modules), by device; shared tensors counted once"""
    if dep_>4: return
    if torch.is_tensor(o_):
        k_=o_.untyped_storage().data_ptr() if o_.numel() else id(o_)
        if k_ in seen_: return
        seen_.add(k_); acc_['cuda' if o_.is_cuda else 'cpu']+=o_.numel()*o_.element_size(); return
    if isinstance(o_,torch.nn.Module):
        for t_ in list(o_.parameters())+list(o_.buffers()): _tbytes(t_,seen_,acc_,dep_+1)
        return
    if isinstance(o_,dict): it_=list(o_.values())
    elif isinstance(o_,(list,tuple,set)): it_=list(o_)
    else: return
    if len(it_)>20000: it_=it_[:20000]                                          # (big containers of plain data: tensors, if any, sit at the front)
    for v_ in it_:
        if torch.is_tensor(v_) or isinstance(v_,(dict,list,tuple,set,torch.nn.Module)): _tbytes(v_,seen_,acc_,dep_+1)
def _mem_ledger(model,cfg):
    """cg26h MEMORY LEDGER (readout): what holds memory and what grows. GPU and CPU tensor bytes by owner (model parameters, each long-lived
    entry of the model's state, module-level stores); counts of the sleep structures (sentence store, index, facts, conclusions, link graph,
    word counts, fast-store names); host memory; change since the last ledger. Untracked GPU memory = allocated minus everything counted
    (optimiser state and working buffers)."""
    if not int(getattr(cfg,'inf_ledger',0)): return
    G_=1/2**30; seen_=set(); own_={}
    def add_(k_,o_):
        a_={'cuda':0,'cpu':0}; _tbytes(o_,seen_,a_); own_[k_]=a_
    add_('model weights',model)
    for k_,v_ in list(model.__dict__.items()):
        if k_ in ('_modules','_parameters','_buffers') or k_.startswith('__'): continue
        if torch.is_tensor(v_) or isinstance(v_,(dict,list,tuple,set,torch.nn.Module)): add_('model.'+k_,v_)
    for k_,v_ in list(_pkg_globals().items()):
        if k_.startswith('_') and k_[1:2].isupper() and isinstance(v_,dict): add_(k_,v_)          # module-level stores (_INFM, _LM, ...)
    S_=_inf_store(model); it_=S_.get('it',[])
    cnt_={'store sentences':len(it_),'store live':sum(1 for x_ in it_[-400000:] if not x_.get('gone')),'name index':len(S_.get('idx',{})),'facts':len(S_.get('facts',[])),
          'conclusions':len(S_.get('conc',[])),'conclusions freed':sum(1 for c_ in S_.get('conc',[]) if c_.get('freed')),'link graph links':len(S_.get('W1',{})),'link graph entities':len({x_ for k_ in list(S_.get('W1',{}).keys())[:400000] if isinstance(k_,tuple) for x_ in k_}),
          'word counts':len(S_.get('df',{})),'evidence record':len(S_.get('rel',{})),'fact pairs':len(S_.get('fpair',{})),'known pairs':len(S_.get('pairs',())),'fast-store names':len(_INFM.get('names',{}) or {})}
    try: rss_=int([l_ for l_ in open('/proc/self/status') if l_.startswith('VmRSS')][0].split()[1])/2**20
    except Exception: rss_=float('nan')
    pv_=model.__dict__.get('_ledger_prev') or {}; dl_=lambda k_,v_: (f" ({v_-pv_[k_]:+.2f})" if isinstance(v_,float) else f" ({v_-pv_[k_]:+d})") if k_ in pv_ else ""
    gp_=sorted(((k_,a_['cuda']*G_) for k_,a_ in own_.items() if a_['cuda']>0),key=lambda z_: -z_[1])[:10]
    cp_=sorted(((k_,a_['cpu']*G_) for k_,a_ in own_.items() if a_['cpu']>0),key=lambda z_: -z_[1])[:6]
    al_=torch.cuda.memory_allocated()*G_ if torch.cuda.is_available() else float('nan'); tr_=sum(a_['cuda'] for a_ in own_.values())*G_
    cur_={**{('gpu '+k_):v_ for k_,v_ in gp_},**{('cpu '+k_):v_ for k_,v_ in cp_},**cnt_,'host GB':rss_,'gpu untracked':al_-tr_ if al_==al_ else float('nan')}
    print("  [inference MEMORY LEDGER] GPU (GB): "+", ".join(f"{k_} {v_:.2f}{dl_('gpu '+k_,v_)}" for k_,v_ in gp_)+(f" | untracked (optimiser, buffers) {al_-tr_:.2f}{dl_('gpu untracked',al_-tr_)}" if al_==al_ else ""),flush=True)
    print("  [inference MEMORY LEDGER] CPU tensors (GB): "+", ".join(f"{k_} {v_:.2f}{dl_('cpu '+k_,v_)}" for k_,v_ in cp_)+f" | host process {rss_:.1f} GB{dl_('host GB',rss_)}",flush=True)
    print("  [inference MEMORY LEDGER] counts: "+", ".join(f"{k_} {v_}{dl_(k_,v_)}" for k_,v_ in cnt_.items()),flush=True)
    model.__dict__['_ledger_prev']=cur_
