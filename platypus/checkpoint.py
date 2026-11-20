"""Checkpoint and resume at visit boundaries.

PLATYPUS_CKPT=<folder> turns it on. After every visit run() writes <folder>/ckpt.pt (the previous one is
kept as ckpt.prev.pt); a run started with the same folder, build and options continues after the last
saved visit. A resumed run is the same computation as an uninterrupted one: tools/ckpt/equiv.py checks
this on the CPU (the logs match line for line after the resume point).

What a checkpoint holds, in one pickle (so objects shared between them stay shared):
  - the model's and the configuration's attributes (restored into the live objects that run() built);
  - the run() locals that carry from one visit to the next (runner.CK_LOCALS) and the read positions of
    the data loaders (their text is not saved: the resumed run reads it again from the Pile cache);
  - the contents of every package module's module-level state (the dicts, lists and sets, the other
    rebound names, function attributes such as train_domain._avg_difficulty, class attributes);
  - the torch (CPU and CUDA), numpy and Python random states.

Nested functions, lambdas and closure cells (forward hooks, loader closures) are pickled by value with
their module's globals by reference; torch.Generator objects by state; weak references (the identity
checks of caches keyed by id()) come back dead, so those caches miss once and recompute. Module-level containers, the
model, the configuration and modules are pickled by reference to the live objects of the resumed run.

Other switches: PLATYPUS_CKPT_EVERY=N (every Nth visit, default 1), PLATYPUS_CKPT_STRICT=1 (a failed
save stops the run; default: report it and go on without a checkpoint), PLATYPUS_CKPT_STOP_AFTER=V
(tests: exit with status 75 after saving visit V), PLATYPUS_CKPT_FORCE=1 (resume even if the build
or the options changed).
"""
import collections
import hashlib
import json
import marshal
import os
import pickle
import random
import sys
import time
import types
import weakref

import numpy as np
import torch

_TAG = 'platypus-ckpt'
CK_FORMAT = 1
# module-level names that are caches or setup products, not run state: kept as the resumed run built them
_KEEP_LIVE = {'_BYTES_CACHE', '_COMPILED', '_FLA', '_PKG_MODULES', '_PKG_ORDER', '_LATE', '_CONFIG_DEFAULTS', 'MODULES',
              '_REFS', '_PM', '_ckpt', '_KEEP_LIVE', '_TIME_ORIGINS', '_CONTAINERS', '_LOADER_SKIP'}   # (the last: this module, in the single-file export)
_TIME_ORIGINS = {'_RUN_T0'}                       # wall-clock origins: shifted on resume so elapsed times continue
_CONTAINERS = (dict, list, set, bytearray, collections.defaultdict, collections.OrderedDict, collections.deque, collections.Counter)

_REFS = {'save': None, 'load': None}             # id -> key while saving; key -> live object while loading


def ck_folder():
    p = os.environ.get('PLATYPUS_CKPT', '').strip()
    return p if p and p != '0' else None


def ck_every():
    return max(1, int(os.environ.get('PLATYPUS_CKPT_EVERY', '1') or 1))


# ---------------- pickling of closures, cells and generators ----------------
def _by_reference(fn):
    m_ = sys.modules.get(fn.__module__)
    if m_ is None: return False
    o_ = m_
    for p_ in fn.__qualname__.split('.'):
        o_ = getattr(o_, p_, None)
        if o_ is None: return False
    return o_ is fn


def _mk_fn(code, module, name, qualname, defaults, closure):
    g_ = sys.modules[module].__dict__ if module in sys.modules else {'__builtins__': __builtins__}
    f_ = types.FunctionType(marshal.loads(code), g_, name, defaults, closure)
    f_.__qualname__ = qualname
    return f_


def _set_fn_state(f_, state):
    kw_, d_ = state
    f_.__kwdefaults__ = kw_
    f_.__dict__.update(d_)


def _mk_cell():
    return types.CellType()


def _set_cell(c_, state):
    c_.cell_contents = state[0]


def _mk_gen(device, state):
    g_ = torch.Generator(device=device)
    g_.set_state(state)
    return g_


class _Gone:
    pass


def _dead_ref():
    return weakref.ref(_Gone())                                          # the instance is freed at once: a dead reference


def _ck_reduce(obj):
    if type(obj) is weakref.ReferenceType: return (_dead_ref, ())       # weak references (identity caches) come back dead
    if isinstance(obj, types.FunctionType) and not _by_reference(obj):
        return (_mk_fn, (marshal.dumps(obj.__code__), obj.__module__, obj.__name__, obj.__qualname__, obj.__defaults__, obj.__closure__),
                (obj.__kwdefaults__, dict(obj.__dict__)), None, None, _set_fn_state)
    if isinstance(obj, types.CellType):
        try: st_ = (obj.cell_contents,)
        except ValueError: st_ = None                                    # an empty cell
        return (_mk_cell, (), st_, None, None, _set_cell)
    if isinstance(obj, torch.Generator):
        return (_mk_gen, (str(obj.device), obj.get_state()))
    return NotImplemented


class _Pickler(pickle.Pickler):
    """pickle.Pickler for torch.save: our references first, then torch's own persistent ids (storages)"""
    def __setattr__(self, name, value):
        if name == 'persistent_id':
            torch_pid, refs = value, _REFS['save']
            def value(obj):
                k_ = refs.get(id(obj))
                if k_ is not None: return (_TAG,) + k_
                if isinstance(obj, types.ModuleType): return (_TAG, 'module', obj.__name__)
                return torch_pid(obj)
        super().__setattr__(name, value)

    def reducer_override(self, obj):
        return _ck_reduce(obj)


class _Unpickler(pickle.Unpickler):
    def __setattr__(self, name, value):
        if name == 'persistent_load':
            torch_pl, live = value, _REFS['load']
            def value(pid):
                if isinstance(pid, tuple) and pid and pid[0] == _TAG:
                    if pid[1] == 'module': return sys.modules[pid[2]]
                    return live[pid[1:]]
                return torch_pl(pid)
        super().__setattr__(name, value)


# ---------------- id() keys ----------------
# Much of the run state is keyed by id() of a parameter, a module or the model. Those numbers change in a
# resumed process, so the checkpoint carries (old id, object) for every int it holds that is the id of an
# object it also holds; after loading, those ints are replaced by the new objects' ids.
class _Sink:
    def write(self, b): return len(b)


_ATOMS = (str, float, bool, bytes, type(None), complex)


class _IdScan(pickle.Pickler):
    """a dry pickling pass (tensors and arrays not written) that records every object and every int it meets"""
    def __init__(self, refs):
        super().__init__(_Sink(), protocol=4)
        self.refs, self.objs, self.ints = refs, {}, set()

    def persistent_id(self, obj):
        t_ = type(obj)
        if t_ is int: self.ints.add(obj); return None
        if t_ in _ATOMS: return None
        self.objs[id(obj)] = obj
        if id(obj) in self.refs or isinstance(obj, (types.ModuleType, torch.Tensor, torch.UntypedStorage, torch.TypedStorage, np.ndarray)):
            return 'x'                                                   # recorded; not descended into
        return None

    def reducer_override(self, obj):
        return _ck_reduce(obj)


def _id_map(payload, refs):
    sc_ = _IdScan(refs); sc_.dump(payload)
    return [(i_, sc_.objs[i_]) for i_ in sc_.ints if i_ in sc_.objs]


def _remap_ids(root, remap, skip):
    """replace the old ids in dict keys and values, list, deque and set items and tuple keys of everything reachable from root"""
    if not remap: return 0
    def m_(x):
        if type(x) is int: return remap.get(x, x)
        if type(x) is tuple: return tuple(m_(y) for y in x)
        return x
    def hit(x):
        if type(x) is int: return x in remap
        if type(x) is tuple: return any(hit(y) for y in x)
        return False
    seen, stack, n_ = set(), [root], 0
    while stack:
        o_ = stack.pop()
        if id(o_) in seen or id(o_) in skip or type(o_) in _ATOMS or type(o_) is int: continue
        seen.add(id(o_))
        if isinstance(o_, (torch.Tensor, np.ndarray, types.ModuleType, type)): continue
        if isinstance(o_, dict):
            if any(hit(k_) or hit(v_) for k_, v_ in o_.items()):
                it_ = [(m_(k_), m_(v_)) for k_, v_ in o_.items()]; o_.clear(); o_.update(it_); n_ += 1
            for k_, v_ in o_.items(): stack.append(k_); stack.append(v_)
        elif isinstance(o_, (list, collections.deque)):
            for j_, v_ in enumerate(o_):
                if hit(v_): o_[j_] = m_(v_); n_ += 1
                else: stack.append(v_)
        elif isinstance(o_, set):
            if any(hit(v_) for v_ in o_):
                it_ = [m_(v_) for v_ in o_]; o_.clear(); o_.update(it_); n_ += 1
            stack.extend(o_)
        elif isinstance(o_, (tuple, frozenset)): stack.extend(o_)
        elif isinstance(o_, types.FunctionType):
            for c_ in o_.__closure__ or (): stack.append(c_)
            stack.append(o_.__dict__); stack.append(o_.__defaults__)
        elif isinstance(o_, types.CellType):
            try: v_ = o_.cell_contents
            except ValueError: continue
            if hit(v_): o_.cell_contents = m_(v_); n_ += 1
            else: stack.append(v_)
        else:
            d_ = getattr(o_, '__dict__', None)
            if isinstance(d_, dict): stack.append(d_)
            for sl_ in getattr(type(o_), '__slots__', ()) or ():
                if isinstance(sl_, str) and hasattr(o_, sl_): stack.append(getattr(o_, sl_))
    return n_


_PM = types.ModuleType('platypus_ckpt_pickle')
_PM.Pickler, _PM.Unpickler = _Pickler, _Unpickler
_PM.load, _PM.dump = pickle.load, pickle.dump


# ---------------- module-level state ----------------
def _ck_modules():
    """the package modules (this one aside); in the single-file export, the one module that holds everything"""
    try: from platypus.common import _PKG_MODULES
    except ImportError: _PKG_MODULES = []
    me = sys.modules[__name__]
    return [m_ for m_ in _PKG_MODULES if m_ is not me] or [me]


def _is_value(v):
    """a rebindable module-level value (not a module, class, function, or callable from outside the package)"""
    if isinstance(v, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType, types.MethodType)): return False
    if callable(v) and type(v).__module__ not in {m_.__name__ for m_ in _ck_modules()}: return False
    return True


def _ck_snap(c_):
    if type(c_) is collections.defaultdict: return ('defaultdict', c_.default_factory, dict(c_))
    if type(c_) is collections.deque: return ('deque', c_.maxlen, list(c_))
    if type(c_) is collections.OrderedDict: return ('ordered', list(c_.items()))
    if type(c_) is collections.Counter: return ('counter', dict(c_))
    return (type(c_).__name__, type(c_)(c_))


def _ck_fill(c_, s_):
    kind = s_[0]
    if kind == 'defaultdict': c_.default_factory = s_[1]; c_.clear(); c_.update(s_[2])
    elif kind == 'deque': c_.clear(); c_.extend(s_[2])
    elif kind == 'ordered': c_.clear(); c_.update(s_[1])
    elif kind in ('dict', 'counter', 'set'): c_.clear(); c_.update(s_[1])
    else: c_[:] = s_[1]                                                 # list, bytearray


def _collect_refs(anchors):
    """id -> key of every object that is pickled by reference; key -> object (the same map, inverted)"""
    refs, live = {}, {}
    for k_, o_ in anchors.items():
        refs[id(o_)] = (k_,); live[(k_,)] = o_
    for m_ in _ck_modules():
        for n_, v_ in vars(m_).items():
            if type(v_) in _CONTAINERS and id(v_) not in refs:
                refs[id(v_)] = ('g', m_.__name__, n_); live[('g', m_.__name__, n_)] = v_
    return refs, live


def _module_state():
    cont, vals, fattrs, cls = {}, {}, {}, {}
    seen = set()
    for m_ in _ck_modules():
        for n_, v_ in vars(m_).items():
            if n_.startswith('__') or n_ in _KEEP_LIVE: continue
            if type(v_) in _CONTAINERS:
                if id(v_) not in seen: seen.add(id(v_)); cont[('g', m_.__name__, n_)] = _ck_snap(v_)
            elif isinstance(v_, types.FunctionType):
                if getattr(v_, '__module__', None) == m_.__name__:
                    d_ = {a_: b_ for a_, b_ in vars(v_).items() if not a_.startswith('__')}
                    if d_: fattrs[(m_.__name__, n_)] = d_
            elif isinstance(v_, type):
                if v_.__module__ == m_.__name__:
                    d_ = {a_: b_ for a_, b_ in vars(v_).items() if not a_.startswith('__') and _is_value(b_)
                          and not isinstance(b_, (staticmethod, classmethod, property))}
                    if d_: cls[(m_.__name__, n_)] = d_
            elif _is_value(v_):
                vals[(m_.__name__, n_)] = v_
    return cont, vals, fattrs, cls


def _rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(), 'python': random.getstate()}


def _set_rng(r_):
    torch.set_rng_state(r_['torch'])
    if r_['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(r_['cuda'])
    np.random.set_state(r_['numpy']); random.setstate(r_['python'])


# ---------------- loaders ----------------
_LOADER_SKIP = {'src', 'cfg'}                     # the text and the configuration: rebuilt, not saved


def ck_loader_state(fn):
    """read position of a make_loader closure: its closure variables except the text, the configuration and functions"""
    st_ = {}
    for n_, c_ in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        if n_ in _LOADER_SKIP: continue
        try: v_ = c_.cell_contents
        except ValueError: continue
        if callable(v_) and not isinstance(v_, torch.Generator): continue
        st_[n_] = v_
    return {'cells': st_, 'attrs': {a_: b_ for a_, b_ in vars(fn).items() if not callable(b_)}}


def ck_set_loader_state(fn, s_):
    for n_, c_ in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        if n_ in s_['cells']: c_.cell_contents = s_['cells'][n_]
    for a_, b_ in s_['attrs'].items(): setattr(fn, a_, b_)


# ---------------- build identity ----------------
def ck_build_id(cfg):
    h_ = hashlib.sha256()
    root = os.path.dirname(os.path.abspath(__file__))
    files = [] if len(_ck_modules()) == 1 else [os.path.join(r_, f_) for r_, ds_, fs_ in sorted(os.walk(root)) for f_ in sorted(fs_) if f_.endswith('.py')]
    for p_ in files or [os.path.abspath(__file__)]:                     # the package's .py files; the single-file export: that file
        h_.update(os.path.relpath(p_, root).encode() + b'\0' + open(p_, 'rb').read() + b'\0')
    opts = json.dumps({k_: repr(v_) for k_, v_ in sorted(vars(cfg).items())}, sort_keys=True)
    return {'build': h_.hexdigest(), 'options': hashlib.sha256(opts.encode()).hexdigest(), 'python': sys.version.split()[0],
            'torch': torch.__version__}


def _ck_nbytes(obj):
    import io
    b_ = io.BytesIO()
    try: torch.save(obj, b_, pickle_module=_PM, pickle_protocol=4)
    except Exception as e: return -1
    return b_.tell()


def _report_sizes(payload):
    """PLATYPUS_CKPT_SIZES=1: the largest parts of the checkpoint (each measured alone, so shared objects count in each)"""
    rows = []
    for k_ in ('containers', 'values', 'fattrs', 'classes', 'cfg', 'model', 'locals', 'loaders'):
        for n_, v_ in payload[k_].items() if isinstance(payload[k_], dict) else []:
            rows.append((_ck_nbytes(v_), f"{k_}[{n_!r}]"))
    rows.sort(reverse=True)
    print("  [checkpoint] largest parts: " + " | ".join(f"{n_} {s_/2**20:.1f} MB" for s_, n_ in rows[:12]), flush=True)


# ---------------- save and load ----------------
def ck_save(vi, n_visits, cfg, model, run_locals, loaders, wall_run, ident):
    """write the state after visit vi; returns the size in bytes"""
    d_ = ck_folder(); os.makedirs(d_, exist_ok=True)
    t0 = time.perf_counter()
    cont, vals, fattrs, cls = _module_state()
    now = time.time()
    payload = {'format': CK_FORMAT, 'visit': int(vi), 'n_visits': int(n_visits), 'ident': ident, 'saved_at': now, 'wall_run': float(wall_run),
               'containers': cont, 'values': vals, 'fattrs': fattrs, 'classes': cls,
               'cfg': dict(vars(cfg)), 'model': dict(vars(model)), 'locals': run_locals, 'loaders': loaders, 'rng': _rng_state()}
    refs, _ = _collect_refs({'cfg': cfg, 'model': model})
    payload['idmap'] = _id_map(payload, refs)
    _REFS['save'] = refs
    if os.environ.get('PLATYPUS_CKPT_SIZES', '0') == '1': _report_sizes(payload)
    tmp = os.path.join(d_, 'ckpt.pt.tmp'); cur = os.path.join(d_, 'ckpt.pt'); prev = os.path.join(d_, 'ckpt.prev.pt')
    try:
        torch.save(payload, tmp, pickle_module=_PM, pickle_protocol=4)
    finally:
        _REFS['save'] = None
    if os.path.exists(cur): os.replace(cur, prev)
    os.replace(tmp, cur)
    size = os.path.getsize(cur)
    with open(os.path.join(d_, 'ckpt.json'), 'w') as f_:
        json.dump({'visit': int(vi), 'n_visits': int(n_visits), 'bytes': size, 'saved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)),
                   'seconds': round(time.perf_counter() - t0, 1), **ident}, f_, indent=2)
    return size, time.perf_counter() - t0


def ck_load(cfg, model, ident):
    """the saved payload (unpickled against the live cfg, model and module state), or None when there is no checkpoint"""
    d_ = ck_folder()
    if d_ is None: return None
    for name in ('ckpt.pt', 'ckpt.prev.pt'):
        p_ = os.path.join(d_, name)
        if not os.path.exists(p_): continue
        _, live = _collect_refs({'cfg': cfg, 'model': model})
        _REFS['load'] = live
        try:
            payload = torch.load(p_, map_location=None, pickle_module=_PM, weights_only=False)
        except Exception as e:
            print(f"  [checkpoint] {p_} could not be read ({repr(e)[:200]}); trying the previous one", flush=True)
            continue
        finally:
            _REFS['load'] = None
        if payload.get('format') != CK_FORMAT: raise RuntimeError(f"checkpoint {p_}: format {payload.get('format')}, this build writes {CK_FORMAT}")
        old = payload['ident']
        diff = [k_ for k_ in ('build', 'options', 'python', 'torch') if old.get(k_) != ident.get(k_)]
        if diff and os.environ.get('PLATYPUS_CKPT_FORCE', '0') != '1':
            raise RuntimeError(f"checkpoint {p_} was written by a different {', '.join(diff)} (saved {[old.get(k_) for k_ in diff]}, now "
                               f"{[ident.get(k_) for k_ in diff]}); use a new PLATYPUS_CKPT folder, or PLATYPUS_CKPT_FORCE=1 to resume anyway")
        payload['path'] = p_
        remap = {o_: id(x_) for o_, x_ in payload.pop('idmap', []) if o_ != id(x_)}
        payload['ids_remapped'] = _remap_ids(payload, remap, {id(v_) for v_ in live.values()})
        return payload
    return None


def ck_restore(payload, cfg, model):
    """put the saved module state, configuration, model attributes and random states back (in place)"""
    _, live = _collect_refs({'cfg': cfg, 'model': model})
    for k_, s_ in payload['containers'].items():
        if k_ in live: _ck_fill(live[k_], s_)
    mods = {m_.__name__: m_ for m_ in _ck_modules()}
    shift = time.time() - payload['saved_at']
    for (mn_, n_), v_ in payload['values'].items():
        if mn_ in mods: setattr(mods[mn_], n_, v_ + shift if n_ in _TIME_ORIGINS else v_)
    for (mn_, n_), d_ in payload['fattrs'].items():
        f_ = getattr(mods.get(mn_), n_, None)
        if f_ is not None:
            for a_ in [a_ for a_ in vars(f_) if not a_.startswith('__')]: delattr(f_, a_)
            for a_, b_ in d_.items(): setattr(f_, a_, b_)
    for (mn_, n_), d_ in payload['classes'].items():
        c_ = getattr(mods.get(mn_), n_, None)
        if c_ is not None:
            for a_, b_ in d_.items(): setattr(c_, a_, b_)
    vars(cfg).clear(); vars(cfg).update(payload['cfg'])
    vars(model).clear(); vars(model).update(payload['model'])
    _set_rng(payload['rng'])


def ck_stop_after(vi):
    s_ = os.environ.get('PLATYPUS_CKPT_STOP_AFTER', '').strip()
    return s_.lstrip('-').isdigit() and int(s_) == int(vi)


_ckpt = sys.modules[__name__]                    # the name runner.py calls through (also in the single-file export)
