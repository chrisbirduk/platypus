"""Split the stage-0 single-file build into the platypus package (real modules, explicit imports).
python3 split.py S0.py OUTDIR [--report]
Every top-level statement keeps its text verbatim; modules are contiguous runs of the original file (anchors below).
Names a module needs from a module that loads EARLIER are imported at the top; names it needs only at run time
from a module that loads LATER are listed in _LATE and linked by platypus/__init__.py once every module is loaded."""
import ast, sys, os, re, json, builtins, collections, textwrap
SRC, OUT = sys.argv[1], sys.argv[2]
REPORT = '--report' in sys.argv
src = open(SRC).read(); L = src.split('\n'); tree = ast.parse(src)

# (module, first top-level name it defines, one-line description)
SPEC = [
 ('common',              None,               'Shared imports, the device, and the helpers that keep module-level names consistent across modules.'),
 ('config',              'Config',           'Config: every option with its default; the base domains and their difficulty.'),
 ('gating',              '_NOISE',           'Write gating: thresholds from impedance and importance, drive statistics, polarisation-gated writes with split weight gradients.'),
 ('protection',          '_STF',             'Weight protection: replay stiffness, projections, Kalman/Fisher factors, protected Muon, varpro head refits, probes.'),
 ('private',             'ngp_init',         'Per-domain private parts: masks, private trunk units, private n-gram tables.'),
 ('slots',               '_slot_bpc',        'Replay and stream sources, slot scoring and slot (domain) inference.'),
 ('layers',              'Head',             'Output heads and the delta-rule core: WY solve, chunked and blocked kernels, DeltaBlock.'),
 ('capacity',            'Reserve',          'Dormant reserve, uniqueness-driven activation, per-domain weight deltas.'),
 ('sampler',             '_TEACH_DROP',      'Frozen teachers, stacked teacher pass and the recurrent dream sampler.'),
 ('dream_bank',          'build_teachers',   'Dream bank: teacher building, stream generation, bank refresh and dream batches.'),
 ('ngrams',              'adam_capture',     'Hashed n-gram embeddings and count features, the main optimiser and compilation.'),
 ('model',               'Model',            'The Model.'),
 ('meta',                'MetaController',   'Learned meta-controller, differentiable meta helpers and threshold gates.'),
 ('diagnostics',         'ctx_profile',      'Topic identification, reconstruction and forgetting diagnostics.'),
 ('importance',          'install_cov_hooks','Data-free input covariances, importance and revisit bookkeeping.'),
 ('data',                '_rank',            'Text data: Pile streaming with a disk cache, fact injection, loaders.'),
 ('heads',               'head_stats_update','Inferred heads (domain not given at test time) and BPC evaluation.'),
 ('train',               '_NOV',             'One visit of training (train_domain) and its step-level helpers.'),
 ('orth',                'ORTH_KEYS',        'Orthogonality diagnostics.'),
 ('conduit',             'set_seed',         'Conduit dreams: replay rows, hints, CUDA-graph and slot-memory generation, NREM losses.'),
 ('inference.store',     '_ng_zero',         'The fast store: name-indexed sparse logit corrector, and evaluation with it.'),
 ('inference.units',     '_INF_REL',         'Rule-free units: the learned lexicon and the MDL reader.'),
 ('inference.probes',    '_INF',             'Planted probe chains, prose variants and the real-text benchmark.'),
 ('inference.sentences', '_inf_store',       'The sentence store: adding, evicting, relations.'),
 ('inference.links',     '_LM',              'Link memory and cue repair.'),
 ('inference.infer',     'inf_infer',        'Offline inference: bridges, conclusions, staging and salience.'),
 ('inference.evaluate',  'inf_eval',         'Probe evaluation, CLS measurements and REM examination.'),
 ('inference.ledger',    '_tbytes',          'Memory ledger.'),
 ('inference.decay',     '_HL_FEAT',         'Decay model of store-off recall.'),
 ('inference.closed_form','_ws_eager',       'Closed-form edits of the last block MLP: layer states, drift, application.'),
 ('inference.sleep',     'inf_sleep',        'Sleep: NREM, REM, store writes, replay into the trunk.'),
 ('inference.readouts',  'inf_final_bpc',    'End-of-run readouts: dual pathway, CLS probes, ceiling, transfer.'),
 ('inference.selftaught','_inf_free',        'Self-taught inference: freeing, occlusion, attended-span splices, spectral word codes.'),
 ('inference.mixing',    '_inf_mix',         'Mixing (splice) dreams: proposal, judging, dreaming.'),
 ('inference.jepa',      '_jepa_emb',        'JEPA dream selector and latent readouts.'),
 ('inference.rem',       '_inf_gen',         'Greedy generation, self-inference, episodic fact records, the rule layer and REM recombination.'),
 ('inference.qtime',     '_inf_wm_write',    'Question-time chaining with a working memory.'),
 ('inference.audit',     '_inf_heldout',     'Hypotheses, pruning, abstention and the grounding audit.'),
 ('inference.need',      'ngp_step',         'Store-off helpers, feature-need signals, sleep-only feature learning, private-into-shared consolidation.'),
 ('losses',              '_nrem_loss_v1',    'Conduit losses, replay diagnostics, reconstruction fidelity and oracle readouts.'),
 ('runner',              'run',              'run(cfg): the whole continual-learning run.'),
 ('profiling',           '_PRF',             'Named regions for the one-off GPU profile.'),
]

def bound(n):
    out = []
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): out.append(n.name)
    elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        ts = n.targets if isinstance(n, ast.Assign) else [n.target]
        def tg(t):
            if isinstance(t, ast.Name): out.append(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts: tg(e)
            elif isinstance(t, ast.Starred): tg(t.value)
        for t in ts: tg(t)
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names: out.append((a.asname or a.name).split('.')[0])
    elif isinstance(n, ast.For):
        for x in ast.walk(n.target):
            if isinstance(x, ast.Name): out.append(x.id)
    return out

body = tree.body
main_i = next(i for i, n in enumerate(body) if isinstance(n, ast.If) and 'main' in ast.unparse(n.test))
stmts = body[:main_i]
# statement text spans: from the line after the previous statement's end (comments travel with the statement below them)
first = stmts[0].lineno - 1
if stmts[0].decorator_list if hasattr(stmts[0], 'decorator_list') else False: pass
spans = []
prev_end = None
for i, n in enumerate(stmts):
    st = min([n.lineno] + [d.lineno for d in getattr(n, 'decorator_list', [])])
    s0 = (prev_end + 1) if prev_end is not None else st
    spans.append((s0, n.end_lineno)); prev_end = n.end_lineno
header = '\n'.join(L[:stmts[0].lineno - 1])                     # the file's leading comment block
# a line holding two statements ("a=1; b=2") puts both in the same span: merge spans that share a line
anchors = {}
for i, n in enumerate(stmts):
    for b in bound(n): anchors.setdefault(b, i)
starts = []
for m, a, d in SPEC:
    starts.append(0 if a is None else anchors[a])
assert starts == sorted(starts), list(zip([s[0] for s in SPEC], starts))
mod_of_stmt = []
for i in range(len(stmts)):
    k = max(j for j, s in enumerate(starts) if s <= i); mod_of_stmt.append(SPEC[k][0])

# imports at the top of the file: alias -> import line
IMPORT_LINES = {}
for n in stmts:
    if isinstance(n, ast.Import):
        for a in n.names:
            al = a.asname or a.name
            IMPORT_LINES[al] = f"import {a.name}" + (f" as {a.asname}" if a.asname else "")
IMPORT_LINES['nn'] = 'import torch.nn as nn'; IMPORT_LINES['F'] = 'import torch.nn.functional as F'

# per module: own statements, defined names, used names
mods = collections.OrderedDict((m, {'stmts': [], 'def': [], 'use': set(), 'desc': d}) for m, a, d in SPEC)
defined_in = {}
for i, n in enumerate(stmts):
    m = mod_of_stmt[i]
    if isinstance(n, ast.Import): continue                      # imports are regenerated per module
    mods[m]['stmts'].append(i)
    for b in bound(n):
        if b in defined_in and defined_in[b] != m: print('NAME BOUND IN TWO MODULES:', b, defined_in[b], m)
        defined_in.setdefault(b, m)
        if b not in mods[m]['def']: mods[m]['def'].append(b)
    for x in ast.walk(n):
        if isinstance(x, ast.Name): mods[m]['use'].add(x.id)
        elif isinstance(x, ast.Global): mods[m]['use'].update(x.names)
order = list(mods)
BUILTINS = set(dir(builtins))

def modpath(m): return 'platypus.' + m
def fname(m): return os.path.join(OUT, 'platypus', *m.split('.')) + '.py'
report = []
late_all = {}
for k, m in enumerate(order):
    M = mods[m]
    need = sorted(u for u in M['use'] if u not in M['def'])
    std = sorted({IMPORT_LINES[u] for u in need if u in IMPORT_LINES and u not in defined_in})
    back = collections.OrderedDict(); late = {}
    for u in need:
        if u in IMPORT_LINES and u not in defined_in: continue
        if u not in defined_in: continue                         # builtins, locals, attributes of objects
        d = defined_in[u]
        if order.index(d) < k: back.setdefault(d, []).append(u)
        else: late[u] = modpath(d)
    late_all[m] = late
    lines = [f'"""{M["desc"]}"""']
    if m == 'common': lines += ['import os as _os0', "_os0.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')   # less fragmentation (only takes effect before CUDA starts)"]
    lines += [x for x in std if not (m == 'common' and x == 'import os as _os0')]
    for d, names in sorted(back.items(), key=lambda kv: order.index(kv[0])):
        names = sorted(names)
        chunk = [f"from {modpath(d)} import ("]
        row = '    '
        for nm in names:
            if len(row) + len(nm) + 2 > 150: chunk.append(row.rstrip()); row = '    '
            row += nm + ', '
        chunk.append(row.rstrip()); chunk.append(')')
        lines += chunk
    lines.append('')
    for i in M['stmts']:
        s, e = spans[i]
        txt = '\n'.join(L[s - 1:e])
        if m == 'common' and isinstance(stmts[i], ast.Expr) and '_os0.environ.setdefault' in txt: continue
        lines.append(txt)
    if m == 'profiling':                                       # the labelling loop runs once every module is loaded (platypus/__init__.py)
        body_ = '\n'.join(lines)
        loop_ = re.search(r"\nfor _nm in \[.*\n    if _nm in _pkg_globals\(\): .*", body_)
        assert loop_, 'profiling loop not found'
        body_ = body_.replace(loop_.group(0), "\ndef _label_regions():\n    \"\"\"label the named regions (called by platypus/__init__.py once every module is loaded)\"\"\"" + textwrap.indent(loop_.group(0), '    '))
        lines = body_.split('\n')
    if late:
        lines += ['', '# names this module uses at run time from modules that load after it; platypus/__init__.py links them once every module is loaded',
                  '_LATE = {']
        for u in sorted(late): lines.append(f"    {u!r}: {late[u]!r},")
        lines.append('}')
    os.makedirs(os.path.dirname(fname(m)), exist_ok=True)
    open(fname(m), 'w').write('\n'.join(lines).rstrip('\n') + '\n')
    report.append((m, sum(spans[i][1] - spans[i][0] + 1 for i in M['stmts']), len(M['def']), sum(len(v) for v in back.values()), len(late)))

# package order of the original module-level names (for _pkg_globals)
seen = set(); pkg_order = []
for i, n in enumerate(stmts):
    for b in bound(n):
        if b in seen or isinstance(n, ast.Import): continue
        seen.add(b); pkg_order.append((defined_in[b], b))
open(os.path.join(OUT, 'platypus', 'inference', '__init__.py'), 'w').write('"""Sleep inference: the fast store, offline inference, sleep, dreams, JEPA, readouts."""\n')
json.dump({'order': order, 'pkg_order': pkg_order, 'header': header, 'late': late_all}, open(os.path.join(OUT, 'split_meta.json'), 'w'))
if REPORT or True:
    print(f"{'module':24} {'lines':>6} {'names':>6} {'imports':>8} {'late':>5}")
    for r in report: print(f"{r[0]:24} {r[1]:6} {r[2]:6} {r[3]:8} {r[4]:5}")

# ---------------- presets.py, __main__.py, _order.py, __init__.py ----------------
mb = body[main_i]
seg = lambda n: textwrap.dedent('\n'.join(L[n.lineno - 1:n.end_lineno]))
nm1, allp = mb.body[0], mb.body[1]
assert isinstance(nm1, ast.Assign) and nm1.targets[0].id == 'NAME' and allp.targets[0].id == 'ALLPILE_CONFIG'
ovr_ = next(n for n in mb.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == '_OVR')
default_ovr = ovr_.value.args[1].value
nm2 = [n for n in mb.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'NAME'][-1]
v = nm2.value                                                           # ((CONST + NAME) + ' | overrides ') + _OVR
assert isinstance(v, ast.BinOp) and isinstance(v.right, ast.Name) and v.right.id == '_OVR'
assert isinstance(v.left.right, ast.Constant) and v.left.right.value == ' | overrides '
assert isinstance(v.left.left.right, ast.Name) and v.left.left.right.id == 'NAME'
build_name = v.left.left.left.value
pre = ['"""The GPU run: model and run options (ALLPILE_CONFIG), the default overrides (DEFAULT_LAB_OVR, replaced by the LAB_OVR',
       'environment variable) and the build name with its change history (BUILD_NAME, BASE_NAME)."""',
       '',
       'BASE_NAME = ' + repr(nm1.value.value), '',
       seg(allp), '',
       'DEFAULT_LAB_OVR = ' + repr(default_ovr), '',
       '# The build and its change history (newest first). The run prints BUILD_NAME + BASE_NAME + the overrides in use.',
       'BUILD_NAME = ' + repr(build_name), '']
open(os.path.join(OUT, 'platypus', 'presets.py'), 'w').write('\n'.join(pre))
mn = ['"""python -m platypus: one GPU run. Options: LAB_OVR (JSON overrides) and the PLATYPUS_* switches (see README)."""',
      'import os',
      'import numpy as np',
      'import platypus',
      'from platypus.config import Config, _CONFIG_DEFAULTS, PILE_DOMAINS',
      'from platypus.runner import run',
      'from platypus.presets import ALLPILE_CONFIG, BASE_NAME, BUILD_NAME, DEFAULT_LAB_OVR',
      '',
      'NAME = BASE_NAME']
_prev_ln = None
for n in mb.body[2:]:
    if n.lineno == _prev_ln: continue
    _prev_ln = n.end_lineno
    t = seg(n)
    if n is ovr_: t = "_OVR=os.environ.get('LAB_OVR',DEFAULT_LAB_OVR)"
    if n is nm2: t = "NAME=BUILD_NAME+NAME+' | overrides '+_OVR"
    mn.append(t)
src_main = '\n'.join(mn) + '\n'
# statements sharing a line (cfg=Config(); cfg.seed=0) were emitted twice: keep the first copy of each source line
open(os.path.join(OUT, 'platypus', '__main__.py'), 'w').write(src_main)
open(os.path.join(OUT, 'platypus', '_order.py'), 'w').write(
    '"""Every module-level name with its module, in the order of the original single-file build (used by _pkg_globals)."""\nORDER = (\n'
    + ''.join(f"    ({m!r}, {n!r}),\n" for m, n in pkg_order) + ')\n')
hdr = header
init = ['"""Platypus: a continual-learning byte-level language model with sleep inference, a complementary learning',
        'system design and JEPA dream selection. Copyright Christian Bird, Apache 2.0.',
        '',
        'The package is the GPU build split into modules; python -m platypus runs it (see __main__.py)."""',
        hdr, '',
        'import importlib as _il, sys as _sys',
        'MODULES = (' + ', '.join(repr(m) for m in order) + ')',
        "_loaded = [_il.import_module('platypus.' + m_) for m_ in MODULES]",
        '# link the names each module uses at run time from modules that load after it',
        'for m_ in _loaded:',
        "    for n_, p_ in getattr(m_, '_LATE', {}).items(): setattr(m_, n_, getattr(_sys.modules[p_], n_))",
        'from platypus import common as _common, _order as _ord',
        "_common._PKG_MODULES[:] = _loaded",
        "_common._PKG_ORDER[:] = [(_sys.modules['platypus.' + m_], n_) for m_, n_ in _ord.ORDER]",
        'from platypus import profiling as _profiling',
        '_profiling._label_regions()',
        'from platypus import presets',
        'from platypus.config import Config, _CONFIG_DEFAULTS',
        'from platypus.data import set_stream_source',
        'from platypus.runner import run',
        '']
open(os.path.join(OUT, 'platypus', '__init__.py'), 'w').write('\n'.join(init))
print('package written to', os.path.join(OUT, 'platypus'))
