"""Smoke test of the INTEGRATED keyed memory (platypus.inference.kmem: kmi_refresh, kmi_rows_loss, kmi_readout) on a tiny untrained platypus
Model on the CPU: a fabricated pool (toy facts in six relations, written inferences right / wrong, held back), keys refreshed as at a
readout, then the question rows of the pool fed to the memory as the wake and dose rows would (trunk logits detached), then the readout:
the model's answers WITH the memory vs the trunk alone. Usage: python3 tools/kmi_smoke.py [steps (300)]"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus
from platypus.config import Config, _CONFIG_DEFAULTS
from platypus.presets import ALLPILE_CONFIG
for k, v in _CONFIG_DEFAULTS.items(): setattr(Config, k, v)
cfg = Config(); cfg.seed = 0
for k, v in ALLPILE_CONFIG.items(): setattr(cfg, k, v)
ovr = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'presets', os.environ.get('KM_PRESET', 'cg27a_kmi') + '.overrides.json')))
for k, v in ovr.items():
    if not k.startswith('_'): setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
cfg.d_model = 64; cfg.n_layers = 2; cfg.seq_len = 128
if os.environ.get('KM_CSTEPS'): cfg.inf_kmc_steps = int(os.environ['KM_CSTEPS'])
if os.environ.get('KM_MODE'): cfg.inf_kmc_mode = os.environ['KM_MODE']
if os.environ.get('KM_MARGIN'): cfg.inf_kmc_margin = float(os.environ['KM_MARGIN'])
from platypus.model import Model
from platypus.inference.distill import _dt_forms
from platypus.inference.closed_form import _ws_batch, _ws_run
from platypus.inference.kmem import kmi_refresh, kmi_rows_loss, kmi_readout
torch.manual_seed(0); model = Model(cfg); model.eval(); rs = np.random.RandomState(0); T0 = time.time()
syl = ['ka', 'lo', 'mi', 'ner', 'tas', 'vol', 'ru', 'den', 'sa', 'por', 'el', 'ith']
nm = lambda: (''.join(rs.choice(syl, 2)) + ' ' + ''.join(rs.choice(syl, 3))).title()
rels = ['works with', 'lives near', 'reports to', 'is a partner of', 'trained under', 'knows']
people = list(dict.fromkeys(nm() for _ in range(400))); ents, held = [], []
for i, A in enumerate(people[:120]):
    for r in rs.choice(rels, 2, replace=False):
        C = people[160 + rs.randint(200)]; e = dict(key=('fact', A, r, C), A=A, r=r, ans=C, d=0); (held if i % 5 == 0 else ents).append(e)
for e0 in [e for e in list(ents) if e['r'] != 'is connected to'][:40]:     # chains: some answers are subjects of their own facts (two-hop readout)
    r = rs.choice(rels); C = people[360 + rs.randint(40)]; ents.append(dict(key=('fact', e0['ans'], r, C), A=e0['ans'], r=r, ans=C, d=0))
for i in range(80):
    A, C = people[rs.randint(400)], people[rs.randint(400)]
    if A != C:
        e = dict(key=('fact', A, 'is connected to', C), A=A, r='is connected to', ans=C, d=0, truth=bool(i % 3)); (held if i % 4 == 0 else ents).append(e)
S_slot = lambda d: 0; FR = _dt_forms(cfg)
kmi_readout(model, cfg, S_slot, ents, held, 0); K = kmi_refresh(model, cfg, ents, held, 0)
print(f"refreshed: {len(K['nk'])} name keys, {len(K['wk'])} wording keys, V {K['V']}, all-text {K.get('allt')} (head loss {K.get('hloss', float('nan')):.2f})  ({time.time() - T0:.0f}s)", flush=True)
fs = [e for e in ents if e['key'][0] == 'fact']
for st in range(int(sys.argv[1]) if len(sys.argv) > 1 else 300):
    items = [(FR[rs.randint(len(FR))].format(A=e['A'], r=e['r']).encode('latin-1'), (e['ans'] + '.').encode('latin-1'), 0, repr(e['key'])) for e in (fs[rs.randint(len(fs))] for _ in range(16))]
    X, _ = _ws_batch(model, [it[0] + it[1] for it in items])
    with torch.no_grad(): lg = _ws_run(model, X, 0)
    kmi_rows_loss(model, cfg, items, lg, X, 0)
print(f"memory steps {K['steps']}, rows addressed {K['hit']} of {K['rows']}, loss {np.mean(K['loss'][:20]):.3f} -> {np.mean(K['loss'][-20:]):.3f}  ({time.time() - T0:.0f}s)", flush=True)
kmi_readout(model, cfg, S_slot, ents, held, 1)
if int(os.environ.get('KM_KMC', 0)):                                  # consolidation through the memory: judged / kept / retracted inferences
    from platypus.inference.premises import _PM, _RTI_R
    from platypus.inference.distill import _kmc_pending
    from platypus.inference.kmem import kmc_retract, _kmi_counter_readout
    inf_ = [e for e in fs if e['r'] == 'is connected to']; W = {}
    for i, e in enumerate(inf_):
        W[(e['A'], e['ans'])] = dict(vis=0, d=0, truth=e['truth'], judged=(None if i % 3 == 0 else 1), flag=(i % 3 == 1 and not e['truth']), ret=(i % 3 == 1 and not e['truth']))
    _PM['rti'] = W; cfg.inf_kmc = 1
    items = [(FR[0].format(A=e['A'], r=e['r']).encode('latin-1'), (e['ans'] + '.').encode('latin-1'), 0, repr(('fact', e['A'], e['r'], e['ans']))) for e in inf_]
    sk = _kmc_pending(cfg, items)
    print(f"consolidation: {len(items)} inference rows, {len(sk)} memory-only (unjudged or flagged), {len(items) - len(sk)} taught to the trunk too", flush=True)
    RX = [e for e in inf_ if W[(e['A'], e['ans'])]['ret']]; KP = [e for e in inf_ if not W[(e['A'], e['ans'])]['flag']]
    _kmi_counter_readout(model, cfg, K, S_slot, KP, RX, [e for e in fs if e['r'] != 'is connected to'], 9)   # on a copy: before -> after
    kmc_retract(model, cfg, S_slot, ents, 1)
    print(f"retracted marked done: {sum(1 for w in W.values() if w.get('kmc_done') is not None)} of {sum(1 for w in W.values() if w.get('ret'))}", flush=True)
if K.get('allt'):                                                     # 15 Nov ALL-TEXT: bits on ordinary text and on fact sentences, trunk vs trunk + gated memory
    from platypus.inference.kmem import kma_logits, _kma_run
    import torch.nn.functional as F9
    def tb(texts, gate=None):
        X, M = _ws_batch(model, texts); lg, H = _kma_run(model, X, 0); lm, opn, na, no = kma_logits(model, K, X, lg, H, gate)
        m = M[:, 1:lg.shape[1] + 1]; f = lambda z: float((F9.cross_entropy(z.reshape(-1, z.shape[-1]).float(), X[:, 1:lg.shape[1] + 1].reshape(-1), reduction='none') * m.reshape(-1)).sum() / m.sum() / np.log(2))
        return f(lg), f(lm), 1000 * na / float(m.sum()), 1000 * no / float(m.sum())
    plain = [b'The weather was mild and the river ran slowly past the old mill, where Thomas Grey kept his accounts in a ledger. ' * 2] * 8
    plain += [f"Yesterday {people[300 + q]} said the road to {people[350 + q]} was closed. ".encode('latin-1') for q in range(8)]
    factsent = [f"Notes say that {e['A']} {e['r']} {e['ans']}. Nothing more.".encode('latin-1') for e in fs[:16]]
    for nm_, T_ in (('ordinary text (unwritten names)', plain), ('sentences stating TAUGHT facts', factsent)):
        for g_ in ((1, 0) if K.get('gate') else (0,)):
            a0, a1, na, no = tb(T_, g_); print(f"ALL TEXT | {nm_} | gate {g_}: bits {a0:.4f} -> {a1:.4f} ({a1 - a0:+.4f}); addressed {na:.0f}, open {no:.1f} per 1000 bytes", flush=True)
for p in model.parameters(): p.data += 0.01 * torch.randn_like(p)   # the trunk moves; keys refreshed (drift), read again
kmi_refresh(model, cfg, ents, held, 1); kmi_readout(model, cfg, S_slot, ents, held, 2)
print(f"DONE ({time.time() - T0:.0f}s)")
