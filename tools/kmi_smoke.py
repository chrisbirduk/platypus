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
for i in range(80):
    A, C = people[rs.randint(400)], people[rs.randint(400)]
    if A != C:
        e = dict(key=('fact', A, 'is connected to', C), A=A, r='is connected to', ans=C, d=0, truth=bool(i % 3)); (held if i % 4 == 0 else ents).append(e)
S_slot = lambda d: 0; FR = _dt_forms(cfg)
kmi_readout(model, cfg, S_slot, ents, held, 0); K = kmi_refresh(model, cfg, ents, held, 0)
print(f"refreshed: {len(K['nk'])} name keys, {len(K['wk'])} wording keys, V {K['V']}  ({time.time() - T0:.0f}s)", flush=True)
fs = [e for e in ents if e['key'][0] == 'fact']
for st in range(int(sys.argv[1]) if len(sys.argv) > 1 else 300):
    items = [(FR[rs.randint(len(FR))].format(A=e['A'], r=e['r']).encode('latin-1'), (e['ans'] + '.').encode('latin-1'), 0, repr(e['key'])) for e in (fs[rs.randint(len(fs))] for _ in range(16))]
    X, _ = _ws_batch(model, [it[0] + it[1] for it in items])
    with torch.no_grad(): lg = _ws_run(model, X, 0)
    kmi_rows_loss(model, cfg, items, lg, X)
print(f"memory steps {K['steps']}, rows addressed {K['hit']} of {K['rows']}, loss {np.mean(K['loss'][:20]):.3f} -> {np.mean(K['loss'][-20:]):.3f}  ({time.time() - T0:.0f}s)", flush=True)
kmi_readout(model, cfg, S_slot, ents, held, 1)
for p in model.parameters(): p.data += 0.01 * torch.randn_like(p)   # the trunk moves; keys refreshed (drift), read again
kmi_refresh(model, cfg, ents, held, 1); kmi_readout(model, cfg, S_slot, ents, held, 2)
print(f"DONE ({time.time() - T0:.0f}s)")
