"""Smoke test of the KEYED MEMORY readout (platypus.inference.kmem) on a tiny untrained platypus Model on the CPU, with a fabricated question
pool: toy facts in six relations (two per subject), written real-text inferences RIGHT / WRONG / flagged, held-back controls and two-hop
chains. Checks the code paths and the readout lines; an untrained trunk gives near-random name codes, so the numbers only show that
writing works (taught far above held back).
Usage: python3 tools/kmem_smoke.py [visits (2)]"""
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
ovr = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'presets', 'cg27a_rasymaa.overrides.json')))
for k, v in ovr.items():
    if not k.startswith('_'): setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
cfg.d_model = 64; cfg.n_layers = 2; cfg.seq_len = 128; cfg.inf_kmem = 1; cfg.inf_kmem_n = int(os.environ.get('KM_N', 32)); cfg.inf_kmem_rows = 20
from platypus.model import Model
torch.manual_seed(0); model = Model(cfg); model.eval()
rs = np.random.RandomState(0); T0 = time.time()
syl = ['ka', 'lo', 'mi', 'ner', 'tas', 'vol', 'ru', 'den', 'sa', 'por', 'el', 'ith']
nm = lambda: (''.join(rs.choice(syl, 2)) + ' ' + ''.join(rs.choice(syl, 3))).title()
rels = ['works with', 'lives near', 'reports to', 'is a partner of', 'trained under', 'knows']
people = list(dict.fromkeys(nm() for _ in range(400)))
ents, held = [], []
for i, A in enumerate(people[:160]):
    for r in rs.choice(rels, 2, replace=False):
        C = people[160 + rs.randint(200)]; e = dict(key=('fact', A, r, C), A=A, r=r, ans=C, d=0)
        (held if i % 5 == 0 else ents).append(e)
for i in range(120):
    A, C = people[rs.randint(400)], people[rs.randint(400)]
    if A == C: continue
    e = dict(key=('fact', A, 'is connected to', C), A=A, r='is connected to', ans=C, d=0, truth=bool(i % 3), flag=(i % 7 == 0))
    (held if i % 4 == 0 else ents).append(e)
for e in ents[:60]:                                                    # chains: A r B, B r' C
    if not any(x['A'] == e['ans'] for x in ents):
        ents.append(dict(key=('fact', e['ans'], 'knows', people[300 + rs.randint(100)]), A=e['ans'], r='knows', ans=people[300 + rs.randint(100)], d=0))
from platypus.inference.kmem import kmem_readout
S_slot = lambda d: 0
for vis in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
    if vis == 1:                                                       # a second readout: more facts arrive, the trunk moves a little
        for p in model.parameters(): p.data += 0.01 * torch.randn_like(p)
        for A in people[200:240]:
            ents.append(dict(key=('fact', A, 'works with', people[rs.randint(400)]), A=A, r='works with', ans=people[rs.randint(400)], d=0))
    kmem_readout(model, cfg, {}, S_slot, ents, held, vis)
print(f"DONE ({time.time() - T0:.0f}s)")
