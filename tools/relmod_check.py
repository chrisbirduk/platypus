"""Unit check of the relational module (inf_distill_relmod) on the rule world plus 20 arbitrary relations of 1500 names: python3 tools/relmod_check.py [steps]"""
import sys, time, types, numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
torch.set_num_threads(1)
from platypus.config import *  # noqa
import platypus.config as C
from platypus.inference import probes, distill
cfgc = [C.Config]
cfg = C.Config()
cfg.seed = 0; cfg.inf_rule_world = 12; cfg.inf_distill_relmod = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
W = probes.inf_rule_world(cfg)
doms = sorted({r[0] for r in W['planted']}); print('domains', doms, 'planted', len(W['planted']), 'held', len(W['held']))
model = types.SimpleNamespace(); model.__dict__['_t2s'] = {d: 1 for d in doms}
ents = []
for d, s, r, a, b in W['planted']:
    ents.append(dict(key=('fact', a, W['rel'][r], b), A=a, r=W['rel'][r], ans=b))
rs = np.random.RandomState(1); nm = [f'Pq{i}x' for i in range(1500)]
for rel in [f'rel{k} of' for k in range(20)]:
    for a in rs.choice(1500, 80, replace=False): ents.append(dict(key=('fact', nm[a], rel, nm[rs.randint(1500)]), A=nm[a], r=rel, ans=nm[rs.randint(1500)]))
t = time.time(); distill._dw_relmod(model, cfg, ents, [], 1); print('t', time.time() - t)
