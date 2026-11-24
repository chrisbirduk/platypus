"""24 Nov: how fast does the in-model JEPA memory (platypus/inference/jmem.py) learn when facts arrive domain by domain? The five-level world
of tools/jepa_store.py, its people split into 3 domains (geography in every domain); one 'sleep' per domain trains JMem on everything read so
far for JS_SLEEP steps (data replay, as kra50); after each sleep: one-hop 'lives' recall per domain and 2-hop chaining (lives -> city_in).
Usage: python3 tools/jmem_sim.py  (env JS_SLEEP steps a sleep, JS_WARM warm-up steps, JS_DIM, JS_LAYERS)"""
import os, sys, types, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2); T0 = time.time(); P = lambda s: print(s, flush=True)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jepa_store.py')).read(); cut = src.index('PRE1 = sorted(')
g = {'__name__': 'jm'}; exec(compile(src[:cut], 'jepa_store_world', 'exec'), g)
W, FACTS, CIT, REG, PEO = g['W'], g['FACTS'], g['CIT'], g['REG'], g['TRP'] + g['HEP']
from platypus.inference.jmem import JMem, _items_from
cfg = types.SimpleNamespace(inf_jmem_dim=int(os.environ.get('JS_DIM', 192)), inf_jmem_layers=int(os.environ.get('JS_LAYERS', 3)), inf_jmem_lr=1e-3, inf_jmem_warm=int(os.environ.get('JS_WARM', 0)))
J = JMem(cfg, torch.device('cpu')); SLP = int(os.environ.get('JS_SLEEP', 1500))
geo = [(t.format(a=a, b=b), (a, b)) for rel in ('city_in', 'region_of', 'country_in') for a, b in FACTS[rel] for t in W[rel][:1]]
GR = [PEO[k::3] for k in range(3)]; read = list(geo)
P(f"JEPA MEMORY SIMULATION: {len(PEO)} people in 3 domains, {len(geo)} geography sentences; {SLP} steps a sleep, warm-up {cfg.inf_jmem_warm}")
for dom in range(3):
    read += [(W['lives'][0].format(a=a, b=CIT[ci][0]), (a, CIT[ci][0])) for a, ci in GR[dom]]
    ls = J.train(_items_from(read), SLP)
    rec = []
    for k in range(dom + 1):
        o, _ = J.hop([a for a, ci in GR[k]], 'lives in'); rec.append(np.mean([x == CIT[ci][0] for x, (a, ci) in zip(o, GR[k])]))
    o1, _ = J.hop([a for a, ci in GR[0]], 'lives in'); o2, _ = J.hop(o1, 'lies in')
    t2 = np.mean([y == REG[CIT[ci][1]][0] for y, (a, ci) in zip(o2, GR[0])])
    P(f"   after sleep {dom + 1} ({J.steps} steps, {len(J.keys)} slots, loss {ls:.3f}): 'lives' recall by domain " + ' | '.join(f"D{k} {v:.2f}" for k, v in enumerate(rec)) + f" | 2 hops (D0) {t2:.2f}  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
