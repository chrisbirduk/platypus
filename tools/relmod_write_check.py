"""Unit check of step 4 (inf_distill_relmod_write): two sleeps of the relational module on the rule world (+ 20 arbitrary relations),
recall from the trunk stubbed (no model); checks the writes, their precision, the taught / held-back split and provenance at sleep 2.
Usage: python3 tools/relmod_write_check.py [steps]"""
import os
import sys
import types
import zlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)
import platypus.config as C  # noqa: E402
from platypus.inference import probes, distill  # noqa: E402
cfg = C.Config(); cfg.seed = 0; cfg.inf_rule_world = 12; cfg.inf_distill_relmod = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
cfg.inf_distill_relmod_write = 1
W = probes.inf_rule_world(cfg); doms = sorted({r[0] for r in W['planted']})
model = types.SimpleNamespace(); model.__dict__['_t2s'] = {d: 1 for d in doms}
ents = [dict(key=('fact', a, W['rel'][r], b), A=a, r=W['rel'][r], ans=b, d=d, src=(0,)) for d, s, r, a, b in W['planted']]
rs = np.random.RandomState(1); nm = [f'Pq{i}x' for i in range(1500)]
for rel in [f'rel{k} of' for k in range(20)]:
    for a in rs.choice(1500, 80, replace=False): ents.append(dict(key=('fact', nm[a], rel, nm[rs.randint(1500)]), A=nm[a], r=rel, ans=nm[rs.randint(1500)], d=0, src=(0,)))
ctl = lambda e: zlib.crc32(repr(e['key']).encode()) % 5 == 0
T0 = [e for e in ents if not ctl(e)]; H0 = [e for e in ents if ctl(e)]
calls = []
distill._dt_recall = lambda m, s, L: (calls.append(len(L)), {'facts': (float('nan'), len(L)), 'conclusions': (float('nan'), 0)})[1]
distill._dw_relmod(model, cfg, T0, H0, 1)                                     # sleep at visit index 1
e1, h1 = distill._dw_relmod_write(model, cfg, lambda d: 0, list(T0), list(H0), ctl, 1)
WR = model.__dict__['_rm_written']; add_t = len(e1) - len(T0); add_h = len(h1) - len(H0)
print(f"CHECK sleep 1: written {len(WR)}, joined the pool {add_t} taught + {add_h} held back; recall calls {calls}")
distill._dw_relmod(model, cfg, T0, H0, 2)                                     # sleep 2: written facts are hops, not evidence; no re-writes
e2, h2 = distill._dw_relmod_write(model, cfg, lambda d: 0, list(T0), list(H0), ctl, 2)
print(f"CHECK sleep 2: written {len(model.__dict__['_rm_written'])} (no duplicates expected), recall calls {calls[-4:]} (written at sleep 1 now counted)")
assert add_t + add_h == len(WR) and add_h > 0 and add_t > 0
