"""DOES A TRUNK-SIZED TRANSFORMER LEARN TO COMPOSE FACTS WITHOUT RULES? (13 Nov 2026; tools/compose_test.py) - offline, no effect on the package.
Rule-free inference inside the weights, in the setting of Wang et al. 2024 ('Grokked transformers are implicit reasoners'): E entities, R
relations, each entity with K atomic facts (h, r) -> t; two-hop facts (h, r1, r2) -> t through the bridge b = t(h, r1). Trained on ALL
atomic facts plus a share of the two-hop facts (ratio PHI = two-hop / atomic in training); tested on HELD-OUT two-hop facts whose atomic
facts were all seen (ID), and on two-hop facts from OOD entities (their atomic facts seen, no two-hop fact about them ever trained).
Wang: ID composition generalises only after long training past fitting (grokking), faster with larger PHI; OOD never. A token-level model
of the stand-in trunk's size (width 128, 4 layers), AdamW weight decay 0.1, batch 512. PHI 0 is the control (no composition taught).
Readouts every EVAL steps: atomic train accuracy, two-hop train, two-hop ID HELD OUT, two-hop OOD.
Usage: python3 tools/compose_test.py [steps, default 40000] [phis, default 0,1,3.6,7.2]   env CT_E (400), CT_R (20), CT_K (8), CT_THREADS"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('CT_THREADS', 1)))
P = lambda *a: print(*a, flush=True); T0 = time.time()
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
PHIS = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '0,1,3.6,7.2').split(',')]
E = int(os.environ.get('CT_E', 400)); R = int(os.environ.get('CT_R', 20)); K = int(os.environ.get('CT_K', 8)); EVAL = int(os.environ.get('CT_EVAL', 4000))
DW, NL, BS = 128, 4, 512
rs = np.random.RandomState(0)
OOD = set(rs.choice(E, E // 20, replace=False).tolist())
ATOM = {}
for h in range(E):
    for r in rs.choice(R, K, replace=False): ATOM[(h, int(r))] = int(rs.randint(E))
TWO = {}
for (h, r1), b in ATOM.items():
    for r2 in range(R):
        if (b, r2) in ATOM: TWO[(h, r1, r2)] = ATOM[(b, r2)]
ID2 = [k for k in TWO if k[0] not in OOD]; OOD2 = [k for k in TWO if k[0] in OOD]
rs.shuffle(ID2); TEST = ID2[:2000]; POOL2 = ID2[2000:]
NONE = E + R; VOC = E + R + 1
P(f"COMPOSITION TEST: {E} entities ({len(OOD)} OOD), {R} relations, {K} facts each -> {len(ATOM)} atomic, {len(TWO)} two-hop ({len(TEST)} ID held out, "
  f"{len(POOL2)} trainable, {len(OOD2)} OOD); model {DW} wide, {NL} layers, {STEPS} steps, batch {BS}")


class M(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(VOC, DW); s.pos = nn.Parameter(torch.randn(3, DW) * 0.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(DW, 4, 4 * DW, 0.0, batch_first=True, norm_first=True), NL)
        s.n = nn.LayerNorm(DW); s.o = nn.Linear(DW, E)
    def forward(s, X): return s.o(s.n(s.enc(s.emb(X) + s.pos)[:, -1]))


def tens(keys):
    X = torch.tensor([[h, E + r1, (E + r2) if r2 is not None else NONE] for h, r1, r2 in keys]); return X


def acc(m, keys, truth):
    if not keys: return float('nan')
    with torch.no_grad():
        out = []
        for i in range(0, len(keys), 4096): out.append(m(tens(keys[i:i + 4096])).argmax(-1))
        return float((torch.cat(out) == torch.tensor(truth)).float().mean())


for phi in PHIS:
    n2 = min(len(POOL2), int(round(phi * len(ATOM)))); tr2 = POOL2[:n2]
    atoms = [(h, r, None) for (h, r) in ATOM]; ya = [ATOM[(h, r)] for h, r, _ in atoms]
    keys = atoms + tr2; ys = ya + [TWO[k] for k in tr2]; Xall = tens(keys); Yall = torch.tensor(ys)
    torch.manual_seed(0); m = M(); op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1); r2 = np.random.RandomState(1)
    P(f"\n== PHI {phi}: {len(atoms)} atomic + {len(tr2)} two-hop facts in training")
    oodk = OOD2[:2000]
    for st in range(STEPS):
        ix = torch.tensor(r2.randint(0, len(keys), BS)); l = F.cross_entropy(m(Xall[ix]), Yall[ix])
        op.zero_grad(); l.backward(); op.step()
        if (st + 1) % EVAL == 0:
            m.eval()
            P(f"   step {st + 1}: loss {l.item():.3f} | atomic {acc(m, atoms, ya):.3f} | two-hop TRAIN {acc(m, tr2[:4000], [TWO[k] for k in tr2[:4000]]):.3f} | "
              f"two-hop ID HELD OUT {acc(m, TEST, [TWO[k] for k in TEST]):.3f} | OOD {acc(m, oodk, [TWO[k] for k in oodk]):.3f}  ({time.time() - T0:.0f}s)")
            m.train()
P(f"DONE ({time.time() - T0:.0f}s)")
