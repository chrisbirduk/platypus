"""STAND-IN TRUNK WITH A PRETRAINING STAGE (15 Nov 2026; tools/pretrain_lm.py) - offline, no effect on the package.
The same byte LM as tools/trunk_guides.py (d TG_D, 3 blocks, windows of 128) trained in STAGES, each a byte file and a number of steps with
its own one-cycle schedule, continuing from the last: e.g. TinyStories (basic English, no encyclopedic facts) then the Pile's Wikipedia,
against Wikipedia alone for the same or the same total steps. The checkpoint loads in every tool that takes TG_CKPT.
Usage: TG_D=192 python3 tools/pretrain_lm.py <out.pt> <file>:<steps> [<file>:<steps> ...]   (files: raw text; newlines become spaces)"""
import os, sys, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(int(os.environ.get('PT_THREADS', 2)))
DW = int(os.environ.get('TG_D', 192)); NL = 3; L = 128; PAD = 256; V = 257
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=DW, nl=NL):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(L, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl))
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V)
    def forward(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.o(s.n(x))
out = sys.argv[1]; stages = [(a.rsplit(':', 1)[0], int(a.rsplit(':', 1)[1])) for a in sys.argv[2:]]
torch.manual_seed(0); rs = np.random.RandomState(0); m = LM()
ev = {}
for path, steps in stages:
    raw = open(path, 'rb').read().replace(b'\n', b' '); corpus = np.frombuffer(raw[:-200000], dtype=np.uint8); ev[path] = np.frombuffer(raw[-200000:], dtype=np.uint8)
    op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(op, 1e-3, total_steps=max(1, steps), pct_start=0.05); ls = []
    m.train()
    for st in range(steps):
        i0 = rs.randint(0, len(corpus) - L - 1, 32); X = torch.tensor(np.stack([corpus[i:i + L] for i in i0]).astype(np.int64))
        Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); l = F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1))
        op.zero_grad(); l.backward(); op.step(); sch.step(); ls.append(l.item())
        if (st + 1) % 1000 == 0: P(f"   {os.path.basename(path)} step {st + 1}/{steps}: {np.mean(ls[-200:]) / math.log(2):.3f} bits per byte  ({time.time() - T0:.0f}s)")
m.eval()
with torch.no_grad():
    for path, e in ev.items():
        X = torch.tensor(np.stack([e[k * L:(k + 1) * L] for k in range(len(e) // L)][:400]).astype(np.int64))
        Xi = torch.cat([torch.full((len(X), 1), PAD), X[:, :-1]], 1); b = float(F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1))) / math.log(2)
        P(f"HELD-OUT {os.path.basename(path)} (its last 200 KB): {b:.3f} bits per byte")
torch.save(m.state_dict(), out); P(f"saved {out}  ({time.time() - T0:.0f}s)")
