"""DOES THE EXTRACTION SIGNAL GROW AS THE TRUNK LEARNS? (13 Nov 2026; tools/extract_curve.py) - offline, no effect on the package.
The premise of trunk-guided extraction: a better trunk tells related name pairs from unrelated ones better. A byte LM of the stand-in's
size (tools/trunk_guides.py) is trained from scratch on the text of the Re-DocRED train + dev documents (the text it will be asked about,
as the model reads the benchmark), and at checkpoints the EXTRACTION GAIN of tools/extract_gain.py is measured on a fixed set of candidate
records (two names in one sentence and the words between; scored where the names meet again in another document):
  AUC related vs not - the words between beyond the names, the whole record, the names only; closeness of the names as the no-trunk baseline
  THE LABEL-FREE KEEP RULE: keep a record when its words save more bits than a NULL - the same two names with the words between taken from a
  random other candidate record; precision (related among kept) and recall (share of related records kept), against the base rate.
Usage: python3 tools/extract_curve.py [steps, default 12000] [checkpoints, default 1000,3000,6000,12000] [candidates, default 3000]
       env EC_D (width 192), EC_NL (layers 3), EC_THREADS"""
import os
import sys
import json
import math
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('EC_THREADS', 1)))
P = lambda *a: print(*a, flush=True); T0 = time.time()
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
CKS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '1000,3000,6000,12000').split(',')]
NC = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
DW = int(os.environ.get('EC_D', 192)); NL = int(os.environ.get('EC_NL', 3)); L = 128; PAD = 256; V = 257; B = 32
DIR = os.environ.get('DV_DIR', '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad/bench_dl')
docs = json.load(open(DIR + '/train_revised.json')) + json.load(open(DIR + '/dev_revised.json'))
txt = lambda toks: ' '.join(toks).encode('utf-8', 'replace')
corpus = np.frombuffer(b' '.join(txt([w for s in d['sents'] for w in s]) for d in docs), dtype=np.uint8)
TH = b'Thing'
rs = np.random.RandomState(0)

# ---- candidates (as tools/extract_gain.py)
cands = []; occ = collections.defaultdict(list)
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    gold = {(l['h'], l['t']) for l in d['labels']} | {(l['t'], l['h']) for l in d['labels']}
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms); toks = d['sents'][si]
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[y]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb]: continue
                key = tuple(sorted((cn[ea], cn[eb]))); occ[key].append((di, si, a0, a1, b0, b1))
                if b0 - a1 <= 12 and len(' '.join(toks[a0:b1])) <= 60:
                    cands.append(dict(di=di, si=si, a=(a0, a1), b=(b0, b1), key=key, gold=(ea, eb) in gold, dist=b0 - a1))
U = []
for c in cands:
    oth = [o for o in occ[c['key']] if o[0] != c['di']]
    if oth: c['tgt'] = oth[rs.randint(len(oth))]; U.append(c)
rs.shuffle(U); U = U[:NC]; y = np.array([c['gold'] for c in U])
P(f"EXTRACTION CURVE: {len(docs)} documents, corpus {len(corpus)} bytes | {len(cands)} candidate records, {len(U)} scored (where the names meet again "
  f"in another document), related {y.mean():.3f} | byte LM {DW} wide, {NL} layers, {STEPS} steps, checkpoints {CKS}")
SEQ = {k: [] for k in ('w', 'o', 'pw', 'po', 'nw', 'no')}
perm = rs.permutation(len(U))
for n, c in enumerate(U):
    d = docs[c['di']]; toks = d['sents'][c['si']]; (a0, a1), (b0, b1) = c['a'], c['b']
    r = U[int(perm[n])] if int(perm[n]) != n else U[(n + 1) % len(U)]; rt = docs[r['di']]['sents'][r['si']]; mid = txt(rt[r['a'][1]:r['b'][0]])
    di2, si2, x0, x1, y0, y1 = c['tgt']; t2 = docs[di2]['sents'][si2]; pre = txt(t2[max(0, y0 - 6):y0])[-36:] + b' '; tg = txt(t2[y0:y1])[:24]
    A, Bn = txt(toks[a0:a1]), txt(toks[b0:b1]); M = txt(toks[a1:b0])
    for k, rec in (('w', A + b' ' + M + b' ' + Bn), ('o', TH + b' ' + M + b' ' + Bn), ('pw', A + b' ' + Bn), ('po', TH + b' ' + Bn),
                   ('nw', A + b' ' + mid + b' ' + Bn), ('no', TH + b' ' + mid + b' ' + Bn)):
        q = rec.replace(b'  ', b' ') + b' . ' + pre; SEQ[k].append((q + tg, (len(q), len(q) + len(tg))))


class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))


class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, DW); s.pos = nn.Parameter(torch.randn(L, DW) * 0.02); s.b = nn.ModuleList(Blk(DW, 4) for _ in range(NL))
        s.n = nn.LayerNorm(DW); s.o = nn.Linear(DW, V)
    def forward(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.o(s.n(x))


@torch.no_grad()
def bits(m, items, bs=128):
    out = np.zeros(len(items))
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; cut = [max(0, len(s) - L) for s, _ in ch]; X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for j, (s, _) in enumerate(ch): X[j, :len(s) - cut[j]] = torch.tensor(list(s[cut[j]:]))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(m(Xi), -1)
        nl = -lp.gather(-1, X.clamp(max=V - 1)[..., None])[..., 0] / math.log(2)
        for j, ((s, (a, b)), c0) in enumerate(zip(ch, cut)): out[i + j] = float(nl[j, max(0, a - c0):max(0, b - c0)].sum())
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    _u, _i, _c = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True); r = (np.cumsum(_c) - (_c - 1) / 2.0)[_i]   # ties share their mean rank (13 Nov fix)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def measure(m, tag):
    m.eval(); b = {k: bits(m, v) for k, v in SEQ.items()}; m.train()
    g = b['o'] - b['w']; pg = b['po'] - b['pw']; ng = b['no'] - b['nw']; wb = g - pg; cl = -np.array([c['dist'] for c in U], float)
    o = np.argsort(-wb)[:len(wb) // 4]
    msg = (f"   {tag}: AUC related vs not - WORDS BETWEEN beyond the names {auc(wb[y], wb[~y]):.3f}, whole record {auc(g[y], g[~y]):.3f}, names only "
           f"{auc(pg[y], pg[~y]):.3f}, closeness {auc(cl[y], cl[~y]):.3f} | top quarter by words-between related {y[o].mean():.3f} (base {y.mean():.3f}) | "
           f"bits saved by the record, related vs not {g[y].mean():+.2f} vs {g[~y].mean():+.2f}")
    for mg in (0.0, 0.5, 1.0):
        keep = (g - ng) > mg
        msg += f"\n      KEEP RULE (own words beat another record's words by > {mg:.1f} bits): kept {keep.mean():.3f}, precision {y[keep].mean() if keep.any() else float('nan'):.3f}, recall {keep[y].mean():.3f}"
    P(msg + f"  ({time.time() - T0:.0f}s)")


torch.manual_seed(0); m = LM(); op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
sch = torch.optim.lr_scheduler.OneCycleLR(op, 1e-3, total_steps=STEPS, pct_start=0.05); ls = []; r2 = np.random.RandomState(2)
measure(m, "untrained")
for st in range(STEPS):
    i0 = r2.randint(0, len(corpus) - L - 1, B); X = torch.tensor(np.stack([corpus[i:i + L] for i in i0]).astype(np.int64))
    Xi = torch.cat([torch.full((B, 1), PAD), X[:, :-1]], 1); l = F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1))
    op.zero_grad(); l.backward(); op.step(); sch.step(); ls.append(l.item())
    if st + 1 in CKS: measure(m, f"step {st + 1} ({np.mean(ls[-200:]) / math.log(2):.2f} bits/byte)")
P(f"DONE ({time.time() - T0:.0f}s)")
