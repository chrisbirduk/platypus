"""SUBSTITUTION READER (19 Nov 2026; tools/sub_reader.py) - offline, distant set, LABEL-FREE (distant labels for evaluation only).
The reader's records are name pairs within 12 words of one sentence (AUC 0.62 label-free by the words between them). Here the TRUNK's
own reading judges a record: how much the first name helps it predict the second. For a pair (X before Y in a sentence), the trunk's
surprise on Y's bytes after the sentence prefix, with X as written vs with X replaced by a random name of the same type from another
document (SB_SWAPS swaps). A relational pair should lose more when X is swapped (Y depends on X); a list or chance pair should not.
Scores: the surprise GAIN in bits (swapped - original), per byte and summed; and combined with distance and cross-document agreement
(ranks summed, label-free). AUC and precision on records from SB_DOCS documents, against distance and agreement.
Usage: TG_D=192 TG_CKPT=<stand-in trunk> python3 tools/sub_reader.py <distant.json.gz>   env SB_DOCS (3000) SB_SWAPS (2) SB_THREADS (2)"""
import os, sys, gzip, json, math, time, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
T0 = time.time(); P = lambda s: print(s, flush=True); torch.set_num_threads(int(os.environ.get('SB_THREADS', 2)))
L = 128; PAD = 256; V = 257; DW = int(os.environ.get('TG_D', 192)); NL = int(os.environ.get('TG_NL', 3))
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
    def t(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for b in s.b: x = b(x, am)
        return s.n(x)
    def forward(s, X): return s.o(s.t(X))
lm = LM(); lm.load_state_dict(torch.load(os.environ['TG_CKPT'])); lm.eval()
DOCS = json.load(gzip.open(sys.argv[1])); ND = len(DOCS); rs = np.random.RandomState(0)
NOSP_B = {',', '.', ';', ':', ')', "'s", "n't", '%', '!', '?', "'", "''"}; NOSP_A = {'(', '$', '``'}
def detok(ws):
    out, prev = '', None
    for w in ws:
        if out and not (w in NOSP_B or prev in NOSP_A): out += ' '
        out += w; prev = w
    return out
LAB = set(); bytype = collections.defaultdict(list); pdocs = collections.defaultdict(set); recs = []
for di, d in enumerate(DOCS):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']: LAB.add(frozenset((cn[l['h']], cn[l['t']])))
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei, m.get('type', 'X'))); bytype[m.get('type', 'X')].append(m['name'])
    for si, ms in bys.items():
        ms = sorted(ms)
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                (a0, a1, ea, ta), (b0, b1, eb, tb) = ms[x], ms[y]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb] or b0 - a1 > 12: continue
                pdocs[frozenset((cn[ea], cn[eb]))].add(di)
                if di < int(os.environ.get('SB_DOCS', 3000)): recs.append((di, si, a0, a1, b0, b1, ta, cn[ea], cn[eb], b0 - a1))
P(f"{ND} documents; {len(recs)} records from the first {os.environ.get('SB_DOCS', 3000)} documents  ({time.time() - T0:.0f}s)")
def seq(d, si, a0, a1, b0, b1, xrep=None):
    """the sentence prefix up to and including Y, with X replaced by xrep: (bytes, start of Y's bytes)"""
    ws = d['sents'][si]; pre = ws[:a0] + (list(xrep) if xrep is not None else ws[a0:a1]) + ws[a1:b0]
    p = detok(pre); full = detok(pre + ws[b0:b1]); pb = p.encode('utf-8', 'replace'); fb = full.encode('utf-8', 'replace')
    ys = len(pb) + (1 if len(fb) > len(pb) and fb[len(pb):len(pb) + 1] == b' ' else 0); cut = max(0, len(fb) - L)
    return fb[cut:], ys - cut
@torch.no_grad()
def surprise(items):
    """mean and summed bits over Y's bytes for each (bytes, Y start)"""
    out = []
    for i0 in range(0, len(items), 256):
        ch = items[i0:i0 + 256]; X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for k, (b, ys) in enumerate(ch): X[k, :len(b)] = torch.tensor(list(b))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); nl = -F.log_softmax(lm(Xi), -1).gather(-1, X[..., None])[..., 0] / math.log(2)
        for k, (b, ys) in enumerate(ch):
            v = nl[k, max(0, ys):len(b)]; out.append((float(v.mean()) if len(v) else 0.0, float(v.sum())))
    return out
NS = int(os.environ.get('SB_SWAPS', 2)); orig, sw = [], [[] for _ in range(NS)]
for (di, si, a0, a1, b0, b1, ta, A, B, dist) in recs:
    d = DOCS[di]; orig.append(seq(d, si, a0, a1, b0, b1))
    for j in range(NS):
        pool = bytype.get(ta) or bytype['X']; r_ = pool[rs.randint(len(pool))]
        while r_ == A: r_ = pool[rs.randint(len(pool))]
        sw[j].append(seq(d, si, a0, a1, b0, b1, r_.split()))
so = surprise(orig); ss = [surprise(x) for x in sw]
P(f"surprise computed for {len(recs)} records x {NS + 1} versions  ({time.time() - T0:.0f}s)")
y = np.array([frozenset((r[7], r[8])) in LAB for r in recs], float); dist = np.array([r[9] for r in recs], float)
agr = np.array([len(pdocs[frozenset((r[7], r[8]))] - {r[0]}) for r in recs], float)
gm = np.mean([[s[k][0] - so[k][0] for k in range(len(recs))] for s in ss], 0); gs = np.mean([[s[k][1] - so[k][1] for k in range(len(recs))] for s in ss], 0)
P(f"base rate {y.mean():.3f}")
def rep(nm, s):
    o = np.argsort(-s); P(f"   {nm:62s} AUC {roc_auc_score(y, s):.3f} | precision top 5% {y[o[:len(o) // 20]].mean():.3f}, top 20% {y[o[:len(o) // 5]].mean():.3f}")
rk = lambda v: np.argsort(np.argsort(v)).astype(float)
rep('distance only (nearer = better)', -dist)
rep('cross-document agreement', np.log1p(agr))
rep('SUBSTITUTION GAIN, bits per byte of Y', gm)
rep('SUBSTITUTION GAIN, bits summed over Y', gs)
rep('original surprise of Y (lower = better; control)', -np.array([x[0] for x in so]))
rep('GAIN (per byte) + distance (ranks)', rk(gm) + rk(-dist))
rep('GAIN (per byte) + distance + agreement (ranks)', rk(gm) + rk(-dist) + rk(np.log1p(agr)))
rep('distance + agreement (ranks; no trunk)', rk(-dist) + rk(np.log1p(agr)))
P(f"DONE ({time.time() - T0:.0f}s)")
