"""ARE HELD-BACK RIGHT INFERENCES RANKED ABOVE CHANCE BY GENERALISATION OR BY PRIOR? (14 Nov 2026; tools/gen_diag.py) - offline.
GPU cg27a_rcode3: rank among itself and 9 held-back answers (chance 0.10), held back RIGHT 0.18 / 0.24 / 0.22 at visits 3-5 against
held back WRONG 0.10 / 0.10 / 0.15. Either the trunk generalises from the taught inferences, or it already preferred the right answers
(a prior from what it read, e.g. familiar or fitting names). The stand-in trunk (TG_CKPT) gets 4 x 90 pairs: TAUGHT right / wrong and
HELD BACK right / wrong (wrong = A with a random name of the answer's type); the taught ones in the system's question frames at a high
dose. Rank of each pair's answer, BEFORE and AFTER teaching, two ways: among 9 random names of its type, and as the GPU readout does,
among itself and 9 held-back answers. If held-back right is already above held-back wrong BEFORE teaching, it is a prior.
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/gen_diag.py   env TD_SEED (0), GD_STEPS (400)"""
import os
import json
import math
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
P = lambda *a: print(*a, flush=True); T0 = time.time()
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
DW = int(os.environ.get('TG_D', 192)); NL = int(os.environ.get('TG_NL', 3)); CK = os.environ['TG_CKPT']; torch.set_num_threads(1)
L = 128; PAD = 256; V = 257; SEED = int(os.environ.get('TD_SEED', 0)); rs = np.random.RandomState(SEED); torch.manual_seed(SEED)
FORMS = ("In short, {A} {r} ", "Notes on {A}: {A} {r} ", "{A}, as noted before, {r} ", "As for {A}, people say {A} {r} ",
         "Q: {A} {r} whom? A: ", "{A}, whose name came up earlier in the report, {r} ", "From the notes: {A} {r} ")
REL = {'P17': 'is in the country', 'P131': 'is located in', 'P27': 'is a citizen of', 'P19': 'was born in', 'P20': 'died in', 'P108': 'worked for',
       'P54': 'played for', 'P463': 'is a member of', 'P102': 'is a member of the party', 'P175': 'was performed by', 'P495': 'comes from',
       'P159': 'is based in', 'P50': 'was written by', 'P57': 'was directed by', 'P361': 'is part of', 'P150': 'contains', 'P527': 'includes',
       'P740': 'was formed in', 'P276': 'took place in', 'P26': 'was married to', 'P69': 'was educated at'}
tr_docs = json.load(open(SCR + '/bench_dl/train_revised.json')); dv_docs = json.load(open(SCR + '/bench_dl/dev_revised.json'))
cn_ = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
tp_ = lambda d: [vs[0].get('type', 'X') for vs in d['vertexSet']]
dev_names = {n for d in dv_docs for n in cn_(d)}
def pairs(docs, familiar):
    out, used, byt = [], set(), collections.defaultdict(set)
    for d in docs:
        for n, t in zip(cn_(d), tp_(d)): byt[t].add(n)
    for di in rs.permutation(len(docs)):
        d = docs[int(di)]; cn, tp = cn_(d), tp_(d)
        for l in d['labels']:
            A, C = cn[l['h']], cn[l['t']]
            if l['r'] not in REL or A in used or not (3 <= len(A) <= 30 and 3 <= len(C) <= 30) or not all(ord(ch) < 256 for ch in A + C): continue
            if familiar != (A in dev_names and C in dev_names): continue
            sent = next((' '.join(s) for si, s in enumerate(d['sents']) if any(m['sent_id'] == si for m in d['vertexSet'][l['h']]) and any(m['sent_id'] == si for m in d['vertexSet'][l['t']])), None)
            out.append(dict(A=A, C=C, rel=REL[l['r']], t=tp[l['t']], sent=sent)); used.add(A); break
        if len(out) >= 360: break
    return out, byt
wiki = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().replace(b'\n', b' '); WK = np.frombuffer(wiki, dtype=np.uint8)
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
def tens(seqs):
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
    return X
@torch.no_grad()
def recall(m, facts, rel_of, byt):
    m.eval(); oe, rk = [], []; rr = np.random.RandomState(5)
    for f in facts:
        pb = f"It is on record that {f['A']} {rel_of(f)} ".encode('latin-1'); tb = (f['C'] + '.').encode('latin-1')
        X = tens([pb + tb]); Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1); lg = m(Xi)[0]
        ps = torch.arange(len(pb), len(pb) + len(tb)); top = lg[ps].argmax(-1); oe.append(int(bool((top[:-1] == X[0, ps][:-1]).all())))
        alts = sorted(n for n in byt[f['t']] if n != f['C'] and 3 <= len(n) <= 30 and all(ord(ch) < 256 for ch in n))
        cands = [f['C']] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)]
        seqs = [pb + c.encode('latin-1') + b'.' for c in cands]; X = tens(seqs); Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1)
        nl = -F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[..., 0]; sc = [float(nl[k, len(pb):len(q) - 1].mean()) for k, q in enumerate(seqs)]
        rk.append(int(np.argmin(sc) == 0))
    m.train(); return f"open-ended {np.mean(oe):.2f} | rank {np.mean(rk):.2f}"

ST = int(os.environ.get('GD_STEPS', 400))
FACTS, BYT = pairs(tr_docs, False); rs.shuffle(FACTS); n4 = len(FACTS) // 4
TR, TW, HR, HW_ = FACTS[:n4], [dict(f) for f in FACTS[n4:2 * n4]], FACTS[2 * n4:3 * n4], [dict(f) for f in FACTS[3 * n4:4 * n4]]
for f in TW + HW_:
    alts = sorted(n for n in BYT[f['t']] if n != f['C'] and 3 <= len(n) <= 30 and all(ord(ch) < 256 for ch in n)); f['C'] = alts[rs.randint(len(alts))]
CONN = lambda f: 'is connected to'; HPOOL = sorted({f['C'] for f in HR + HW_})
@torch.no_grad()
def rank_pool(m, facts):
    m.eval(); rr = np.random.RandomState(11); hit = []
    for f in facts:
        pb = f"It is on record that {f['A']} is connected to ".encode('latin-1'); al = [c for c in HPOOL if c != f['C']]
        cs = [f['C']] + [al[j] for j in rr.choice(len(al), 9, replace=False)]; seqs = [pb + (c + '.').encode('latin-1') for c in cs]
        X = tens(seqs); Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); nl = -F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[..., 0]
        hit.append(int(np.argmin([float(nl[k, len(pb):len(q) - 1].mean()) for k, q in enumerate(seqs)]) == 0))
    m.train(); return np.mean(hit)
def rep(m, tag):
    a = " | ".join(f"{nm} {recall(m, fs, CONN, BYT).split('rank ')[1]}" for nm, fs in (('taught R', TR), ('taught W', TW), ('held R', HR), ('held W', HW_)))
    b = " | ".join(f"{nm} {rank_pool(m, fs):.2f}" for nm, fs in (('taught R', TR), ('taught W', TW), ('held R', HR), ('held W', HW_)))
    P(f"{tag}: among 9 of its type: {a} || among held-back answers (as the GPU readout): {b}  ({time.time() - T0:.0f}s)")
np.random.seed(SEED); m = LM(); m.load_state_dict(torch.load(CK)); m.train(); op = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
P(f"pairs: 4 x {n4} (unfamiliar names), held-back answer pool {len(HPOOL)}; chance 0.10")
rep(m, "BEFORE TEACHING")
T_ = TR + TW
for st in range(ST):
    seqs = [(FORMS[np.random.randint(len(FORMS))].format(A=f['A'], r='is connected to') + f['C'] + '.').encode('latin-1') for f in (T_[np.random.randint(len(T_))] for _ in range(16))]
    seqs += [bytes(WK[i0:i0 + L]) for i0 in np.random.randint(0, len(WK) - L - 1, 16)]
    X = tens(seqs); Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg = X.reshape(-1)
    l = F.cross_entropy(lo[tg != PAD], tg[tg != PAD]); op.zero_grad(); l.backward(); op.step()
rep(m, f"AFTER TEACHING ({ST} steps, {ST * 16 / len(T_):.0f} rows per taught pair)")
P(f"DONE ({time.time() - T0:.0f}s)")
