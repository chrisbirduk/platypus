"""CONTRADICTION BY SURPRISE: CAN THE TRUNK FLAG A WRONG TAUGHT CLAIM WHEN LATER TEXT DISAGREES? (14 Nov 2026; tools/surprise_trigger.py)
Offline, no effect on the package. The idea (the author's feedback loop): the store teaches TYPED claims ('A was born in C'); the trunk then
EXPECTS C after A in that relation; when later text states another object, the trunk is SURPRISED - and that surprise is the retraction
signal (and, per type, a check on the type). Untyped claims ('A is connected to C') never collide with real text, which is why every trigger
for them failed (tools/trigger_retest.py). The stand-in trunk (TG_CKPT) is taught, in the system's question frames at a high dose:
  RIGHT   (A, r, C) - the gold relation's phrase and object
  WRONG   (A, r, C') - the same phrase, a random name of C's type
  CONTROL untaught pairs
in two arms that differ only in the phrase: GOLD PHRASE (typed) and 'is connected to' (untyped). LATER READING: each pair's real source
sentence from Re-DocRED (A's mention before C's; the true object C, in the document's own words). Signal: the trunk's surprise at C's bytes
in that sentence, after teaching minus before (a wrong claim should make the true object MORE surprising, a right one less). AUC right vs
wrong (wrong = higher surprise change), and the flagged share at a threshold set on the untaught control (its 90th percentile).
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/surprise_trigger.py   env ST_N (120 per group), ST_STEPS (400), TD_SEED (0)"""
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
NG = int(os.environ.get('ST_N', 120)); ST = int(os.environ.get('ST_STEPS', 400))
def build(docs):
    out, used, byt = [], set(), collections.defaultdict(set)
    for d in docs:
        for n, t in zip(cn_(d), tp_(d)): byt[t].add(n)
    for di in rs.permutation(len(docs)):
        d = docs[int(di)]; cn, tp = cn_(d), tp_(d)
        for l in d['labels']:
            if l['r'] not in REL: continue
            A, C = cn[l['h']], cn[l['t']]
            if A in used or not (3 <= len(A) <= 30 and 3 <= len(C) <= 30) or not all(ord(ch) < 256 for ch in A + C): continue
            hit = None
            for ma in d['vertexSet'][l['h']]:
                for mc in d['vertexSet'][l['t']]:
                    if ma['sent_id'] == mc['sent_id'] and ma['pos'][1] <= mc['pos'][0]: hit = (ma, mc); break
                if hit: break
            if not hit: continue
            ma, mc = hit; toks = d['sents'][ma['sent_id']]
            pre = ' '.join(toks[:mc['pos'][0]]); obj = ' '.join(toks[mc['pos'][0]:mc['pos'][1]])
            sent = (pre + ' ' + obj + ' ' + ' '.join(toks[mc['pos'][1]:])).strip()
            pb = (pre + ' ').encode('latin-1', 'replace')[-(L - 40):]; ob = obj.encode('latin-1', 'replace')[:36]
            if len(pb) < 8 or len(ob) < 2: continue
            out.append(dict(A=A, C=C, rel=REL[l['r']], t=tp[l['t']], pb=pb, ob=ob)); used.add(A); break
        if len(out) >= 3 * NG: break
    return out, byt
FACTS, BYT = build(tr_docs); rs.shuffle(FACTS); RIGHT, WRONG, CTL = FACTS[:NG], [dict(f) for f in FACTS[NG:2 * NG]], FACTS[2 * NG:3 * NG]
for f in WRONG:
    alts = sorted(n for n in BYT[f['t']] if n != f['C'] and 3 <= len(n) <= 30 and all(ord(ch) < 256 for ch in n)); f['taught'] = alts[rs.randint(len(alts))]
for f in RIGHT: f['taught'] = f['C']
@torch.no_grad()
def surprise(m, facts):
    m.eval(); out = []
    for i in range(0, len(facts), 32):
        ch = facts[i:i + 32]; seqs = [f['pb'] + f['ob'] for f in ch]; X = tens(seqs); Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1)
        nl = -F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[..., 0] / math.log(2)
        for j, f in enumerate(ch): a = len(f['pb']); out.append(float(nl[j, a:a + len(f['ob'])].sum()))
    m.train(); return np.array(out)
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    _u, _i, _c = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True); r = (np.cumsum(_c) - (_c - 1) / 2.0)[_i]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
P(f"pairs with a later sentence (A before C): right {len(RIGHT)}, wrong {len(WRONG)}, control {len(CTL)} | relations " + ", ".join(f"{k} {v}" for k, v in collections.Counter(f['rel'] for f in FACTS).most_common(6)))
for arm, phr in (('GOLD PHRASE (typed)', lambda f: f['rel']), ("'is connected to' (untyped)", lambda f: 'is connected to')):
    m = LM(); m.load_state_dict(torch.load(CK)); m.train(); op = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); rr = np.random.RandomState(SEED + 3)
    b = {k: surprise(m, g) for k, g in (('R', RIGHT), ('W', WRONG), ('C', CTL))}
    T_ = RIGHT + WRONG
    for st in range(ST):
        seqs = [(FORMS[rr.randint(len(FORMS))].format(A=f['A'], r=phr(f)) + f['taught'] + '.').encode('latin-1', 'replace') for f in (T_[rr.randint(len(T_))] for _ in range(16))]
        seqs += [bytes(WK[i0:i0 + L]) for i0 in rr.randint(0, len(WK) - L - 1, 16)]
        X = tens(seqs); Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg = X.reshape(-1)
        l = F.cross_entropy(lo[tg != PAD], tg[tg != PAD]); op.zero_grad(); l.backward(); op.step()
    a = {k: surprise(m, g) for k, g in (('R', RIGHT), ('W', WRONG), ('C', CTL))}
    d = {k: a[k] - b[k] for k in a}; thr = np.quantile(d['C'], 0.9)
    P(f"{arm}: surprise at the true object in the later sentence, change after teaching (bits): right {d['R'].mean():+.2f}, wrong {d['W'].mean():+.2f}, "
      f"control {d['C'].mean():+.2f} || AUC wrong vs right (wrong more surprised) {auc(d['W'], d['R']):.3f} | flagged above the control's 90th percentile: "
      f"wrong {(d['W'] > thr).mean():.2f}, right {(d['R'] > thr).mean():.2f}  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
