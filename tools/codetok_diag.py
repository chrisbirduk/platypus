"""CODES AS TOKENS ON REAL TEXT (14 Nov 2026; tools/codetok_diag.py) - offline, no effect on the package.
Synthetic world (tools/compose_synth.py CS_PCODE 2): a trunk that reads compression codes as tokens after names infers held-out facts in one
hop. Here the same on Re-DocRED. Each name gets a code from the store's own graph (names x names, linked when they share a sentence of a
kept document; PPMI, SVD, k-means into RD_K clusters - no labels). Held-out documents (RD_NT) are never read, and a candidate pair is a
gold-related pair of a held-out document whose names never share a sentence anywhere in the kept documents (nothing states it).
The stand-in trunk (TG_CKPT; never read these documents) is trained on the kept documents' sentences and on 'A is connected to C' question
rows for names that share a kept sentence, in two arms that differ only in the cluster token after each name ('Paris [k017]'):
  TOKENS  every name in sentences and rows is followed by its cluster token
  RANDOM  the same tokens shuffled across names (same text added, no code information; RD_ARMS=RANDOM)
  NONE    the same text without tokens
Readout ('It is on record that A [kA] is connected to C [kC].', tokens only in the TOKENS arm): RANK of C among it and 9 names of its
type whose store degree is in the same quintile (popularity matched), for the held-out pairs (never stated: the inference) and for
stated pairs (taught); the code itself (cosine, no trunk) on the same candidates; and text loss on held wiki (cost, no tokens).
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/codetok_diag.py   env RD_NT (300), RD_K (128), RD_STEPS (1500), TD_SEED (0)"""
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
NT = int(os.environ.get('RD_NT', 300)); KC = int(os.environ.get('RD_K', 128)); STEPS = int(os.environ.get('RD_STEPS', 1500))
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
docs = tr_docs; HOLD = set(rs.permutation(len(docs))[:NT].tolist())
ok_n = lambda n: 3 <= len(n) <= 30 and all(ord(ch) < 256 for ch in n)
NI = {}; E = collections.Counter(); TY = {}; GREL = set(); SENTS = []; DEG = collections.Counter()
for di, d in enumerate(docs):
    cn, tp = cn_(d), tp_(d)
    for n, t in zip(cn, tp): TY.setdefault(n, t)
    for l in d['labels']:
        if cn[l['h']] != cn[l['t']]: GREL.add(frozenset((cn[l['h']], cn[l['t']])))
    if di in HOLD: continue
    for si, sw in enumerate(d['sents']):
        ms = [(m['pos'][0], m['pos'][1], cn[e]) for e, vs in enumerate(d['vertexSet']) for m in vs if m['sent_id'] == si and ok_n(cn[e])]
        ns = sorted({x[2] for x in ms})
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)): E[(ns[i], ns[j])] += 1
        if ms: SENTS.append((sw, ms))
for (a, b) in E: NI.setdefault(a, len(NI)); NI.setdefault(b, len(NI)); DEG[a] += 1; DEG[b] += 1
NN = len(NI); r_ = np.array([NI[a] for a, b in E]); c_ = np.array([NI[b] for a, b in E])
G = csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(NN, NN)); G = G + G.T; deg = np.asarray(G.sum(1)).ravel(); Gc = G.tocoo()
v = np.log(G.sum() * Gc.data / (deg[Gc.row] * deg[Gc.col])); mk = v > 0
U = normalize(TruncatedSVD(128, random_state=0).fit_transform(csr_matrix((v[mk], (Gc.row[mk], Gc.col[mk])), shape=G.shape)))
CL = MiniBatchKMeans(KC, random_state=0, n_init=3, batch_size=4096).fit_predict(U)
CLR = np.random.RandomState(1234).permutation(CL) if True else CL                    # RANDOM arm: the same tokens, shuffled across names (each name keeps one)
TOKMODE = ['real']
TOK = lambda n: (f" [k{(CL if TOKMODE[0] == 'real' else CLR)[NI[n]]:03d}]" if n in NI else '')
# held-out candidates: gold pairs of held-out documents, both names coded, never sharing a kept sentence
CAND = []
for di in sorted(HOLD):
    d = docs[di]; cn = cn_(d)
    for l in d['labels']:
        A, C = cn[l['h']], cn[l['t']]
        if A != C and A in NI and C in NI and ok_n(A) and ok_n(C) and tuple(sorted((A, C))) not in E: CAND.append((A, C))
CAND = sorted(set(CAND)); rs.shuffle(CAND); CAND = CAND[:200]
STATED = [k for k in E if frozenset(k) in GREL]; rs.shuffle(STATED); STATED = STATED[:200]
QP = list(E.keys())
qs = np.quantile(np.array([DEG[n] for n in NI]), [0.2, 0.4, 0.6, 0.8]); QB = lambda n: int(np.searchsorted(qs, DEG[n]))
BYQ = collections.defaultdict(list)
for n in NI:
    if ok_n(n): BYQ[(TY.get(n, 'X'), QB(n))].append(n)
def cands(A, C, rr):
    pool = [n for n in BYQ[(TY.get(C, 'X'), QB(C))] if n != C and n != A and frozenset((A, n)) not in GREL]
    if len(pool) < 9: return None
    return [C] + [pool[j] for j in rr.choice(len(pool), 9, replace=False)]
def sent_text(sw, ms, tok):
    ins = collections.defaultdict(str)
    for a, b, n in ms:
        if tok: ins[b - 1] = TOK(n)
    return ' '.join(w + ins.get(i, '') for i, w in enumerate(sw)).encode('latin-1', 'replace')
P(f"documents: kept {len(docs) - NT}, held out {NT} | names coded {NN}, links {len(E)}, clusters {KC} | held-out candidates {len(CAND)}, stated {len(STATED)} | kept sentences {len(SENTS)}  ({time.time() - T0:.0f}s)")
HW = [bytes(WK[i0:i0 + L]) for i0 in np.random.RandomState(99).randint(0, len(WK) - L - 1, 128)]
def tens2(seqs):
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
    return X
@torch.no_grad()
def rank(m, pairs, tok):
    m.eval(); rr = np.random.RandomState(5); hit, hc, n = 0, 0, 0
    for A, C in pairs:
        cs = cands(A, C, rr)
        if cs is None: continue
        pb = f"It is on record that {A}{TOK(A) if tok else ''} is connected to ".encode('latin-1')
        seqs = [pb + (c + (TOK(c) if tok else '') + '.').encode('latin-1') for c in cs]; X = tens2(seqs); Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1)
        nl = -F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[..., 0]; sc = [float(nl[k, len(pb):len(q) - 1].mean()) for k, q in enumerate(seqs)]
        hit += int(np.argmin(sc) == 0); n += 1; cs_ = [float(U[NI[A]] @ U[NI[c]]) for c in cs]; hc += int(np.argmax(cs_) == 0)
    m.train(); return hit / max(1, n), hc / max(1, n), n
@torch.no_grad()
def wiki_loss(m):
    m.eval(); X = tens2(HW); Xi = torch.cat([torch.full((len(HW), 1), PAD), X[:, :-1]], 1); l = F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1)); m.train(); return float(l) / math.log(2)
FORMS = ("In short, {A} is connected to ", "Notes on {A}: {A} is connected to ", "{A}, as noted before, is connected to ", "Q: {A} is connected to whom? A: ", "From the notes: {A} is connected to ")
ARMS = os.environ.get('RD_ARMS', 'TOKENS,NONE').split(',')
for arm in ARMS:
    tok = arm != 'NONE'; TOKMODE[0] = 'random' if arm == 'RANDOM' else 'real'
    m = LM(); m.load_state_dict(torch.load(CK)); m.train(); op = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); rr = np.random.RandomState(SEED + 3)
    h0 = rank(m, CAND, tok); s0 = rank(m, STATED, tok)
    P(f"{arm}: before training - held-out (never stated) rank {h0[0]:.3f} (code itself {h0[1]:.3f}, n {h0[2]}) | stated rank {s0[0]:.3f} (code {s0[1]:.3f}, n {s0[2]}) | wiki {wiki_loss(m):.3f} bits  ({time.time() - T0:.0f}s)")
    for st in range(1, STEPS + 1):
        seqs = []
        for _ in range(12):
            a, b = QP[rr.randint(len(QP))]
            if rr.rand() < 0.5: a, b = b, a
            seqs.append((FORMS[rr.randint(len(FORMS))].format(A=a + (TOK(a) if tok else '')) + b + (TOK(b) if tok else '') + '.').encode('latin-1'))
        for _ in range(12): sw, ms = SENTS[rr.randint(len(SENTS))]; seqs.append(sent_text(sw, ms, tok))
        seqs += [bytes(WK[i0:i0 + L]) for i0 in rr.randint(0, len(WK) - L - 1, 8)]
        X = tens2(seqs); Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg = X.reshape(-1)
        l = F.cross_entropy(lo[tg != PAD], tg[tg != PAD]); op.zero_grad(); l.backward(); op.step()
        if st % 500 == 0 or st == STEPS:
            h = rank(m, CAND, tok); s_ = rank(m, STATED, tok)
            P(f"{arm}: step {st} - held-out (never stated) rank {h[0]:.3f} (code itself {h[1]:.3f}) | stated rank {s_[0]:.3f} | wiki {wiki_loss(m):.3f} bits  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
