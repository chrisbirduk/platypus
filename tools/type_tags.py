"""DO COMPRESSED TYPES HELP THE TRUNK WITH NEW NAMES ON REAL TEXT? (14 Nov 2026; tools/type_tags.py) - offline, no effect on the package.
After tools/compose_synth.py: in a synthetic world a type code on a new name is used at once (a new city's country 0.85-0.92 never taught,
chance 0.10) but does not make two hops compose. Real compressed types are coarser. Here the stand-in trunk (TG_CKPT, trained on wiki +
Re-DocRED dev) is fine-tuned on Re-DocRED TRAINING documents 0..NF-1 with every entity mention followed by a type tag ' [tN]', four arms:
  NONE     no tags
  COMP     the store's compressed type: the label-free block model's name class (tools/record_mdl.py, 16 name x 32 wording classes)
  ORACLE   the gold entity type (PER, LOC, ORG, TIME, NUM, MISC) - the upper bound
  RANDOM   a random tag per name (16 values) - the control
Test (v2): held-out training documents NF..: bits per byte on the 32 bytes after each mention of a name never seen in fine-tuning (tags excluded). A relation
template is prompted with the head and its tag ('{h} [tN] was born in '), and the gold tail's bytes are scored against 9 others (chance 0.10):
  CLASS  9 names of OTHER gold types - does the trunk know what kind of thing comes next for a new name? (what a type can carry)
  FACT   9 names of the SAME gold type - the specific fact (never read: should stay near chance; a check on the measure)
CLASS is split by whether the head's compressed class is PURE (its class's majority gold type is the head's) or not: the cost of
inherited errors. Usage: TG_D=192 TG_CKPT=<model file> TT_STEPS=1500 TT_SEED=0 python3 tools/type_tags.py"""
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
DW = int(os.environ.get('TG_D', 192)); NL = int(os.environ.get('TG_NL', 3)); CK = os.environ['TG_CKPT']; STEPS = int(os.environ.get('TT_STEPS', 1500))
SEED = int(os.environ.get('TT_SEED', 0)); NF = int(os.environ.get('TT_NF', 2500)); torch.set_num_threads(1); torch.manual_seed(SEED); rs = np.random.RandomState(SEED)
L = 128; PAD = 256; V = 257
docs = json.load(open(SCR + '/bench_dl/train_revised.json'))
RMJ = json.load(open(os.environ.get('TT_TYPES', SCR + '/record_mdl.json'))); col = RMJ['k'].index(os.environ.get('TT_KM', '16x32')); CT = RMJ['types']
cn_ = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
tp_ = lambda d: [vs[0].get('type', 'X') for vs in d['vertexSet']]
GT = {}; ndocs = collections.defaultdict(set)
for di, d in enumerate(docs):
    for n, t in zip(cn_(d), tp_(d)): GT.setdefault(n, t); ndocs[n].add(di)
GTS = sorted(set(GT.values())); RAND = {n: int(rs.randint(16)) for n in GT}
tag = {'NONE': lambda n: None, 'COMP': lambda n: CT[n][col] if n in CT else None, 'ORACLE': lambda n: GTS.index(GT[n]) + 20, 'RANDOM': lambda n: RAND.get(n)}
def text(di, arm, marks=None):
    d = docs[di]; cn = cn_(d); ends = {}
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: ends[(m['sent_id'], m['pos'][1] - 1)] = cn[ei]
    out = b''
    for si, s in enumerate(d['sents']):
        for k, w in enumerate(s):
            out += (b' ' if out else b'') + w.encode('utf-8', 'replace'); n = ends.get((si, k))
            if n is not None and tag[arm](n) is not None: out += f" [t{tag[arm](n)}]".encode()
            if n is not None and marks is not None and n not in seenf: marks.append((len(out), n))
    return out
# purity of compressed classes: majority gold type per class (by names)
cls = collections.defaultdict(collections.Counter)
for n, t in GT.items():
    if n in CT: cls[CT[n][col]][t] += 1
MAJ = {c: v.most_common(1)[0][0] for c, v in cls.items()}
pure = lambda n: n in CT and MAJ.get(CT[n][col]) == GT[n]
P(f"compressed classes ({RMJ['k'][col]}): names covered {sum(1 for n in GT if n in CT)} of {len(GT)}, names in their class's majority gold type "
  f"{np.mean([pure(n) for n in GT if n in CT]):.3f}")
TPL = {'P17': '{h} is in the country of ', 'P131': '{h} is located in ', 'P27': '{h} is a citizen of ', 'P19': '{h} was born in ', 'P20': '{h} died in ',
       'P569': '{h} was born on ', 'P570': '{h} died on ', 'P108': '{h} worked for ', 'P54': '{h} played for ', 'P463': '{h} is a member of ',
       'P102': '{h} is a member of the party ', 'P175': '{h} was performed by ', 'P577': '{h} was released in ', 'P495': '{h} comes from ',
       'P571': '{h} was founded in ', 'P159': '{h} is based in ', 'P50': '{h} was written by ', 'P57': '{h} was directed by ', 'P161': '{h} stars ',
       'P150': '{h} contains ', 'P150i': '', 'P264': '{h} was released by the label ', 'P150x': ''}
fine = set(range(NF)); seenf = {n for di in fine for n in cn_(docs[di])}
TEST = []; pool = collections.defaultdict(set)
for di in range(NF, len(docs)):
    d = docs[di]; cn, tp = cn_(d), tp_(d)
    for n, t in zip(cn, tp):
        if n not in seenf and 3 <= len(n) <= 30: pool[t].add(n)
    for l in d['labels']:
        h, t_ = cn[l['h']], cn[l['t']]
        if TPL.get(l['r']) and h not in seenf and 3 <= len(h) <= 30 and 2 <= len(t_) <= 30: TEST.append((h, l['r'], t_, tp[l['t']]))
TEST = list({(h, r): (h, r, t, ty) for h, r, t, ty in TEST}.values()); rs.shuffle(TEST); TEST = TEST[:600]
pool = {t: sorted(v) for t, v in pool.items()}
def cands(t_, ty, same, rr):
    if same: alts = [n for n in pool.get(ty, []) if n != t_]
    else: alts = [n for t, v in pool.items() if t != ty for n in v]
    return [t_] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)] if len(alts) >= 9 else None
CAND = {}
rr0 = np.random.RandomState(SEED + 3)
for it in TEST:
    CAND[(it, True)] = cands(it[2], it[3], True, rr0); CAND[(it, False)] = cands(it[2], it[3], False, rr0)
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
def score(m, arm):
    """bits per byte on the 32 bytes that FOLLOW each mention of a new name in the held-out documents (after its tag, if any), and on all
    held-out text outside the tags; split by whether the name's compressed class is pure"""
    m.eval(); res = collections.defaultdict(list); rr = np.random.RandomState(SEED + 5)
    for di in HOLD:
        mk = []; t = text(di, arm, mk); tb = np.frombuffer(t, dtype=np.uint8)
        for e, n in mk:
            if e + 32 > len(tb) or rr.rand() > 0.5: continue
            a0 = max(0, e + 32 - L); X = tens([bytes(tb[a0:e + 32])]); Xi = torch.cat([torch.full((1, 1), PAD), X[:, :-1]], 1)
            nl = -F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[0, :, 0]; b = float(nl[e - a0:e + 32 - a0].mean()) / math.log(2)
            res['after a new name'].append(b); res['after a new name, class pure' if pure(n) else 'after a new name, class impure'].append(b)
    m.train(); return " | ".join(f"{k} {np.mean(v):.3f} ({len(v)})" for k, v in sorted(res.items()))
wiki = open(os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile') + '/wiki.bin', 'rb').read().replace(b'\n', b' ')
WK = np.frombuffer(wiki, dtype=np.uint8)
m0 = LM(); m0.load_state_dict(torch.load(CK))
HOLD = list(range(NF, min(len(docs), NF + int(os.environ.get('TT_NH', 300)))))
P(f"held-out documents {len(HOLD)} ({NF}..): bits per byte after mentions of names never seen in fine-tuning (lower = the trunk knows better what is "
  f"said about a new name) | before fine-tuning (no tags): {score(m0, 'NONE')}  ({time.time() - T0:.0f}s)")
for arm in ('NONE', 'COMP', 'ORACLE', 'RANDOM'):
    TX = np.frombuffer(b' '.join(text(di, arm) for di in range(NF)), dtype=np.uint8)
    m = LM(); m.load_state_dict(torch.load(CK)); m.train(); op = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); rr = np.random.RandomState(SEED + 7); ls = []
    for st in range(1, STEPS + 1):
        src = [TX] * 24 + [WK] * 8; seqs = []
        for s_ in src: i0 = rr.randint(0, len(s_) - L - 1); seqs.append(bytes(s_[i0:i0 + L]))
        X = tens(seqs); Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); l = F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1))
        op.zero_grad(); l.backward(); op.step(); ls.append(float(l))
    P(f"   {arm:7s} ({STEPS} steps, {np.mean(ls[-200:]) / math.log(2):.3f} bits/byte; tagged text {len(TX)} bytes): {score(m, arm)}  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
