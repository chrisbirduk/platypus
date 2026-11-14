"""HOW TO UNLEARN A WRONG WRITTEN INFERENCE (14 Nov 2026; tools/retract_diag.py) - offline, no effect on the package.
The store will sometimes write a wrong inference ('A is connected to C') and teach it to the trunk (GPU cg27a_rcode3: about a third of
writes are wrong). If the store later sees the evidence turn (tools/fact_compress.py FC_SAVE / FC_LOAD), can the trunk be made to drop it
without damaging the rest? The stand-in trunk (TG_CKPT) is taught 360 pairs in the system's question frames at a high dose (teach_diag:
dose dominates): 120 RIGHT (gold pairs), 120 WRONG (A with a random name of the right type) and 120 BYSTANDERS (gold pairs, not touched
afterwards). Then 200 retraction steps, the right ones rehearsed in every arm, the wrong ones retracted by:
  STOP          no longer taught (the slot goes to text)
  NEUTRAL       taught a neutral answer instead ('A is connected to no one on record.')
  UNLIKELIHOOD  -log(1 - p(next byte)) on the wrong answer's bytes after the test-like frames (plus the right rows as anchor)
  NEUTRAL + UL  both
Readout (test wording 'It is on record that A is connected to '): RANK of the taught answer among it and 9 of its kind (chance 0.10;
untaught prior about 0.28), for wrong / right / bystander, and text loss on held wiki (collateral).
Usage: TG_D=192 TG_CKPT=<model file> python3 tools/retract_diag.py   env RD_NF (360), RD_STEPS1 (400), RD_STEPS2 (200), RD_ULW (1.0, unlikelihood
 weight), RD_ARMS (comma list), TD_SEED (0)"""
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
        if len(out) >= NF: break
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
NF = int(os.environ.get('RD_NF', 360)); ULW = float(os.environ.get('RD_ULW', 1.0)); ARMS = os.environ.get('RD_ARMS', 'STOP,NEUTRAL,UNLIKELIHOOD,NEUTRAL + UL').split(','); S1 = int(os.environ.get('RD_STEPS1', 400)); S2 = int(os.environ.get('RD_STEPS2', 200))
FACTS, BYT = pairs(tr_docs, False); rs.shuffle(FACTS); FACTS = FACTS[:NF]; n3 = len(FACTS) // 3
RIGHT, WRONG, BYST = FACTS[:n3], [dict(f) for f in FACTS[n3:2 * n3]], FACTS[2 * n3:3 * n3]
for f in WRONG:
    alts = sorted(n for n in BYT[f['t']] if n != f['C'] and 3 <= len(n) <= 30 and all(ord(ch) < 256 for ch in n)); f['C'] = alts[rs.randint(len(alts))]
REL_ = lambda f: 'is connected to'; NEUT = 'no one on record'
HW = [bytes(WK[i0:i0 + L]) for i0 in np.random.RandomState(99).randint(0, len(WK) - L - 1, 128)]
@torch.no_grad()
def wiki_loss(m):
    m.eval(); X = tens(HW); Xi = torch.cat([torch.full((len(HW), 1), PAD), X[:, :-1]], 1); l = F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1)); m.train(); return float(l) / math.log(2)
def row(f, c=None): return (FORMS[np.random.randint(len(FORMS))].format(A=f['A'], r=REL_(f)) + (c or f['C']) + '.').encode('latin-1')
def lm_loss(m, seqs):
    X = tens(seqs); Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg = X.reshape(-1)
    return F.cross_entropy(lo[tg != PAD], tg[tg != PAD])
def ul_loss(m, facts):
    seqs, spans = [], []
    for f in facts:
        pre = (FORMS[np.random.randint(len(FORMS))].format(A=f['A'], r=REL_(f))).encode('latin-1'); ans = (f['C'] + '.').encode('latin-1')
        seqs.append(pre + ans); spans.append((len(pre), len(pre) + len(ans)))
    X = tens(seqs); Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[..., 0]
    mk = torch.zeros_like(lp, dtype=torch.bool)
    for k, (a, b) in enumerate(spans): mk[k, a:min(b, L)] = True
    p = lp[mk].exp().clamp(max=0.999); return -torch.log1p(-p).mean()
def rep(m): return " | ".join(f"{nm} {recall(m, fs, REL_, BYT).split('rank ')[1]}" for nm, fs in (('WRONG', WRONG), ('RIGHT', RIGHT), ('BYSTANDER', BYST))) + f" | wiki {wiki_loss(m):.3f} bits"
np.random.seed(SEED)
m = LM(); m.load_state_dict(torch.load(CK)); m.train(); op = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
P(f"facts: right {len(RIGHT)}, wrong {len(WRONG)}, bystanders {len(BYST)} (unfamiliar names); rank chance 0.10")
P(f"BEFORE TEACHING (rank of the answer to be taught):   {rep(m)}  ({time.time() - T0:.0f}s)")
ALL = RIGHT + WRONG + BYST
for st in range(S1):
    seqs = [row(ALL[np.random.randint(len(ALL))]) for _ in range(16)] + [bytes(WK[i0:i0 + L]) for i0 in np.random.randint(0, len(WK) - L - 1, 16)]
    l = lm_loss(m, seqs); op.zero_grad(); l.backward(); op.step()
P(f"TAUGHT ({S1} steps, {S1 * 16 / len(ALL):.1f} rows per fact):            {rep(m)}  ({time.time() - T0:.0f}s)")
SD = {k: v.clone() for k, v in m.state_dict().items()}; OD = op.state_dict()
import copy
for arm in ARMS:
    m = LM(); m.load_state_dict(SD); m.train(); op = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); op.load_state_dict(copy.deepcopy(OD)); np.random.seed(SEED + 7)
    for st in range(S2):
        seqs = [row(RIGHT[np.random.randint(len(RIGHT))]) for _ in range(8)]
        if 'NEUTRAL' in arm: seqs += [row(WRONG[np.random.randint(len(WRONG))], NEUT) for _ in range(8)]
        seqs += [bytes(WK[i0:i0 + L]) for i0 in np.random.randint(0, len(WK) - L - 1, 32 - len(seqs))]
        l = lm_loss(m, seqs)
        if 'UL' in arm or arm == 'UNLIKELIHOOD': l = l + ULW * ul_loss(m, [WRONG[np.random.randint(len(WRONG))] for _ in range(8)])
        op.zero_grad(); l.backward(); op.step()
    P(f"RETRACTED by {arm:14s} (weight {ULW:g}, {S2} steps):            {rep(m)}  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
