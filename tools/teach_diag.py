"""WHY ARE WRITTEN REAL-TEXT INFERENCES NOT RECALLED FROM THE WEIGHTS? (14 Nov 2026; tools/teach_diag.py) - offline, no effect on the package.
GPU cg27a_rcode visits 3-5: inferences written into the question pool ('A is connected to C') are recalled from the trunk at 0.00-0.02,
taught or held back, right or wrong, while planted facts in the same pool reach 0.2-0.4. On the GPU each pool fact gets about 2-3 question
rows a visit (399 steps x 4 rows over about 650 facts); planted facts are also READ 30 times each in the text, the inferences never.
The stand-in trunk (TG_CKPT; trained on wiki + Re-DocRED dev) is taught 120 pairs the same way - question rows in the system's own seven
frames (_DT_FORMS, none the test wording), mixed into batches of wiki text - and recall is measured in the system's test wording
('It is on record that A r '), two ways: OPEN-ENDED exact (greedy bytes, as the model's _dt_recall) and RANK (the answer first among it and 9
others of its kind, chance 0.10). Conditions, one factor at a time from a base (GPU-like dose, 'is connected to', unfamiliar names):
  DOSE        question rows per fact over the teaching: about 2.5 (GPU-like), 10, 40
  NAMES       unfamiliar (Re-DocRED training documents, never read by this trunk) vs familiar (dev documents, which it read)
  RELATION    'is connected to' vs the gold relation's own phrase ('was born in', 'is located in', ...)
  READ TOO    the pair's source sentence also in the text batches (as planted facts are read)
Then 300 steps of other text: what is kept. Usage: TG_D=192 TG_CKPT=<model file> python3 tools/teach_diag.py"""
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
        if len(out) >= 120: break
    return out, byt
UNF, BYT_U = pairs(tr_docs, False); FAM, BYT_F = pairs(dv_docs, True)
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
def run(tag, facts, byt, k, rel_of, read=False, steps=300):
    m = LM(); m.load_state_dict(torch.load(CK)); m.train(); op = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01); rr = np.random.RandomState(SEED + 3)
    b0 = recall(m, facts, rel_of, byt)
    sents = [f['sent'].encode('latin-1', 'replace') for f in facts if f['sent']]
    for st in range(steps):
        nk = int(k) + (1 if rr.rand() < k - int(k) else 0); seqs = []
        for _ in range(nk):
            f = facts[rr.randint(len(facts))]; seqs.append((FORMS[rr.randint(len(FORMS))].format(A=f['A'], r=rel_of(f)) + f['C'] + '.').encode('latin-1'))
        if read and sents: seqs += [sents[rr.randint(len(sents))] for _ in range(2)]
        while len(seqs) < 32: i0 = rr.randint(0, len(WK) - L - 1); seqs.append(bytes(WK[i0:i0 + L]))
        X = tens(seqs); Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg = X.reshape(-1)
        l = F.cross_entropy(lo[tg != PAD], tg[tg != PAD]); op.zero_grad(); l.backward(); op.step()
    b1 = recall(m, facts, rel_of, byt)
    for st in range(300):
        seqs = [bytes(WK[i0:i0 + L]) for i0 in rr.randint(0, len(WK) - L - 1, 32)]; X = tens(seqs); Xi = torch.cat([torch.full((32, 1), PAD), X[:, :-1]], 1)
        l = F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1)); op.zero_grad(); l.backward(); op.step()
    P(f"   {tag:44s} rows per fact {steps * k / len(facts):5.1f} | before: {b0} || taught: {b1} || +300 steps other text: {recall(m, facts, rel_of, byt)}  ({time.time() - T0:.0f}s)")
CONN = lambda f: 'is connected to'; GOLD = lambda f: f['rel']
P(f"facts: {len(UNF)} unfamiliar (Re-DocRED training documents), {len(FAM)} familiar (dev documents, read by this trunk); 300 teaching steps, batches of 32")
run('BASE: GPU-like dose, connected, unfamiliar', UNF, BYT_U, 1.0, CONN)
run('DOSE x4 (4 rows a step)', UNF, BYT_U, 4.0, CONN)
run('DOSE x16 (16 rows a step)', UNF, BYT_U, 16.0, CONN)
run('FAMILIAR names, GPU-like dose', FAM, BYT_F, 1.0, CONN)
run('FAMILIAR names, dose x4', FAM, BYT_F, 4.0, CONN)
run('GOLD RELATION phrase, dose x4', UNF, BYT_U, 4.0, GOLD)
run('READ TOO (source sentences in the text), dose x4', UNF, BYT_U, 4.0, CONN, read=True)
P(f"DONE ({time.time() - T0:.0f}s)")
