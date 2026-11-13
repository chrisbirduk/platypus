"""DO SEVERAL WORDINGS MAKE A READ FACT ANSWERABLE? (13 Nov 2026; tools/paraphrase_test.py) - offline, no effect on the package.
The 'named facts do not generalise' gap (real-text replay: facts recalled in their own wording, at the fresh-subject prior in a new one), in
the setting of Allen-Zhu & Li 2023 (Physics of Language Models 3.1): a byte-level LM (the stand-in trunk of tools/trunk_guides.py) reads
synthetic biographies of N invented people (birth city, university, employer; 20 invented values each, chance 0.05) and is TAUGHT questions
about half of them (A); the test is the OTHER half (B): facts only ever read, never asked - can the model answer them? Two conditions, the
same exposure (each biography 5 times):
  SINGLE   every biography in one fixed wording and order (5 identical copies)
  MULTI    5 copies, each attribute in one of 5 wordings at random and the sentences shuffled (paraphrase augmentation)
Readouts (multiple choice by bits among the 20 values, every 2,000 steps): STORAGE (the biography's own first wording completes the fact),
taught questions on A, the SAME question forms on B (extraction from reading), and question forms NEVER trained, on A and B.
PT_ZIPF=1 (13 Nov 16:15): values Zipf-skewed; reports whether wrong answers drift to the commonest value (prototype pull, Spens & Burgess).
PT_NEED=uniform|need: extra biography copies for B, spread evenly or only to the half of B that will be asked (replay by need, Mattar & Daw).
PT_CONDS: the conditions to run (default SINGLE,MULTI).
Usage: python3 tools/paraphrase_test.py [steps, default 8000] [people, default 200]   env PT_D (width 128), PT_NL (layers 3), PT_THREADS"""
import os
import sys
import math
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('PT_THREADS', 1)))
P = lambda *a: print(*a, flush=True); T0 = time.time()
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000; N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
DW = int(os.environ.get('PT_D', 128)); NL = int(os.environ.get('PT_NL', 3)); L = 128; PAD = 256; V = 257; B = 32
rs = np.random.RandomState(0)
SY = ['ba', 'ko', 'ri', 'tel', 'man', 'vor', 'su', 'ne', 'dal', 'gri', 'po', 'lin', 'ka', 'mer', 'tu', 'zen', 'ho', 'fi', 'rad', 'ul', 'sen', 'mo', 'quil', 'tar']
def word(k): return ''.join(rs.choice(SY) for _ in range(k)).capitalize()
def uniq(n, f):
    out = []
    while len(out) < n:
        w = f()
        if w not in out: out.append(w)
    return out
PEOPLE = uniq(N, lambda: word(2) + ' ' + word(2))
POOL = {'city': uniq(20, lambda: word(3)), 'uni': [w + ' University' for w in uniq(20, lambda: word(2))], 'emp': [w + ' Industries' for w in uniq(20, lambda: word(2))]}
ATTR = list(POOL)
ZIPF = int(os.environ.get('PT_ZIPF', 0))                                        # 13 Nov: values drawn Zipf-skewed (a prototype exists)
zp = (1 / np.arange(1, 21)) / (1 / np.arange(1, 21)).sum()
FACT = {p: {a: POOL[a][rs.choice(20, p=zp) if ZIPF else rs.randint(20)] for a in ATTR} for p in PEOPLE}
TPL = {'city': ['{N} was born in {v}.', '{N} came into the world in {v}.', "{N}'s birthplace is {v}.", '{N} is a native of {v}.', 'The birthplace of {N} was {v}.'],
       'uni': ['{N} studied at {v}.', '{N} graduated from {v}.', '{N} attended {v}.', '{N} received a degree from {v}.', "{N}'s alma mater is {v}."],
       'emp': ['{N} worked for {v}.', '{N} was employed by {v}.', '{N} had a job at {v}.', '{N} joined {v}.', "{N}'s employer was {v}."]}
QTRAIN = {'city': 'Q: Where was {N} born? A:', 'uni': 'Q: Where did {N} study? A:', 'emp': 'Q: Who employed {N}? A:'}
QNEW = {'city': 'Q: What city is {N} from? A:', 'uni': 'Q: Which university did {N} attend? A:', 'emp': 'Q: Which company did {N} work for? A:'}
HA, HB = PEOPLE[:N // 2], PEOPLE[N // 2:]
NEED = os.environ.get('PT_NEED', '')                                            # 13 Nov: replay by need - half of B will be asked (NEEDED);
NB = HB[:len(HB) // 2]                                                          #   extra biography copies: 'uniform' 2 per B person, 'need' 4 per
EXTRA = {p: (2 if NEED == 'uniform' else (4 if (NEED == 'need' and p in NB) else 0)) for p in HB} if NEED else {}   # needed person (same total)


def corpus(cond):
    r = np.random.RandomState(1); docs = []
    for p in PEOPLE:
        for c in range(5 + EXTRA.get(p, 0)):
            if cond == 'SINGLE': ss = [TPL[a][0].format(N=p, v=FACT[p][a]) for a in ATTR]
            else: ss = [TPL[a][r.randint(5)].format(N=p, v=FACT[p][a]) for a in ATTR]; r.shuffle(ss)
            docs.append(' '.join(ss))
    for p in HA:
        for a in ATTR:
            for c in range(5): docs.append(QTRAIN[a].format(N=p) + ' ' + FACT[p][a] + '.')
    r.shuffle(docs); return np.frombuffer((' \n '.join(docs)).encode(), dtype=np.uint8)


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


@torch.no_grad()
def choose(m, prompts, cands):
    """for each (prompt, attribute): the candidate whose ' value.' costs the fewest bits after the prompt"""
    seqs, spans = [], []
    for pr, a in prompts:
        for v in cands[a]:
            pre = (pr + ' ').encode(); s_ = pre + (v + '.').encode(); seqs.append(s_[-L:]); spans.append((len(pre) - (len(s_) - len(s_[-L:])), len(s_[-L:])))
    out = np.zeros(len(seqs))
    for i in range(0, len(seqs), 256):
        ch = seqs[i:i + 256]; X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for j, c_ in enumerate(ch): X[j, :len(c_)] = torch.tensor(list(c_))
        Xi = torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1); lp = F.log_softmax(m(Xi), -1)
        nl = -lp.gather(-1, X.clamp(max=V - 1)[..., None])[..., 0]
        for j, (a_, b_) in enumerate(spans[i:i + 256]): out[i + j] = float(nl[j, max(0, a_):b_].sum())
    res, k = [], 0
    for pr, a in prompts:
        n = len(cands[a]); res.append(cands[a][int(np.argmin(out[k:k + n]))]); k += n
    return res


def evaluate(m, tag):
    m.eval(); r = {}
    def acc(people, form):
        pr = [(form[a].format(N=p) if '{N}' in form[a] else form[a], a) for p in people for a in ATTR]; truth = [FACT[p][a] for p in people for a in ATTR]
        return float(np.mean([x == y for x, y in zip(choose(m, pr, POOL), truth)]))
    st = {a: TPL[a][0].split('{v}')[0] for a in ATTR}
    r['storage A'] = acc(HA, st); r['storage B'] = acc(HB, st); r['taught Q, A'] = acc(HA, QTRAIN); r['SAME Q forms, B (read only)'] = acc(HB, QTRAIN)
    r['NEW Q forms, A'] = acc(HA, QNEW); r['NEW Q forms, B'] = acc(HB, QNEW)
    if NEED: r['NEEDED B, new forms'] = acc(NB, QNEW); r['NEEDED B, same forms'] = acc(NB, QTRAIN)
    if ZIPF:                                                                    # prototype pull: wrong answers on B that are the commonest value
        top = {a: collections.Counter(FACT[p][a] for p in PEOPLE).most_common(1)[0][0] for a in ATTR}
        pr = [(QNEW[a].format(N=p), a) for p in HB for a in ATTR]; ch = choose(m, pr, POOL); tr_ = [FACT[p][a] for p in HB for a in ATTR]
        wr = [(c, a) for (c, t), (_, a) in zip(zip(ch, tr_), pr) if c != t]
        r['wrong answers that are the commonest value'] = float(np.mean([c == top[a] for c, a in wr])) if wr else float('nan')
        r['commonest value share'] = float(np.mean([FACT[p][a] == top[a] for p in PEOPLE for a in ATTR]))
    m.train()
    P(f"   {tag}: " + " | ".join(f"{k} {v:.2f}" for k, v in r.items()) + f"  ({time.time() - T0:.0f}s)")
    return r


P(f"PARAPHRASE TEST: {N} people ({len(HA)} asked, {len(HB)} only read), attributes {ATTR}, 20 values each (chance 0.05); byte LM {DW} wide, {NL} layers, "
  f"{STEPS} steps, batch {B} x {L} bytes | e.g. {PEOPLE[0]}: {FACT[PEOPLE[0]]}")
for cond in os.environ.get('PT_CONDS', 'SINGLE,MULTI').split(','):
    C = corpus(cond); torch.manual_seed(0); m = LM(); op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(op, 1e-3, total_steps=STEPS, pct_start=0.05); ls = []; r2 = np.random.RandomState(2)
    P(f"\n== {cond}: corpus {len(C)} bytes, e.g. {bytes(C[:160]).decode()!r}")
    for st in range(STEPS):
        i0 = r2.randint(0, len(C) - L - 1, B); X = torch.tensor(np.stack([C[i:i + L] for i in i0]).astype(np.int64))
        Xi = torch.cat([torch.full((B, 1), PAD), X[:, :-1]], 1); l = F.cross_entropy(m(Xi).reshape(-1, V), X.reshape(-1))
        op.zero_grad(); l.backward(); op.step(); sch.step(); ls.append(l.item())
        if (st + 1) % 2000 == 0 or st + 1 == STEPS: evaluate(m, f"step {st + 1}, {np.mean(ls[-200:]) / math.log(2):.2f} bits/byte")
P(f"DONE ({time.time() - T0:.0f}s)")
