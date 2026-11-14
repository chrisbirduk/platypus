"""CAN COMPRESSION CARRY COMPOSITION? (14 Nov 2026; tools/compose_synth.py) - offline, synthetic, no effect on the package.
After the two-hop literature (Balesni et al. 2411.16353: two NEW facts learned apart are not composed in the weights; one new fact + one
well-known fact is, about 20%; Feng et al. 2412.04614: new facts generalise to implications through structures learned when implications
of known facts were seen; Wang et al. 2405.15071: a composing circuit forms when it is cheaper than memorising, in distribution only).
A small byte transformer (the stand-in trunk's architecture) is trained from scratch on a synthetic world:
  countries (20), each with a TYPE CODE - the stand-in for the store's compressed type of a name (written as a suffix: 'Velmora-qa');
  OLD cities (300): 'B lies in C', many times, many wordings; a city's code is its country's code 80% of the time (an imperfect compressor)
  people: 'A lives in B'; for 400 TRAINING people also the implication 'A is a citizen of C' (implications of known facts in pretraining);
          100 HELD-OUT people have 'A lives in B' only
IDEA 1 (compression as training pressure; Wang et al.): do held-out people get the right citizenship without ever being told (in
distribution)? Measured during pretraining.
IDEA 2 (compression as shared representation; Feng et al.): after pretraining, new people and cities are taught briefly (300 steps, 8 new
fact rows + 24 replayed pretraining rows per batch), four arms from the same checkpoint:
  F1 OLD CITY       'A lives in B' with B an old, well-consolidated city                     (one new fact + one known: should work)
  F2 NEW, RANDOM    a new city with a RANDOM code: 'A lives in B' and 'B lies in C' both new  (two new facts: the two-hop curse)
  F3 NEW, TYPED     the same, but the new city's code is its country's (the compressed type)
  F4 TYPED ONLY     the new city's code is its country's; 'B lies in C' is never taught        (inheritance from the type alone)
Recall: the right answer first among it and 9 others of its kind (chance 0.10): citizenship ('A is a citizen of '), the city ('A lives
in '), the city's country ('B lies in ').
Usage: CS_D=192 CS_NL=3 CS_STEPS=6000 CS_IMPL=1 CS_SEED=0 python3 tools/compose_synth.py"""
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
P = lambda *a: print(*a, flush=True); T0 = time.time()
DW = int(os.environ.get('CS_D', 192)); NL = int(os.environ.get('CS_NL', 3)); STEPS = int(os.environ.get('CS_STEPS', 6000)); IMPL = int(os.environ.get('CS_IMPL', 1))
SEED = int(os.environ.get('CS_SEED', 0)); torch.set_num_threads(int(os.environ.get('CS_THREADS', 1))); torch.manual_seed(SEED); rs = np.random.RandomState(SEED)
L = 64; PAD = 256; V = 257; B_ = 32
SY = ['ka', 'lo', 'mi', 'ra', 'ven', 'tor', 'sa', 'bel', 'dun', 'fi', 'gar', 'hu', 'jo', 'ke', 'lin', 'mor', 'na', 'pe', 'qui', 'ro', 'sel', 'ta', 'ul', 'vo', 'wen', 'yo', 'zen']
used = set()
def name(nsy):
    while True:
        n = ''.join(rs.choice(SY) for _ in range(nsy)).capitalize()
        if n not in used: used.add(n); return n
CODES = ['q' + c for c in 'abcdefghijklmnopqrst']
NC = 20; COUNTRY = [name(3) for _ in range(NC)]
def city(c, informative=0.8):
    code = CODES[c] if rs.rand() < informative else CODES[rs.randint(NC)]
    return f"{name(3)}-{code}", c
OLD = [city(rs.randint(NC)) for _ in range(300)]
TRAINP = [(name(2) + name(1).lower() + ' ' + name(3), int(rs.randint(len(OLD)))) for _ in range(400)]
HELDP = [(name(2) + name(1).lower() + ' ' + name(3), int(rs.randint(len(OLD)))) for _ in range(100)]
CT = ["{B} lies in {C}.", "{B} is a city in {C}.", "The city of {B} is in {C}.", "{B}, a town of {C}, is old."]
PT = ["{A} lives in {B}.", "{A} has a home in {B}.", "{A} moved to {B} years ago."]
IT = ["{A} is a citizen of {C}.", "{A} holds the passport of {C}."]
def pre_sentences():
    out = []
    for b, c in OLD: out += [t.format(B=b, C=COUNTRY[c]) for t in CT] * 3
    for a, bi in TRAINP + HELDP: out += [t.format(A=a, B=OLD[bi][0]) for t in PT] * 2
    if IMPL:
        for a, bi in TRAINP: out += [t.format(A=a, C=COUNTRY[OLD[bi][1]]) for t in IT] * 2
    return out
PRE = pre_sentences()
def stream(sents, rr):
    o = rr.permutation(len(sents)); return np.frombuffer((' '.join(sents[i] for i in o) + ' ').encode('latin-1'), dtype=np.uint8)
PRES = stream(PRE * 4, rs)
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
def win(src, n, rr):
    i0 = rr.randint(0, len(src) - L - 1, n); return [bytes(src[i:i + L]) for i in i0]
def tens(seqs):
    X = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for k, q in enumerate(seqs): X[k, :len(q[:L])] = torch.tensor(list(q[:L]))
    return X
def step(m, op, seqs, sch=None):
    X = tens(seqs); Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); lo = m(Xi).reshape(-1, V); tg = X.reshape(-1)
    l = F.cross_entropy(lo[tg != PAD], tg[tg != PAD]); op.zero_grad(); l.backward(); op.step()
    if sch: sch.step()
    return float(l)
@torch.no_grad()
def rank1(m, items, rr):
    """items: (prefix, right answer, pool of alternatives) -> share with the right answer cheapest among it and 9 others"""
    m.eval(); hit = []
    for pre, ans, pool in items:
        alts = [x for x in pool if x != ans]; cands = [ans] + [alts[j] for j in rr.choice(len(alts), 9, replace=False)]
        p = pre.encode('latin-1'); seqs = [p + c.encode('latin-1') + b'.' for c in cands]; X = tens(seqs)
        Xi = torch.cat([torch.full((len(seqs), 1), PAD), X[:, :-1]], 1); nl = -F.log_softmax(m(Xi), -1).gather(-1, X[..., None])[..., 0]
        sc = [float(nl[k, len(p):min(L, len(q)) - 1].mean()) for k, q in enumerate(seqs)]; hit.append(int(np.argmin(sc) == 0))
    m.train(); return float(np.mean(hit))
CN = COUNTRY
cit = lambda people, cities: [(f"{a} is a citizen of ", CN[cities[bi][1]], CN) for a, bi in people]
liv = lambda people, cities: [(f"{a} lives in ", cities[bi][0], [c[0] for c in cities]) for a, bi in people]
lie = lambda cities: [(f"{b} lies in ", CN[c], CN) for b, c in cities]
P(f"synthetic world: {NC} countries, {len(OLD)} old cities (code = country's 80%), {len(TRAINP)} training people{' with implications' if IMPL else ' (NO implications)'}, "
  f"{len(HELDP)} held-out people | pretraining sentences {len(PRE)}, stream {len(PRES)} bytes | LM {NL} layers x {DW}, {STEPS} steps, seed {SEED}")
m = LM(); op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1); sch = torch.optim.lr_scheduler.OneCycleLR(op, 1e-3, total_steps=STEPS, pct_start=0.05)
rr = np.random.RandomState(SEED + 1); ls = []
for st in range(1, STEPS + 1):
    ls.append(step(m, op, win(PRES, B_, rr), sch))
    if st % 1000 == 0 or st == STEPS:
        re_ = np.random.RandomState(7)
        P(f"   pretraining step {st}: {np.mean(ls[-200:]) / math.log(2):.3f} bits/byte | city's country {rank1(m, lie(OLD[:100]), re_):.2f} | person's city {rank1(m, liv(HELDP, OLD), re_):.2f} | "
          f"citizenship TAUGHT (training people) {rank1(m, cit(TRAINP[:100], OLD), re_):.2f} | IDEA 1 citizenship NEVER TAUGHT (held-out people) {rank1(m, cit(HELDP, OLD), re_):.2f}  ({time.time() - T0:.0f}s)")
st0 = {k: v.clone() for k, v in m.state_dict().items()}
# ---- IDEA 2: four arms from the same checkpoint
def arm(tag, people, cities, teach_lies):
    sents = [t.format(A=a, B=cities[bi][0]) for a, bi in people for t in PT[:1]]
    if teach_lies: sents += [CT[0].format(B=b, C=CN[c]) for b, c in cities]
    mm = LM(); mm.load_state_dict(st0); mm.train(); op2 = torch.optim.AdamW(mm.parameters(), lr=3e-4, weight_decay=0.01); r2 = np.random.RandomState(SEED + 11); out = []
    for s_ in range(1, 301):
        step(mm, op2, [sents[r2.randint(len(sents))].encode('latin-1') for _ in range(8)] + win(PRES, B_ - 8, r2))
        if s_ in (100, 300):
            re_ = np.random.RandomState(9)
            out.append(f"{s_}: CITIZENSHIP {rank1(mm, cit(people, cities), re_):.2f} | city {rank1(mm, liv(people, cities), re_):.2f} | city's country {rank1(mm, lie(cities), re_):.2f}")
    P(f"   {tag:16s} " + " || ".join(out) + f"  ({time.time() - T0:.0f}s)")
newp = lambda n: [(name(2) + name(1).lower() + ' ' + name(3), i) for i in range(n)]
P("IDEA 2 - taught briefly after pretraining (chance 0.10; citizenship is never taught for these people):")
arm('F1 OLD CITY', [(name(2) + name(1).lower() + ' ' + name(3), int(rs.randint(len(OLD)))) for _ in range(100)], OLD, False)
NEWC = [rs.randint(NC) for _ in range(100)]
arm('F2 NEW, RANDOM', newp(100), [(f"{name(3)}-{CODES[rs.randint(NC)]}", c) for c in NEWC], True)
arm('F3 NEW, TYPED', newp(100), [(f"{name(3)}-{CODES[c]}", c) for c in NEWC], True)
arm('F4 TYPED ONLY', newp(100), [(f"{name(3)}-{CODES[c]}", c) for c in NEWC], False)
P(f"DONE ({time.time() - T0:.0f}s)")
