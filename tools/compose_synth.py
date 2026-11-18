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
CS_COT 1 (18 Nov, LEARNED DECOMPOSITION): the training people's implication is written as its chain - 'A is a citizen of B > C.' - so the
model can learn to produce the bridge itself; held-out people are asked 'A is a citizen of ' and their greedy continuation is parsed (B, then C).
CS_REV 1 (with CS_COT): belief revision - held-out people move to another old city (taught briefly); is the chain's answer the NEW country?
CS_LOOP (18 Nov): passes through the same blocks - a weight-shared LOOPED trunk (the two-hop literature: a second pass lets the bridge entity
resolved in pass 1 be used in pass 2).
Usage: CS_D=192 CS_NL=3 CS_STEPS=6000 CS_IMPL=1 CS_SEED=0 [CS_NTRAIN=400 CS_NHELD=100 CS_LOOP=1] python3 tools/compose_synth.py"""
import os
import math
import collections
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
P = lambda *a: print(*a, flush=True); T0 = time.time()
DW = int(os.environ.get('CS_D', 192)); NL = int(os.environ.get('CS_NL', 3)); STEPS = int(os.environ.get('CS_STEPS', 6000)); IMPL = int(os.environ.get('CS_IMPL', 1))
COT = int(os.environ.get('CS_COT', 0)); COTLATE = int(os.environ.get('CS_COTLATE', 0)); OPJ = int(os.environ.get('CS_OPJ', 0)); REV = int(os.environ.get('CS_REV', 0)); SEED = int(os.environ.get('CS_SEED', 0)); LOOP = int(os.environ.get('CS_LOOP', 1)); torch.set_num_threads(int(os.environ.get('CS_THREADS', 1))); torch.manual_seed(SEED); rs = np.random.RandomState(SEED)
L = int(os.environ.get('CS_L', 64)); PAD = 256; V = 257; B_ = 32
SY = ['ka', 'lo', 'mi', 'ra', 'ven', 'tor', 'sa', 'bel', 'dun', 'fi', 'gar', 'hu', 'jo', 'ke', 'lin', 'mor', 'na', 'pe', 'qui', 'ro', 'sel', 'ta', 'ul', 'vo', 'wen', 'yo', 'zen']
used = set()
def name(nsy):
    while True:
        n = ''.join(rs.choice(SY) for _ in range(nsy)).capitalize()
        if n not in used: used.add(n); return n
def person():                                                                  # first + family name; unique as a whole
    while True:
        n = ''.join(rs.choice(SY) for _ in range(2)).capitalize() + ' ' + ''.join(rs.choice(SY) for _ in range(3)).capitalize()
        if n not in used: used.add(n); return n
CODES = ['q' + c for c in 'abcdefghijklmnopqrst']
NC = 20; COUNTRY = [name(3) for _ in range(NC)]
def city(c, informative=0.8):
    code = CODES[c] if rs.rand() < informative else CODES[rs.randint(NC)]
    return f"{name(3)}-{code}", c
OLD = [city(rs.randint(NC)) for _ in range(300)]
PCODE = int(os.environ.get('CS_PCODE', 0))                                    # 14 Nov: CS_PCODE 1 - every PERSON also carries a code, the compressed
def pcode(bi, informative=0.8):                                                #   fact table's summary of their neighbourhood: their city's country's
    c = OLD[bi][1]; return CODES[c] if rs.rand() < informative else CODES[rs.randint(NC)]   #   code (80%, like the cities' codes)
def pnew(bi):
    n = person(); return (f"{n}-{pcode(bi)}" if PCODE else n), bi
TRAINP = [pnew(int(rs.randint(len(OLD)))) for _ in range(int(os.environ.get('CS_NTRAIN', 400)))]   # 17 Nov: CS_NTRAIN / CS_NHELD
HELDP = [pnew(int(rs.randint(len(OLD)))) for _ in range(int(os.environ.get('CS_NHELD', 100)))]
F1P = [pnew(int(rs.randint(len(OLD)))) for _ in range(100)]
if PCODE == 2:                                                                 # 14 Nov CS_PCODE 2 - codes from REAL COMPRESSION of the fact graph: names lose the
    from scipy.sparse import csr_matrix                                        #   handed-out codes; the graph of what is stated (person-city, city-country, and
    from sklearn.decomposition import TruncatedSVD                             #   person-country for training people if CS_IMPL) is coded by PPMI + truncated SVD
    from sklearn.cluster import KMeans                                         #   (CS_PK dims) and k-means (CS_PKC clusters, no labels); each person and city
    strip = lambda n: n.rsplit('-', 1)[0]                                      #   carries its cluster as a token ('Velmora-k07')
    cities = [strip(b) for b, c in OLD]; people = [(strip(a), bi) for a, bi in TRAINP + HELDP + F1P]
    nid = {}; ed = []
    for i, (b, c) in enumerate(OLD): ed.append((nid.setdefault('C:' + cities[i], len(nid)), nid.setdefault('K:' + COUNTRY[c], len(nid))))
    for j, (a, bi) in enumerate(people):
        ed.append((nid.setdefault('P:' + a, len(nid)), nid['C:' + cities[bi]]))
        if IMPL and j < len(TRAINP): ed.append((nid['P:' + a], nid['K:' + COUNTRY[OLD[bi][1]]]))
        nz = float(os.environ.get('CS_NOISE', 0))                              #   CS_NOISE: co-occurrence noise - random person-city links (on average this
        for _ in range(int(nz) + (1 if rs.rand() < nz - int(nz) else 0)):      #   many per person) that the store records like facts, as real text does
            ed.append((nid['P:' + a], nid['C:' + cities[rs.randint(len(cities))]]))
    NN_ = len(nid); ea = np.array(ed); G_ = csr_matrix((np.ones(len(ea)), (ea[:, 0], ea[:, 1])), shape=(NN_, NN_)); G_ = G_ + G_.T; G_.data[:] = 1.0
    G2 = (G_ @ G_ + G_).tocoo(); dg = np.asarray(G2.sum(1)).ravel(); v_ = np.log(G2.sum() * G2.data / (dg[G2.row] * dg[G2.col]) + 1e-12); m_ = v_ > 0
    U_ = TruncatedSVD(int(os.environ.get('CS_PK', 16)), random_state=SEED).fit_transform(csr_matrix((v_[m_], (G2.row[m_], G2.col[m_])), shape=G2.shape))
    U_ /= np.linalg.norm(U_, axis=1, keepdims=True) + 1e-9
    ent = [k for k in nid if not k.startswith('K:')]; KC = int(os.environ.get('CS_PKC', 40))
    lab = KMeans(KC, random_state=SEED, n_init=5).fit_predict(U_[[nid[k] for k in ent]]); TOK = {k: f"k{l:02d}" for k, l in zip(ent, lab)}
    def pur(keys, ctry):
        cc = collections.defaultdict(collections.Counter)
        for k, c in zip(keys, ctry): cc[TOK[k]][c] += 1
        return sum(v.most_common(1)[0][1] for v in cc.values()) / max(1, len(keys))
    pp = pur(['P:' + a for a, bi in people], [OLD[bi][1] for a, bi in people]); pc = pur(['C:' + c for c in cities], [c for b, c in OLD])
    OLD = [(f"{cities[i]}-{TOK['C:' + cities[i]]}", c) for i, (b, c) in enumerate(OLD)]
    rn = lambda L: [(f"{strip(a)}-{TOK['P:' + strip(a)]}", bi) for a, bi in L]
    TRAINP, HELDP, F1P = rn(TRAINP), rn(HELDP), rn(F1P)
    print(f"CODES FROM COMPRESSION: {NN_} nodes, PPMI of 1-2 hop graph, SVD {U_.shape[1]} dims, {KC} clusters | country purity of the clusters: people {pp:.2f}, "
          f"cities {pc:.2f} (a handed-out code: 0.80; chance about {1 / NC:.2f})", flush=True)
CT = ["{B} lies in {C}.", "{B} is a city in {C}.", "The city of {B} is in {C}.", "{B}, a town of {C}, is old."]
PT = ["{A} lives in {B}.", "{A} has a home in {B}.", "{A} moved to {B} years ago."]
IT = ["{A} is a citizen of {C}.", "{A} holds the passport of {C}."]
def pre_sentences():
    out = []
    for b, c in OLD: out += [t.format(B=b, C=COUNTRY[c]) for t in CT] * 3
    for a, bi in TRAINP + HELDP: out += [t.format(A=a, B=OLD[bi][0]) for t in PT] * 2
    if IMPL and COT:
        for a, bi in TRAINP: out += [f"{a} is a citizen of {OLD[bi][0]} > {COUNTRY[OLD[bi][1]]}."] * 4
    elif IMPL:
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
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V); s.lp = nn.Parameter(torch.zeros(LOOP, d))   # 18 Nov: CS_LOOP passes through the SAME blocks
    def hid(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]
        for p_ in range(LOOP):
            x = x + s.lp[p_]
            for b in s.b: x = b(x, am)
        return s.n(x)
    def forward(s, X): return s.o(s.hid(X))                                    #   (weight-shared looped trunk; a learned per-pass offset)
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
@torch.no_grad()
def gen(m, pre, n=48):
    m.eval(); b = bytearray(pre.encode('latin-1'))
    for _ in range(n):
        if len(b) >= L: break
        X = torch.tensor([[PAD] + list(b)])[:, -L:]; c = int(m(X)[0, -1].argmax())
        if c in (46, 10, PAD): break
        b.append(c)
    m.train(); return bytes(b[len(pre.encode('latin-1')):]).decode('latin-1', 'replace').strip()
def chain(m, people, cities):
    rb, rc = [], []
    for a, bi in people:
        g = gen(m, f"{a} is a citizen of "); parts = [x.strip() for x in g.split('>')]
        rb.append(len(parts) == 2 and parts[0] == cities[bi][0]); rc.append(len(parts) == 2 and parts[1] == CN[cities[bi][1]])
    return float(np.mean(rb)), float(np.mean(rc))
cit = lambda people, cities: [(f"{a} is a citizen of ", CN[cities[bi][1]], CN) for a, bi in people]
liv = lambda people, cities: [(f"{a} lives in ", cities[bi][0], [c[0] for c in cities]) for a, bi in people]
lie = lambda cities: [(f"{b} lies in ", CN[c], CN) for b, c in cities]
P(f"synthetic world: {NC} countries, {len(OLD)} old cities (code = country's 80%), {len(TRAINP)} training people{' with implications' if IMPL else ' (NO implications)'}, "
  f"{len(HELDP)} held-out people | pretraining sentences {len(PRE)}, stream {len(PRES)} bytes | LM {NL} layers x {DW} x {LOOP} pass(es), {STEPS} steps, seed {SEED}")
m = LM(); op = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1); sch = torch.optim.lr_scheduler.OneCycleLR(op, 1e-3, total_steps=STEPS, pct_start=0.05)
rr = np.random.RandomState(SEED + 1); ls = []; CK_ = os.environ.get('CS_CKPT', '')   # CS_CKPT: load the pretrained model if the file exists, else pretrain and save it
if CK_ and os.path.exists(CK_): m.load_state_dict(torch.load(CK_)); P(f"   loaded {CK_}"); STEPS_RUN = 0
else: STEPS_RUN = STEPS
for st in range(1, STEPS_RUN + 1):
    ls.append(step(m, op, win(PRES, B_, rr), sch))
    if st % 1000 == 0 or st == STEPS:
        re_ = np.random.RandomState(7)
        P(f"   pretraining step {st}: {np.mean(ls[-200:]) / math.log(2):.3f} bits/byte | city's country {rank1(m, lie(OLD[:100]), re_):.2f} | person's city {rank1(m, liv(HELDP, OLD), re_):.2f} | "
          f"citizenship TAUGHT (training people) {rank1(m, cit(TRAINP[:100], OLD), re_):.2f} | IDEA 1 citizenship NEVER TAUGHT (held-out people) {rank1(m, cit(HELDP, OLD), re_):.2f}  ({time.time() - T0:.0f}s)"
          + (" || LEARNED DECOMPOSITION (greedy chain 'B > C'): taught B {:.2f} C {:.2f} | NEVER TAUGHT B {:.2f} C {:.2f}".format(*chain(m, TRAINP[:100], OLD), *chain(m, HELDP, OLD)) if COT else ""))
if CK_ and not os.path.exists(CK_): torch.save(m.state_dict(), CK_)
st0 = {k: v.clone() for k, v in m.state_dict().items()}
if OPJ:                                                                        # 18 Nov ROUTE J IN THE SYNTHETIC WORLD: one linear operator per relation on the model's
    @torch.no_grad()                                                           #   name codes (mean final state over the name's bytes in 'About X.'), fitted on facts
    def codes(names):                                                          #   ('lives in' for every person, 'lies in' for every old city); citizenship of the HELD-OUT
        out = []                                                               #   people (never taught) by composing the two operators in latent space
        for n_ in names:
            b_ = f"About {n_}.".encode('latin-1'); X = torch.tensor([[PAD] + list(b_)])[:, :L]; h = m.hid(X)[0]
            out.append(h[7:7 + len(n_)].mean(0))
        Z_ = torch.stack(out); Z_ = Z_ - Z_.mean(0); return Z_ / (Z_.norm(dim=-1, keepdim=True) + 1e-8)
    ppl = TRAINP + HELDP; CIT = [b for b, c in OLD]; ALL = [a for a, _ in ppl] + CIT + CN
    Z = codes(ALL); ix = {n_: k for k, n_ in enumerate(ALL)}; lam = float(os.environ.get('CS_OPJ_LAM', 0.1)); Id = torch.eye(Z.shape[1])
    def fit(pairs, shuf=False):
        X = Z[[ix[a] for a, b in pairs]]; Y = Z[[ix[b] for a, b in pairs]]
        if shuf: Y = Y[torch.randperm(len(Y), generator=torch.Generator().manual_seed(3))]
        return torch.linalg.solve(X.t() @ X + lam * Id, X.t() @ Y)
    Pl = fit([(a, OLD[bi][0]) for a, bi in ppl]); Pc = fit([(b, CN[c]) for b, c in OLD]); Pls = fit([(a, OLD[bi][0]) for a, bi in ppl], True)
    ci = torch.tensor([ix[c] for c in CIT]); ki = torch.tensor([ix[c] for c in CN]); ai = torch.arange(len(ALL))
    h1, cc, cs, call, nul = [], [], [], [], []
    for a, bi in HELDP:
        v1 = Z[ix[a]] @ Pl; b1 = CIT[int((v1 @ Z[ci].t()).argmax())]; h1.append(b1 == OLD[bi][0])
        v2 = v1 @ Pc; cc.append(CN[int((v2 @ Z[ki].t()).argmax())] == CN[OLD[bi][1]]); call.append(ALL[int((v2 @ Z.t()).argmax())] == CN[OLD[bi][1]])
        cs.append(CN[int(((Z[ix[b1]] @ Pc) @ Z[ki].t()).argmax())] == CN[OLD[bi][1]])
        nul.append(CN[int(((Z[ix[a]] @ Pls @ Pc) @ Z[ki].t()).argmax())] == CN[OLD[bi][1]])
    P(f"ROUTE J (synthetic, {Z.shape[1]}-d codes, {len(ALL)} names): HELD-OUT people's citizenship never taught (n {len(HELDP)}) || first hop: city {np.mean(h1):.2f} (among {len(CIT)}) | "
      f"COMPOSITE country {np.mean(cc):.2f} (among {NC}; chance {1 / NC:.2f}), among ALL names {np.mean(call):.2f} | with B SNAPPED to the nearest city {np.mean(cs):.2f} | "
      f"NULL (lives-in operator fitted on shuffled facts) {np.mean(nul):.2f} | the model asked directly (rank among 10) {rank1(m, cit(HELDP, OLD), np.random.RandomState(9)):.2f}  ({time.time() - T0:.0f}s)")
    if int(os.environ.get('CS_OPJ_MLP', 0)):                                   # HIGHER-CAPACITY ROUTE J: a JEPA-style PREDICTOR per relation (MLP, InfoNCE over the names'
        Zt = Z.detach()                                                         #   codes, temperature 0.05) instead of one linear operator; the same composition and null
        def mlp_fit(pairs, shuf=False, hid=int(os.environ.get('CS_OPJ_H', 1024)), steps=int(os.environ.get('CS_OPJ_STEPS', 3000))):
            g_ = torch.Generator().manual_seed(11); net = nn.Sequential(nn.Linear(Zt.shape[1], hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, Zt.shape[1]))
            A_ = torch.tensor([ix[a] for a, b in pairs]); B9 = torch.tensor([ix[b] for a, b in pairs])
            if shuf: B9 = B9[torch.randperm(len(B9), generator=g_)]
            tgt = torch.unique(B9); pos = {int(t): k for k, t in enumerate(tgt)}; yb = torch.tensor([pos[int(b)] for b in B9]); o9 = torch.optim.Adam(net.parameters(), lr=1e-3)
            for _ in range(steps):
                k9 = torch.randint(0, len(A_), (256,), generator=g_); v = F.normalize(net(Zt[A_[k9]]) + Zt[A_[k9]] * 0, dim=-1)
                l9 = F.cross_entropy(v @ Zt[tgt].t() / 0.05, yb[k9]); o9.zero_grad(); l9.backward(); o9.step()
            return net
        Ml = mlp_fit([(a, OLD[bi][0]) for a, bi in ppl]); Mc = mlp_fit([(b, CN[c]) for b, c in OLD]); Mls = mlp_fit([(a, OLD[bi][0]) for a, bi in ppl], True)
        with torch.no_grad():
            h1, cc, cs, nul = [], [], [], []
            for a, bi in HELDP:
                v1 = F.normalize(Ml(Zt[ix[a]]), dim=-1); b1 = CIT[int((v1 @ Zt[ci].t()).argmax())]; h1.append(b1 == OLD[bi][0])
                cc.append(CN[int((F.normalize(Mc(v1), dim=-1) @ Zt[ki].t()).argmax())] == CN[OLD[bi][1]])
                cs.append(CN[int((F.normalize(Mc(Zt[ix[b1]]), dim=-1) @ Zt[ki].t()).argmax())] == CN[OLD[bi][1]])
                nul.append(CN[int((F.normalize(Mc(F.normalize(Mls(Zt[ix[a]]), dim=-1)), dim=-1) @ Zt[ki].t()).argmax())] == CN[OLD[bi][1]])
        P(f"ROUTE J, MLP PREDICTOR (InfoNCE): first hop city {np.mean(h1):.2f} | COMPOSITE country {np.mean(cc):.2f} | with B SNAPPED {np.mean(cs):.2f} | NULL {np.mean(nul):.2f}  ({time.time() - T0:.0f}s)")
if COTLATE:                                                                    # 18 Nov LATE CHAIN TEACHING (the package's situation): the model was pretrained with NO citizenship
    NT = int(os.environ.get('CS_COTLATE_N', 300)); TP = TRAINP[:NT]           #   at all; then the chain 'A is a citizen of B > C.' is taught briefly for NT people (8 rows +
    for tag, fmt in (('CHAIN', lambda a, bi: f"{a} is a citizen of {OLD[bi][0]} > {CN[OLD[bi][1]]}."), ('DIRECT', lambda a, bi: f"{a} is a citizen of {CN[OLD[bi][1]]}.")):
        sents = [fmt(a, bi) for a, bi in TP]                                   #   24 replayed rows a batch); held-out people never taught
        mm = LM(); mm.load_state_dict(st0); mm.train(); op2 = torch.optim.AdamW(mm.parameters(), lr=3e-4, weight_decay=0.01); r2 = np.random.RandomState(SEED + 31); out = []
        for s_ in range(1, int(os.environ.get('CS_COTLATE_STEPS', 1500)) + 1):
            step(mm, op2, [sents[r2.randint(len(sents))].encode('latin-1') for _ in range(8)] + win(PRES, B_ - 8, r2))
            if s_ in (300, 750, 1500, 3000):
                if tag == 'CHAIN': bb_, c_ = chain(mm, HELDP, OLD); tb_, tc_ = chain(mm, TP[:100], OLD); out.append(f"{s_}: taught C {tc_:.2f} | NEVER TAUGHT B {bb_:.2f} C {c_:.2f}")
                else: out.append(f"{s_}: taught {rank1(mm, cit(TP[:100], OLD), np.random.RandomState(9)):.2f} | NEVER TAUGHT {rank1(mm, cit(HELDP, OLD), np.random.RandomState(9)):.2f} (rank among 10)")
        P(f"   LATE TEACHING, {tag} form, {NT} people: " + " || ".join(out) + f"  ({time.time() - T0:.0f}s)")
if OPJ or COTLATE: P(f"DONE ({time.time() - T0:.0f}s)"); raise SystemExit
if COT and REV:                                                                # BELIEF REVISION: 100 held-out people move to an old city in ANOTHER country
    mv = []
    for a, bi in HELDP[:100]:
        nb = int(rs.randint(len(OLD)))
        while OLD[nb][1] == OLD[bi][1]: nb = int(rs.randint(len(OLD)))
        mv.append((a, bi, nb))
    sents = [f"{a} lives in {OLD[nb][0]}." for a, bi, nb in mv] + [f"{a} has a home in {OLD[nb][0]}." for a, bi, nb in mv]
    mm = LM(); mm.load_state_dict(st0); mm.train(); op2 = torch.optim.AdamW(mm.parameters(), lr=3e-4, weight_decay=0.01); r2 = np.random.RandomState(SEED + 21)
    for s_ in range(1, 301):
        step(mm, op2, [sents[r2.randint(len(sents))].encode('latin-1') for _ in range(8)] + win(PRES, B_ - 8, r2))
        if s_ in (100, 300):
            new = [(a, nb) for a, bi, nb in mv]; old = [(a, bi) for a, bi, nb in mv]
            nbb, ncc = chain(mm, new, OLD); obb, occ = chain(mm, old, OLD)
            P(f"   BELIEF REVISION {s_} steps: the chain gives the NEW city {nbb:.2f}, NEW country {ncc:.2f} | still the OLD city {obb:.2f}, OLD country {occ:.2f} | "
              f"'lives in' the new city {rank1(mm, liv(new, OLD), np.random.RandomState(3)):.2f}  ({time.time() - T0:.0f}s)")
    P(f"DONE ({time.time() - T0:.0f}s)"); raise SystemExit
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
newp = lambda n: [(person(), i) for i in range(n)]
P("IDEA 2 - taught briefly after pretraining (chance 0.10; citizenship is never taught for these people):")
arm('F1 OLD CITY', F1P, OLD, False)
if PCODE == 2: P(f"DONE ({time.time() - T0:.0f}s)"); raise SystemExit
NEWC = [rs.randint(NC) for _ in range(100)]
arm('F2 NEW, RANDOM', newp(100), [(f"{name(3)}-{CODES[rs.randint(NC)]}", c) for c in NEWC], True)
arm('F3 NEW, TYPED', newp(100), [(f"{name(3)}-{CODES[c]}", c) for c in NEWC], True)
arm('F4 TYPED ONLY', newp(100), [(f"{name(3)}-{CODES[c]}", c) for c in NEWC], False)
P(f"DONE ({time.time() - T0:.0f}s)")
