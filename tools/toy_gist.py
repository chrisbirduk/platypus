"""TOY SEGMENTATION AND GIST TEST (no effect on the package). A small 2-block delta-rule model, token level, is trained on streams of short
documents. Each document is a chain of facts 'A r B t (t) .' (t: one or two of the document's three topic tokens); the next fact starts with
the previous object, so inside a document the next subject is predictable and at a new document it is not. The model also learns to read a
fact back from memory (one-hop, with a distractor fact superposed). Entities of the test streams are never seen in training.

TEST 1, SEGMENTATION (held-out streams, carried memory): where do three signals peak?
  token surprise (-log p of the token), JEPA surprise (a LeJEPA head predicts the trace of the next 4 tokens from the trace of the memory so
  far; error at each position), write novelty (delta rule: |v - S k| / |v| at each write, last block). AUC of each signal at document starts
  against ordinary tokens and against within-document fact starts (the event-boundary test; fixed-length cuts score 0.5 by construction),
  and at fact starts against other tokens. Plus the CAPACITY CURVE: one-hop recall of a stream's first fact from one memory that has gone on
  to read n more facts.
TEST 2, GIST (disambiguation): the same name has a fact with the same relation in two documents ('X r Y' in one, 'X r Z' in the other).
  A LeJEPA head maps the trace of the question 'X r' to the trace of the fact; candidates are all test facts. Gist = the trace of the
  document's other facts. Variants: PLAIN (no gist), ADD (gist added in the same trace space), ORTHOGONAL (gist in its own subspace,
  concatenated), MEMORY (the fact and the question read after the document's other facts, i.e. superposed in the delta memory itself).
  Reported: top-1 among all facts, the twin test (right fact over its same-name twin; chance 0.5), and errors that pick another fact of the
  same document (what a too-strong gist does). Training and test names are disjoint.
usage: python3 toy_gist.py [seed] [steps_A]"""
import sys, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS_A = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
rs = np.random.RandomState(SEED); torch.manual_seed(SEED)

# ---------------------------------------------------------------- vocabulary and documents
NREL, NTOP, NENT = 8, 40, 3000
SO, CONN, DOT, PAD = 0, 1, 2, 3; R0 = 4; T0 = R0 + NREL; E0 = T0 + NTOP; V = E0 + NENT
ent = rs.permutation(np.arange(E0, V)); TRAIN_E, TEST_E = list(ent[:2200]), list(ent[2200:])
def make_doc(pool, nf=None, names=None):
    nf = nf or int(rs.randint(2, 6)); tp = [T0 + int(x) for x in rs.choice(NTOP, 3, replace=False)]
    nm = names or [int(pool[int(i)]) for i in rs.choice(len(pool), nf + 1, replace=False)]
    sents = []
    for i in range(nf):
        s = [nm[i], R0 + int(rs.randint(NREL)), nm[i + 1]] + [tp[int(j)] for j in rs.choice(3, int(rs.randint(1, 3)), replace=False)] + [DOT]
        sents.append(s)
    return {'s': sents, 'tp': tp}
def make_stream(pool, nd=3):
    docs = [make_doc(pool) for _ in range(nd)]; toks, doc_start, fact_start = [], [], []
    for di, d in enumerate(docs):
        for si, s in enumerate(d['s']):
            if si == 0 and di > 0: doc_start.append(len(toks))
            elif si > 0: fact_start.append(len(toks))
            toks += s
    return toks, doc_start, fact_start, docs

# ---------------------------------------------------------------- model (as toy_compose.py, with write novelty and per-position states)
class Delta(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.dh = d // h; s.Wk = nn.Linear(d, d, bias=False); s.Wv = nn.Linear(d, d, bias=False); s.Wb = nn.Linear(d, h); s.Wo = nn.Linear(d, d, bias=False)
    def forward(s, x, S0=None, rec=None):
        B, T, D = x.shape; k = F.normalize(s.Wk(x).view(B, T, s.h, s.dh), dim=-1); v = s.Wv(x).view(B, T, s.h, s.dh); b = torch.sigmoid(s.Wb(x))
        S = torch.zeros(B, s.h, s.dh, s.dh) if S0 is None else S0; out = []
        for t in range(T):
            kt, vt = k[:, t], v[:, t]; r_ = vt - torch.einsum('bhvk,bhk->bhv', S, kt)
            if rec is not None: rec.setdefault('nov', []).append((r_.norm(dim=-1) / vt.norm(dim=-1).clamp_min(1e-6)).mean(-1)); rec.setdefault('S', []).append(S)
            S = S + b[:, t, :, None, None] * torch.einsum('bhv,bhk->bhvk', r_, kt); out.append(torch.einsum('bhvk,bhk->bhv', S, kt))
        return s.Wo(torch.stack(out, 1).reshape(B, T, D)), S
class Toy(nn.Module):
    def __init__(s, d=128, h=4, nb=2):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Embedding(256, d); s.nb = nb
        s.mix = nn.ModuleList(Delta(d, h) for _ in range(nb)); s.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(nb))
        s.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(nb)); s.n2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(nb)); s.out = nn.Linear(d, V)
    def run(s, X, S0=None, rec=None):
        x = s.emb(X) + s.pos(torch.arange(X.shape[1]) % 256); Ss = []
        for i in range(s.nb):
            y, S = s.mix[i](s.n1[i](x), None if S0 is None else S0[i], rec if (rec is not None and i == s.nb - 1) else None)
            x = x + y; x = x + s.mlp[i](s.n2[i](x)); Ss.append(S)
        return s.out(x), Ss
def pad(seqs):
    L = max(len(q) for q in seqs); return torch.tensor([q + [PAD] * (L - len(q)) for q in seqs]), torch.tensor([[1.0] * len(q) + [0.0] * (L - len(q)) for q in seqs])
def addm(a, b): return [x + y for x, y in zip(a, b)]

# ---------------------------------------------------------------- stage A: language model on streams + reading facts back from memory
def lossA(m):
    st = [make_stream(TRAIN_E)[0] for _ in range(16)]; X, M_ = pad(st); lg, _ = m.run(X)
    l1 = (F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction='none') * M_[:, 1:].reshape(-1)).sum() / M_[:, 1:].sum()
    fa = [make_doc(TRAIN_E, 1)['s'][0] for _ in range(32)]; fd = [make_doc(TRAIN_E, 1)['s'][0] for _ in range(32)]
    Xa, _ = pad(fa); Xd, _ = pad(fd); Mm = addm(m.run(Xa)[1], m.run(Xd)[1])
    lg2, _ = m.run(torch.tensor([[f[0], f[1]] for f in fa]), Mm)
    l2 = F.cross_entropy(lg2[:, -1], torch.tensor([f[2] for f in fa]))
    return l1 + l2
def sigreg(z, ns=128, nk=17, tm=5.0):
    U = F.normalize(torch.randn(z.shape[1], ns), dim=0); x = z @ U; t = torch.linspace(0, tm, (nk + 1) // 2); tx = x[:, :, None] * t
    err = ((torch.cos(tx).mean(0) - torch.exp(-t ** 2 / 2)) ** 2 + torch.sin(tx).mean(0) ** 2) * torch.exp(-t ** 2 / 2)
    return z.shape[0] * 2 * torch.trapezoid(err, t, dim=-1).mean()
def auc(pos, neg):
    p, n = np.asarray(pos), np.asarray(neg)
    if not len(p) or not len(n): return float('nan')
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())

t0 = time.time(); m = Toy(); op = torch.optim.Adam(m.parameters(), lr=1e-3)
for st in range(STEPS_A):
    l = lossA(m); op.zero_grad(); l.backward(); op.step()
    if (st + 1) % max(1, STEPS_A // 5) == 0: print(f"stage A {st+1}: loss {float(l):.3f} ({time.time()-t0:.0f}s)", flush=True)
for p in m.parameters(): p.requires_grad_(False)

# ---------------------------------------------------------------- TEST 1: segmentation
def states_stream(toks):
    rec = {}; lg, _ = m.run(torch.tensor([toks]), None, rec); lp = F.log_softmax(lg[0], -1)
    sur = [0.0] + [float(-lp[t - 1, toks[t]]) for t in range(1, len(toks))]
    S = torch.stack(rec['S'], 1)[0].reshape(len(toks), -1)                  # memory BEFORE each token's write (what the stream so far left)
    nov = torch.stack(rec['nov'], 1)[0].tolist(); return sur, S, nov
class Head(nn.Module):
    def __init__(s, din, dz=64): super().__init__(); s.f = nn.Linear(din, dz)
    def forward(s, x): return s.f(x)
# JEPA surprise head: trace of the memory before position t -> trace of the span t..t+3 read alone
def span_mem(toks, t):
    _, Ss = m.run(torch.tensor([toks[t:t + 4]])); return Ss[-1].reshape(1, -1)
def jepa_data(n):
    Z = [];
    for _ in range(n):
        toks, ds, fs, _ = make_stream(TRAIN_E); _, S, _ = states_stream(toks)
        for t in range(1, len(toks) - 4): Z.append((S[t], toks[t:t + 4]))
    return Z
with torch.no_grad(): TR = jepa_data(120)
with torch.no_grad():
    SP = torch.cat([m.run(pad([s for _, s in TR[i:i + 256]])[0])[1][-1].reshape(len(TR[i:i + 256]), -1) for i in range(0, len(TR), 256)])
    SM = torch.stack([s for s, _ in TR])
mu_s, sd_s = SM.mean(0), SM.std(0) + 1e-6; mu_p, sd_p = SP.mean(0), SP.std(0) + 1e-6
Eq, Ek, Pj = Head(SM.shape[1]), Head(SP.shape[1]), nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
oj = torch.optim.AdamW(list(Eq.parameters()) + list(Ek.parameters()) + list(Pj.parameters()), lr=1e-3, weight_decay=0.05)
for st in range(1500):
    ix = torch.as_tensor(rs.choice(len(TR), 256, replace=False)); zq = Eq((SM[ix] - mu_s) / sd_s); zk = Ek((SP[ix] - mu_p) / sd_p)
    l = 0.95 * ((Pj(zq) - zk) ** 2).sum(1).mean() + 0.05 * sigreg(torch.cat([zq, zk])); oj.zero_grad(); l.backward(); oj.step()
R = {'tok': [[], [], []], 'jepa': [[], [], []], 'nov': [[], [], []]}           # [doc starts, fact starts, other tokens]
with torch.no_grad():
    for _ in range(60):
        toks, ds, fs, _ = make_stream(TEST_E, 4); sur, S, nov = states_stream(toks)
        span = torch.cat([span_mem(toks, t) for t in range(1, len(toks) - 4)]); jz = ((Pj(Eq((S[1:len(toks) - 4] - mu_s) / sd_s)) - Ek((span - mu_p) / sd_p)) ** 2).sum(1).tolist()
        js = [float('nan')] + jz
        for t in range(1, len(toks) - 4):
            g = 0 if t in ds else 1 if t in fs else 2
            R['tok'][g].append(sur[t]); R['jepa'][g].append(js[t]); R['nov'][g].append(nov[t])
    cap = []                                                                  # capacity: recall of fact 1 after n more facts in one memory
    for n in (0, 1, 2, 4, 8, 12, 16):
        ok = []
        for _ in range(40):
            facts = [make_doc(TEST_E, 1)['s'][0] for _ in range(n + 1)]; flat = [x for f in facts for x in f]
            _, Ss = m.run(torch.tensor([flat])); lg, _ = m.run(torch.tensor([[facts[0][0], facts[0][1]]]), Ss)
            ok.append(int(int(lg[0, -1].argmax()) == facts[0][2]))
        cap.append((n, float(np.mean(ok))))
print("TEST 1 SEGMENTATION (held-out streams, 4 documents each): AUC at DOCUMENT STARTS vs other tokens | vs within-document FACT STARTS | FACT STARTS vs other tokens")
for k, nm in (('tok', 'token surprise'), ('jepa', 'JEPA surprise'), ('nov', 'write novelty')):
    d, f, o = R[k]; print(f"  {nm:15s} {auc(d, o):.2f} | {auc(d, f):.2f} | {auc(f, o):.2f}   (n {len(d)}, {len(f)}, {len(o)})")
print("  CAPACITY: one-hop recall of the first fact after n more facts in the same memory: " + ", ".join(f"n={n} {r:.2f}" for n, r in cap) + f"  ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- TEST 2: gist
def twin_docs(pool, n):
    """n pairs of documents; each pair shares a name X with the same relation r: 'X r Y' in one document, 'X r Z' in the other"""
    out = []
    for _ in range(n):
        a = make_doc(pool, int(rs.randint(3, 5))); b = make_doc(pool, int(rs.randint(3, 5)))
        i, j = int(rs.randint(len(a['s']))), int(rs.randint(len(b['s'])))
        X = a['s'][i][0]; r = a['s'][i][1]; b['s'][j][0] = X; b['s'][j][1] = r
        if j > 0: b['s'][j - 1][2] = X                                        # keep the chain inside b
        out.append((a, i, b, j))
    return out
def items(pairs):
    """per twin fact: (question, fact, context = the document's other facts, doc id, twin index)"""
    it = []
    for p, (a, i, b, j) in enumerate(pairs):
        for d, k, tw in ((a, i, 2 * p + 1), (b, j, 2 * p)):
            ctx = [x for q, s in enumerate(d['s']) if q != k for x in s][:24]
            it.append({'q': d['s'][k][:2], 'f': d['s'][k], 'c': ctx, 'doc': 2 * p + (d is b), 'tw': tw})
    for q in it: q['ans'] = q['f'][2]
    return it
def final_states(seqs):
    res = [None] * len(seqs); byL = {}
    for i, q in enumerate(seqs): byL.setdefault(len(q), []).append(i)
    for L, ix in byL.items():
        for a in range(0, len(ix), 128):
            jj = ix[a:a + 128]; _, Ss = m.run(torch.tensor([seqs[j] for j in jj])); v = Ss[-1].reshape(len(jj), -1)
            for q, j in enumerate(jj): res[j] = v[q]
    return torch.stack(res)
with torch.no_grad():
    TRI, TEI = items(twin_docs(TRAIN_E, 300)), items(twin_docs(TEST_E, 100))
    def feats(I):
        return {'q': final_states([x['q'] for x in I]), 'f': final_states([x['f'] for x in I]), 'c': final_states([x['c'] for x in I]),
                'qc': final_states([x['c'] + x['q'] for x in I]), 'fc': final_states([x['c'] + x['f'] for x in I])}
    FT, FE = feats(TRI), feats(TEI)
nz = {k: (FT[k].mean(0), FT[k].std(0) + 1e-6) for k in FT}
N = lambda Fd, k: (Fd[k] - nz[k][0]) / nz[k][1]
def train_variant(var):
    din = FT['q'].shape[1]; Eq_, Ef_, Eg_ = Head(din), Head(din), Head(din); P_ = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    o = torch.optim.AdamW([*Eq_.parameters(), *Ef_.parameters(), *Eg_.parameters(), *P_.parameters()], lr=1e-3, weight_decay=0.05)
    def enc(Fd):
        if var == 'memory': return P_(Eq_(N(Fd, 'qc'))), Ef_(N(Fd, 'fc'))
        q, f = P_(Eq_(N(Fd, 'q'))), Ef_(N(Fd, 'f'))
        if var == 'plain': return q, f
        if var == 'add': g = Ef_(N(Fd, 'c')); return q + g, f + g                    # the gist in the same space as the facts
        g = Eg_(N(Fd, 'c')); return torch.cat([q, g], 1), torch.cat([f, g], 1)      # orthogonal: its own subspace
    for st in range(1500):
        ix = torch.as_tensor(rs.choice(len(TRI), 128, replace=False)); Fd = {k: v[ix] for k, v in FT.items()}
        q, f = enc(Fd); qn, fn = F.normalize(q, dim=-1), F.normalize(f, dim=-1)
        l = 0.95 * ((q - f) ** 2).sum(1).mean() + 0.05 * sigreg(torch.cat([q, f])) + F.cross_entropy(qn @ fn.T / 0.1, torch.arange(len(ix)))
        o.zero_grad(); l.backward(); o.step()
    with torch.no_grad():
        q, f = enc(FE); sim = F.normalize(q, dim=-1) @ F.normalize(f, dim=-1).T; top = sim.argmax(1)
        n = len(TEI); right = (top == torch.arange(n)).float().mean()
        tw = float(np.mean([float(sim[i, i] > sim[i, TEI[i]['tw']]) for i in range(n)]))
        same = float(np.mean([float(int(top[i]) != i and TEI[int(top[i])]['doc'] == TEI[i]['doc']) for i in range(n)]))
    return float(right), tw, same
print(f"TEST 2 GIST: {len(TRI)} training and {len(TEI)} held-out twin facts (names never seen in training); retrieval of the fact from the question among all {len(TEI)} held-out facts")
for var, nm in (('plain', 'PLAIN (no gist)'), ('add', 'ADD (gist in the same space)'), ('orth', 'ORTHOGONAL (gist subspace)'), ('memory', 'MEMORY (read after the document)')):
    r, tw, same = train_variant(var)
    print(f"  {nm:34s} top-1 {r:.2f} | twin test {tw:.2f} (chance 0.5) | errors to another fact of the same document {same:.2f}", flush=True)
print(f"DONE {time.time()-t0:.0f}s")
