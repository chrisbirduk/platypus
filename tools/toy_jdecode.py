"""TOY: COMPOSE TWO JEPA TRACES AND GENERATE TEXT FROM THE COMPOSITION (no effect on the package).
Question: the model keys each block of context by a JEPA code and finds blocks that are alike. Can two such codes be COMPOSED and the
composition turned back into text - a generated dream that holds what both blocks hold (e.g. A r1 B in one block, B r2 C in another ->
text linking A to C)?
DATA      128-byte blocks of the five local domains (mini-Pile), each with one planted fact sentence 'A r B.' (invented names with q/x/z,
          8 relations) at a random place; facts form chains A r1 B, B r2 C (the two facts in different blocks, usually different domains).
          Each fact is planted in 3 blocks. The domains are VISITED IN ORDER (continual; nothing is pretrained).
ENCODERS  JEPA (online + predictor, moving-average target, L1 on masked spans and whole words), two arms:
          PLAIN    the bytes as they are
          GENERIC  every name (capitalised word of 4-12 letters) masked before encoding, as the index's generic arm; names kept as slots
DECODERS  per arm, a small causal byte transformer reading the block's TARGET code (projected into 4 prefix positions) and trained to
          write the block (GENERIC: the masked block). Trained beside the encoder, on the visits so far.
COMPOSERS MEAN (z_a + z_b) / 2; CENTRED SUM mu + (z_a - mu) + (z_b - mu) (mu = mean code of the stored blocks); NEAREST STORED BLOCK to
          the mean (retrieval instead of generation: does decoding add anything over finding a block?).
READOUTS  1 RECONSTRUCTION of held-out blocks from their own code: byte accuracy (teacher forced), greedy decode keeps the fact's
            names (PLAIN) / its relation phrase (both), re-encoded code cosine to the source code.
          2 COMPOSITION of chain pairs (A-B block + B-C block) against CONTROL pairs (A-B block + an unrelated chain's block): the decoded
            text holds names / relation phrases of both blocks; LINKED A..C (A and C within one sentence, PLAIN); re-encoded cosine to the
            composed code and to each source code.
          3 JUDGE (as the dream judge, by a small byte LM trained on the same visits): bits saved on the two fact sentences 'A r1 B.' and
            'B r2 C.' when the composed text is read first, against a random block read first - for the decoded composition, the nearest
            stored block and a TEXT SPLICE of the two fact sentences (what dreams do now); chain and control pairs.
usage: python3 toy_jdecode.py [seed] [steps per visit]"""
import sys, os, re, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SPV = int(sys.argv[2]) if len(sys.argv) > 2 else 600
torch.set_num_threads(int(os.environ.get('JD_THREADS', '1'))); torch.manual_seed(SEED); t0 = time.time()
DATA = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
DOMS = ['prose', 'wiki', 'arxiv', 'pubmed', 'law']; ND = len(DOMS)
RAW = {i: open(os.path.join(DATA, d + '.bin'), 'rb').read()[:3_000_000] for i, d in enumerate(DOMS)}
CH = 128; MASK = 257; V = 258; NP = 4; DZ = 96
NAME = re.compile(rb'\b[A-Z][a-z]{3,11}\b')
r = np.random.RandomState(SEED)
SYL = ['qui', 'zo', 'xa', 'ber', 'tal', 'mon', 'ris', 'van', 'dru', 'pel', 'kor', 'sen', 'lu', 'fa', 'ni', 'gal']
def name(r):
    while True:
        n = ''.join(r.choice(SYL, r.randint(2, 4))); n = n[0].upper() + n[1:]
        if 5 <= len(n) <= 11 and any(c in n for c in 'qxz'): return n
RELS = ['works with', 'reports to', 'lives near', 'is linked to', 'trained under', 'is a partner of', 'answers to', 'writes to']
NCH = 400; used = set(); CHAINS = []
while len(CHAINS) < NCH:
    a, b, c = name(r), name(r), name(r)
    if len({a, b, c}) < 3 or {a, b, c} & used: continue
    used |= {a, b, c}; CHAINS.append((a, b, c, int(r.randint(8)), int(r.randint(8)), int(r.randint(ND)), int(r.randint(ND))))
HELD = set(range(0, NCH, 4))                                        # chains whose blocks are never trained on (readouts)
def fact(a, k, b): return f"{a} {RELS[k]} {b}.".encode()
def block(d, f, r):
    """a 128-byte block of domain d's text with the fact sentence f at a random word boundary"""
    i = r.randint(0, len(RAW[d]) - 400); t = RAW[d][i:i + 400]; t = t[t.find(b' ') + 1:]
    p = r.randint(0, CH - len(f) - 1); b = t[:p] + b' ' + f + b' ' + t[p:]; return b[:CH].ljust(CH, b' ')
BLK = {}                                                           # (chain, 0/1) -> 3 blocks
for ci, (a, b, c, k1, k2, d1, d2) in enumerate(CHAINS):
    BLK[(ci, 0)] = [(d1, block(d1, fact(a, k1, b), r)) for _ in range(3)]; BLK[(ci, 1)] = [(d2, block(d2, fact(b, k2, c), r)) for _ in range(3)]
def plain_text(d, n, r):
    out = []
    for _ in range(n):
        i = r.randint(0, len(RAW[d]) - CH - 1); out.append(RAW[d][i:i + CH])
    return out
def ids(b, gen):
    x = list(b)
    if gen:
        for m in NAME.finditer(b):
            for k in range(m.start(), m.end()): x[k] = MASK
    return x
def T(bs, gen=False): return torch.tensor([ids(b, gen) for b in bs])
class Blk(nn.Module):
    def __init__(s, d, h, causal=False):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d); s.causal = causal
    def forward(s, x):
        h = s.n1(x); m = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), 1) if s.causal else None
        x = x + s.a(h, h, h, attn_mask=m, need_weights=False)[0]; return x + s.m(s.n2(x))
class Enc(nn.Module):
    def __init__(s, d=DZ, nl=3, L=CH):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(L, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl)); s.n = nn.LayerNorm(d)
    def forward(s, X, x_in=None):
        x = (s.emb(X) if x_in is None else x_in) + s.pos[:X.shape[1]]
        for b in s.b: x = b(x)
        return s.n(x)
def mask_spans(X, r):
    M = torch.zeros_like(X, dtype=torch.bool)
    for i in range(X.shape[0]):
        row = bytes([v if v < 256 else 32 for v in X[i].tolist()]); sp = [m.span() for m in re.finditer(rb'[A-Za-z]+', row)]
        for a, b in sp:
            if r.rand() < 0.15: M[i, a:b] = True
        for _ in range(r.randint(1, 4)): a = r.randint(0, CH - 4); M[i, a:a + r.randint(2, 8)] = True
    return X.masked_fill(M, MASK), M
class Dec(nn.Module):
    """causal byte LM reading a code through NP prefix positions; vocabulary 258 (MASK = a name slot)"""
    def __init__(s, d=128, nl=3, ctx=False):
        super().__init__(); s.emb = nn.Embedding(V + 1, d); s.pos = nn.Parameter(torch.randn(NP + 2 * CH + 1, d) * 0.02); s.zp = nn.Linear(DZ, NP * d)
        s.b = nn.ModuleList(Blk(d, 4, causal=True) for _ in range(nl)); s.n = nn.LayerNorm(d); s.out = nn.Linear(d, V); s.d = d
    def forward(s, z, X):
        """z (B, DZ) or None; X (B, T) inputs (BOS = V prepended by the caller); returns logits for every input position"""
        x = s.emb(X)
        if z is not None: x = torch.cat([s.zp(z).view(-1, NP, s.d), x], 1)
        x = x + s.pos[:x.shape[1]]
        for b in s.b: x = b(x)
        x = s.n(x); return s.out(x[:, NP:] if z is not None else x)
def bos(X): return torch.cat([torch.full((X.shape[0], 1), V), X[:, :-1]], 1)
class Arm:
    def __init__(s, gen):
        s.gen = gen; s.on, s.tg, s.pr = Enc(), Enc(), Enc(nl=1); s.tg.load_state_dict(s.on.state_dict()); s.dec = Dec()
        for p in s.tg.parameters(): p.requires_grad_(False)
        s.op = torch.optim.AdamW(list(s.on.parameters()) + list(s.pr.parameters()), lr=5e-4, weight_decay=0.01)
        s.od = torch.optim.AdamW(s.dec.parameters(), lr=1e-3, weight_decay=0.01); s.r = np.random.RandomState(SEED + 11 + gen); s.lj, s.ld = [], []
    def step(s, bs):
        X = T(bs, s.gen); Xm, M = mask_spans(X, s.r)
        with torch.no_grad(): z = s.tg(X); z = F.layer_norm(z, z.shape[-1:])
        p = s.pr(Xm, x_in=s.on(Xm)); l = F.smooth_l1_loss(p[M], z[M]); s.op.zero_grad(); l.backward(); s.op.step(); s.lj.append(l.item())
        with torch.no_grad():
            for a, b in zip(s.tg.parameters(), s.on.parameters()): a.mul_(0.996).add_(b, alpha=0.004)
        zc = s.code(bs); lg = s.dec(zc, bos(X)); ld = F.cross_entropy(lg.reshape(-1, V), X.reshape(-1)); s.od.zero_grad(); ld.backward(); s.od.step(); s.ld.append(ld.item())
    @torch.no_grad()
    def code(s, bs):
        out = []
        for a in range(0, len(bs), 256):
            z = s.tg(T(bs[a:a + 256], s.gen)); out.append(F.layer_norm(z, z.shape[-1:]).mean(1))
        return torch.cat(out)
    @torch.no_grad()
    def decode(s, z, n=CH):
        X = torch.full((z.shape[0], 1), V)
        for t in range(n):
            lg = s.dec(z, X)[:, -1]; X = torch.cat([X, lg.argmax(-1, keepdim=True)], 1)
        return [bytes([v if v < 256 else 0 for v in row]) for row in X[:, 1:].tolist()]           # name slots shown as byte 0
ARMS = {'PLAIN': Arm(False), 'GENERIC': Arm(True)}
LM = Dec(nl=2); olm = torch.optim.AdamW(LM.parameters(), lr=1e-3, weight_decay=0.01); llm = []   # reference byte LM (the judge): 2-block windows
rt = np.random.RandomState(SEED + 41)
print(f"TOY JEPA DECODE, seed {SEED}, {SPV} steps per visit, {NCH} chains ({len(HELD)} held out), visits in domain order (continual, nothing pretrained)", flush=True)
for d in range(ND):
    mine = [blk for (ci, j), L in BLK.items() if ci not in HELD for (dd, blk) in L if dd == d]
    for st in range(SPV):
        bs = [mine[i] for i in rt.randint(0, len(mine), 12)] + plain_text(d, 12, rt)
        for a in ARMS.values(): a.step(bs)
        pairs = [b1 + b2 for b1, b2 in zip(bs[:12], plain_text(d, 12, rt))] + [p1 + p2 for p1, p2 in zip(plain_text(d, 6, rt), plain_text(d, 6, rt))]   # judge LM: two blocks
        X = torch.tensor([list(p) for p in pairs]); lg = LM(None, bos(X)); l = F.cross_entropy(lg.reshape(-1, V), X.reshape(-1)); olm.zero_grad(); l.backward(); olm.step(); llm.append(l.item())
    print(f"visit {d + 1} {DOMS[d]:6s}: JEPA loss PLAIN {np.mean(ARMS['PLAIN'].lj[-50:]):.3f} GENERIC {np.mean(ARMS['GENERIC'].lj[-50:]):.3f} | decoder bits/byte "
          f"PLAIN {np.mean(ARMS['PLAIN'].ld[-50:]) / math.log(2):.3f} GENERIC {np.mean(ARMS['GENERIC'].ld[-50:]) / math.log(2):.3f} | judge LM {np.mean(llm[-50:]) / math.log(2):.3f}  ({time.time() - t0:.0f}s)", flush=True)
# ---------------- readouts on held-out chains ----------------
HL = sorted(HELD); HL = HL[:80]; rr = np.random.RandomState(SEED + 77)
def has(t, s): return s.encode() in t
def linked(t, a, c):
    for sent in re.split(rb'[.\n]', t):
        if a.encode() in sent and c.encode() in sent: return True
    return False
def rel_hit(t, k): return RELS[k].encode() in t
print("1 RECONSTRUCTION of held-out blocks from their own code (greedy decode)", flush=True)
for an, A in ARMS.items():
    bs = [BLK[(ci, 0)][0][1] for ci in HL]; X = T(bs, A.gen); z = A.code(bs)
    with torch.no_grad(): acc = float((A.dec(z, bos(X)).argmax(-1) == X).float().mean())
    dec = A.decode(z); z2 = A.code([t.replace(b'\x00', b' ') for t in dec])
    nm = np.mean([has(t, CHAINS[ci][0]) and has(t, CHAINS[ci][1]) for t, ci in zip(dec, HL)]); rl = np.mean([rel_hit(t, CHAINS[ci][3]) for t, ci in zip(dec, HL)])
    print(f"  {an:7s} | next-byte accuracy (teacher forced) {acc:.2f} | greedy decode holds both names {nm:.2f} | the relation phrase {rl:.2f} | "
          f"re-encoded cosine to the source {float(F.cosine_similarity(z, z2).mean()):.2f} | e.g. {dec[0][:90]!r} <- {bs[0][:90]!r}", flush=True)
@torch.no_grad()
def judge_bits(ctx, tgt):
    """bits of tgt after ctx under the judge LM (ctx, tgt bytes; both padded/cut to CH)"""
    X = torch.tensor([list(c[:CH].ljust(CH, b' ') + t[:CH].ljust(CH, b' ')) for c, t in zip(ctx, tgt)])
    lp = F.log_softmax(LM(None, bos(X)), -1); out = []
    for i, t in enumerate(tgt):
        n = min(len(t), CH); out.append(float(-lp[i, CH:CH + n].gather(-1, X[i, CH:CH + n, None]).sum()) / math.log(2))
    return np.array(out)
print("2 COMPOSITION of two blocks (held-out chains: A-B block + B-C block; CONTROL: A-B block + another chain's block)", flush=True)
ctrl = [HL[(i + 7) % len(HL)] for i in range(len(HL))]
for an, A in ARMS.items():
    za = A.code([BLK[(ci, 0)][0][1] for ci in HL]); zb = A.code([BLK[(ci, 1)][0][1] for ci in HL]); zx = A.code([BLK[(cj, 1)][0][1] for cj in ctrl])
    store = [b for L in BLK.values() for _, b in L[:1]]; Zs = A.code(store); mu = Zs.mean(0)
    for kind, z2 in (('CHAIN', zb), ('CONTROL', zx)):
        for cn, zc in (('MEAN', (za + z2) / 2), ('CENTRED SUM', mu + (za - mu) + (z2 - mu))):
            dec = A.decode(zc); zr = A.code([t.replace(b'\x00', b' ') for t in dec])
            o2 = [CHAINS[ci] if kind == 'CHAIN' else CHAINS[cj] for ci, cj in zip(HL, ctrl)]
            nA = np.mean([has(t, CHAINS[ci][0]) for t, ci in zip(dec, HL)]); nE = np.mean([has(t, o[2]) for t, o in zip(dec, o2)])
            both_rel = np.mean([rel_hit(t, CHAINS[ci][3]) and rel_hit(t, o[4]) for t, ci, o in zip(dec, HL, o2)])
            lk = np.mean([linked(t, CHAINS[ci][0], o[2]) for t, ci, o in zip(dec, HL, o2)])
            print(f"  {an:7s} {kind:7s} {cn:11s} | holds A {nA:.2f}, the other block's end {nE:.2f}, both relation phrases {both_rel:.2f}, LINKED A..end {lk:.2f} | "
                  f"re-encoded cosine to the composition {float(F.cosine_similarity(zr, zc).mean()):.2f}, to block 1 {float(F.cosine_similarity(zr, za).mean()):.2f}, "
                  f"to block 2 {float(F.cosine_similarity(zr, z2).mean()):.2f} | e.g. {dec[0][:100]!r}", flush=True)
print("3 JUDGE: bits saved on the two fact sentences when the text is read first (against a random block first); held-out chains; TEXT SPLICE holds both sentences verbatim, so it is the copying ceiling", flush=True)
rnd = plain_text(0, len(HL), rr)
for an, A in ARMS.items():
    za = A.code([BLK[(ci, 0)][0][1] for ci in HL]); store = [b for L in BLK.values() for _, b in L[:1]]; Zs = A.code(store)
    for kind in ('CHAIN', 'CONTROL'):
        o2 = [ci if kind == 'CHAIN' else cj for ci, cj in zip(HL, ctrl)]
        z2 = A.code([BLK[(cj, 1)][0][1] for cj in o2]); zc = (za + z2) / 2
        f1 = [fact(*[CHAINS[ci][i] for i in (0, 3, 1)]) for ci in HL]; f2 = [fact(*[CHAINS[cj][i] for i in (1, 4, 2)]) for cj in o2]
        base = judge_bits(rnd, f1) + judge_bits(rnd, f2)
        cand = {'DECODED': [t.replace(b'\x00', b' ') for t in A.decode(zc)],
                'NEAREST STORED': [store[int(i)] for i in torch.cdist(zc, Zs).argmin(1)],
                'TEXT SPLICE': [x + b' ' + y for x, y in zip(f1, f2)]}
        out = []
        for k, ctx in cand.items():
            g = base - (judge_bits(ctx, f1) + judge_bits(ctx, f2)); out.append(f"{k} {np.mean(g):+.1f} bits")
        print(f"  {an:7s} {kind:7s} | " + ' | '.join(out), flush=True)
print(f"done ({time.time() - t0:.0f}s)", flush=True)
