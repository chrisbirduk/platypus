"""TOY: ROUTING BY A JEPA TRACE OF THE CONTEXT (no effect on the package).
The five local domains (the harness's mini-Pile: Gutenberg, Wikipedia, ArXiv, PubMed, FreeLaw). A FIRST VISIT (the start of each
domain's training region) builds one prototype per slot; a REVISIT is text from the domain's held-out region (documents never seen).
ROUTERS
  BYTES      byte and hashed byte-pair frequency profile (square-root, Hellinger) - a stand-in for the current sequence statistics
  JEPA       a V-JEPA-style byte encoder (online + predictor, moving-average target, L1, whole-word and span masking) trained on 128-byte
             chunks of all five domains; a chunk's code = the target's pooled latent; a window's code = the mean of its chunks' codes
  JEPA-GEN   the same encoder trained and used with every capitalised word of 4-12 letters masked (the store's regex name finder)
  BYTES+JEPA the two distances, each divided by its median, added
PROTOTYPES  centroid (mean code of the first visit), kNN (mean distance to the 10 nearest first-visit windows), diagonal Gaussian.
CONTEXT     one 128-byte chunk, one 1024-byte window, or 2 / 4 / 8 / 16 windows averaged.
READOUTS    1 ROUTING ACCURACY on revisit text (chance 0.20); 2 NOVELTY: each domain left out of the prototypes in turn, AUC of the
            distance to the nearest prototype (unseen domain against seen ones); 3 the CONFUSIONS at one window.
usage: python3 toy_jroute.py [seed] [steps]"""
import sys, os, re, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
torch.set_num_threads(1); rs = np.random.RandomState(SEED); torch.manual_seed(SEED); t0 = time.time()
DATA = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
DOMS = ['prose', 'wiki', 'arxiv', 'pubmed', 'law']; TRAIN_BYTES, VAL_BYTES = 2_600_000, 500_000
RAW = {d: open(os.path.join(DATA, d + '.bin'), 'rb').read() for d in DOMS}
TR = {d: b[:TRAIN_BYTES] for d, b in RAW.items()}; VA = {d: b[-VAL_BYTES:] for d, b in RAW.items()}
FIRST = {d: b[:400_000] for d, b in TR.items()}                            # the first visit: the start of each training region
CH, W = 128, 1024; PAD, MASK = 256, 257; V = 258
NAME = re.compile(rb'\b[A-Z][a-z]{3,11}\b')
def ids(b, gen):
    x = list(b)
    if gen:
        for m in NAME.finditer(b):
            for k in range(m.start(), m.end()): x[k] = MASK
    return x
def chunks(b, n, r, L=CH): return [b[i:i + L] for i in r.randint(0, len(b) - L, size=n)]
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x): h = s.n1(x); x = x + s.a(h, h, h, need_weights=False)[0]; return x + s.m(s.n2(x))
class Enc(nn.Module):
    def __init__(s, d=128, nl=4):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(CH, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl)); s.n = nn.LayerNorm(d)
    def forward(s, X, x_in=None):
        x = (s.emb(X) if x_in is None else x_in) + s.pos[:X.shape[1]]
        for b in s.b: x = b(x)
        return s.n(x)
def mask_spans(X, r):
    M = torch.zeros_like(X, dtype=torch.bool)
    for i in range(X.shape[0]):
        row = bytes([v for v in X[i].tolist() if v < 256]); sp = [m.span() for m in re.finditer(rb'[A-Za-z]+', row)]
        for a, b in sp:
            if r.rand() < 0.15: M[i, a:b] = True
        for _ in range(r.randint(0, 3)): a = r.randint(0, CH - 4); M[i, a:a + r.randint(2, 8)] = True
    return X.masked_fill(M, MASK), M
def train(gen, steps):
    on, tg = Enc(), Enc(); tg.load_state_dict(on.state_dict()); pr = Enc(nl=2)
    for p in tg.parameters(): p.requires_grad_(False)
    op = torch.optim.AdamW(list(on.parameters()) + list(pr.parameters()), lr=5e-4, weight_decay=0.01); r = np.random.RandomState(SEED + 11); ls = []
    for st in range(steps):
        cs = [c for d in DOMS for c in chunks(TR[d], 7, r)]; X = torch.tensor([ids(c, gen) for c in cs])
        Xm, M = mask_spans(X, r)
        with torch.no_grad(): z = tg(X); z = F.layer_norm(z, z.shape[-1:])
        p = pr(Xm, x_in=on(Xm)); l = F.smooth_l1_loss(p[M], z[M])
        op.zero_grad(); l.backward(); op.step(); ls.append(float(l))
        with torch.no_grad():
            for a, b in zip(tg.parameters(), on.parameters()): a.mul_(0.996).add_(b, alpha=0.004)
    print(f"  JEPA{'-GEN' if gen else ''} trained: loss {np.mean(ls[:20]):.3f} -> {np.mean(ls[-100:]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    tg.eval(); return tg
@torch.no_grad()
def jcode(tg, bs, gen):
    """one code per 128-byte chunk: the target's pooled latent"""
    out = []
    for a in range(0, len(bs), 256):
        z = tg(torch.tensor([ids(b, gen) for b in bs[a:a + 256]])); out.append(F.layer_norm(z, z.shape[-1:]).mean(1))
    return torch.cat(out)
def bcode(bs):
    """byte unigram + hashed byte-pair profile, square-rooted"""
    out = np.zeros((len(bs), 256 + 4096), dtype=np.float32)
    for i, b in enumerate(bs):
        a = np.frombuffer(b, dtype=np.uint8).astype(np.int64)
        out[i, :256] = np.bincount(a, minlength=256); out[i, 256:] = np.bincount((a[:-1] * 257 + a[1:]) % 4096, minlength=4096)
    out[:, :256] /= out[:, :256].sum(1, keepdims=True) + 1e-9; out[:, 256:] /= out[:, 256:].sum(1, keepdims=True) + 1e-9
    return torch.tensor(np.sqrt(out))
# windows: first-visit windows (prototypes) and revisit windows (test), each split into 8 chunks
r = np.random.RandomState(SEED + 21); NF, NV = 160, 320
WIN = {('first', d): chunks(FIRST[d], NF, r, W) for d in DOMS}; WIN.update({('re', d): chunks(VA[d], NV, r, W) for d in DOMS})
def win_codes(fn):
    """codes per window (mean over its 8 chunks) and per first chunk"""
    out = {}
    for k, ws in WIN.items():
        cs = [w[i:i + CH] for w in ws for i in range(0, W, CH)]; z = fn(cs).reshape(len(ws), W // CH, -1); out[k] = (z.mean(1), z[:, 0])
    return out
print(f"TOY JEPA ROUTING, seed {SEED}, {STEPS} steps; first visit {NF} windows per domain, revisit {NV} windows per domain from unseen text", flush=True)
CODES = {'BYTES': win_codes(bcode)}
for gen in (False, True):
    tg = train(gen, STEPS); CODES['JEPA-GEN' if gen else 'JEPA'] = win_codes(lambda cs: jcode(tg, cs, gen))
def dists(name, X, protos, kind):
    """distance of each row of X to each slot (rows: contexts, cols: slots)"""
    out = []
    for P in protos:
        if kind == 'centroid': out.append((X - P.mean(0)).norm(dim=-1))
        elif kind == 'knn': out.append(torch.cdist(X, P).topk(min(10, P.shape[0]), largest=False).values.mean(1))
        else: mu, sd = P.mean(0), P.std(0) + 1e-3; out.append((((X - mu) / sd) ** 2).mean(-1).sqrt())
    return torch.stack(out, 1)
def contexts(name, d, k, r):
    """revisit contexts of k windows (k = 0: one 128-byte chunk), as the mean of their codes"""
    win, first = CODES[name][('re', d)]
    if k == 0: return first
    idx = torch.tensor(r.randint(0, win.shape[0], size=(200, k))); return win[idx].mean(1)
def route(names, kind, k, r, doms=DOMS, protos_d=DOMS):
    """distances of revisit contexts of each domain in doms to the prototypes of protos_d; names combined by median-normalised sum"""
    D = None
    for nm in names:
        rr = np.random.RandomState(r)                                       # the same contexts for every router
        P = [CODES[nm][('first', d)][0] for d in protos_d]; X = torch.cat([contexts(nm, d, k, rr) for d in doms]); Dn = dists(nm, X, P, kind)
        Dn = Dn / Dn.median(); D = Dn if D is None else D + Dn
    return D
ROUTERS = {'BYTES': ['BYTES'], 'JEPA': ['JEPA'], 'JEPA-GEN': ['JEPA-GEN'], 'BYTES+JEPA': ['BYTES', 'JEPA'], 'BYTES+JEPA-GEN': ['BYTES', 'JEPA-GEN']}
KS = [(0, '1 chunk (128 B)'), (1, '1 window (1 KB)'), (2, '2 windows'), (4, '4 windows'), (8, '8 windows'), (16, '16 windows')]
print("1 ROUTING ACCURACY on revisit text (chance 0.20), centroid / kNN / Gaussian prototypes:")
y_ = None
for kk, kn in KS:
    line = []
    for rn, nms in ROUTERS.items():
        acc = []
        for kind in ('centroid', 'knn', 'gauss'):
            D = route(nms, kind, kk, 100 + kk); n_ = D.shape[0] // len(DOMS); y = torch.arange(len(DOMS)).repeat_interleave(n_)
            acc.append(float((D.argmin(1) == y).float().mean()))
        line.append(f"{rn} {acc[0]:.2f}/{acc[1]:.2f}/{acc[2]:.2f}")
    print(f"  {kn:16s} | " + " | ".join(line) + f"  ({time.time()-t0:.0f}s)", flush=True)
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); rk = np.concatenate([pos, neg]).argsort().argsort().astype(float) + 1
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
print("2 NOVELTY: each domain left out of the prototypes in turn; AUC of the nearest-prototype distance, unseen domain against seen ones (centroid):")
for kk, kn in KS[1:5:2] + KS[4:5]:
    line = []
    for rn, nms in ROUTERS.items():
        a = []
        for h in DOMS:
            seen = [d for d in DOMS if d != h]; D = route(nms, 'centroid', kk, 200 + kk, doms=DOMS, protos_d=seen); n_ = D.shape[0] // len(DOMS)
            m = D.min(1).values; hi = DOMS.index(h); a.append(auc(m[hi * n_:(hi + 1) * n_].numpy(), np.concatenate([m[i * n_:(i + 1) * n_].numpy() for i in range(len(DOMS)) if i != hi])))
        line.append(f"{rn} {np.mean(a):.2f} (min {min(a):.2f})")
    print(f"  {kn:16s} | " + " | ".join(line) + f"  ({time.time()-t0:.0f}s)", flush=True)
print("3 CONFUSIONS at one window (rows: true domain, columns: routed slot; centroid):")
for rn in ('BYTES', 'JEPA', 'JEPA-GEN'):
    D = route(ROUTERS[rn], 'centroid', 1, 300); n_ = D.shape[0] // len(DOMS); pr_ = D.argmin(1)
    print(f"  {rn:9s} " + " | ".join(f"{d}: " + " ".join(f"{int((pr_[i * n_:(i + 1) * n_] == j).sum())}" for j in range(len(DOMS))) for i, d in enumerate(DOMS)), flush=True)
print(f"DONE {time.time()-t0:.0f}s")
