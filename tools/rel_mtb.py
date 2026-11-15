"""RELATION TYPES WITH THE NAMES MASKED, AND A MATCHING-THE-BLANKS RELATION KEY (15 Nov 2026; tools/rel_mtb.py) - offline, stand-in trunk.
tools/rel_types.py: pooled states of 'A <words> C.' type real records better than their wordings (purity 0.40 vs 0.34) but names dominate;
projecting a learned name subspace out hurt. Research:
  1 MASK THE NAMES (Peng et al., EMNLP 2020, 'Learning from Context or Names?'): read 'someone <words> something.' - or with the names'
    entity types as placeholders ('a person <words> a place.') - so the representation carries the context, not the names
  2 MATCHING THE BLANKS (Baldini Soares et al., ACL 2019): a relation key trained label-free - two records of the SAME name pair (different
    sentences) are positives, other pairs negatives (InfoNCE); with the names masked the key must read the relation from the words
All label-free; scored (evaluation only) as rel_types: k-means types (64) - purity / NMI against Wikidata on the sample's records with one
relation; AMIE rules over the types and their typed proposals right (the head type's commonest relation holds).
Usage: TG_D=192 TG_CKPT=<model> python3 tools/rel_mtb.py <train_distant.json.gz>   env RT_N (30000), RT_K (64), MT_PAIRS (20000), MT_STEPS (3000)"""
import os, sys, json, gzip, time, types, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
sys.path.insert(0, '.')
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2)
DW = int(os.environ.get('TG_D', 192)); CK = os.environ['TG_CKPT']; L = 128; PAD = 256; V = 257
NREC, KT, NPAIR, STEPS = int(os.environ.get('RT_N', 30000)), int(os.environ.get('RT_K', 64)), int(os.environ.get('MT_PAIRS', 20000)), int(os.environ.get('MT_STEPS', 3000))
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.a = nn.MultiheadAttention(d, h, batch_first=True); s.m = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d)
    def forward(s, x, am):
        h = s.n1(x); x = x + s.a(h, h, h, attn_mask=am, need_weights=False)[0]; return x + s.m(s.n2(x))
class LM(nn.Module):
    def __init__(s, d=DW, nl=3):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(L, d) * 0.02); s.b = nn.ModuleList(Blk(d, 4) for _ in range(nl))
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, V)
    def states(s, X):
        T = X.shape[1]; am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1); x = s.emb(X) + s.pos[:T]; hs = []
        for b in s.b: x = b(x, am); hs.append(x)
        return hs
lm = LM(); lm.load_state_dict(torch.load(CK)); lm.eval()
@torch.no_grad()
def pooled(texts):
    out = []
    for i0 in range(0, len(texts), 128):
        ch = [t.encode('latin-1', 'replace')[:L] for t in texts[i0:i0 + 128]]; T = max(len(b) for b in ch)
        X = torch.full((len(ch), T), PAD, dtype=torch.long)
        for k, b in enumerate(ch): X[k, :len(b)] = torch.tensor(list(b))
        hs = lm.states(torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1)); m = (X != PAD).float()[:, :, None]
        out.append(torch.cat([(h * m).sum(1) / m.sum(1) for h in (hs[1], hs[2])], -1).numpy())   # middle + last block, mean
    return np.concatenate(out)
import platypus.inference.premises as pm
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10)); R = pm._PM['R']; ND = pm._PM['ND']; LAB = pm._PM['LAB']; TYP = pm._PM['TY']
PR = pm._PM.get('PRD') or {}
n1 = int(round(ND * 0.2)); g = np.random.RandomState(0)
early = [r for r in R if r[0] < n1 and 1 <= r[5] <= 8 and r[1] != r[2] and len(r[1]) < 40 and len(r[2]) < 40]
nc = collections.Counter(n for r in early for n in (r[1], r[2])); PAIR = {frozenset((r[1], r[2])) for r in early}
pool = [r for r in early if nc[r[1]] >= 2 and nc[r[2]] >= 2]; S = [pool[i] for i in g.choice(len(pool), min(NREC, len(pool)), replace=False)]
# Matching-the-Blanks training pairs: one name pair stated in two different documents (disjoint from nothing: label-free, any records)
byp = collections.defaultdict(list)
for r in early: byp[frozenset((r[1], r[2]))].append(r)
mp = [(L_[i], L_[j]) for L_ in byp.values() if len({r[0] for r in L_}) >= 2 for i, j in [(0, next(k for k in range(1, len(L_)) if L_[k][0] != L_[0][0]))]]
mp = [mp[i] for i in g.choice(len(mp), min(NPAIR, len(mp)), replace=False)]
TN = {'PER': 'a person', 'LOC': 'a place', 'ORG': 'an organisation', 'TIME': 'a date', 'NUM': 'a number', 'MISC': 'a thing'}
VIEWS = {'names': lambda r: f"{r[1]} {r[3]} {r[2]}.", 'masked': lambda r: f"someone {r[3]} something.",
         'type-masked': lambda r: f"{TN.get(TYP.get(r[1]), 'a thing')} {r[3]} {TN.get(TYP.get(r[2]), 'a thing')}."}
P(f"{len(early)} early records; evaluation sample {len(S)}; Matching-the-Blanks pairs (one name pair in two documents) {len(mp)}  ({time.time() - T0:.0f}s)")
lab = np.array([next(iter(PR[(r[1], r[2])])) if len(PR.get((r[1], r[2]), ())) == 1 else '' for r in S]); has = lab != ''
def purity(c):
    ct = collections.defaultdict(collections.Counter)
    for a, b in zip(c[has], lab[has]): ct[a][b] += 1
    return sum(v.most_common(1)[0][1] for v in ct.values()) / has.sum(), normalized_mutual_info_score(lab[has], c[has])
def mine(c):
    OUT = collections.defaultdict(lambda: collections.defaultdict(set)); HEAD = {}
    for r, t in zip(S, c):
        k = int(t); OUT[r[1]][(k, 0)].add(r[2]); OUT[r[2]][(k, 1)].add(r[1]); HEAD.setdefault((r[1], r[2]), set()).add((k, 0)); HEAD.setdefault((r[2], r[1]), set()).add((k, 1))
    HASR = collections.defaultdict(set)
    for (x, z), ks in HEAD.items(): HASR[x] |= ks
    BODY = collections.defaultdict(set)
    for y, d in OUT.items():
        nin = [(k, x) for k, xs in d.items() for x in xs]
        if len(nin) > 60: continue
        for k1, x in nin:
            for k2, z in nin:
                if z != x: BODY[((k1[0], 1 - k1[1]), k2)].add((x, z))
    rules = []
    for (r1, r2), bp in BODY.items():
        if len(bp) < 5: continue
        cnt = collections.Counter(k for x, z in bp for k in HEAD.get((x, z), ()))
        for k, s_ in cnt.items():
            if s_ >= 5:
                pca = s_ / max(1, sum(1 for x, z in bp if k in HASR[x]))
                if pca >= 0.3: rules.append((r1, r2, k, pca))
    TREL = collections.defaultdict(collections.Counter)
    for r, t in zip(S, c):
        for rr in PR.get((r[1], r[2]), ()): TREL[(int(t), 0)][rr] += 1
        for rr in PR.get((r[2], r[1]), ()): TREL[(int(t), 1)][rr] += 1
    props = {}
    for r1, r2, k, pca in rules:
        for x, z in BODY[(r1, r2)]:
            if frozenset((x, z)) not in PAIR and ((x, z) not in props or props[(x, z)][1] < pca): props[(x, z)] = (k, pca)
    typ = [int(TREL[k].most_common(1)[0][0] in PR.get((x, z), ())) for (x, z), (k, p) in props.items() if TREL[k]]
    return f"{len(rules)} rules, {len(props)} typed proposals, TYPED right {np.mean(typ) if typ else float('nan'):.3f}, untyped {np.mean([frozenset(p) in LAB for p in props]) if props else float('nan'):.3f}"
def report(name, Z):
    c = MiniBatchKMeans(KT, random_state=0, n_init=3, batch_size=4096).fit_predict(Z); pu, nmi = purity(c)
    P(f"   {name:44s} purity {pu:.3f}, NMI {nmi:.3f} | {mine(c)}")
P(f"TYPES ({KT}) on {has.sum()} records with one Wikidata relation (evaluation only); baseline: surface wordings")
wc = collections.Counter(r[3] for r in S); tw = {w: i for i, (w, _) in enumerate(wc.most_common(KT - 1))}; cw = np.array([tw.get(r[3], KT - 1) for r in S]); pu, nmi = purity(cw)
P(f"   {'surface wordings':44s} purity {pu:.3f}, NMI {nmi:.3f} | {mine(cw)}")
for vn, f in VIEWS.items():
    Xr = pooled([f(r) for r in S]); Xs = Xr - Xr.mean(0); report(f"pooled, {vn}", Xs)
    if vn == 'names': continue
    A_ = pooled([f(a) for a, b in mp]); B_ = pooled([f(b) for a, b in mp]); mu = np.concatenate([A_, B_]).mean(0); sd = np.concatenate([A_, B_]).std(0) + 1e-6
    A_, B_ = torch.tensor((A_ - mu) / sd, dtype=torch.float32), torch.tensor((B_ - mu) / sd, dtype=torch.float32)
    head = nn.Sequential(nn.Linear(A_.shape[1], 256), nn.GELU(), nn.Linear(256, 64)); opt = torch.optim.Adam(head.parameters(), lr=1e-3); tr = np.random.RandomState(3)
    for st in range(STEPS):
        ix = torch.as_tensor(tr.choice(len(A_), 256, replace=False)); z1 = F.normalize(head(A_[ix]), dim=-1); z2 = F.normalize(head(B_[ix]), dim=-1)
        lo = 0.5 * (F.cross_entropy(z1 @ z2.t() / 0.1, torch.arange(256)) + F.cross_entropy(z2 @ z1.t() / 0.1, torch.arange(256))); opt.zero_grad(); lo.backward(); opt.step()
    with torch.no_grad(): Zk = F.normalize(head(torch.tensor((Xr - mu) / sd, dtype=torch.float32)), dim=-1).numpy()
    report(f"MATCHING-THE-BLANKS key, {vn} (loss {float(lo):.2f})", Zk)
    P(f"  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
