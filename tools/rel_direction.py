"""DOES THE RELATION WORD GIVE A DIRECTION OF ITS OWN? (15 Nov 2026; tools/rel_direction.py) - offline, stand-in trunk.
The author's hypothesis: in a plain (JEPA-style, pooled) representation of 'A r C', the RELATION phrase adds a direction - a subspace
roughly orthogonal to what the NAMES (subject, object) and NUMBERS add - so relations could be read off (typed) by projection, label-free.
PART A, factorial: S subjects x R relation phrases x O objects (x Y years: 'A r C in YEAR.'), the trunk's states pooled over the sentence
(mean) and at the end of the relation phrase, middle and last block. Per factor: share of the variance (main effects, ANOVA), the
ADDITIVE fit (main effects only: R^2), and the PRINCIPAL ANGLES between the factors' effect subspaces (mean cos^2 of the top-k bases:
0 = orthogonal, 1 = the same; random k-dim subspaces of d dims give about k/d).
PART B, real text: Re-DocRED-style distant records 'A <words> C' (one Wikidata relation, evaluation only), pooled states (i) raw, (ii)
with the NAME subspaces of part A projected out, (iii) projected ONTO the relation subspace of part A: how well each groups records by
their relation - leave-one-out 5-nearest-neighbour accuracy and k-means NMI (labels only score).
Usage: TG_D=192 TG_CKPT=<model> python3 tools/rel_direction.py <train_distant.json.gz>   env RD_K (8: subspace dims)"""
import os, sys, re, json, gzip, math, time, collections, types
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
sys.path.insert(0, '.')
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2)
DW = int(os.environ.get('TG_D', 192)); CK = os.environ['TG_CKPT']; L = 128; PAD = 256; V = 257; KD = int(os.environ.get('RD_K', 8))
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
def reps(texts, ends=None):
    """per text: {(layer, 'mean' | 'end')}: mean over its bytes, and the state at byte index ends[i] (the relation phrase's last byte)"""
    out = collections.defaultdict(list)
    for i0 in range(0, len(texts), 64):
        ch = [t.encode('latin-1', 'replace')[:L] for t in texts[i0:i0 + 64]]
        X = torch.full((len(ch), L), PAD, dtype=torch.long)
        for k, b in enumerate(ch): X[k, :len(b)] = torch.tensor(list(b))
        hs = lm.states(torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1))   # state at output t has read bytes < t; index t = len(b) reads all
        for li, nm in ((1, 'mid'), (2, 'last')):
            H = hs[li]
            for k, b in enumerate(ch):
                n = len(b); out[(nm, 'mean')].append(H[k, 1:n + 1].mean(0).numpy())
                if ends is not None: out[(nm, 'end')].append(H[k, min(ends[i0 + k] + 1, n)].numpy())
    return {k: np.array(v) for k, v in out.items()}
def basis(E, k):
    E = E - E.mean(0); U, S, Vt = np.linalg.svd(E, full_matrices=False); return Vt[:k].T
def overlap(B1, B2): return float(np.mean(np.linalg.svd(B1.T @ B2, compute_uv=False) ** 2))
REL = ['was born in', 'died in', 'is a citizen of', 'worked for', 'played for', 'is a member of', 'is located in', 'was written by',
       'was directed by', 'is part of', 'comes from', 'is based in']
syl = ['ka', 'lo', 'mi', 'ner', 'tas', 'vol', 'ru', 'den', 'sa', 'por', 'el', 'ith', 'qua', 'bren', 'dor', 'fen']; g = np.random.RandomState(0)
nm = lambda: (''.join(g.choice(syl, 2)) + ' ' + ''.join(g.choice(syl, 3))).title()
SUB = [nm() for _ in range(16)]; OBJ = [nm() for _ in range(16)]; YR = ['1874', '1912', '1950', '1987', '2003']
# PART A
for withyear in (False, True):
    rows = [(s, r, o, y) for s in range(len(SUB)) for r in range(len(REL)) for o in range(len(OBJ)) for y in (range(len(YR)) if withyear else [0])]
    if withyear: rows = [rows[i] for i in g.choice(len(rows), 4000, replace=False)]
    tx = [f"{SUB[s]} {REL[r]} {OBJ[o]}" + (f" in {YR[y]}." if withyear else ".") for s, r, o, y in rows]
    ends = [len(f"{SUB[s]} {REL[r]}") - 1 for s, r, o, y in rows]; Z = reps(tx, ends); idx = np.array(rows)
    fac = [('subject', 0), ('RELATION', 1), ('object', 2)] + ([('year', 3)] if withyear else [])
    P(f"PART A{' (with years)' if withyear else ''}: {len(tx)} sentences, {len(SUB)} subjects x {len(REL)} relations x {len(OBJ)} objects" + (f" x {len(YR)} years" if withyear else ''))
    for key in sorted(Z):
        X = Z[key] - Z[key].mean(0); tot = (X ** 2).sum(); eff = {}; fit = np.zeros_like(X); parts = []
        for name, c in fac:
            M = np.array([X[idx[:, c] == v].mean(0) for v in range(idx[:, c].max() + 1)]); eff[name] = M; fit += M[idx[:, c]]
            parts.append(f"{name} {((M[idx[:, c]]) ** 2).sum() / tot:.2f}")
        r2 = 1 - ((X - fit) ** 2).sum() / tot; B = {n: basis(eff[n], min(KD, eff[n].shape[0] - 1)) for n, c in fac}
        ang = [f"{a}-{b} {overlap(B[a], B[b]):.2f}" for a, b in (('RELATION', 'subject'), ('RELATION', 'object'), ('subject', 'object')) + ((('RELATION', 'year'),) if withyear else ())]
        P(f"   {key[0]:4s} {key[1]:4s}: variance share {', '.join(parts)}; ADDITIVE R^2 {r2:.2f} | subspace overlap (mean cos^2, top {KD}; random ~{KD / DW:.2f}): {', '.join(ang)}")
    if not withyear: BASES = {k: {n: basis(np.array([(Z[k] - Z[k].mean(0))[idx[:, c] == v].mean(0) for v in range(idx[:, c].max() + 1)]), KD) for n, c in fac} for k in Z if k[1] == 'mean'}
P(f"  ({time.time() - T0:.0f}s)")
# PART B: real records
import platypus.inference.premises as pm
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10)); R = pm._PM['R']
PR = collections.defaultdict(set)
for d in json.load(gzip.open(sys.argv[1], 'rt')):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']: PR[(cn[l['h']], cn[l['t']])].add(l['r'])
rec = [(r[1], r[3], r[2], next(iter(PR[(r[1], r[2])]))) for r in R if 1 <= r[5] <= 8 and len(PR.get((r[1], r[2]), ())) == 1 and len(r[1]) < 40 and len(r[2]) < 40]
cnt = collections.Counter(t[3] for t in rec); top = {k for k, c in cnt.most_common(12)}
byr = collections.defaultdict(list)
for t in rec:
    if t[3] in top: byr[t[3]].append(t)
rec = [t for k in sorted(byr) for t in [byr[k][i] for i in g.choice(len(byr[k]), min(150, len(byr[k])), replace=False)]]
y = np.array([t[3] for t in rec]); Zr = reps([f"{a} {w} {c}." for a, w, c, _ in rec])
P(f"PART B: {len(rec)} real records 'A <words> C' with one Wikidata relation, the 12 commonest relations ({', '.join(sorted(top))})")
def knn(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); S = Xn @ Xn.T; np.fill_diagonal(S, -9); nn_ = np.argsort(-S, 1)[:, :5]
    return float(np.mean([collections.Counter(y[j] for j in row).most_common(1)[0][0] == y[i] for i, row in enumerate(nn_)]))
nmi = lambda X: normalized_mutual_info_score(y, KMeans(12, random_state=0, n_init=4).fit_predict(X))
for key in sorted(Zr):
    X = Zr[key] - Zr[key].mean(0); Bn = np.linalg.qr(np.concatenate([BASES[key]['subject'], BASES[key]['object']], 1))[0]; Br = BASES[key]['RELATION']
    Xo = X - (X @ Bn) @ Bn.T; Xr = X @ Br
    P(f"   {key[0]:4s} {key[1]:4s}: 5-NN relation accuracy raw {knn(X):.3f}, names projected OUT {knn(Xo):.3f}, ONTO the relation subspace ({KD} dims) {knn(Xr):.3f} "
      f"(chance {max(collections.Counter(y).values()) / len(y):.3f}) | k-means NMI raw {nmi(X):.3f}, names out {nmi(Xo):.3f}, relation subspace {nmi(Xr):.3f}")
P(f"DONE ({time.time() - T0:.0f}s)")
