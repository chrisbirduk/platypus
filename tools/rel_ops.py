"""RULES BY COMPOSING RELATION MAPS, AND TYPING THE WRITTEN INFERENCES (15 Nov 2026; tools/rel_ops.py) - offline, label-free; labels score.
Counting chains (tools/typed_rules.py, rel_types.py, rel_mtb.py) and counting shared name pairs (tools/ent_graph.py) find almost no rules
on the distant set: it is too sparse. After Hernandez et al. (ICLR 2024: a relation acts as a near-linear map between representations)
and the relational module's operator algebra (rule world: composed operator 1.00), a rule is found WITHOUT chains in the data: learn each
relation type as a linear map between NAME CODES and ask whether map r1 then map r2 lands on map r3.
  TYPES       as tools/rel_mtb.py's best: records read with the names replaced by their entity types, a Matching-the-Blanks key (two records
              of one name pair in different documents), 64 k-means types (on 30,000 records, documents 0 -> 2/10)
  NAME CODES  the store's name x name record graph (all records of documents 0 -> 2/10; PPMI, truncated SVD, 128 dims, unit length)
  MAPS        per type and direction, ridge regression code(A) -> code(C) over its records (40+); fit = held-out cosine
  RULES       for every t1, t2: W(t2) W(t1) against every W(t3): normalised distance; kept if below the 1st percentile of a NULL (maps fitted on
              records with shuffled types); examples described by their commonest wording (label-free) and Wikidata relation (evaluation)
  TYPING      the GPU path's 200 written inferences (tools/select_judge.py, code + AA, 0.74 right): each (A, C) gets the type whose map sends
              code(A) nearest code(C) (either direction), and the best composed-rule fit; scored: typed right (that type's commonest relation
              holds for A-C), and each fit as a JUDGE (AUC right vs wrong; the current judge 0.64)
Usage: TG_D=192 TG_CKPT=<model> python3 tools/rel_ops.py <train_distant.json.gz> <writes .json>"""
import os, sys, json, time, types, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
sys.path.insert(0, '.')
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2)
DW = int(os.environ.get('TG_D', 192)); CK = os.environ['TG_CKPT']; L = 128; PAD = 256; V = 257; KT = 64
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
        out.append(torch.cat([(h * m).sum(1) / m.sum(1) for h in (hs[1], hs[2])], -1).numpy())
    return np.concatenate(out)
import platypus.inference.premises as pm
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10)); R = pm._PM['R']; ND = pm._PM['ND']; LAB = pm._PM['LAB']; TYP = pm._PM['TY']
PR = pm._PM.get('PRD') or {}; n1 = int(round(ND * 0.2)); g = np.random.RandomState(0)
early = [r for r in R if r[0] < n1 and 1 <= r[5] <= 8 and r[1] != r[2] and len(r[1]) < 40 and len(r[2]) < 40]
# NAME CODES from the whole early record graph
NI = {}; ia, ib = [], []
for r in R:
    if r[0] < n1 and r[1] != r[2]: ia.append(NI.setdefault(r[1], len(NI))); ib.append(NI.setdefault(r[2], len(NI)))
NN = len(NI); G = csr_matrix((np.ones(len(ia), np.float32), (ia, ib)), shape=(NN, NN)); G = G + G.T; G.sum_duplicates(); G.data[:] = 1.0
deg = np.asarray(G.sum(1)).ravel(); Gc = G.tocoo(); tot = float(G.sum()); v = np.log(tot * Gc.data / (deg[Gc.row] * deg[Gc.col]) + 1e-12); mk = v > 0
Pm = csr_matrix((v[mk].astype(np.float32), (Gc.row[mk], Gc.col[mk])), shape=(NN, NN)); del G, Gc, ia, ib
CODE = TruncatedSVD(128, random_state=0).fit_transform(Pm).astype(np.float32); CODE /= (np.linalg.norm(CODE, axis=1, keepdims=True) + 1e-9); del Pm
P(f"name codes: {NN} names, 128 dims  ({time.time() - T0:.0f}s)")
# TYPES (type-masked + Matching-the-Blanks key)
nc = collections.Counter(n for r in early for n in (r[1], r[2])); pool = [r for r in early if nc[r[1]] >= 2 and nc[r[2]] >= 2]
S = [pool[i] for i in g.choice(len(pool), min(30000, len(pool)), replace=False)]
TN = {'PER': 'a person', 'LOC': 'a place', 'ORG': 'an organisation', 'TIME': 'a date', 'NUM': 'a number', 'MISC': 'a thing'}
tm = lambda r: f"{TN.get(TYP.get(r[1]), 'a thing')} {r[3]} {TN.get(TYP.get(r[2]), 'a thing')}."
byp = collections.defaultdict(list)
for r in early: byp[frozenset((r[1], r[2]))].append(r)
mp = []
for L_ in byp.values():
    o = [r for r in L_[1:] if r[0] != L_[0][0]]
    if o: mp.append((L_[0], o[0]))
mp = [mp[i] for i in g.choice(len(mp), min(20000, len(mp)), replace=False)]
Xr = pooled([tm(r) for r in S]); A_ = pooled([tm(a) for a, b in mp]); B_ = pooled([tm(b) for a, b in mp]); mu = np.concatenate([A_, B_]).mean(0); sd = np.concatenate([A_, B_]).std(0) + 1e-6
A_, B_ = torch.tensor((A_ - mu) / sd, dtype=torch.float32), torch.tensor((B_ - mu) / sd, dtype=torch.float32)
head = nn.Sequential(nn.Linear(A_.shape[1], 256), nn.GELU(), nn.Linear(256, 64)); opt = torch.optim.Adam(head.parameters(), lr=1e-3); tr = np.random.RandomState(3)
for st in range(3000):
    ix = torch.as_tensor(tr.choice(len(A_), 256, replace=False)); z1 = F.normalize(head(A_[ix]), dim=-1); z2 = F.normalize(head(B_[ix]), dim=-1)
    lo = 0.5 * (F.cross_entropy(z1 @ z2.t() / 0.1, torch.arange(256)) + F.cross_entropy(z2 @ z1.t() / 0.1, torch.arange(256))); opt.zero_grad(); lo.backward(); opt.step()
with torch.no_grad(): Zk = F.normalize(head(torch.tensor((Xr - mu) / sd, dtype=torch.float32)), dim=-1).numpy()
TYPE = MiniBatchKMeans(KT, random_state=0, n_init=3, batch_size=4096).fit_predict(Zk)
P(f"types: {KT} over {len(S)} records (type-masked Matching-the-Blanks key, loss {float(lo.detach()):.2f})  ({time.time() - T0:.0f}s)")
# MAPS per type and direction: code(head) -> code(tail)
def fit(pairs, lam=1.0):
    Xa = CODE[[NI[a] for a, b in pairs]]; Xb = CODE[[NI[b] for a, b in pairs]]; n = len(pairs); k = max(1, n // 5); idx = g.permutation(n); te, trn = idx[:k], idx[k:]
    W = np.linalg.solve(Xa[trn].T @ Xa[trn] + lam * np.eye(128), Xa[trn].T @ Xb[trn])
    pr = Xa[te] @ W; cs = (pr * Xb[te]).sum(1) / (np.linalg.norm(pr, axis=1) + 1e-9)
    W = np.linalg.solve(Xa.T @ Xa + lam * np.eye(128), Xa.T @ Xb); return W, float(cs.mean())
recs = collections.defaultdict(list)
for r, t in zip(S, TYPE):
    if r[1] in NI and r[2] in NI: recs[(int(t), 0)].append((r[1], r[2])); recs[(int(t), 1)].append((r[2], r[1]))
MAP = {}; FIT = {}
for k, pairs in recs.items():
    if len(pairs) >= 40: MAP[k], FIT[k] = fit(pairs)
sh = np.random.RandomState(5).permutation(len(S)); NUL = {}                    # NULL maps: the same record counts, types shuffled
recn = collections.defaultdict(list)
for r, t in zip(S, TYPE[sh]):
    if r[1] in NI and r[2] in NI: recn[(int(t), 0)].append((r[1], r[2])); recn[(int(t), 1)].append((r[2], r[1]))
for k, pairs in recn.items():
    if len(pairs) >= 40: NUL[k] = fit(pairs)[0]
P(f"maps: {len(MAP)} type-directions with 40+ records; held-out cosine code(A) W -> code(C): median {np.median(list(FIT.values())):.3f} (best {max(FIT.values()):.3f})  ({time.time() - T0:.0f}s)")
# RULES: W(t2) W(t1) vs W(t3)
nrm = lambda W: W / (np.linalg.norm(W) + 1e-9)
def rules(M):
    ks = list(M); N3 = np.stack([nrm(M[k]).ravel() for k in ks]); out = []
    for a in ks:
        for b in ks:
            if a[0] == b[0]: continue
            c = nrm(M[a] @ M[b]).ravel(); s = N3 @ c; j = int(np.argmax(s)); out.append((a, b, ks[j], float(s[j])))
    return out
RU = rules(MAP); RN = rules(NUL); thr = np.quantile([x[3] for x in RN], 0.99)
kept = sorted([x for x in RU if x[3] > thr], key=lambda x: -x[3])
wd = collections.defaultdict(collections.Counter); rel = collections.defaultdict(collections.Counter)
for r, t in zip(S, TYPE):
    wd[int(t)][r[3]] += 1
    for rr in PR.get((r[1], r[2]), ()): rel[(int(t), 0)][rr] += 1
    for rr in PR.get((r[2], r[1]), ()): rel[(int(t), 1)][rr] += 1
desc = lambda k: f"'{wd[k[0]].most_common(1)[0][0]}'{'^-1' if k[1] else ''} [{rel[k].most_common(1)[0][0] if rel[k] else '-'}]"
P(f"RULES W(t2) W(t1) ~ W(t3): {len(RU)} compositions; NULL 99th percentile similarity {thr:.3f}; kept {len(kept)} above it (top similarity {kept[0][3] if kept else float('nan'):.3f})")
for a, b, c, s in kept[:10]: P(f"   {desc(a)} then {desc(b)} => {desc(c)}  ({s:.3f})")
# TYPING the GPU path's writes
W8 = [(tuple(k), bool(t)) for k, t in json.load(open(sys.argv[2]))]; W8 = [(k, t) for k, t in W8 if k[0] in NI and k[1] in NI]
y = np.array([t for _, t in W8]); best = []; bfit = []; cfit = []
for (a, c), t in W8:
    ca, cc = CODE[NI[a]], CODE[NI[c]]; sc = {k: float((ca @ W) @ cc / (np.linalg.norm(ca @ W) + 1e-9)) for k, W in MAP.items()}
    k = max(sc, key=sc.get); best.append(k); bfit.append(sc[k])
    cfit.append(max([float((ca @ (MAP[x] @ MAP[z])) @ cc / (np.linalg.norm(ca @ MAP[x] @ MAP[z]) + 1e-9)) for x, z, _, _ in kept[:200]] or [0.0]))
typ = [int(bool(rel[k]) and rel[k].most_common(1)[0][0] in (PR.get((a, c), ()) if k[1] == 0 else PR.get((c, a), ()))) for ((a, c), _), k in zip(W8, best)]
anyrel = [int(bool(PR.get((a, c)) or PR.get((c, a)))) for (a, c), _ in W8]
bfit, cfit = np.array(bfit), np.array(cfit)
P(f"TYPING the {len(W8)} written inferences (right {y.mean():.3f}): the best map's type is the pair's relation {np.mean(typ):.3f} (among those with a relation "
  f"{np.mean([t for t, a in zip(typ, anyrel) if a]) if any(anyrel) else float('nan'):.3f}) | AS A JUDGE: best map fit AUC {_cg_auc(bfit[y], bfit[~y]):.3f}, "
  f"best composed-rule fit AUC {_cg_auc(cfit[y], cfit[~y]):.3f} (the current judge 0.64)")
P(f"DONE ({time.time() - T0:.0f}s)")
