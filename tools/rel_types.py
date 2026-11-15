"""RELATION TYPES FROM THE REPRESENTATION, THEN RULES OVER THEM (15 Nov 2026; tools/rel_types.py) - offline, stand-in trunk, label-free.
tools/rel_direction.py: a pooled representation of 'A <words> C' is ADDITIVE (subject + relation + object), the relation's subspace largely
apart from the names'; on real records, projecting the names out raised relation grouping 0.50 -> 0.54. tools/typed_rules.py: rules over
SURFACE wordings fail (a wording is not a relation). Here, label-free:
  1 REPRESENT  a sample of real records (documents 0 -> 2/10; both names in 2+ records) as the trunk's pooled state of 'A <words> C.'
  2 SUBSPACES  learned from real text: NAME subspace = top PCs of per-name means (names in 5+ records: the relation mix averages out);
               WORDING subspace = top PCs of per-wording means (wordings in 5+ records: the names average out)
  3 TYPES      k-means over the records with the name subspace projected out (and onto the wording subspace) -> relation types
  4 RULES      AMIE closed paths over the TYPED facts (type ids, either direction), proposals = body pairs never stated
Scored (EVALUATION ONLY): the types' purity / NMI against Wikidata relations (records with one), against surface-wording identity and
raw clustering; the rules' typed proposals - the head type's commonest Wikidata relation holds for (x, z) - and untyped (some relation).
Usage: TG_D=192 TG_CKPT=<model> python3 tools/rel_types.py <train_distant.json.gz>   env RT_N (30000 records), RT_K (64 types), RT_NS (24),
RT_WS (24), RT_SYNTH (0: name subspace from real per-name means; 1: from a synthetic factorial), RT_SUP (5), RT_CONF (0.3), RT_HUB (60)"""
import os, sys, json, gzip, time, types, collections
import numpy as np
import torch, torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
sys.path.insert(0, '.')
P = lambda *a: print(*a, flush=True); T0 = time.time(); torch.set_num_threads(2)
DW = int(os.environ.get('TG_D', 192)); CK = os.environ['TG_CKPT']; L = 128; PAD = 256; V = 257
NREC, KT, NS, WSD = int(os.environ.get('RT_N', 30000)), int(os.environ.get('RT_K', 64)), int(os.environ.get('RT_NS', 24)), int(os.environ.get('RT_WS', 24))
SUP, CONF, HUB = int(os.environ.get('RT_SUP', 5)), float(os.environ.get('RT_CONF', 0.3)), int(os.environ.get('RT_HUB', 60))
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
        H = lm.states(torch.cat([torch.full((len(ch), 1), PAD), X[:, :-1]], 1))[-1]
        m = (X != PAD).float()[:, :, None]; m = torch.cat([m[:, 1:], torch.zeros_like(m[:, :1])], 1) + torch.cat([torch.zeros_like(m[:, :1]), m[:, :-1]], 1).clamp(max=0)
        out.append(((H * (X != PAD).float()[:, :, None]).sum(1) / (X != PAD).float().sum(1, keepdim=True)).numpy())
    return np.concatenate(out)
import platypus.inference.premises as pm
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10)); R = pm._PM['R']; ND = pm._PM['ND']; LAB = pm._PM['LAB']
n1 = int(round(ND * 0.2)); g = np.random.RandomState(0)
early = [r for r in R if r[0] < n1 and 1 <= r[5] <= 8 and r[1] != r[2] and len(r[1]) < 40 and len(r[2]) < 40]
nc = collections.Counter(n for r in early for n in (r[1], r[2])); PAIR = {frozenset((r[1], r[2])) for r in early}
pool = [r for r in early if nc[r[1]] >= 2 and nc[r[2]] >= 2]
S = [pool[i] for i in g.choice(len(pool), min(NREC, len(pool)), replace=False)]
PR = collections.defaultdict(set)
for d in json.load(gzip.open(sys.argv[1], 'rt')):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']: PR[(cn[l['h']], cn[l['t']])].add(l['r'])
P(f"{len(early)} early records, {len(pool)} with both names in 2+ records; sample {len(S)}  ({time.time() - T0:.0f}s)")
X = pooled([f"{r[1]} {r[3]} {r[2]}." for r in S]); X = X - X.mean(0); P(f"represented  ({time.time() - T0:.0f}s)")
def pcs(groups, k):
    M = np.array([X[ix].mean(0) for ix in groups.values() if len(ix) >= 5]); M = M - M.mean(0)
    return np.linalg.svd(M, full_matrices=False)[2][:k].T, len(M)
byn = collections.defaultdict(list); byw = collections.defaultdict(list)
for i, r in enumerate(S): byn[r[1]].append(i); byn[r[2]].append(i); byw[r[3]].append(i)
Bn, nn_ = pcs(byn, NS); Bw, nw_ = pcs(byw, WSD)
if int(os.environ.get('RT_SYNTH', 0)):                                         # the NAME subspace from a factorial of synthetic sentences instead
    REL9 = ['was born in', 'died in', 'is a citizen of', 'worked for', 'played for', 'is a member of', 'is located in', 'was written by', 'was directed by', 'is part of', 'comes from', 'is based in']
    syl = ['ka', 'lo', 'mi', 'ner', 'tas', 'vol', 'ru', 'den', 'sa', 'por', 'el', 'ith', 'qua', 'bren', 'dor', 'fen']; g2 = np.random.RandomState(1)
    nmf = lambda: (''.join(g2.choice(syl, 2)) + ' ' + ''.join(g2.choice(syl, 3))).title(); SB = [nmf() for _ in range(16)]; OB = [nmf() for _ in range(16)]
    rows = [(a_, r_, o_) for a_ in range(16) for r_ in range(12) for o_ in range(16)]; Zs = pooled([f"{SB[a_]} {REL9[r_]} {OB[o_]}." for a_, r_, o_ in rows]); Zs -= Zs.mean(0); ix = np.array(rows)
    Ms = [np.array([Zs[ix[:, c] == v].mean(0) for v in range(16)]) for c in (0, 2)]
    Bn = np.linalg.qr(np.concatenate([np.linalg.svd(M - M.mean(0), full_matrices=False)[2][:NS // 2].T for M in Ms], 1))[0]; nn_ = 'synthetic 16 + 16'
ov = float(np.mean(np.linalg.svd(Bn.T @ Bw, compute_uv=False) ** 2))
P(f"NAME subspace from {nn_} names (top {NS}), WORDING subspace from {nw_} wordings (top {WSD}); their overlap (mean cos^2) {ov:.2f} (random ~{max(NS, WSD) / DW:.2f})")
Xo = X - (X @ Bn) @ Bn.T; Xw = Xo @ Bw
lab = np.array([next(iter(PR[(r[1], r[2])])) if len(PR.get((r[1], r[2]), ())) == 1 else '' for r in S]); has = lab != ''
wid = {w: i for i, w in enumerate(byw)}; widx = np.array([wid[r[3]] for r in S])
def purity(c):
    m = has; ct = collections.defaultdict(collections.Counter)
    for a, b in zip(c[m], lab[m]): ct[a][b] += 1
    return sum(v.most_common(1)[0][1] for v in ct.values()) / m.sum(), normalized_mutual_info_score(lab[m], c[m])
TY = {}
for nm, Z in (('raw', X), ('names out', Xo), ('names out, onto wording subspace', Xw)):
    c = MiniBatchKMeans(KT, random_state=0, n_init=3, batch_size=4096).fit_predict(Z); TY[nm] = c; pu, nmi = purity(c)
    P(f"   TYPES ({KT}) from {nm:34s}: purity {pu:.3f}, NMI {nmi:.3f} vs Wikidata on {has.sum()} records with one relation")
top = collections.Counter(widx).most_common(KT - 1); tw = {w: i for i, (w, _) in enumerate(top)}; cw = np.array([tw.get(w, KT - 1) for w in widx]); pu, nmi = purity(cw)
P(f"   TYPES ({KT}) = the commonest SURFACE WORDINGS (rest one type): purity {pu:.3f}, NMI {nmi:.3f}")
def mine(c, name):
    OUT = collections.defaultdict(lambda: collections.defaultdict(set)); HEAD = {}
    for r, t in zip(S, c):
        k = int(t); OUT[r[1]][(k, 0)].add(r[2]); OUT[r[2]][(k, 1)].add(r[1]); HEAD.setdefault((r[1], r[2]), set()).add((k, 0)); HEAD.setdefault((r[2], r[1]), set()).add((k, 1))
    HASR = collections.defaultdict(set)
    for (x, z), ks in HEAD.items(): HASR[x] |= ks
    BODY = collections.defaultdict(set)
    for y, d in OUT.items():
        nin = [(k, x) for k, xs in d.items() for x in xs]
        if len(nin) > HUB: continue
        for k1, x in nin:
            r1 = (k1[0], 1 - k1[1])
            for k2, z in nin:
                if z != x: BODY[(r1, k2)].add((x, z))
    rules = []
    for (r1, r2), bp in BODY.items():
        if len(bp) < SUP: continue
        cnt = collections.Counter(k for x, z in bp for k in HEAD.get((x, z), ()))
        for k, s in cnt.items():
            if s < SUP: continue
            pca = s / max(1, sum(1 for x, z in bp if k in HASR[x]))
            if pca >= CONF: rules.append((r1, r2, k, s, pca))
    TREL = collections.defaultdict(collections.Counter)                       # evaluation only: a type's commonest relation, by direction
    for r, t in zip(S, c):
        for rr in PR.get((r[1], r[2]), ()): TREL[(int(t), 0)][rr] += 1
        for rr in PR.get((r[2], r[1]), ()): TREL[(int(t), 1)][rr] += 1
    props = {}
    for r1, r2, k, s, pca in rules:
        for x, z in BODY[(r1, r2)]:
            if frozenset((x, z)) in PAIR: continue
            if (x, z) not in props or props[(x, z)][1] < pca: props[(x, z)] = (k, pca)
    if not props: P(f"   RULES over {name}: {len(rules)} kept, no proposals"); return
    unt = np.mean([frozenset(p) in LAB for p in props]); typ = []
    for (x, z), (k, pca) in props.items():
        if TREL[k]: typ.append(int(TREL[k].most_common(1)[0][0] in PR.get((x, z), ())))
    hi = [(p, v) for p, v in props.items() if v[1] >= 0.6]
    P(f"   RULES over {name:34s}: {len(rules)} kept (support >= {SUP}, PCA >= {CONF}); {len(props)} typed proposals: UNTYPED right {unt:.3f}, TYPED right "
      f"{np.mean(typ) if typ else float('nan'):.3f} | PCA >= 0.6: {len(hi)}, untyped {np.mean([frozenset(p) in LAB for p, v in hi]) if hi else float('nan'):.3f}")
for nm in TY: mine(TY[nm], f"types from {nm}")
mine(cw, 'surface-wording types')
P(f"DONE ({time.time() - T0:.0f}s)")
