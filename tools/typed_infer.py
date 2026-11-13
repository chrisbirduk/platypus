"""UNIVERSAL SCHEMA + TYPED, COUPLED INFERENCE (13 Nov 2026; tools/typed_infer.py) - offline, no effect on the package.
(1) UNIVERSAL SCHEMA (Riedel et al. 2013): the store's records form a matrix of entity pairs x wordings; wordings that occur with the same
    pairs are the same relation. PPMI over wordings x ordered pairs, SVD -> a vector per wording, k-means -> LATENT RELATIONS (no labels).
    RELATION IDENTITY on held-out gold-related records (as tools/record_mdl.py): cosine of the wordings' vectors, alone and with the compressed
    name-type pair; against the wording class (0.56-0.61) and the gold type pair (0.80).
(2) TYPED, COUPLED INFERENCE (NELL-style coupling, Carlson et al. 2010): A-C through B, each premise typed by its latent relation (and the
    bridge's side); a COMPOSITION TABLE (premise relations -> the relation A and C are later read in, in a relational wording) learned on
    TRAIN pairs (label-free); an inference is accepted when its composition is confident AND the predicted relation's argument types fit A and
    C (type coupling: the name-type pair has been seen with that relation). Gold rate on TEST pairs; TYPED accuracy (the predicted latent
    relation's majority gold relation is a gold relation of A-C; the mapping uses gold on training records, evaluation only).
(3) TYPED HOP 2: accepted typed inferences + stored records -> A-D, typed and composability-gated again; against untyped hop 2 (0.14).
Usage: RM_EXTRA=<distant documents> python3 tools/typed_infer.py   env TI_DIM (64), TI_R (latent relations, 64)"""
import os
import sys
import math
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('RM_K', '16'); os.environ.setdefault('RM_M', '128'); os.environ.setdefault('RM_R', '1'); os.environ.setdefault('RM_INIT', 'split')
_rm = open(os.path.join(HERE, 'record_mdl.py')).read()
exec(compile(_rm[:_rm.index('KBEST = min(FITS')], 'record_mdl', 'exec'))
gn, gw = FITS[(int(os.environ['RM_K']), int(os.environ['RM_M']))]
import zlib
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
DIM = int(os.environ.get('TI_DIM', 64)); RR = int(os.environ.get('TI_R', 64))
# ---- (1) universal schema: wordings x ordered pairs
PI = {}; pid = np.array([PI.setdefault((int(x), int(z)), len(PI)) for x, z in zip(Af, Bf)])
M_ = csr_matrix((np.ones(len(pid)), (Wf, pid)), shape=(NW, len(PI))); M_.data[:] = 1.0; M_.sum_duplicates(); M_.data[:] = 1.0
rs_ = np.asarray(M_.sum(1)).ravel(); cs_ = np.asarray(M_.sum(0)).ravel(); tot = M_.sum()
Mc = M_.tocoo(); pm = np.log(tot / (rs_[Mc.row] * cs_[Mc.col]) + 1e-12); keep = pm > 0
X = csr_matrix((pm[keep], (Mc.row[keep], Mc.col[keep])), shape=M_.shape)
E = normalize(TruncatedSVD(DIM, random_state=0).fit_transform(X))
rel = MiniBatchKMeans(RR, random_state=0, batch_size=4096, n_init=3).fit_predict(E)
P(f"UNIVERSAL SCHEMA: {NW} wordings x {len(PI)} ordered pairs ({M_.nnz} cells; pairs seen 2+ times {int((cs_ >= 2).sum())}), PPMI + SVD {DIM}, {RR} latent relations  ({time.time() - T0:.0f}s)")
# gold relations per pair (evaluation only)
GR = collections.defaultdict(set); ANN = set(); dfreq = collections.Counter()
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for l in d['labels']: GR[frozenset((cn[l['h']], cn[l['t']]))].add(l['r'])
    ns = set(cn); ANN |= {frozenset((x, z)) for x, z in ((p, q) for p in ns for q in ns) if x != z}
    for n in ns: dfreq[n] += 1
GOLDP = set(GR)
RMD = {}
for di, d in enumerate(docs):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]; R_ = collections.defaultdict(set)
    for l in d['labels']: R_[(cn[l['h']], cn[l['t']])].add(l['r']); R_[(cn[l['t']], cn[l['h']])].add(l['r'] + '^-1')
    RMD[di] = R_
HR = [i for i in TE if y[i]]; RS = [RMD[C[i]['di']][(C[i]['cA'], C[i]['cB'])] for i in HR]
rq = np.random.RandomState(1); qs = rq.choice(len(HR), 600, replace=False); PAIRS = []
for q in qs: PAIRS += [(int(q), int(c)) for c in rq.choice(len(HR), 40, replace=False) if C[HR[int(c)]]['di'] != C[HR[q]]['di']][:30]
same = np.array([bool(RS[i] & RS[j]) for i, j in PAIRS])
def rep(nm_, sim):
    top, ch = [], []
    for n0 in range(0, len(PAIRS), 30):
        blk = list(range(n0, min(n0 + 30, len(PAIRS)))); bb = max(blk, key=lambda n: sim[n] + 1e-9 * rq.rand()); top.append(same[bb]); ch.append(np.mean(same[blk]))
    P(f"   {nm_:56s} top-1 same relation {np.mean(top):.3f} (chance {np.mean(ch):.3f}) | AUC {auc(sim[same], sim[~same]):.3f}")
eq = lambda f: np.array([float(f(i) == f(j)) for i, j in PAIRS])
P(f"RELATION IDENTITY (held-out gold-related records; {len(PAIRS)} pairs):")
cosw = np.array([float(E[w[HR[i]]] @ E[w[HR[j]]]) for i, j in PAIRS]); tpair = eq(lambda i: (gn[a[HR[i]]], gn[b[HR[i]]]))
rep('wording class (record compression, M 128)', eq(lambda i: gw[w[HR[i]]])); rep('UNIVERSAL SCHEMA wording vector (cosine)', cosw)
rep('UNIVERSAL SCHEMA latent relation (equal)', eq(lambda i: rel[w[HR[i]]])); rep('universal schema cosine + name-type pair', cosw + tpair)
rep('latent relation + name-type pair', eq(lambda i: rel[w[HR[i]]]) + tpair); rep('GOLD type pair (reference)', eq(lambda i: (GT[a[HR[i]]], GT[b[HR[i]]])))
# ---- (2) typed, coupled inference
thr_rel = np.quantile(s_all, 0.75); rr_ = lambda i: int(rel[w[i]])
SREL = collections.defaultdict(list)                                                # pair -> (relation, document) of later relational readings
for i, c in enumerate(C):
    if s_all[i] >= thr_rel: SREL[frozenset((c['cA'], c['cB']))].append((rr_(i), c['di']))
ARG = collections.defaultdict(collections.Counter)                                 # type coupling: name-type pairs seen with each latent relation
for i in np.where(s_all >= np.quantile(s_all, 0.5))[0]: ARG[rr_(i)][tuple(sorted((int(gn[a[i]]), int(gn[b[i]]))))] += 1
side = lambda i, B: 0 if C[i]['cB'] == B else 1
nb = collections.defaultdict(list)
for i, c in enumerate(C): nb[c['cA']].append((c['cB'], c['di'], i)); nb[c['cB']].append((c['cA'], c['di'], i))
PR = collections.defaultdict(lambda: dict(k=set(), docs=set(), br=set()))
for i, c in enumerate(C):
    for B, Cn in ((c['cA'], c['cB']), (c['cB'], c['cA'])):
        if dfreq[B] > 200: continue
        for A, dA, j in nb[B]:
            if dA == c['di'] or A == Cn: continue
            k1, k2 = (rr_(j), side(j, B)), (rr_(i), side(i, B)); kd = (k1, k2) if k1 <= k2 else (k2, k1)
            p = PR[frozenset((A, Cn))]; p['k'].add(kd); p['docs'] |= {dA, c['di']}; p['br'].add(B)
keys = list(PR); tr = np.array([zlib.crc32(repr(sorted(k)).encode()) % 2 == 0 for k in keys])
cf = [[r for r, d_ in SREL.get(k, []) if d_ not in PR[k]['docs']] for k in keys]
CT = collections.defaultdict(collections.Counter); KN = collections.Counter()
for k, t, c_ in zip(keys, tr, cf):
    if not t: continue
    for kd in PR[k]['k']:
        KN[kd] += 1
        for r in set(c_): CT[kd][r] += 1
def wilson(s, n, z=1.64):
    if n == 0: return 0.0
    p = s / n; return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
def typed(k):
    """best (confidence, predicted relation) over the proposal's premise kinds"""
    best = (0.0, -1)
    for kd in PR[k]['k']:
        if CT[kd]:
            r, n_ = CT[kd].most_common(1)[0]; best = max(best, (wilson(n_, KN[kd]), r))
    return best
TY = [typed(k) for k in keys]; comp = np.array([t[0] for t in TY]); prel = np.array([t[1] for t in TY])
def fits(k, r):
    A, Cn = tuple(k); tp = tuple(sorted((int(gn[NI[A]]), int(gn[NI[Cn]])))); n_ = sum(ARG[r].values())
    return r >= 0 and n_ > 0 and ARG[r][tp] / n_ >= 0.01
cpl = np.array([fits(k, r) for k, r in zip(keys, prel)])
GMAP = {}                                                                           # evaluation only: latent relation -> its majority gold relation
for i in TR:
    if y[i]:
        for g_ in RMD[C[i]['di']].get((C[i]['cA'], C[i]['cB']), ()): GMAP.setdefault(rr_(i), collections.Counter())[g_.replace('^-1', '')] += 1
GMAP = {r: c_.most_common(1)[0][0] for r, c_ in GMAP.items()}
TEv = np.where(~tr & np.array([k in ANN for k in keys]))[0]; yte = np.array([keys[i] in GOLDP for i in TEv])
tacc = np.array([GMAP.get(prel[i], '') in GR.get(keys[i], set()) for i in TEv])
P(f"TYPED INFERENCE: {len(keys)} proposals, {sum(1 for k in KN)} premise kinds, composition learned on train pairs ({sum(1 for t, c_ in zip(tr, cf) if t and c_)} confirmed "
  f"in a relational wording) | TEST verifiable {len(TEv)}, gold {yte.mean():.3f}  ({time.time() - T0:.0f}s)")
def pol(tag, m):
    ix = np.where(m[TEv])[0]
    P(f"   {tag:58s} {len(ix):6d} inferences, gold {yte[ix].mean() if len(ix) else float('nan'):.3f} | TYPED right (relation too) {tacc[ix].mean() if len(ix) else float('nan'):.3f}")
q90 = np.quantile(comp[TEv], 0.90); q75 = np.quantile(comp[TEv], 0.75)
pol('all proposals', np.ones(len(keys), bool)); pol('typed composability top 25%', comp >= q75); pol('typed composability top 10%', comp >= q90)
pol('type coupling only', cpl); pol('typed composability top 25% AND type coupling', (comp >= q75) & cpl); pol('typed composability top 10% AND type coupling', (comp >= q90) & cpl)
# ---- (3) typed hop 2: accepted typed inferences + stored records, composability-gated again
acc = [i for i in np.where((comp >= q90) & cpl)[0]]; smp = [acc[i] for i in rq.permutation(len(acc))[:3000]]; H2 = {}
for i in smp:
    X_, Y_ = tuple(keys[i]); r3 = int(prel[i])
    for a_, b_ in ((X_, Y_), (Y_, X_)):
        if dfreq[b_] > 200: continue
        for D_, dD, j in nb[b_]:
            if D_ == a_: continue
            k2 = frozenset((a_, D_))
            if k2 not in ANN or k2 in PR: continue
            k1, kk2 = (r3, 0), (rr_(j), side(j, b_)); kd = (k1, kk2) if k1 <= kk2 else (kk2, k1)
            c2 = (wilson(CT[kd].most_common(1)[0][1], KN[kd]) if CT[kd] else 0.0)
            H2[k2] = max(H2.get(k2, 0.0), c2)
k2s = list(H2); g2 = np.array([k in GOLDP for k in k2s]); c2s = np.array([H2[k] for k in k2s])
if len(k2s):
    m2 = c2s >= (np.quantile(c2s[c2s > 0], 0.75) if (c2s > 0).any() else 1.0)
    P(f"TYPED HOP 2 from {len(smp)} accepted typed inferences (hop-1 gold {np.mean([keys[i] in GOLDP for i in smp if keys[i] in ANN]):.3f}): {len(k2s)} verifiable, gold {g2.mean():.3f} | "
      f"composability-gated (top 25% of nonzero) {int(m2.sum())}, gold {g2[m2].mean() if m2.any() else float('nan'):.3f} | untyped hop 2 was 0.14")
P(f"DONE ({time.time() - T0:.0f}s)")
