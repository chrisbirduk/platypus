"""CODES THAT CARRY THE RELATION (14 Nov 2026; tools/rel_codes.py) - offline, no effect on the package.
tools/fact_compress.py: a low-rank code per name infers THAT two names never stated together are related (3-4x base where a chain exists).
Here the code also says WHICH relation. Every store record (A, wording, B) gets a label-free LATENT RELATION (universal schema: PPMI of
wordings x name pairs, SVD, k-means, as tools/typed_infer.py), and a DistMult-style model learns a code per name and a diagonal operator per
latent relation from the store's records (negatives: a random B), no labels: score(A, r, B) = sum_k a_k r_k b_k.
Held out as in tools/fact_compress.py (FC_NT documents; every record pairing two names of one of them removed; nothing in the store states a
candidate). For each candidate pair:
  RELATED    max over relations of the score, against the fact_compress code (cosine of PPMI-SVD codes) - AUC and top 5 / 10 / 25 %
  WHICH      for gold-related pairs, the predicted latent relation (argmax) mapped to its majority gold relation on the store's own
             gold-related records (evaluation only) - right if it is one of the pair's gold relations; against the commonest gold relation
             and the commonest gold relation of the pair's (gold) entity-type pair (an oracle-typed reference)
Usage: python3 tools/rel_codes.py   env FC_NT (400), FC_SEED (0), RC_D (128), RC_EPOCHS (8), RC_R (64 latent relations)"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_rm = open(os.path.join(HERE, 'record_mdl.py')).read()
exec(compile(_rm[:_rm.index('FITS = {}')], 'record_mdl', 'exec'))
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
torch.set_num_threads(1)
NT = int(os.environ.get('FC_NT', 400)); FS = int(os.environ.get('FC_SEED', 0)); DD = int(os.environ.get('RC_D', 128)); EP = int(os.environ.get('RC_EPOCHS', 8)); RR = int(os.environ.get('RC_R', 64))
rh = np.random.RandomState(100 + FS); HOLD = set(rh.permutation(len(docs))[:NT].tolist())
cnz = lambda d: [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
CAND = {}; GREL = collections.defaultdict(set); GTY = {}
for di, d in enumerate(docs):
    cn = cnz(d); tp = [vs[0].get('type', 'X') for vs in d['vertexSet']]
    for n, t in zip(cn, tp): GTY.setdefault(n, t)
    if di in HOLD:
        ns = sorted(set(cn))
        for i_ in range(len(ns)):
            for j_ in range(i_ + 1, len(ns)): CAND[frozenset((ns[i_], ns[j_]))] = di
    for l in d['labels']:
        if cn[l['h']] != cn[l['t']]: GREL[frozenset((cn[l['h']], cn[l['t']]))].add(l['r'])
dvec = np.array([c['di'] for c in C] + [-1] * (len(Af) - N))
keep = np.array([(dvec[i] not in HOLD) and (frozenset((NAMES[Af[i]], NAMES[Bf[i]])) not in CAND) and Af[i] != Bf[i] for i in range(len(Af))])
# ---- latent relations (universal schema) over all kept records
PI = {}; pid = np.array([PI.setdefault((int(x), int(z)), len(PI)) for x, z in zip(Af[keep], Bf[keep])])
M_ = csr_matrix((np.ones(len(pid)), (Wf[keep], pid)), shape=(NW, len(PI))); M_.sum_duplicates(); M_.data[:] = 1.0
rs_ = np.asarray(M_.sum(1)).ravel(); cs_ = np.asarray(M_.sum(0)).ravel(); tot = M_.sum(); Mc = M_.tocoo()
pm = np.log(tot / (rs_[Mc.row] * cs_[Mc.col]) + 1e-12); kp = pm > 0
E = normalize(TruncatedSVD(64, random_state=0).fit_transform(csr_matrix((pm[kp], (Mc.row[kp], Mc.col[kp])), shape=M_.shape)))
rel = MiniBatchKMeans(RR, random_state=0, batch_size=4096, n_init=3).fit_predict(E)
KA, KB, KR = Af[keep], Bf[keep], rel[Wf[keep]]
P(f"held-out documents {NT}: candidates {len(CAND)} | store records kept {int(keep.sum())}, latent relations {RR}  ({time.time() - T0:.0f}s)")
# ---- gold mapping latent relation -> commonest gold relation, on the store's own (kept, Re-DocRED) gold-related records (evaluation only)
GMAP = collections.defaultdict(collections.Counter)
for i in np.where(keep[:N])[0]:
    for g in GREL.get(frozenset((C[i]['cA'], C[i]['cB'])), ()): GMAP[int(rel[Wf[i]])][g] += 1
GM = {r: c.most_common(1)[0][0] for r, c in GMAP.items()}
# ---- DistMult on (A, r, B), both directions, negatives by a random B
torch.manual_seed(FS); emb = torch.nn.Embedding(NN, DD); rop = torch.nn.Embedding(RR, DD)
torch.nn.init.normal_(emb.weight, std=0.1); torch.nn.init.normal_(rop.weight, mean=1.0, std=0.1)
op = torch.optim.Adam(list(emb.parameters()) + list(rop.parameters()), lr=0.01)
TA = torch.tensor(np.r_[KA, KB]); TB = torch.tensor(np.r_[KB, KA]); TR = torch.tensor(np.r_[KR, KR]); nT = len(TA); bs = 4096
for ep in range(EP):
    perm = torch.randperm(nT); ls = []
    for s0 in range(0, nT, bs):
        ix = perm[s0:s0 + bs]; a = emb(TA[ix]); r = rop(TR[ix]); b = emb(TB[ix]); bn = emb(torch.randint(0, NN, (len(ix), 4)))
        pos = (a * r * b).sum(-1); neg = ((a * r)[:, None, :] * bn).sum(-1)
        l = F.softplus(-pos).mean() + F.softplus(neg).mean() + 1e-4 * (a.pow(2).sum(-1).mean() + b.pow(2).sum(-1).mean())
        op.zero_grad(); l.backward(); op.step(); ls.append(float(l))
    P(f"   epoch {ep + 1}: loss {np.mean(ls):.4f}  ({time.time() - T0:.0f}s)")
G1 = csr_matrix((np.ones(int(keep.sum())), (KA, KB)), shape=(NN, NN)); G1 = G1 + G1.T; G1.data[:] = 1.0; deg = np.asarray(G1.sum(1)).ravel()
ok = [k for k in CAND if all(NI.get(n) is not None and deg[NI[n]] > 0 for n in k)]
XA = np.array([NI[sorted(k)[0]] for k in ok]); XB = np.array([NI[sorted(k)[1]] for k in ok]); yy = np.array([k in GREL for k in ok])
with torch.no_grad():
    ea = emb.weight[torch.tensor(XA)]; eb = emb.weight[torch.tensor(XB)]; S = torch.einsum('nd,rd,nd->nr', ea, rop.weight, eb).numpy()
smax = S.max(1); rhat = S.argmax(1)
Gc = G1.tocoo(); v = np.log(G1.sum() * Gc.data / (deg[Gc.row] * deg[Gc.col]) + 1e-12); m = v > 0
U = normalize(TruncatedSVD(512, random_state=0).fit_transform(csr_matrix((v[m].astype(np.float32), (Gc.row[m], Gc.col[m])), shape=G1.shape)).astype(np.float32))
code = np.einsum('ij,ij->i', U[XA], U[XB]); cn_cnt = np.asarray(G1[XA].multiply(G1[XB]).sum(1)).ravel()
def rep(mask, title):
    yv = yy[mask]; P(f"{title}: {int(mask.sum())} candidates, gold {yv.mean():.3f}")
    for k, sv in (('CODE 512 (fact_compress)', code[mask]), (f'RELATION CODES {DD} (max over relations)', smax[mask])):
        o = np.argsort(-sv); P(f"   {k:44s} AUC {auc(sv[yv], sv[~yv]):.3f} | gold in top 5/10/25%: " + "/".join(f"{yv[o[:max(1, len(o) * q // 100)]].mean():.3f}" for q in (5, 10, 25)))
rep(np.ones(len(ok), bool), "RELATED, ALL CANDIDATES"); rep(cn_cnt > 0, "RELATED, WITH A BRIDGE"); rep(cn_cnt == 0, "RELATED, NO BRIDGE")
gi = np.where(yy)[0]; rels = [GREL[ok[i]] for i in gi]
allg = collections.Counter(g for i in np.where(keep[:N])[0] for g in GREL.get(frozenset((C[i]['cA'], C[i]['cB'])), ()))
top1 = allg.most_common(1)[0][0]
tpc = collections.defaultdict(collections.Counter)
for i in np.where(keep[:N])[0]:
    for g in GREL.get(frozenset((C[i]['cA'], C[i]['cB'])), ()): tpc[tuple(sorted((GTY.get(C[i]['cA'], 'X'), GTY.get(C[i]['cB'], 'X'))))][g] += 1
acc = np.mean([GM.get(int(rhat[i]), '') in rs__ for i, rs__ in zip(gi, rels)])
base1 = np.mean([top1 in rs__ for rs__ in rels])
tp_ref = np.mean([(tpc.get(tuple(sorted((GTY.get(sorted(ok[i])[0], 'X'), GTY.get(sorted(ok[i])[1], 'X'))))) or collections.Counter({'': 1})).most_common(1)[0][0] in rs__ for i, rs__ in zip(gi, rels)])
conf = smax[gi]; hi = conf >= np.quantile(conf, 0.75)
acc_hi = np.mean([GM.get(int(rhat[i]), '') in rs__ for i, rs__, h in zip(gi, rels, hi) if h])
P(f"WHICH RELATION, gold-related candidates ({len(gi)}): predicted latent relation -> its majority gold relation right {acc:.3f} (most confident quarter {acc_hi:.3f}) | "
  f"commonest gold relation ({top1}) {base1:.3f} | commonest for the pair's GOLD entity types (oracle-typed reference) {tp_ref:.3f}")
P(f"DONE ({time.time() - T0:.0f}s)")
