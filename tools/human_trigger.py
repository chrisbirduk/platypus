"""ARE THE TRIGGERS WEAK, OR ARE THE LABELS? (15 Nov 2026; tools/human_trigger.py) - offline.
On the GPU path every label-free trigger stops near AUC 0.64 (tools/trigger_retest.py, tools/stab_trigger.py) - but those AUCs are against
the DISTANT (Wikidata) labels, which agree with human labels only 0.85. Here the same signals score the Re-DocRED writes of
tools/fact_compress.py FC_SAVE (200 per split; HUMAN labels from Re-DocRED, and the distant labels for the same pairs), with the distant
corpus as the store's later reading:
  RESTATEMENT   documents stating the pair (distant documents 0-30,000), and records with 3 or fewer words between
  PATH VIEWS    on the distant graph (documents 0-30,000; records stating ANY written pair left out: indirect support only): Adamic-Adar,
                resource allocation, 3-hop paths (degree-normalised)
  TEMPORAL      the change of resource allocation and 3-hop support from documents 0-10,000 to 0-30,000
  COMBINED      rank sums: restatement + resource allocation + 3-hop; and all of them
AUC right vs wrong against HUMAN and DISTANT labels, and the lowest 20% flagged (wrong among them).
Usage: python3 tools/human_trigger.py <train_distant.json.gz> <ret_s0.json> [<ret_s1.json>]"""
import sys, json, types, time, collections
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1])
pm._load_distant(cfg); R = pm._PM['R']; LAB = pm._PM['LAB']
WS = [(p.split('/')[-1], json.load(open(p))) for p in sys.argv[2:]]
ALLP = {frozenset((w[0], w[1])) for _, Wr in WS for w in Wr}
D1, D2 = 10000, 30000
docs = collections.defaultdict(set); short = collections.Counter()
for r in R:
    if r[0] >= D2: continue
    fk = frozenset((r[1], r[2]))
    if fk in ALLP: docs[fk].add(r[0]); short[fk] += int(r[5] <= 3)
def graph(dmax):
    NI = {}; ia, ib = [], []
    for r in R:
        if r[0] >= dmax or r[1] == r[2] or frozenset((r[1], r[2])) in ALLP: continue
        ia.append(NI.setdefault(r[1], len(NI))); ib.append(NI.setdefault(r[2], len(NI)))
    NN = len(NI); G = csr_matrix((np.ones(len(ia), np.float32), (np.array(ia), np.array(ib))), shape=(NN, NN)); G = G + G.T; G.sum_duplicates(); G.data[:] = 1.0
    return NI, G, np.asarray(G.sum(1)).ravel()
def views(T, pairs):
    NI, G, deg = T; iw = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0).astype(np.float32); iw1 = (1.0 / np.maximum(deg, 1)).astype(np.float32)
    Gn = G.multiply(iw1[:, None]).tocsr(); out = np.full((len(pairs), 3), np.nan)
    for i, (a, c) in enumerate(pairs):
        if a not in NI or c not in NI: continue
        ga, gc = G[NI[a]], G[NI[c]]; m = ga.multiply(gc)
        out[i] = [float(m.multiply(iw).sum()), float(m.multiply(iw1).sum()), float((ga @ Gn).multiply(gc).sum())]
    return out
T1 = graph(D1); P(f"graph to document {D1}  ({time.time() - T0:.0f}s)"); T2 = graph(D2); P(f"graph to document {D2}  ({time.time() - T0:.0f}s)")
rk = lambda x: (lambda z: np.argsort(np.argsort(z, kind='stable'), kind='stable') / max(1, len(z) - 1))(np.where(np.isnan(x), -np.inf, x))
def rep(name, s, yh, yd):
    out = []
    for lab, y in (('HUMAN', yh), ('distant', yd)):
        m = ~np.isnan(s); s_, y_ = s[m], y[m]
        if y_.all() or (~y_).all(): out.append(f"{lab} -"); continue
        fl = s_ <= np.quantile(s_, 0.2)
        out.append(f"{lab} AUC {_cg_auc(s_[y_], s_[~y_]):.3f}, flagged wrong {1 - y_[fl].mean():.2f} (base {1 - y_.mean():.2f})")
    P(f"   {name:44s} " + " | ".join(out))
for nm, Wr in WS:
    pairs = [(w[0], w[1]) for w in Wr]; fk = [frozenset(p) for p in pairs]
    yh = np.array([bool(w[2]) for w in Wr]); yd = np.array([f in LAB for f in fk])
    nd = np.array([len(docs.get(f, ())) for f in fk], float); ns = np.array([short.get(f, 0) for f in fk], float)
    V1, V2 = views(T1, pairs), views(T2, pairs)
    P(f"{nm}: {len(Wr)} writes, HUMAN-labelled right {yh.mean():.2f}, distant {yd.mean():.2f} (agree {np.mean(yh == yd):.2f}); stated in a distant document "
      f"{np.mean(nd > 0):.2f} (human right {np.mean(nd[yh] > 0):.2f} / wrong {np.mean(nd[~yh] > 0):.2f})")
    rep('RESTATEMENT: documents stating the pair', nd, yh, yd); rep('RESTATEMENT: short-gap records', ns, yh, yd)
    for j, v in enumerate(('Adamic-Adar', 'resource allocation', '3-hop paths')): rep(f'PATH VIEW {v}', V2[:, j], yh, yd)
    rep('TEMPORAL change of resource allocation', V2[:, 1] - V1[:, 1], yh, yd); rep('TEMPORAL change of 3-hop', V2[:, 2] - V1[:, 2], yh, yd)
    rep('COMBINED restatement + RA + 3-hop', rk(nd) + rk(V2[:, 1]) + rk(V2[:, 2]), yh, yd)
    rep('COMBINED all', rk(nd) + rk(ns) + rk(V2[:, 0]) + rk(V2[:, 1]) + rk(V2[:, 2]) + rk(V2[:, 1] - V1[:, 1]) + rk(V2[:, 2] - V1[:, 2]), yh, yd)
P(f"DONE ({time.time() - T0:.0f}s)")
