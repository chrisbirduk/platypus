"""SPREADING ACTIVATION FOR INFERENCE (13 Nov 2026; tools/spread_infer.py) - offline, no effect on the package.
REMERGE (Kumaran & McClelland 2012) and HippoRAG (Gutierrez et al. 2024): inference by activation spreading over stored associations, not by one
recalled bridge. tools/integrate_test.py proposed A-C from single shared names across documents (top 10% by the rarest bridge 0.21 gold; both
premises true 0.48; base 0.09). Here the same store (every name pair in a sentence of Re-DocRED train + dev; hub names in 200+ documents left
out) is a graph of names, and each verifiable A-C (A and C named in one document) is scored by activation reaching C from A through OTHER
documents' records only (the pair's own co-occurrence never counts):
  RA     resource allocation: sum over bridges B of 1 / degree(B)          (each bridge shares its activation among its neighbours)
  AA     Adamic-Adar: sum over bridges of 1 / log degree(B)
  RA-w   RA with each edge weighted by the learned extractor's score (tools/consistency_teacher.py, label-free)
  L3     three-step paths A-X-Y-C, degree-normalised (Kovacs et al. 2019), on a sample of pairs
Usage: python3 tools/spread_infer.py"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
_it = open(os.path.join(HERE, 'integrate_test.py')).read()
exec(compile(_it[:_it.index('keys = [k for k in PROP if k in ANN]')], 'integrate_test', 'exec'))
keys = [k for k in PROP if k in ANN]; yk = np.array([k in GOLDP for k in keys])
P(f"verifiable proposals {len(keys)}, gold {yk.mean():.3f}")
def rep(tag, sc):
    o = np.argsort(-sc + 1e-9 * np.random.RandomState(5).rand(len(sc))); k1, k2 = max(1, len(sc) // 10), max(1, len(sc) // 4)
    P(f"   {tag:58s} AUC {auc(sc[yk], sc[~yk]):.3f} | top 10%: gold {yk[o[:k1]].mean():.3f} | top 25%: gold {yk[o[:k2]].mean():.3f}")
# graph from records, edges keep (document) so the pair's own documents can be excluded
from scipy.sparse import csr_matrix
names = sorted({x for c in C for x in (c['cA'], c['cB']) if dfreq[x] <= 200}); ni = {n: i for i, n in enumerate(names)}
ew = collections.defaultdict(float); ewx = collections.defaultdict(float); edocs = collections.defaultdict(set)
sig = lambda z: 1 / (1 + np.exp(-z))
for i, c in enumerate(C):
    if c['cA'] in ni and c['cB'] in ni:
        a, b = sorted((ni[c['cA']], ni[c['cB']])); ew[(a, b)] += 1; ewx[(a, b)] += float(sig(s_all[i])); edocs[(a, b)].add(c['di'])
deg = collections.Counter()
for (a, b) in ew: deg[a] += 1; deg[b] += 1
adj = collections.defaultdict(dict)
for (a, b), w in ew.items(): adj[a][b] = (w, ewx[(a, b)], edocs[(a, b)]); adj[b][a] = (w, ewx[(a, b)], edocs[(a, b)])
RA, AA, RAW = [], [], []
for k in keys:
    A, Cn = tuple(k); ia, ic = ni.get(A), ni.get(Cn); ra = aa = rw = 0.0
    if ia is not None and ic is not None:
        own = adj[ia].get(ic, (0, 0, set()))[2]
        for b, (w1, x1, d1) in adj[ia].items():
            if b == ic or b not in adj[ic]: continue
            w2, x2, d2 = adj[ic][b]
            if not ((d1 | d2) - own): continue                                 # only through records of OTHER documents
            ra += 1 / deg[b]; aa += 1 / np.log(1 + deg[b]); rw += (x1 * x2) / deg[b]
    RA.append(ra); AA.append(aa); RAW.append(rw)
RA, AA, RAW = map(np.array, (RA, AA, RAW))
rare = np.array([-min(dfreq[b] for b in PROP[k]['bridges']) for k in keys], float)
ext = np.array([max(min(s_all[j], s_all[i]) for j, i in PROP[k]['rec']) for k in keys])
rep('rarest bridge (tools/integrate_test.py)', rare); rep('learned extractor, both records', ext)
rep('RA  resource allocation', RA); rep('AA  Adamic-Adar', AA); rep('RA-w  weighted by the learned extractor', RAW)
rk = lambda x: np.argsort(np.argsort(x)) / len(x)
rep('RA-w + rarest bridge (rank sum)', rk(RAW) + rk(rare))
# L3 on a sample
rq = np.random.RandomState(2); S = rq.choice(len(keys), min(6000, len(keys)), replace=False)
rows, cols, vals = [], [], []
for (a, b), w in ew.items():
    v = 1.0 / np.sqrt(deg[a] * deg[b]); rows += [a, b]; cols += [b, a]; vals += [v, v]
Wn = csr_matrix((vals, (rows, cols)), shape=(len(names), len(names)))
L3 = np.zeros(len(S))
for t, j in enumerate(S):
    A, Cn = tuple(keys[j]); ia, ic = ni.get(A), ni.get(Cn)
    if ia is None or ic is None: continue
    ra_ = Wn[ia].toarray().ravel(); ra_[ic] = 0; rc_ = Wn[ic].toarray().ravel(); rc_[ia] = 0
    L3[t] = float(ra_ @ (Wn @ rc_))
ys = yk[S]
o = np.argsort(-L3 + 1e-9 * rq.rand(len(L3))); k1 = len(L3) // 10
P(f"   {'L3  three-step paths (sample of ' + str(len(S)) + ')':58s} AUC {auc(L3[ys], L3[~ys]):.3f} | top 10%: gold {ys[o[:k1]].mean():.3f} | top 25%: gold {ys[o[:len(L3) // 4]].mean():.3f} (sample base {ys.mean():.3f})")
P(f"DONE ({time.time() - T0:.0f}s)")
