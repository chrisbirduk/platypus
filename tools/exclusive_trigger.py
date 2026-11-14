"""AN EXCLUSIVITY TRIGGER FOR RETRACTION (14 Nov 2026; tools/exclusive_trigger.py) - offline, the package's own write path.
tools/compress_typing.py: the exact relation of an inference cannot be typed without labels, but whether it is FUNCTIONAL (0.75-0.77) and
its DIRECTION (the more specific name is the subject, 0.75) can. tools/trigger_retest.py: 'is the pair mentioned again' fails on the
GPU's writes (93% are). Here the trigger asks instead whether later reading points the SUBJECT at a COMPETITOR: the subject S of a write
(the name with the lower store degree), the object O, the CLASS of O (k-means of the names' compressed codes, no labels); in the documents
read after the write, how often S appears with O against with other names of O's class. Flag when a competitor beats O. Restricted, then,
to writes the asymmetry calls functional-like (S clearly more specific than O).
Writes: the GPU path (premises._rcode_step at 20,375 distant documents, seed 4, distant labels), judged on documents 20,375-40,749.
Usage: python3 tools/exclusive_trigger.py <train_distant.json.gz>"""
import sys, types, math, collections, time
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+ra',
                            inf_rcode_selreport=0, inf_rti_max=int(sys.argv[2]) if len(sys.argv) > 2 else 400, inf_rret=0, inf_rti_ctl=0.333)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']
n1 = int(round(ND * 0.2)); n2 = int(round(ND * 0.4))
pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < n1], 1)
W = pm._PM['rti']; ks = list(W); y = np.array([bool(W[k]['truth']) for k in ks])
P(f"{len(ks)} written at {n1} documents, right {y.mean():.3f}; judged on {n1}-{n2}  ({time.time() - T0:.0f}s)")
NI = {}; ia, ib = [], []
for r in R:
    if r[0] < n1: ia.append(NI.setdefault(r[1], len(NI))); ib.append(NI.setdefault(r[2], len(NI)))
NN = len(NI); G = csr_matrix((np.ones(len(ia), np.float32), (ia, ib)), shape=(NN, NN)); G = G + G.T; G.data[:] = 1.0; deg = np.asarray(G.sum(1)).ravel()
Gc = G.tocoo(); v = np.log(G.sum() * Gc.data / (deg[Gc.row] * deg[Gc.col]) + 1e-12); m = v > 0
U = normalize(TruncatedSVD(128, random_state=0).fit_transform(csr_matrix((v[m], (Gc.row[m], Gc.col[m])), shape=G.shape)).astype(np.float32))
KC = 64; cl = MiniBatchKMeans(KC, random_state=0, batch_size=8192, n_init=3).fit_predict(U)
P(f"name classes {KC} over {NN} names  ({time.time() - T0:.0f}s)")
later = collections.defaultdict(collections.Counter)                               # name -> partners in later documents (document counts)
seen = set()
for r in R:
    if n1 <= r[0] < n2:
        for a, b in ((r[1], r[2]), (r[2], r[1])):
            if (r[0], a, b) not in seen: seen.add((r[0], a, b)); later[a][b] += 1
rows = []
for (A, C_) in ks:
    a, c = NI.get(A), NI.get(C_)
    if a is None or c is None: rows.append(None); continue
    S, O, s, o = (A, C_, a, c) if deg[a] <= deg[c] else (C_, A, c, a)
    asym = math.log((deg[o] + 1) / (deg[s] + 1))
    cnt = later.get(S, {}); co = cnt.get(O, 0); comp = max([n for x, n in cnt.items() if x != O and x in NI and cl[NI[x]] == cl[o]] + [0])
    tot_cls = sum(n for x, n in cnt.items() if x in NI and cl[NI[x]] == cl[o])
    rows.append((asym, co, comp, tot_cls, sum(cnt.values())))
ok = np.array([r_ is not None for r_ in rows]); R_ = np.array([r_ if r_ is not None else (0, 0, 0, 0, 0) for r_ in rows], float)
asym, co, comp, totc, tot = R_.T
share = np.where(totc > 0, co / np.maximum(totc, 1), np.nan)
def rep(name, flag, mask):
    f = flag & mask; yy = y[mask]
    if f.sum() == 0 or (~f & mask).sum() == 0: P(f"   {name}: flagged {int(f.sum())}"); return
    caught = (f & ~y).sum() / max(1, (mask & ~y).sum()); lost = (f & y).sum() / max(1, (mask & y).sum())
    P(f"   {name}: writes {int(mask.sum())} (wrong {1 - yy.mean():.2f}) | FLAGGED {int(f.sum())}: wrong {1 - y[f].mean():.2f} | likelihood ratio {caught / max(lost, 1e-9):.1f}")
P("EXCLUSIVITY: later reading ties the subject to a competitor of the object's class at least as often as to the object")
for nm, msk in (('all writes', ok), ('functional-like (subject clearly more specific: asymmetry > 1)', ok & (asym > 1.0)),
                ('strongly functional-like (asymmetry > 2)', ok & (asym > 2.0)), ('subject read again with that class (2+ times)', ok & (totc >= 2)),
                ('functional-like AND subject read again with that class', ok & (asym > 1.0) & (totc >= 2))):
    rep(f"{nm}, competitor >= object", comp >= np.maximum(co, 1), msk)
    rep(f"{nm}, object's share of the class < 0.25", (share < 0.25) & (totc >= 2), msk)
vv = ok & (totc >= 2)
if vv.sum() > 10 and (y[vv]).any() and (~y[vv]).any():
    P(f"   AUC (object's share of the class among the subject's later partners), writes with 2+ such mentions ({int(vv.sum())}): {_cg_auc(share[vv & y], share[vv & ~y]):.3f}")
P(f"DONE ({time.time() - T0:.0f}s)")
