"""LABEL-FREE TRIGGERS FROM CONSISTENCY (15 Nov 2026; tools/stab_trigger.py) - offline, the GPU path's own writes.
Restatement fails (tools/trigger_retest.py: right 0.95 / wrong 0.92 restated, best AUC 0.64). Research routes instead ask whether a
derived link is STABLE and CONSISTENT across views of the evidence:
  1 BOOTSTRAP STABILITY  (semantic entropy, Farquhar et al. 2024; SelfCheckGPT; FactSelfCheck 2025: confabulations are unstable under
                          resampling) - the fact table is rebuilt from NB document-level resamples (70% of the documents read), the code +
                          Adamic-Adar score recomputed; signal = the written pair's mean percentile among all candidates, and its worst
  2 VIEW AGREEMENT       (CAGED, Zhang et al. AAAI 2023; CKRL; KGTtm: consistency across views, local + path evidence) - percentiles of
                          independent predictors at write time: the code alone, Adamic-Adar, resource allocation, 3-hop paths (A-x-y-C,
                          degree-normalised); signal = the WORST of them (all views must support it) and their spread
  3 TEMPORAL STABILITY   (the same principle as evidence accumulates) - the table rebuilt after the next 2/10 of the documents (records
                          stating any written pair or candidate pair left out: INDIRECT support only); signal = the percentile then, and its
                          change since the write
AUC against the distant labels (right vs wrong), and what flagging each signal's lowest 20% does to the kept precision (base 0.74).
Usage: python3 tools/stab_trigger.py <train_distant.json.gz>   env ST_NB (5), TR_SEL (code+aa), TR_ASYM (1.0)"""
import sys, os, types, time, zlib, collections
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4,
                            inf_rcode_sel=os.environ.get('TR_SEL', 'code+aa'), inf_rcode_asym=float(os.environ.get('TR_ASYM', 1.0)),
                            inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; LAB = pm._PM['LAB']
n1 = int(round(ND * 0.2)); n2 = int(round(ND * 0.4))
pm._PM['rti'] = {}
R1 = [r for r in R if r[0] < n1]
pm._rcode_step(None, cfg, lambda d: 0, [1], R1, 1)
W = pm._PM['rti']; WK = [frozenset(k) for k in W]; y = np.array([bool(W[k]['truth']) for k in W])
P(f"{len(WK)} writes at document {n1}, labelled right {y.mean():.3f}  ({time.time() - T0:.0f}s)")
hold = lambda d: zlib.crc32(b'rcode%d' % int(d)) % 1000 < 0.1 * 1000
co = collections.defaultdict(set)
for r in R1:
    if hold(r[0]): co[r[0]].update((r[1], r[2]))
CAND = set()
for ns in co.values():
    ns = sorted(ns)[:60]; CAND.update(frozenset((ns[x], ns[y])) for x in range(len(ns)) for y in range(x + 1, len(ns)))
P(f"{len(CAND)} candidate pairs (the written ones among them: {sum(1 for k in WK if k in CAND)})")

NAMEID = {}
def compact(recs):
    """records as arrays (document, name id, name id), held-out documents and candidate-stating records already dropped (memory)"""
    d, a, b = [], [], []
    for r in recs:
        if hold(r[0]) or r[1] == r[2] or frozenset((r[1], r[2])) in CAND: continue
        d.append(r[0]); a.append(NAMEID.setdefault(r[1], len(NAMEID))); b.append(NAMEID.setdefault(r[2], len(NAMEID)))
    return np.array(d, np.int32), np.array(a, np.int32), np.array(b, np.int32)
def table(recs, seed, k=512):
    """the store's name x name graph without held-out documents and without any record stating a candidate pair; codes; per-name degree"""
    d_, a_, b_ = recs; ua = np.unique(np.concatenate([a_, b_])); loc = {int(x): i for i, x in enumerate(ua)}
    inv = {i: n for n, i in NAMEID.items()}; NI = {inv[int(x)]: i for i, x in enumerate(ua)}
    ia = np.searchsorted(ua, a_); ib = np.searchsorted(ua, b_)
    NN = len(NI); G = csr_matrix((np.ones(len(ia), np.float32), (np.array(ia), np.array(ib))), shape=(NN, NN)); G = G + G.T; G.sum_duplicates(); G.data[:] = 1.0
    deg = np.asarray(G.sum(1)).ravel(); Gc = G.tocoo(); tot = float(G.sum())
    v = np.log(tot * Gc.data / (deg[Gc.row] * deg[Gc.col]) + 1e-12); mk = v > 0
    Pm = csr_matrix((v[mk].astype(np.float32), (Gc.row[mk], Gc.col[mk])), shape=(NN, NN))
    U = TruncatedSVD(k, random_state=seed).fit_transform(Pm).astype(np.float32); U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
    return NI, G, deg, U
def views(T, pairs):
    """per pair: code, Adamic-Adar, resource allocation, 3-hop paths (degree-normalised); nan where a name is missing"""
    NI, G, deg, U = T; ok = np.array([all(n in NI and deg[NI[n]] > 0 for n in p) for p in pairs])
    out = {k: np.full(len(pairs), np.nan) for k in ('code', 'aa', 'ra', 'l3')}
    idx = np.where(ok)[0]
    if not len(idx): return out
    XA = np.array([NI[sorted(pairs[i])[0]] for i in idx]); XB = np.array([NI[sorted(pairs[i])[1]] for i in idx])
    out['code'][idx] = np.einsum('ij,ij->i', U[XA], U[XB])
    iw = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0).astype(np.float32); iw1 = (1.0 / np.maximum(deg, 1)).astype(np.float32)
    Gn = G.multiply(iw1[:, None]).tocsr()
    for s0 in range(0, len(idx), 2000):
        sl = slice(s0, s0 + 2000); GA = G[XA[sl]]; GB = G[XB[sl]]; m = GA.multiply(GB)
        out['aa'][idx[sl]] = np.asarray(m.multiply(iw[None, :]).sum(1)).ravel(); out['ra'][idx[sl]] = np.asarray(m.multiply(iw1[None, :]).sum(1)).ravel()
        out['l3'][idx[sl]] = np.asarray((GA @ Gn).multiply(GB).sum(1)).ravel()
    return out
rk = lambda x: (lambda z: np.argsort(np.argsort(z, kind='stable'), kind='stable') / max(1, len(z) - 1))(np.where(np.isnan(x), -np.inf, x))
CL0 = sorted(CAND - set(WK), key=lambda c: tuple(sorted(c)))                 # percentiles among a fixed sample of 20,000 candidates + the writes (memory)
CL = [CL0[i] for i in np.random.RandomState(9).choice(len(CL0), min(20000, len(CL0)), replace=False)] + WK
WI = {p: i for i, p in enumerate(CL)}; wi = np.array([WI[p] for p in WK])
def percentiles(T):
    """the written pairs' percentiles among ALL candidates under each view and under the write rule (code + AA rank sum)"""
    V = views(T, CL); R_ = {k: rk(v) for k, v in V.items()}; both = R_['code'] + R_['aa']; R_['rule'] = rk(both)
    return {k: v[wi] for k, v in R_.items()}
R2 = compact([r for r in R if r[0] < n2])                                     # later reading (built now, then the full record list is freed: memory)
R1 = compact(R1); del R; pm._PM['R'] = None; import gc; gc.collect()
# write time (all documents read)
KB = int(os.environ.get('ST_K', 256))                                           # resample / later tables at 256 dims (memory: the 512-dim SVD peaks ~3 GB)
P0 = percentiles(table(R1, 4)); P0b = percentiles(table(R1, 4, KB)); P(f"write-time table rebuilt  ({time.time() - T0:.0f}s)")
# 1 bootstrap: 70% of the documents read
NB = int(os.environ.get('ST_NB', 5)); docs = np.arange(n1); BS = []
for b in range(NB):
    keep = set(np.random.RandomState(100 + b).choice(docs, int(0.7 * n1), replace=False).tolist())
    km = np.isin(R1[0], np.fromiter(keep, np.int32)); BS.append(percentiles(table((R1[0][km], R1[1][km], R1[2][km]), 100 + b, KB))['rule']); gc.collect(); P(f"   bootstrap {b + 1} of {NB}  ({time.time() - T0:.0f}s)")
BS = np.array(BS)
# 3 temporal: after the next 2/10 of the documents, records stating a written or candidate pair left out
P2 = percentiles(table(R2, 4, KB)); P(f"later table (documents < {n2})  ({time.time() - T0:.0f}s)")
def report(name, s):
    m = ~np.isnan(s); s_, y_ = s[m], y[m]
    if m.sum() < 20 or y_.all() or (~y_).all(): P(f"   {name}: too few"); return
    auc = _cg_auc(s_[y_], s_[~y_]); q = np.quantile(s_, 0.2); fl = s_ <= q
    P(f"   {name:58s} AUC {auc:.3f} (n {m.sum()}) | lowest 20% flagged: wrong among them {1 - y_[fl].mean():.2f} (overall {1 - y_.mean():.2f}), kept right {y_[~fl].mean():.3f}")
P("SIGNALS vs the distant labels (higher = right):")
report('write-time rule percentile (code + AA)', P0['rule']); report(f'write-time rule percentile at {KB} dims', P0b['rule'])
P(" 1 BOOTSTRAP STABILITY")
report('mean percentile over resamples', BS.mean(0)); report('worst percentile over resamples', BS.min(0)); report('minus spread over resamples', -BS.std(0))
P(" 2 VIEW AGREEMENT (write time)")
for k in ('code', 'aa', 'ra', 'l3'): report(f'view {k}', P0[k])
VV = np.stack([P0[k] for k in ('code', 'aa', 'ra', 'l3')]); report('WORST view (all must support)', VV.min(0)); report('minus spread of the views', -VV.std(0))
P(" 3 TEMPORAL STABILITY (indirect support only)")
report('rule percentile after more reading', P2['rule']); report('change of the rule percentile (same dims)', P2['rule'] - P0b['rule'])
for k in ('code', 'l3'): report(f'change of view {k} (same dims)', P2[k] - P0b[k])
P(" COMBINED")
comb = rk(BS.min(0)) + rk(VV.min(0)) + rk(P2['rule'] - P0b['rule']); report('bootstrap worst + view worst + temporal change', comb)
P(f"DONE ({time.time() - T0:.0f}s)")
