"""HIGH-PRECISION READER (19 Nov 2026; tools/prec_reader.py) - offline, distant set, LABEL-FREE (distant labels for evaluation only).
With the exact-name gate the reader does not need a good ranking of every record, only a clean top slice. This scores held-out records
(the last 20% of documents) with label-free signals from everything else read, and reports precision at the top 1/2/5/10% and AUC:
  A. single signals: distance; agreement (other documents recording the pair, any or only EARLIER documents); repeats of the pair within its
     own document; pair specificity (PMI-like: pair documents / sqrt(documents of A x documents of B)).
  B. ROUTE 2, USEFULNESS OVER TIME: if the pair were stored, would it help predict later text? Over the other documents that mention A (or B),
     the share that also record the pair (a memory hit) against those that do not (a miss: the memory's prediction would cost bits),
     weighted by how surprising the predicted name is (log of documents / documents of the name). Online variant: earlier documents only.
  C. CONJUNCTIONS declared in advance (no tuning on labels): yield (share of records kept) and precision.
Usage: python3 tools/prec_reader.py <distant.json.gz>"""
import os, sys, time, types, bisect, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, LAB, ND = pm._PM['R'], pm._PM['LAB'], pm._PM['ND']
GLUE = {',', 'and', 'or', 'the', 'a', 'an', '(', ')', '-', '"', "'", ';', ':', ''}
y = np.array([frozenset((r[1], r[2])) in LAB for r in R], float) if os.environ.get('PR_Y', 'any') == 'any' else np.array([bool(r[6]) for r in R], float)   # PR_Y doc: labelled in ITS OWN document (no bias toward pairs seen in many documents)
doc = np.array([r[0] for r in R]); test = doc >= int(ND * 0.8)
pd, nd, pdl = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set); inrep = collections.Counter()
for r in R:
    k = frozenset((r[1], r[2])); pd[k].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0]); inrep[(r[0], k)] += 1
    if r[4] not in GLUE: pdl[k].add(r[0])
pds = {k: sorted(v) for k, v in pd.items()}; nds = {k: sorted(v) for k, v in nd.items()}
idx = np.where(test)[0]; T = len(idx); yt = y[idx]
P(f"{ND} documents, {len(R)} records; held out {T} (base rate {yt.mean():.3f})  ({time.time() - T0:.0f}s)")
F = collections.defaultdict(list)
for i in idx:
    d, A, B = R[i][0], R[i][1], R[i][2]; k = frozenset((A, B))
    ag = len(pd[k]) - 1; nA = len(nd[A]) - 1; nB = len(nd[B]) - 1
    agp = bisect.bisect_left(pds[k], d); nAp = bisect.bisect_left(nds[A], d); nBp = bisect.bisect_left(nds[B], d)
    F['dist'].append(-R[i][5]); F['agree'].append(ag); F['agree_link'].append(len(pdl[k] - {d})); F['agree_past'].append(agp)
    F['repeat_in_doc'].append(inrep[(d, k)]); F['pmi'].append(ag / np.sqrt((nA + 1) * (nB + 1)))
    sA, sB = np.log(ND / (len(nd[A]) + 1)), np.log(ND / (len(nd[B]) + 1))
    for nm, a_, nA_, nB_ in (('use', ag, nA, nB), ('use_past', agp, nAp, nBp)):
        hA = a_ / (nA_ + 2); hB = a_ / (nB_ + 2)                      # hit rate when A (B) turns up again
        F[nm].append(max(hA * sB - (1 - hA) * 0.5, hB * sA - (1 - hB) * 0.5) if a_ else -1.0 - 0.01 * (nA_ + nB_))
F = {k: np.array(v, float) for k, v in F.items()}
def rep(nm, s, m=None):
    yy, ss = (yt, s) if m is None else (yt[m], s[m]); o = np.argsort(-ss + 1e-9 * np.random.RandomState(0).rand(len(ss)))
    pr = ' '.join(f"{q}% {yy[o[:max(1, len(o) * q // 100)]].mean():.3f}" for q in (1, 2, 5, 10))
    P(f"   {nm:58s} AUC {roc_auc_score(yy, ss):.3f} | precision top {pr}")
rk = lambda v: np.argsort(np.argsort(v + 1e-9 * np.random.RandomState(1).rand(len(v)))).astype(float)
P("A. SINGLE SIGNALS (held-out records):")
for k in ('dist', 'agree', 'agree_link', 'agree_past', 'repeat_in_doc', 'pmi'): rep(k, F[k])
P("B. ROUTE 2, USEFULNESS OVER TIME (memory hits vs misses on other documents naming A or B, weighted by the predicted name's surprise):")
rep('usefulness (all other documents)', F['use']); rep('usefulness (EARLIER documents only: online)', F['use_past'])
P("   ranks summed:")
rep('usefulness + distance', rk(F['use']) + rk(F['dist']))
rep('usefulness + distance + repeats in own document', rk(F['use']) + rk(F['dist']) + rk(F['repeat_in_doc']))
rep('agreement + distance + repeats + pmi', rk(F['agree']) + rk(F['dist']) + rk(F['repeat_in_doc']) + rk(F['pmi']))
rep('online: usefulness (earlier) + distance + repeats', rk(F['use_past']) + rk(F['dist']) + rk(F['repeat_in_doc']))
kn = (F['agree'] > 0); P(f"   records whose pair is recorded in another document: {kn.mean():.3f} of held-out (base rate there {yt[kn].mean():.3f}):")
rep('  usefulness, these records only', F['use'], kn); rep('  usefulness + distance, these only', rk(F['use']) + rk(F['dist']), kn)
P("C. CONJUNCTIONS declared in advance: kept share of held-out records, precision (base rate {:.3f})".format(yt.mean()))
D = -F['dist']
for nm, m in (('agree >= 1 and distance <= 3', (F['agree'] >= 1) & (D <= 3)), ('agree >= 2', F['agree'] >= 2), ('agree >= 3', F['agree'] >= 3),
              ('agree >= 2 and distance <= 5', (F['agree'] >= 2) & (D <= 5)), ('agree >= 1 and repeated in own document', (F['agree'] >= 1) & (F['repeat_in_doc'] >= 2)),
              ('agree >= 2 and repeated in own document', (F['agree'] >= 2) & (F['repeat_in_doc'] >= 2)),
              ('agree with a link word >= 1 and distance <= 5', (F['agree_link'] >= 1) & (D <= 5)), ('pmi >= 0.2 and agree >= 1', (F['pmi'] >= 0.2) & (F['agree'] >= 1)),
              ('pmi >= 0.5 and agree >= 2', (F['pmi'] >= 0.5) & (F['agree'] >= 2)), ('ONLINE: agree_past >= 2 and distance <= 5', (F['agree_past'] >= 2) & (D <= 5))):
    P(f"   {nm:52s} kept {m.mean():.4f} ({m.sum():5d}) | precision {yt[m].mean() if m.any() else float('nan'):.3f}")
P(f"DONE ({time.time() - T0:.0f}s)")
