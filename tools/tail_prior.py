"""NAME PRIOR FORECAST (21 Nov 2026; tools/tail_prior.py) - offline, distant set, the package's own writes.
kra20 GPU: after one visit teaching confirmed inferences, HELD-BACK right inferences rose (rank among 10: 0.15 -> 0.33) while held-back WRONG
ones fell (0.19 -> 0.06). Generalisation - or a preference for answer NAMES the taught inferences made familiar? Here: the package's code writes
(as tools/retract_min.py) and the confirmed ones (one record rated top half, kra20's rule); split them by hash as the package does (a fifth
held back). For each held-back inference: how often its ANSWER (tail) name is also the answer of a TAUGHT one, and the tail's store degree;
the AUC of each for right vs wrong held-back, and the rank-among-10 a pure name prior would give (the candidate taught most often wins).
Usage: python3 tools/tail_prior.py <distant_50k.json.gz>"""
import os, sys, types, time, collections, zlib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '.')
import platypus.inference.premises as pm
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa', inf_rcode_asym=1.0,
                            inf_rcode_selreport=0, inf_rti_max=2400, inf_rret=0, inf_rti_ctl=0.0, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rcode_reader=0.5, inf_premises_seed_ctx=1, inf_rti_typed=1)
pm._load_distant(cfg); ND = pm._PM['ND']; CX = pm._PM.get('CTX') or {}; R = pm._PM['R']; n0 = int(round(ND * 0.4)); n1 = int(round(ND * 0.6))
y = np.array([bool(r[6]) for r in R]); hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
wtx = lambda r: f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(int(r[5]), 12) // 3} " + ' '.join('C_' + w for w in CX.get(id(r), '').split())
iRD = np.array([i for i, r in enumerate(R) if r[0] < n0]); sd = iRD[np.random.RandomState(4 + 707).permutation(len(iRD))[:20000]]
c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(hv.transform([wtx(R[i]) for i in sd]), y[sd])
score = np.concatenate([c.decision_function(hv.transform([wtx(R[i]) for i in range(j, min(len(R), j + 200000))])) for j in range(0, len(R), 200000)])
RD = [R[i] for i in iRD]; pm._PM['rdscore'] = score[iRD]; pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], RD, 3)
W = pm._PM['rti']; q50 = float(np.median(score[iRD])); FK = {frozenset(k): k for k in W}; cf = collections.Counter(); DEG = collections.Counter()
for i, r in enumerate(R):
    if r[0] < n0: DEG[r[1]] += 1; DEG[r[2]] += 1
    elif r[0] < n1 and 1 <= r[5] <= 8 and not pm._rret_listlike(r) and score[i] >= q50 and frozenset((r[1], r[2])) in FK: cf[FK[frozenset((r[1], r[2]))]] += 1
C = [k for k in W if cf[k] >= 1]; held = lambda k: zlib.crc32((repr(('fact', k[0], pm._RTI_R, k[1])) + '|rti').encode()) % 5 == 0
TA = [k for k in C if not held(k)]; HB = [k for k in C if held(k)]; tl = lambda k: W[k].get('tt') or k[1]
TC = collections.Counter(tl(k) for k in TA)
P(f"{len(W)} written, {len(C)} confirmed (right {np.mean([W[k]['truth'] for k in C]):.3f}): {len(TA)} taught, {len(HB)} held back  ({time.time() - T0:.0f}s)")
from sklearn.metrics import roc_auc_score
yr = np.array([bool(W[k]['truth']) for k in HB]); f_t = np.array([TC[tl(k)] for k in HB], float); f_d = np.array([np.log1p(DEG[tl(k)]) for k in HB])
P(f"HELD BACK: right {yr.sum()} / wrong {(~yr).sum()} | answer also a TAUGHT answer: right {np.mean(f_t[yr] > 0):.2f} vs wrong {np.mean(f_t[~yr] > 0):.2f} "
  f"(times taught, mean {f_t[yr].mean():.1f} vs {f_t[~yr].mean():.1f}) | AUC right vs wrong: times taught {roc_auc_score(yr, f_t):.3f}, store degree {roc_auc_score(yr, f_d):.3f}")
pool = sorted({tl(k) for k in HB}); rs = np.random.RandomState(7); hit = {True: [], False: []}
for k in HB:                                                                  # a pure NAME PRIOR: the candidate taught most often wins (ties: random)
    alts = [x for x in pool if x not in (tl(k), W[k].get('th') or k[0])]
    if len(alts) < 9: continue
    cs = [tl(k)] + [alts[j] for j in rs.choice(len(alts), 9, replace=False)]; sc = np.array([TC[x] for x in cs], float) + 1e-3 * rs.rand(10)
    hit[bool(W[k]['truth'])].append(int(np.argmax(sc) == 0))
P(f"A PURE NAME PRIOR (the most-taught candidate wins), rank among 10 held-back answers: HELD BACK right {np.mean(hit[True]):.2f} ({len(hit[True])}) | "
  f"wrong {np.mean(hit[False]):.2f} ({len(hit[False])})  (kra20 GPU visit 5: 0.31-0.34 / 0.06)")
P(f"DONE ({time.time() - T0:.0f}s)")
